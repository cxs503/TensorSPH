"""Continuum Surface Force (CSF) model for SPH surface tension.

Implements the CSF model of Brackbill, Kothe & Zemach (1992) adapted for
weakly-compressible SPH. The surface tension force is:

    f_st = sigma * kappa * n * delta_s

where:
    sigma: surface tension coefficient (N/m)
    kappa: curvature (1/m), computed as divergence of the normal field
    n: interface normal = grad(c) / |grad(c)|
    delta_s: surface delta function approximated by |grad(c)|

The color function c is taken as the density ratio rho / rho0, which provides
a smooth indicator of the interface in single-fluid SPH. For multi-phase
extensions, c would be a phase indicator.

References:
    Brackbill, Kothe & Zemach (1992): "A continuum surface force method"
    Monaghan (1992): "Smoothed particle hydrodynamics"
    Tartakovsky et al. (2005): "Modeling of physically adsorbing surfaces"
"""
from __future__ import annotations

import math

import torch

from .droplet_impact import Config, cubic_spline_gradient, cubic_spline_kernel


def compute_color_gradient(
    pos: torch.Tensor,
    color: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Compute gradient of color function using SPH.

    grad(c)_i = sum_j V_j * (c_j - c_i) * grad_i W_ij,   V_j = m_j / rho_j

    The pair list holds each unordered pair once, so particle ``j`` receives its
    own contribution from the same entry. Written out for ``j`` the term is

        grad(c)_j = V_i * (c_i - c_j) * grad_j W_ji
                  = V_i * (c_j - c_i) * grad_i W_ij

    because ``grad_j W_ji = -grad_i W_ij``; both the volume and the sign belong
    to the partner particle, which is what makes the discrete operator reproduce
    the gradient of a linear field (``tests/test_surface_tension.py``).

    Args:
        pos: (N, 2) particle positions
        color: (N,) color function values
        mass: (N,) particle masses
        density: (N,) particle densities
        pairs: (P, 2) neighbor pair indices
        h: smoothing length

    Returns:
        grad_c: (N, 2) gradient of color function
    """
    n = pos.shape[0]
    grad_c = torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    if pairs.numel() == 0:
        return grad_c

    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(delta, h)

    dc = color[j] - color[i]
    vol = mass / density

    grad_c = grad_c.index_add(0, i, vol[j, None] * dc[:, None] * grad_w)
    grad_c = grad_c.index_add(0, j, vol[i, None] * dc[:, None] * grad_w)

    return grad_c


def compute_normal(
    grad_c: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute interface normal from color gradient.

    n = grad(c) / |grad(c)|

    Args:
        grad_c: (N, 2) gradient of color function
        eps: small value to avoid division by zero

    Returns:
        normal: (N, 2) unit normal vectors (zero where |grad_c| is small)
    """
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    # Only normalize where gradient is significant (interface region)
    mask = mag.squeeze() > eps
    normal = torch.zeros_like(grad_c)
    normal[mask] = grad_c[mask] / mag[mask]
    return normal


def compute_curvature(
    pos: torch.Tensor,
    normal: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Compute curvature as divergence of normal field.

    kappa = div(n) = sum_j m_j / rho_j * (n_j - n_i) . grad_W_ij

    Note: The sign convention is such that a convex droplet (with outward-pointing
    normal) has positive curvature (div(n) > 0), giving an inward-pointing force
    when combined with f = -sigma * kappa * n.

    Args:
        pos: (N, 2) particle positions
        normal: (N, 2) interface normals (pointing outward)
        mass: (N,) particle masses
        density: (N,) particle densities
        pairs: (P, 2) neighbor pair indices
        h: smoothing length

    Returns:
        kappa: (N,) curvature values (positive for convex surfaces)
    """
    n = pos.shape[0]
    kappa = torch.zeros(n, dtype=pos.dtype, device=pos.device)

    if pairs.numel() == 0:
        return kappa

    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(delta, h)

    # Divergence of normal: (n_j - n_i) . grad_W
    dn = normal[j] - normal[i]
    div_n = (dn * grad_w).sum(dim=1)
    vol_j = mass[j] / density[j]
    vol_i = mass[i] / density[i]

    # Symmetric contribution:
    # For particle i: V_j * (n_j - n_i) . grad_W_ij
    # For particle j: V_i * (n_j - n_i) . grad_W_ij (note: grad_W_ji = -grad_W_ij, but we use the same pair)
    kappa = kappa.index_add(0, i, vol_j * div_n)
    kappa = kappa.index_add(0, j, vol_i * div_n)

    return kappa


def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall: torch.Tensor,
    h: float,
    theta: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Modify the color gradient near walls to enforce a contact angle.

    For particles within interaction range of the wall, the color gradient is
    rotated so that the interface normal makes the desired angle theta with the
    wall. The gradient magnitude is preserved.

    The contact angle theta is measured through the fluid:
        - theta < pi/2: hydrophilic (wetting)
        - theta = pi/2: neutral
        - theta > pi/2: hydrophobic (non-wetting)

    Algorithm:
        1. Find the closest wall point for each particle
        2. Compute wall normal (from wall to fluid)
        3. Decompose grad_c into wall-normal and wall-tangential components
        4. Adjust the normal component to achieve cos(theta) alignment
        5. Preserve the gradient magnitude

    Args:
        grad_c: (N, 2) color gradient (points from surface to interior)
        pos: (N, 2) particle positions
        wall: (Nw, 2) wall particle positions (may be empty)
        h: smoothing length
        theta: desired contact angle in radians
        eps: small value to avoid division by zero

    Returns:
        grad_c_modified: (N, 2) modified color gradient
    """
    n = pos.shape[0]
    grad_c_out = grad_c.clone()

    if wall.numel() == 0 or n == 0:
        return grad_c_out

    # Find closest wall point for each fluid particle
    # pos: (N, 2), wall: (Nw, 2)
    # dist_sq: (N, Nw)
    dist_sq = torch.sum((pos.unsqueeze(1) - wall.unsqueeze(0)) ** 2, dim=2)
    min_dist_sq, wall_idx = torch.min(dist_sq, dim=1)
    closest_wall = wall[wall_idx]  # (N, 2)

    # Wall normal: direction from wall to fluid particle
    wall_normal = pos - closest_wall  # (N, 2)
    wall_normal_mag = torch.linalg.vector_norm(wall_normal, dim=1, keepdim=True)
    wall_normal = wall_normal / wall_normal_mag.clamp_min(eps)  # (N, 2)

    # Only modify particles within 2h of the wall
    near_wall = min_dist_sq < (2.0 * h) ** 2

    if not near_wall.any():
        return grad_c_out

    # Gradient magnitude (preserved)
    grad_mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)  # (N, 1)

    # Current interface normal (grad_c points from surface to interior,
    # so this is the inward normal)
    n_interface = grad_c / grad_mag.clamp_min(eps)  # (N, 2)

    # Current alignment with wall normal
    cos_current = (n_interface * wall_normal).sum(dim=1, keepdim=True)  # (N, 1)

    # Desired alignment: cos(theta)
    # The interface normal should make angle theta with the wall
    # n_interface · wall_normal = cos(theta)
    cos_target = math.cos(theta)

    # Decompose n_interface into wall-normal and wall-tangential components
    n_normal = cos_current * wall_normal  # wall-normal component
    n_tangent = n_interface - n_normal    # wall-tangential component
    n_tangent_mag = torch.linalg.vector_norm(n_tangent, dim=1, keepdim=True)

    # New interface normal with desired contact angle
    # n_new = cos(theta) * wall_normal + sin(theta) * t_hat
    # where t_hat is the unit tangential direction
    sin_theta = math.sin(theta)
    t_hat = n_tangent / n_tangent_mag.clamp_min(eps)
    n_new = cos_target * wall_normal + sin_theta * t_hat

    # Reconstruct gradient with preserved magnitude
    grad_c_new = grad_mag * n_new

    # Apply only to near-wall particles
    grad_c_out[near_wall] = grad_c_new[near_wall]

    return grad_c_out


def compute_surface_tension_force(
    pos: torch.Tensor,
    density: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    cfg: Config,
    wall: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute surface tension acceleration using CSF model.

    For single-fluid SPH with free surfaces (droplets/bubbles), we detect
    surface particles by counting neighbors. Surface particles have fewer
    neighbors than interior particles.

    The surface tension force is applied to surface particles only:
        f_st = sigma * kappa * n

    where n is the inward normal and kappa is the local curvature.

    When a wall is provided and cfg.contact_angle is set, the color gradient
    is modified near the wall to enforce the specified contact angle.

    Args:
        pos: (N, 2) particle positions
        density: (N,) particle densities
        mass: (N,) particle masses
        pairs: (P, 2) neighbor pair indices
        cfg: simulation configuration
        wall: (Nw, 2) wall particle positions, or None

    Returns:
        acc_st: (N, 2) surface tension acceleration
    """
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient

    if sigma <= 0.0 or pairs.numel() == 0:
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    # Count neighbors for each particle
    i, j = pairs.unbind(dim=1)
    neighbor_count = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    neighbor_count = neighbor_count.index_add(0, i, torch.ones_like(i, dtype=pos.dtype))
    neighbor_count = neighbor_count.index_add(0, j, torch.ones_like(j, dtype=pos.dtype))

    # Detect surface particles: those with fewer neighbors than average
    # Interior particles typically have ~20-30 neighbors in 2D
    # Surface particles have significantly fewer
    avg_neighbors = neighbor_count.mean()
    surface_threshold = 0.7 * avg_neighbors  # 70% of average
    is_surface = neighbor_count < surface_threshold

    if not is_surface.any():
        # No surface particles detected
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    # For surface particles, compute a color function based on neighbor count
    # Interior particles: c = 1.0
    # Surface particles: c = neighbor_count / avg_neighbors (0 to 0.7)
    color = torch.ones(n, dtype=pos.dtype, device=pos.device)
    color[is_surface] = neighbor_count[is_surface] / avg_neighbors

    # Compute gradient of color function
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)

    # Apply contact angle boundary condition if wall and contact_angle are provided
    if wall is not None and cfg.contact_angle is not None and wall.numel() > 0:
        grad_c = apply_contact_angle_boundary(
            grad_c, pos, wall, h, cfg.contact_angle
        )

    # Compute normal from color gradient
    # Note: color is high (1.0) in interior and low at surface, so grad(c) points inward.
    # For CSF model, we need normal pointing outward (from fluid to vacuum).
    # So we negate the gradient to get outward normal.
    normal = compute_normal(-grad_c)

    # Compute curvature
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)

    # Surface delta function: |grad(c)|
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)

    # CSF force per unit volume: f = sigma * kappa * n * delta_s
    # For a convex surface (droplet) with outward normal, kappa > 0 (divergence of outward normal)
    # The force should point inward (toward center) to contract the droplet.
    # Since n points outward and kappa > 0, f = sigma * kappa * n points outward.
    # But we want inward force! So we need to negate: f = -sigma * kappa * n * delta_s
    # OR: use inward normal (which we already have from grad(c)) and positive kappa.

    # Let's use the convention: force points inward for convex surfaces.
    # With outward normal and kappa = div(n) > 0 for convex surface:
    # f = -sigma * kappa * n points inward (correct!)
    force_mag = -sigma * kappa * delta_s
    acc_st = force_mag[:, None] * normal / density[:, None]

    return acc_st


def capillary_dt_limit(cfg: Config, safety: float = 0.25) -> float:
    """Time step limit from capillary waves.

    The capillary time step is:
        dt_cap = sqrt(rho * h^3 / sigma)

    This is analogous to the Courant limit but for surface tension waves.
    See Brackbill et al. (1992) or Lapinskas et al. (2022).

    Args:
        cfg: simulation configuration
        safety: safety factor (default 0.25)

    Returns:
        dt_cap: maximum stable time step from capillary limit
    """
    if cfg.surface_tension_coefficient <= 0.0:
        return float("inf")
    h = cfg.h_factor * cfg.dx
    rho = cfg.rho0
    sigma = cfg.surface_tension_coefficient
    # Capillary time scale: sqrt(rho * h^3 / sigma)
    dt_cap = safety * (rho * h**3 / sigma) ** 0.5
    return dt_cap
