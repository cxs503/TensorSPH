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

Contact angle boundary condition
    When a fluid interface meets a solid wall the contact angle theta is
    enforced by rotating the colour gradient of near-wall particles so that

        n . n_wall = cos(theta)

    where n_wall is the wall normal pointing into the fluid (Adami et al.
    2012; Tartakovsky & Meakin 2005). See
    ``docs/contact_angle_boundary.md`` and ``apply_contact_angle_boundary``.

References:
    Brackbill, Kothe & Zemach (1992): "A continuum surface force method"
    Monaghan (1992): "Smoothed particle hydrodynamics"
    Tartakovsky et al. (2005): "Modeling of physically adsorbing surfaces"
    Adami et al. (2012): "Generalized boundary condition for SPH"
"""
from __future__ import annotations

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


def csf_colour_field(
    pos: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Colour function, its gradient and the free-surface mask of the CSF model.

    Interior particles carry ``c = 1``; particles whose neighbour count falls
    below 70 % of the mean carry ``c = n_neighbours / mean(n_neighbours)``.
    ``grad(c)`` of that field feeds the CSF normal, curvature and surface delta
    function, so the force routine and the verification harnesses share this
    single implementation instead of keeping drifting copies.

    Args:
        pos: (N, 2) particle positions
        mass: (N,) particle masses
        density: (N,) particle densities
        pairs: (P, 2) neighbour pair indices
        h: smoothing length

    Returns:
        color: (N,) colour function
        grad_c: (N, 2) colour gradient
        is_surface: (N,) boolean free-surface mask
    """
    n = pos.shape[0]
    color = torch.ones(n, dtype=pos.dtype, device=pos.device)
    grad_c = torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)
    if pairs.numel() == 0:
        return color, grad_c, torch.zeros(n, dtype=torch.bool, device=pos.device)

    i, j = pairs.unbind(dim=1)
    neighbor_count = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    neighbor_count = neighbor_count.index_add(0, i, torch.ones_like(i, dtype=pos.dtype))
    neighbor_count = neighbor_count.index_add(0, j, torch.ones_like(j, dtype=pos.dtype))

    # Interior particles have ~20-30 neighbours in 2-D; free-surface particles
    # have roughly half of that, so 70 % of the mean separates the two.
    avg_neighbors = neighbor_count.mean()
    is_surface = neighbor_count < 0.7 * avg_neighbors
    if is_surface.any():
        color[is_surface] = neighbor_count[is_surface] / avg_neighbors

    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    return color, grad_c, is_surface


def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall_positions: torch.Tensor,
    h: float,
    contact_angle_rad: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Apply contact angle boundary condition to color gradient near walls.

    When a fluid interface meets a solid wall, the contact angle θ determines
    how the interface meets the wall. This function modifies the color gradient
    of surface particles near the wall to enforce the desired contact angle.

    The method follows Adami et al. (2012) and Tartakovsky & Meakin (2005):
    - For particles near the wall, the wall-normal component of grad(c) is
      adjusted to satisfy: n · n_wall = cos(θ)
    - The wall outward normal n_wall points into the fluid domain.

    Args:
        grad_c: (N, 2) color gradient (modified in place for near-wall particles)
        pos: (N, 2) particle positions
        wall_positions: (M, 2) wall particle positions
        h: smoothing length
        contact_angle_rad: desired contact angle in radians (0 < θ < π)
            - θ < π/2: hydrophilic (wetting)
            - θ = π/2: neutral
            - θ > π/2: hydrophobic (non-wetting)
        eps: small value for numerical stability

    Returns:
        grad_c: (N, 2) modified color gradient with contact angle enforced
    """
    if wall_positions.numel() == 0:
        return grad_c

    n = pos.shape[0]
    cos_theta = torch.cos(torch.tensor(contact_angle_rad, dtype=pos.dtype, device=pos.device))
    sin_theta = torch.sin(torch.tensor(contact_angle_rad, dtype=pos.dtype, device=pos.device))

    # For large wall counts, use bounds-based approach
    if wall_positions.shape[0] > 5000:
        return _apply_contact_angle_by_bounds(grad_c, pos, h, contact_angle_rad, eps)

    # Find particles near the wall (within 2h)
    search_radius = 2.0 * h
    diff = pos[:, None, :] - wall_positions[None, :, :]  # (N, M, 2)
    dist_sq = (diff ** 2).sum(dim=2)  # (N, M)
    min_dist_sq = dist_sq.min(dim=1).values  # (N,)
    near_wall = min_dist_sq < (search_radius ** 2)

    if not near_wall.any():
        return grad_c

    # Wall normal: from wall to fluid particle
    nearest_wall_idx = dist_sq.argmin(dim=1)
    nearest_wall_pos = wall_positions[nearest_wall_idx]
    wall_normal = pos - nearest_wall_pos
    wall_normal_mag = torch.linalg.vector_norm(wall_normal, dim=1, keepdim=True)
    wall_normal = wall_normal / (wall_normal_mag + eps)

    # Current normal from color gradient
    grad_mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    current_normal = grad_c / (grad_mag + eps)

    # Project onto tangent plane and reconstruct with contact angle
    dot_product = (current_normal * wall_normal).sum(dim=1, keepdim=True)
    n_tangent = current_normal - dot_product * wall_normal
    n_tangent_mag = torch.linalg.vector_norm(n_tangent, dim=1, keepdim=True)

    safe_tangent = torch.where(
        n_tangent_mag > eps,
        n_tangent / (n_tangent_mag + eps),
        torch.zeros_like(n_tangent)
    )

    new_normal = cos_theta * wall_normal + sin_theta * safe_tangent

    significant_grad = grad_mag.squeeze() > eps
    apply_mask = near_wall & significant_grad
    grad_c[apply_mask] = grad_mag[apply_mask] * new_normal[apply_mask]

    return grad_c


def _apply_contact_angle_by_bounds(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    h: float,
    contact_angle_rad: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Simplified contact angle enforcement using domain bounds."""
    cos_theta = torch.cos(torch.tensor(contact_angle_rad, dtype=pos.dtype, device=pos.device))
    sin_theta = torch.sin(torch.tensor(contact_angle_rad, dtype=pos.dtype, device=pos.device))

    near_left = pos[:, 0] < 2.0 * h
    near_right = pos[:, 0] > pos[:, 0].max() - 2.0 * h
    near_bottom = pos[:, 1] < 2.0 * h
    near_wall = near_left | near_right | near_bottom
    if not near_wall.any():
        return grad_c

    grad_mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    current_normal = grad_c / (grad_mag + eps)

    wall_normal = torch.zeros_like(grad_c)
    wall_normal[near_left, 0] = 1.0
    wall_normal[near_right, 0] = -1.0
    wall_normal[near_bottom, 1] = 1.0

    dot_product = (current_normal * wall_normal).sum(dim=1, keepdim=True)
    n_tangent = current_normal - dot_product * wall_normal
    n_tangent_mag = torch.linalg.vector_norm(n_tangent, dim=1, keepdim=True)

    safe_tangent = torch.where(
        n_tangent_mag > eps,
        n_tangent / (n_tangent_mag + eps),
        torch.zeros_like(n_tangent)
    )

    new_normal = cos_theta * wall_normal + sin_theta * safe_tangent

    significant_grad = grad_mag.squeeze() > eps
    apply_mask = near_wall & significant_grad
    grad_c[apply_mask] = grad_mag[apply_mask] * new_normal[apply_mask]

    return grad_c


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

    Args:
        pos: (N, 2) particle positions
        density: (N,) particle densities
        mass: (N,) particle masses
        pairs: (P, 2) neighbor pair indices
        cfg: simulation configuration
        wall: (M, 2) wall particle positions; when given together with
            ``cfg.contact_angle`` the colour gradient of near-wall particles is
            rotated to realise the requested contact angle

    Returns:
        acc_st: (N, 2) surface tension acceleration
    """
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient

    if sigma <= 0.0 or pairs.numel() == 0:
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    color, grad_c, is_surface = csf_colour_field(pos, mass, density, pairs, h)

    if not is_surface.any():
        # No surface particles detected
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    # Enforce the contact angle on near-wall particles before the normal and
    # curvature are formed, so the whole CSF chain sees the wetted interface.
    if wall is not None and wall.numel() > 0 and cfg.contact_angle is not None:
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
