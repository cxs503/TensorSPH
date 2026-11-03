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
        wall: optional (M, 2) static wall particles. When given *and*
            ``cfg.contact_angle`` is set, the colour gradient near the wall is
            rotated so that the interface meets the solid at that angle
            (:func:`apply_contact_angle_boundary`).

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

    # Wall contact angle: rotate the colour gradient where the interface meets
    # the solid, so that the CSF normal satisfies Young's relation. No wall or
    # ``contact_angle=None`` leaves the gradient (and the old behaviour) intact.
    if cfg.contact_angle is not None and wall is not None and wall.numel() > 0:
        grad_c = apply_contact_angle_boundary(grad_c, pos, wall, h,
                                              cfg.contact_angle)

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


# --------------------------------------------------------------------------- #
# Wall contact-angle boundary condition
# --------------------------------------------------------------------------- #
def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall: torch.Tensor | None,
    h: float,
    theta: float,
    reach: float = 2.0,
    chunk: int = 4096,
) -> torch.Tensor:
    """Impose a static contact angle where the interface meets a solid wall.

    Physics
    -------
    Young's relation ``sigma_sv = sigma_sl + sigma_lv cos(theta)`` fixes the
    angle between the liquid-vapour interface and the solid surface. With
    ``n_w`` the wall normal pointing *from the wall into the fluid* and
    ``n_out = -grad(c)/|grad(c)|`` the outward interface normal, the condition is

        angle(n_out, n_w) = theta                                   (1)

    (``theta = 0`` wetting film: interface parallel to the wall, normal along
    ``n_w``; ``theta = pi/2`` neutral: interface perpendicular to the wall,
    normal tangent to it; ``theta -> pi`` completely non-wetting).

    Implementation
    --------------
    For every fluid particle within ``reach * h`` of a wall particle, the colour
    gradient is decomposed into its wall-normal part and its tangential part
    ``g_t``.  The tangential *direction* ``t_hat = g_t / |g_t|`` is kept, so the
    interface tilts to the side it already tilts to, and the direction is
    rebuilt so that (1) holds while the magnitude is preserved:

        g_new = |g| * ( -cos(theta) * n_w + sin(theta) * t_hat )      (2)

    (2) is exactly ``-|g| n_out`` with ``n_out`` satisfying (1): the sign of the
    tangential term carries the original tilt side, therefore the operator is
    reversible in theta and reduces to the identity for the angle the gradient
    already has.  This is the "gradient modification" variant of the contact
    angle treatment of Tartakovsky & Meakin (2005) and Adami, Hu & Adams (2012).

    Args:
        grad_c: (N, D) colour-function gradient, modified in place semantics
            (a new tensor is returned; the input is not mutated).
        pos: (N, D) fluid particle positions.
        wall: (M, D) static wall particle positions, or ``None``/empty for no
            wall (the gradient is returned unchanged).
        h: smoothing length.
        theta: contact angle in radians, ``0 <= theta <= pi``.
        reach: wall influence measured in smoothing lengths.
        chunk: block size of the nearest-wall search (bounds peak memory).

    Returns:
        (N, D) modified colour gradient, same dtype/device as ``grad_c``.

    Raises:
        ValueError: if ``theta`` is outside ``[0, pi]`` or ``reach`` is
            not positive.
    """
    if not math.isfinite(theta) or theta < 0.0 or theta > math.pi:
        raise ValueError(f"contact angle must lie in [0, pi], got {theta!r}")
    if reach <= 0.0:
        raise ValueError(f"reach must be positive, got {reach!r}")
    if wall is None or wall.numel() == 0 or grad_c.numel() == 0:
        return grad_c

    out = grad_c.clone()
    wall = wall.to(dtype=pos.dtype, device=pos.device)
    reach_dist = reach * h

    # Nearest wall particle for each fluid particle (blocked to bound memory).
    idx = torch.empty(pos.shape[0], dtype=torch.long, device=pos.device)
    dist = torch.empty(pos.shape[0], dtype=pos.dtype, device=pos.device)
    for start in range(0, pos.shape[0], chunk):
        stop = min(start + chunk, pos.shape[0])
        d = torch.cdist(pos[start:stop], wall)
        dmin, argmin = d.min(dim=1)
        dist[start:stop] = dmin
        idx[start:stop] = argmin

    near = dist <= reach_dist
    if not bool(near.any()):
        return out

    # Wall normal: from the nearest wall particle into the fluid.
    delta = pos[near] - wall[idx[near]]
    d_near = dist[near]
    safe = d_near > 1.0e-12
    n_w = torch.zeros_like(delta)
    n_w[safe] = delta[safe] / d_near[safe, None]
    # A particle sitting exactly on a wall particle has no defined wall normal;
    # leave its gradient untouched rather than inventing a direction.
    active = safe
    if not bool(active.any()):
        return out

    g = out[near]
    mag = torch.linalg.vector_norm(g, dim=1)
    g_n = (g * n_w).sum(dim=1, keepdim=True)
    g_t = g - g_n * n_w
    gt_mag = torch.linalg.vector_norm(g_t, dim=1)

    # Tangential direction: keep the existing tilt side; if the gradient is
    # already along the wall normal, fall back to a fixed in-plane rotation.
    t_hat = torch.empty_like(g_t)
    have_t = gt_mag > 1.0e-12 * mag.clamp_min(1.0e-300)
    t_hat[have_t] = g_t[have_t] / gt_mag[have_t, None]
    if bool((~have_t).any()):
        perp = torch.stack((-n_w[~have_t, 1], n_w[~have_t, 0]), dim=1)
        t_hat[~have_t] = perp

    direction = -math.cos(theta) * n_w + math.sin(theta) * t_hat
    d_mag = torch.linalg.vector_norm(direction, dim=1).clamp_min(1.0e-30)
    direction = direction / d_mag[:, None]

    # Particles with an undefined wall normal keep their original gradient.
    block = out[near]
    new_block = torch.where(active[:, None], mag[:, None] * direction, block)
    out[near] = new_block
    return out


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
