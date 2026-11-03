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
    reach: float = 2.0,
    chunk: int = 2048,
) -> torch.Tensor:
    """Enforce a prescribed contact angle on the interface normal near a wall.

    The interface normal is taken to be collinear with ``grad_c`` (in this module
    the outward normal is ``-grad(c)``, so the same function serves both sign
    conventions: it only *reorients* a vector field, it never flips it).  For a
    fluid particle within ``reach * h`` of a wall particle, the local wall normal

        n_w = (x_i - x_w) / |x_i - x_w|

    (pointing from the wall into the fluid) together with the contact angle theta
    fixes the admissible normal, following the colour-gradient reconstruction of
    Adami, Hu & Adams (2012), *A generalized wall boundary condition for
    smoothed particle hydrodynamics*, J. Comput. Phys. 231:7057-7075, and
    Tartakovsky & Meakin (2005), *Modeling of interfacial dynamics in
    smoothed particle hydrodynamics*, J. Comput. Phys. 207:610-628:

        n_new = cos(theta) * n_w + sin(theta) * t_hat

    where ``t_hat`` is the unit tangential direction obtained by projecting the
    incoming normal onto the wall plane.  ``theta = 0`` gives a normal along
    ``n_w`` (a film lying flat on the wall, complete wetting), ``theta = pi/2``
    gives a normal in the wall plane (the interface meets the wall at 90 deg),
    and ``theta = pi`` would invert it (non-wetting limit).  The magnitude
    ``|grad_c|`` is preserved, so the surface delta function of the CSF model is
    unaffected and only the direction (hence the curvature) changes.

    The function is non-destructive: the input tensor is not modified.

    Args:
        grad_c: (N, 2) colour gradient / interface normal field.
        pos: (N, 2) fluid particle positions.
        wall: (M, 2) static wall particle positions.
        h: smoothing length.
        theta: prescribed contact angle (rad) measured inside the fluid between
            the interface normal and the wall normal.
        reach: wall influence range in units of ``h`` (default 2h, the kernel
            support).
        chunk: rows per distance block, bounds the (chunk, M) cdist workspace.

    Returns:
        (N, 2) reoriented field; equal to ``grad_c`` for particles outside the
        wall influence range or with a vanishing gradient.
    """
    if wall.numel() == 0 or pos.numel() == 0 or grad_c.numel() == 0:
        return grad_c
    if not math.isfinite(theta):
        return grad_c

    n = pos.shape[0]
    wall = wall.to(dtype=pos.dtype, device=pos.device)

    # Nearest wall particle for every fluid particle (chunked cdist keeps the
    # workspace at chunk x M instead of N x M).
    dist = torch.empty(n, dtype=pos.dtype, device=pos.device)
    nearest = torch.empty((n, pos.shape[1]), dtype=pos.dtype, device=pos.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(pos[start:stop], wall)
        dmin, idx = d.min(dim=1)
        dist[start:stop] = dmin
        nearest[start:stop] = wall[idx]

    n_wall = pos - nearest
    n_wall_mag = torch.linalg.vector_norm(n_wall, dim=1, keepdim=True)
    g_mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True).clamp_min(1.0e-30)

    active = ((dist < reach * h)
              & (n_wall_mag.squeeze() > 1.0e-12)
              & (g_mag.squeeze() > 1.0e-12))
    if not bool(active.any()):
        return grad_c

    n_hat = n_wall[active] / n_wall_mag[active]
    g_hat = grad_c[active] / g_mag[active]

    cos_orig = (g_hat * n_hat).sum(dim=1, keepdim=True)
    tangent = g_hat - cos_orig * n_hat
    t_mag = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)

    # If the incoming normal is already wall-normal the tangential part is zero
    # and there is no tilt direction to preserve; fall back to a fixed 90 deg
    # rotation of the wall normal so the prescribed angle is still reachable.
    perp = torch.stack((-n_hat[:, 1], n_hat[:, 0]), dim=1)
    t_hat = torch.where(t_mag > 1.0e-12, tangent / t_mag.clamp_min(1.0e-30), perp)

    target = math.cos(theta) * n_hat + math.sin(theta) * t_hat
    target = target / torch.linalg.vector_norm(target, dim=1, keepdim=True).clamp_min(1.0e-30)

    out = grad_c.clone()
    out[active] = g_mag[active] * target
    return out


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
        wall: (M, 2) static wall particles, or None. When given together with
            ``cfg.contact_angle`` the interface normal within 2h of the wall is
            reoriented by :func:`apply_contact_angle_boundary`.

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

    # Compute normal from color gradient
    # Note: color is high (1.0) in interior and low at surface, so grad(c) points inward.
    # For CSF model, we need normal pointing outward (from fluid to vacuum).
    # So we negate the gradient to get outward normal.
    normal = compute_normal(-grad_c)

    # Contact-angle boundary condition: reorient the outward normal next to a
    # wall.  |grad(c)| is preserved, so delta_s below is unchanged and only the
    # curvature (hence the force direction) is affected.
    if wall is not None and cfg.contact_angle is not None and wall.numel() > 0:
        normal = apply_contact_angle_boundary(
            normal, pos, wall, h, float(cfg.contact_angle))

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
