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

Contact angle boundary condition:
    When a fluid interface meets a solid wall, the contact angle theta is
    enforced by rotating the color gradient of the near-wall particles so
    that  n . n_wall = cos(theta),  with n_wall the wall normal pointing
    into the fluid (Tartakovsky & Meakin 2005; Adami et al. 2012). The
    rotation preserves |grad c|, so the CSF force magnitude is unchanged
    and only its direction at the wall is corrected.

References:
    Brackbill, Kothe & Zemach (1992): "A continuum surface force method"
    Monaghan (1992): "Smoothed particle hydrodynamics"
    Tartakovsky et al. (2005): "Modeling of physically adsorbing surfaces"
    Adami, Hu & Adams (2012): "A generalized wall boundary condition for SPH"
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



# Tangential fraction below which the gradient counts as parallel to the
# wall normal (no defined tilt direction).
TANGENT_TOL = 1.0e-4


def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall_positions: torch.Tensor,
    h: float,
    contact_angle_rad: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Rotate the color gradient near a wall so that ``n . n_wall = cos(theta)``.

    The wall normal ``n_wall`` points from the nearest wall particle into the
    fluid. For each particle within ``2h`` of the wall the gradient is split into
    its wall-normal and wall-tangential parts and reassembled as

        n_new = cos(theta) * n_wall + sin(theta) * t_hat,

    where ``t_hat`` is the unit tangential direction of the *existing* gradient.
    ``|n_new| = 1`` because the two basis vectors are orthonormal, so the
    magnitude of ``grad_c`` -- and therefore ``|grad c| = delta_s`` in the CSF
    force -- is preserved: only the direction at the wall is corrected.

    Two degenerate cases are left untouched rather than guessed at:

    * ``|grad_c|`` is zero (no interface at that particle), and
    * the gradient is parallel to the wall normal, so the tangential direction
      and hence the sign of the tilt are undefined. The previous revision
      collapsed the magnitude to ``|cos theta|`` here, which silently deleted the
      interface; the tests only checked ``|grad|`` preservation, so the failure
      was invisible.

    Args:
        grad_c: (N, 2) color gradient; not modified in place
        pos: (N, 2) fluid particle positions
        wall_positions: (M, 2) wall particle positions
        h: smoothing length
        contact_angle_rad: contact angle in radians, 0 < theta < pi
            (theta < pi/2 wetting, pi/2 neutral, > pi/2 non-wetting)
        eps: guard against division by zero

    Returns:
        (N, 2) gradient with the contact angle enforced near the wall.
    """
    out = grad_c.clone()
    if wall_positions.numel() == 0 or out.numel() == 0:
        return out
    if wall_positions.shape[0] > 5000:
        return _apply_contact_angle_by_bounds(out, pos, h, contact_angle_rad, eps)

    cos_theta = torch.cos(torch.tensor(contact_angle_rad, dtype=pos.dtype,
                                       device=pos.device))
    sin_theta = torch.sin(torch.tensor(contact_angle_rad, dtype=pos.dtype,
                                       device=pos.device))

    # Pair each fluid particle with its nearest wall particle (chunked over the
    # fluid side so the (N, M) distance matrix never has to exist in full).
    min_dist = torch.full((pos.shape[0],), float("inf"), dtype=pos.dtype,
                          device=pos.device)
    nearest_wall = torch.zeros_like(pos)
    chunk = max(1, int(2e6 // max(1, wall_positions.shape[0])))
    for start in range(0, pos.shape[0], chunk):
        block = pos[start:start + chunk]
        diff = block[:, None, :] - wall_positions[None, :, :]
        dist_sq = (diff * diff).sum(dim=2)
        best = dist_sq.argmin(dim=1)
        min_dist[start:start + chunk] = dist_sq.gather(1, best[:, None]).sqrt().squeeze(1)
        nearest_wall[start:start + chunk] = wall_positions[best]

    near_wall = min_dist < 2.0 * h
    if not bool(near_wall.any()):
        return out

    wall_normal = pos - nearest_wall
    # clamp_min, not "+ eps": adding eps leaves |n_wall| = 1 - eps/|v|, i.e. a
    # 1e-6 residual for the default parameters, and that residual survived the
    # tangent projection as a fake tangential direction.
    wall_normal = wall_normal / torch.linalg.vector_norm(
        wall_normal, dim=1, keepdim=True).clamp_min(eps)

    grad_mag = torch.linalg.vector_norm(out, dim=1, keepdim=True)
    has_interface = (grad_mag > eps).squeeze(1)
    current_normal = out / grad_mag.clamp_min(eps)

    dot = (current_normal * wall_normal).sum(dim=1, keepdim=True)
    tangent = current_normal - dot * wall_normal
    tangent_mag = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    # Relative test: a tangential component below TANGENT_TOL is round-off from a
    # gradient that is parallel to the wall normal, where the tilt has no defined
    # direction. Dividing such a residue by a clamped magnitude produces a
    # "unit" vector of magnitude 1/tol and inflates |grad c| by two orders of
    # magnitude, so the case is skipped instead of amplified.
    has_tangent = (tangent_mag > TANGENT_TOL).squeeze(1)
    tangent = tangent / tangent_mag.clamp_min(TANGENT_TOL)

    new_normal = (cos_theta * wall_normal
                  + sin_theta * torch.where(has_tangent[:, None], tangent,
                                            torch.zeros_like(tangent)))
    mask = near_wall & has_interface & has_tangent
    out = torch.where(mask[:, None], grad_mag * new_normal, out)
    return out


def _apply_contact_angle_by_bounds(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    h: float,
    contact_angle_rad: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Bounded-memory fallback: infer the walls from the domain bounds.

    Used when the wall cloud is large enough that the nearest-wall search would
    dominate the run. It assumes the pool occupies ``[0, x_max] x [0, inf)``,
    which is how ``Config`` and ``wall_lattice`` build every case in this
    repository; it is not a general geometry. Same rotation and same degenerate
    cases as :func:`apply_contact_angle_boundary`.
    """
    cos_theta = torch.cos(torch.tensor(contact_angle_rad, dtype=pos.dtype,
                                       device=pos.device))
    sin_theta = torch.sin(torch.tensor(contact_angle_rad, dtype=pos.dtype,
                                       device=pos.device))

    near_left = pos[:, 0] < 2.0 * h
    near_right = pos[:, 0] > pos[:, 0].max() - 2.0 * h
    near_bottom = pos[:, 1] < 2.0 * h
    near_wall = near_left | near_right | near_bottom
    if not bool(near_wall.any()):
        return grad_c.clone()

    wall_normal = torch.zeros_like(pos)
    wall_normal[near_left, 0] = 1.0
    wall_normal[near_right, 0] = -1.0
    wall_normal[near_bottom, 1] = 1.0
    wall_normal = wall_normal / torch.linalg.vector_norm(
        wall_normal, dim=1, keepdim=True).clamp_min(eps)

    grad_mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    has_interface = (grad_mag > eps).squeeze(1)
    current_normal = grad_c / grad_mag.clamp_min(eps)
    dot = (current_normal * wall_normal).sum(dim=1, keepdim=True)
    tangent = current_normal - dot * wall_normal
    tangent_mag = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    has_tangent = (tangent_mag > TANGENT_TOL).squeeze(1)
    tangent = tangent / tangent_mag.clamp_min(TANGENT_TOL)
    new_normal = (cos_theta * wall_normal
                  + sin_theta * torch.where(has_tangent[:, None], tangent,
                                            torch.zeros_like(tangent)))
    mask = near_wall & has_interface & has_tangent
    return torch.where(mask[:, None], grad_mag * new_normal, grad_c)


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
        wall: (M, 2) wall particle positions. Required only when
            ``cfg.contact_angle`` is set; it supplies the wall normal for the
            contact-angle rotation of the color gradient.

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

    # Contact angle: rotate the near-wall gradient before the normal and
    # the curvature are extracted from it, so the enforced direction is
    # what the CSF force actually sees.
    contact_angle = getattr(cfg, "contact_angle", None)
    if contact_angle is not None and wall is not None and wall.numel() > 0:
        grad_c = apply_contact_angle_boundary(grad_c, pos, wall, h, contact_angle)

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
