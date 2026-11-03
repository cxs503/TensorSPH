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
        wall: (W, 2) optional wall particle positions; when given together with
            ``cfg.contact_angle`` the near-wall colour gradient is rotated by
            :func:`apply_contact_angle_boundary` so the interface meets the wall
            at the requested angle.

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

    # Contact-angle BC: steer the near-wall gradient before it sets the normal.
    if wall is not None and cfg.contact_angle is not None:
        grad_c = apply_contact_angle_boundary(grad_c, pos, wall, h, cfg.contact_angle)

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


# --------------------------------------------------------------------------- #
# Opt-in corrections for the CSF colour-function / surface-delta defects
# --------------------------------------------------------------------------- #
# The default path above builds the colour function from a neighbour-count
# ratio thresholded at 0.7 * mean. That field is a 1-particle-thick step: it is
# not a smooth indicator, so |grad c| is an arbitrary scale rather than the
# surface delta function, and div(n) of its normal is noise. Five independent
# missions (m2142, m2146, m2174, m2229, m2233) measured the consequences; none
# corrected the field. The functions below are the corrected pair and are
# opt-in: ``compute_surface_tension_force`` keeps its historical behaviour so
# that no existing verification record silently changes meaning.
#
# The two corrections are independent and each is scored separately in
# ``benchmarks/laplace_rootcause``:
#
# 1. ``free_surface_color`` -- Shepard-normalised kernel occupancy deficiency
#        c_i = 1 - sum_j V_j W_ij,      V_j = m_j / rho_j
#    The partition of unity sum_j V_j W_ij -> 1 in the interior and -> 0 for an
#    isolated particle, so c is dimensionless, monotone outward and varies over
#    ~2h instead of one lattice spacing. This is the standard free-surface
#    deficiency indicator; it is not a phase indicator.
#
# 2. ``normalise_delta_s`` -- CSF requires the surface delta function to satisfy
#        int delta_s dn = 1
#    across the interface. For a colour function whose range is
#    ``c_max - c_min`` rather than {0, 1}, the compensating normalisation is
#        delta_s = |grad c| / (c_max - c_min)
#    measured with robust percentiles so that a handful of isolated particles
#    cannot set the scale.


def free_surface_color(
    pos: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Shepard-normalised kernel occupancy deficiency used as colour function.

    Args:
        pos: (N, 2) particle positions
        mass: (N,) particle masses
        density: (N,) particle densities
        pairs: (P, 2) unordered neighbour pairs (each pair once)
        h: smoothing length

    Returns:
        color: (N,) deficiency in [0, 1]; ~0 in the interior, ~0.5 on a flat
        free surface, -> 1 for an isolated particle.

    Notes:
        The self contribution W(0) is included analytically, so a particle with
        a full neighbour shell returns exactly 0 rather than the truncation
        error of the pair sum.
    """
    n = pos.shape[0]
    color = torch.zeros(n, dtype=pos.dtype, device=pos.device)

    # self term: V_i * W(0, h)
    w0 = cubic_spline_kernel(torch.zeros(1, dtype=pos.dtype, device=pos.device), h)
    color = color + (mass / density) * w0[0]

    if pairs.numel() > 0:
        i, j = pairs.unbind(dim=1)
        delta = pos[i] - pos[j]
        w = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
        vol = mass / density
        occ = torch.zeros(n, dtype=pos.dtype, device=pos.device)
        occ = occ.index_add(0, i, vol[j] * w)
        occ = occ.index_add(0, j, vol[i] * w)
        color = color + occ

    return (1.0 - color).clamp(0.0, 1.0)


def normalise_delta_s(
    delta_s: torch.Tensor,
    color: torch.Tensor,
    lower: float = 0.01,
    upper: float = 0.99,
) -> torch.Tensor:
    """Scale ``|grad c|`` by the colour-function range so that int delta_s dn = 1.

    Args:
        delta_s: (N,) raw |grad c|
        color: (N,) the colour function that produced it
        lower, upper: quantiles used for the range; robust to isolated particles.

    Returns:
        delta_s_norm: (N,) surface delta function consistent with the colour
        function's actual jump.
    """
    span = torch.quantile(color, torch.tensor(upper, dtype=color.dtype)) - torch.quantile(
        color, torch.tensor(lower, dtype=color.dtype)
    )
    return delta_s / span.clamp_min(1.0e-12)


def compute_surface_tension_force_corrected(
    pos: torch.Tensor,
    density: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    cfg: Config,
    fields: dict | None = None,
) -> torch.Tensor:
    """CSF surface tension with the corrected colour function and delta_s.

    Same force expression as :func:`compute_surface_tension_force`
    (``f = -sigma * kappa * n * delta_s``) and the same sign convention, so the
    two differ only in how ``c``, ``n`` and ``delta_s`` are built. That is the
    point: the factorial probe can then attribute a change in the measured
    Laplace/momentum criteria to a specific ingredient.

    Args:
        pos, density, mass, pairs, cfg: as for the uncorrected routine
        fields: optional dict filled in place with the intermediate fields, so a
            benchmark can score each ingredient without recomputing.

    Returns:
        acc_st: (N, 2) surface tension acceleration
    """
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient
    n = pos.shape[0]
    if sigma <= 0.0 or pairs.numel() == 0:
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    color = free_surface_color(pos, mass, density, pairs, h)
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    # c grows outward (interior 0 -> surface 0.5), so grad(c) already points
    # from fluid to vacuum and needs no negation.
    normal = compute_normal(grad_c)
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)
    delta_s = normalise_delta_s(torch.linalg.vector_norm(grad_c, dim=1), color)

    force_mag = -sigma * kappa * delta_s
    acc_st = force_mag[:, None] * normal / density[:, None]

    if fields is not None:
        fields.update(
            color=color, grad_c=grad_c, normal=normal, kappa=kappa,
            delta_s=delta_s, acc_st=acc_st,
        )
    return acc_st


# --------------------------------------------------------------------------- #
# Wall contact-angle boundary condition
# --------------------------------------------------------------------------- #
# The colour gradient of a free surface that meets a wall is set by the CSF
# force on the *fluid* particles only: no wall-fluid term carries the wetting
# information, so the modelled interface always meets the wall at 90 deg and the
# configured ``Config.contact_angle`` had no effect. ``apply_contact_angle_boundary``
# rotates the near-wall colour gradient so that the interface normal makes the
# requested angle with the wall normal, which is the standard way of imposing a
# static contact angle on a CSF colour function without resolving the precursor
# film.
#
# The transformation is deliberately magnitude preserving:
#
#     g = g_n n_w + g_t,   |target| = |g|,   result = slerp(g, target, w)
#
# so ``delta_s = |grad c|`` -- and therefore the capillary force scale -- is
# untouched and only the direction is steered. ``w`` ramps from 1 on the wall to
# 0 at ``2h``, so particles further than one support radius and empty wall sets
# are returned unchanged. The target satisfies
#
#     |cos angle(n, n_w)| = cos(theta),
#
# i.e. theta < pi/2 tilts the normal *towards* the wall normal (wetting) and
# theta > pi/2 tilts it away, with theta = pi/2 giving a wall-normal interface.


def _wall_normal(
    points: torch.Tensor,
    wall_pos: torch.Tensor,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locally fitted wall normal, wall distance and wall centroid for ``points``.

    The normal is the smallest-eigenvector direction of the covariance of the
    wall particles within ``2h``, oriented towards the fluid. A least-squares
    plane fit is used rather than "direction to the nearest wall particle",
    because for a flat wall the latter returns the in-plane offset and would
    make the contact angle depend on which wall particle happens to be closest.

    Returns:
        (normal (M,2), distance (M,), centroid (M,2)); the normal is zero where
        fewer than two wall points are in range.
    """
    m = points.shape[0]
    dtype, device = points.dtype, points.device
    normal = torch.zeros((m, 2), dtype=dtype, device=device)
    dist = torch.full((m,), float("inf"), dtype=dtype, device=device)
    centroid = points.clone()

    if wall_pos.numel() == 0 or m == 0:
        return normal, dist, centroid

    # The wall lattice is routinely built in the default float32 while the
    # solution runs in float64; cdist rejects mixed dtypes.
    wall_pos = wall_pos.to(dtype=dtype, device=device)
    d = torch.cdist(points, wall_pos)          # (M, W)
    near = d <= 2.0 * h
    nearest = d.argmin(dim=1)
    dist = d.gather(1, nearest[:, None]).squeeze(1)

    for k in range(m):
        idx = torch.nonzero(near[k], as_tuple=False).squeeze(1)
        if idx.numel() < 2:
            continue
        pts = wall_pos[idx]
        mu = pts.mean(dim=0)
        centred = pts - mu
        cov = centred.T @ centred / max(int(idx.numel()) - 1, 1)
        vals, vecs = torch.linalg.eigh(cov)
        n = vecs[:, 0]                          # smallest eigenvalue
        if torch.dot(n, points[k] - mu) < 0.0:  # orient towards the fluid
            n = -n
        normal[k] = n
        centroid[k] = mu
        # Perpendicular distance to the fitted wall plane. Measuring to the
        # nearest wall *particle* instead quantises the distance at half the wall
        # lattice spacing, so the blend weight would never reach 1 on the wall.
        dist[k] = torch.dot(points[k] - mu, n).abs()
    return normal, dist, centroid


def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall_pos: torch.Tensor,
    h: float,
    theta: float,
) -> torch.Tensor:
    """Rotate the near-wall colour gradient to impose a static contact angle.

    Args:
        grad_c: (N, dim) colour gradient, modified copy returned
        pos: (N, dim) fluid particle positions
        wall_pos: (W, dim) wall particle positions; empty disables the BC
        h: smoothing length; only particles within ``2h`` of the wall are touched
        theta: contact angle in radians, measured through the fluid
            (``pi/2`` = neutral, ``< pi/2`` = wetting, ``> pi/2`` = non-wetting)

    Returns:
        (N, dim) gradient with the same magnitudes, reference unchanged.

    Notes:
        ``theta = pi/2`` makes the interface normal tangential to the wall, which
        is what a neutral contact angle means; it is therefore *not* the identity
        for a gradient that already points along the wall normal. Pass
        ``cfg.contact_angle = None`` to recover the historical 90 deg-imposed
        behaviour exactly.
    """
    if wall_pos is None or wall_pos.numel() == 0 or pos.numel() == 0:
        return grad_c

    out = grad_c.clone()
    normal, dist, centroid = _wall_normal(pos, wall_pos, h)
    active = (normal.norm(dim=1) > 0.0) & (dist < 2.0 * h)
    if not bool(active.any()):
        return out

    idx = torch.nonzero(active, as_tuple=False).squeeze(1)
    n_w = normal[idx]
    g = grad_c[idx]
    mag = torch.linalg.vector_norm(g, dim=1)
    live = mag > 1.0e-12
    if not bool(live.any()):
        return out

    idx = idx[live]
    n_w, g, mag = n_w[live], g[live], mag[live]
    g_hat = g / mag[:, None]

    g_n = (g_hat * n_w).sum(dim=1, keepdim=True)
    t_vec = g_hat - g_n * n_w
    t_mag = torch.linalg.vector_norm(t_vec, dim=1, keepdim=True)

    # Tangential basis: keep the existing in-plane direction where it exists,
    # otherwise spread outwards along the wall from the wall patch centroid.
    offset = pos[idx] - centroid[idx]
    offset = offset - (offset * n_w).sum(dim=1, keepdim=True) * n_w
    fallback = offset / torch.linalg.vector_norm(offset, dim=1, keepdim=True).clamp_min(1.0e-12)
    degenerate = torch.linalg.vector_norm(offset, dim=1, keepdim=True) < 1.0e-12
    if bool(degenerate.any()):
        basis = torch.zeros_like(n_w)
        basis[:, 0] = 1.0
        tangent_basis = basis - (basis * n_w).sum(dim=1, keepdim=True) * n_w
        fallback = torch.where(degenerate, tangent_basis, fallback)
    e_t = torch.where(t_mag > 1.0e-12, t_vec / t_mag.clamp_min(1.0e-12), fallback)

    # Target normal direction: |n.n_w| = cos(theta), sign kept from the input.
    cos_t = math.cos(float(theta))
    sin_t = math.sin(float(theta))
    sign = torch.where(g_n >= 0.0, torch.ones_like(g_n), -torch.ones_like(g_n))
    target = sign * cos_t * n_w + sin_t * e_t

    # Blend by wall proximity: identity at 2h, full BC on the wall.
    w = (1.0 - dist[idx] / (2.0 * h)).clamp(0.0, 1.0)[:, None]
    mixed = (1.0 - w) * g_hat + w * target
    mixed_mag = torch.linalg.vector_norm(mixed, dim=1, keepdim=True)
    bad = mixed_mag < 1.0e-12
    mixed = torch.where(bad, g_hat, mixed / mixed_mag.clamp_min(1.0e-12))

    out[idx] = mixed * mag[:, None]
    return out