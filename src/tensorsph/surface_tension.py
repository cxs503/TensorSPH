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
from .csf_consistent import (  # noqa: F401  (re-exported: public API of this module)
    CSFState,
    apply_contact_angle_boundary,
    csf_fields,
    kernel_sum_color,
    mls_divergence,
    mls_gradient,
    surface_tension_acceleration,
)


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


# --------------------------------------------------------------------------- #
# Consistent interface fields: colour, normal, surface delta, curvature
# --------------------------------------------------------------------------- #
#
# These four fields are produced by one function so that the force, the tests
# and the verification benchmark cannot drift apart. The recipe differs from the
# neighbour-count heuristic that shipped in 0.8.0 in three independent places,
# each of which is needed for the 2-D Laplace law ``dp = sigma/R`` to hold:
#
# 1. **Colour field from kernel completeness**, ``c_i = sum_j V_j W_ij``, not
#    ``count_i / mean(count)``.  ``c`` is the fraction of a full kernel support
#    occupied by fluid: ``1`` in the interior, ``~1/2`` on a planar free surface
#    and ``0`` in vacuum. A neighbour count is a raw particle tally and carries
#    Poisson noise of order ``sqrt(n)`` into ``grad c``; the volume-weighted
#    kernel sum is a smooth volumetric measure whose gradient is the surface
#    area density of CSF by construction (see 3).
# 2. **One smoothing length for the whole interface**, ``h_c = 2 h`` by default
#    (``Config.surface_tension_h_factor``). ``n`` and ``kappa`` live on the
#    interface shell; with ``h_c = h = 1.3 dx`` that shell is two particles wide
#    and the difference-form derivatives are dominated by particle noise. The
#    curvature of a circle of radius ``R`` is only resolvable while ``h_c << R``,
#    which is what the benchmark ladder checks.
# 3. **Fluid-side normalisation of the surface delta**: the continuum identity
#    for a plane interface is ``integral_{-inf}^{0} |grad c| dxi = 1/2`` (the
#    colour drops from 1 to 0 across the interface and the kernel is even), but
#    only the fluid half is discretised. The momentum balance of the interface,
#    ``integral_fluid rho a . (-n) dxi = sigma kappa``, therefore requires
#    ``delta_s = 2 |grad c|``. Shipping ``|grad c|`` unnormalised loses a factor
#    of two before any modelling error is considered.
#
# Curvature uses the difference-form divergence of the unit normal,
# ``div n = sum_j V_j (n_j - n_i) . grad_i W_ij``, which is consistent for
# linear fields (so the operator was not the cause of the 0.8.0 defect -- the
# noisy normal was). The scalar support factor ``scale = -2 / sum_j V_j
# (x_i - x_j) . grad W`` equals one for a complete kernel and corrects the
# truncation of the support near the free surface; it is clamped so that a
# nearly empty neighbour list cannot amplify noise.

_EPS = 1.0e-12


def surface_smoothing_length(cfg: "Config") -> float:
    """Smoothing length ``h_c`` used for the interface fields (metres)."""
    factor = getattr(cfg, "surface_tension_h_factor", 2.0)
    return float(factor) * cfg.h_factor * cfg.dx


def _pair_support(cfg: "Config") -> float:
    """Cutoff of the pair list a caller is expected to supply."""
    return 2.0 * cfg.h_factor * cfg.dx


def interface_fields(
    pos: torch.Tensor,
    density: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    cfg: Config,
    wall: torch.Tensor | None = None,
    scheme: str | None = None,
) -> torch.Tensor:
    """Compute surface tension acceleration using the CSF model.

    Two schemes are available through ``cfg.csf_scheme`` (or the ``scheme``
    argument):

    ``"consistent"`` (default)
        kernel-sum colour function + first-order (MLS) consistent gradient and
        divergence, see :mod:`tensorsph.csf_consistent`.  This is the model that
        reproduces the Young-Laplace jump (docs/verification_surface_tension_laplace.md).
    ``"legacy"``
        neighbour-count colour function with the uncorrected SPH operators, kept
        for the A/B comparison and for reproducing pre-m2188 results.

    ``wall`` (particle positions) together with ``cfg.contact_angle`` activates
    the wall contact-angle boundary condition.

    For every particle within ``support`` of a wall particle the local wall
    frame (normal ``w``, tangent ``t``) is fitted to the nearby wall particles
    and the normal is replaced by

        n_new = |n| * ( sin(theta) * t_hat + cos(theta) * w_hat ),

    where ``t_hat`` is the tangential projection of the current normal. The
    result is blended with the incoming normal by a linear ramp in the wall
    distance, so the condition fades out at ``support`` instead of jumping.
    Particles whose normal is (numerically) parallel to the wall normal keep
    their normal: their tangential orientation is not defined by the fields.

    Args:
        grad_c: ``(N, 2)`` colour gradient; the outward normal is ``-grad_c``
            and the input is not modified.
        pos: ``(N, 2)`` particle positions.
        wall: ``(M, 2)`` wall particle positions (may be empty).
        h: smoothing length that sets the default support ``2 h``.
        theta: contact angle in radians, measured inside the liquid.
        support: override for the wall influence distance.

    Returns:
        ``(N, 2)`` gradient with the contact-angle boundary applied.
    """
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient
    scheme = scheme if scheme is not None else getattr(cfg, "csf_scheme", "consistent")

    dtype, device = grad_c.dtype, grad_c.device
    out = grad_c.clone()
    wall = wall.to(dtype=dtype, device=device)
    if support is None:
        support = 2.0 * h

    mag = torch.linalg.vector_norm(grad_c, dim=1)
    candidates = torch.nonzero(mag > _EPS, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return out

    # distance from every candidate particle to the closest wall particle
    d = torch.cdist(pos[candidates].to(dtype=dtype), wall).min(dim=1).values
    near = candidates[d < support]
    if near.numel() == 0:
        return out

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for idx in near.tolist():
        patch = wall[torch.cdist(pos[idx][None, :], wall)[0] < support]
        frame = _wall_frame(patch)
        if frame is None:
            continue
        _, w_hat, t_hat = frame
        p = pos[idx]
        # orient the fitted normal towards the fluid
        if float(torch.dot(p - patch.mean(dim=0), w_hat)) < 0.0:
            w_hat = -w_hat

        n = -grad_c[idx] / mag[idx]          # outward normal
        n_t = n - torch.dot(n, w_hat) * w_hat
        n_t_norm = torch.linalg.vector_norm(n_t)
        if float(n_t_norm) < 1.0e-6:
            continue                          # tangential orientation undefined
        target = sin_t * (n_t / n_t_norm) + cos_t * w_hat
        target = target / torch.linalg.vector_norm(target).clamp_min(_EPS)

        weight = float(1.0 - torch.cdist(p[None, :], wall).min() / support)
        weight = min(max(weight, 0.0), 1.0)
        blended = (1.0 - weight) * n + weight * target
        blended = blended / torch.linalg.vector_norm(blended).clamp_min(_EPS)

        out[idx] = -blended * mag[idx]
        del t_hat

    return out

    if scheme not in ("consistent", "legacy"):
        raise ValueError(
            f"unknown csf_scheme {scheme!r}; expected 'consistent' or 'legacy'")

    if scheme == "consistent":
        acc, _ = surface_tension_acceleration(
            pos, density, mass, pairs, h, sigma, rho0=cfg.rho0, wall=wall,
            contact_angle=getattr(cfg, "contact_angle", None))
        return acc

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

    fields = interface_fields(pos, density, mass, pairs, cfg)

    normal = fields["normal"]
    if wall is not None and getattr(cfg, "contact_angle", None) is not None:
        grad_c = apply_contact_angle_boundary(
            fields["grad_colour"], pos, wall, surface_smoothing_length(cfg),
            float(cfg.contact_angle))
        grad_mag = torch.linalg.vector_norm(grad_c, dim=1)
        normal = torch.where(grad_mag[:, None] > _EPS,
                             -grad_c / grad_mag.clamp_min(_EPS)[:, None], normal)

    force_mag = -sigma * fields["curvature"] * fields["delta_s"]
    return force_mag[:, None] * normal / density[:, None]

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
