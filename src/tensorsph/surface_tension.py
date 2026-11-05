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
    # For particle j: V_i * (n_j - n_i) . grad_W_ij
    # (note: grad_W_ji = -grad_W_ij, but we use the same pair)
    kappa = kappa.index_add(0, i, vol_j * div_n)
    kappa = kappa.index_add(0, j, vol_i * div_n)

    return kappa


def kernel_sum_color_field(
    pos: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Phase indicator from the kernel sum, ``c_i = sum_j V_j W_ij``.

    Local-density variant (``V_j = m_j / rho_j``) used by the ``"kernel_sum"``
    colour model. The consistent scheme has its own kernel-sum colour
    (``tensorsph.csf_consistent.kernel_sum_color``, ``V_j = m_j / rho0``,
    clipped to [0, 1]); the two names are distinct because their
    signatures differ.

    With a kernel normalised to unit area and volumes ``V_j = m_j / rho_j`` the
    discrete sum is ~1 in the bulk, ~0.5 on a free surface (half the support
    missing) and 0 outside the fluid. Unlike the neighbour count this is a
    *smooth* function of position, so ``grad(c)`` -- and therefore the CSF
    normal, the curvature and the surface delta function -- is defined
    everywhere instead of jumping between two discrete levels.
    """
    n = pos.shape[0]
    zero = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    color = mass * cubic_spline_kernel(zero, h) / density
    if pairs.numel() == 0:
        return color
    i, j = pairs.unbind(dim=1)
    rij = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(rij, h)
    color = color.index_add(0, i, mass[j] * w / density[j])
    color = color.index_add(0, j, mass[i] * w / density[i])
    return color


def support_vector(
    pos: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Missing-support vector ``s_i = sum_j V_j grad_i W_ij``.

    This is the gradient of a *constant* colour field evaluated without
    differencing: for a particle with full support the gradients cancel, and for
    one on a free surface the half-missing support is what remains. ``s`` points
    *into* the fluid, and ``|s|`` is the surface delta function of the CSF model
    (zero in the bulk, maximum on the interface).

    Evaluating ``grad(c)`` by differencing a colour that is built from a
    neighbour count or a kernel sum does not work at this resolution: the two
    fields differ by O(0.2) across the interface while their lattice noise is of
    the same order, so the differenced gradient is not correlated with the
    surface normal (measured correlation with ``r_hat``: -0.02 for the neighbour
    count, -0.01 for the kernel sum, versus -0.92 here). See
    ``docs/benchmarks/surface-tension.md``.
    """
    i, j = pairs.unbind(dim=1)
    grad_w = cubic_spline_gradient(pos[i] - pos[j], h)
    vol_j = mass[j] / density[j]
    vol_i = mass[i] / density[i]
    s = torch.zeros((pos.shape[0], 2), dtype=pos.dtype, device=pos.device)
    s = s.index_add(0, i, vol_j[:, None] * grad_w)
    s = s.index_add(0, j, -vol_i[:, None] * grad_w)
    return s


def smooth_vector_field(
    vec: torch.Tensor,
    pos: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
    passes: int,
) -> torch.Tensor:
    """Shepard-weighted kernel smoothing of a vector field, in place per pass.

    ``n`` is obtained by normalising ``s``; the raw ``s`` is computed from a
    *truncated* kernel support and is therefore under-resolved exactly where it
    matters, so its unit vector carries lattice-scale noise and ``div(n)``
    amplifies that noise instead of returning the curvature. Each pass is
    ``v_i <- sum_j V_j v_j W_ij / sum_j V_j W_ij``, which leaves a constant
    field unchanged and a linear field unchanged to kernel accuracy, so it
    smooths the noise without moving the surface.
    """
    if passes <= 0 or pairs.numel() == 0:
        return vec
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel(torch.linalg.vector_norm(pos[i] - pos[j], dim=1), h)
    vol_i = mass[i] / density[i]
    vol_j = mass[j] / density[j]
    out = vec
    for _ in range(passes):
        acc = torch.zeros_like(out)
        acc = acc.index_add(0, i, (vol_j * w)[:, None] * out[j])
        acc = acc.index_add(0, j, (vol_i * w)[:, None] * out[i])
        wsum = torch.zeros(out.shape[0], dtype=out.dtype, device=out.device)
        wsum = wsum.index_add(0, i, vol_j * w)
        wsum = wsum.index_add(0, j, vol_i * w)
        out = acc / wsum.clamp_min(1.0e-12)[:, None]
    return out


def surface_tension_field(
    pos: torch.Tensor,
    density: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    cfg,
    color: str | None = None,
):
    """CSF field quantities, separated from the acceleration they produce.

    Returns ``(normal, kappa, delta_s, acc)``:

    * ``normal``  -- unit normal pointing out of the fluid;
    * ``kappa``   -- curvature ``div(n)``, positive for a convex droplet;
    * ``delta_s`` -- surface delta function ``|grad(c)|``;
    * ``acc``     -- ``-sigma * kappa * delta_s * n / rho``, the acceleration
      ``compute_surface_tension_force`` returns.

    Exposing the three field quantities is what makes the model verifiable: a
    benchmark can compare ``kappa`` with ``1/R`` and the integral of
    ``sigma * kappa * delta_s`` with the Laplace pressure ``sigma/R`` without
    having to re-derive the discretisation outside the repository.

    ``cfg.surface_tension_color`` selects the colour function:

    * ``"neighbor_count"`` -- the historical heuristic: interior particles get
      ``c = 1`` and particles whose neighbour count is below 70 % of the mean
      get ``c = count / mean``. Two-valued, hence not differentiable.
    * ``"kernel_sum"`` -- ``c = sum_j V_j W_ij`` (see
      :func:`kernel_sum_color_field`).
    * ``"support_sum"`` -- the normal comes from the missing-support vector of
      :func:`support_vector` after ``cfg.surface_tension_smoothing`` smoothing
      passes, and the surface delta function is ``|s|``. This is the only one of
      the three that reproduces the Laplace pressure of a static droplet; the
      measured comparison is in ``docs/benchmarks/surface-tension.md``.

    ``color`` overrides ``cfg.surface_tension_color`` for this call; the
    ``"legacy"`` alias of :func:`compute_surface_tension_force` uses it to pin
    the historical neighbour-count colour.
    """
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient
    zero_vec = torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)
    zero_scalar = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    if sigma <= 0.0 or pairs.numel() == 0:
        return zero_vec, zero_scalar, zero_scalar, zero_vec

    model = color if color is not None else getattr(
        cfg, "surface_tension_color", "neighbor_count")
    if model == "support_sum":
        s = support_vector(pos, mass, density, pairs, h)
        delta_s = torch.linalg.vector_norm(s, dim=1)
        # The normal is taken from the *smoothed* support vector: dividing the
        # raw one by its own magnitude amplifies lattice noise, and the
        # resulting div(n) does not converge (measured 6.4x / 9.4x / 13.1x the
        # analytic curvature at R/dx = 8 / 12 / 16, versus 1.04 / 1.02 / 1.02
        # with the smoothing below). |s| itself is left unsmoothed so that the
        # force sheet keeps the thickness the kernel gives it.
        delta_s = delta_s * float(getattr(cfg, "surface_tension_delta_scale", 1.0))
        passes = int(getattr(cfg, "surface_tension_smoothing", 4))
        s_smooth = smooth_vector_field(s, pos, mass, density, pairs, h, passes)
        normal = compute_normal(-s_smooth)
        kappa = compute_curvature(pos, normal, mass, density, pairs, h)
        acc = (-sigma * kappa * delta_s)[:, None] * normal / density[:, None]
        return normal, kappa, delta_s, acc
    if model == "kernel_sum":
        color = kernel_sum_color_field(pos, mass, density, pairs, h)
    elif model == "neighbor_count":
        i, j = pairs.unbind(dim=1)
        count = torch.zeros(n, dtype=pos.dtype, device=pos.device)
        count = count.index_add(0, i, torch.ones_like(i, dtype=pos.dtype))
        count = count.index_add(0, j, torch.ones_like(j, dtype=pos.dtype))
        avg = count.mean()
        if float(avg) <= 0.0:
            return zero_vec, zero_scalar, zero_scalar, zero_vec
        is_surface = count < 0.7 * avg
        if not bool(is_surface.any()):
            return zero_vec, zero_scalar, zero_scalar, zero_vec
        color = torch.ones(n, dtype=pos.dtype, device=pos.device)
        color = torch.where(is_surface, count / avg, color)
    else:
        raise ValueError(
            f"surface_tension_color must be 'neighbor_count' or 'kernel_sum', "
            f"got {model!r}")

    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    # c is maximal inside the fluid and decays outward, so -grad(c) points out.
    normal = compute_normal(-grad_c)
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    acc = (-sigma * kappa * delta_s)[:, None] * normal / density[:, None]
    return normal, kappa, delta_s, acc


def compute_surface_tension_force(
    pos: torch.Tensor,
    density: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    cfg: Config,
    wall: torch.Tensor | None = None,
    scheme: str | None = None,
) -> torch.Tensor:
    """Compute surface-tension acceleration using the CSF model.

    Two model families are available:

    ``"consistent"`` (default)
        kernel-sum colour function + first-order (MLS) consistent gradient and
        divergence, see :mod:`tensorsph.csf_consistent`.  This is the model that
        reproduces the Young-Laplace jump
        (docs/verification_surface_tension_laplace.md).
    ``"colour"`` (alias ``"legacy"``)
        the 0.9.0 colour-function study path, :func:`surface_tension_field`.
        ``cfg.surface_tension_color`` selects the colour function:
        ``"neighbor_count"`` is the historical two-valued heuristic,
        ``"kernel_sum"`` the smooth kernel-sum indicator and ``"support_sum"``
        the missing-support normal (the only one that reproduces the Laplace
        pressure at the resolutions measured in
        ``docs/benchmarks/surface-tension.md``).  ``"legacy"`` always uses
        ``"neighbor_count"``, i.e. exactly the pre-0.9.0 behaviour, and is kept
        for the A/B comparison and for reproducing pre-m2188 results.

    Selection order, so that neither branch is silent about which model runs:

    1. an explicit ``scheme`` argument wins;
    2. otherwise a non-default ``cfg.surface_tension_color`` is an explicit
       request for the colour-function path (the consistent scheme carries its
       own colour function and ignores that setting);
    3. otherwise ``cfg.csf_scheme`` (``"consistent"`` by default).

    ``wall`` (particle positions) together with ``cfg.contact_angle`` activates
    the wall contact-angle boundary condition in the consistent scheme.
    """
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient

    if scheme is None:
        scheme = getattr(cfg, "csf_scheme", "consistent")
        color = getattr(cfg, "surface_tension_color", "neighbor_count")
        if scheme != "legacy" and color != "neighbor_count":
            scheme = "colour"

    if scheme not in ("consistent", "colour", "legacy"):
        raise ValueError(
            f"unknown csf_scheme {scheme!r}; expected 'consistent', 'colour' or "
            f"'legacy'")

    if sigma <= 0.0 or pairs.numel() == 0:
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    if scheme == "consistent":
        acc, _ = surface_tension_acceleration(
            pos, density, mass, pairs, h, sigma, rho0=cfg.rho0, wall=wall,
            contact_angle=getattr(cfg, "contact_angle", None),
            curvature_scheme=getattr(cfg, "csf_curvature", "fit_quadratic"))
        return acc

    # Colour-function family.  "legacy" pins the historical neighbour count so
    # that the alias cannot silently change what it measures.
    color = "neighbor_count" if scheme == "legacy" else None
    return surface_tension_field(
        pos, density, mass, pairs, cfg, color=color)[3]


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
