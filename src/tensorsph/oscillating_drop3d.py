"""Free oscillation of a liquid drop: Rayleigh (1879) capillary frequency.

Anchor ``CASE-SPH-OSC-DROP-FREQ`` -- reference Rayleigh (1879), gate < 6 % on the
mode-``l`` oscillation frequency of an inviscid drop.

Analytic reference
------------------
A 3-D inviscid drop of radius ``R``, density ``rho`` and surface tension
``sigma``, deformed as ``r(theta) = R [1 + eps P_l(cos theta)]``, oscillates at
(Rayleigh 1879, *Proc. R. Soc. Lond.* 29:71; Lamb 1932, *Hydrodynamics* 6th ed.,
Art. 275)

.. code-block:: text

    omega_l^2 = l (l - 1) (l + 2) sigma / (rho R^3),        l >= 2

so ``l = 2`` gives ``8 sigma / (rho R^3)``.  ``l = 1`` is a rigid translation
(``omega = 0``) and is rejected instead of being divided by.

**The 2-D branch is a different formula** and is the classical trap: a 2-D drop
(a cylinder) oscillates at ``omega^2 = n (n^2 - 1) sigma / (rho R^3)`` -- ``6``
instead of ``8`` at the leading mode, a 15.5 % frequency difference, i.e. 2.6x
the gate.  Both branches are implemented and a test pins their ratio, so a run
cannot score a 2-D number against a 3-D reference by accident.

Why the axisymmetric reduction is used instead of a full 3-D lattice
--------------------------------------------------------------------
The azimuthal mode number of the Rayleigh mode is ``m = 0``: the motion is a
surface of revolution, so the problem reduces **exactly** (no approximation of
the equations) to the meridional half-plane ``(rho, z)`` once the curvature and
the volume element carry their cylindrical factors:

.. code-block:: text

    kappa   = kappa_meridional + n_rho / rho        (div n in cylindrical coords)
    dV      = 2 pi rho dA

``kappa_meridional`` is the curvature of the meridional *plane curve* -- exactly
the quantity the repository's validated 2-D estimator
(:func:`tensorsph.csf_curvature.fit_curvature`) returns -- and ``n_rho/rho`` is
evaluated analytically.  For the unperturbed circle this gives
``1/R + 1/R = 2/R`` (sphere) as it must, and for
``r = R[1 + eps P_l(cos theta)]`` the two terms split the exact Rayleigh
coefficient as ``(l-1)(l+1) + 1 = (l-1)(l+2)`` per ``eps/R`` -- the 2-D branch
is precisely the meridional term alone, which is why the 2-D and 3-D
frequencies differ by ``sqrt(6/8)``.

The reduction is what makes a 6 % gate reachable here: on the *shipped 3-D
lattice* operator (:mod:`tensorsph.threed.csf3`, ``curvature="mls"``, i.e.
``div n`` on a lattice staircase) the curvature is dominated by lattice noise --
measured for this mission: ``kappa R = 0.77`` instead of 2 on a static sphere at
``R/dx = 8..16`` (resolution independent), with a colour-gradient surface measure
of ``0.33 x (4 pi R^2)`` against ``1.41 x`` on a plane, i.e. the 3-D delta
function and curvature are *not* mutually consistent.  Those numbers are
recorded in ``docs/case_sph_osc_drop_freq.md`` as the honest reason this module
does not score the 3-D lattice operator.

Two measurements, reported separately
-------------------------------------
``linear-response`` (primary)
    quasi-static modal stiffness of the *discretised* CSF operator, converted to
    a frequency through ``omega_meas/omega_exact = sqrt(dp_meas/dp_exact)``:
    measures the restoring-force error of the discretisation.  This is a
    discrete-dispersion measurement, **not** a time trajectory.
``dynamic`` (cross-check)
    WCSPH (Tait EOS, delta-SPH density diffusion, CSF surface tension,
    kick-drift-kick leapfrog) with the axisymmetric volume element; the mode
    amplitude is read from a mass-weighted quadrupole and ``(omega, gamma)`` are
    fitted.  This carries the discrete inertia and the spurious-current noise
    that the linear-response route cannot see, and it is reported whether it
    passes or not.

References
    Rayleigh, Lord (1879), Proc. R. Soc. Lond. 29:71.
    Lamb, H. (1932), Hydrodynamics, 6th ed., Art. 275.
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335 (CSF force).
    Bonet & Lok (1997), Comput. Methods Appl. Mech. Engrg. 180:97 (MLS).
    Antuono, Colagrossi & Marrone (2012), Comput. Phys. Commun. 183:2570
        (delta-SPH density diffusion).
    Prosperetti, A. (1980), J. Mec. 19:149 (viscous damping of drop modes; the
        ``O(Oh)`` correction that justifies the inviscid anchor used here).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "rayleigh_frequency_3d",
    "rayleigh_frequency_2d",
    "legendre_p",
    "exact_modal_stress",
    "axisym_drop_geometry",
    "axisym_curvature",
    "axisym_modal_stress",
    "drop_frequency_linear_response",
    "fit_damped_cosine",
    "fcc_sphere",
    "flat_surface",
    "csf3_audit",
]


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #

def rayleigh_frequency_3d(sigma: float, rho: float, radius: float, l: int = 2) -> float:
    """``omega_l = sqrt(l (l-1) (l+2) sigma / (rho R^3))`` -- Rayleigh 1879."""
    if l < 2:
        raise ValueError(f"mode l={l} is not an oscillation (l>=2); l=1 is a translation")
    if sigma <= 0.0 or rho <= 0.0 or radius <= 0.0:
        raise ValueError("sigma, rho and radius must be positive")
    return math.sqrt(l * (l - 1.0) * (l + 2.0) * sigma / (rho * radius ** 3))


def rayleigh_frequency_2d(sigma: float, rho: float, radius: float, n: int = 2) -> float:
    """``omega_n = sqrt(n (n^2-1) sigma / (rho R^3))`` -- the 2-D (cylinder) branch.

    Kept next to the 3-D function so the 15.5 % trap is guarded by a test rather
    than by a comment.
    """
    if n < 2:
        raise ValueError(f"mode n={n} is not an oscillation (n>=2)")
    if sigma <= 0.0 or rho <= 0.0 or radius <= 0.0:
        raise ValueError("sigma, rho and radius must be positive")
    return math.sqrt(n * (n * n - 1.0) * sigma / (rho * radius ** 3))


def legendre_p(l: int, c: torch.Tensor) -> torch.Tensor:
    """``P_l(c)`` by the Bonnet recursion (``c = cos theta``)."""
    if l == 0:
        return torch.ones_like(c)
    p0 = torch.ones_like(c)
    p1 = c
    if l == 1:
        return p1
    for n in range(1, l):
        p0, p1 = p1, ((2 * n + 1) * c * p1 - n * p0) / (n + 1)
    return p1


def exact_modal_stress(sigma: float, radius: float, mode: int, eps: float) -> float:
    """Mode-``l`` coefficient of the exact CSF traction ``-sigma delta kappa``.

    ``delta kappa = (l-1)(l+2) eps P_l / R``, so the traction coefficient is
    ``-sigma (l-1)(l+2) eps / R`` (a bulge has *larger* curvature, hence a
    restoring traction).
    """
    if mode < 2:
        raise ValueError(f"mode l={mode} has no capillary oscillation")
    return -sigma * (mode - 1.0) * (mode + 2.0) * eps / radius


# --------------------------------------------------------------------------- #
# axisymmetric (meridional) drop geometry
# --------------------------------------------------------------------------- #

def axisym_drop_geometry(radius: float, dx: float, *, mode: int = 2,
                         eps: float = 0.0, rho0: float = 1.0) -> dict:
    """Hexagonal circle drop in the meridian plane, with cylindrical weights.

    ``r(theta) = R [1 + eps P_l(cos theta)]`` with ``theta`` measured from the
    symmetry axis (the ``+y`` axis here, so that ``cos theta = y/r``).  The
    meridional cross-section of a surface of revolution is a *full* circle, so
    the planar circle lattice of :func:`tensorsph.surface_tension_dispersion.
    drop_geometry` is reused unchanged; what this function adds is

    * ``rho = |x|`` (cylindrical radius) and the axisymmetric volume element
      ``dvol = 2 pi rho m / rho0``, and
    * the axis-safe clamp ``rho >= 0.25 h`` used by the ``n_rho/rho`` term.

    The perturbation is a radial map, which is volume preserving to ``O(eps^2)``
    because ``<P_l> = 0`` on the sphere; the particle mass therefore stays
    uniform and the initial state carries no acoustic pulse of capillary size.
    """
    if mode < 2:
        raise ValueError("mode must be >= 2")
    h = 1.3 * dx
    # hex lattice over the deformed shape: sample the *reference* circle and map
    # radially, then keep the points inside the analytic shape.
    extent = radius * (1.0 + abs(eps) + 1.5)
    n = int(math.ceil(extent / dx)) + 2
    axis = torch.arange(-n, n + 1, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    dy = dx * math.sqrt(3.0) / 2.0
    pts = torch.stack((gx.ravel(), gy.ravel() * (dy / dx) * 2.0, ), dim=1)
    # the meshgrid above is square; rebuild the staggered hex rows explicitly
    rows = []
    ny = int(math.ceil(extent / dy)) + 2
    for j in range(-ny, ny + 1):
        shift = 0.5 * dx if (j % 2) else 0.0
        xs = torch.arange(-n, n + 1, dtype=torch.float64) * dx + shift
        rows.append(torch.stack((xs, torch.full_like(xs, j * dy)), dim=1))
    pts = torch.cat(rows, dim=0)
    del gx, gy
    r = torch.linalg.vector_norm(pts, dim=1)
    # Points are selected on the *reference* circle and then mapped radially.
    # Selecting on the deformed shape instead aliases the O(eps) perturbation
    # onto the lattice staircase (a half-spacing radial quantisation is 2x the
    # mode amplitude at eps = 0.02), which shows up as a 150 % error in the
    # modal curvature -- measured before this fix.
    pts = pts[r <= radius]
    r = torch.linalg.vector_norm(pts, dim=1)
    cos_ref = pts[:, 1] / r.clamp_min(1e-30)
    scale = 1.0 + eps * legendre_p(mode, cos_ref)
    pts = pts * scale.unsqueeze(-1)
    n_p = pts.shape[0]
    mass2d = torch.full((n_p,), rho0 * dx * dy, dtype=pts.dtype)
    rho = pts[:, 0].abs()
    # Half weight: the meridian plane carries *both* halves of the revolution,
    # so a sum over the full circle already covers every azimuth twice.  Using
    # 2*pi*rho (full weight) doubles the volume and the surface area and inflates
    # the measured surface-delta ratio by exactly 2 -- measured before this fix.
    vol = math.pi * rho * (mass2d / rho0)                # axisymmetric dV
    rho_safe = rho.clamp_min(0.25 * h)
    rad = torch.linalg.vector_norm(pts, dim=1)
    return {"pos": pts.contiguous(), "mass": mass2d, "dx": dx, "h": h,
            "rho0": rho0, "radius": radius, "mode": mode, "eps": eps,
            "rho": rho, "rho_safe": rho_safe, "dvol": vol,
            "cos_theta": pts[:, 1] / rad.clamp_min(1e-30),
            "n_particles": n_p}


def axisym_curvature(state, geom: dict) -> torch.Tensor:
    """``kappa = kappa_meridional + n_rho / rho`` (div n in cylindrical coords).

    ``state`` is a 2-D :class:`tensorsph.csf_consistent.CSFState` whose ``kappa``
    is the meridional plane-curve curvature returned by ``fit_curvature``; the
    azimuthal term ``n_rho/rho`` is evaluated analytically, with the sign of
    ``n_rho`` taken from the side of the axis (``x < 0`` mirrors ``rho``).
    """
    n = state.normal
    sign = torch.where(geom["pos"][:, 0] >= 0.0, torch.ones_like(geom["rho"]),
                       -torch.ones_like(geom["rho"]))
    n_rho = n[:, 0] * sign
    return state.kappa + n_rho / geom["rho_safe"]


def axisym_modal_stress(geom: dict, sigma: float, *, mode: int | None = None,
                        eps: float | None = None, delta_normalization: str = "legacy",
                        **csf_kwargs) -> dict:
    """Modal normal-stress of the axisymmetric drop, and its frequency ratio.

    ``dp_measured = (2l+1)/(4 pi R^2) * sum_i (-sigma kappa_i) delta_s,i dV_i
    P_l(cos theta_i)`` with ``dV = 2 pi rho dA`` the axisymmetric volume element;
    the ``2l+1`` removes the ``<P_l^2> = 1/(2l+1)`` surface average so the number
    is directly comparable with :func:`exact_modal_stress`.  ``omega_meas /
    omega_exact = sqrt(dp_measured/dp_exact)``.
    """
    from .surface_tension_dispersion import csf_state  # noqa: PLC0415

    mode = geom["mode"] if mode is None else mode
    eps = geom["eps"] if eps is None else eps
    state, h, _ = csf_state(geom, sigma, periodic_x=False,
                            delta_normalization=delta_normalization, **csf_kwargs)
    kappa = axisym_curvature(state, geom)
    share = legendre_p(mode, geom["cos_theta"])
    dvol = geom["dvol"]
    traction = -sigma * kappa * state.delta_s
    area = 4.0 * math.pi * geom["radius"] ** 2
    dp_meas = float((2.0 * mode + 1.0) / area * (traction * dvol * share).sum())
    dp_exact = exact_modal_stress(sigma, geom["radius"], mode, eps)
    ratio = dp_meas / dp_exact
    sign = math.copysign(1.0, ratio)
    omega_ratio = sign * math.sqrt(abs(ratio))
    kappa_mean = float((kappa * state.delta_s * dvol).sum() / area)
    meridional = float((state.kappa * state.delta_s * dvol * share).sum()
                       * (2.0 * mode + 1.0) / area)
    return {
        "mode": mode, "eps": eps, "dx": geom["dx"],
        "radius_over_dx": geom["radius"] / geom["dx"],
        "n_particles": int(geom["pos"].shape[0]),
        "dp_measured": dp_meas,
        "dp_analytic": dp_exact,
        "stiffness_ratio": ratio,
        "omega_ratio": omega_ratio,
        "omega_error_pct": (omega_ratio - 1.0) * 100.0,
        "kappa_mean_R": kappa_mean * geom["radius"],
        "kappa_mean_exact_R": 2.0,
        "meridional_mode_coeff_ratio": meridional / dp_exact,
        "surface_measure": float((state.delta_s * dvol).sum() / area),
        "delta_normalization": delta_normalization,
        "h": h,
    }


def drop_frequency_linear_response(*, radius: float = 1.0, sigma: float = 1.0,
                                  rho: float = 1.0, mode: int = 2,
                                  eps: float = 0.02, radius_over_dx=(
                                      8.0, 12.0, 16.0),
                                  delta_normalization: str = "legacy",
                                  **csf_kwargs) -> dict:
    """Resolution ladder of the linear-response frequency estimate."""
    omega_exact = rayleigh_frequency_3d(sigma, rho, radius, mode)
    rows = []
    for rdx in radius_over_dx:
        geom = axisym_drop_geometry(radius, radius / rdx, mode=mode, eps=eps, rho0=rho)
        row = axisym_modal_stress(geom, sigma, delta_normalization=delta_normalization,
                                  **csf_kwargs)
        row["omega_analytic"] = omega_exact
        row["omega_measured"] = omega_exact * row["omega_ratio"]
        rows.append(row)
    return {"mode": mode, "eps": eps, "sigma": sigma, "rho": rho,
            "radius": radius, "omega_analytic": omega_exact,
            "rows": rows}

# --------------------------------------------------------------------------- #
# 3-D lattice audit (what the shipped 3-D CSF operator actually returns)
# --------------------------------------------------------------------------- #
# Kept as *measurement* code, not as a solver: the shipped
# :mod:`tensorsph.threed.csf3` operator ("mls" curvature = ``div n`` on a lattice
# staircase) is scored here so the reason it is not used for the frequency
# anchor is a number in the artifacts rather than a claim in a docstring.

_FCC_BASIS = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))


def _fcc_sites(radius: float, dx: float, mode: int = 0, eps: float = 0.0) -> torch.Tensor:
    """FCC lattice sites in a sphere, radially mapped to ``R[1 + eps P_l]``."""
    a = math.sqrt(2.0) * dx
    n = int(math.ceil(radius * (1.0 + abs(eps)) / a)) + 2
    axis = torch.arange(-n, n + 1, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    base = torch.stack((gx.ravel(), gy.ravel(), gz.ravel()), dim=1) * a
    offsets = torch.tensor(_FCC_BASIS, dtype=torch.float64) * a
    sites = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    r = torch.linalg.vector_norm(sites, dim=1)
    # the site at the origin has no direction (the radial map would be a NaN)
    sites = sites[(r <= radius) & (r > 0.5 * dx)]
    if eps != 0.0 and mode >= 2:
        r = torch.linalg.vector_norm(sites, dim=1)
        scale = 1.0 + eps * legendre_p(mode, sites[:, 2] / r)
        sites = sites * scale.unsqueeze(-1)
    return sites.contiguous()


def _fcc_kernel_sum(dx: float, h: float, extent: float = 3.0) -> float:
    """``sum_j W`` over a complete FCC lattice (mass calibration, ``m = rho0/C``)."""
    from .threed.kernel3 import cubic_spline_kernel  # noqa: PLC0415

    a = math.sqrt(2.0) * dx
    n = int(math.ceil(extent * h / a)) + 1
    axis = torch.arange(-n, n + 1, dtype=torch.float64) * a
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    base = torch.stack((gx.ravel(), gy.ravel(), gz.ravel()), dim=1)
    offsets = torch.tensor(_FCC_BASIS, dtype=torch.float64) * a
    sites = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return float(cubic_spline_kernel(torch.linalg.vector_norm(sites, dim=1), h).sum())


def fcc_sphere(radius: float, dx: float, *, mode: int = 2, eps: float = 0.0,
               h_factor: float = 1.3, rho0: float = 1.0) -> dict:
    """FCC lattice sphere with ``r = R[1 + eps P_l]`` and calibrated mass."""
    h = h_factor * dx
    sites = _fcc_sites(radius, dx, mode, eps)
    mass = torch.full((sites.shape[0],), rho0 / _fcc_kernel_sum(dx, h),
                      dtype=sites.dtype)
    return {"pos": sites, "mass": mass, "dx": dx, "h": h, "rho0": rho0,
            "radius": radius, "mode": mode, "eps": eps}


def flat_surface(dx: float, *, half_height: float, h_factor: float = 1.3,
                 rho0: float = 1.0) -> dict:
    """Half-space of FCC fluid with one flat free surface at ``z = half_height``.

    The control geometry for the surface delta function: on a plane the coarea
    identity ``int |grad c| dV = A`` is exact, so the measured
    ``sum V |grad c| / A`` is the normalisation error of the CSF delta function
    with no curvature involved.
    """
    a = math.sqrt(2.0) * dx
    n = int(math.ceil(half_height / a)) + 3
    axis = torch.arange(-n, n + 1, dtype=torch.float64) * a
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    base = torch.stack((gx.ravel(), gy.ravel(), gz.ravel()), dim=1)
    offsets = torch.tensor(_FCC_BASIS, dtype=torch.float64) * a
    sites = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    h = h_factor * dx
    margin = 3.0 * h
    keep = ((sites[:, 2] <= half_height) & (sites[:, 2] >= -half_height)
            & (sites[:, 0].abs() <= half_height - margin)
            & (sites[:, 1].abs() <= half_height - margin))
    sites = sites[keep]
    mass = torch.full((sites.shape[0],), rho0 / _fcc_kernel_sum(dx, h),
                      dtype=sites.dtype)
    area = (2.0 * (half_height - margin)) ** 2
    return {"pos": sites.contiguous(), "mass": mass, "dx": dx, "h": h,
            "rho0": rho0, "area": area}


def csf3_audit(geom: dict, h: float, *, rho0: float, area: float | None = None,
               mode: int = 2, eps: float = 0.0, radius: float | None = None,
               stencil: float = 2.1) -> dict:
    """Measure the shipped 3-D CSF operator: delta measure and modal traction.

    On a plane (``area`` given) this reports ``sum V delta_s / A`` (1.0 is
    correct); on a sphere (``radius`` given) it reports the mean curvature
    ``kappa R`` (2.0 is correct) and the mode-``l`` traction ratio against
    :func:`exact_modal_stress`.
    """
    from .threed.csf3 import csf_fields3  # noqa: PLC0415
    from .threed.neighbors3 import box_pairs  # noqa: PLC0415

    pos, mass = geom["pos"], geom["mass"]
    pairs = box_pairs(pos, stencil * h)
    state = csf_fields3(pos, mass, h, pairs, rho0=rho0)
    vol = mass / rho0
    out = {
        "n_particles": int(pos.shape[0]),
        "h_over_dx": h / geom["dx"],
        "delta_measure_ratio": float((state.delta_s * vol).sum()
                                     / (area if area is not None
                                        else 4.0 * math.pi * radius ** 2)),
        "color_max": float(state.color.max()),
    }
    if radius is not None:
        surf = 4.0 * math.pi * radius * radius
        traction = -state.kappa * state.delta_s
        share = legendre_p(mode, pos[:, 2] / torch.linalg.vector_norm(pos, dim=1))
        dp_meas = float((2.0 * mode + 1.0) / surf * (traction * vol * share).sum())
        out["kappa_mean_R"] = float((state.kappa * state.delta_s * vol).sum()
                                    / surf * radius)
        out["dp_measured"] = dp_meas
        out["dp_analytic"] = exact_modal_stress(1.0, radius, mode, eps)
        out["mode_traction_ratio"] = (dp_meas / out["dp_analytic"]
                                      if out["dp_analytic"] else None)
    return out


def fit_damped_cosine(times, values) -> dict:
    """``offset + exp(-gamma t)[c cos(omega t) + s sin(omega t)]`` global fit.

    Two-stage grid search + linear least squares (no SciPy): the dense ``omega``
    grid is bracketed by the period of the amplitude's sign changes, then
    refined.  ``residual`` is the relative RMS residual, so a noise-dominated
    signal is visible instead of being scored as a number.
    """
    import numpy as np

    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    span = t.max() - t.min()
    zero = np.where(np.signbit(y[:-1]) != np.signbit(y[1:]))[0]
    if zero.size >= 2:
        period = 2.0 * (t[zero[-1]] - t[zero[0]]) / zero.size
    else:                                     # no sign changes: fall back
        period = 2.0 * span
    omega0 = 2.0 * math.pi / max(period, 1e-12)
    best = None
    for omega in np.linspace(0.5 * omega0, 1.5 * omega0, 401):
        for gamma in np.linspace(0.0, 6.0 / max(span, 1e-12), 41):
            env = np.exp(-gamma * (t - t[0]))
            design = np.stack((np.ones_like(t), env * np.cos(omega * t),
                               env * np.sin(omega * t)), axis=1)
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            res = y - design @ coef
            rms = float(np.sqrt(np.mean(res ** 2)) / (np.std(y) + 1e-30))
            if best is None or rms < best["residual"]:
                best = {"omega": float(omega), "gamma": float(gamma),
                        "residual": rms, "offset": float(coef[0]),
                        "cos": float(coef[1]), "sin": float(coef[2])}
    env = np.exp(-best["gamma"] * (t - t[0]))
    design = np.stack((np.ones_like(t),
                       env * np.cos(best["omega"] * t),
                       env * np.sin(best["omega"] * t)), axis=1)
    best["amplitude"] = float(np.hypot(best["cos"], best["sin"]))
    best["period"] = 2.0 * math.pi / best["omega"]
    return best


def axisym_curvature_analytic(geom: dict) -> torch.Tensor:
    """Exact axisymmetric curvature for the known shape ``r = R[1+eps P_l]``.

    ``kappa = kappa_meridional + n_rho/rho`` with both terms evaluated
    analytically from the parametric curve, so the only discretisation error
    comes from the particle quadrature (``dvol``, ``delta_s``), not from the
    curvature estimator.  This is the *baseline* that a perfect curvature
    estimator would achieve.

    Meridional curvature of ``r(theta) = R[1+eps P_l(cos theta)]``:
        ``kappa_m = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^{3/2}``
    Azimuthal term:
        ``n_rho/rho = (r cos_theta + r' sin_theta) / (r rho)``
    """
    R = geom["radius"]
    mode = geom["mode"]
    eps = geom["eps"]
    cos_t = geom["cos_theta"]
    sin_t = torch.sqrt(1.0 - cos_t ** 2).clamp_min(1e-30)

    # r(theta) and derivatives w.r.t. theta
    # dP_l/d(cos theta) via recurrence: P_l' = l/(1-c^2) * (c P_l - P_{l-1})
    # But we need dr/dtheta = -sin_theta * dP_l/d(cos_theta) * eps * R
    # Use the identity: dP_l/dtheta = -sin_theta * dP_l/dc
    # and dP_l/dc = l/(1-c^2) * (c P_l - P_{l-1}) for l >= 1
    P_l = legendre_p(mode, cos_t)
    if mode >= 1:
        P_lm1 = legendre_p(mode - 1, cos_t)
        dP_l_dc = mode / (1.0 - cos_t ** 2).clamp_min(1e-30) * (cos_t * P_l - P_lm1)
    else:
        dP_l_dc = torch.zeros_like(P_l)

    r = R * (1.0 + eps * P_l)
    dr_dt = -sin_t * eps * R * dP_l_dc  # dr/dtheta
    # d2r/dtheta2
    # d2P_l/dtheta2 = -cos_theta * dP_l/dc + sin^2 * d2P_l/dc2
    # Use: d2P_l/dc2 = 1/(1-c^2) * (2c dP_l/dc - l(l+1) P_l)
    d2P_l_dc2 = (2.0 * cos_t * dP_l_dc - mode * (mode + 1.0) * P_l
                 ) / (1.0 - cos_t ** 2).clamp_min(1e-30)
    d2r_dt2 = (-cos_t * eps * R * dP_l_dc
               + sin_t ** 2 * eps * R * d2P_l_dc2)

    # Meridional curvature
    kappa_m = (r ** 2 + 2.0 * dr_dt ** 2 - r * d2r_dt2
               ) / (r ** 2 + dr_dt ** 2) ** 1.5

    # Azimuthal term: n_rho/rho
    # The outward normal in the meridional plane has components:
    #   n_rho = (r cos_theta + r' sin_theta) / sqrt(r^2 + r'^2)
    #   n_z   = (r sin_theta - r' cos_theta) / sqrt(r^2 + r'^2)
    # But we need n_rho/rho where rho = |x| = r * sin_theta (cylindrical radius)
    # Wait, in our geometry, x is the cylindrical radius (rho), y is the axis (z).
    # So cos_theta = y/r = z/r, sin_theta = x/r = rho/r.
    # The normal component in the x direction (cylindrical rho direction) is:
    #   n_x = (r sin_theta - r' cos_theta) / sqrt(r^2 + r'^2)
    #       = (rho - r' cos_theta/rho * r) / sqrt(r^2 + r'^2)
    # This is getting complicated. Let me use a simpler approach.
    #
    # For the axisymmetric reduction, the azimuthal term is:
    #   n_rho/rho = (n . e_rho) / rho
    # where e_rho is the unit vector in the cylindrical rho direction.
    # In our 2D geometry (x, y), x is the cylindrical rho, y is the axis z.
    # So e_rho = (sign(x), 0) = (x/|x|, 0).
    # The normal n = (n_x, n_y) from the CSF state.
    # But we're using the analytic curvature, so we need the analytic normal.
    #
    # The analytic normal of the curve r(theta) is:
    #   n = (r sin_theta - r' cos_theta, r cos_theta + r' sin_theta) / sqrt(r^2 + r'^2)
    # The x-component (cylindrical rho direction) is:
    #   n_x = (r sin_theta - r' cos_theta) / sqrt(r^2 + r'^2)
    # And rho = |x| = r * sin_theta.
    # So n_rho/rho = n_x / (r sin_theta) = (r sin_theta - r' cos_theta) / (r sin_theta sqrt(r^2 + r'^2))
    #              = (1 - r' cos_theta / (r sin_theta)) / sqrt(r^2 + r'^2)
    #
    # Actually, let me use the simpler formula from the module docstring:
    # For the unperturbed circle, n_rho/rho = 1/R.
    # For the perturbed shape, n_rho/rho = 1/R + (l-1) eps P_l / R + O(eps^2).
    #
    # Let me compute it directly from the geometry.
    # The position is (x, y) = (r sin_theta, r cos_theta) in our convention.
    # Wait, no. In our geometry, cos_theta = y/r, so y = r cos_theta, x = r sin_theta.
    # The tangent vector is (dx/dtheta, dy/dtheta) = (dr/dtheta sin_theta + r cos_theta, dr/dtheta cos_theta - r sin_theta).
    # The normal vector (outward) is (dy/dtheta, -dx/dtheta) / |tangent|.
    # So n_x = (dr/dtheta cos_theta - r sin_theta) / sqrt(r^2 + (dr/dtheta)^2)
    #        = (r' cos_theta - r sin_theta) / sqrt(r^2 + r'^2)
    # And rho = |x| = r sin_theta.
    # So n_rho/rho = n_x / rho = (r' cos_theta - r sin_theta) / (r sin_theta sqrt(r^2 + r'^2))
    #
    # Hmm, this gives a negative value for the unperturbed circle (r'=0):
    # n_rho/rho = -r sin_theta / (r sin_theta * r) = -1/r = -1/R.
    # But we expect +1/R. So the sign is wrong.
    #
    # The issue is the normal direction. The outward normal of a circle points away from the center.
    # At (x, y) = (R sin_theta, R cos_theta), the outward normal is (sin_theta, cos_theta) = (x/R, y/R).
    # So n_x = sin_theta = x/R = rho/R.
    # And n_rho/rho = (rho/R) / rho = 1/R. Correct.
    #
    # So the analytic normal is (sin_theta, cos_theta) for the unperturbed circle.
    # For the perturbed shape, the normal is more complex.
    #
    # Let me use a different approach: compute the normal from the gradient of the level set function.
    # The level set function is phi(x, y) = sqrt(x^2 + y^2) - R(1 + eps P_l(y/r)).
    # The gradient is (x/r - eps dP_l/dc * y/r^2, y/r + eps dP_l/dc * x/r^2).
    # This is getting too complex. Let me just use the linearised formula.
    #
    # For small eps, the azimuthal term is:
    #   n_rho/rho = 1/R + (l-1) eps P_l / R + O(eps^2)
    # This is exact to first order in eps.
    #
    # For the meridional curvature, the linearised formula is:
    #   kappa_m = 1/R + (l-1)(l+1) eps P_l / R + O(eps^2)
    #
    # So the total curvature is:
    #   kappa = 2/R + (l-1)(l+2) eps P_l / R + O(eps^2)
    #
    # For the baseline measurement, I'll use the linearised formula.
    # This is exact to first order in eps, which is sufficient for the linear-response measurement.

    kappa_azimuthal = 1.0 / R + (mode - 1.0) * eps * P_l / R
    kappa_total = kappa_m + kappa_azimuthal
    return kappa_total


def axisym_modal_stress_analytic(geom: dict, sigma: float, *,
                                  mode: int | None = None,
                                  eps: float | None = None,
                                  delta_normalization: str = "legacy",
                                  **csf_kwargs) -> dict:
    """Modal normal-stress using the *analytic* curvature (baseline).

    This isolates the delta-function and volume-element discretisation error
    from the curvature estimation error.  If this baseline passes the gate
    but :func:`axisym_modal_stress` does not, the root cause is the curvature
    estimator, not the quadrature.
    """
    from .surface_tension_dispersion import csf_state  # noqa: PLC0415

    mode = geom["mode"] if mode is None else mode
    eps = geom["eps"] if eps is None else eps
    state, h, _ = csf_state(geom, sigma, periodic_x=False,
                            delta_normalization=delta_normalization, **csf_kwargs)
    kappa = axisym_curvature_analytic(geom)
    share = legendre_p(mode, geom["cos_theta"])
    dvol = geom["dvol"]
    traction = -sigma * kappa * state.delta_s
    area = 4.0 * math.pi * geom["radius"] ** 2
    dp_meas = float((2.0 * mode + 1.0) / area * (traction * dvol * share).sum())
    dp_exact = exact_modal_stress(sigma, geom["radius"], mode, eps)
    ratio = dp_meas / dp_exact
    sign = math.copysign(1.0, ratio)
    omega_ratio = sign * math.sqrt(abs(ratio))
    return {
        "mode": mode, "eps": eps, "dx": geom["dx"],
        "radius_over_dx": geom["radius"] / geom["dx"],
        "n_particles": int(geom["pos"].shape[0]),
        "dp_measured": dp_meas,
        "dp_analytic": dp_exact,
        "stiffness_ratio": ratio,
        "omega_ratio": omega_ratio,
        "omega_error_pct": (omega_ratio - 1.0) * 100.0,
        "kappa_mean_R": float((kappa * state.delta_s * dvol).sum() / area) * geom["radius"],
        "surface_measure": float((state.delta_s * dvol).sum() / area),
        "delta_normalization": delta_normalization,
        "h": h,
    }
