"""Sub-particle interface reconstruction on the SPH colour field (``c = 1/2``).

Motivation (``CASE-SPH-OSC-DROP-FREQ``)
--------------------------------------
The two surface operators in the repository that a capillary oscillation case
can reach are both *particle-valued*:

* the colour-gradient CSF operator (``csf_consistent.csf_fields``) evaluates the
  curvature at particle centres, so the interface it differentiates is the
  lattice staircase, not the surface; the modal *stiffness* it returns for an
  ``l = 2`` drop mode has the wrong sign and a 200 % error
  (``docs/case_sph_osc_drop_freq.md``, ``..._round2.md``);
* an interface-*marker* quadrature over the shell inherits the same staircase in
  its arc length (``dl/2 pi R = 1.64 .. 1.78``).

What is missing is a representation of the interface *between* particles.  This
module builds one and nothing else: it evaluates the same colour field the CSF
operator uses,

    c(x) = sum_j V_j W(|x - x_j|, h)          V_j = m_j / rho0,

at arbitrary query points (not only at particle centres), follows it to the
``c = 1/2`` level along rays from the symmetry axis, and hands back a surface
of revolution ``r(theta)``.  Curvature, area and modal traction are then
computed analytically *on that surface* -- the standard split used by
interface-reconstruction surface-tension models (Cummins, Francois & Kothe 2005;
Marchandise et al. 2007): the discrete part does interface *resolution*, the
continuous part does differential geometry.  The accuracy of the whole operator
is therefore one measurable number -- the amplitude transfer of the
reconstruction -- which :func:`measure_revolution` reports per mode.

Everything here is ``m = 0`` axisymmetric (a surface of revolution about
``+y``) and pure tensor code; nothing is tuned to a particular case.

Caveats that a user must respect
--------------------------------
* The reconstruction is a truncated Legendre fit, so it can represent a mode it
  was built to fit.  It is a *measurement of how well the SPH colour field
  resolves the interface*, not a curvature operator for an unstructured mesh.
* The ``c = 1/2`` level set of a one-sided (free-surface) kernel sum sits inside
  the particle cloud by ``O(h)`` -- a constant offset that changes ``r_0`` and is
  reported, but that does not by itself bias a modal amplitude.  The *modal*
  bias is measured, never assumed: use :func:`measure_revolution` on the
  undeformed shape (``eps = 0``) as the null input of any estimator.

Usage::

    PYTHONPATH=src python -c "import tensorsph.interface_level_set as ils; print(ils.COLOR_LEVEL)"
"""
from __future__ import annotations

import math

import torch

from .droplet_impact import cubic_spline_kernel
from .oscillating_drop3d import legendre_p

COLOR_LEVEL = 0.5


# --------------------------------------------------------------------------- #
# colour field at arbitrary points
# --------------------------------------------------------------------------- #
def color_at_points(
    pos: torch.Tensor,
    mass: torch.Tensor,
    xq: torch.Tensor,
    h: float,
    rho0: float,
    *,
    chunk: int = 4096,
) -> torch.Tensor:
    """``c(x_q) = sum_j V_j W(|x_q - x_j|, h)`` clipped to ``[0, 1]``.

    Identical field to :func:`tensorsph.csf_consistent.kernel_sum_color` -- the
    self term is included because the sum here runs over *all* particles, not over
    a pair list -- evaluated at arbitrary query points instead of at particle
    centres.  The quadrature is exact in the sense that no neighbourhood cut-off
    is applied: the cubic spline has compact support ``2h`` and every pair is
    evaluated, so the value does not depend on a cell-list radius.
    """
    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError("pos must be (n, 2)")
    if xq.dim() != 2 or xq.shape[1] != 2:
        raise ValueError("xq must be (m, 2)")
    if h <= 0.0:
        raise ValueError("h must be positive")
    vol = mass / rho0
    out = torch.empty(xq.shape[0], dtype=pos.dtype, device=pos.device)
    for a in range(0, xq.shape[0], chunk):
        q = xq[a:a + chunk]
        r = torch.cdist(q, pos)
        out[a:a + chunk] = (cubic_spline_kernel(r, h) * vol[None, :]).sum(dim=1)
    return out.clamp(0.0, 1.0)


def level_set_radius(
    geom: dict,
    *,
    n_rays: int = 240,
    band: float = 0.20,
    n_radial: int = 81,
    level: float = COLOR_LEVEL,
    progress: bool = False,
) -> dict:
    """Radius where the colour crosses ``level`` along rays ``theta`` from ``+y``.

    Rays are spaced uniformly in ``theta`` over ``(0, pi)``; along each ray the
    colour is sampled on ``n_radial`` radii in ``[R(1-band), R(1+band)]`` and the
    crossing is linearly interpolated between the two samples that bracket the
    level.  With ``n_radial = 81`` the sample spacing is ``0.005 R`` -- an order
    of magnitude finer than ``h`` -- so the interpolation error is far below the
    particle spacing that the whole exercise is about.

    Returns ``theta``, ``r`` (both ``[0, 1]``-shaped, with non-crossing rays
    dropped), the fraction of rays that crossed, and the sampled colour grid
    (``(n_rays, n_radial)``) so an audit can look at the profile itself.
    """
    pos, mass, h, rho0 = geom["pos"], geom["mass"], geom["h"], geom["rho0"]
    radius = float(geom["radius"])
    if not 0.0 < band < 1.0:
        raise ValueError("band must be in (0, 1)")
    theta = torch.linspace(1.0e-3, math.pi - 1.0e-3, n_rays, dtype=pos.dtype)
    radii = torch.linspace(radius * (1.0 - band), radius * (1.0 + band),
                           n_radial, dtype=pos.dtype)
    dirs = torch.stack((torch.sin(theta), torch.cos(theta)), dim=1)
    grid = (dirs[:, None, :] * radii[None, :, None]).reshape(-1, 2)
    col = color_at_points(pos, mass, grid, h, rho0).reshape(n_rays, n_radial)

    cross = torch.full((n_rays,), float("nan"), dtype=pos.dtype)
    for i in range(n_rays):
        ci = col[i]
        below = ci < level
        if not bool(ci[0] > level) or not bool(below.any()):
            continue
        j = int(torch.nonzero(below, as_tuple=False)[0])
        c0, c1 = float(ci[j - 1]), float(ci[j])
        r0, r1 = float(radii[j - 1]), float(radii[j])
        if c0 == c1:
            continue
        cross[i] = r0 + (c0 - level) / (c0 - c1) * (r1 - r0)
    ok = torch.isfinite(cross)
    if progress:
        print(f"    level set: {int(ok.sum())}/{n_rays} rays crossed c={level}",
              flush=True)
    return {"theta": theta[ok], "r": cross[ok], "col": col,
            "n_rays": n_rays, "n_crossed": int(ok.sum()),
            "crossed_fraction": float(ok.sum()) / n_rays,
            "r_lo": float(radii[0]), "r_hi": float(radii[-1])}


# --------------------------------------------------------------------------- #
# Legendre reconstruction of the meridian curve
# --------------------------------------------------------------------------- #
def legendre_fit(theta: torch.Tensor, r: torch.Tensor, *,
                 kmax: int = 6, weight_sin: bool = True) -> dict:
    """Area-weighted Legendre least squares for ``r(theta)``.

    The surface element of a surface of revolution carries ``sin(theta)``, so the
    default weights are ``sin(theta)`` (uniform rays in ``theta`` then integrate
    as a sphere does).  ``P_0`` absorbs the mean radius, i.e. the ``O(h)``
    offset of the level set, so the modal coefficients are unaffected by it.
    """
    if theta.shape != r.shape:
        raise ValueError("theta and r must have the same shape")
    if theta.numel() < kmax + 2:
        raise ValueError(f"{theta.numel()} crossings cannot fit {kmax + 1} coefficients")
    basis = torch.stack([legendre_p(k, torch.cos(theta)) for k in range(kmax + 1)], dim=1)
    w = torch.sin(theta) if weight_sin else torch.ones_like(theta)
    sw = w / w.sum()
    ata = (basis * sw[:, None]).T @ basis
    atb = (basis * sw[:, None]).T @ r
    coef = torch.linalg.solve(ata + 1.0e-14 * torch.eye(kmax + 1, dtype=ata.dtype), atb)
    resid = r - basis @ coef
    return {"coef": coef, "kmax": kmax,
            "rms_residual": float((resid.pow(2) * sw).sum().sqrt())}


def rms_amplitude(theta: torch.Tensor, r: torch.Tensor) -> float:
    """Non-modal amplitude estimator: ``sqrt(<(r - <r>)^2>_sin)``.

    For ``r = R(1 + eps P_l)`` this is ``eps R / sqrt(2 l + 1)``.  It is the
    *roughness* of the measured interface and needs no assumption about which
    mode is present, which is why it is used as an independent cross-check on the
    ``P_l`` projection (a mode-truncated fit could in principle report a
    different amplitude than the surface actually has).
    """
    w = torch.sin(theta)
    mean = float((w * r).sum() / w.sum())
    return float((w * (r - mean).pow(2)).sum().div(w.sum()).sqrt())


# --------------------------------------------------------------------------- #
# analytic geometry of a surface of revolution
# --------------------------------------------------------------------------- #
def reconstructed_surface(coef: torch.Tensor, kmax: int, *, radius: float,
                          n_theta: int = 4001) -> dict:
    """Evaluate ``r, r', r'', kappa(meridional+azimuthal), dA`` on a fine grid.

    ``kappa = kappa_m + n_rho/rho`` is ``div n`` of the revolution in cylindrical
    coordinates, computed from the *analytic* derivatives of the Legendre series
    (no finite differences on particle values).  A sphere gives ``kappa R = 2``
    to machine precision -- ``tests/test_oscillating_drop_freq_round3.py`` pins
    that, so a later change to this file cannot silently degrade the geometry.
    """
    theta = torch.linspace(1.0e-6, math.pi - 1.0e-6, n_theta, dtype=coef.dtype)
    c = torch.cos(theta)
    basis = torch.stack([legendre_p(k, c) for k in range(kmax + 1)], dim=1)
    r = basis @ coef
    dth = float(theta[1] - theta[0])
    rp = torch.gradient(r, spacing=dth)[0]
    rpp = torch.gradient(rp, spacing=dth)[0]
    rho = r * torch.sin(theta)
    dl = torch.sqrt(r ** 2 + rp ** 2)
    darea = 2.0 * math.pi * rho * dl * dth
    kappa_m = (r ** 2 + 2.0 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5
    n_rho = torch.sin(theta) - (rp / r) * torch.cos(theta)
    kappa_az = n_rho / rho.clamp_min(1.0e-30)
    return {"theta": theta, "cos_theta": c, "r": r, "dl": dl, "darea": darea,
            "kappa_m": kappa_m, "kappa_az": kappa_az, "kappa": kappa_m + kappa_az,
            "radius": radius}


def modal_capillary_stress(surface: dict, *, sigma: float, radius: float,
                          mode: int, reference_radius: float | None = None,
                          subtract_base_curvature: bool = True,
                          reference_measure: bool = True) -> dict:
    """``P_l`` coefficient of the *restoring* traction ``-(sigma/R)(delta kappa)``.

    Two convention choices decide whether this number is right, and getting them
    wrong over-counts the stiffness by exactly **2x** (``CASE`` round 2 did, see
    below):

    ``subtract_base_curvature``
        The traction ``-sigma kappa`` carries the equilibrium part ``-2 sigma/R``,
        which is balanced in the base state and drives nothing.  Only the
        perturbation ``delta kappa = kappa - 2/R`` restores the mode
        (``delta kappa = (l-1)(l+2) a_l P_l / R`` for ``r = R(1+a_l P_l)``,
        i.e. Prosperetti's ``p' = sigma (l-1)(l+2) delta/R^2``).
    ``reference_measure``
        The projection is onto the *undistorted* sphere measure
        ``2 pi R^2 sin(theta) dtheta``.  Projecting the deformed measure instead
        lets the base curvature ``2 sigma/R`` multiply the ``O(a)`` area
        perturbation and adds another ``-2 sigma a (2l+1)/R`` -- a second factor
        of the same size, with nothing physical behind it: the P_l content of the
        area change is already carried by ``delta kappa``.

    Both effects are ``O(a)`` and identical in size, so the wrong pair of
    conventions looks *right* whenever the measured amplitude happens to be
    under-recovered by ~2x.  ``tests/test_oscillating_drop_freq_round3.py`` pins
    an analytic mode-l surface against
    :func:`tensorsph.oscillating_drop3d.exact_modal_stress`, which no measured
    amplitude can fake.
    """
    theta = surface["theta"]
    share = legendre_p(mode, surface["cos_theta"])
    r_ref = float(surface["radius"] if reference_radius is None else reference_radius)
    area_ref = 4.0 * math.pi * r_ref ** 2
    kappa = surface["kappa"]
    if subtract_base_curvature:
        kappa = kappa - 2.0 / r_ref
    if reference_measure:
        dth = float(theta[1] - theta[0])
        weight = 2.0 * math.pi * r_ref ** 2 * torch.sin(theta) * dth
    else:
        weight = surface["darea"]
    dp = float(-(2.0 * mode + 1.0) / area_ref
               * (sigma * kappa * weight * share).sum())
    darea = surface["darea"]
    return {"dp": dp,
            "area_ratio": float(darea.sum()) / area_ref,
            "kappaR_mean": float((surface["kappa"] * darea).sum()
                                 / darea.sum()) * r_ref}


def measure_revolution(geom: dict, *, mode: int, sigma: float | None = None,
                       kmax: int = 6, n_rays: int = 240, n_radial: int = 81,
                       band: float = 0.20, level: float = COLOR_LEVEL) -> dict:
    """Full measurement chain on one geometry: level set -> fit -> traction.

    ``mode`` selects which Legendre coefficient is read out and which traction
    harmonic is projected; ``sigma`` may be ``None`` when only geometry is
    wanted.  Reported per-mode quantities:

    ``amplitude``
        ``coef[mode] / R`` -- the reconstructed relative amplitude;
    ``amplitude_recovery``
        ``amplitude / eps`` when the geometry carries a seed ``eps`` (this is the
        amplitude transfer of the whole operator);
    ``spurious_amplitude``
        the same coefficient on an undeformed geometry, i.e. the null input the
        estimator invents out of lattice noise;
    """
    radius = float(geom["radius"])
    ls = level_set_radius(geom, n_rays=n_rays, band=band, n_radial=n_radial,
                          level=level)
    fit = legendre_fit(ls["theta"], ls["r"], kmax=kmax)
    coef = fit["coef"]
    out = {
        "mode": mode, "kmax": kmax, "n_crossed": ls["n_crossed"],
        "crossed_fraction": ls["crossed_fraction"],
        "mean_radius": float(coef[0]) / radius,
        "amplitude": float(coef[mode]) / radius,
        "rms_amplitude": rms_amplitude(ls["theta"], ls["r"]) / radius,
        "rms_residual": fit["rms_residual"] / radius,
        "level_set_offset": float(coef[0]) / radius - 1.0,
        "theta": ls["theta"], "r": ls["r"], "coef": coef, "col": ls["col"],
    }
    eps = float(geom.get("eps", 0.0))
    if eps > 0.0:
        out["seeded_eps"] = eps
        out["amplitude_recovery"] = out["amplitude"] / eps
        # eps R / sqrt(2 l + 1) is the exact roughness of the seeded shape
        out["rms_amplitude_recovery"] = (out["rms_amplitude"]
                                         / (eps / math.sqrt(2 * mode + 1)))
    if sigma is not None:
        surf = reconstructed_surface(coef, kmax, radius=radius)
        st = modal_capillary_stress(surf, sigma=sigma, radius=radius, mode=mode,
                                    reference_radius=float(coef[0]))
        out.update({"dp": st["dp"], "area_ratio": st["area_ratio"],
                    "kappaR_mean": st["kappaR_mean"]})
    return out