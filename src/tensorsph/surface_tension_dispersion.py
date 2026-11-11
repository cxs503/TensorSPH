"""Capillary-wave dispersion and Lamb-mode frequency of the CSF surface-tension operator.

What this module is
-------------------
A *measurement* library for the discrete surface-tension stiffness of the CSF
operator shipped in :mod:`tensorsph.csf_consistent`.  It answers one question:
for a free-surface normal mode with an exactly known frequency,

* capillary wave, deep, inviscid, 2-D       ``omega^2 = sigma k^3 / rho``
  (Lamb 1932, *Hydrodynamics* 6th ed., Art. 267),
* oscillating 2-D drop (cylinder), mode ``n`` ``omega^2 = n (n^2 - 1) sigma / (rho R^3)``
  (Rayleigh 1879; Lamb Art. 275),

how large is ``omega_measured / omega_analytic`` when the restoring force is
produced by the *discretised* CSF operator instead of its exact value?

In the linear regime the only physics of the restoring force is the normal
surface stress

* wave: ``Delta p(x) = sigma kappa(x) = sigma k^2 a cos(k x)``
* drop: ``Delta p(theta) = sigma (n^2 - 1) eps cos(n theta) / R^2``

and the inertia is the exact potential-flow inertia (``rho / k`` per unit area
for the wave, ``rho R / n`` for the drop), so

.. code-block:: text

    omega_measured^2 / omega_theory^2  =  Delta p_measured / Delta p_theory

i.e. the frequency ratio is the *modal normal-stress ratio* of the discretised
operator.  This is a linear-response (discrete-dispersion) measurement, not a
time integration; :mod:`docs/verification_capillary_wave_open_surface.md`
documents why the trajectory of the same operator is not admissible at HEAD
(spurious acceleration ~23x the wave's own ``omega^2 A``).

Geometry defects this module fixes (m14536)
-------------------------------------------
The first version of this harness (m13895) measured a **closed block**: fluid
bounded by a free bottom, two free sides and the wave on top.  Four free
surfaces each carry their own ``-sigma kappa delta_s n`` force, whose modal
projection onto ``cos(k x)`` is noise, and the contamination was *larger* than
the signal: the measured stiffness had the wrong sign (``-0.25`` at ``k=1``) and
the reported error was ``-50 %`` with an ensemble spread larger than the mean.
Two structural fixes are implemented here:

``periodic_x=True``
    the particle cloud is triplicated along ``x`` before the neighbour search,
    so every particle on the primary copy sees its periodic images (the wave is
    posed on a periodic channel).  This removes the two free side faces.
``wall_layers>0``
    the lowest ``wall_layers`` rows of the slab are flagged as *solid wall*: they
    stay in the colour field and in the neighbour list (so the fluid above them
    sees full kernel support and its colour is 1, not 1/2) but their acceleration
    is zeroed.  This removes the free bottom face.

The remaining single free surface is the wave itself.  ``periodic_x=False,
wall_layers=0`` reproduces the old defective geometry, so the before/after
comparison in ``docs/verification_surface_tension_dispersion.md`` is a
controlled A/B on the same code.

Second defect fixed: the ``k`` sweep was degenerate.  The old harness set
``dx = lambda / ppw`` with a fixed ``ppw``, so every ``k`` produced a
*geometrically identical* problem (the CSF ratio is scale invariant) and
``k=1, 2, 3`` returned the same number to the last digit.  Here the domain
length is fixed at ``L = 2 pi``: ``L`` holds ``k`` wavelengths, ``dx = L/ppw``
does not depend on ``k``, and only the wavelength changes -- so ``k=2`` and
``k=3`` are genuinely shorter (stiffer, harder to resolve) waves.

Stability limit
---------------
:func:`capillary_dt_limit` is the explicit-integration bound for the capillary
force: the stiffest representable wavelength is ``lambda_min ~ 2 h``, so
``omega_max^2 = sigma (pi/h)^3 / rho`` and ``dt <= cfl / omega_max``.  It is used
by :func:`tensorsph.threed.solver.acoustic_dt` as a second limiter, which is the
integration point of this module in the solver stack.

Usage
-----
    PYTHONPATH=src python -m tensorsph.surface_tension_dispersion --quick
    PYTHONPATH=src python -m tensorsph.surface_tension_dispersion \
        --out runs/surface_tension_dispersion/report.json
    PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --quick

The CLI writes a JSON report that carries, per case, the analytic reference, the
measured value, the relative error, the tolerance and a pass flag.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .csf_consistent import csf_fields, mls_divergence
from .neighbor_search import cell_linked_pairs

#: Solver-side call sites of this module (checked by the integration test).
#: ``threed/solver.py::acoustic_dt`` takes the capillary bound as a second
#: limiter, so the 3-D verification drivers cannot out-run the surface force.
INTEGRATION_CALLERS = (
    "src/tensorsph/threed/solver.py:acoustic_dt",
    "src/tensorsph/surface_tension.py:capillary_dt_limit",
)

__all__ = [
    "AMPLITUDE_FRACTION", "DROP_EPS", "H_FACTOR", "INTEGRATION_CALLERS", "RHO0",
    "SIGMA",
    "curvature_audit", "curvature_noise_scan", "delta_measure_audit", "normal_audit",
    "capillary_dt_limit", "capillary_wave_case", "convergence_order",
    "csf_state", "domain_length", "drop_geometry", "drop_modal_stiffness",
    "ensemble", "hex_lattice", "lamb_case", "main", "periodic_images",
    "run_validation", "sigma_sensitivity", "wave_geometry",
    "wave_modal_stiffness",
]

DTYPE = torch.float64
RHO0 = 1000.0          # kg/m^3, water at 20 C
SIGMA = 0.072          # N/m, water/air at 20 C
H_FACTOR = 1.3         # h / dx
AMPLITUDE_FRACTION = 0.02   # a / lambda of the capillary wave
DROP_EPS = 0.10        # eps of r(theta) = R (1 + eps cos n theta)
DOMAIN_LENGTH = 2.0 * math.pi

#: platform gates (mission p97w4-n06-sph-surface-tension)
TOL_WAVE_PCT = 5.0
TOL_LAMB_PCT = 3.0
MIN_ORDER = 2.0
TOL_SIGMA_PCT = 5.0


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hex_lattice(dx: float, x0: float, x1: float, y0: float, y1: float,
                sx: float = 0.0, sy: float = 0.0) -> torch.Tensor:
    """Hexagonal particle lattice with a sub-lattice offset ``(sx, sy) * dx``.

    A hexagonal packing is used because it is the densest 2-D arrangement and its
    spacing is isotropic to within 8 %, so the surface tension operator does not
    see a preferred lattice direction.  ``sx``/``sy`` translate the lattice by a
    fraction of a cell; sweeping them gives the ensemble used by
    :func:`ensemble` to separate the smooth (physical) part of the modal response
    from the lattice-phase noise.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    ys = torch.arange(y0 + 0.5 * dy + sy * dy, y1, dy, dtype=DTYPE)
    rows = []
    for index, y in enumerate(ys):
        # Alternate rows are shifted by half a cell; the parity is taken from the
        # row index, not from ``round((y - y0)/dy)`` -- torch rounds half to even,
        # which silently aligns every second pair of rows into a square lattice
        # (nearest neighbour dy = 0.87 dx instead of dx, m13895 inherited that bug).
        x_start = x0 + 0.5 * dx + (0.5 * dx if index % 2 else 0.0) + sx * dx
        xs = torch.arange(x_start, x1, dx, dtype=DTYPE)
        rows.append(torch.stack((xs, torch.full_like(xs, float(y))), dim=1))
    return torch.cat(rows, dim=0)


def domain_length(k: float = 1.0) -> float:
    """Length of the periodic channel for wavenumber ``k``.

    Fixed at ``2 pi`` on purpose: ``k`` then sets how many wavelengths fit in the
    domain while ``dx`` stays the same, which is what makes the ``k = 1, 2, 3``
    sweep an independent test instead of three copies of the same problem.
    """
    return DOMAIN_LENGTH


def wave_geometry(k: float, amplitude: float, dx: float, *,
                  depth_factor: float = 0.8, wall_layers: int = 3,
                  sx: float = 0.0, sy: float = 0.0) -> dict:
    """Deep capillary wave on a periodic channel with a frozen wall bottom.

    The free surface is ``eta(x) = -amplitude * cos(k x)`` (a trough at ``x = 0``),
    the fluid occupies ``-depth <= y <= eta(x)``, and the lowest ``wall_layers``
    lattice rows are marked in ``wall_mask`` (frozen, but kept in the colour field
    so the fluid above them is bulk).  ``copies`` copies of the slab are needed for
    the ``x`` periodicity; :func:`periodic_images` builds them.
    """
    lam = 2.0 * math.pi / k
    length = domain_length(k)
    depth = depth_factor * lam
    cell = dx * dx * math.sqrt(3.0) / 2.0
    wall_depth = (wall_layers + 1) * dx
    pts = hex_lattice(dx, 0.0, length, -depth - wall_depth, 0.0, sx, sy)
    eta = -amplitude * torch.cos(k * pts[:, 0])
    pos = pts[pts[:, 1] < eta]
    mass = torch.full((pos.shape[0],), RHO0 * cell, dtype=DTYPE)
    y_min = pos[:, 1].min()
    wall_mask = pos[:, 1] < y_min + (wall_layers + 0.5) * dx
    # exact outward normal of the interface graph y = eta(x)
    slope = amplitude * k * torch.sin(k * pos[:, 0])
    normal = torch.stack((-slope, torch.ones_like(slope)), dim=1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    return dict(pos=pos, mass=mass, normal=normal, wall_mask=wall_mask,
                lam=lam, length=length, depth=depth, cell=cell,
                amplitude=amplitude, k=k, dx=dx)


def drop_geometry(mode: int, eps: float, radius: float, dx: float, *,
                  sx: float = 0.0, sy: float = 0.0) -> dict:
    """Lobed 2-D drop ``r(theta) = R (1 + eps cos(n theta))`` centred at the origin."""
    cell = dx * dx * math.sqrt(3.0) / 2.0
    reach = 1.4 * radius
    pts = hex_lattice(dx, -reach, reach, -reach, reach, sx, sy)
    r = torch.linalg.vector_norm(pts, dim=1)
    theta = torch.atan2(pts[:, 1], pts[:, 0])
    r_surf = radius * (1.0 + eps * torch.cos(mode * theta))
    pos = pts[r < r_surf]
    mass = torch.full((pos.shape[0],), RHO0 * cell, dtype=DTYPE)
    rs = torch.linalg.vector_norm(pos, dim=1)
    cos_t, sin_t = pos[:, 0] / rs, pos[:, 1] / rs
    r_prime = -radius * eps * mode * torch.sin(mode * torch.atan2(pos[:, 1], pos[:, 0]))
    nx = r_prime * sin_t + rs * cos_t
    ny = rs * sin_t - r_prime * cos_t
    normal = torch.stack((nx, ny), dim=1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    return dict(pos=pos, mass=mass, normal=normal,
                wall_mask=torch.zeros(pos.shape[0], dtype=torch.bool),
                radius=radius, cell=cell, eps=eps, mode=mode, dx=dx,
                length=2.0 * math.pi * radius, k=float(mode))


def periodic_images(pos: torch.Tensor, mass: torch.Tensor, length: float,
                    copies: int = 3):
    """Triplicate the cloud along ``x`` so the primary copy sees periodic images.

    Returns ``(pos_tiled, mass_tiled, n_primary)``; the first ``n_primary``
    particles are the primary copy, in the original order.
    """
    if copies % 2 != 1:
        raise ValueError("copies must be odd so the primary copy is centred")
    n = pos.shape[0]
    # the primary copy comes FIRST: csf_state slices the state back with ``[:n]``,
    # so any other ordering silently pairs one copy's fields with another copy's
    # positions (m14536 fixed exactly that).
    shifts = [0] + [s * length for k in range(1, copies // 2 + 1)
                    for s in (k, -k)]
    pts, ms = [], []
    for shift in shifts:
        pts.append(pos + torch.tensor([shift, 0.0], dtype=pos.dtype))
        ms.append(mass)
    return torch.cat(pts), torch.cat(ms), n


# --------------------------------------------------------------------------- #
# CSF evaluation
# --------------------------------------------------------------------------- #
def csf_state(geom: dict, sigma: float, *, h_factor: float = H_FACTOR,
              curvature_scheme: str = "fit_quadratic",
              curvature_stencil: float = 4.0,
              delta_normalization: str = "legacy",
              periodic_x: bool = True, copies: int = 3):
    """Run the CSF operator on a geometry and return ``(state, h, n_primary)``.

    ``periodic_x=True`` tiles the cloud (see :func:`periodic_images`) and the
    returned state is sliced back to the primary copy, so all downstream modal
    projections see one particle per physical point.
    """
    pos, mass = geom["pos"], geom["mass"]
    h = h_factor * geom["dx"]
    if periodic_x:
        pos_t, mass_t, n = periodic_images(pos, mass, geom["length"], copies)
    else:
        pos_t, mass_t, n = pos, mass, pos.shape[0]
    pairs = cell_linked_pairs(pos_t, curvature_stencil * h)
    state = csf_fields(pos_t, mass_t, pairs, h, RHO0,
                       curvature_scheme=curvature_scheme,
                       curvature_stencil=curvature_stencil,
                       delta_normalization=delta_normalization)
    if periodic_x:
        state = _slice_state(state, n)
    return state, h, n


def _slice_state(state, n: int):
    """Slice every per-particle tensor of a :class:`CSFState` to the primary copy."""
    from dataclasses import replace  # noqa: PLC0415  (cheap, keeps import surface small)

    np_ = state.kappa.shape[0]
    out = {}
    for name, value in vars(state).items():
        if torch.is_tensor(value) and value.shape[:1] == (np_,):
            out[name] = value[:n]
        else:
            out[name] = value
    return replace(state, **out)


def _csf_force(state, geom: dict, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``f = -sigma kappa delta_s n`` per particle, wall masked, plus surface measure."""
    mask = geom.get("wall_mask")
    delta_s = state.delta_s.clone()
    if mask is not None:
        delta_s = torch.where(mask, torch.zeros_like(delta_s), delta_s)
    force = -sigma * state.kappa * delta_s          # scalar traction [N/m^2] (2-D)
    return force, delta_s


def _modal_projector(geom: dict):
    vol = geom["mass"] / RHO0
    mask = geom.get("wall_mask")
    if mask is not None:
        vol = torch.where(mask, torch.zeros_like(vol), vol)
    return vol


def wave_modal_stiffness(geom: dict, sigma: float, **kwargs) -> dict:
    """Modal normal-stress ratio of the capillary wave.

    Returns the measured/analytic coefficient of ``Delta p`` on ``cos(k x)``, the
    implied frequency ratio ``sqrt(ratio)`` and the CSF surface measure
    ``sum_i delta_s,i V_i / L`` (1.0 when the delta function integrates to one
    interface length).
    """
    state, h, _ = csf_state(geom, sigma, **kwargs)
    vol = _modal_projector(geom)
    traction, delta_s = _csf_force(state, geom, sigma)
    k = geom["k"]
    cos_kx = torch.cos(k * geom["pos"][:, 0])
    dp_meas = 2.0 / geom["length"] * (traction * vol * cos_kx).sum()
    dp_exact = sigma * exact_wave_stress_coeff(k, geom["amplitude"])
    measure = float((delta_s * vol).sum() / geom["length"])
    kappa_mode = 2.0 / geom["length"] * (state.kappa * delta_s * vol * cos_kx).sum()
    kappa_exact = geom["amplitude"] * k * k
    ratio = float(dp_meas / dp_exact)
    return dict(dp_measured=float(dp_meas), dp_analytic=float(dp_exact),
                stiffness_ratio=ratio,
                omega_ratio=math.copysign(math.sqrt(abs(ratio)), ratio),
                omega_error_pct=(math.copysign(math.sqrt(abs(ratio)), ratio) - 1.0) * 100.0,
                surface_measure_ratio=measure,
                kappa_mode_ratio=float(kappa_mode / kappa_exact),
                n_particles=float(geom["pos"].shape[0]), h=h)


def drop_modal_stiffness(geom: dict, sigma: float, **kwargs) -> dict:
    """Modal normal-stress ratio of the lobed drop (Rayleigh/Lamb mode ``n``)."""
    state, h, _ = csf_state(geom, sigma, **kwargs)
    vol = _modal_projector(geom)
    traction, delta_s = _csf_force(state, geom, sigma)
    n = geom["mode"]
    theta = torch.atan2(geom["pos"][:, 1], geom["pos"][:, 0])
    cos_nt = torch.cos(n * theta)
    dp_meas = 2.0 / (2.0 * math.pi * geom["radius"]) * (traction * vol * cos_nt).sum()
    # Sign convention (shared with the wave case): the modal coefficient of the
    # traction ``t = f . n`` is ``-sigma kappa``; a *restoring* mode needs ``t``
    # opposite to the displacement, which is what makes the ratio below positive.
    # The mode-``n`` coefficient of the exact curvature of
    # ``r = R (1 + eps cos n theta)`` is ``(n^2 - 1) eps / R`` [1/m] (first order
    # in ``eps``; ``tests/test_surface_tension_dispersion.py`` pins it against the
    # closed-form curvature), so the analytic traction coefficient is negative.
    kappa_exact = (n * n - 1.0) * geom["eps"] / geom["radius"]
    dp_exact = -sigma * kappa_exact
    ratio = float(dp_meas / dp_exact)
    measure = float((delta_s * vol).sum() / (2.0 * math.pi * geom["radius"]))
    kappa_mode = 2.0 / (2.0 * math.pi * geom["radius"]) * (
        state.kappa * delta_s * vol * cos_nt).sum()
    return dict(dp_measured=float(dp_meas), dp_analytic=float(dp_exact),
                stiffness_ratio=ratio,
                omega_ratio=math.copysign(math.sqrt(abs(ratio)), ratio),
                omega_error_pct=(math.copysign(math.sqrt(abs(ratio)), ratio) - 1.0) * 100.0,
                surface_measure_ratio=measure,
                kappa_mode_ratio=float(kappa_mode / kappa_exact),
                n_particles=float(geom["pos"].shape[0]), h=h)


def exact_wave_stress_coeff(k: float, amplitude: float, n: int = 720) -> float:
    """Mode-``k`` coefficient of the exact normal stress ``sigma kappa ds/dx``.

    ``eta(x) = -a cos(kx)``; ``kappa = -eta'' (1 + eta'^2)^{-3/2}``; the arclength
    factor ``ds/dx`` is kept, because the modal projection is an area integral.
    Keeping the exact geometric curvature (instead of the linear ``a k^2``) means
    the reference stays valid at the finite amplitudes where the lattice noise is
    resolvable.
    """
    x = torch.linspace(0.0, 2.0 * math.pi / k, n + 1, dtype=DTYPE)[:-1]
    eta1 = amplitude * k * torch.sin(k * x)
    eta2 = amplitude * k * k * torch.cos(k * x)
    kappa = -eta2 / (1.0 + eta1 ** 2) ** 1.5
    weight = torch.sqrt(1.0 + eta1 ** 2)
    return float(-2.0 / n * (kappa * weight * torch.cos(k * x)).sum())


# --------------------------------------------------------------------------- #
# ensembles, cases, convergence
# --------------------------------------------------------------------------- #
def ensemble(builder, modal, shifts: int = 2) -> tuple[dict, dict]:
    """Average a modal estimator over a ``shifts x shifts`` sub-lattice ensemble.

    The interface cuts the particle lattice at an arbitrary phase and the CSF
    fields are not invariant under it (staircase).  The physical part of the
    modal stress is, so the estimator is averaged; the ensemble standard
    deviation is reported as the lattice noise a single realisation would carry.
    """
    rows = []
    for a in range(shifts):
        for b in range(shifts):
            geom = builder(a / shifts, b / shifts)
            rows.append(modal(geom))
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    std = {}
    for k in keys:
        vals = torch.tensor([r[k] for r in rows], dtype=torch.float64)
        std[k] = float(vals.std(unbiased=False))
    return mean, std


def capillary_wave_case(k: float, ppw: int, sigma: float = SIGMA, *,
                        amplitude_fraction: float = AMPLITUDE_FRACTION,
                        shifts: int = 2, periodic_x: bool = True,
                        wall_layers: int = 3,
                        curvature_scheme: str = "fit_quadratic",
                        delta_normalization: str = "legacy",
                        h_factor: float = H_FACTOR,
                        curvature_stencil: float = 4.0) -> dict:
    """One capillary-wave dispersion datum: wavenumber ``k``, ``ppw`` cells per domain."""
    dx = domain_length(k) / ppw
    lam = 2.0 * math.pi / k
    amplitude = amplitude_fraction * lam

    def builder(sx, sy):
        return wave_geometry(k, amplitude, dx, wall_layers=wall_layers, sx=sx, sy=sy)

    modal = lambda g: wave_modal_stiffness(  # noqa: E731
        g, sigma, curvature_scheme=curvature_scheme,
        delta_normalization=delta_normalization, h_factor=h_factor,
        curvature_stencil=curvature_stencil, periodic_x=periodic_x)
    mean, std = ensemble(builder, modal, shifts)
    omega_theory = math.sqrt(sigma * k ** 3 / RHO0)
    return dict(case=f"capillary_wave_k{k}", family="capillary_wave", k=k,
                wavelength=lam, ppw_domain=ppw,
                particles_per_wavelength=ppw / k, dx=dx, amplitude=amplitude,
                periodic_x=bool(periodic_x), wall_layers=int(wall_layers),
                curvature_scheme=curvature_scheme,
                delta_normalization=delta_normalization,
                n_particles=mean["n_particles"],
                dp_measured=mean["dp_measured"], dp_analytic=mean["dp_analytic"],
                stiffness_ratio=mean["stiffness_ratio"],
                stiffness_ratio_std=std["stiffness_ratio"],
                omega_theory=omega_theory,
                omega_measured=mean["omega_ratio"] * omega_theory,
                omega_ratio=mean["omega_ratio"],
                omega_error_pct=mean["omega_error_pct"],
                surface_measure_ratio=mean["surface_measure_ratio"],
                kappa_mode_ratio=mean["kappa_mode_ratio"])


def lamb_case(mode: int, radius_over_dx: float, sigma: float = SIGMA, *,
              radius: float = 0.02, eps: float = DROP_EPS, shifts: int = 2,
              curvature_scheme: str = "fit_quadratic",
              delta_normalization: str = "legacy",
              h_factor: float = H_FACTOR,
              curvature_stencil: float = 4.0) -> dict:
    """One Lamb-mode datum: azimuthal mode ``mode`` at ``radius_over_dx`` resolution."""
    dx = radius / radius_over_dx

    def builder(sx, sy):
        return drop_geometry(mode, eps, radius, dx, sx=sx, sy=sy)

    modal = lambda g: drop_modal_stiffness(  # noqa: E731
        g, sigma, curvature_scheme=curvature_scheme,
        delta_normalization=delta_normalization, h_factor=h_factor,
        curvature_stencil=curvature_stencil, periodic_x=False)
    mean, std = ensemble(builder, modal, shifts)
    omega_theory = math.sqrt(mode * (mode * mode - 1.0) * sigma / (RHO0 * radius ** 3))
    return dict(case=f"lamb_mode{mode}", family="lamb", mode=mode,
                radius=radius, radius_over_dx=radius_over_dx, eps=eps, dx=dx,
                periodic_x=False, wall_layers=0,
                curvature_scheme=curvature_scheme,
                delta_normalization=delta_normalization,
                n_particles=mean["n_particles"],
                dp_measured=mean["dp_measured"], dp_analytic=mean["dp_analytic"],
                stiffness_ratio=mean["stiffness_ratio"],
                stiffness_ratio_std=std["stiffness_ratio"],
                omega_theory=omega_theory,
                omega_measured=mean["omega_ratio"] * omega_theory,
                omega_ratio=mean["omega_ratio"],
                omega_error_pct=mean["omega_error_pct"],
                surface_measure_ratio=mean["surface_measure_ratio"],
                kappa_mode_ratio=mean["kappa_mode_ratio"])


def convergence_order(rows: list[dict], key: str = "omega_error_pct") -> float | None:
    """Least-squares slope of ``log|error|`` vs ``log(dx)`` (needs >= 2 points).

    A positive slope means the error shrinks under refinement; the mission gate
    is ``slope >= 2`` (second-order).
    """
    xs, ys = [], []
    for r in rows:
        err = abs(r[key])
        if err <= 0.0:
            continue
        xs.append(math.log(r["dx"]))
        ys.append(math.log(err))
    if len(xs) < 2:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def sigma_sensitivity(sigma: float = SIGMA, *, mode: int = 2,
                      radius_over_dx: float = 16.0, factors=(0.5, 1.0, 2.0),
                      shifts: int = 2, **kwargs) -> dict:
    """Linearity of the operator in ``sigma``.

    The CSF traction is ``-sigma kappa delta_s``, so the *modal stress* must be
    exactly proportional to ``sigma`` and the measured ``omega`` must scale as
    ``sqrt(sigma)``.  The reported spread of ``omega_ratio`` over the factors is
    therefore a check of the implementation (a sigma-dependence of the *shape*
    fields would show up here), not of the physics.  ``slope`` is the
    least-squares exponent of the *dimensional* modal stress ``dp_measured`` vs
    ``sigma`` (1.0 = linear); the quoted ``stiffness_ratio`` is sigma-normalised,
    so a correct operator has ``max_omega_ratio_spread ~ 0``.
    """
    rows = {}
    for factor in factors:
        row = lamb_case(mode, radius_over_dx, sigma * factor, shifts=shifts, **kwargs)
        rows[f"sigma_x{factor}"] = row
    base = rows["sigma_x1.0"]
    spread = max(abs(r["omega_ratio"] - base["omega_ratio"]) / abs(base["omega_ratio"])
                 for r in rows.values())
    xs = [math.log(f) for f in factors]
    # the *dimensional* modal stress must scale as sigma^1 (the ratio above is
    # sigma-normalised by construction, so its slope is 0 by definition)
    ys = [math.log(abs(rows[f"sigma_x{f}"]["dp_measured"])) for f in factors]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else None
    return dict(factors=list(factors),
                omega_ratio={key: r["omega_ratio"] for key, r in rows.items()},
                stiffness_ratio={key: r["stiffness_ratio"] for key, r in rows.items()},
                max_omega_ratio_spread=spread,
                max_spread_pct=spread * 100.0,
                stiffness_log_slope=slope,
                within_5pct=bool(spread * 100.0 < TOL_SIGMA_PCT))


# --------------------------------------------------------------------------- #
# audits: curvature recovery and the surface-delta measure
# --------------------------------------------------------------------------- #
def _surface_mask(geom: dict, state, rows: float = 3.0) -> torch.Tensor:
    """Particles in the top ``rows`` lattice rows *and* carrying interface weight."""
    y = geom["pos"][:, 1]
    delta_s = state.delta_s.clone()
    mask = geom.get("wall_mask")
    if mask is not None:
        delta_s = torch.where(mask, torch.zeros_like(delta_s), delta_s)
    top = y > y.max() - rows * geom["dx"]
    return top & (delta_s > 0.05 * delta_s.max())


def curvature_audit(k: float = 1.0, ppw: int = 32, *,
                    amplitude_fraction: float = AMPLITUDE_FRACTION,
                    shifts: int = 2, delta_normalization: str = "legacy",
                    curvature_scheme: str = "fit_quadratic") -> dict:
    """Curvature recovery of the CSF operator at a trough, a crest and a flat surface.

    The capillary-wave dispersion test needs ``kappa`` to have the right *sign*
    where the surface is gently curved.  This audit measures exactly that,
    independently of the modal projection:

    ``trough``   ``kappa`` at ``x = 0`` where the exact value is ``-a k^2``
    ``crest``    ``kappa`` at ``x = pi/k`` where the exact value is ``+a k^2``
    ``flat``     a flat free surface (``a = 0``, exact ``kappa = 0``); the
                 delta_s-weighted RMS there is the *noise floor* of the operator

    All three are averaged over the ``shifts x shifts`` sub-lattice ensemble.  A
    negative ``trough`` measurement (the exact value is negative) is a *correct*
    sign; ``sign_recovered`` is the honest verdict on the operator.
    """
    lam = 2.0 * math.pi / k
    dx = domain_length(k) / ppw

    def _measure(amplitude: float, centres: tuple[float, ...]):
        values = {c: [] for c in centres}
        for a in range(shifts):
            for b in range(shifts):
                geom = wave_geometry(k, amplitude, dx, sx=a / shifts, sy=b / shifts)
                state, _, _ = csf_state(geom, SIGMA,
                                        delta_normalization=delta_normalization,
                                        curvature_scheme=curvature_scheme)
                surface = _surface_mask(geom, state)
                x = geom["pos"][:, 0]
                weight = state.delta_s * geom["mass"] / RHO0
                for centre in centres:
                    sel = surface & (torch.abs(x - centre) < 0.15 * lam)
                    if not bool(sel.any()):
                        continue
                    values[centre].append(float((state.kappa[sel] * weight[sel]).sum()
                                                / weight[sel].sum()))
        return {c: (sum(v) / len(v) if v else float("nan")) for c, v in values.items()}

    amp = amplitude_fraction * lam
    trough_c, crest_c = 0.0, math.pi / k
    moving = _measure(amp, (trough_c, crest_c))
    flat = _measure(0.0, (trough_c,))
    kappa_exact = amp * k * k

    def _rms_flat(amp0: float) -> float:
        vals = []
        for a in range(shifts):
            for b in range(shifts):
                geom = wave_geometry(k, amp0, dx, sx=a / shifts, sy=b / shifts)
                state, _, _ = csf_state(geom, SIGMA,
                                        delta_normalization=delta_normalization,
                                        curvature_scheme=curvature_scheme)
                surface = _surface_mask(geom, state)
                weight = state.delta_s * geom["mass"] / RHO0
                vals.append(float((state.kappa[surface] ** 2 * weight[surface]).sum()
                                  / weight[surface].sum()))
        return math.sqrt(sum(vals) / len(vals))

    noise = _rms_flat(0.0)
    trough_meas, crest_meas = moving[trough_c], moving[crest_c]
    sign_ok = (trough_meas < 0.0 < crest_meas)
    return dict(k=k, ppw=ppw, dx=dx, amplitude=amp,
                exact_kappa=kappa_exact,
                trough=dict(measured_kappa=trough_meas, exact_kappa=-kappa_exact,
                            ratio=trough_meas / -kappa_exact if kappa_exact else None),
                crest=dict(measured_kappa=crest_meas, exact_kappa=kappa_exact,
                           ratio=crest_meas / kappa_exact if kappa_exact else None),
                flat_noise_rms=noise,
                flat=flat,
                sign_recovered=bool(sign_ok),
                signal_to_noise=abs(kappa_exact) / noise if noise > 0 else float("inf"),
                delta_normalization=delta_normalization,
                curvature_scheme=curvature_scheme)


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    na = torch.linalg.vector_norm(a, dim=1).clamp_min(1e-30)
    nb = torch.linalg.vector_norm(b, dim=1).clamp_min(1e-30)
    cos = ((a * b).sum(dim=1) / (na * nb)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def normal_audit(k: float = 1.0, ppw: int = 24, *, shifts: int = 2,
                 amplitude_fraction: float = AMPLITUDE_FRACTION) -> dict:
    """Angular error of the CSF surface normal, and the divergence control.

    ``kappa = div n`` inherits the angular error of ``n`` divided by the stencil
    length, so this audit separates the two halves of the curvature operator:

    ``angle_mean_deg`` / ``angle_p95_deg``
        angle between the CSF normal (``-grad c / |grad c|``, Shepard smoothed)
        and the *analytic* normal of the wave surface, at the surface particles.
    ``trough_csf_normal`` / ``trough_analytic_normal``
        the same modal curvature as :func:`curvature_audit`, computed (a) with the
        CSF normal and (b) with the exact normal fed into the *same* MLS
        divergence.  If (b) is close to the exact value while (a) is not, the
        defect is the normal estimator, not the divergence.
    """
    lam = 2.0 * math.pi / k
    dx = domain_length(k) / ppw
    amp = amplitude_fraction * lam
    angles, kappa_csf, kappa_exact_normal = [], [], []
    for a in range(shifts):
        for b in range(shifts):
            geom = wave_geometry(k, amp, dx, sx=a / shifts, sy=b / shifts)
            state, h, _ = csf_state(geom, SIGMA)
            vol = geom["mass"] / RHO0
            surface = _surface_mask(geom, state)
            weight = state.delta_s * vol
            angles.append(float((_angle_deg(state.normal, geom["normal"])[surface]
                                 * weight[surface]).sum() / weight[surface].sum()))
            pairs = cell_linked_pairs(geom["pos"], 4.0 * h)
            kappa_ref = mls_divergence(geom["pos"], geom["normal"], vol, pairs, h)
            x = geom["pos"][:, 0]
            trough = surface & (torch.abs(x) < 0.15 * lam)
            kappa_csf.append(float((state.kappa[trough] * weight[trough]).sum()
                                   / weight[trough].sum()))
            kappa_exact_normal.append(float((kappa_ref[trough] * weight[trough]).sum()
                                            / weight[trough].sum()))
    exact = -amp * k * k
    return dict(k=k, ppw=ppw, exact_trough_kappa=exact,
                angle_mean_deg=sum(angles) / len(angles),
                angle_p95_deg=float(torch.tensor(angles).quantile(0.95)),
                trough_csf_normal=sum(kappa_csf) / len(kappa_csf),
                trough_analytic_normal=sum(kappa_exact_normal) / len(kappa_exact_normal),
                normal_error_dominates=bool(
                    abs(sum(kappa_exact_normal) / len(kappa_exact_normal) - exact)
                    < 0.5 * abs(sum(kappa_csf) / len(kappa_csf) - exact)))


def curvature_noise_scan(ppws=(16, 24, 32, 48), *, k: float = 1.0,
                         shifts: int = 2) -> list[dict]:
    """Noise floor of the CSF curvature versus resolution (the core defect)."""
    rows = []
    for ppw in ppws:
        audit = curvature_audit(k, ppw, shifts=shifts)
        rows.append(dict(ppw=ppw, dx=audit["dx"],
                         flat_noise_rms=audit["flat_noise_rms"],
                         signal=audit["exact_kappa"],
                         signal_to_noise=audit["signal_to_noise"],
                         trough_measured=audit["trough"]["measured_kappa"],
                         crest_measured=audit["crest"]["measured_kappa"],
                         sign_recovered=audit["sign_recovered"]))
    return rows


def delta_measure_audit(ppw: int = 32, *, radius_over_dx: float = 32.0,
                        mode: int = 2, shifts: int = 2) -> dict:
    """``sum_i delta_s,i V_i / A`` on the wave and on the drop, both normalisations.

    ``1.0`` is the CSF requirement (Brackbill et al. 1992: the surface delta
    function integrates to one interface area).  Both geometries are averaged over
    the ``shifts x shifts`` sub-lattice ensemble because the integral is *also*
    phase sensitive: on the free surface a single phase can sit anywhere in
    0.7-1.3, which is why the ensemble mean and its spread are both reported.
    """
    out: dict[str, dict] = {}
    for norm in ("legacy", "local_range"):
        wave_vals, drop_vals = [], []
        for a in range(shifts):
            for b in range(shifts):
                geom = wave_geometry(1.0, AMPLITUDE_FRACTION * 2.0 * math.pi,
                                     domain_length(1.0) / ppw, sx=a / shifts,
                                     sy=b / shifts)
                state, _, _ = csf_state(geom, SIGMA, delta_normalization=norm)
                delta_s = state.delta_s.clone()
                delta_s[geom["wall_mask"]] = 0.0
                wave_vals.append(float((delta_s * geom["mass"] / RHO0).sum()
                                       / geom["length"]))
                drop = drop_geometry(mode, DROP_EPS, 0.02, 0.02 / radius_over_dx,
                                     sx=a / shifts, sy=b / shifts)
                state, _, _ = csf_state(drop, SIGMA, delta_normalization=norm,
                                        periodic_x=False)
                drop_vals.append(float((state.delta_s * drop["mass"] / RHO0).sum()
                                       / (2.0 * math.pi * drop["radius"])))
        wave_t = torch.tensor(wave_vals, dtype=torch.float64)
        drop_t = torch.tensor(drop_vals, dtype=torch.float64)
        slot = out.setdefault("capillary_wave_k1", {})
        slot[norm] = float(wave_t.mean())
        slot[norm + "_std"] = float(wave_t.std(unbiased=False))
        slot2 = out.setdefault(f"drop_l{mode}", {})
        slot2[norm] = float(drop_t.mean())
        slot2[norm + "_std"] = float(drop_t.std(unbiased=False))
    return out


# --------------------------------------------------------------------------- #
# stability limit (integration point of this module)
# --------------------------------------------------------------------------- #
def capillary_dt_limit(h: float, sigma: float, rho0: float = RHO0,
                       cfl: float = 0.25, k_max: float | None = None) -> float:
    """Largest explicit step for the capillary force at smoothing length ``h``.

    A capillary wave of wavenumber ``k`` oscillates at ``omega^2 = sigma k^3/rho0``,
    so an explicit integrator with safety factor ``cfl`` needs
    ``dt <= cfl / sqrt(sigma k_max^3 / rho0)`` where ``k_max`` is the stiffest
    wavenumber the discretisation carries.

    ``k_max = pi/h`` (default)
        the grid-scale wave, ``lambda_min ~ 2 h`` -- the honest limit for a
        particle method: ``dt = cfl sqrt(rho0 h^3 / (sigma pi^3))``.
    ``k_max = 1/h``
        the historical convention of :func:`tensorsph.surface_tension.capillary_dt_limit`
        (``dt = cfl sqrt(rho0 h^3 / sigma)``), kept reachable so that wrapper stays
        bit-for-bit compatible; it is ``pi^{3/2} = 5.6``x looser.

    ``cfl = 0.25`` matches the acoustic limit of the solvers.  ``sigma = 0``
    returns ``inf`` so callers can take a plain ``min`` without branching.
    """
    if sigma <= 0.0:
        return math.inf
    k_star = (math.pi / h) if k_max is None else k_max
    return cfl / math.sqrt(sigma * k_star ** 3 / rho0)


# --------------------------------------------------------------------------- #
# validation driver
# --------------------------------------------------------------------------- #
def _repro_cmd(args_namespace=None) -> str:
    return ("PYTHONPATH=src python -m tensorsph.surface_tension_dispersion "
            "--ppw 24,32,48 --ks 1,2,3 --modes 2,3,4 --rdx 16,24,32 --shifts 2 "
            "--out runs/surface_tension_dispersion/report.json")


def run_validation(*, ks=(1.0, 2.0, 3.0), ppws=(24, 32, 48), modes=(2, 3, 4),
                   rdxs=(16, 24, 32), sigma: float = SIGMA, shifts: int = 2,
                   periodic_x: bool = True, wall_layers: int = 3,
                   compare_defective: bool = False, with_sigma: bool = True,
                   option_screen: bool = True,
                   curvature_scheme: str = "fit_quadratic",
                   delta_normalization: str = "legacy") -> dict:
    """Run the full dispersion / Lamb / convergence / sigma-sensitivity campaign.

    ``compare_defective=True`` adds the old (closed-block, non-periodic) geometry
    as an A/B control; it is the configuration that made m13895's ``C1`` fail and
    is kept so the fix can be audited.
    """
    t0 = time.time()
    common = dict(sigma=sigma, shifts=shifts,
                  curvature_scheme=curvature_scheme,
                  delta_normalization=delta_normalization)
    rows: list[dict] = []
    for k in ks:
        for ppw in ppws:
            rows.append(capillary_wave_case(k, ppw, periodic_x=periodic_x,
                                            wall_layers=wall_layers, **common))
    for mode in modes:
        for rdx in rdxs:
            rows.append(lamb_case(mode, rdx, **common))

    control: list[dict] = []
    if compare_defective:
        for k in ks:
            for ppw in (ppws[-1],):
                control.append(capillary_wave_case(k, ppw, periodic_x=False,
                                                   wall_layers=0, **common))

    conv = {}
    for k in ks:
        conv[f"capillary_wave_k{k}"] = convergence_order(
            [r for r in rows if r["family"] == "capillary_wave" and r["k"] == k])
    for mode in modes:
        conv[f"lamb_mode{mode}"] = convergence_order(
            [r for r in rows if r["family"] == "lamb" and r["mode"] == mode])

    # option screen: does *any* shipped combination of curvature scheme and delta
    # normalisation meet the gates?  Run at the *finest* grid of the campaign so the
    # screen is directly comparable with the gate numbers above.
    screen = []
    for norm in (("legacy", "local_range") if option_screen else ()):
        for scheme in ("mls", "fit_quadratic"):
            wave = capillary_wave_case(ks[0], max(ppws), sigma=sigma, shifts=shifts,
                                       curvature_scheme=scheme,
                                       delta_normalization=norm)
            drop = lamb_case(modes[0], max(rdxs), sigma=sigma, shifts=shifts,
                             curvature_scheme=scheme, delta_normalization=norm)
            screen.append(dict(delta_normalization=norm, curvature_scheme=scheme,
                               wave_error_pct=wave["omega_error_pct"],
                               wave_surface_measure=wave["surface_measure_ratio"],
                               lamb_error_pct=drop["omega_error_pct"],
                               lamb_surface_measure=drop["surface_measure_ratio"],
                               wave_pass=bool(abs(wave["omega_error_pct"]) <= TOL_WAVE_PCT),
                               lamb_pass=bool(abs(drop["omega_error_pct"]) <= TOL_LAMB_PCT)))

    sens = sigma_sensitivity(sigma, factors=(0.5, 1.0, 2.0), shifts=shifts,
                             curvature_scheme=curvature_scheme,
                             delta_normalization=delta_normalization) if with_sigma else {}

    # benchmark table: analytic reference vs measured, tolerance and verdict
    table = []
    for r in rows:
        tol = TOL_WAVE_PCT if r["family"] == "capillary_wave" else TOL_LAMB_PCT
        table.append(dict(case=r["case"], ppw_domain=r.get("ppw_domain"),
                          radius_over_dx=r.get("radius_over_dx"),
                          quantity="omega", ref=r["omega_theory"],
                          got=r["omega_measured"],
                          rel_err_pct=r["omega_error_pct"], tolerance_pct=tol,
                          passed=bool(abs(r["omega_error_pct"]) <= tol)))
    finest_wave = [r for r in table if r["case"].startswith("capillary_wave")
                   and r["ppw_domain"] == max(ppws)]
    finest_lamb = [r for r in table if r["case"].startswith("lamb")
                   and r["radius_over_dx"] == max(rdxs)]
    c1 = max((abs(r["rel_err_pct"]) for r in finest_wave), default=None)
    c2 = max((abs(r["rel_err_pct"]) for r in finest_lamb), default=None)
    orders = [v for v in conv.values() if v is not None]
    gates = dict(
        C1_capillary_wave=dict(max_abs_error_pct=c1, tolerance_pct=TOL_WAVE_PCT,
                               passed=bool(c1 is not None and c1 <= TOL_WAVE_PCT)),
        C2_lamb_modes=dict(max_abs_error_pct=c2, tolerance_pct=TOL_LAMB_PCT,
                           passed=bool(c2 is not None and c2 <= TOL_LAMB_PCT)),
        C3_convergence=dict(orders=conv, min_order=(min(orders) if orders else None),
                            required=MIN_ORDER,
                            passed=bool(orders and min(orders) >= MIN_ORDER)),
        C4_sigma_sensitivity=dict(max_spread_pct=sens.get("max_spread_pct"),
                                  tolerance_pct=TOL_SIGMA_PCT,
                                  passed=bool(sens.get("within_5pct", False))) if sens else
        dict(passed=False, note="skipped"),
    )
    report = dict(
        benchmark="surface_tension_dispersion",
        schema="tensorsph.surface_tension_dispersion/1",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        environment=dict(python=os.sys.version.split()[0], torch=torch.__version__,
                         threads=int(torch.get_num_threads())),
        reference=dict(
            capillary_wave="omega^2 = sigma k^3 / rho (Lamb 1932, Art. 267)",
            drop="omega^2 = n(n^2-1) sigma / (rho R^3) (Rayleigh 1879; Lamb Art. 275)",
            measurement=("omega_measured^2/omega_theory^2 = modal normal-stress "
                         "ratio of the discretised CSF operator; the inertia is "
                         "the exact potential-flow value, so the score isolates the "
                         "surface-force operator (linear-response / discrete "
                         "dispersion, not a time integration)")),
        setup=dict(sigma=sigma, rho0=RHO0, h_factor=H_FACTOR,
                   amplitude_fraction=AMPLITUDE_FRACTION, drop_eps=DROP_EPS,
                   shifts=shifts, periodic_x=periodic_x, wall_layers=wall_layers,
                   curvature_scheme=curvature_scheme,
                   delta_normalization=delta_normalization,
                   domain_length=DOMAIN_LENGTH, ks=list(ks), ppws=list(ppws),
                   modes=list(modes), rdxs=list(rdxs)),
        rows=rows, control_defective=control, convergence=conv, option_screen=screen,
        option_screen_best=dict(
            wave_min_abs_error_pct=min((abs(r["wave_error_pct"]) for r in screen),
                                       default=None),
            lamb_min_abs_error_pct=min((abs(r["lamb_error_pct"]) for r in screen),
                                       default=None),
            any_wave_pass=any(r["wave_pass"] for r in screen),
            any_lamb_pass=any(r["lamb_pass"] for r in screen),
            screened=len(screen)),
        sigma_sensitivity=sens, benchmark_table=table, gates=gates,
        wall_clock_s=time.time() - t0, repro_cmd=_repro_cmd())
    report["failed_gates"] = [k for k, v in gates.items() if not v.get("passed")]
    report["status"] = "PASS" if not report["failed_gates"] else "FAIL"
    return report


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ppw", default="24,32,48")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--modes", default="2,3,4")
    ap.add_argument("--rdx", default="16,24,32")
    ap.add_argument("--shifts", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--scheme", default="fit_quadratic")
    ap.add_argument("--norm", default="legacy")
    ap.add_argument("--defective-control", action="store_true",
                    help="also measure the old closed-block geometry (A/B control)")
    ap.add_argument("--skip-sigma", action="store_true")
    ap.add_argument("--skip-screen", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="runs/surface_tension_dispersion/report.json")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    ints = lambda txt: [int(v) for v in str(txt).split(",") if v]   # noqa: E731
    floats = lambda txt: [float(v) for v in str(txt).split(",") if v]  # noqa: E731
    if args.quick:
        ppws, ks, modes, rdxs, shifts = (24, 32), (1.0, 2.0), (2, 3), (16, 24), 1
    else:
        ppws, ks = ints(args.ppw), floats(args.ks)
        modes, rdxs, shifts = ints(args.modes), ints(args.rdx), args.shifts
    report = run_validation(ks=ks, ppws=ppws, modes=modes, rdxs=rdxs,
                            sigma=args.sigma, shifts=shifts,
                            compare_defective=args.defective_control,
                            with_sigma=not args.skip_sigma,
                            option_screen=not args.skip_screen,
                            curvature_scheme=args.scheme,
                            delta_normalization=args.norm)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "failed_gates": report["failed_gates"],
                      "gates": {k: {kk: vv for kk, vv in v.items() if kk != "orders"}
                                for k, v in report["gates"].items()},
                      "out": str(out)}, indent=1))
    return report


if __name__ == "__main__":
    main()