"""Oscillating 2-D drop: free-oscillation frequency vs the analytic capillary law.

Why this benchmark exists
-------------------------
``docs/algorithm-gap-analysis.md`` closes B3 (surface tension) only to the extent
that the *static* Young-Laplace law is checked.  The oscillating-droplet half of
B3 is still open (the same row also asks for a "quiet free-surface case in zero
g" to size the noise floor of the diffusive density term).  A static drop tests
only the force *amplitude*; a free oscillation additionally tests the *dynamics*
(force -> acceleration -> interface kinematics) and it has a closed-form
reference, so it is the cheapest honest way to score surface tension beyond the
Laplace jump.

Analytic reference (2-D, azimuthal mode m, inviscid)
---------------------------------------------------
For an infinitely long inviscid cylinder whose surface is perturbed as
``r(theta) = R (1 + eps cos m theta)`` the linearised Laplace pressure of the
in-plane curvature is

    p_in - p_out = sigma kappa,
    kappa = (1/R) [1 + (m^2 - 1) eta / R] + O(eta^2)                    (1)

(the ``m^2 - 1`` follows from expanding
``kappa = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^{3/2}`` for
``r = R + eta cos m theta``).  Balancing (1) against the Bernoulli pressure of
the irrotational interior field ``phi = A r^m cos m theta`` and the kinematic
condition at ``r = R`` gives the Rayleigh-Lamb frequency

    omega_m^2 = sigma * m (m^2 - 1) / (rho R^3),   m >= 2                    (2)

so mode 2, the mode an ellipse excites, is ``omega_2 = sqrt(6 sigma / (rho R^3))``.
References: Rayleigh (1879) Proc. London Math. Soc. 10:4; Lamb (1932)
*Hydrodynamics* 6th ed. art. 267; Chandrasekhar (1961) *Hydrodynamic and
Hydromagnetic Stability* ch. XII.

Two traps this harness is written against
-----------------------------------------
* The **3-D** sphere law is ``omega_m^2 = sigma m (m-1)(m+2)/(rho R^3)``, i.e.
  ``8 sigma/(rho R^3)`` for m = 2, which is 15.5 % high in *frequency* -- the
  same size as the platform gate.  Using it on a 2-D drop silently turns every
  "pass" into a coin flip.  ``tests/test_oscillating_droplet.py`` pins the 2-D
  coefficient with an independent discrete check.
* A drop released from an *ellipse* conserves area only if
  ``a = R (1 + eps)``, ``b = R / (1 + eps)``.  Using ``b = R (1 - eps)``
  injects an ``O(eps^2)`` volume error which shifts the mean radius and biases
  the frequency.

Measured quantities
-------------------
``s(t) = (A - B) / (A + B)`` where ``(A, B) = 2 sqrt(eigenvalues)`` of the
mass-weighted second-moment tensor of the particle cloud (a uniform ellipse has
``<x^2> = a^2/4``).  For ``r = R (1 + eps cos 2 theta)`` this is ``s = eps`` to
first order in ``eps``.  The mode-2 projection of ``r - R_eff`` is evaluated as
an independent estimator.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_droplet/run.py --mode frequency
    PYTHONPATH=src python benchmarks/oscillating_droplet/run.py --mode ladder --plot
    PYTHONPATH=src python benchmarks/oscillating_droplet/run.py --mode damping --plot
    PYTHONPATH=src python benchmarks/oscillating_droplet/run.py --mode all --plot
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    rhs,
    stable_dt,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent

# Platform acceptance bar (docs/PROJECT_CHARTER.md): >15 % is unacceptable.
GATE_PCT = 15.0
MODE = 2


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def elliptical_drop(radius: float, dx: float, rho0: float,
                    eps: float):
    """Hexagonal-packed drop shaped as an area-conserving ellipse.

    ``a = R (1 + eps)``, ``b = R / (1 + eps)`` so that ``a * b = R^2``: the
    reference area is preserved to machine precision, which removes the
    ``O(eps^2)`` mean-radius bias of the naive ``b = R (1 - eps)``.
    """
    a = radius * (1.0 + eps)
    b = radius / (1.0 + eps)
    dy = dx * math.sqrt(3.0) / 2.0
    mass = torch.tensor(rho0 * dx * dy, dtype=torch.float64)
    ys = torch.arange(-b, b + 0.5 * dy, dy, dtype=torch.float64)
    xs = torch.arange(-a, a + 0.5 * dx, dx, dtype=torch.float64)
    pts = []
    for row, y in enumerate(ys):
        offset = 0.5 * dx if (row % 2) else 0.0
        xr = xs + offset
        keep = (xr / a) ** 2 + (y / b) ** 2 <= 1.0
        if keep.any():
            pts.append(torch.stack([xr[keep], y.expand_as(xr)[keep]], dim=1))
    pos = torch.cat(pts, dim=0)
    pos = pos - pos.mean(dim=0)
    # ``rhs``/``density_from_neighbor_pairs`` require an (N,) mass vector.
    return pos, mass.expand(pos.shape[0]).clone()


def lobed_droplet(radius: float, dx: float, rho0: float, eps: float,
                  mode: int = MODE) -> tuple[torch.Tensor, torch.Tensor]:
    """Hexagonal drop with ``r(theta) = R (1 + eps cos m theta)``, centred at 0.

    :func:`elliptical_drop` is the ``m = 2`` special case; this generalisation
    covers the higher azimuthal modes of the same Rayleigh-Lamb family
    (``omega_3^2 = 24 sigma / (rho R^3)``), which is what the ``m >= 3`` branch
    of eq. 2 needs to be scored at all.

    The boundary is the analytic lobe ``r <= R (1 + eps cos m theta)``, so the
    enclosed area is ``(R^2/2) Int (1 + eps cos m theta)^2 dtheta =
    pi R^2 (1 + eps^2/2)`` for every ``m >= 1`` -- the ``eps^2/2`` term is the
    lattice-level area inflation, i.e. the same ``O(eps^2)`` trap that
    :func:`elliptical_drop` removes by construction for ``m = 2``.

    Merged from the m2919 lineage of this benchmark (PR #87): the m2919
    implementation scored the ``m = 3`` case with it, and its measurement is
    preserved verbatim in ``validation_report_m2919.json``.
    """
    if mode < 1:
        raise ValueError(f"mode m = {mode} is not an azimuthal mode")
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    reach = radius * (1.0 + abs(eps))
    pts = []
    j = 0
    y = -reach
    while y <= reach:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -reach + offset
        while x <= reach:
            r = math.hypot(x, y)
            if r > 1e-12:
                theta = math.atan2(y, x)
                if r <= radius * (1.0 + eps * math.cos(mode * theta)):
                    pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    pos = pos - pos.mean(dim=0)
    return pos, torch.full((pos.shape[0],), mass, dtype=torch.float64)


def analytic_omega(radius: float, sigma: float, rho0: float,
                   m: int = MODE) -> float:
    """``omega_m = sqrt(sigma m (m^2 - 1) / (rho R^3))`` (2-D, eq. 2).

    ``m = 1`` is a rigid translation of the cylinder (``m^2 - 1 = 0``, no
    restoring force), so it is rejected instead of silently returning 0.
    """
    if m < 2:
        raise ValueError(
            f"mode m = {m} is a translation of the cylinder: no restoring "
            "force, so there is no frequency (m >= 2 required)")
    return math.sqrt(sigma * m * (m * m - 1.0) / (rho0 * radius ** 3))


def analytic_omega_3d(radius: float, sigma: float, rho0: float,
                      m: int = MODE) -> float:
    """3-D sphere law, kept only to document the 15.5 % trap in the docstring."""
    return math.sqrt(sigma * m * (m - 1.0) * (m + 2.0) / (rho0 * radius ** 3))


# --------------------------------------------------------------------------- #
# falsifiable gauges (verifier seat, 2026-11-06)
# --------------------------------------------------------------------------- #
# A free drop has no wall and no box: the only thing that keeps the cloud
# together is the surface-tension force itself.  "The fit missed the analytic
# frequency" is therefore a statement about surface tension only as long as the
# cloud *stayed a drop*.  The gauges below make that precondition falsifiable.
# The gate that used to carry it, ``diagnostics.mass_relative_drift``, cannot
# fail by construction: ``mass`` is a per-particle attribute that the
# integrator never touches, so it is 0.0 for *any* trajectory, including one
# that has flung 97 % of the mass off the drop (see ``docs/
# verification_oscillating_droplet_gauge_audit.md``).
RADIUS_GATE = 1.30        # rms radius / R; a uniform ellipse sits at 0.707-0.714
ESCAPE_FACTOR = 1.5       # a particle farther than this * R has left the drop
ESCAPE_GATE = 0.02        # ... and this mass fraction outside is already fatal
SPEED_GATE_FACTOR = 1.0   # |v| / c0 above which the run is not a capillary flow


def containment_diagnostics(pos: torch.Tensor, mass: torch.Tensor,
                            radius: float,
                            escape_factor: float = ESCAPE_FACTOR) -> dict:
    """Is the cloud still a drop of radius ``R``?

    Three dimensionless, wall-free gauges: the mass-weighted rms radius in units
    of the reference radius, the outermost particle radius, and the mass
    fraction beyond ``escape_factor * R``.  All three are 0.71 / ~1.1 / 0.0 for
    a healthy ellipse and grow without bound once the drop disperses, so unlike
    the tracked mass they can actually fail.
    """
    if mass.dim() == 0:
        mass = mass.expand(pos.shape[0])
    m_tot = mass.sum()
    com = (mass[:, None] * pos).sum(dim=0) / m_tot
    r = torch.linalg.vector_norm(pos - com, dim=1)
    rms = float(((mass * r.square()).sum() / m_tot).sqrt())
    escaped = float((mass * (r > escape_factor * radius)).sum() / m_tot)
    rms_over_r = rms / radius
    return {
        "rms_radius_over_radius": rms_over_r,
        "max_radius_over_radius": float(r.max()) / radius,
        "escaped_mass_fraction": escaped,
        "intact": bool(rms_over_r <= RADIUS_GATE and escaped <= ESCAPE_GATE),
    }


def spurious_momentum(vel: torch.Tensor, mass: torch.Tensor,
                      reference_velocity: float) -> float:
    """``|sum m v| / (M v_ref)`` -- the momentum residual of a free drop.

    Pressure and viscous interactions are antisymmetric over particle pairs, so
    the total momentum of an isolated drop must stay at its initial value
    (zero here).  The CSF surface-tension force is a per-particle continuum
    force and is not pairwise symmetric, so whatever residual shows up is a
    genuine, falsifiable conservation error of the scheme -- in contrast with
    the (structurally conserved) Lagrangian mass.
    """
    if mass.dim() == 0:
        mass = mass.expand(vel.shape[0])
    m_tot = mass.sum()
    p = (mass[:, None] * vel).sum(dim=0)
    return float(torch.linalg.vector_norm(p) / (m_tot * abs(reference_velocity)))


def divergence_onset(ts, max_speed, speed_limit: float) -> dict:
    """First sampled time at which some particle exceeds ``speed_limit``.

    The reference speed here is the sound speed ``c0`` of the *same* run: a
    capillary flow of this drop moves at ``sqrt(sigma/(rho R)) ~ 0.6 m/s``, i.e.
    3 % of ``c0 = 20 m/s``, so anything above ``c0`` is no longer the physics
    the analytic law describes.
    """
    for t, v in zip(ts, max_speed):
        if v > speed_limit:
            return {"detected": True, "onset_time_s": t, "speed_at_onset": v}
    return {"detected": False, "onset_time_s": None, "speed_at_onset": None}


def windowed_frequency(ts, s, omega_ref: float,
                       windows=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float("inf")),
                       min_samples: int = 8) -> list:
    """Fit the damped cosine on ``t <= w * T_ref`` for a sweep of windows ``w``.

    The frequency of a diverging run is a function of the fit window, and the
    window is not a physical parameter: a stable reading must be flat in ``w``.
    Reporting the sweep makes the collapse visible instead of hiding it behind
    one number fitted over the whole (blown-up) trace.
    """
    t_ref = 2.0 * math.pi / omega_ref
    rows = []
    for w in windows:
        limit = w * t_ref if math.isfinite(w) else float("inf")
        idx = [i for i, t in enumerate(ts) if t <= limit]
        if len(idx) < min_samples:
            continue
        fit = damped_cosine_fit([ts[i] for i in idx], [s[i] for i in idx],
                                omega_ref)
        rows.append({
            "window_periods": w if math.isfinite(w) else None,
            "window_end_s": (limit if math.isfinite(w) else ts[int(idx[-1])]),
            "n_samples": len(idx),
            "omega_rad_per_s": fit["omega"],
            "rel_error_pct": (fit["omega"] - omega_ref) / omega_ref * 100.0,
            "lambda_per_s": fit["lambda_per_s"],
            "r2": fit["r2"],
        })
    return rows


# --------------------------------------------------------------------------- #
# mode amplitude estimators
# --------------------------------------------------------------------------- #
def mode_amplitude(pos: torch.Tensor, mass: torch.Tensor) -> dict:
    """Mode-2 amplitude from the mass-weighted second-moment tensor.

    A uniform ellipse has ``<x^2> = a^2/4``, ``<y^2> = b^2/4``, so the
    eigenvalues of the covariance give ``A = 2 sqrt(lambda_1)``,
    ``B = 2 sqrt(lambda_2)`` and ``s = (A - B)/(A + B) = eps`` to first order
    in ``eps`` for ``r = R (1 + eps cos 2 theta)``.  The second moment is an
    integral over the whole drop, so it is far less sensitive to the lattice
    jaggedness than an interface-radius estimator (see the note in
    ``docs/verification_oscillating_droplet.md``: the radius-hull Fourier
    coefficient of this very lattice returns 0.055-0.13 for ``eps = 0.10``).
    """
    # ``mass`` is a broadcast scalar for a monodisperse lattice (as in
    # benchmarks/laplace_law); expand it so the estimator is dimension-safe.
    if mass.dim() == 0:
        mass = mass.expand(pos.shape[0])
    m_tot = mass.sum()
    com = (mass[:, None] * pos).sum(dim=0) / m_tot
    d = pos - com
    cov = (mass[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(dim=0) / m_tot
    lam = torch.linalg.eigvalsh(cov)          # ascending
    lam1 = float(lam[-1].clamp_min(1e-30))
    lam2 = float(lam[0].clamp_min(1e-30))
    big, small = 2.0 * math.sqrt(lam1), 2.0 * math.sqrt(lam2)
    return {"s_moment": (big - small) / (big + small),
            "semi_major": big, "semi_minor": small,
            "r_eff": 2.0 * math.sqrt(0.5 * (lam1 + lam2))}


def mode_quadrupole(pos: torch.Tensor, mass: torch.Tensor) -> float:
    """Signed mode-2 quadrupole ``sum m (x^2 - y^2) / sum m r^2`` about the COM.

    The *oriented* twin of :func:`mode_amplitude`: for
    ``r = R (1 + eps cos 2 theta)`` a uniform ellipse gives ``3 eps / 2`` up to
    the ``r^2`` weighting, and a 90 degree rotation flips its sign, whereas the
    eigenvalue estimator above returns ``|s|`` and is blind to that flip.
    Both are kept (merged from the m2919 lineage, PR #87) because they answer
    different questions: :func:`mode_amplitude` is the amplitude used for the
    frequency fit (robust to lattice jaggedness, no sign), and this one is the
    estimator the m2919 runs measured -- the numbers in
    ``validation_report_m2919.json`` are its output.
    """
    if mass.dim() == 0:
        mass = mass.expand(pos.shape[0])
    com = (mass[:, None] * pos).sum(dim=0) / mass.sum()
    rel = pos - com
    r2 = rel.square().sum(dim=1)
    quad = (mass * (rel[:, 0] ** 2 - rel[:, 1] ** 2)).sum()
    return float(quad / (mass * r2).sum())


# --------------------------------------------------------------------------- #
# signal analysis
# --------------------------------------------------------------------------- #
def zero_crossing_period(t: list[float], s: list[float]) -> dict:
    """Period from linearly interpolated zero crossings (both directions).

    Every sign change is a half period, so the estimator needs only ~1.5
    oscillations to report a period; the mean of the trace is removed first so
    the residual offset of a damped signal does not fake a crossing.
    """
    n = len(s)
    mean = sum(s) / n
    cross = []
    for i in range(n - 1):
        a, b = s[i] - mean, s[i + 1] - mean
        if (a <= 0.0 < b) or (a >= 0.0 > b):
            w = -a / (b - a)
            cross.append(t[i] + w * (t[i + 1] - t[i]))
    if len(cross) < 3:
        return {"n_crossings": len(cross), "period": float("nan"),
                "omega": float("nan")}
    half = (cross[-1] - cross[0]) / (len(cross) - 1)
    return {"n_crossings": len(cross), "half_period": half,
            "period": 2.0 * half, "omega": 2.0 * math.pi / (2.0 * half)}


def damped_cosine_fit(t: list[float], s: list[float],
                      omega_guess: float) -> dict:
    """Least-squares fit of ``s = c + e^{-lam t}(P cos wt + Q sin wt)``.

    Grid search over ``(omega, lam)`` around ``omega_guess``; for each pair the
    three linear coefficients ``(c, P, Q)`` are solved in closed form, so the
    search has no local minima of its own.  The trace is subsampled for the
    coarse grid and the winner is refined on the full trace.
    """
    ts = torch.tensor(t, dtype=torch.float64)
    ss = torch.tensor(s, dtype=torch.float64)
    stride = max(1, len(t) // 900)
    tc, sc = ts[::stride], ss[::stride]

    def solve(tt, ss_, om, lam):
        env = torch.exp(-lam * (tt - tt[0]))
        A = torch.stack([torch.ones_like(tt), env * torch.cos(om * tt),
                         env * torch.sin(om * tt)], dim=1)
        coef, *_ = torch.linalg.lstsq(A, ss_[:, None])
        coef = coef[:, 0]
        res = A @ coef - ss_
        return coef, float(torch.sqrt((res * res).mean()))

    best = None
    for om in torch.linspace(0.6 * omega_guess, 1.6 * omega_guess, 161):
        for lam in torch.linspace(0.0, 12.0, 41):
            coef, rms = solve(tc, sc, float(om), float(lam))
            if best is None or rms < best[2]:
                best = (float(om), float(lam), rms)
    om0, lam0 = best[0], best[1]
    # refine on the full trace
    for _ in range(3):
        span_om = 0.02 * om0
        for om in torch.linspace(om0 - span_om, om0 + span_om, 81):
            coef, rms = solve(ts, ss, float(om), lam0)
            if rms < best[2]:
                best = (float(om), lam0, rms)
        om0 = best[0]
        for lam in torch.linspace(max(0.0, lam0 - 0.25 * max(lam0, 1.0)),
                                  lam0 + 0.25 * max(lam0, 1.0), 41):
            coef, rms = solve(ts, ss, om0, float(lam))
            if rms < best[2]:
                best = (om0, float(lam), rms)
        lam0 = best[1]
    coef, rms = solve(ts, ss, best[0], best[1])
    amp = float(torch.sqrt(coef[1] ** 2 + coef[2] ** 2))
    var = float(((ss - ss.mean()) ** 2).mean())
    return {"omega": best[0], "lambda_per_s": best[1], "rms": rms,
            "amplitude0": amp, "offset": float(coef[0]),
            "r2": 1.0 - rms ** 2 / var if var > 0 else float("nan")}


def peak_decay(t: list[float], s: list[float]) -> dict:
    """Amplitude decay rate from successive extrema of ``s(t)``."""
    n = len(s)
    mean = sum(s) / n
    y = [v - mean for v in s]
    peaks = []
    for i in range(1, n - 1):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0.0 and abs(y[i]) > 1e-12:
            peaks.append((t[i], abs(y[i])))
    if len(peaks) < 3:
        return {"n_peaks": len(peaks), "lambda_per_s": float("nan")}
    t0 = peaks[0][0]
    xs = torch.tensor([p[0] - t0 for p in peaks], dtype=torch.float64)
    ys = torch.log(torch.tensor([p[1] for p in peaks], dtype=torch.float64))
    A = torch.stack([torch.ones_like(xs), xs], dim=1)
    coef, *_ = torch.linalg.lstsq(A, ys[:, None])
    slope = float(coef[1, 0])
    pred = A @ coef[:, 0]
    res = pred - ys
    var = float(((ys - ys.mean()) ** 2).mean())
    return {"n_peaks": len(peaks), "lambda_per_s": -slope,
            "r2": 1.0 - float((res * res).mean()) / var if var > 0 else float("nan"),
            "peak_times_s": [float(p[0]) for p in peaks],
            "peak_amplitudes": [float(p[1]) for p in peaks]}


# --------------------------------------------------------------------------- #
# time integration
# --------------------------------------------------------------------------- #
def run_case(radius: float = 0.02, dx: float = 0.002, sigma: float = 7.2,
             rho0: float = 1000.0, sound_speed: float = 20.0,
             viscosity: float = 5e-5, eps: float = 0.10,
             periods: float = 3.2, scheme: str = "consistent",
             curvature_scheme: str = "fit_quadratic",
             density_diffusion: float = 0.0, steps: int | None = None,
             sample_every: int = 1) -> dict:
    """Release an elliptical drop and record the mode-2 oscillation."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme=scheme,
                 csf_curvature=curvature_scheme,
                 density_diffusion=density_diffusion)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    omega_ref = analytic_omega(radius, sigma, rho0)
    # ``sigma = 0`` is the negative control of the whole benchmark (it must NOT
    # oscillate at the capillary frequency), and it has no reference period to
    # scale from: the caller then has to pass an explicit ``steps``.
    if sigma > 0.0:
        period_ref = 2.0 * math.pi / omega_ref
        t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    elif steps is None:
        raise ValueError("sigma = 0 has no reference period: pass steps=...")
    else:
        period_ref = float("inf")
        t_sigma = float("inf")
    if steps is None:
        steps = int(math.ceil(periods * period_ref / dt))

    pos, mass = elliptical_drop(radius, dx, rho0, eps)
    vel = torch.zeros_like(pos)
    n0 = pos.shape[0]
    m0 = float(mass.sum())
    a0 = mode_amplitude(pos, mass)

    ts, s_mom, ke_hist = [], [], []
    vmax_hist, rms_hist, esc_hist, mom_hist = [], [], [], []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample_every == 0 or step == steps - 1:
            amp = mode_amplitude(pos, mass)
            ts.append((step + 1) * dt)
            s_mom.append(amp["s_moment"])
            # Kinetic energy is a kinematic observable independent of the
            # geometric estimator: KE ~ amplitude^2 * cos^2(omega t), i.e. it
            # oscillates at 2*omega.  It is the cross-check on the frequency.
            ke_hist.append(float(0.5 * (mass * vel.square().sum(dim=1)).sum()))
            v_sampled = float(torch.linalg.vector_norm(vel, dim=1).max())
            vmax_hist.append(v_sampled)
            cont = containment_diagnostics(pos, mass, radius)
            rms_hist.append(cont["rms_radius_over_radius"])
            esc_hist.append(cont["escaped_mass_fraction"])
            mom_hist.append(spurious_momentum(
                vel, mass, math.sqrt(sigma / (rho0 * radius)) if sigma > 0.0
                else 1.0))

    fit_mom = damped_cosine_fit(ts, s_mom, omega_ref)
    fit_ke = damped_cosine_fit(ts, ke_hist, 2.0 * omega_ref)
    zc_mom = zero_crossing_period(ts, s_mom)
    peaks = peak_decay(ts, s_mom)
    final = mode_amplitude(pos, mass)

    if omega_ref > 0.0:
        err_pct = (fit_mom["omega"] - omega_ref) / omega_ref * 100.0
        err_zc = ((zc_mom["omega"] - omega_ref) / omega_ref * 100.0
                  if math.isfinite(zc_mom["omega"]) else float("nan"))
    else:
        # sigma = 0 negative control: there is no target frequency to miss.
        err_pct = float("nan")
        err_zc = float("nan")
    # numerical (inviscid) dissipation floor implied by the fitted lambda
    return {
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
            "n_particles": n0, "sigma_N_per_m": sigma, "rho0": rho0,
            "sound_speed": sound_speed, "viscosity_m2_per_s": viscosity,
            "gravity": 0.0, "initial_eps": eps, "steps": steps,
            "physical_time_s": steps * dt,
            "periods_simulated": steps * dt / period_ref,
            "csf_scheme": scheme, "csf_curvature": curvature_scheme,
            "density_diffusion": density_diffusion,
            "lattice": "hexagonal", "mode": MODE,
        },
        "reference": {
            "law": "omega^2 = sigma m (m^2-1) / (rho R^3)  [2-D cylinder]",
            "omega_rad_per_s": omega_ref,
            "period_s": period_ref,
            "omega_3d_sphere_rad_per_s": analytic_omega_3d(radius, sigma, rho0),
            "capillary_time_s": t_sigma,
            "period_over_t_sigma": (period_ref / t_sigma
                                    if sigma > 0.0 else float("nan")),
            "damping_estimate_per_s": 2.0 * viscosity * MODE ** 2 / radius ** 2,
            "damping_estimate_provenance":
                "irrotational plane-capillary-wave estimate lambda = 2 nu k^2 "
                "with k = m/R; order-of-magnitude only (no closed-form 2-D "
                "viscous coefficient, Chandrasekhar 1961 ch. XII)",
        },
        "initial_state": a0,
        "final_state": final,
        "frequency": {
            "fit_second_moment_rad_per_s": fit_mom["omega"],
            "fit_second_moment_rel_error_pct": err_pct,
            "fit_kinetic_energy_rad_per_s": fit_ke["omega"],
            "fit_kinetic_energy_rel_error_pct":
                ((fit_ke["omega"] - 2.0 * omega_ref) / (2.0 * omega_ref) * 100.0
                 if omega_ref > 0.0 else float("nan")),
            "zero_crossing_rad_per_s": zc_mom["omega"],
            "zero_crossing_rel_error_pct": err_zc,
            # KE carries 2*omega; halves it so the two estimators are comparable.
            "estimator_agreement_pct":
                ((0.5 * fit_ke["omega"] - fit_mom["omega"])
                 / fit_mom["omega"] * 100.0
                 if fit_mom["omega"] else float("nan")),
            "within_3pct": abs(err_pct) < 3.0,
            "within_15pct": abs(err_pct) < GATE_PCT,
            "fit": fit_mom,
            "fit_kinetic_energy": fit_ke,
            "zero_crossing": zc_mom,
        },
        "damping": {
            "fit_lambda_per_s": fit_mom["lambda_per_s"],
            "peak_lambda_per_s": peaks["lambda_per_s"],
            "peak_fit_r2": peaks.get("r2", float("nan")),
            "ke_over_lambda": (fit_mom["lambda_per_s"] /
                               (2.0 * viscosity * MODE ** 2 / radius ** 2)
                               if viscosity > 0 else float("nan")),
            "peaks": peaks,
        },
        "diagnostics": {
            # Structurally zero: ``mass`` is never advected, only its (constant)
            # sum is compared with itself.  Kept for artifact compatibility and
            # flagged so no reader mistakes it for a conservation measurement.
            "mass_relative_drift": abs(float(mass.sum()) - m0) / m0,
            "mass_relative_drift_is_structural": True,
            "max_speed_m_per_s": max(vmax_hist),
            "capillary_velocity_m_per_s": (math.sqrt(sigma / (rho0 * radius))
                                           if sigma > 0.0 else float("nan")),
            "amplitude_after_sim_periods": abs(final["s_moment"]),
            "containment": {**containment_diagnostics(pos, mass, radius),
                            "gate_rms_radius_over_radius": RADIUS_GATE,
                            "gate_escaped_mass_fraction": ESCAPE_GATE,
                            "escape_factor": ESCAPE_FACTOR},
            "liveness": {
                **divergence_onset(ts, vmax_hist,
                                   SPEED_GATE_FACTOR * sound_speed),
                "speed_limit_m_per_s": SPEED_GATE_FACTOR * sound_speed,
                "kinetic_energy_max_J": max(ke_hist),
                "kinetic_energy_final_J": ke_hist[-1],
                "kinetic_energy_over_capillary_scale":
                    max(ke_hist) / (sigma * radius) if sigma > 0.0
                    else float("nan"),
            },
            "spurious_momentum": {
                "max_over_mv": max(mom_hist),
                "final_over_mv": mom_hist[-1],
            },
            "windowed_frequency": (windowed_frequency(ts, s_mom, omega_ref)
                                   if omega_ref > 0.0 else []),
            "trace": {"t": ts, "s_second_moment": s_mom,
                      "kinetic_energy_J": ke_hist,
                      "max_speed_m_per_s": vmax_hist,
                      "rms_radius_over_radius": rms_hist,
                      "escaped_mass_fraction": esc_hist,
                      "spurious_momentum_over_mv": mom_hist},
        },
    }


def strip_trace(case: dict) -> dict:
    """Report copy without the raw trace (keeps validation_report.json small)."""
    out = {k: v for k, v in case.items() if k != "diagnostics"}
    out["diagnostics"] = {k: v for k, v in case["diagnostics"].items()
                          if k != "trace"}
    return out


# --------------------------------------------------------------------------- #
# study definitions
# --------------------------------------------------------------------------- #
def study_frequency(**kw) -> dict:
    return {"frequency": run_case(**kw)}


def study_ladder(dx_list=(), **kw) -> dict:
    out = {}
    for dx in dx_list:
        case = run_case(dx=dx, **kw)
        key = f"dx_{dx * 1000:.1f}mm"
        (HERE / "convergence").mkdir(exist_ok=True)
        (HERE / "convergence" / f"frequency_{key}.json").write_text(
            json.dumps(strip_trace(case), indent=2))
        out[key] = case
    return {"ladder": out}


def study_damping(nu_list=(1e-4, 4e-4), **kw) -> dict:
    out = {}
    for nu in nu_list:
        case = run_case(viscosity=nu, **kw)
        key = f"nu_{nu:.1e}"
        (HERE / "convergence").mkdir(exist_ok=True)
        (HERE / "convergence" / f"damping_{key}.json").write_text(
            json.dumps(strip_trace(case), indent=2))
        out[key] = case
    return {"damping_series": out}


def study_quiet(nu_list=(5e-5,), dd_list=(0.1,), **kw) -> dict:
    """Zero-g quiet-drop noise floor, with and without the diffusive density term.

    ``docs/algorithm-gap-analysis.md`` (row P1) asks for exactly this case: a
    quiet free surface in zero g, to size the noise that the diffusive density
    term (Antuono et al. 2012) is supposed to remove and hence to decide
    whether it should be on by default.
    """
    out = {}
    for nu in nu_list:
        for dd in dd_list:
            case = run_case(viscosity=nu, density_diffusion=dd, **kw)
            key = f"nu_{nu:.1e}_dd_{dd:.2f}"
            (HERE / "convergence").mkdir(exist_ok=True)
            (HERE / "convergence" / f"quiet_{key}.json").write_text(
                json.dumps(strip_trace(case), indent=2))
            out[key] = case
    return {"quiet_case": out}


def summarise(study: dict) -> dict:
    """Verdict block shared by the JSON report and the doc generator."""
    rows = []
    if "frequency" in study:
        c = study["frequency"]
        rows.append({"case": "frequency_dx2.0mm",
                     "omega_measured": c["frequency"]["fit_second_moment_rad_per_s"],
                     "omega_reference": c["reference"]["omega_rad_per_s"],
                     "rel_error_pct": c["frequency"]["fit_second_moment_rel_error_pct"]})
    for block in ("ladder", "damping_series", "quiet_case"):
        for key, c in study.get(block, {}).items():
            rows.append({"case": f"{block}:{key}",
                         "omega_measured": c["frequency"]["fit_second_moment_rad_per_s"],
                         "omega_reference": c["reference"]["omega_rad_per_s"],
                         "rel_error_pct": c["frequency"]["fit_second_moment_rel_error_pct"],
                         "lambda_per_s": c["damping"]["fit_lambda_per_s"],
                         "lambda_estimate_per_s": c["reference"]["damping_estimate_per_s"]})
    errs = [abs(r["rel_error_pct"]) for r in rows]
    worst = max(errs) if errs else float("nan")
    return {"rows": rows, "worst_abs_rel_error_pct": worst,
            "worst_within_15pct": worst < GATE_PCT,
            "worst_within_3pct": worst < 3.0}


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot(study: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = {}
    if "frequency" in study:
        cases["dx = 2.0 mm"] = study["frequency"]
    for key, c in study.get("ladder", {}).items():
        cases[f"dx = {c['setup']['dx_m'] * 1000:.1f} mm"] = c
    for key, c in study.get("damping_series", {}).items():
        cases[f"nu = {c['setup']['viscosity_m2_per_s']:.1e}"] = c
    for key, c in study.get("quiet_case", {}).items():
        cases[f"dd = {c['setup']['density_diffusion']:.2f}"] = c
    if not cases:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=140)
    ax = axes[0]
    for i, (label, c) in enumerate(cases.items()):
        tr = c["diagnostics"]["trace"]
        ax.plot(tr["t"], tr["s_second_moment"], lw=1.0, alpha=0.85, label=label)
    c0 = next(iter(cases.values()))
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$s=(A-B)/(A+B)$  (mode-2 amplitude)")
    ax.set_title("Oscillating 2-D drop: second-moment amplitude")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ref = c0["reference"]["omega_rad_per_s"]
    for i, (label, c) in enumerate(cases.items()):
        w = c["frequency"]["fit_second_moment_rad_per_s"]
        ax.bar(i, w / ref, color="tab:blue", alpha=0.75)
        ax.text(i, w / ref + 0.01, f"{(w / ref - 1) * 100:+.1f}%",
                ha="center", fontsize=7)
    ax.axhline(1.0, color="k", lw=1.0)
    ax.axhspan(0.85, 1.15, color="tab:green", alpha=0.12,
               label="+/-15 % platform gate")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(list(cases), rotation=30, fontsize=7)
    ax.set_ylabel(r"$\omega_{\rm measured} / \sqrt{\sigma m(m^2-1)/\rho R^3}$")
    ax.set_title("Frequency vs Rayleigh-Lamb (2-D)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="all",
                    choices=("frequency", "ladder", "damping", "quiet", "all"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--periods", type=float, default=2.2)
    args = ap.parse_args(argv)

    common = {"eps": args.eps, "periods": args.periods}
    study: dict = {}
    if args.mode in ("frequency", "all"):
        study.update(study_frequency(**common))
    if args.mode in ("ladder", "all"):
        study.update(study_ladder(**common))
    if args.mode in ("damping", "all"):
        study.update(study_damping(**common))
    if args.mode in ("quiet", "all"):
        study.update(study_quiet(**common))
    study["verdict"] = summarise(study)

    out = {k: (strip_trace(v) if isinstance(v, dict) and "setup" in v else v)
           for k, v in study.items() if k != "verdict"}
    for block in ("ladder", "damping_series", "quiet_case"):
        if block in out:
            out[block] = {k: strip_trace(v) for k, v in out[block].items()}
    out["verdict"] = study["verdict"]
    (HERE / "validation_report.json").write_text(json.dumps(out, indent=2))
    (HERE / "trace_data.json").write_text(json.dumps(
        {k: (v["diagnostics"]["trace"] if "diagnostics" in v else None)
         for k, v in study.items() if isinstance(v, dict) and "diagnostics" in v},
        indent=2))
    if args.plot:
        plot(study, HERE / "oscillating_droplet.png")
    print(json.dumps(study["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
