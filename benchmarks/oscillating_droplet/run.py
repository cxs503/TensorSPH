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


def analytic_omega(radius: float, sigma: float, rho0: float,
                   m: int = MODE) -> float:
    """``omega_m = sqrt(sigma m (m^2 - 1) / (rho R^3))`` (2-D, eq. 2)."""
    return math.sqrt(sigma * m * (m * m - 1.0) / (rho0 * radius ** 3))


def analytic_omega_3d(radius: float, sigma: float, rho0: float,
                      m: int = MODE) -> float:
    """3-D sphere law, kept only to document the 15.5 % trap in the docstring."""
    return math.sqrt(sigma * m * (m - 1.0) * (m + 2.0) / (rho0 * radius ** 3))


# --------------------------------------------------------------------------- #
# mode amplitude estimators
# --------------------------------------------------------------------------- #
def mode_amplitude(pos: torch.Tensor, mass: torch.Tensor,
                   reference_axis=None) -> dict:
    """Mode-2 amplitude from the mass-weighted second-moment tensor.

    A uniform ellipse has ``<x^2> = a^2/4``, ``<y^2> = b^2/4``, so the
    eigenvalues of the covariance give ``A = 2 sqrt(lambda_1)``,
    ``B = 2 sqrt(lambda_2)`` and ``s = (A - B)/(A + B) = eps`` to first order
    in ``eps`` for ``r = R (1 + eps cos 2 theta)``.  The second moment is an
    integral over the whole drop, so it is far less sensitive to the lattice
    jaggedness than an interface-radius estimator (see the note in
    ``docs/verification_oscillating_droplet.md``: the radius-hull Fourier
    coefficient of this very lattice returns 0.055-0.13 for ``eps = 0.10``).

    ``s_moment`` is a *magnitude*: ``A >= B`` by construction, so the sign of
    ``eps`` is lost and a released drop reads as ``|cos(w t)|`` -- rectified at
    ``2 w``.  ``s_signed = s_moment * cos(2 theta)`` restores it, with ``theta``
    the angle between the current major axis and ``reference_axis`` (the initial
    major axis): ``cos 2 theta = 2 (e_major . e_ref)^2 - 1`` is smooth, equals
    +1 when the ellipse is still aligned with the launch axis and -1 when it has
    turned through 90 degrees.  Without a reference axis the major axis is its
    own reference and ``s_signed == s_moment``.  See
    ``docs/verification_oscillating_droplet_fit.md`` for the measurement this
    repairs.
    """
    # ``mass`` is a broadcast scalar for a monodisperse lattice (as in
    # benchmarks/laplace_law); expand it so the estimator is dimension-safe.
    if mass.dim() == 0:
        mass = mass.expand(pos.shape[0])
    m_tot = mass.sum()
    com = (mass[:, None] * pos).sum(dim=0) / m_tot
    d = pos - com
    cov = (mass[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(dim=0) / m_tot
    lam, vecs = torch.linalg.eigh(cov)        # ascending
    lam1 = float(lam[-1].clamp_min(1e-30))
    lam2 = float(lam[0].clamp_min(1e-30))
    big, small = 2.0 * math.sqrt(lam1), 2.0 * math.sqrt(lam2)
    s_moment = (big - small) / (big + small)
    major = vecs[:, -1]
    cos2 = 1.0
    if reference_axis is not None:
        ref = torch.as_tensor(reference_axis, dtype=pos.dtype, device=pos.device)
        norm = float(torch.linalg.vector_norm(ref))
        if norm > 0.0:
            cos2 = float(2.0 * (float(major @ ref) / norm) ** 2 - 1.0)
    return {"s_moment": s_moment,
            "s_signed": s_moment * cos2,
            "cos2theta": cos2,
            # JSON-safe, so the report/manifest can carry it and the next call
            # can use it as reference_axis.
            "major_axis": [float(major[0]), float(major[1])],
            "semi_major": big, "semi_minor": small,
            "r_eff": 2.0 * math.sqrt(0.5 * (lam1 + lam2))}


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


def _damped_cosine_solve(tt: torch.Tensor, ss: torch.Tensor, om: float,
                         lam: float):
    """Closed-form ``(c, P, Q)`` for a fixed ``(omega, lambda)`` plus its rms.

    ``s = c + exp(-lam (t - t0)) (P cos(w t) + Q sin(w t))`` is linear in
    ``(c, P, Q)``, so only the two nonlinear parameters are searched.  The
    envelope is measured from the first sample, which keeps ``P`` and ``Q``
    O(amplitude) instead of exp(+lam t0) times it.
    """
    env = torch.exp(-lam * (tt - tt[0]))
    design = torch.stack((torch.ones_like(tt), env * torch.cos(om * tt),
                          env * torch.sin(om * tt)), dim=1)
    coef, *_ = torch.linalg.lstsq(design, ss[:, None])
    coef = coef[:, 0]
    residual = design @ coef - ss
    return coef, float(torch.sqrt((residual * residual).mean()))


def _refine_damped_cosine(ts: torch.Tensor, ss: torch.Tensor, om: float,
                          lam: float, iterations: int = 6):
    """Coordinate descent in ``(omega, lambda)`` with a bracket that shrinks.

    The bracket halves every ``1.6`` iterations, so the first sweep can move
    the seed by up to 2 % in ``omega`` and the last one resolves 1e-3 of it --
    the same policy regardless of which basin the seed came from, which is what
    makes the multi-start result independent of the coarse-grid ordering.
    """
    _, rms = _damped_cosine_solve(ts, ss, om, lam)
    for step in range(iterations):
        shrink = 1.6 ** step
        span = max(1e-9, 0.02 * om / shrink)
        for cand in torch.linspace(om - span, om + span, 41):
            _, r = _damped_cosine_solve(ts, ss, float(cand), lam)
            if r < rms:
                om, rms = float(cand), r
        span_lam = max(1e-6, 0.25 * max(lam, 1.0) / shrink)
        for cand in torch.linspace(max(0.0, lam - span_lam), lam + span_lam, 21):
            _, r = _damped_cosine_solve(ts, ss, om, float(cand))
            if r < rms:
                lam, rms = float(cand), r
    return om, lam, rms


#: Basins refined by :func:`damped_cosine_fit` by default.  ``1`` restores the
#: historical single-start behaviour and is only used to demonstrate the defect.
FIT_STARTS = 6


def damped_cosine_fit(t: list[float], s: list[float], omega_guess: float,
                      starts: int = FIT_STARTS) -> dict:
    """Least-squares fit of ``s = c + e^{-lam t}(P cos wt + Q sin wt)``.

    ``(omega, lambda)`` are searched on a coarse ``(omega, lambda)`` grid (the
    trace is subsampled for it) and then refined on the full trace; the three
    linear coefficients are solved in closed form at every grid point.

    Why the search is multi-start
    -----------------------------
    A release trace covers only a couple of periods, so the rms landscape has
    several near-equal basins and the historical single-start search reported
    whichever basin the *subsampled* coarse grid happened to win.  A round-off
    level difference between BLAS implementations can reorder two near-equal
    coarse minima, after which the local refinement (which only descends) locks
    onto a different basin: identical code then measured ``+8.64 %`` on x86_64
    CI and ``-31.78 %`` on loong64.  That spread is an instrument defect, not
    physics, and it is diagnosed in ``fit_robustness.py``.

    Here the coarse grid is clustered into basins separated by at least one
    Rayleigh bin ``2 pi / T`` (the finest frequency difference the recorded
    window can resolve), up to ``starts`` basins are refined, and the lowest
    rms wins.  Since every competitive basin is refined, the winner no longer
    depends on the coarse-grid ordering.

    The returned ``ambiguity_rms_gap`` is the honest precision statement: it is
    the relative rms advantage of the winner over the best *other* basin.  A
    gap near zero means the data do not determine the frequency, whatever the
    point estimate says; callers should then quote an uncertainty of one
    Rayleigh bin instead of the fitted value to more digits.
    """
    if starts < 1:
        raise ValueError("starts must be >= 1")
    ts = torch.tensor(t, dtype=torch.float64)
    ss = torch.tensor(s, dtype=torch.float64)
    if ts.numel() < 8:
        raise ValueError("need at least 8 samples to fit a damped cosine")
    stride = max(1, ts.numel() // 900)
    tc, sc = ts[::stride], ss[::stride]

    window = float(ts[-1] - ts[0])
    rayleigh = 2.0 * math.pi / window if window > 0.0 else float("inf")
    separation = max(rayleigh, 0.02 * omega_guess)

    coarse = []
    for om in torch.linspace(0.6 * omega_guess, 1.6 * omega_guess, 121):
        for lam in torch.linspace(0.0, 12.0, 25):
            _, rms = _damped_cosine_solve(tc, sc, float(om), float(lam))
            coarse.append((rms, float(om), float(lam)))
    coarse.sort(key=lambda row: row[0])
    seeds: list[tuple[float, float]] = []
    for _, om, lam in coarse:
        if all(abs(om - seed_om) > separation for seed_om, _ in seeds):
            seeds.append((om, lam))
        if len(seeds) >= starts:
            break

    iterations = 6 if starts > 1 else 3
    refined = []
    for om, lam in seeds:
        r_om, r_lam, r_rms = _refine_damped_cosine(ts, ss, om, lam, iterations)
        refined.append((r_rms, r_om, r_lam))
    refined.sort(key=lambda row: row[0])
    rms_best, om_best, lam_best = refined[0]
    runner = next((row for row in refined
                   if abs(row[1] - om_best) > separation), None)
    ambiguity = ((runner[0] - rms_best) / rms_best
                 if runner is not None and rms_best > 0.0 else float("inf"))

    coef, rms = _damped_cosine_solve(ts, ss, om_best, lam_best)
    amp = float(torch.sqrt(coef[1] ** 2 + coef[2] ** 2))
    var = float(((ss - ss.mean()) ** 2).mean())
    return {"omega": om_best, "lambda_per_s": lam_best, "rms": rms,
            "amplitude0": amp, "offset": float(coef[0]),
            "r2": 1.0 - rms ** 2 / var if var > 0 else float("nan"),
            "starts": len(refined), "rayleigh_rad_per_s": rayleigh,
            "ambiguity_rms_gap": ambiguity,
            "omega_alternatives": [
                {"omega": row[1], "lambda_per_s": row[2], "rms": row[0]}
                for row in refined[1:]]}


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
    period_ref = 2.0 * math.pi / omega_ref
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    if steps is None:
        steps = int(math.ceil(periods * period_ref / dt))

    pos, mass = elliptical_drop(radius, dx, rho0, eps)
    vel = torch.zeros_like(pos)
    n0 = pos.shape[0]
    m0 = float(mass.sum())
    a0 = mode_amplitude(pos, mass)
    reference_axis = a0["major_axis"]

    ts, s_mom, s_sig, ke_hist = [], [], [], []
    vmax_hist = []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample_every == 0 or step == steps - 1:
            amp = mode_amplitude(pos, mass, reference_axis=reference_axis)
            ts.append((step + 1) * dt)
            s_mom.append(amp["s_moment"])
            s_sig.append(amp["s_signed"])
            # Kinetic energy is a kinematic observable independent of the
            # geometric estimator: KE ~ amplitude^2 * cos^2(omega t), i.e. it
            # oscillates at 2*omega.  It is the cross-check on the frequency.
            ke_hist.append(float(0.5 * (mass * vel.square().sum(dim=1)).sum()))
            vmax_hist.append(float(torch.linalg.vector_norm(vel, dim=1).max()))

    fit_mom = damped_cosine_fit(ts, s_mom, omega_ref)
    fit_sig = damped_cosine_fit(ts, s_sig, omega_ref)
    fit_ke = damped_cosine_fit(ts, ke_hist, 2.0 * omega_ref)
    zc_mom = zero_crossing_period(ts, s_mom)
    zc_sig = zero_crossing_period(ts, s_sig)
    peaks = peak_decay(ts, s_mom)
    final = mode_amplitude(pos, mass)

    err_pct = (fit_mom["omega"] - omega_ref) / omega_ref * 100.0
    err_sig = (fit_sig["omega"] - omega_ref) / omega_ref * 100.0
    err_zc = ((zc_mom["omega"] - omega_ref) / omega_ref * 100.0
              if math.isfinite(zc_mom["omega"]) else float("nan"))
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
            "period_over_t_sigma": period_ref / t_sigma,
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
                (fit_ke["omega"] - 2.0 * omega_ref) / (2.0 * omega_ref) * 100.0,
            "zero_crossing_rad_per_s": zc_mom["omega"],
            "zero_crossing_rel_error_pct": err_zc,
            # KE carries 2*omega; halves it so the two estimators are comparable.
            "estimator_agreement_pct":
                (0.5 * fit_ke["omega"] - fit_mom["omega"])
                / fit_mom["omega"] * 100.0,
            "within_3pct": abs(err_pct) < 3.0,
            "within_15pct": abs(err_pct) < GATE_PCT,
            "fit": fit_mom,
            # Signed-observable path: the same estimator on the rectification-
            # free trace.  ``fit_second_moment_*`` above is kept verbatim as
            # the historical reading, so the two can be compared directly.
            "fit_signed_rad_per_s": fit_sig["omega"],
            "fit_signed_rel_error_pct": err_sig,
            "fit_signed": fit_sig,
            "zero_crossing_signed_rad_per_s": zc_sig["omega"],
            "zero_crossing_signed_rel_error_pct":
                ((zc_sig["omega"] - omega_ref) / omega_ref * 100.0
                 if math.isfinite(zc_sig["omega"]) else float("nan")),
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
            "mass_relative_drift": abs(float(mass.sum()) - m0) / m0,
            "max_speed_m_per_s": max(vmax_hist),
            "capillary_velocity_m_per_s": math.sqrt(sigma / (rho0 * radius)),
            "amplitude_after_sim_periods": abs(final["s_moment"]),
            "trace": {"t": ts, "s_second_moment": s_mom,
                      "s_signed": s_sig, "kinetic_energy_J": ke_hist},
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
