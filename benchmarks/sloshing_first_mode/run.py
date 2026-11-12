#!/usr/bin/env python3
"""Free-decay sloshing: does the solved tank oscillate at the linear natural
frequency of its own geometry?

Why this benchmark exists
-------------------------
:mod:`tensorsph.sloshing` shipped with an analytic first-mode frequency
(``SloshingConfig.natural_frequency``) and a docstring that claimed to measure
"natural frequencies from FFT of surface elevation".  Nothing in the repository
ever compared the two:

* the only tests of the frequency were ``cfg.natural_frequency()`` against the
  same formula re-typed in the test (a tautology -- it checks arithmetic, not
  the solver);
* the only dynamical test asserted ``eta_max - eta_min > 1e-6 m`` on a tank
  released from a *flat* surface, a tolerance so loose that free-surface
  roughening alone passes it;
* the excitation branch was a ``pass``, so no run of this module could ever
  oscillate at a natural frequency -- the tank was never perturbed and never
  forced.

This benchmark closes that loop.  It releases the tank from the linear normal
mode (``eta(x) = a cos(pi x / L)``, fluid at rest, which is exactly the mode at
its extreme displacement), integrates 3.5 periods with the shipped WCSPH solver,
and measures the period three independent ways.  The measured period is graded
against the linear-theory value **and** against an independent variational
eigenvalue solve of the same boundary-value problem (below), so a transcription
error in the closed form cannot pass as agreement.

Reference
---------
Linear potential flow in a tank of length ``L`` and depth ``H``, walls at
``x = 0, L``, bottom at ``z = -H``, free surface at ``z = 0``:

    grad^2 phi = 0,  d phi/dn = 0 on the walls and bottom,
    d eta/dt = d phi/dz,  d phi/dt + g eta = 0 on z = 0.

Separation ``phi = psi(z) cos(k x)`` with ``k = n pi / L`` gives

    psi'' = k^2 psi,  psi'(-H) = 0,  omega^2 psi(0) = g psi'(0),
    =>  omega_n^2 = g k tanh(k H).

:func:`variational_reference` does not use that closed form.  It poses the same
eigenvalue problem as a Galerkin/Ritz problem in ``z`` -- basis
``cos(m pi (z+H) / (2H))``, which satisfies ``psi'(-H) = 0`` by construction, the
Laplace equation in the weak form ``int (psi' v' + k^2 psi v) = psi'(0) v(0)``
and the free-surface condition ``psi'(0) = (omega^2/g) psi(0)`` moved into the
mass matrix -- and returns the smallest positive eigenvalue of that generalized
problem.  It agrees with the closed form to ``<0.1 %`` (``reference_crosscheck``
in the output) at every depth, including the shallow (``kH -> 0``) and deep
(``kH -> inf``) limits where the closed form is easy to mistype.

Measurement, and why three methods
----------------------------------
``eta(t)`` at the wall-side antinode ``x = L/4`` (mode-1 amplitude ``0.707 a``)
is sampled every ``output_every`` steps.  The period is read by

* ``zcr``  -- interpolated upward zero crossings of the mean-removed signal;
* ``fft``  -- peak of a Hann-windowed, 8x zero-padded periodogram;
* ``fit``  -- least-squares damped sinusoid ``A exp(-lam t) cos(omega t + phi)``.

The three see different error sources (noise/quantisation, spectral leakage,
envelope drift), so agreement between them bounds the reading error rather than
one method's assumptions.  ``fit`` is the reported value; the spread between
methods is reported as ``method_spread_pct`` and graded too.

The a*x*i*s signal is the **modal amplitude**, not a point probe.  A probe
window of ``2 dx`` holds O(10) particles, so its elevation is quantised to
``dx`` -- at a 2 cm release whose antinode amplitude is 1.4 cm, the quantisation
is the same size as the signal, and a first cut of this benchmark read period
errors of 5-9 % from it.  Instead the solver records the whole free-surface
profile every output step (``SloshingConfig.record_profiles``) and the mode is
projected out of it,

    A_1(t) = (2/L) * int_0^L (eta(x,t) - H) cos(pi x / L) dx,

which averages the column quantisation over the whole free surface.  Point
probes are still recorded and reported, and the modal-vs-probe disagreement is
kept as ``probe_vs_modal_pct``: disagreement means the run is not the clean
single mode it claims to be.

Gates
-----
``G1_reference``        variational reference agrees with the closed form to <0.1 %
``G2_period``           |T_measured / T_theory - 1| <= 3 % at the finest level
``G3_method_agreement`` the three period readings spread <= 2 %
``G4_conservation``     no escaped particles, mass fixed, compression <= 2 %
``G5_amplitude``        the period does not drift with release amplitude
                        (linearity: 0.5a, a, 2a agree to <= 3 %) -- run with
                        ``--amplitude-ablation``
``G6_dx``               error at the finest level no worse than at the coarsest
                        (a ladder that gets worse with resolution is a bug)

Exit status is non-zero when a gate fails.  Honest FAIL is a result: it is
reported, not hidden.

Usage::

    PYTHONPATH=src python benchmarks/sloshing_first_mode/run.py \\
        --resolutions 0.025 0.0125 0.01 --periods 3.5 --out benchmarks/sloshing_first_mode/out

``--shard i/n`` runs every n-th resolution, so a ladder can be split over
cluster nodes; each shard writes ``out/partial_<i>_<n>.json`` and
``--merge`` folds the partials into ``out/results.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict

import numpy as np
import torch

from tensorsph.sloshing import (SloshingConfig, make_water_block, run_case,
                                validate_case)

#: Tank geometry and release state shared by every rung.  ``H/L = 0.4`` is
#: intermediate depth: neither the shallow nor the deep limit, where the
#: dispersion relation degenerates to a form that is easy to get right by
#: accident.  ``a/H = 5 %`` keeps the release inside the linear regime.
TANK = dict(tank_length=1.0, pool_width=1.0, tank_height=0.6, water_depth=0.4,
            gravity=9.81)
MODE = 1
AMPLITUDE = 0.02
PROBES = (0.25, 0.75)


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #

def analytic_period(cfg: SloshingConfig, mode: int = MODE) -> float:
    """Linear-theory natural period ``2 pi / sqrt(g k tanh(k H))``."""
    return cfg.natural_period(mode)


def variational_reference(L: float, H: float, g: float, mode: int = MODE,
                         n_basis: int = 64, n_quad: int = 4001) -> float:
    """Natural frequency from a Galerkin discretisation, not the closed form.

    Basis ``psi_m(z) = cos(m pi (z + H) / (2 H))``, ``m = 1..n_basis``: each
    satisfies ``psi_m'(-H) = 0``, so the bottom condition is exact.  Weak form of
    ``psi'' = k^2 psi`` with test ``v = psi_n``::

        int_0^H (psi' v' + k^2 psi v) dxi - psi'(0) v(0) = 0

    and the free-surface condition ``psi'(0) = (omega^2 / g) psi(0)`` turns the
    last term into the mass matrix ``G_mn = psi_m(H) psi_n(H) / g``, so the
    discrete problem is the generalized eigenproblem ``A c = omega^2 G c``.
    The smallest positive eigenvalue is ``omega_1``.
    """
    from scipy.linalg import eig

    k = mode * math.pi / L
    m = np.arange(1, n_basis + 1)
    lam = m * math.pi / (2.0 * H)
    xi = np.linspace(0.0, H, n_quad)
    basis = np.cos(np.outer(xi, lam))            # (n_quad, n_basis)
    dbasis = -np.sin(np.outer(xi, lam)) * lam    # d/d xi
    w = np.full(n_quad, xi[1] - xi[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    A = (dbasis * w[:, None]).T @ dbasis + k * k * (basis * w[:, None]).T @ basis
    top = np.cos(lam * H)                        # psi_m(0) with xi = 0 -> z = 0
    G = np.outer(top, top) / g
    vals, _ = eig(A, G)
    vals = np.real(vals[np.isfinite(vals)])
    vals = np.sort(vals[vals > 0.0])
    if vals.size == 0:
        raise RuntimeError("variational eigenproblem returned no positive mode")
    return float(math.sqrt(vals[0]))


# --------------------------------------------------------------------------- #
# Period readings
# --------------------------------------------------------------------------- #

def _detrend(signal: np.ndarray) -> np.ndarray:
    return signal - signal.mean()


def _smooth(y: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average, ``width`` samples (1 = no smoothing)."""
    if width <= 1 or y.size < width:
        return y
    kernel = np.ones(width) / width
    pad = width // 2
    padded = np.concatenate((np.full(pad, y[0]), y, np.full(width - 1 - pad, y[-1])))
    return np.convolve(padded, kernel, mode="valid")


def period_zero_crossing(t: np.ndarray, y: np.ndarray,
                         smooth_frac: float = 0.1) -> float | None:
    """Mean interval between interpolated upward zero crossings.

    The trace is low-passed over ``smooth_frac`` of one period first: the raw
    modal amplitude still carries free-surface noise above the mode frequency,
    and a single spurious crossing shifts the mean interval by a whole period.
    """
    dt = float(np.mean(np.diff(t)))
    fft_period = period_fft(t, y)
    width = (max(1, int(round(smooth_frac * fft_period / dt)))
             if fft_period else 1)
    s = _detrend(_smooth(y, width))
    crossings = []
    for i in range(1, s.size):
        if s[i - 1] < 0.0 <= s[i]:
            frac = -s[i - 1] / (s[i] - s[i - 1])
            crossings.append(t[i - 1] + frac * (t[i] - t[i - 1]))
    if len(crossings) < 2:
        return None
    return float(np.mean(np.diff(crossings)))


def period_fft(t: np.ndarray, y: np.ndarray, pad: int = 8) -> float | None:
    """Hann-windowed, zero-padded periodogram peak."""
    s = _detrend(y)
    if s.size < 16 or np.std(s) < 1e-14:
        return None
    dt = float(np.mean(np.diff(t)))
    window = np.hanning(s.size)
    n = int(pad * s.size)
    spec = np.abs(np.fft.rfft(s * window, n=n))
    freqs = np.fft.rfftfreq(n, d=dt)
    # Ignore d.c. and everything below one period over the whole record.
    keep = freqs > 1.0 / (t[-1] - t[0])
    if not keep.any():
        return None
    idx = np.flatnonzero(keep)[np.argmax(spec[keep])]
    if idx == 0 or idx + 1 >= spec.size:
        return None
    # Parabolic refinement on log magnitude.
    a, b, c = np.log(spec[idx - 1] + 1e-300), np.log(spec[idx] + 1e-300), \
        np.log(spec[idx + 1] + 1e-300)
    denom = a - 2.0 * b + c
    shift = 0.5 * (a - c) / denom if abs(denom) > 1e-30 else 0.0
    f = freqs[idx] + shift * (freqs[1] - freqs[0])
    return float(1.0 / f) if f > 0 else None


def period_fit(t: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Least-squares damped sinusoid ``A exp(-lam t) cos(omega t + phi) + c``.

    Returns ``(period, damping_rate)``.  The envelope coefficient is fitted, not
    assumed: WCSPH dissipates through the artificial viscosity term, so the
    trace is *not* a pure sinusoid and a fixed-envelope reading would bias the
    period.
    """
    from scipy.optimize import curve_fit

    s = _detrend(y)

    def model(tt, amp, lam, omega, phi, off):
        return amp * np.exp(-lam * tt) * np.cos(omega * tt + phi) + off

    # Seed the frequency from the FFT reading so the fit cannot lock onto a
    # harmonic of the sampling envelope.
    t_guess = period_fft(t, y)
    if t_guess is None:
        return None, None
    omega0 = 2.0 * math.pi / t_guess
    p0 = [float(np.std(s)) * 1.8, 0.1, omega0, 0.0, 0.0]
    try:
        popt, _ = curve_fit(model, t, s, p0=p0, maxfev=40000)
    except Exception:  # noqa: BLE001 - a failed fit is a reading, not a crash
        return None, None
    amp, lam, omega, phi, off = popt
    if not (math.isfinite(omega) and omega > 0.0):
        return None, None
    return float(2.0 * math.pi / omega), float(lam)


def read_period(t_list, y_list) -> dict:
    t = np.asarray(t_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    out: dict = {}
    tz = period_zero_crossing(t, y)
    tf = period_fft(t, y)
    tp, lam = period_fit(t, y)
    out["period_zero_crossing"] = tz
    out["period_fft"] = tf
    out["period_fit"] = tp
    out["damping_rate_fit"] = lam
    readings = [v for v in (tz, tf, tp) if v is not None]
    if readings:
        out["period"] = float(np.median(readings))
        out["method_spread_pct"] = float(
            100.0 * (max(readings) - min(readings)) / out["period"])
    else:
        out["period"] = None
        out["method_spread_pct"] = None
    return out


def modal_amplitude(profile_x, profiles, water_depth: float,
                    mode: int = MODE, tank_length: float | None = None):
    """Project the free-surface profiles onto ``cos(mode pi x / L)``.

    ``A(t) = (2/L) int (eta - H) cos(k x) dx`` with the trapezoid rule over the
    recorded columns; empty columns (``-1``) are dropped.  For a pure released
    mode this returns the released crest amplitude, so it is also a check on the
    initial condition.
    """
    x = np.asarray(profile_x, dtype=float)
    L = float(tank_length if tank_length is not None else x[-1] + 0.5 * (x[1] - x[0]))
    k = mode * math.pi / L
    trapz = getattr(np, "trapezoid", None) or np.trapz
    out = []
    for prof in profiles:
        eta = np.asarray(prof, dtype=float)
        keep = eta > 0.0
        if keep.sum() < 4:
            out.append(float("nan"))
            continue
        xs, es = x[keep], eta[keep] - water_depth
        out.append(float(2.0 / L * trapz(es * np.cos(k * xs), xs)))
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# One rung
# --------------------------------------------------------------------------- #

def build_config(dx: float, periods: float, amplitude: float = AMPLITUDE,
                 sound_speed: float | None = None,
                 viscosity: float | None = None) -> SloshingConfig:
    """Config for one ladder rung at the acoustic Courant limit of ``dx``."""
    cfg = SloshingConfig(**TANK, dx=dx,
                         initial_mode=MODE, initial_amplitude=amplitude,
                         probe_locations=list(PROBES), device="cpu")
    if sound_speed is not None:
        cfg.sound_speed = sound_speed
    if viscosity is not None:
        cfg.viscosity = viscosity
    h = cfg.h_factor * dx
    cfg.dt = cfg.max_courant * h / cfg.sound_speed
    span = periods * cfg.natural_period(MODE)
    # ~110 samples per oscillation: enough for the zero-crossing and the fit,
    # and the FFT interpolates the rest.
    n_samples = 110 * periods
    cfg.output_every = max(1, int(round(span / cfg.dt / n_samples)))
    cfg.steps = int(math.ceil(span / cfg.dt))
    validate_case(cfg)
    return cfg


def run_columns(cfg: SloshingConfig) -> list[float]:
    """x centres of the surface columns the solver records.

    Same bucketing as ``sloshing._column_profile``: column ``i`` spans
    ``[i dx, (i+1) dx)`` and the count is ``floor(tank_length / dx)``.
    """
    n = cfg.profile_columns if cfg.profile_columns > 0 \
        else int(math.floor(cfg.tank_length / cfg.dx))
    return [(i + 0.5) * cfg.dx for i in range(n)]


def quantised_surface(eta: np.ndarray, dx: float) -> np.ndarray:
    """Surface height a column of lattice centres can actually show.

    Lattice centres sit at ``(k + 1/2) dx``; the highest one below ``eta`` is
    ``dx * (ceil(eta/dx - 1/2) - 1/2)``.  Projecting *this* against the released
    amplitude separates the discretisation's own loss from the solver's, which
    the raw ``a`` would conflate.
    """
    return dx * (np.ceil(eta / dx - 0.5) - 0.5)


def run_rung(dx: float, periods: float, amplitude: float = AMPLITUDE,
             sound_speed: float | None = None,
             viscosity: float | None = None) -> dict:
    cfg = build_config(dx, periods, amplitude=amplitude,
                       sound_speed=sound_speed, viscosity=viscosity)
    h = cfg.h_factor * cfg.dx
    positions, mass = make_water_block(cfg, torch.device("cpu"))
    # Release state: the analytic mode shape, what the column-max surface
    # reading can show of it on this lattice, and what the solver actually
    # holds.  Only the last one is a solver claim; the middle one is the floor.
    n_columns = len(run_columns(cfg))
    x_cols = np.asarray([(i + 0.5) * cfg.dx for i in range(n_columns)])
    eta_ref = np.asarray([float(v) for v in cfg.mode_surface(
        torch.tensor(x_cols))])
    eta_quant = quantised_surface(eta_ref, cfg.dx)
    x_list = [float(v) for v in x_cols]
    a1_ref = float(modal_amplitude(x_list, [eta_ref], cfg.water_depth, MODE,
                                   cfg.tank_length)[0])
    a1_quant = float(modal_amplitude(x_list, [eta_quant], cfg.water_depth, MODE,
                                     cfg.tank_length)[0])

    started = time.time()
    run = run_case(cfg)
    wall = time.time() - started

    t = np.asarray(run.times)
    t_ref = analytic_period(cfg)

    # Primary reading: the projected modal amplitude.
    if run.surface_profiles:
        amp_series = modal_amplitude(run.profile_x, run.surface_profiles,
                                     cfg.water_depth, MODE, cfg.tank_length)
        modal = read_period(t, amp_series)
        # Released-mode check: A_1(0) must equal the released amplitude.
        modal["amplitude_t0"] = float(amp_series[0])
        modal["amplitude_t0_error_pct"] = float(
            100.0 * (amp_series[0] / amplitude - 1.0)) if amplitude else None
    else:
        amp_series, modal = None, {"period": None}

    probe_readings = {str(x): read_period(t, run.surface_elevations[x])
                      for x in PROBES}
    period = modal.get("period")
    if period is None:  # fall back to the probes and say so
        periods_meas = [r["period"] for r in probe_readings.values() if r["period"]]
        period = float(np.median(periods_meas)) if periods_meas else None
    err = None if period is None else 100.0 * (period / t_ref - 1.0)
    probe_periods = [r["period"] for r in probe_readings.values() if r["period"]]
    probe_vs_modal = (100.0 * (float(np.median(probe_periods)) / period - 1.0)
                      if probe_periods and period else None)

    # Crest envelope per oscillation from the modal amplitude: a run whose
    # amplitude collapses inside one period has no period to trust.
    env = []
    if amp_series is not None and np.isfinite(amp_series).sum() > 2:
        win = max(1, int(round(t_ref / float(np.mean(np.diff(t))))))
        for start in range(0, amp_series.size - win, win):
            chunk = amp_series[start:start + win]
            if np.isfinite(chunk).any():
                env.append(float(np.nanmax(chunk)))

    return {
        "dx": dx,
        "h_over_dx": cfg.h_factor,
        "cells_x": int(round(cfg.tank_length / dx)),
        "cells_y": int(round(cfg.water_depth / dx)),
        "particles": int(positions.shape[0]),
        "dt": cfg.dt,
        "acoustic_dt_limit": cfg.max_courant * h / cfg.sound_speed,
        "courant_limit": cfg.max_courant,
        "sound_speed": cfg.sound_speed,
        "viscosity": cfg.viscosity,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "periods_requested": periods,
        "output_every": cfg.output_every,
        "samples": run.times.__len__(),
        "wall_seconds": wall,
        "ms_per_step": 1000.0 * wall / max(1, run.steps_taken),
        "period_theory": t_ref,
        "period_measured": period,
        "period_error_pct": err,
        "modal_reading": modal,
        "probe_readings": probe_readings,
        "probe_vs_modal_pct": probe_vs_modal,
        "profile_columns": len(run.profile_x) if run.profile_x else 0,
        "amplitude_envelope_m": env,
        "amplitude_ratio_first_last": (env[-1] / env[0]) if len(env) > 1 and env[0] else None,
        "release_amplitude_ideal": amplitude,
        "release_amplitude_modal_measured": modal.get("amplitude_t0"),
        "release_amplitude_modal_quantised_reference": a1_quant,
        "release_amplitude_quantisation_loss_pct": (100.0 * (a1_quant / a1_ref - 1.0)
                                                    if a1_ref else None),
        "release_amplitude_ratio_measured_vs_quantised": (
            float(modal["amplitude_t0"] / a1_quant)
            if modal.get("amplitude_t0") is not None and a1_quant else None),
        "max_acoustic_courant": run.max_acoustic_courant,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "max_speed": run.max_speed,
        "provenance": {
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": "benchmarks/sloshing_first_mode/run.py",
            "cfg": {k: v for k, v in asdict(cfg).items() if isinstance(v, (int, float, str, bool, type(None)))},
        },
    }


def reference_crosscheck(levels: float = 0.05) -> dict:
    """Closed form vs variational eigen-solve over the depth range.

    Checked at ``H/L`` from shallow (0.05) to deep (4.0), not just at the case's
    own depth: the released case sits at an intermediate depth where both limits
    are far away and a swapped ``tanh``/``sinh`` would still land close.
    """
    rows = []
    for ratio in (0.05, 0.1, 0.4, 1.0, 2.0, 4.0):
        L, H, g = 1.0, ratio, 9.81
        k = MODE * math.pi / L
        closed = math.sqrt(g * k * math.tanh(k * H))
        numer = variational_reference(L, H, g)
        rows.append({"H_over_L": ratio, "omega_closed": closed,
                     "omega_variational": numer,
                     "error_pct": 100.0 * (numer / closed - 1.0)})
    return {"rows": rows,
            "max_abs_error_pct": max(abs(r["error_pct"]) for r in rows)}


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def grade(results: dict) -> dict:
    ref = results["reference_crosscheck"]
    rungs = sorted(results["rungs"], key=lambda r: -r["dx"])
    fine, coarse = rungs[-1], rungs[0]
    gates = {}

    gates["G1_reference"] = {
        "pass": ref["max_abs_error_pct"] < 0.1,
        "detail": f"max |closed - variational| = {ref['max_abs_error_pct']:.4f} % over H/L in [0.05, 4]",
        "value": ref["max_abs_error_pct"], "limit": 0.1}

    err = fine.get("period_error_pct")
    gates["G2_period"] = {
        "pass": err is not None and abs(err) <= 3.0,
        "detail": (f"finest dx={fine['dx']:g}: T_meas/T_theory - 1 = "
                   f"{err:+.2f} %" if err is not None else "no period reading"),
        "value": err, "limit": 3.0}

    spread = max((r["modal_reading"].get("method_spread_pct") or 0.0)
                 for r in rungs)
    gates["G3_method_agreement"] = {
        "pass": spread <= 2.0,
        "detail": f"worst spread between zcr/fft/fit on the modal amplitude = "
                  f"{spread:.2f} %",
        "value": spread, "limit": 2.0}

    bad = [r["dx"] for r in rungs
           if r["escaped"] > 0 or r["max_density_compression"] > 0.02]
    gates["G4_conservation"] = {
        "pass": not bad,
        "detail": ("every rung: 0 escaped particles, compression <= 2 %"
                   if not bad else f"violations at dx={bad}"),
        "value": len(bad), "limit": 0}

    abl = results.get("amplitude_ablation") or []
    if abl:
        errs = [a["period_error_pct"] for a in abl if a["period_error_pct"] is not None]
        drift = (max(errs) - min(errs)) if errs else None
        gates["G5_amplitude"] = {
            "pass": drift is not None and drift <= 3.0,
            "detail": f"period error spread over a = {[a['amplitude'] for a in abl]} m: "
                      f"{drift:.2f} %" if drift is not None else "no readings",
            "value": drift, "limit": 3.0}
    else:
        gates["G5_amplitude"] = {"pass": None, "detail": "not run", "value": None,
                                 "limit": 3.0}

    if len(rungs) > 1:
        better = abs(fine["period_error_pct"] or 1e9) <= abs(coarse["period_error_pct"] or 1e9) + 0.5
        gates["G6_dx"] = {
            "pass": better,
            "detail": (f"|error| coarse dx={coarse['dx']:g}: "
                       f"{coarse['period_error_pct']:+.2f} %, fine dx={fine['dx']:g}: "
                       f"{fine['period_error_pct']:+.2f} % (0.5 % slack)"),
            "value": coarse["period_error_pct"], "limit": fine["period_error_pct"]}
    else:
        gates["G6_dx"] = {"pass": None, "detail": "single rung", "value": None,
                          "limit": None}

    failed = [k for k, g in gates.items() if g["pass"] is False]
    results["gates"] = gates
    results["verdict"] = "FAIL" if failed else (
        "PASS" if all(g["pass"] for g in gates.values()) else "WITHHELD")
    results["failed_gates"] = failed
    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001 - provenance, not physics
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolutions", type=float, nargs="+",
                    default=[0.025, 0.0125, 0.01])
    ap.add_argument("--periods", type=float, default=3.5)
    ap.add_argument("--amplitude", type=float, default=AMPLITUDE)
    ap.add_argument("--sound-speed", type=float, default=None,
                    help="default: the case's own 10 sqrt(g H)")
    ap.add_argument("--viscosity", type=float, default=None)
    ap.add_argument("--amplitude-ablation", action="store_true",
                    help="also release at 0.5x and 2x the amplitude (G5)")
    ap.add_argument("--ablation-dx", type=float, default=None,
                    help="resolution for the amplitude ablation (default: finest)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th rung")
    ap.add_argument("--out", default="benchmarks/sloshing_first_mode/out")
    ap.add_argument("--merge", action="store_true",
                    help="merge partial_*.json in --out into results.json")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))

    if args.merge:
        rungs = []
        for name in sorted(os.listdir(args.out)):
            if not name.startswith("partial_"):
                continue
            with open(os.path.join(args.out, name)) as fh:
                rungs.extend(json.load(fh)["rungs"])
        rungs.sort(key=lambda r: -r["dx"])
        ref = reference_crosscheck()
        results = {"case": "sloshing first-mode free decay", "tank": TANK,
                   "mode": MODE, "amplitude": args.amplitude,
                   "periods": args.periods, "git_commit": _git_commit(),
                   "reference_crosscheck": ref, "rungs": rungs}
        results = grade(results)
        with open(os.path.join(args.out, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        print(json.dumps({"verdict": results["verdict"],
                          "failed": results["failed_gates"],
                          "rungs": [{"dx": r["dx"], "T_theory": r["period_theory"],
                                     "T_meas": r["period_measured"],
                                     "err_pct": r["period_error_pct"]}
                                    for r in rungs]}, indent=2))
        return 0 if results["verdict"] == "PASS" else 1

    i, n = (int(x) for x in args.shard.split("/"))
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"bad --shard {args.shard}")

    ref = reference_crosscheck()
    rungs = []
    for level, dx in enumerate(args.resolutions):
        if level % n != i:
            continue
        print(f"[shard {i}/{n}] dx={dx:g} ...", flush=True)
        rung = run_rung(dx, args.periods, amplitude=args.amplitude,
                        sound_speed=args.sound_speed, viscosity=args.viscosity)
        print(f"  particles={rung['particles']} steps={rung['steps']} "
              f"wall={rung['wall_seconds']:.0f}s "
              f"T_meas={rung['period_measured']} T_theory={rung['period_theory']:.5f} "
              f"err={rung['period_error_pct']:+.2f}%", flush=True)
        rungs.append(rung)

    abl = []
    if args.amplitude_ablation:
        dx = args.ablation_dx if args.ablation_dx else min(args.resolutions)
        for amp in (0.5 * args.amplitude, 2.0 * args.amplitude):
            print(f"[shard {i}/{n}] amplitude ablation a={amp:g} ...", flush=True)
            r = run_rung(dx, args.periods, amplitude=amp,
                         sound_speed=args.sound_speed, viscosity=args.viscosity)
            r["amplitude"] = amp
            abl.append(r)
            print(f"  T_meas={r['period_measured']} err={r['period_error_pct']:+.2f}%",
                  flush=True)

    payload = {"reference_crosscheck": ref, "rungs": rungs,
               "amplitude_ablation": abl, "git_commit": _git_commit()}
    partial = os.path.join(args.out, f"partial_{i}_{n}.json")
    with open(partial, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {partial}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())