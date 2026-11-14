#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH -- case-pool entry: plane Poiseuille flow, three forcings.

The case
--------
One geometry, one equation, three forcings.  Fluid fills ``0 <= y <= H``
between no-slip walls, ``x`` is periodic, and a uniform body force per unit mass
``g(t)`` (a pressure gradient) drives the channel::

    u_t = nu u_yy + g(t),        u(0,t) = u(H,t) = 0,        u(y,0) = 0

* **R1 constant** ``g = g0``  -> Boussinesq start-up towards the parabola
  ``u = g0 (H-y) y / (2 nu)`` (:func:`tensorsph.womersley.startup_velocity`).
* **R2 harmonic** ``g = g0 cos(omega t)`` -> the Womersley periodic state
  (:func:`tensorsph.womersley.complex_amplitude`);
  ``Wo = (H/2) sqrt(omega/nu)``, Stokes layer ``delta = sqrt(2 nu/omega)``.
  This is the regime that no other benchmark in the repository measures: the
  steady channel, the start-up and Stokes' second problem (wall-driven) all
  reproduce a *mean* balance, while R2 grades the **frequency response** --
  amplitude *and* phase lag of every row.

Reference and independence
--------------------------
The analytic references are re-derived in :mod:`tensorsph.womersley`
(``verify_symbolic`` proves the ODE residual and both no-slip conditions with
sympy; ``omega -> 0`` recovers the steady parabola).  A second, independent
reference is the Crank-Nicolson finite-difference solution of the same PDE with
the same forcing history (:mod:`benchmarks.case_poiseuille.ref`), which shares no
code with the SPH path; its agreement with the closed form is itself a gate
(G9), so a reference error cannot be hidden inside the physics error.

Numerical side
--------------
The production WCSPH solver (``tensorsph.droplet_impact.rhs``: Tait EOS,
density summation, Morris/Brookshaw viscosity, ``wall_friction = 1``) with the
mirror/ghost no-slip planes of :mod:`tensorsph.wall_mirror`.  The geometry
builder, config and row binning are the ones used by
``benchmarks/poiseuille_mirror`` (imported, not copied).  Row-mean ``u_x`` is
sampled uniformely in time during the measurement window and a harmonic fit at
``omega`` (:func:`tensorsph.stokes_second.harmonic_fit`) gives the measured
amplitude and phase of every row.

Predeclared gates (``DESIGN.md``, written before the final ladder)
------------------------------------------------------------------
G1 centre-line amplitude error (harmonic)                     <= 5 %
G2 amplitude RMS over rows 0.15 <= y/H <= 0.85                <= 5 %
G3 centre-line phase error                                    <= 5 % of pi/2
G4 phase RMS over the same rows                               <= 5 % of pi/2
G5 window stability: two consecutive periods agree            <= 5 %
G6 frequency discrimination: A(omega_fast)/A(omega) vs analytic<= 5 %
G7 anchor R1: steady-profile RMS error                        <= 5 %
G8 anchor R1: start-up profile RMS at t = 0.40 s              <= 5 %
G9 reference self-check: CN-FD vs analytic, amplitude+phase   <= 1 %

Usage
-----
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py --quick
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py --ladder \
        --out benchmarks/case_poiseuille/out

Exit status is non-zero when any gate fails.  ``out/result.json`` carries every
number quoted in the verification note.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.poiseuille.run import (  # noqa: E402
    bin_rows,
    build_fluid,
    make_config,
    periodic_images,
)
from benchmarks.case_poiseuille import ref as refmod  # noqa: E402
from tensorsph import womersley as wm  # noqa: E402
from tensorsph.droplet_impact import resolve_device, rhs  # noqa: E402
from tensorsph.stokes_second import harmonic_fit  # noqa: E402
from tensorsph.wall_mirror import plane_walls  # noqa: E402

CASE_KEY = wm.CASE_KEY
TAG = "CASE-POISEUILLE-SPH"

HEIGHT = 0.1
VISCOSITY = 0.01
RHO0 = 1000.0
BODY_FORCE = 9.81
OMEGA = 13.0            # Wo = 1.80, delta/H = 0.39
OMEGA_FAST = 31.4       # Wo = 2.80, delta/H = 0.25  (frequency discrimination)
COLUMNS = 8
SAMPLES_PER_PERIOD = 64

INTERIOR_LO, INTERIOR_HI = 0.15, 0.85
QUARTER_TURN = 0.5 * math.pi     # the phase scale the 5 % tolerance is relative to

GATES = {
    "G1_amp_centre_pct": 5.0,
    "G2_amp_profile_rms_pct": 5.0,
    "G3_phase_centre_pct_of_quarter_turn": 5.0,
    "G4_phase_profile_rms_pct_of_quarter_turn": 5.0,
    "G5_window_stability_pct": 5.0,
    "G6_frequency_ratio_pct": 5.0,
    "G7_steady_anchor_rms_pct": 5.0,
    "G8_startup_anchor_rms_pct": 5.0,
    "G9_reference_fd_vs_analytic_pct": 1.0,
}


# --------------------------------------------------------------------------- #
# Driving loop
# --------------------------------------------------------------------------- #

def make_case_config(dx: float, viscosity: float, device, friction: float = 1.0,
                     sound_speed: float = 20.0):
    """The Poiseuille-channel config, pinned to this case's domain."""
    base = make_config(dx, viscosity, friction, device)
    return dataclasses.replace(base, sound_speed=sound_speed,
                               pool_width=COLUMNS * dx, pool_depth=HEIGHT)


def drive(cfg, height: float, length: float, forcing, t_end: float,
          sample_times, device, samples_per_period: float | None = None,
          omega: float | None = None) -> dict:
    """Integrate the channel from rest and record row-mean profiles.

    ``forcing(t)`` returns the body force per unit mass along ``x``.  Profiles
    are recorded at every requested time; the measurement window is sampled at
    ``samples_per_period`` points per period so a harmonic fit at ``omega`` has
    an integer number of periods (orthogonal ``cos``/``sin``).
    """
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (height, -1)]

    dt = min(0.3 * h / cfg.sound_speed, 0.125 * h * h / cfg.viscosity)
    steps = int(round(t_end / dt))

    targets = sorted(float(s) for s in sample_times)
    recorded_t, recorded_u = [], []
    y_rows = None
    n_ghost = 0
    trace = []
    wall_clock = time.time()

    for step in range(steps + 1):
        t = step * dt
        while targets and t >= targets[0] - 0.5 * dt:
            _, profile = bin_rows(pos, vel, height, dx)
            ys = np.array([r["y"] for r in profile], dtype=float)
            ux = np.array([r["u_mean"] for r in profile], dtype=float)
            if y_rows is None:
                y_rows = ys
            recorded_t.append(t)
            recorded_u.append(ux)
            targets.pop(0)
        if step == steps:
            break

        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                              planes, support, slip="no_slip")
        n_ghost = int(ghost_pos.shape[0])
        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                        wall=ghost_pos if n_ghost else None,
                        wall_vel=ghost_vel if n_ghost else None)
        acc = acc[:n].clone()
        acc[:, 0] += forcing(t)

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            trace.append({"t": t, "u_centre": float(vel[:, 0].max()),
                          "u_mean": float(vel[:, 0].mean()),
                          "max_mach": float(vel.abs().max()) / cfg.sound_speed,
                          "wall_clock_s": time.time() - wall_clock})

    return {
        "n_fluid": int(n),
        "n_ghost": n_ghost,
        "dt": dt,
        "steps": steps,
        "sample_times": np.array(recorded_t),
        "u_samples": np.array(recorded_u),
        "y_rows": y_rows,
        "trace": trace,
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def harmonic_profile(case: dict, omega: float, height: float) -> list[dict]:
    """Per-row harmonic fit of the recorded series, against the closed form."""
    y = case["y_rows"]
    times = case["sample_times"]
    u = case["u_samples"]
    amp_ana = wm.amplitude(y, height, BODY_FORCE, VISCOSITY, omega)
    pha_ana = wm.phase(y, height, BODY_FORCE, VISCOSITY, omega)
    rows = []
    for k in range(y.size):
        fit = harmonic_fit(times, u[:, k], omega)
        rows.append({
            "row": k,
            "y": float(y[k]),
            "y_over_H": float(y[k] / height),
            "amplitude": fit["amplitude"],
            "amplitude_analytic": float(amp_ana[k]),
            "amplitude_rel_error": (float(fit["amplitude"]) - float(amp_ana[k]))
                                   / float(amp_ana[k]) if amp_ana[k] else float("nan"),
            "phase": fit["phase"],
            "phase_analytic": float(pha_ana[k]),
            "phase_error_deg": math.degrees(_wrap(fit["phase"] - float(pha_ana[k]))),
            "fit_residual": fit["residual"],
        })
    return rows


def _interior(rows: list[dict]) -> list[dict]:
    return [r for r in rows if INTERIOR_LO <= r["y_over_H"] <= INTERIOR_HI]


def _centre(rows: list[dict]) -> dict:
    return min(rows, key=lambda r: abs(r["y_over_H"] - 0.5))


def _rms(values) -> float:
    values = list(values)
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def evaluate_harmonic(case: dict, omega: float, height: float, resolution: int,
                      windows: int = 2) -> dict:
    """Gates G1..G5 for one harmonic run."""
    rows = harmonic_profile(case, omega, height)
    inner = _interior(rows)
    centre = _centre(rows)
    period = 2.0 * math.pi / omega

    # G5: fit each measurement period separately; a run that has not reached the
    # periodic state shows it here before any comparison with theory is made.
    times, u = case["sample_times"], case["u_samples"]
    k_c = int(np.argmin(np.abs(case["y_rows"] - centre["y"])))
    per_window = []
    for w in range(windows):
        sel = (times >= times[0] + w * period - 0.5 * case["dt"]) & \
              (times < times[0] + (w + 1) * period - 0.5 * case["dt"])
        if int(sel.sum()) < 8:
            continue
        fit = harmonic_fit(times[sel], u[sel, k_c], omega)
        per_window.append({"amplitude": fit["amplitude"], "phase": fit["phase"],
                           "n_samples": int(sel.sum())})
    amp_drift = float("nan")
    phase_drift = float("nan")
    if len(per_window) >= 2:
        a0, a1 = per_window[0]["amplitude"], per_window[1]["amplitude"]
        amp_drift = abs(a1 - a0) / abs(a0) if a0 else float("nan")
        phase_drift = abs(math.degrees(_wrap(per_window[1]["phase"] - per_window[0]["phase"])))

    phase_centre_pct = 100.0 * abs(math.radians(centre["phase_error_deg"])) / QUARTER_TURN
    phase_rms_pct = 100.0 * _rms(math.radians(r["phase_error_deg"]) for r in inner) / QUARTER_TURN

    out = {
        "resolution_H_over_dx": resolution,
        "omega": omega,
        "period": period,
        "womersley_number": wm.womersley_number(height, VISCOSITY, omega),
        "stokes_layer": wm.stokes_layer_thickness(VISCOSITY, omega),
        "stokes_layer_over_dx": (wm.stokes_layer_thickness(VISCOSITY, omega)
                                 / (height / resolution)),
        "dt": case["dt"],
        "steps": case["steps"],
        "n_fluid": case["n_fluid"],
        "n_ghost": case["n_ghost"],
        "n_samples": int(case["u_samples"].shape[0]),
        "max_mach": max(p["max_mach"] for p in case["trace"]),
        "wall_clock_s": case["wall_clock_s"],
        "amp_centre": centre["amplitude"],
        "amp_centre_analytic": centre["amplitude_analytic"],
        "amp_centre_error_pct": 100.0 * centre["amplitude_rel_error"],
        "amp_profile_rms_pct": 100.0 * _rms(r["amplitude_rel_error"] for r in inner),
        "amp_worst_row_pct": 100.0 * max(abs(r["amplitude_rel_error"]) for r in inner),
        "phase_centre_deg": centre["phase"],
        "phase_centre_analytic_deg": centre["phase_analytic"],
        "phase_centre_error_deg": centre["phase_error_deg"],
        "phase_centre_pct_of_quarter_turn": phase_centre_pct,
        "phase_profile_rms_pct_of_quarter_turn": phase_rms_pct,
        "phase_worst_row_deg": max(abs(r["phase_error_deg"]) for r in inner),
        "window_amplitude_drift_pct": 100.0 * amp_drift,
        "window_phase_drift_deg": phase_drift,
        "fit_residual_max": max(r["fit_residual"] for r in inner),
        "centre_series": {
            "t": times.tolist(),
            "u": u[:, k_c].tolist(),
            "y": float(case["y_rows"][k_c]),
        },
        "rows": rows,
        "windows": per_window,
    }
    stability = None
    if len(per_window) >= 2:
        stability = bool(100.0 * amp_drift <= 5.0
                         and phase_drift <= math.degrees(0.05 * QUARTER_TURN))
    out["gates"] = {
        "G1_amp_centre_5pct": abs(out["amp_centre_error_pct"]) <= 5.0,
        "G2_amp_profile_rms_5pct": out["amp_profile_rms_pct"] <= 5.0,
        "G3_phase_centre_5pct_of_quarter_turn": phase_centre_pct <= 5.0,
        "G4_phase_profile_rms_5pct_of_quarter_turn": phase_rms_pct <= 5.0,
        "G5_window_stability_5pct": stability,   # None = not graded (single window)
    }
    return out


def evaluate_constant(case: dict, height: float, resolution: int,
                      startup_times=(0.15, 0.40),
                      steady_from: float | None = None) -> dict:
    """Gates G7 (steady parabola) and G8 (start-up series) for the constant run."""
    y = case["y_rows"]
    times = case["sample_times"]
    u = case["u_samples"]

    if steady_from is None:
        steady_from = float(times.max()) - 0.20
    steady_sel = times >= steady_from - 0.5 * case["dt"]
    u_steady = u[steady_sel].mean(axis=0)
    u_parabola = wm.steady_velocity(y, height, BODY_FORCE, VISCOSITY)
    interior = (y / height >= INTERIOR_LO) & (y / height <= INTERIOR_HI)
    rel = (u_steady - u_parabola) / u_parabola
    steady_rms = 100.0 * _rms(rel[interior])

    startup = []
    for t_s in startup_times:
        k = int(np.argmin(np.abs(times - t_s)))
        u_ref = wm.startup_velocity(y, float(times[k]), height, BODY_FORCE,
                                    VISCOSITY)
        rel_s = (u[k] - u_ref) / u_ref
        startup.append({
            "t_requested": t_s,
            "t_actual": float(times[k]),
            "profile_rms_pct": 100.0 * _rms(rel_s[interior]),
            "profile_worst_pct": 100.0 * float(np.max(np.abs(rel_s[interior]))),
            "u_max_measured": float(u[k].max()),
            "u_max_series": float(u_ref.max()),
        })

    out = {
        "resolution_H_over_dx": resolution,
        "dt": case["dt"],
        "steps": case["steps"],
        "n_fluid": case["n_fluid"],
        "n_ghost": case["n_ghost"],
        "max_mach": max(p["max_mach"] for p in case["trace"]),
        "wall_clock_s": case["wall_clock_s"],
        "steady_from": steady_from,
        "steady_rows_used": int(steady_sel.sum()),
        "steady_profile_rms_pct": steady_rms,
        "steady_centre_pct": 100.0 * float(
            (u_steady[np.argmin(np.abs(y - height / 2))]
             - wm.steady_velocity([height / 2], height, BODY_FORCE, VISCOSITY)[0])
            / wm.steady_velocity([height / 2], height, BODY_FORCE, VISCOSITY)[0]),
        "steady_rows": [{"y": float(y[k]), "y_over_H": float(y[k] / height),
                         "u_measured": float(u_steady[k]),
                         "u_analytic": float(u_parabola[k]),
                         "rel_error": float(rel[k])} for k in range(y.size)],
        "startup": startup,
        "startup_rms_pct_at_040": next((s["profile_rms_pct"] for s in startup
                                        if abs(s["t_requested"] - 0.40) < 1e-9),
                                       float("nan")),
    }
    out["gates"] = {
        "G7_steady_anchor_5pct": steady_rms <= 5.0,
        "G8_startup_anchor_5pct": out["startup_rms_pct_at_040"] <= 5.0,
    }
    return out


def evaluate_reference(height: float, omega: float, y_rows, t_end: float,
                       sample_times) -> dict:
    """Gate G9: the independent CN-FD solution against the closed form."""
    sol = refmod.cn_solve(lambda t: BODY_FORCE * math.cos(omega * t), height,
                          VISCOSITY, t_end, sample_times, nx=201, n_steps=12000)
    rows = []
    for k, yk in enumerate(y_rows):
        sig = np.array([refmod.interpolate_to_rows(sol["y"], s, [yk])[0]
                        for s in sol["samples"]])
        fit = harmonic_fit(sol["times"], sig, omega)
        amp_ana = float(wm.amplitude([yk], height, BODY_FORCE, VISCOSITY, omega)[0])
        pha_ana = float(wm.phase([yk], height, BODY_FORCE, VISCOSITY, omega)[0])
        rows.append({
            "y": float(yk),
            "y_over_H": float(yk / height),
            "amplitude_fd": fit["amplitude"],
            "amplitude_analytic": amp_ana,
            "amplitude_rel_error": (fit["amplitude"] - amp_ana) / amp_ana,
            "phase_fd_deg": math.degrees(fit["phase"]),
            "phase_analytic_deg": math.degrees(pha_ana),
            "phase_error_deg": math.degrees(_wrap(fit["phase"] - pha_ana)),
        })
    inner = _interior(rows)
    amp_pct = 100.0 * _rms(r["amplitude_rel_error"] for r in inner)
    phase_pct = 100.0 * _rms(math.radians(r["phase_error_deg"]) for r in inner) / QUARTER_TURN
    return {
        "scheme": sol["scheme"],
        "nx": sol["nx"],
        "n_steps": sol["n_steps"],
        "dy": sol["dy"],
        "dt": sol["dt"],
        "amp_profile_rms_pct": amp_pct,
        "amp_worst_row_pct": 100.0 * max(abs(r["amplitude_rel_error"]) for r in inner),
        "phase_profile_rms_pct_of_quarter_turn": phase_pct,
        "phase_worst_row_deg": max(abs(r["phase_error_deg"]) for r in inner),
        "rows": rows,
        "gates": {"G9_reference_fd_vs_analytic_1pct": (amp_pct <= 1.0 and phase_pct <= 1.0)},
    }


# --------------------------------------------------------------------------- #
# Case driver
# --------------------------------------------------------------------------- #

def _window_times(t_start: float, period: float, periods: int = 2) -> list[float]:
    n = periods * SAMPLES_PER_PERIOD
    return [t_start + i * period / SAMPLES_PER_PERIOD for i in range(n)]


def run_harmonic(resolution: int, omega: float, device, settle_periods: int = 2,
                 window_periods: int = 2) -> dict:
    period = 2.0 * math.pi / omega
    dx = HEIGHT / resolution
    length = COLUMNS * dx
    cfg = make_case_config(dx, VISCOSITY, device)
    t_start = settle_periods * period
    times = _window_times(t_start, period, window_periods)
    t_end = times[-1]
    case = drive(cfg, HEIGHT, length, lambda t: BODY_FORCE * math.cos(omega * t),
                 t_end, times, device)
    res = evaluate_harmonic(case, omega, HEIGHT, resolution, windows=window_periods)
    res["length"] = length
    res["settle_periods"] = settle_periods
    return res


def run_constant(resolution: int, device, t_end: float = 1.0) -> dict:
    dx = HEIGHT / resolution
    length = COLUMNS * dx
    cfg = make_case_config(dx, VISCOSITY, device)
    startup = [0.15, 0.40]
    steady = [0.80 + 0.02 * i for i in range(11)]
    case = drive(cfg, HEIGHT, length, lambda t: BODY_FORCE, t_end,
                 startup + steady, device)
    res = evaluate_constant(case, HEIGHT, resolution)
    res["length"] = length
    return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "benchmarks/case_poiseuille/out")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true",
                        help="one coarse resolution, shorter window (smoke)")
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args(argv)

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.quick:
        args.resolutions = [8]

    settle, window = (1, 1) if args.quick else (2, 2)
    y_rows_ladder = {}

    harmonic = []
    for r in args.resolutions:
        print(f"[{TAG}] harmonic R2 H/dx={r} (Wo={wm.womersley_number(HEIGHT, VISCOSITY, OMEGA):.2f}) ...",
              flush=True)
        res = run_harmonic(r, OMEGA, device, settle, window)
        print(f"[{TAG}]   amp centre {res['amp_centre_error_pct']:+.2f}%  "
              f"profile {res['amp_profile_rms_pct']:.2f}%  "
              f"phase {res['phase_centre_error_deg']:+.2f} deg "
              f"({res['phase_centre_pct_of_quarter_turn']:.2f}% of pi/2)  "
              f"({res['wall_clock_s']:.0f}s)", flush=True)
        y_rows_ladder[r] = np.array([row["y"] for row in res["rows"]])
        harmonic.append(res)

    fast = None
    if not args.quick:
        print(f"[{TAG}] harmonic R2 H/dx=12 at omega={OMEGA_FAST} (frequency discrimination) ...",
              flush=True)
        fast = run_harmonic(12, OMEGA_FAST, device, settle, window)
        print(f"[{TAG}]   amp centre {fast['amp_centre_error_pct']:+.2f}%  "
              f"profile {fast['amp_profile_rms_pct']:.2f}%  "
              f"({fast['wall_clock_s']:.0f}s)", flush=True)

    const = None
    print(f"[{TAG}] constant R1 H/dx=12 (start-up + steady anchors) ...", flush=True)
    const = run_constant(12, device, t_end=1.0)
    print(f"[{TAG}]   steady {const['steady_profile_rms_pct']:.2f}%  "
          f"startup(t=0.40) {const['startup_rms_pct_at_040']:.2f}%  "
          f"({const['wall_clock_s']:.0f}s)", flush=True)

    # G6: the measured frequency response must follow the analytic ratio, which
    # a quasi-steady (omega-independent) solver cannot do.
    freq = None
    if fast is not None:
        base = next(r for r in harmonic if r["resolution_H_over_dx"] == 12)
        ampl_ratio = fast["amp_centre"] / base["amp_centre"]
        ampl_ratio_ana = fast["amp_centre_analytic"] / base["amp_centre_analytic"]
        phase_ratio = fast["phase_centre_deg"] - base["phase_centre_deg"]
        phase_ratio_ana = fast["phase_centre_analytic_deg"] - base["phase_centre_analytic_deg"]
        freq = {
            "omega_slow": OMEGA,
            "omega_fast": OMEGA_FAST,
            "amplitude_ratio_measured": ampl_ratio,
            "amplitude_ratio_analytic": ampl_ratio_ana,
            "amplitude_ratio_error_pct": 100.0 * (ampl_ratio - ampl_ratio_ana) / ampl_ratio_ana,
            "phase_difference_measured_deg": phase_ratio,
            "phase_difference_analytic_deg": phase_ratio_ana,
            "quasi_steady_ratio_would_be": 1.0,
            "gates": {"G6_frequency_ratio_5pct":
                      abs(100.0 * (ampl_ratio - ampl_ratio_ana) / ampl_ratio_ana) <= 5.0},
        }
        print(f"[{TAG}]   frequency ratio {ampl_ratio:.3f} vs analytic "
              f"{ampl_ratio_ana:.3f} ({freq['amplitude_ratio_error_pct']:+.2f}%)", flush=True)

    reference = None
    if not args.no_reference:
        print(f"[{TAG}] G9 independent CN-FD reference ...", flush=True)
        period = 2.0 * math.pi / OMEGA
        times = _window_times(settle * period, period, window)
        reference = evaluate_reference(HEIGHT, OMEGA, y_rows_ladder[args.resolutions[-1]],
                                       times[-1], times)
        print(f"[{TAG}]   FD vs analytic: amp {reference['amp_profile_rms_pct']:.3f}%  "
              f"phase {reference['phase_profile_rms_pct_of_quarter_turn']:.3f}% of pi/2",
              flush=True)

    gates = {}
    gates.update(harmonic[-1]["gates"])                  # finest resolution grades G1..G5
    if freq:
        gates.update(freq["gates"])
    gates.update(const["gates"])
    if reference:
        gates.update(reference["gates"])

    payload = {
        "case": CASE_KEY,
        "note": "算例池·泊肃叶流: constant force, start-up and Womersley harmonic regimes",
        "geometry": {"height": HEIGHT, "viscosity": VISCOSITY, "rho0": RHO0,
                     "body_force": BODY_FORCE, "columns": COLUMNS},
        "forcing_regimes": {
            "R1_constant": "g(t) = g0  ->  Boussinesq start-up / steady parabola",
            "R2_harmonic": "g(t) = g0 cos(omega t)  ->  Womersley periodic state",
        },
        "parameters": {"omega": OMEGA, "omega_fast": OMEGA_FAST,
                       "womersley_number": wm.womersley_number(HEIGHT, VISCOSITY, OMEGA),
                       "stokes_layer": wm.stokes_layer_thickness(VISCOSITY, OMEGA),
                       "settle_periods": settle, "window_periods": window,
                       "samples_per_period": SAMPLES_PER_PERIOD},
        "thresholds": GATES,
        "reference_provenance": {
            "symbolic": wm.verify_symbolic(),
            "numeric": wm.verify_numeric(HEIGHT, VISCOSITY, BODY_FORCE, OMEGA,
                                         2.0 * 2.0 * math.pi / OMEGA),
            "literature": ["Womersley 1955 J. Physiol. 127:553",
                           "Batchelor 1967 §4.2",
                           "Landau & Lifshitz Fluid Mechanics 2nd ed. §24"],
        },
        "harmonic_runs": harmonic,
        "harmonic_fast": fast,
        "frequency_discrimination": freq,
        "constant_run": const,
        "reference_check": reference,
        "gates": gates,
        "gates_graded": {k: v for k, v in gates.items() if v is not None},
        "gates_not_graded": [k for k, v in gates.items() if v is None],
        "all_gates_pass": bool(all(v for v in gates.values() if v is not None)),
        "device": str(device),
        "torch": torch.__version__,
        "wall_clock_s": (sum(r["wall_clock_s"] for r in harmonic)
                         + (fast["wall_clock_s"] if fast else 0.0)
                         + const["wall_clock_s"]),
        "generated_by": "benchmarks/case_poiseuille/run.py",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(payload, indent=2))
    print(f"[{TAG}] wrote {args.out / 'result.json'}")
    for name, ok in gates.items():
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{TAG}] {status} {name}")
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())