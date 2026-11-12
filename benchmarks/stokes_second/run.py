#!/usr/bin/env python3
"""Stokes' second problem: oscillating flat plate, 2-D SPH (case-pool anchor).

Physics and reference
---------------------
An incompressible fluid fills ``0 <= y <= H``; the plate at ``y = 0``
oscillates in its own plane at ``u(0,t) = U0 cos(omega t)`` and
``x`` is periodic.  In a half space the exact solution is Landau & Lifshitz
(*Fluid Mechanics*, 2nd English ed., §24)

    u(y,t) = U0 exp(-y/delta) cos(omega t - y/delta),  delta = sqrt(2 nu / omega)

so the **amplitude** decays as ``A(y) = U0 exp(-y/delta)``.  The case is graded
on that decay: the wave is solenoidal, so no pressure work is involved and the
only physics under test is the viscous operator plus the *moving* no-slip wall.

The reference is re-derived, not transcribed: :func:`tensorsph.stokes_second.
verify_symbolic` proves with sympy that ``u_t = nu u_yy`` identically, that
``u(0,t) = U0 cos(omega t)``, that ``A(delta)/U0 = 1/e`` and that
``delta = sqrt(2 nu/omega)``; the artifact carries those residuals.

Two references are reported, because they answer different questions:

* ``half_space``   ``U0 exp(-y/delta)`` — the case-pool referent.  A numerical
  channel of depth ``H`` cannot be a half space, so its exact solution differs
  by ``O(exp(-(H-y)/delta))``; that truncation is part of the error budget and
  is quantified, not hidden.
* ``finite_depth`` ``Re{U0 sinh(m(H-y))/sinh(mH) e^{i omega t}}``, ``m = (1+i)/delta``
  — the exact solution *of the problem actually run*.  Numerics minus this is
  pure discretisation error.

Wall model
----------
The moving no-slip wall is the mirror (ghost) construction of
:mod:`tensorsph.wall_mirror` driven with ``plane_vel = (U0 cos omega t, 0)``:
the ghost carries ``2w - v``, so the fluid/ghost viscous pair has mean ``w`` and
the no-slip plane moves *on* ``y = 0``.  The solver itself is the production
WCSPH ``rhs`` (Tait EOS, density summation, Morris/Brookshaw viscosity,
``wall_friction = 1``); nothing is calibrated against the run — ``U0``, ``nu``,
``omega``, ``H`` and ``dx`` are inputs and the run starts from rest.

Predeclared gates (``benchmarks/stokes_second/DESIGN.md``, written first)
-------------------------------------------------------------------------
G1 amplitude L2 error over the Stokes layer vs ``U0 exp(-y/delta)``  <= 5 %
G2 worst single row in the same set                                <= 10 %
G3 decay-rate error, fitted ``-dlnA/dy`` vs ``1/delta``              <= 5 %
G4 finite-depth (discretisation-only) L2 error                      <= 5 %
G5 negative control: free-slip oscillating wall drives < 5 % of U0
G6 resolution: G1 does not grow from the coarsest to the finest mesh

Usage
-----
    PYTHONPATH=.:src python benchmarks/stokes_second/run.py --quick          # smoke
    PYTHONPATH=.:src python benchmarks/stokes_second/run.py --ladder         # G1..G6
    PYTHONPATH=.:src python benchmarks/stokes_second/run.py --sweep          # + sensitivity
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.poiseuille.run import (  # noqa: E402
    build_fluid,
    make_config,
    periodic_images,
)
from tensorsph.droplet_impact import resolve_device, rhs  # noqa: E402
from tensorsph.stokes_second import (  # noqa: E402
    finite_depth,
    finite_depth_amplitude,
    harmonic_fit,
    penetration_depth,
    semi_infinite_amplitude,
    verify_numeric,
    verify_symbolic,
)
from tensorsph.wall_mirror import plane_walls  # noqa: E402

OUT = REPO / "benchmarks" / "stokes_second" / "out"


def _jsonable(obj):
    """numpy scalars/bools -> plain JSON types (artifacts must be pure JSON)."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")

#: Case constants.  ``H = 6 delta`` and ``delta/dx`` on the ladder; nothing here
#: is fitted to a measured profile.
NU = 0.01                 # m^2/s
OMEGA = 2.0 * math.pi     # rad/s   -> T = 1 s, delta = 5.64 cm
U0 = 0.01                 # m/s     -> Ma = 1e-3 at c0 = 10 m/s
H_OVER_DELTA = 6.0
COLUMNS = 6               # fluid columns in x; x is periodic and invariant
RHO0 = 1000.0
SOUND_SPEED = 10.0
GAMMA = 7.0
H_FACTOR = 1.3
FRICTION = 1.0
LADDER = (8, 12, 18)      # delta/dx
PERIODS = 6.0             # run length in oscillation periods
MEASURE_PERIODS = 2.0     # harmonic fit window = last 2 periods
SETTLE_PERIODS = 4.0      # rest -> periodic transient decays ~ exp(-t/1.16 T)
SAMPLES_PER_PERIOD = 160
GATES = {"G1_amplitude_l2_pct": 5.0, "G2_worst_row_pct": 10.0,
         "G3_decay_rate_pct": 5.0, "G4_finite_depth_l2_pct": 5.0,
         "G5_control_amp_pct": 5.0}


# --------------------------------------------------------------------------- #
# Driving loop
# --------------------------------------------------------------------------- #

def make_case_config(dx: float, height: float, viscosity: float,
                     sound_speed: float, h_factor: float, friction: float,
                     device):
    """The Poiseuille channel config with this case's sound speed and support."""
    base = make_config(dx, viscosity, friction, device)
    return dataclasses.replace(
        base, sound_speed=sound_speed, h_factor=h_factor, gravity=0.0,
        wall_friction=friction, pool_width=COLUMNS * dx, pool_depth=height)


def drive(cfg, height: float, length: float, viscosity: float, omega: float,
          u0: float, t_end: float, t_measure: float, samples_per_period: int,
          device, slip: str = "no_slip", h_oscillate: bool = True,
          label: str = "") -> dict:
    """Integrate the oscillating-plate channel and fit the amplitude per row.

    The plate velocity ``U0 cos(omega t)`` is prescribed exactly (it is the
    boundary condition, not a fit).  Row-mean ``u_x`` is sampled during the
    measurement window; a harmonic fit at ``omega`` gives the amplitude and
    phase of every row, which is what the gates grade.
    """
    dx = cfg.dx
    delta = penetration_depth(viscosity, omega)
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (height, -1)]

    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    period = 2.0 * math.pi / omega
    sample_every = max(1, int(round(period / samples_per_period / dt)))
    measure_from = t_end - t_measure

    # Row index of each particle (the flow is x-invariant, y does not drift)
    row_of = torch.round((pos[:, 1] - 0.5 * dx) / dx).to(torch.long)
    n_rows = int(round(height / dx))

    sample_t: list[float] = []
    sample_u: list[np.ndarray] = []
    trace: list[dict] = []
    wall_clock = time.time()
    n_ghost = 0
    n_ghost_max = 0

    def plate_vel(t: float):
        if not h_oscillate:
            return (0.0, 0.0)
        return (u0 * math.cos(omega * t), 0.0)

    for step in range(steps):
        t = step * dt
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)

        w = plate_vel(t)
        plane_list = [(planes[0][0], planes[0][1], w), planes[1]]
        ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                             plane_list, support, slip=slip)
        n_ghost = int(ghost_pos.shape[0])
        n_ghost_max = max(n_ghost_max, n_ghost)

        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                        wall=ghost_pos if n_ghost else None,
                        wall_vel=ghost_vel if n_ghost else None)
        acc = acc[:n].clone()

        if t >= measure_from:
            idx = row_of.clamp(0, n_rows - 1).numpy()
            sums = np.bincount(idx, weights=vel[:, 0].double().numpy(),
                               minlength=n_rows)
            counts = np.bincount(idx, minlength=n_rows)
            row_mean = np.divide(sums, counts,
                                 out=np.full(n_rows, np.nan),
                                 where=counts > 0)
            sample_t.append(t)
            sample_u.append(row_mean)

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % max(1, steps // 24) == 0 or step == steps - 1:
            trace.append({
                "t": t,
                "phase": float((t / period) % 1.0),
                "u_max": float(vel[:, 0].max()),
                "u_mean": float(vel[:, 0].mean()),
                "max_mach": float(vel.abs().max()) / cfg.sound_speed,
                "wall_clock_s": time.time() - wall_clock,
            })

    times = np.asarray(sample_t)
    u_mat = np.asarray(sample_u)            # (n_samples, n_rows)
    rows = []
    for k in range(n_rows):
        signal = u_mat[:, k]
        finite = np.isfinite(signal)
        if finite.sum() < 8:
            continue
        fit = harmonic_fit(times[finite], signal[finite], omega)
        y = 0.5 * dx + k * dx
        rows.append({
            "row": k,
            "y": float(y),
            "y_over_delta": float(y / delta),
            "amplitude": fit["amplitude"],
            "amplitude_over_U0": fit["amplitude"] / u0 if u0 else float("nan"),
            "phase": fit["phase"],
            "mean_over_U0": fit["mean"] / u0 if u0 else float("nan"),
            "fit_residual": fit["residual"],
            "n_samples": fit["n_samples"],
        })

    # Residual start-up transient: the same harmonic fit applied to each of the
    # last three single-period windows.  A converged periodic state gives
    # identical amplitudes; anything left of the rest-to-oscillation transient
    # shows up as a trend, which is what the drift number reports.
    period = 2.0 * math.pi / omega
    windows = []
    for back in (3, 2, 1):
        lo = times[-1] - back * period
        hi = lo + period
        sel = (times >= lo) & (times < hi)
        if sel.sum() < 8:
            continue
        amps = {}
        for target in (0.5 * delta, 1.0 * delta):
            k = int(min(range(n_rows), key=lambda i: abs(0.5 * dx + i * dx - target)))
            amps[f"y_over_delta_{target / delta:.1f}"] = harmonic_fit(
                times[sel], u_mat[sel, k], omega)["amplitude"]
        windows.append({"t_start": float(lo), "t_end": float(hi), "amps": amps})
    drift = None
    if len(windows) >= 2:
        a, b = windows[-2]["amps"], windows[-1]["amps"]
        drift = {k: (b[k] - a[k]) / a[k] for k in a if a[k]}

    # space-time field of the fitted measurement window (row-mean u_x)
    field = {
        "t": [float(x) for x in times.tolist()],
        "y": [float(0.5 * dx + k * dx) for k in range(n_rows)],
        "u": [[float(v) if np.isfinite(v) else None for v in row]
              for row in u_mat.T.tolist()],
    }

    # instantaneous velocity profile at the phase where the plate is at peak
    last_period = times[times >= times[-1] - period]
    return {
        "label": label,
        "slip": slip,
        "h_oscillate": h_oscillate,
        "dx": dx,
        "h": h,
        "delta": delta,
        "delta_over_dx": delta / dx,
        "height": height,
        "height_over_delta": height / delta,
        "length": length,
        "viscosity": viscosity,
        "omega": omega,
        "period": period,
        "u0": u0,
        "sound_speed": cfg.sound_speed,
        "h_factor": cfg.h_factor,
        "wall_friction": cfg.wall_friction,
        "rho0": cfg.rho0,
        "mach_number": u0 / cfg.sound_speed,
        "stokes_reynolds": u0 * delta / viscosity,
        "n_fluid": int(n),
        "n_ghost_typical": n_ghost,
        "n_ghost_max": n_ghost_max,
        "dt": dt,
        "steps": steps,
        "steps_per_period": period / dt,
        "sample_every": sample_every,
        "n_samples": int(times.size),
        "t_end": t_end,
        "measure_from": measure_from,
        "rows": rows,
        "field": field,
        "trace": trace,
        "wall_clock_s": time.time() - wall_clock,
        "window_amplitudes": windows,
        "window_drift": drift,
        "sample_window_last_period_end": float(last_period[-1])
        if last_period.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Metrics and gates
# --------------------------------------------------------------------------- #

def evaluate(case: dict, height: float, viscosity: float, omega: float,
             u0: float) -> dict:
    """Row-by-row comparison with both references plus the gates' ingredients."""
    delta = penetration_depth(viscosity, omega)
    rows = case["rows"]
    out_rows = []
    for r in rows:
        y = r["y"]
        ref_half = semi_infinite_amplitude(y, u0, viscosity, omega)
        ref_fd = float(finite_depth_amplitude(np.array([y]), height, u0,
                                              viscosity, omega)[0])
        out_rows.append({
            **r,
            "amplitude_half_space": float(ref_half),
            "amplitude_finite_depth": ref_fd,
            "rel_err_half_space": (r["amplitude"] - ref_half) / ref_half
            if ref_half > 0 else float("nan"),
            "rel_err_finite_depth": (r["amplitude"] - ref_fd) / ref_fd
            if ref_fd > 0 else float("nan"),
            "phase_half_space": float(-y / delta),
            "phase_err": r["phase"] - float(-y / delta),
        })

    layer = [r for r in out_rows if r["y"] <= delta]
    outside = [r for r in out_rows if r["y"] > delta]

    def l2(rs, key):
        if not rs:
            return float("nan")
        return math.sqrt(sum(r[key] ** 2 for r in rs) / len(rs))

    # decay rate: least-squares slope of ln A vs y over 0 < y <= delta
    fit_pts = [r for r in out_rows if r["y"] <= delta and r["amplitude"] > 0]
    if len(fit_pts) >= 3:
        ys = np.array([r["y"] for r in fit_pts])
        lnA = np.log(np.array([r["amplitude"] for r in fit_pts]))
        slope = float(np.polyfit(ys, lnA, 1)[0])
    else:
        slope = float("nan")

    metrics = {
        "n_rows_stokes_layer": len(layer),
        "n_rows_outside_layer": len(outside),
        "amplitude_l2_half_space_pct": 100.0 * l2(layer, "rel_err_half_space"),
        "amplitude_l2_finite_depth_pct": 100.0 * l2(layer, "rel_err_finite_depth"),
        "amplitude_worst_row_half_space_pct": 100.0 * max(
            (abs(r["rel_err_half_space"]) for r in layer), default=float("nan")),
        "amplitude_worst_row_y_over_delta": max(
            layer, key=lambda r: abs(r["rel_err_half_space"]))["y_over_delta"]
        if layer else float("nan"),
        # Relative error is meaningless where the amplitude itself is below the
        # discretisation noise (A < 1 % of U0 beyond y ~ 4.6 delta), so the
        # all-row numbers are quoted two ways: relative but limited to y<=2delta,
        # and absolute (normalised by U0) over every row.
        "amplitude_l2_rel_to_2delta_pct": 100.0 * l2(
            [r for r in out_rows if r["y"] <= 2.0 * delta], "rel_err_half_space"),
        "amplitude_abs_l2_all_rows_pct": 100.0 * math.sqrt(
            sum(((r["amplitude"] - r["amplitude_half_space"]) / case["u0"]) ** 2
                for r in out_rows) / max(len(out_rows), 1)),
        "decay_rate_measured": slope,
        "decay_rate_analytic": -1.0 / delta,
        "decay_rate_error_pct": 100.0 * (slope * delta + 1.0),
        "delta_measured": -1.0 / slope if slope else float("nan"),
        "delta_analytic": delta,
        "delta_ratio": (-1.0 / slope) / delta if slope else float("nan"),
        "phase_max_abs_err_rad": max((abs(r["phase_err"]) for r in layer),
                                     default=float("nan")),
        "phase_l2_err_rad": l2(layer, "phase_err"),
        "rectified_mean_max_over_U0": max(
            (abs(r["mean_over_U0"]) for r in out_rows), default=float("nan")),
        "fit_residual_max": max((r["fit_residual"] for r in out_rows),
                                default=float("nan")),
        "amplitude_max_over_U0": max((r["amplitude_over_U0"] for r in out_rows),
                                     default=float("nan")),
        "first_row_amplitude_over_U0": out_rows[0]["amplitude_over_U0"]
        if out_rows else float("nan"),
        "first_row_amplitude_analytic_over_U0": float(
            math.exp(-(0.5 * case["dx"]) / delta)),
        "window_drift_max_pct": 100.0 * max(
            (abs(v) for v in (case.get("window_drift") or {}).values()),
            default=float("nan")),
        "truncation_bias_at_delta_pct": 100.0 * (
            float(finite_depth_amplitude(np.array([delta]), height, u0,
                                         viscosity, omega)[0])
            / float(semi_infinite_amplitude(delta, u0, viscosity, omega)) - 1.0),
    }
    return {"rows": out_rows, "metrics": metrics}


def gates(res: dict) -> dict:
    m = res["metrics"]
    out = {
        "G1_amplitude_l2_5pct": abs(m["amplitude_l2_half_space_pct"])
        <= GATES["G1_amplitude_l2_pct"],
        "G2_worst_row_10pct": abs(m["amplitude_worst_row_half_space_pct"])
        <= GATES["G2_worst_row_pct"],
        "G3_decay_rate_5pct": abs(m["decay_rate_error_pct"])
        <= GATES["G3_decay_rate_pct"],
        "G4_finite_depth_l2_5pct": abs(m["amplitude_l2_finite_depth_pct"])
        <= GATES["G4_finite_depth_l2_pct"],
        "stokes_layer_has_5_rows": m["n_rows_stokes_layer"] >= 5,
    }
    if res.get("slip") == "free_slip" or not res.get("h_oscillate", True):
        # negative control: an inviscid tangential wall (or a still wall) must
        # not drive the flow at all
        out["G5_control_drives_no_flow_5pct"] = m["amplitude_max_over_U0"] <= 0.05
    return out


def run_one(delta_over_dx: int, device, *, u0: float = U0,
            viscosity: float = NU, omega: float = OMEGA, h_factor: float = H_FACTOR,
            sound_speed: float = SOUND_SPEED, slip: str = "no_slip",
            h_oscillate: bool = True, periods: float = PERIODS,
            measure_periods: float = MEASURE_PERIODS, label: str = "") -> dict:
    delta = penetration_depth(viscosity, omega)
    dx = delta / float(delta_over_dx)
    height = H_OVER_DELTA * delta
    length = COLUMNS * dx
    cfg = make_case_config(dx, height, viscosity, sound_speed, h_factor,
                           FRICTION, device)
    period = 2.0 * math.pi / omega
    t_end = periods * period
    t_measure = measure_periods * period
    case = drive(cfg, height, length, viscosity, omega, u0, t_end, t_measure,
                 SAMPLES_PER_PERIOD, device, slip=slip,
                 h_oscillate=h_oscillate, label=label or f"{delta_over_dx}")
    ev = evaluate(case, height, viscosity, omega, u0)
    result = {
        **{k: v for k, v in case.items() if k not in ("rows", "field")},
        "resolution_delta_over_dx": delta_over_dx,
        "metrics": ev["metrics"],
        "rows": ev["rows"],
        "field": case["field"],
    }
    result["gates"] = gates(result)
    return result


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_figures(results: list[dict], out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    finest = max(results, key=lambda r: r["resolution_delta_over_dx"])

    # 1. amplitude decay vs wall distance, log scale, half-space + finite depth
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for res in results:
        y = np.array([r["y_over_delta"] for r in res["rows"]])
        a = np.array([r["amplitude_over_U0"] for r in res["rows"]])
        ax[0].semilogy(y, a, "o-", ms=3, lw=1.0,
                       label=f"SPH $\\delta/dx$={res['resolution_delta_over_dx']}")
    y = np.linspace(0.0, H_OVER_DELTA, 400)
    ax[0].semilogy(y, np.exp(-y), "k-", lw=2.0, label=r"$e^{-y/\delta}$ (Landau §24)")
    ax[0].semilogy(y, finite_depth_amplitude(y * finest["delta"],
                                             finest["height"], finest["u0"],
                                             finest["viscosity"],
                                             finest["omega"]) / finest["u0"],
                   "k--", lw=1.2, label="finite depth (exact, H=6$\\delta$)")
    ax[0].set_xlabel(r"$y/\delta$")
    ax[0].set_ylabel(r"$A(y)/U_0$")
    ax[0].set_title("amplitude decay in the Stokes layer")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # 2. relative error of the amplitude, per row
    for res in results:
        y = np.array([r["y_over_delta"] for r in res["rows"]])
        e = 100.0 * np.array([r["rel_err_half_space"] for r in res["rows"]])
        ax[1].plot(y, e, "o-", ms=3, lw=1.0,
                   label=f"$\\delta/dx$={res['resolution_delta_over_dx']}")
    ax[1].axhspan(-5, 5, color="0.85", label="±5 % gate (G1)")
    ax[1].axvline(1.0, color="k", ls=":", lw=1.0)
    ax[1].set_xlabel(r"$y/\delta$")
    ax[1].set_ylabel(r"$(A_{SPH}-A_{analytic})/A_{analytic}$  [%]")
    ax[1].set_title("amplitude error per row")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "amplitude_decay.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(str(os.path.relpath(p1, REPO)))

    # 3. space-time field of the finest run + phase snapshot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.array(finest["field"]["y"])
    t = np.array(finest["field"]["t"])
    u = np.array([[np.nan if v is None else v for v in row]
                  for row in finest["field"]["u"]])
    mesh = ax[0].pcolormesh(t, y / finest["delta"], u / finest["u0"],
                            shading="auto", cmap="RdBu_r")
    cb = fig.colorbar(mesh, ax=ax[0])
    cb.set_label(r"$u_x/U_0$")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel(r"$y/\delta$")
    ax[0].set_title("SPH velocity field (measurement window)")
    ax[0].invert_yaxis()

    t_cut = t[-1] - finest["period"]
    for res in results:
        tt = np.array(res["field"]["t"])
        yy = np.array(res["field"]["y"]) / res["delta"]
        uu = np.array([[np.nan if v is None else v for v in row]
                       for row in res["field"]["u"]])
        idx = int(np.argmin(np.abs(tt - t_cut)))
        ax[1].plot(uu[:, idx] / res["u0"], yy, "o-", ms=3, lw=1.0,
                   label=f"SPH $\\delta/dx$={res['resolution_delta_over_dx']}")
    yv = np.linspace(0, H_OVER_DELTA, 400)
    ax[1].plot(finite_depth(yv * finest["delta"], t_cut, finest["height"],
                            finest["u0"], finest["viscosity"], finest["omega"])
               / finest["u0"], yv, "k--", lw=1.6,
               label="finite-depth exact")
    ax[1].set_xlabel(r"$u_x/U_0$")
    ax[1].set_ylabel(r"$y/\delta$")
    ax[1].set_title("snapshot profile")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "field_snapshot.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(str(os.path.relpath(p2, REPO)))
    return made


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

SENSITIVITY_SPEC = (
    ("U0_half", dict(u0=0.5 * U0)),
    ("h_factor_1.9", dict(h_factor=1.9)),
    ("free_slip_control", dict(slip="free_slip")),
    ("static_wall_control", dict(h_oscillate=False)),
)


def run_sensitivities(device, args) -> list[dict]:
    """One-at-a-time variation of U0, h/dx and the wall model, at one mesh."""
    out = []
    skip = set(args.skip_sensitivity or ())
    for name, kw in SENSITIVITY_SPEC:
        if name in skip:
            continue
        r = run_one(int(args.sensitivity_resolution), device,
                    sound_speed=args.c0, periods=args.periods,
                    measure_periods=args.measure_periods,
                    label=f"sensitivity_{name}", **kw)
        out.append(r)
        print(f"  {name}: L2(layer)={r['metrics']['amplitude_l2_half_space_pct']:.2f} %"
              f"  A_max/U0={r['metrics']['amplitude_max_over_U0']:.4f}"
              f"  wall={r['wall_clock_s']:.1f} s", flush=True)
    return out


def _round_sig(obj, sig: int = 6):
    """Round floats for persistence (full precision stays in memory)."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}") if math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _round_sig(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_sig(v, sig) for v in obj]
    return obj


def _slim_payload(payload: dict) -> dict:
    """Keep the artefacts reviewable: field only for the finest mesh, decimated.

    A 4 MB result file is dropped by artefact filters (that is exactly how the
    first delivery of this case lost its run artefacts), so the persisted form
    carries every metric and every fitted row, but the space-time field only
    once and every second sample.
    """
    out = _round_sig(payload)
    runs = out.get("runs") or []
    finest = max((r["resolution_delta_over_dx"] for r in runs), default=None)
    for run in runs:
        field = run.get("field")
        if not isinstance(field, dict):
            continue
        if run["resolution_delta_over_dx"] != finest:
            run.pop("field", None)
            run["field_note"] = "dropped: only the finest mesh's field is kept"
        else:
            # 8x in time and 4 significant digits: the figure is rendered from
            # the in-memory array, so the persisted copy only has to let a
            # reviewer re-draw it, not to be the primary store.
            step = 8
            run["field"] = {"t": _round_sig(field["t"][::step], 4), "y": field["y"],
                            "u": _round_sig([row[::step] for row in field["u"]], 4),
                            "decimation": step}
    for sub in ("sensitivity",):
        for run in out.get(sub) or []:
            run.pop("field", None)
            run.pop("trace", None)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _benchmark_rows(run: dict) -> list[dict]:
    """ref vs got, one row per judged quantity (machine-readable gate table)."""
    m = run["metrics"]
    rows = [
        ("amplitude_l2_half_space_pct", 0.0, m["amplitude_l2_half_space_pct"],
         GATES["G1_amplitude_l2_pct"],
         "row-mean rms of (A_SPH-A_ref)/A_ref over y<=delta, %; ref = 0"),
        ("amplitude_worst_row_half_space_pct", 0.0,
         m["amplitude_worst_row_half_space_pct"], GATES["G2_worst_row_pct"],
         "max |(A_SPH-A_ref)/A_ref| over y<=delta, %"),
        ("decay_rate_error_pct", 0.0, m["decay_rate_error_pct"],
         GATES["G3_decay_rate_pct"],
         "100*(-dlnA/dy * delta + 1); 0 = exact exp(-y/delta)"),
        ("amplitude_l2_finite_depth_pct", 0.0,
         m["amplitude_l2_finite_depth_pct"], GATES["G4_finite_depth_l2_pct"],
         "same rms vs the exact finite-depth solution of the run"),
        ("delta_ratio", 1.0, m["delta_ratio"], 0.05,
         "fitted delta = -1/slope vs delta = sqrt(2 nu/omega)"),
        ("first_row_amplitude_over_U0", m["first_row_amplitude_analytic_over_U0"],
         m["first_row_amplitude_over_U0"], 5.0,
         "A at y=dx/2 vs exp(-dx/2/delta)"),
    ]
    out = []
    for name, ref, got, tol, definition in rows:
        err = 100.0 * (got - ref) / ref if ref else float(got)
        out.append({"quantity": name, "reference": ref, "got": got,
                    "tolerance": tol,
                    "error_abs": abs(got - ref),
                    "error_pct_of_ref": err,
                    "pass": bool(abs(got - ref) <= tol) if ref == 0.0
                    else bool(abs(got / ref - 1.0) * 100.0 <= tol),
                    "definition": definition})
    return out


def _write_bundle(out_dir: Path, payload: dict, repro_cmd: str) -> None:
    """benchmark.json (ref vs got) + MANIFEST.json (sha256 of every artefact)."""
    if payload["mode"] == "ladder":
        finest = max(payload["runs"],
                     key=lambda r: r["resolution_delta_over_dx"])
        bench = {
            "anchor": "CASE-SPH-STOKES-2ND-PROBLEM",
            "case": "stokes_second_oscillating_plate",
            "primary_metric": "amplitude_l2_half_space_pct",
            "primary_tolerance_pct": GATES["G1_amplitude_l2_pct"],
            "primary_value_pct": finest["metrics"]["amplitude_l2_half_space_pct"],
            "reference": ("u(y,t)=U0 exp(-y/delta) cos(omega t - y/delta), "
                          "delta=sqrt(2 nu/omega) -- Landau & Lifshitz, Fluid "
                          "Mechanics 2nd English ed., section 24 (Stokes' second "
                          "problem); the finite-depth form is the exact solution "
                          "of the channel actually run"),
            "resolution_delta_over_dx": finest["resolution_delta_over_dx"],
            "rows_of_finest_run": _benchmark_rows(finest),
            "repro_cmd": repro_cmd,
        }
        (out_dir / "benchmark.json").write_text(
            json.dumps(_slim_payload(bench), indent=1))
    files = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            files[path.name] = {"sha256": _sha256(path),
                                "bytes": path.stat().st_size}
    (out_dir / "MANIFEST.json").write_text(json.dumps(
        {"case": "stokes_second_oscillating_plate",
         "anchor": "CASE-SPH-STOKES-2ND-PROBLEM",
         "mode": payload["mode"],
         "provenance": payload["provenance"],
         "gates": payload["provenance"]["gates"],
         "artifacts": files,
         "command": repro_cmd}, indent=1, default=_jsonable))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--quick", action="store_true",
                    help="one coarse run, short window (smoke test)")
    ap.add_argument("--ladder", action="store_true",
                    help="resolution ladder, gates G1..G4, G6")
    ap.add_argument("--sweep", action="store_true",
                    help="ladder plus parameter sensitivity and negative controls")
    ap.add_argument("--probe", default=None,
                    help="one run at the given delta/dx and dump its metrics")
    ap.add_argument("--periods", type=float, default=PERIODS)
    ap.add_argument("--measure-periods", type=float, default=MEASURE_PERIODS)
    ap.add_argument("--c0", type=float, default=SOUND_SPEED)
    ap.add_argument("--sensitivity-resolution", type=int, default=8)
    ap.add_argument("--skip-sensitivity", nargs="*", default=None)
    ap.add_argument("--sensitivity-only", action="store_true",
                    help="skip the ladder, run only the sensitivity/control block")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU threads; 1 is fastest for these tiny tensors")
    args = ap.parse_args(argv)
    repro_cmd = ("PYTHONPATH=.:src " + " ".join(
        ["python", "benchmarks/stokes_second/run.py"] + list(sys.argv[1:])))
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "symbolic": verify_symbolic(),
        "numeric": verify_numeric(),
        # filled from the parsed arguments below: a provenance block that
        # repeats module defaults describes a run that may never have happened.
        "case": {"nu": NU, "omega": OMEGA, "U0": U0, "H_over_delta": H_OVER_DELTA,
                 "columns": COLUMNS, "rho0": RHO0, "c0": args.c0,
                 "periods": args.periods,
                 "measure_periods": args.measure_periods,
                 "ladder": list(LADDER),
                 "gamma": GAMMA, "h_factor": H_FACTOR,
                 "wall_friction": FRICTION,
                 "mach_number": U0 / args.c0,
                 "reynolds_u0_delta_over_nu": U0 * penetration_depth(NU, OMEGA) / NU,
                 "solvers": {"rhs": "tensorsph.droplet_impact.rhs",
                             "wall": "tensorsph.wall_mirror.plane_walls",
                             "mesh": "benchmarks.poiseuille.run.build_fluid"}},
        "gates": GATES,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    started = time.time()

    if args.probe:
        res = run_one(int(args.probe), device, periods=args.periods,
                      measure_periods=args.measure_periods,
                      sound_speed=args.c0, label=f"probe_dx{args.probe}")
        payload = {"mode": "probe", "provenance": provenance, "runs": [res],
                   "repro_cmd": repro_cmd}
        path = out_dir / "results_probe.json"
        path.write_text(json.dumps(_slim_payload(payload), indent=1,
                                   default=_jsonable))
        _write_bundle(out_dir, payload, repro_cmd)
        print(json.dumps(res["metrics"], indent=1))
        print("gates", res["gates"])
        print(f"wall {res['wall_clock_s']:.1f} s, steps {res['steps']}, "
              f"dt {res['dt']:.3e}, n_fluid {res['n_fluid']}")
        return 0

    if args.sensitivity_only:
        payload = {"mode": "sensitivity", "provenance": provenance,
                   "sensitivity": run_sensitivities(device, args),
                   "repro_cmd": repro_cmd}
        path = out_dir / "results_sensitivity.json"
        payload["elapsed_s"] = time.time() - started
        path.write_text(json.dumps(_slim_payload(payload), indent=1,
                                   default=_jsonable))
        _write_bundle(out_dir, payload, repro_cmd)
        print(f"wrote {path}")
        return 0

    if args.quick:
        res = run_one(6, device, periods=2.0, measure_periods=1.0, label="quick",
                      sound_speed=args.c0)
        payload = {"mode": "quick", "provenance": provenance, "runs": [res],
                   "repro_cmd": repro_cmd}
    else:
        ladder = LADDER if (args.ladder or args.sweep) else (LADDER[1],)
        runs = []
        for d in ladder:
            r = run_one(int(d), device, sound_speed=args.c0,
                        periods=args.periods,
                        measure_periods=args.measure_periods,
                        label=f"ladder_dx{d}")
            runs.append(r)
            print(f"  delta/dx={d}: L2(half-space)={r['metrics']['amplitude_l2_half_space_pct']:.2f} %"
                  f"  L2(finite)={r['metrics']['amplitude_l2_finite_depth_pct']:.2f} %"
                  f"  rows in layer={r['metrics']['n_rows_stokes_layer']}"
                  f"  wall={r['wall_clock_s']:.1f} s", flush=True)
        payload = {"mode": "ladder", "provenance": provenance, "runs": runs}

        if args.sweep:
            payload["sensitivity"] = run_sensitivities(device, args)

    # convergence gate G6 across the ladder.  A one-rung "ladder" cannot test
    # convergence, so completeness is part of the gate: the required meshes are
    # the predeclared ones, and a missing rung fails the run instead of being
    # silently skipped (that is how the first delivery shipped n=1).
    if payload["mode"] == "ladder":
        runs = payload["runs"]
        res = [r["resolution_delta_over_dx"] for r in runs]
        l2s = [abs(r["metrics"]["amplitude_l2_half_space_pct"]) for r in runs]
        required = list(LADDER) if (args.ladder or args.sweep) else [LADDER[1]]
        conv = {
            "resolutions": res,
            "required_resolutions": required,
            "l2_by_resolution": l2s,
            "l2_finite_depth_by_resolution": [
                abs(r["metrics"]["amplitude_l2_finite_depth_pct"]) for r in runs],
            "complete": res == required,
        }
        if len(runs) >= 2:
            conv["observed_order_l2_half_space"] = [
                math.log(l2s[i - 1] / l2s[i]) / math.log(res[i] / res[i - 1])
                for i in range(1, len(runs)) if l2s[i] > 0 and l2s[i - 1] > 0]
            conv["G6_not_growing"] = l2s[-1] <= l2s[0] + 0.5
        else:
            conv["observed_order_l2_half_space"] = []
            conv["G6_not_growing"] = False
            conv["G6_reason"] = ("fewer than two rungs were run: convergence "
                                 "cannot be claimed")
        conv["all_gates"] = bool(conv["complete"] and conv["G6_not_growing"]
                                 and all(all(r["gates"].values()) for r in runs))
        payload["convergence"] = conv

    if payload["mode"] == "ladder" and len(payload["runs"]) >= 1:
        try:
            payload["figures"] = make_figures(payload["runs"], out_dir)
        except Exception as exc:           # a broken plot must not lose the run
            payload["figures_error"] = f"{type(exc).__name__}: {exc}"

    payload["elapsed_s"] = time.time() - started
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["repro_cmd"] = repro_cmd
    path = out_dir / f"results_{payload['mode']}.json"
    path.write_text(json.dumps(_slim_payload(payload), indent=1,
                               default=_jsonable))
    print(f"wrote {path}  ({payload['elapsed_s']:.1f} s)")
    _write_bundle(out_dir, payload, repro_cmd)

    if payload["mode"] != "quick":
        ok = payload.get("convergence", {}).get("all_gates", False)
        print("gates:", json.dumps({f"dx{r['resolution_delta_over_dx']}": r["gates"]
                                    for r in payload["runs"]}, indent=1,
                                   default=_jsonable))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())