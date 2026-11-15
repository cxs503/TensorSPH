#!/usr/bin/env python3
"""Start-up of plane Poiseuille flow against the Boussinesq series, *with* walls.

Why this benchmark exists
-------------------------
Every Poiseuille result in this repository that is graded against the analytic
solution is a **steady-state** result: the mirror/ghost no-slip wall
(:mod:`tensorsph.wall_mirror`, ``benchmarks/poiseuille_mirror``) is certified at
``t = 10 tau`` only.  A steady profile constrains the *amplitude* of the
discrete viscous operator; it constrains neither its *rate* nor the time
integration that produced it.  The start-up transient does: the exact solution
at every sampled time is known, so amplitude and rate are separately observable.

The sibling harness ``benchmarks/poiseuille_startup/run.py`` looked like it did
this, but until this mission its wall set was an empty tensor
(``build_mirror_walls`` returned ``torch.empty((0, 2))``) *and* its sampling
buffer was allocated with the wrong rank, so the harness could never have
produced the numbers it advertised.  ``verification_ledger`` already records it
as an ``uncovered_benchmark``.  That path is now repaired (per-step mirror
ghosts) and this benchmark is the graded one.

Physics / reference
-------------------
Fluid at rest in a 2-D channel of height ``H``, walls at ``y = 0, H``, no-slip,
periodic in ``x``, uniform body force ``g_x`` switched on at ``t = 0``:

    u_t = nu u_yy + g,   u(y, 0) = 0,   u(0, t) = u(H, t) = 0,

whose exact solution is the Boussinesq series (Boussinesq 1868; Batchelor,
*An Introduction to Fluid Dynamics*, §4.2; Schlichting, *Boundary-Layer
Theory*, ch. III; Morris, Fox & Zhu, *J. Comput. Phys.* 136 (1997) 214):

    u_s(y)    = g (H - y) y / (2 nu),          u_max,inf = g H^2 / (8 nu)
    u(y, t)   = u_s(y) - (4 g H^2 / (nu pi^3)) sum_{k>=0}
                    sin(n pi y / H) / n^3 * exp(-n^2 pi^2 nu t / H^2),  n = 2k+1
    tau       = H^2 / (nu pi^2)                (slowest-mode e-folding time)

The series is *not* trusted as typed: :func:`check_reference_provenance`
re-solves the same initial-boundary-value problem with an independent
Crank-Nicolson finite-difference code and requires agreement to ``1e-4`` of
``u_max,inf`` at six times.  A transcription or sign error in the series shows
up there, not in the comparison with SPH.

Metrics (predeclared below in ``GATES``, before any ladder is run)
------------------------------------------------------------------
G1  centre-line history:  max |u_c(t) / u_ref(t) - 1| over
                          ``0.25 tau <= t <= 5 tau`` at ``H/dx = 16``   <= 5 %
G2  instantaneous shape:  interior row-RMS error at ``t/tau in {0.5,1,2,4}``
                          at ``H/dx = 16``                              <= 5 %
G3  refinement:           G1 is non-increasing over ``H/dx = 8, 12, 16``
G4a amplitude viscosity:  ``nu_amp / nu - 1`` from the late-time level     <= 5 %
G4b rate viscosity:       ``nu_rate / nu - 1`` from a one-parameter fit of
                          the *shape* of u_c(t)                           <= 5 %
G4c amplitude vs rate:    ``nu_rate / nu_amp - 1``                         <= 5 %
G5  free-slip control:    rigid-body acceleration ``u_c = g t`` at
                          ``t/tau in {1,2,4,8}``                         <= 2 %
G6  wall-less trap:       the same run with the ghost layer switched off
                          must MISS G1 by a wide margin (no silent pass)

G4 is the part a steady-state gate cannot do: ``nu_amp`` uses the plateau,
``nu_rate`` uses how fast the plateau is reached.  If the two disagree, the
solver's viscous operator has an amplitude that is right and a rate that is
wrong (or the reverse) -- which is exactly what a steady-only gate hides.

Usage
-----
    PYTHONPATH=src:. python benchmarks/poiseuille_startup_mirror/run.py \
        --resolutions 8 12 16 --out benchmarks/poiseuille_startup_mirror/out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.poiseuille.run import (
    bin_rows,
    build_fluid,
    make_config,
    periodic_images,
)
from benchmarks.poiseuille_startup.run import boussinesq_series
from tensorsph.droplet_impact import resolve_device, rhs
from tensorsph.wall_mirror import plane_walls

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Predeclared thresholds -- fixed BEFORE the ladder is run
# --------------------------------------------------------------------------- #

GATES = {
    "G1_centreline_history_5pct": 5.0,     # max |u_c/u_ref-1|, 0.25..5 tau, H/dx=16
    "G2_profile_rms_5pct": 5.0,            # interior row RMS at 4 times, H/dx=16
    "G3_refinement_monotone": None,        # boolean, G1 non-increasing in H/dx
    "G4a_amplitude_viscosity_5pct": 5.0,   # |nu_amp/nu - 1|
    "G4b_rate_viscosity_5pct": 5.0,        # |nu_rate/nu - 1|
    "G4c_amp_vs_rate_5pct": 5.0,           # |nu_rate/nu_amp - 1|
    "G5_free_slip_rigid_body_2pct": 2.0,   # |u_c/(g t) - 1|
    "G6_wallless_must_miss": None,         # boolean, wall-less run misses G1
}

# Sampling protocol (also fixed in advance)
HISTORY_WINDOW_LO = 0.25   # in units of tau
HISTORY_WINDOW_HI = 5.0
SMOOTH_HALF_WIDTH = 0.05   # box half-width for the history, in units of tau
PROFILE_TIMES = (0.5, 1.0, 2.0, 4.0)   # t/tau
PROFILE_HALF_WIDTH = 0.05  # in units of tau
N_TRACE_SAMPLES = 600


# --------------------------------------------------------------------------- #
# Reference provenance: independent Crank-Nicolson solve of the same IVP
# --------------------------------------------------------------------------- #

def _thomas(a, b, c, d):
    """Tridiagonal solve (a=sub, b=diag, c=super, d=rhs), in place-safe."""
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def fd_reference(times, height: float, body_force: float, viscosity: float,
                 n_nodes: int = 201, n_steps: int = 4000) -> dict:
    """Crank-Nicolson solution of ``u_t = nu u_yy + g`` with u = 0 at both walls.

    Deliberately independent of the series: no closed form is used anywhere in
    this function, only a tridiagonal solve on a uniform grid.
    """
    t_end = max(times)
    y = np.linspace(0.0, height, n_nodes)
    dy = y[1] - y[0]
    interior = slice(1, -1)
    ni = n_nodes - 2
    u = np.zeros(n_nodes)
    dt = t_end / n_steps
    r = viscosity * dt / (2.0 * dy * dy)

    sub = np.full(ni, -r)
    sup = np.full(ni, -r)
    diag = np.full(ni, 1.0 + 2.0 * r)
    sub[0] = 0.0
    sup[-1] = 0.0
    diag_old = np.full(ni, 1.0 - 2.0 * r)
    sub_old = np.full(ni, r)
    sup_old = np.full(ni, r)
    sub_old[0] = 0.0
    sup_old[-1] = 0.0

    targets = sorted(times)
    out = {}
    k = 0
    for step in range(1, n_steps + 1):
        ui = u[interior]
        rhs = diag_old * ui + body_force * dt
        rhs[1:] += sub_old[1:] * ui[:-1]
        rhs[:-1] += sup_old[:-1] * ui[1:]
        u[interior] = _thomas(sub.copy(), diag.copy(), sup.copy(), rhs)
        t = step * dt
        while k < len(targets) and t >= targets[k] - 1e-15:
            out[targets[k]] = u.copy()
            k += 1
    return {"y": y, "profiles": out, "n_nodes": n_nodes, "n_steps": n_steps,
            "dt": dt, "scheme": "Crank-Nicolson, tridiagonal (Thomas)"}


def check_reference_provenance(height: float, body_force: float,
                               viscosity: float) -> dict:
    """Bind the series to an independent numerical solution of the same IVP."""
    tau = height ** 2 / (viscosity * math.pi ** 2)
    times = [0.25 * tau, 0.5 * tau, 1.0 * tau, 2.0 * tau, 4.0 * tau, 8.0 * tau]
    fd = fd_reference(times, height, body_force, viscosity)
    u_max_inf = body_force * height ** 2 / (8.0 * viscosity)
    rows = []
    worst = 0.0
    for t in times:
        u_fd = fd["profiles"][t]
        u_series = boussinesq_series(fd["y"], t, height, body_force, viscosity,
                                     n_terms=400)
        # compare only interior nodes; the wall nodes are exactly zero in both
        diff = np.abs(u_fd[1:-1] - u_series[1:-1]).max() / u_max_inf
        worst = max(worst, float(diff))
        rows.append({"t_over_tau": t / tau, "max_abs_diff_over_umax": float(diff)})
    return {
        "method": "series (Boussinesq) vs independent Crank-Nicolson FD",
        "fd_grid": {"n_nodes": fd["n_nodes"], "n_steps": fd["n_steps"],
                    "dt": fd["dt"]},
        "tau_s": tau,
        "u_max_inf": u_max_inf,
        "comparison": rows,
        "worst_rel_to_umax": worst,
        "agrees_to_1e-4": bool(worst < 1e-4),
    }


# --------------------------------------------------------------------------- #
# Transient WCSPH run with mirror no-slip walls
# --------------------------------------------------------------------------- #

def _centre_rows(pos, height: float, dx: float):
    """Particles within half a spacing of the centre line."""
    return (pos[:, 1] - 0.5 * height).abs() <= 0.5 * dx


def run_transient(cfg, height: float, length: float, body_force: float,
                  t_end: float, device, slip: str = "no_slip",
                  wallless: bool = False, n_trace: int = N_TRACE_SAMPLES,
                  profile_times=PROFILE_TIMES,
                  profile_half_width: float = PROFILE_HALF_WIDTH,
                  viscosity: float = 0.01) -> dict:
    """Integrate from rest, record the centre-line history and 4 profiles."""
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (height, -1)]

    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    tau = height ** 2 / (viscosity * math.pi ** 2)

    centre_mask = _centre_rows(pos, height, dx)
    trace_every = max(1, steps // n_trace)

    # profile accumulators: window sum of per-particle velocity + row index
    row_index = torch.round((pos[:, 1] - dx * 0.5) / dx).to(torch.long)
    windows = {tf: {"sum": torch.zeros_like(pos), "count": 0,
                    "lo": (tf - profile_half_width) * tau,
                    "hi": (tf + profile_half_width) * tau}
               for tf in profile_times}

    times, u_centre, u_std = [], [], []
    wall_clock = time.time()
    n_ghost = 0

    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)

        if wallless:
            ghost_pos = ghost_vel = None
        else:
            ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                                  planes, support, slip=slip)
            n_ghost = int(ghost_pos.shape[0])

        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                        wall=ghost_pos if ghost_pos is not None else None,
                        wall_vel=ghost_vel if ghost_pos is not None else None)
        acc = acc[:n].clone()
        acc[:, 0] += body_force

        t = step * dt
        if step % trace_every == 0 or step == steps - 1:
            uc = vel[centre_mask, 0]
            times.append(t)
            u_centre.append(float(uc.mean()))
            u_std.append(float(uc.std(unbiased=False)) if uc.numel() > 1 else 0.0)

        for w in windows.values():
            if w["lo"] <= t <= w["hi"]:
                w["sum"] += vel
                w["count"] += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

    profiles = {}
    for tf, w in windows.items():
        if w["count"] == 0:
            profiles[tf] = None
            continue
        u_mean = w["sum"] / w["count"]
        rows, prof = bin_rows(pos, u_mean, height, dx)
        profiles[tf] = {"rows": rows, "profile": prof, "window_count": w["count"]}

    return {
        "n_fluid": int(n),
        "n_ghost": n_ghost,
        "wallless": bool(wallless),
        "slip": slip,
        "dt": dt,
        "steps": steps,
        "trace_every": trace_every,
        "t": np.array(times),
        "u_centre": np.array(u_centre),
        "u_centre_std": np.array(u_std),
        "profiles": profiles,
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def smooth(t, u, half_width):
    """Centred box average; returns smoothed values with the same shape."""
    out = np.empty_like(u)
    for i, ti in enumerate(t):
        sel = np.abs(t - ti) <= half_width
        out[i] = u[sel].mean()
    return out


def series_centre(t, height, body_force, viscosity, n_terms=400):
    """Series centre-line value at times ``t`` (vectorised over t)."""
    y = np.full(1, 0.5 * height)
    return np.array([float(boussinesq_series(y, float(ti), height, body_force,
                                             viscosity, n_terms=n_terms)[0])
                     for ti in np.atleast_1d(t)])


def series_shape(t_over_tau, height, body_force, viscosity):
    """Dimensionless centre-line shape u_c(t)/u_max,inf for the exact solution."""
    tau = height ** 2 / (viscosity * math.pi ** 2)
    u_max = body_force * height ** 2 / (8.0 * viscosity)
    return series_centre(np.asarray(t_over_tau) * tau, height, body_force,
                         viscosity) / u_max


def fit_rate_viscosity(t, u_c, u_steady_meas, height, body_force,
                       viscosity, lo_factor=0.4, hi_factor=2.5):
    """One-parameter fit of nu in the *shape* of the measured centre-line.

    The amplitude is factored out (the measured curve is normalised by its own
    late-time level), so this number measures the *rate* of the transient and
    is largely blind to a uniform multiplicative error in the viscous
    amplitude.  A two-stage geometric search (coarse over ``[0.4, 2.5] nu``,
    then five zoom refinements) keeps it dependency-free.
    """
    uu = np.asarray(u_c) / u_steady_meas
    tt = np.asarray(t)

    def rms_for(nu):
        tau = height ** 2 / (nu * math.pi ** 2)
        shape = series_shape(tt / tau, height, body_force, nu)
        return float(np.sqrt(np.mean((shape - uu) ** 2)))

    grid = np.geomspace(lo_factor, hi_factor, 161) * viscosity
    errs = [rms_for(nu) for nu in grid]
    k = int(np.argmin(errs))
    lo = grid[max(0, k - 1)]
    hi = grid[min(len(grid) - 1, k + 1)]
    for _ in range(60):
        cand = np.linspace(lo, hi, 21)
        e = [rms_for(nu) for nu in cand]
        j = int(np.argmin(e))
        best_nu, best_err = float(cand[j]), float(e[j])
        if (hi - lo) / best_nu < 1e-7:
            break
        lo, hi = float(cand[max(0, j - 1)]), float(cand[min(len(cand) - 1, j + 1)])
    return {"nu_fit": best_nu, "shape_rms": best_err,
            "search_range_over_nu": [lo_factor, hi_factor]}


def solve_amplitude_viscosity(t_late, u_c_late, height, body_force, viscosity,
                              lo_factor=0.5, hi_factor=2.0):
    """nu such that the exact series *level* matches the measured late level."""
    grid = np.geomspace(lo_factor, hi_factor, 200) * viscosity
    vals = [abs(float(series_centre([t_late], height, body_force, nu)[0])
                - u_c_late) for nu in grid]
    return float(grid[int(np.argmin(vals))])


def evaluate(res: dict, height: float, body_force: float, viscosity: float,
             resolution: int, dx: float) -> dict:
    tau = height ** 2 / (viscosity * math.pi ** 2)
    u_max_inf = body_force * height ** 2 / (8.0 * viscosity)
    t, u_raw = res["t"], res["u_centre"]
    u_sm = smooth(t, u_raw, SMOOTH_HALF_WIDTH * tau)

    ref = series_centre(t, height, body_force, viscosity)
    sel = (t >= HISTORY_WINDOW_LO * tau) & (t <= HISTORY_WINDOW_HI * tau)
    rel = np.abs(u_sm[sel] - ref[sel]) / np.abs(ref[sel])
    i_worst = np.argmax(rel)
    # smoothing bias: same box average applied to the exact solution
    ref_sm = smooth(t, ref, SMOOTH_HALF_WIDTH * tau)
    bias = np.max(np.abs(ref_sm[sel] - ref[sel]) / np.abs(ref[sel]))

    # instantaneous-ish profile errors
    prof_rows = []
    for tf in (PROFILE_TIMES if res["profiles"] else ()):
        p = res["profiles"].get(tf)
        if p is None:
            continue
        rows = np.array([r["y"] for r in p["profile"]], dtype=float)
        vals = np.array([r["u_mean"] for r in p["profile"]], dtype=float)
        inter = (rows >= 0.1 * height) & (rows <= 0.9 * height)
        exact = boussinesq_series(rows, tf * tau, height, body_force, viscosity,
                                  n_terms=400)
        diff = vals[inter] - exact[inter]
        rms_rel = float(np.sqrt(np.mean(diff ** 2)) /
                        np.max(np.abs(exact[inter]))) * 100.0
        ic = int(np.argmin(np.abs(rows - 0.5 * height)))
        prof_rows.append({
            "t_over_tau": tf,
            "interior_rms_pct": rms_rel,
            "centre_meas": float(vals[ic]),
            "centre_exact": float(exact[ic]),
            "centre_err_pct": float(100.0 * (vals[ic] / exact[ic] - 1.0)),
            "rows": rows.tolist(),
            "u_meas": vals.tolist(),
            "u_exact": exact.tolist(),
            "window_count": p["window_count"],
        })

    # amplitude viscosity from the late-time level (window-averaged at 8 tau)
    late_sel = np.abs(t - 8.0 * tau) <= 0.25 * tau
    if not late_sel.any():
        late_sel = np.abs(t - t[-1]) <= 1e-12
    i_late = int(np.argmin(np.abs(t - 8.0 * tau)))
    u_late = float(u_sm[late_sel].mean())
    t_late = float(t[late_sel].mean())
    nu_amp = solve_amplitude_viscosity(t_late, u_late, height, body_force,
                                       viscosity)

    # rate viscosity from the transient shape
    sel_fit = (t >= HISTORY_WINDOW_LO * tau) & (t <= HISTORY_WINDOW_HI * tau)
    fit = fit_rate_viscosity(t[sel_fit], u_sm[sel_fit], u_late, height,
                             body_force, viscosity)

    return {
        "H_over_dx": resolution,
        "dx": dx,
        "n_fluid": res["n_fluid"],
        "n_ghost": res["n_ghost"],
        "wallless": res["wallless"],
        "slip": res["slip"],
        "dt": res["dt"],
        "steps": res["steps"],
        "tau": tau,
        "u_max_inf": u_max_inf,
        "history_max_err_pct": float(100.0 * rel[i_worst]),
        "history_max_err_t_over_tau":
            float(t[sel][i_worst] / tau),
        "history_rms_err_pct": float(100.0 * np.sqrt(np.mean(rel ** 2))),
        "smoothing_bias_pct": float(100.0 * bias),
        "u_centre_late": u_late,
        "u_centre_late_t_over_tau": t_late / tau,
        "u_centre_late_over_exact": u_late / float(
            series_centre([t_late], height, body_force, viscosity)[0]),
        "profile_rows": prof_rows,
        "profile_rms_max_pct": (max(r["interior_rms_pct"] for r in prof_rows)
                                if prof_rows else None),
        "u_centre_std_max": float(np.max(res["u_centre_std"])),
        "nu_amp": nu_amp,
        "nu_amp_over_nu": nu_amp / viscosity,
        "nu_rate": fit["nu_fit"],
        "nu_rate_over_nu": fit["nu_fit"] / viscosity,
        "nu_rate_over_nu_amp": fit["nu_fit"] / nu_amp,
        "shape_fit_rms": fit["shape_rms"],
        "t": t.tolist(),
        "u_centre": u_raw.tolist(),
        "u_centre_smoothed": u_sm.tolist(),
        "u_exact_centre": ref.tolist(),
        "wall_clock_s": res["wall_clock_s"],
    }


def gates(ladder: list, control: dict, wallless: dict) -> dict:
    top = [r for r in ladder if r["H_over_dx"] == max(r_["H_over_dx"]
                                                      for r_ in ladder)][0]
    g1 = top["history_max_err_pct"] <= GATES["G1_centreline_history_5pct"]
    g2 = (top["profile_rms_max_pct"] is not None
          and top["profile_rms_max_pct"] <= GATES["G2_profile_rms_5pct"])
    hist = [r["history_max_err_pct"] for r in
            sorted(ladder, key=lambda r: r["H_over_dx"])]
    g3 = all(hist[i + 1] <= hist[i] * 1.02 + 0.05 for i in range(len(hist) - 1))
    g4a = abs(top["nu_amp_over_nu"] - 1.0) * 100 <= GATES["G4a_amplitude_viscosity_5pct"]
    g4b = abs(top["nu_rate_over_nu"] - 1.0) * 100 <= GATES["G4b_rate_viscosity_5pct"]
    g4c = abs(top["nu_rate_over_nu_amp"] - 1.0) * 100 <= GATES["G4c_amp_vs_rate_5pct"]
    g5 = (control["max_rigid_body_err_pct"] <= GATES["G5_free_slip_rigid_body_2pct"]
          if control else False)
    g6 = wallless["history_max_err_pct"] > GATES["G1_centreline_history_5pct"]
    return {
        "G1_centreline_history": bool(g1),
        "G2_profile_rms": bool(g2),
        "G3_refinement_monotone": bool(g3),
        "G4a_amplitude_viscosity": bool(g4a),
        "G4b_rate_viscosity": bool(g4b),
        "G4c_amp_vs_rate": bool(g4c),
        "G5_free_slip_rigid_body": bool(g5),
        "G6_wallless_must_miss": bool(g6),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks/poiseuille_startup_mirror/out")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--viscosity", type=float, default=0.01)
    ap.add_argument("--body-force", type=float, default=9.81)
    ap.add_argument("--height", type=float, default=0.1)
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--t-end-tau", type=float, default=9.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args(argv)

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.quick:
        args.resolutions = [8]
        args.t_end_tau = 2.0

    tau = args.height ** 2 / (args.viscosity * math.pi ** 2)
    t_end = args.t_end_tau * tau
    print(f"[startup-mirror] tau={tau:.5f}s t_end={t_end:.4f}s", flush=True)

    prov = check_reference_provenance(args.height, args.body_force,
                                      args.viscosity)
    print(f"[startup-mirror] reference provenance: worst |series-FD|/u_max = "
          f"{prov['worst_rel_to_umax']:.2e} (agrees={prov['agrees_to_1e-4']})",
          flush=True)

    ladder = []
    raw = []
    for r in args.resolutions:
        dx = args.height / r
        cfg = make_config(dx, args.viscosity, 1.0, device)
        print(f"[startup-mirror] H/dx={r} ...", flush=True)
        res = run_transient(cfg, args.height, args.columns * dx,
                            args.body_force, t_end, device,
                            viscosity=args.viscosity)
        ev = evaluate(res, args.height, args.body_force, args.viscosity, r, dx)
        raw.append(res)
        ladder.append(ev)
        print(f"[startup-mirror]   history max {ev['history_max_err_pct']:+.2f}% "
              f"rms {ev['history_rms_err_pct']:.2f}%  "
              f"profile max {ev['profile_rms_max_pct']:.2f}%  "
              f"nu_amp/nu {ev['nu_amp_over_nu']:.4f}  "
              f"nu_rate/nu {ev['nu_rate_over_nu']:.4f}  "
              f"({ev['wall_clock_s']:.1f}s)", flush=True)

    control = None
    wallless = None
    if not args.skip_controls:
        r0 = args.resolutions[0]
        dx = args.height / r0
        cfg = make_config(dx, args.viscosity, 1.0, device)
        print("[startup-mirror] free-slip control (exact u = g t) ...", flush=True)
        res = run_transient(cfg, args.height, args.columns * dx, args.body_force,
                            t_end, device, slip="free_slip",
                            viscosity=args.viscosity)
        t, u = res["t"], smooth(res["t"], res["u_centre"], SMOOTH_HALF_WIDTH * tau)
        ctrl = {"t_over_tau": (t / tau).tolist(), "u_centre": u.tolist(),
                "u_rigid_body": (args.body_force * t).tolist()}
        errs = []
        for tf in (1.0, 2.0, 4.0, 8.0):
            i = int(np.argmin(np.abs(t - tf * tau)))
            if abs(t[i] - tf * tau) > 0.2 * tau:
                continue
            errs.append({"t_over_tau": float(t[i] / tau),
                         "u_meas": float(u[i]),
                         "u_exact": float(args.body_force * t[i]),
                         "err_pct": float(100.0 * (u[i] / (args.body_force * t[i]) - 1.0))})
        ctrl["checks"] = errs
        ctrl["max_rigid_body_err_pct"] = max((abs(e["err_pct"]) for e in errs),
                                             default=float("nan"))
        ctrl["wall_clock_s"] = res["wall_clock_s"]
        control = ctrl
        print(f"[startup-mirror]   rigid-body max err "
              f"{ctrl['max_rigid_body_err_pct']:.2f}%", flush=True)

        print("[startup-mirror] wall-less trap (ghosts disabled) ...", flush=True)
        res_wl = run_transient(cfg, args.height, args.columns * dx,
                               args.body_force, t_end, device, wallless=True,
                               viscosity=args.viscosity)
        ev_wl = evaluate(res_wl, args.height, args.body_force, args.viscosity,
                         r0, dx)
        wallless = {k: ev_wl[k] for k in
                    ("H_over_dx", "wallless", "n_ghost", "history_max_err_pct",
                     "history_rms_err_pct", "profile_rms_max_pct",
                     "u_centre_late_over_exact", "wall_clock_s")}
        wallless["t"] = ev_wl["t"]
        wallless["u_centre"] = ev_wl["u_centre_smoothed"]
        wallless["u_exact_centre"] = ev_wl["u_exact_centre"]
        print(f"[startup-mirror]   wall-less history max err "
              f"{wallless['history_max_err_pct']:.1f}% (must be >> 5%)",
              flush=True)

    g = gates(ladder, control, wallless) if control else {}
    verdict = {
        "gates": g,
        "all_pass": bool(g) and all(g.values()),
        "honest_fail": [k for k, v in g.items() if not v],
    }

    payload = {
        "case": "plane_poiseuille_startup_mirror_wall",
        "title": "Poiseuille start-up from rest vs the Boussinesq series, "
                 "mirror/ghost no-slip walls",
        "anchor": "CASE-POISEUILLE-SPH",
        "config": {
            "height_m": args.height,
            "viscosity_m2_s": args.viscosity,
            "body_force_m_s2": args.body_force,
            "columns": args.columns,
            "resolutions": args.resolutions,
            "t_end_tau": args.t_end_tau,
            "t_end_s": t_end,
            "tau_s": tau,
            "u_max_inf": args.body_force * args.height ** 2 /
                         (8.0 * args.viscosity),
            "profile_times_tau": list(PROFILE_TIMES),
            "smoothing_half_width_tau": SMOOTH_HALF_WIDTH,
        },
        "reference": "Boussinesq (1868) start-up series; Batchelor §4.2; "
                     "Schlichting ch. III; Morris, Fox & Zhu (1997) JCP 136:214",
        "reference_provenance": prov,
        "gates_predeclared": GATES,
        "ladder": ladder,
        "free_slip_control": control,
        "wallless_trap": wallless,
        "verdict": verdict,
        "wall_clock_s": sum(r["wall_clock_s"] for r in ladder)
                        + (control["wall_clock_s"] if control else 0.0)
                        + (wallless["wall_clock_s"] if wallless else 0.0),
        "device": str(device),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[startup-mirror] wrote {args.out / 'results.json'}")
    for k, v in g.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())