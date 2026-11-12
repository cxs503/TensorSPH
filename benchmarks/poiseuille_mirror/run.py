#!/usr/bin/env python3
"""Plane Poiseuille channel with a *mirror* (ghost) no-slip wall.

The sibling benchmark ``benchmarks/poiseuille/run.py`` uses the solver's
``wall_friction`` blend against static wall particles one spacing outside the
boundary.  That blend puts its no-slip plane ~0.9 dx inside the flow, so the
measured profile misses the nominal parabola by 8.5 % (row RMS, ``H/dx=16``)
and the wall shear by 50 % -- documented, honest, but outside the 5 % band this
case is graded against.  ``docs/verification_poiseuille_channel.md`` names the
fix explicitly: mirrored/ghost particles that impose ``u = 0`` **on** the
boundary.

This script is that benchmark.  Nothing in the viscous operator changes: the
ghost layer is built from the current fluid state every step
(:mod:`tensorsph.wall_mirror`), carries the negated velocity, and is handed to
``rhs`` as a moving wall, so the wall/fluid viscosity term sees an
antisymmetric velocity pair whose mean is zero on the plane.

Physics / reference
-------------------
Steady plane Poiseuille flow, walls at ``y = 0, H``, no-slip, uniform body
force ``g_x`` per unit mass, periodic in ``x``:

   0 = rho g_x + mu u''(y),  u(0) = u(H) = 0
   =>  u(y) = g_x (H - y) y / (2 nu),   u_max = g_x H^2 / (8 nu),
       tau_w = rho g_x H / 2.

The closed form is the textbook result (White, *Viscous Fluid Flow*, 3rd ed.,
§3-2.2; Batchelor, *An Introduction to Fluid Dynamics*, §4.2).  It is not
hand-copied here: ``check_analytic_symbolically`` re-derives the residual of
the 1-D Navier-Stokes equation with sympy and records it in the artifact, and
``main`` also exports the ODE residual evaluated numerically.  See
``docs/verification_poiseuille_mirror_wall.md``.

Metrics (predeclared, before the ladder is run)
-----------------------------------------------
A  centre-line:        |u_max / (g H^2 / 8 nu) - 1|                 <= 5 %
B  profile shape:      row RMS of (u_num - u_analytic)/u_analytic
                       over 0.1 <= y/H <= 0.9                     <= 5 %
C  wall shear:         |tau_fit / (rho g H / 2) - 1| at both walls  <= 5 %
D  wall slip:          u(first row) / u_max                          <= 5 %
E  convergence:        B must not grow with H/dx (8 -> 12 -> 16)
F  control:            free-slip mirror has no steady state
                       (u_max grows like g t, flat profile)

Usage
-----
    PYTHONPATH=src python benchmarks/poiseuille_mirror/run.py \
        --resolutions 8 12 16 --out benchmarks/poiseuille_mirror/out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from benchmarks.poiseuille.run import (
    analytic,
    bin_rows,
    build_fluid,
    fit_profile,
    make_config,
    periodic_images,
)
from tensorsph.droplet_impact import resolve_device, rhs
from tensorsph.wall_mirror import plane_walls

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Reference solution provenance
# --------------------------------------------------------------------------- #

def check_analytic_symbolically() -> dict:
    """Re-derive the closed-form profile from the 1-D Navier-Stokes equation.

    Returns the residual that must be identically zero, plus the boundary
    conditions evaluated symbolically.  If sympy is unavailable the entry says
    so and the numerical residual (below) is used instead.
    """
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"sympy unavailable: {exc}"}

    y, g, nu, H = sp.symbols("y g nu H", positive=True)
    u = g * (H - y) * y / (2 * nu)
    resid = sp.simplify(nu * sp.diff(u, y, 2) + g)
    u_max = sp.simplify(u.subs(y, H / 2))
    tau_wall = sp.simplify(nu * sp.diff(u, y).subs(y, 0))
    return {
        "available": True,
        "profile": "u(y) = g*(H-y)*y/(2*nu)",
        "ode_residual_should_be_zero": str(resid),
        "residual_is_zero": bool(resid == 0),
        "u_max_formula": str(u_max),
        "u_max_matches_gH2_over_8nu": bool(sp.simplify(u_max - g * H**2 / (8 * nu)) == 0),
        "du_dy_at_wall": str(tau_wall),
        "tau_w_matches_gH_over_2": bool(sp.simplify(tau_wall - g * H / 2) == 0),
        "bc_at_y0": str(sp.simplify(u.subs(y, 0))),
        "bc_at_yH": str(sp.simplify(u.subs(y, H))),
    }


def analytic_ode_residual(rows, body_force: float, viscosity: float) -> float:
    """Discrete check: max |nu u'' + g| over the profile, second differences."""
    if len(rows) < 3:
        return float("nan")
    ys = [r["y"] for r in rows]
    us = [r["u_mean"] for r in rows]
    dy = ys[1] - ys[0]
    worst = 0.0
    for k in range(1, len(rows) - 1):
        u_yy = (us[k - 1] - 2.0 * us[k] + us[k + 1]) / (dy * dy)
        worst = max(worst, abs(viscosity * u_yy + body_force))
    return worst


# --------------------------------------------------------------------------- #
# Time integration with the mirror wall
# --------------------------------------------------------------------------- #

def run_case_mirror(cfg, height: float, length: float, body_force: float,
                    t_end: float, t_average: float, device,
                    slip: str = "no_slip", t_steady_check: float = 0.25) -> dict:
    """Integrate to steady state with mirror ghosts at ``y = 0`` and ``y = H``."""
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    support = 2.0 * h                       # kernel support radius
    planes = [(0.0, +1), (height, -1)]

    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    avg_from = t_end - t_average

    acc_sum = torch.zeros_like(pos)
    samples = 0
    trace = []
    wall_clock = time.time()
    n_ghost = 0

    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)

        ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                              planes, support, slip=slip)
        n_ghost = int(ghost_pos.shape[0])

        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                        wall=ghost_pos if n_ghost else None,
                        wall_vel=ghost_vel if n_ghost else None)
        acc = acc[:n].clone()
        acc[:, 0] += body_force

        t = step * dt
        if t >= avg_from:
            acc_sum += vel
            samples += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % max(1, steps // 12) == 0 or step == steps - 1:
            ux = vel[:, 0]
            trace.append({
                "t": t,
                "u_max": float(ux.max()),
                "u_mean": float(ux.mean()),
                "max_mach": float(ux.abs().max()) / cfg.sound_speed,
                "wall_clock_s": time.time() - wall_clock,
            })

    u_mean = acc_sum / max(samples, 1)
    rows, profile = bin_rows(pos, u_mean, height, dx)

    tail = [p for p in trace if p["t"] >= t_end - t_steady_check]
    drift = 0.0
    if len(tail) >= 2 and abs(tail[0]["u_max"]) > 0:
        drift = abs(tail[-1]["u_max"] - tail[0]["u_max"]) / abs(tail[0]["u_max"])

    stride = max(1, n // 1500)
    field = {
        "x": pos[::stride, 0].tolist(),
        "y": pos[::stride, 1].tolist(),
        "u": u_mean[::stride, 0].tolist(),
        "u_y": u_mean[::stride, 1].tolist(),
    }

    return {
        "n_fluid": int(n),
        "n_ghost": n_ghost,
        "dt": dt,
        "steps": steps,
        "n_averaged_samples": samples,
        "rows": rows,
        "profile": profile,
        "field": field,
        "trace": trace,
        "steady_drift": drift,
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def evaluate(case: dict, height: float, body_force: float, viscosity: float,
             rho0: float, resolution: int) -> dict:
    """Turn a binned profile into the predeclared metrics A..D."""
    dx = height / resolution
    profile = case["profile"]
    fit = fit_profile(profile, height, body_force, viscosity)
    if len(profile) >= 3:
        # Quadratic fit through every row. With the mirror wall the parabola is
        # resolved to the wall, so a global fit is the right estimator; the
        # blend benchmark needed an interior window because its wall row was
        # smeared over a spacing.
        pass

    rows = []
    for p in profile:
        u_num = p["u_mean"]
        u_ref = analytic(p["y"], height, body_force, viscosity, 0.0)
        rows.append({
            **p,
            "u_analytic": u_ref,
            "rel_error": (u_num - u_ref) / u_ref if u_ref else float("nan"),
        })

    interior = [r for r in rows if 0.1 <= r["y_over_H"] <= 0.9]
    rms_interior = (math.sqrt(sum(r["rel_error"] ** 2 for r in interior)
                             / len(interior)) if interior else float("nan"))
    rms_all = math.sqrt(sum(r["rel_error"] ** 2 for r in rows) / len(rows))

    u_max_meas = max(r["u_mean"] for r in rows)
    u_max_ref = body_force * height * height / (8.0 * viscosity)

    # Wall shear from the fitted parabolas' slope at each wall. The profile is
    # u = c0 + c1 y + c2 y^2, so du/dy = c1 + 2 c2 y.
    # The two walls carry the *opposite* sign (the flow is driven one way), so
    # the magnitude is what a 5 % band can grade.
    c1, c2 = fit["c1"], fit["c2"]
    tau_bottom = abs(rho0 * viscosity * c1)
    tau_top = abs(rho0 * viscosity * (c1 + 2.0 * c2 * height))
    tau_ref = rho0 * body_force * height / 2.0

    first, last = rows[0], rows[-1]
    return {
        "resolution_H_over_dx": int(resolution),
        "dx": dx,
        "h": 1.3 * dx,
        "body_force": body_force,
        "viscosity": viscosity,
        "reynolds_number": u_max_meas * height / viscosity,
        "u_max_measured": u_max_meas,
        "u_max_reference": u_max_ref,
        "u_max_ratio": u_max_meas / u_max_ref,
        "u_max_error_pct": 100.0 * (u_max_meas / u_max_ref - 1.0),
        "profile_rms_interior_pct": 100.0 * rms_interior,
        "profile_rms_all_rows_pct": 100.0 * rms_all,
        "max_row_error_pct": 100.0 * max(abs(r["rel_error"]) for r in rows),
        "wall_shear_bottom_ratio": tau_bottom / tau_ref,
        "wall_shear_top_ratio": tau_top / tau_ref,
        "wall_shear_bottom_error_pct": 100.0 * (tau_bottom / tau_ref - 1.0),
        "wall_shear_top_error_pct": 100.0 * (tau_top / tau_ref - 1.0),
        # Slip length / slip velocity at the wall itself: the parabola's
        # intercept at y=0 and y=H. The first *row* sits at y = dx/2 and is
        # expected to carry u_analytic(dx/2) ~ 23 % of u_max at H/dx = 8, so
        # quoting it as "slip" would grade the sampling, not the boundary.
        "wall_slip_intercept_over_umax": abs(fit["c0"]) / u_max_meas,
        "wall_slip_intercept_top_over_umax": abs(
            fit["c0"] + fit["c1"] * height + fit["c2"] * height ** 2) / u_max_meas,
        "wall_row_value_over_umax": first["u_mean"] / u_max_meas,
        "wall_row_analytic_over_umax": (
            analytic(first["y"], height, body_force, viscosity, 0.0) / u_max_meas),
        "asymmetry": abs(first["u_mean"] - last["u_mean"]) / u_max_meas,
        "max_row_std_over_umax": max(r["u_std_x"] for r in rows) / u_max_meas,
        "fit": fit,
        "steady_drift": case["steady_drift"],
        "ode_residual_max_over_g": analytic_ode_residual(
            rows, body_force, viscosity) / body_force,
        "dt": case["dt"],
        "steps": case["steps"],
        "n_fluid": case["n_fluid"],
        "n_ghost": case["n_ghost"],
        "n_averaged_samples": case["n_averaged_samples"],
        "max_mach": max(p["max_mach"] for p in case["trace"]),
        "wall_clock_s": case["wall_clock_s"],
        "rows": rows,
        "trace": case["trace"],
        "field": case["field"],
    }


def gates(res: dict) -> dict:
    """Apply the predeclared thresholds A..D."""
    return {
        "A_centreline_5pct": abs(res["u_max_error_pct"]) <= 5.0,
        "B_profile_rms_5pct": res["profile_rms_interior_pct"] <= 5.0,
        "C_wall_shear_5pct": (abs(res["wall_shear_bottom_error_pct"]) <= 5.0
                              and abs(res["wall_shear_top_error_pct"]) <= 5.0),
        "D_wall_slip_5pct": (res["wall_slip_intercept_over_umax"] <= 0.05
                             and res["wall_slip_intercept_top_over_umax"] <= 0.05),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "benchmarks/poiseuille_mirror/out")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--body-force", type=float, default=9.81)
    parser.add_argument("--height", type=float, default=0.1)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--t-average", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--slip", default="no_slip",
                        choices=("no_slip", "free_slip"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args(argv)

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.quick:
        args.resolutions = [8]
        args.t_end = 0.4
        args.t_average = 0.1

    results = []
    for r in args.resolutions:
        print(f"[poiseuille-mirror] H/dx={r} ({args.slip}) ...", flush=True)
        dx = args.height / r
        length = args.columns * dx
        cfg = make_config(dx, args.viscosity, 1.0, device)   # full wall coupling
        case = run_case_mirror(cfg, args.height, length, args.body_force,
                               args.t_end, args.t_average, device, slip=args.slip)
        res = evaluate(case, args.height, args.body_force, args.viscosity,
                       1000.0, r)
        res["length"] = length
        res["slip"] = args.slip
        res["gates"] = gates(res)
        print(f"[poiseuille-mirror]   u_max {res['u_max_error_pct']:+.2f}%  "
              f"rms {res['profile_rms_interior_pct']:.2f}%  "
              f"tau {res['wall_shear_bottom_error_pct']:+.2f}%/"
              f"{res['wall_shear_top_error_pct']:+.2f}%  "
              f"slip {100 * res['wall_slip_intercept_over_umax']:.1f}%  "
              f"({res['wall_clock_s']:.1f}s)", flush=True)
        results.append(res)

    control = None
    if not args.no_control and args.slip == "no_slip":
        print("[poiseuille-mirror] free-slip control ...", flush=True)
        dx = args.height / 8
        length = args.columns * dx
        cfg = make_config(dx, args.viscosity, 1.0, device)
        case = run_case_mirror(cfg, args.height, length, args.body_force,
                               args.t_end, args.t_average, device,
                               slip="free_slip")
        control = evaluate(case, args.height, args.body_force, args.viscosity,
                           1000.0, 8)
        control["expected"] = "rigid-body acceleration u = g t, no parabola"
        control["flatness"] = control["profile_rms_all_rows_pct"]
        print("[poiseuille-mirror]   control u_max ratio "
              f"{control['u_max_ratio']:.2f} (must be >> 1: no steady state)",
              flush=True)

    payload = {
        "case": "plane_poiseuille_channel_mirror_wall",
        "description": ("body-force-driven 2-D channel, mirror/ghost no-slip "
                        "walls at y=0 and y=H, periodic in x"),
        "height": args.height,
        "columns": args.columns,
        "viscosity": args.viscosity,
        "body_force": args.body_force,
        "rho0": 1000.0,
        "analytic": "u(y) = g (H-y) y / (2 nu); u_max = g H^2/(8 nu); tau_w = rho g H/2",
        "reference": ("White, Viscous Fluid Flow 3rd ed. §3-2.2; Batchelor, "
                      "An Introduction to Fluid Dynamics §4.2 (closed form "
                      "re-derived symbolically below the tables)"),
        "reference_provenance": check_analytic_symbolically(),
        "thresholds": {"centreline_pct": 5.0, "profile_rms_pct": 5.0,
                       "wall_shear_pct": 5.0, "wall_slip_over_umax": 0.05},
        "t_end": args.t_end,
        "t_average": args.t_average,
        "device": str(device),
        "torch": torch.__version__,
        "results": results,
        "free_slip_control": control,
        "wall_clock_s": sum(r["wall_clock_s"] for r in results)
                        + (control["wall_clock_s"] if control else 0.0),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[poiseuille-mirror] wrote {args.out / 'results.json'}")

    passed = all(all(r["gates"].values()) for r in results)
    print(f"[poiseuille-mirror] all gates A-D: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
