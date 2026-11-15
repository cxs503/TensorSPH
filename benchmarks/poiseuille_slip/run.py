#!/usr/bin/env python3
"""Plane Poiseuille channel with a *Navier partial-slip* wall.

Why this case
-------------
``benchmarks/poiseuille_mirror`` verifies the two *limits* of the wall
condition family: ``no_slip`` (ghost = -v, slip length 0) and ``free_slip``
(ghost tangential = v, slip length infinite, no steady state).  Real walls sit
between the two: superhydrophobic / gas-cushion / SLIPS surfaces, and rough or
partially wetted surfaces, are modelled by a Navier slip length ``lam``
(``u_wall = lam * du/dn``).  Those numbers are ship-relevant -- the slip length
is what translates a coating into a drag/flux benefit -- and the case pool had
no way to check a solver against them.

This benchmark is the one-parameter analytic rung between the two limits.  The
solver side is ``tensorsph.wall_mirror``'s ``slip="navier"`` mode (this
delivery adds it): the ghost tangential velocity is
``w + alpha (v - w)`` with ``alpha = (lam - d)/(lam + d)``, ``d`` the particle
distance to the plane, which reduces exactly to ``-v`` at ``lam = 0`` and to
``v`` as ``lam -> inf``.

Physics / reference
-------------------
Steady plane Poiseuille flow with body force ``g`` per unit mass, walls at
``y = 0, H``, Navier slip on both walls, periodic in ``x``::

    0 = rho g + mu u''(y),   u(0) = lam u'(0),   u(H) = -lam u'(H)

    =>  u(y) = g (H y - y^2 + lam H) / (2 nu)
        u_max = (g H^2 / (8 nu)) (1 + 4 lam/H)          [at y = H/2, any lam]
        U_slip = u(0) = g lam H / (2 nu)                [= 4 (lam/H) u_max,0]
        tau_w  = rho g H / 2                            [unchanged by slip]
        Q      = (g H^3 / (12 nu)) (1 + 6 lam/H)

Two consequences that make this a real test rather than a tautology:

* the *shear* is independent of slip (``C1 = g H / (2 nu)`` for every
  ``lam``), so the momentum budget ``tau_w = rho g H / 2`` must still close --
  a slip rule that leaks or injects momentum at the wall fails it;
* the *flux enhancement* ``Q/Q_noslip = 1 + 6 lam/H`` is a global functional
  of the profile, not a local wall value.

The closed form is the textbook Navier-slip Poiseuille result (Batchelor,
*An Introduction to Fluid Dynamics*, §4.2 for the no-slip member; the slip
member is the classical Navier/Beavers-Joseph channel limit used in
superhydrophobic drag-reduction literature).  It is re-derived symbolically
with sympy in ``check_analytic_symbolically`` and recorded in the artifact.

Metrics (predeclared before any run)
------------------------------------
A  centre line:     |u_max / u_max,ref - 1|                          <= 5 %
B  profile shape:   row RMS of (u_num - u_analytic)/u_analytic,
                    0.1 <= y/H <= 0.9                                 <= 5 %
C  wall shear:      |tau_fit / (rho g H / 2) - 1|, both walls         <= 5 %
D  wall slip:       |U_slip,fit / U_slip,ref - 1|                     <= 5 %
E  flux:            |(Q_num/Q_noslip)/(1 + 6 lam/H) - 1|              <= 5 %
F  family limit:    the lam = 0 member reproduces the verified no-slip
                    result (gate A at lam = 0)                        <= 5 %
G  hold-out:        lam/H = 0.375 has no special standing in the rule; it is
                    graded by the same closed form, at the same 5 %

Usage
-----
    PYTHONPATH=src python benchmarks/poiseuille_slip/run.py \
        --resolutions 12 16 24 --slip-ratios 0 0.125 0.25 0.5 0.375 \
        --out benchmarks/poiseuille_slip/out
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
# Reference solution
# --------------------------------------------------------------------------- #

def analytic_slip(y: float, height: float, body_force: float, viscosity: float,
                  slip_length: float) -> float:
    """u(y) = g (H y - y^2 + lam H) / (2 nu)."""
    return (body_force * (height * y - y * y + slip_length * height)
            / (2.0 * viscosity))


def u_max_ref(height: float, body_force: float, viscosity: float,
              slip_length: float) -> float:
    """(g H^2 / 8 nu) (1 + 4 lam/H) -- the maximum, still at y = H/2."""
    return (body_force * height * height / (8.0 * viscosity)) * (
        1.0 + 4.0 * slip_length / height)


def slip_ref(height: float, body_force: float, viscosity: float,
             slip_length: float) -> float:
    """Wall slip velocity u(0) = g lam H / (2 nu)."""
    return body_force * slip_length * height / (2.0 * viscosity)


def flux_ref(height: float, body_force: float, viscosity: float,
             slip_length: float) -> float:
    """Q = (g H^3 / 12 nu)(1 + 6 lam/H)."""
    return (body_force * height ** 3 / (12.0 * viscosity)) * (
        1.0 + 6.0 * slip_length / height)


def check_analytic_symbolically() -> dict:
    """Re-derive the slip profile from the ODE *and both slip BCs* with sympy."""
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"sympy unavailable: {exc}"}

    y, g, nu, H, lam = sp.symbols("y g nu H lam", positive=True)
    u = g * (H * y - y * y + lam * H) / (2 * nu)
    resid = sp.simplify(nu * sp.diff(u, y, 2) + g)
    du = sp.diff(u, y)
    bc0 = sp.simplify(u.subs(y, 0) - lam * du.subs(y, 0))
    bcH = sp.simplify(u.subs(y, H) + lam * du.subs(y, H))
    centre = sp.simplify(du.subs(y, H / 2))
    u_max = sp.simplify(u.subs(y, H / 2))
    tau0 = sp.simplify(nu * du.subs(y, 0))
    flux = sp.simplify(sp.integrate(u, (y, 0, H)))
    return {
        "available": True,
        "profile": "u(y) = g*(H*y - y^2 + lam*H)/(2*nu)",
        "ode_residual_is_zero": bool(resid == 0),
        "bc_y0_residual_is_zero": bool(bc0 == 0),
        "bc_yH_residual_is_zero": bool(bcH == 0),
        "centre_gradient_is_zero_any_lam": bool(centre == 0),
        "u_max_minus_closed_form_is_zero": bool(sp.simplify(
            u_max - g * H ** 2 / (8 * nu) * (1 + 4 * lam / H)) == 0),
        "tau_w_minus_gH_over_2_is_zero": bool(sp.simplify(
            tau0 - g * H / 2) == 0),
        "flux_minus_closed_form_is_zero": bool(sp.simplify(
            flux - g * H ** 3 / (12 * nu) * (1 + 6 * lam / H)) == 0),
        "noslip_limit_is_gH2_over_8nu": bool(sp.simplify(
            u_max.subs(lam, 0) - g * H ** 2 / (8 * nu)) == 0),
    }


# --------------------------------------------------------------------------- #
# Solver driver
# --------------------------------------------------------------------------- #

def run_case_slip(cfg, height: float, length: float, body_force: float,
                  t_end: float, t_average: float, device,
                  slip_length: float, t_steady_check: float = 0.25) -> dict:
    """Integrate to steady state with Navier-slip mirror ghosts at y=0 and y=H."""
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

        ghost_pos, ghost_vel, _ = plane_walls(
            pos_ext, vel_ext, mass_ext, planes, support,
            slip="navier" if slip_length > 0.0 else "no_slip",
            slip_length=slip_length if slip_length > 0.0 else None)
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

    # Wall-layer health: the unshifted (Lagrangian) wall layer clusters slowly,
    # so a long run can walk a particle onto / across the plane.  Recorded, not
    # hidden: a crossed particle leaves the mirror support and becomes a
    # ballistic outlier, which would poison the binned profile.
    n_outside = int(((pos[:, 1] < 0.0) | (pos[:, 1] > height)).sum())
    speed = vel.norm(dim=1)
    k = int(speed.argmax())

    stride = max(1, n // 2000)
    field = {
        "x": pos[::stride, 0].tolist(),
        "y": pos[::stride, 1].tolist(),
        "u": u_mean[::stride, 0].tolist(),
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
        "y_min": float(pos[:, 1].min()),
        "y_max": float(pos[:, 1].max()),
        "n_outside_channel": n_outside,
        "max_particle_speed": float(speed[k]),
        "y_at_max_particle_speed": float(pos[k, 1]),
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def evaluate(case: dict, height: float, body_force: float, viscosity: float,
             rho0: float, resolution: int, slip_length: float) -> dict:
    """Turn a binned profile into the predeclared metrics A..E."""
    profile = case["profile"]
    fit = fit_profile(profile, height, body_force, viscosity)

    rows = []
    for p in profile:
        u_ref = analytic_slip(p["y"], height, body_force, viscosity,
                              slip_length)
        rows.append({
            **p,
            "u_analytic": u_ref,
            "rel_error": (p["u_mean"] - u_ref) / u_ref if u_ref else float("nan"),
        })

    interior = [r for r in rows if 0.1 <= r["y_over_H"] <= 0.9]
    rms_interior = math.sqrt(sum(r["rel_error"] ** 2 for r in interior)
                             / len(interior)) if interior else float("nan")
    rms_all = math.sqrt(sum(r["rel_error"] ** 2 for r in rows) / len(rows))

    u_max_meas = max(r["u_mean"] for r in rows)
    u_max_r = u_max_ref(height, body_force, viscosity, slip_length)

    # Wall shear from the fitted parabola's slope at each wall.
    c0, c1, c2 = fit["c0"], fit["c1"], fit["c2"]
    tau_bottom = abs(rho0 * viscosity * c1)
    tau_top = abs(rho0 * viscosity * (c1 + 2.0 * c2 * height))
    tau_r = rho0 * body_force * height / 2.0

    # Slip velocity: intercept of the fitted parabola at each wall plane. For
    # the analytic slip solution the parabola is u0(y) + U_slip, so *both*
    # intercepts equal +U_slip (the flow is driven one way at both walls; only
    # the shear changes sign).
    u_slip_bottom = c0
    u_slip_top = c0 + c1 * height + c2 * height * height
    u_slip_r = slip_ref(height, body_force, viscosity, slip_length)

    # Flux. The row trapezoid misses the two wall slivers, and with slip the
    # near-wall velocity is *not* small there, so the truncation is O(lam dx)
    # and grows with the slip length -- it would masquerade as a slip error.
    # The flux is therefore integrated from the fitted parabola over the whole
    # channel, which is the same profile model the other metrics use; the raw
    # row trapezoid is kept alongside as a diagnostic.
    q_fit = c0 * height + 0.5 * c1 * height ** 2 + c2 * height ** 3 / 3.0
    us = [r["u_mean"] for r in rows]
    ys = [r["y"] for r in rows]
    q_trapz = (float(torch.trapezoid(torch.tensor(us, dtype=torch.float64),
                                    torch.tensor(ys, dtype=torch.float64)))
               if len(ys) > 1 else float("nan"))
    q_meas = q_fit
    q_r = flux_ref(height, body_force, viscosity, slip_length)
    q_noslip = flux_ref(height, body_force, viscosity, 0.0)

    return {
        "resolution_H_over_dx": int(resolution),
        "slip_length": slip_length,
        "slip_ratio_lam_over_H": slip_length / height,
        "slip_over_dx": slip_length / (height / resolution),
        "dx": height / resolution,
        "body_force": body_force,
        "viscosity": viscosity,
        "reynolds_number": u_max_meas * height / viscosity,
        "u_max_measured": u_max_meas,
        "u_max_reference": u_max_r,
        "u_max_error_pct": 100.0 * (u_max_meas / u_max_r - 1.0),
        "profile_rms_interior_pct": 100.0 * rms_interior,
        "profile_rms_all_rows_pct": 100.0 * rms_all,
        "max_row_error_pct": 100.0 * max(abs(r["rel_error"]) for r in rows),
        "wall_shear_bottom_ratio": tau_bottom / tau_r,
        "wall_shear_top_ratio": tau_top / tau_r,
        "wall_shear_bottom_error_pct": 100.0 * (tau_bottom / tau_r - 1.0),
        "wall_shear_top_error_pct": 100.0 * (tau_top / tau_r - 1.0),
        "u_slip_measured_bottom": u_slip_bottom,
        "u_slip_measured_top": u_slip_top,
        "u_slip_reference": u_slip_r,
        "u_slip_error_pct": (100.0 * (u_slip_bottom / u_slip_r - 1.0)
                             if u_slip_r else 0.0),
        "u_slip_top_error_pct": (100.0 * (u_slip_top / u_slip_r - 1.0)
                                 if u_slip_r else 0.0),
        "flux_measured": q_meas,
        "flux_measured_row_trapezoid": q_trapz,
        "flux_reference": q_r,
        "flux_error_pct": 100.0 * (q_meas / q_r - 1.0),
        "flux_enhancement_measured": q_meas / q_noslip if q_noslip else float("nan"),
        "flux_enhancement_reference": 1.0 + 6.0 * slip_length / height,
        "flux_enhancement_error_pct": 100.0 * (
            (q_meas / q_noslip) / (1.0 + 6.0 * slip_length / height) - 1.0),
        "fit": fit,
        "steady_drift": case["steady_drift"],
        "dt": case["dt"],
        "steps": case["steps"],
        "n_fluid": case["n_fluid"],
        "n_ghost": case["n_ghost"],
        "n_averaged_samples": case["n_averaged_samples"],
        "max_mach": max(p["max_mach"] for p in case["trace"]),
        "wall_clock_s": case["wall_clock_s"],
        "y_min": case["y_min"],
        "y_max": case["y_max"],
        "n_outside_channel": case["n_outside_channel"],
        "max_particle_speed": case["max_particle_speed"],
        "y_at_max_particle_speed": case["y_at_max_particle_speed"],
        "rows": rows,
        "trace": case["trace"],
        "field": case["field"],
    }


def gates(res: dict) -> dict:
    """Apply the predeclared thresholds A..F to one run."""
    g = {
        "A_centreline_5pct": abs(res["u_max_error_pct"]) <= 5.0,
        "B_profile_rms_5pct": res["profile_rms_interior_pct"] <= 5.0,
        "C_wall_shear_5pct": (abs(res["wall_shear_bottom_error_pct"]) <= 5.0
                              and abs(res["wall_shear_top_error_pct"]) <= 5.0),
        "E_flux_enhancement_5pct": abs(res["flux_enhancement_error_pct"]) <= 5.0,
    }
    # D: the slip velocity itself.  It is identically zero in the no-slip
    # member, where the gate degenerates to "fitted intercept ~ 0", graded
    # against u_max instead of a vanishing reference.
    if res["slip_length"] > 0.0:
        g["D_wall_slip_5pct"] = (abs(res["u_slip_error_pct"]) <= 5.0
                                 and abs(res["u_slip_top_error_pct"]) <= 5.0)
    else:
        g["D_wall_slip_5pct"] = (
            abs(res["fit"]["c0"]) / res["u_max_measured"] <= 0.05)
    return g


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "benchmarks/poiseuille_slip/out")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[8, 12, 16, 24])
    parser.add_argument("--slip-ratios", type=float, nargs="+",
                        default=[0.0, 0.125, 0.25, 0.5, 0.375])
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--body-force", type=float, default=9.81)
    parser.add_argument("--height", type=float, default=0.1)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--t-average", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.quick:
        args.resolutions = [12]
        args.slip_ratios = [0.0, 0.25]
        args.t_end = 0.5
        args.t_average = 0.15

    results = []
    for lam_over_h in args.slip_ratios:
        lam = lam_over_h * args.height
        for r in args.resolutions:
            print(f"[poiseuille-slip] H/dx={r} lam/H={lam_over_h:g} "
                  f"(lam/dx={lam / (args.height / r):.2f}) ...", flush=True)
            dx = args.height / r
            length = args.columns * dx
            cfg = make_config(dx, args.viscosity, 1.0, device)
            case = run_case_slip(cfg, args.height, length, args.body_force,
                                 args.t_end, args.t_average, device, lam)
            res = evaluate(case, args.height, args.body_force, args.viscosity,
                           1000.0, r, lam)
            res["gates"] = gates(res)
            print(f"[poiseuille-slip]   u_max {res['u_max_error_pct']:+.2f}%  "
                  f"rms {res['profile_rms_interior_pct']:.2f}%  "
                  f"tau {res['wall_shear_bottom_error_pct']:+.2f}%  "
                  f"slip {res['u_slip_error_pct']:+.2f}%  "
                  f"flux+ {res['flux_enhancement_error_pct']:+.2f}%  "
                  f"({res['wall_clock_s']:.1f}s)", flush=True)
            results.append(res)

    payload = {
        "case": "plane_poiseuille_navier_slip",
        "wall_layer_health": [
            {"resolution_H_over_dx": r["resolution_H_over_dx"],
             "slip_ratio_lam_over_H": r["slip_ratio_lam_over_H"],
             "y_min_over_dx": r["y_min"] / r["dx"],
             "y_max_over_H": r["y_max"] / args.height,
             "n_outside_channel": r["n_outside_channel"],
             "max_particle_speed": r["max_particle_speed"],
             "y_at_max_particle_speed": r["y_at_max_particle_speed"],
             "steady_drift_last_quarter": r["steady_drift"]}
            for r in results],
        "description": ("body-force-driven 2-D channel with Navier partial-slip "
                        "mirror/ghost walls (slip length lam) at y=0 and y=H, "
                        "periodic in x"),
        "height": args.height,
        "columns": args.columns,
        "viscosity": args.viscosity,
        "body_force": args.body_force,
        "rho0": 1000.0,
        "analytic": ("u(y)=g(H y - y^2 + lam H)/(2 nu); "
                     "u_max=(gH^2/8nu)(1+4 lam/H); U_slip=g lam H/(2 nu); "
                     "tau_w=rho g H/2; Q=(gH^3/12nu)(1+6 lam/H)"),
        "reference": ("Batchelor, An Introduction to Fluid Dynamics §4.2 "
                      "(no-slip member); Navier-slip channel limit as used for "
                      "superhydrophobic drag reduction. Closed form re-derived "
                      "symbolically in reference_provenance."),
        "reference_provenance": check_analytic_symbolically(),
        "thresholds": {"centreline_pct": 5.0, "profile_rms_pct": 5.0,
                       "wall_shear_pct": 5.0, "wall_slip_pct": 5.0,
                       "flux_enhancement_pct": 5.0},
        "solver_slip_rule": ("ghost tangential = w + alpha (v - w), "
                             "alpha = (lam - d)/(lam + d); lam=0 -> no_slip, "
                             "lam->inf -> free_slip"),
        "t_end": args.t_end,
        "t_average": args.t_average,
        "device": str(device),
        "torch": torch.__version__,
        "results": results,
        "wall_clock_s": sum(r["wall_clock_s"] for r in results),
    }

    by_res = {}
    for res in results:
        by_res.setdefault(str(res["resolution_H_over_dx"]), []).append(
            {k: res[k] for k in (
                "slip_ratio_lam_over_H", "slip_over_dx", "u_max_error_pct",
                "profile_rms_interior_pct", "wall_shear_bottom_error_pct",
                "u_slip_error_pct", "flux_enhancement_measured",
                "flux_enhancement_error_pct", "max_mach", "wall_clock_s")})
    payload["by_resolution"] = by_res

    all_gates = [g for res in results for g in res["gates"].values()]
    payload["summary"] = {
        "all_gates_pass": bool(all(all_gates)),
        "n_gate_checks": len(all_gates),
        "n_gate_fail": int(sum(1 for v in all_gates if not v)),
        "worst_centreline_pct": max(abs(r["u_max_error_pct"]) for r in results),
        "worst_profile_rms_pct": max(r["profile_rms_interior_pct"] for r in results),
        "worst_wall_shear_pct": max(
            abs(r["wall_shear_bottom_error_pct"]) for r in results),
        "worst_slip_pct": max(abs(r["u_slip_error_pct"]) for r in results),
        "worst_flux_enhancement_pct": max(
            abs(r["flux_enhancement_error_pct"]) for r in results),
        "worst_steady_drift": max(r["steady_drift"] for r in results),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"[poiseuille-slip] all gates pass: "
          f"{payload['summary']['all_gates_pass']} "
          f"(failures {payload['summary']['n_gate_fail']}/"
          f"{payload['summary']['n_gate_checks']})")
    print(f"[poiseuille-slip] wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())