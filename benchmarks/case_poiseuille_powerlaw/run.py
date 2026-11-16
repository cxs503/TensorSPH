#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH / non-Newtonian arm: power-law channel flow.

Why this arm exists
-------------------
The case-pool entry ``CASE-POISEUILLE-SPH`` (plane Poiseuille, tolerance 5 %
against the analytic solution) is already covered for a **Newtonian** fluid by
``benchmarks/poiseuille``, ``benchmarks/poiseuille_mirror`` (mirror/ghost
no-slip), ``benchmarks/poiseuille_slip`` (Navier slip), ``benchmarks/case_
poiseuille`` (Womersley pulsatile plus start-up) and ``benchmarks/case_
poiseuille_duct``.  Every one of those passes a *constant* viscosity to the
operator, so none of them can tell whether the solver transports a
**shear-dependent stress** correctly: a closure bug that mis-scales ``tau``
would be invisible.

This benchmark closes that gap with the Ostwald-de Waele fluid
``tau = K |du/dy|**(n-1) du/dy``.  The viscosity is computed per particle from
the SPH strain-rate estimate (:mod:`tensorsph.rheology`) and handed to the
unchanged viscous operator through ``rhs(..., nu_field=...)``; walls are the
same mirror/ghost no-slip condition used by ``benchmarks/poiseuille_mirror``, so
the ghost cloud is the odd extension of the near-wall field and the strain-rate
estimate is exact at the wall to the order of the kernel sum.

Reference
---------
Steady channel, walls ``y = 0, H``, body force per unit volume ``G = rho g``:

    tau(y) = G (H/2 - y),                                     tau_w = G H / 2
    u(y)   = n/(n+1) (G/K)**(1/n) [ (H/2)**((n+1)/n) - |y-H/2|**((n+1)/n) ]
    gd_w   = (G H / (2 K))**(1/n)

Textbook form: Bird, Armstrong & Hassager, *Dynamics of Polymeric Liquids*
vol. 1, ch. 4 (power-law channel flow); the same closed form as the
``dp/dx``-driven solution in Skelland, *Non-Newtonian Flow and Heat Transfer*.
The algebra is independently re-derived here with sympy (``symbolic_reference``)
and the ODE residual is evaluated numerically, so the formula in this file is
not a hand transcription.

Predeclared gates (written before the ladder was run)
-----------------------------------------------------
Graded at the finest resolution of the ladder for P1/P2/P6, at every rung for
P3/P4:

P1  centre-line error    |u_max_sim / u_max_exact - 1|                       <= 5 %
P2  profile shape        row-RMS of (u_sim - u_exact)/u_max over 0.1<=y/H<=0.9 <= 5 %
P3  convergence          row-RMS non-increasing over the ladder 8 -> 12 -> 16 -> 24
P4  closure control      the same n<1 case run with a *constant* viscosity
                         (the wall value) must MISS the power-law profile by
                         much more than 5 % -- otherwise P2 is not evidence
P5  Newtonian consistency  for n = 1 the closure returns exactly K and the run
                         reproduces the Newtonian mirror benchmark (P1,P2 <= 5 %)
P6  stress invariant     tau_w from the near-wall slope, K|du/dy|**n, vs G H/2 <= 5 %
P7  rheology identification  the flow index recovered from log|du/dy| vs
                         log|1-2y/H| over 0.15<=y/H<=0.85 is within          <= 5 %
                         (this is the inverse problem: the only measured input
                         is the velocity field, the only assumed input is that a
                         power law applies)

Usage
-----
    PYTHONPATH=src python benchmarks/case_poiseuille_powerlaw/run.py \
        --resolutions 8 12 16 24 --out benchmarks/case_poiseuille_powerlaw/out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from benchmarks.poiseuille.run import bin_rows, build_fluid, periodic_images
from tensorsph.droplet_impact import (
    Config,
    cubic_spline_gradient,
    density_from_neighbor_pairs,
    drop_wall_wall_pairs,
    resolve_device,
    rhs,
    wall_mass,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.rheology import (
    consistency_for_umax,
    shear_rate,
    power_law_channel_profile,
    power_law_channel_umax,
    power_law_viscosity,
    power_law_wall_shear_rate,
    power_law_wall_shear_stress,
)
from tensorsph.wall_mirror import plane_walls

REPO = Path(__file__).resolve().parents[2]
CASE_KEY = "CASE-POISEUILLE-SPH"

# --------------------------------------------------------------------------- #
# Reference provenance
# --------------------------------------------------------------------------- #


def symbolic_reference() -> dict:
    """Re-derive the power-law channel profile with sympy (not hand-copied).

    Substitutes ``u(y) = A [ (H/2)**((n+1)/n) - |y-H/2|**((n+1)/n) ]`` into the
    momentum balance ``d/dy(K |u'|**(n-1) u') + G = 0`` on the lower half and
    solves for ``A``; then re-derives ``tau_w`` from ``K |u'(0)|**n``.
    """
    import sympy as sp

    y, H, G, K, n, A = sp.symbols("y H G K n A", positive=True)
    s = H / 2 - y                      # distance from the centre line, y < H/2
    u = A * ((H / 2) ** ((n + 1) / n) - s ** ((n + 1) / n))
    du = sp.simplify(sp.diff(u, y))
    tau = sp.simplify(K * du ** n)     # du/dy > 0 on the lower half
    momentum = sp.simplify(sp.diff(tau, y) + G)
    A_sol = sp.solve(sp.Eq(momentum, 0), A)[0]
    A_expected = (G / K) ** (1 / n) * n / (n + 1)
    tau_w = sp.simplify((K * du ** n).subs(y, 0).subs(A, A_sol))
    return {
        "profile_coefficient": str(sp.simplify(A_sol)),
        "matches_closed_form": bool(sp.simplify(A_sol - A_expected) == 0),
        "tau_w": str(sp.simplify(tau_w)),
        "tau_w_matches_G_H_2": bool(
            sp.simplify(tau_w - G * H / 2) == 0),
        "newtonian_limit_n_1": str(sp.simplify(
            (A_sol * ((H / 2) ** ((n + 1) / n) - s ** ((n + 1) / n)))
            .subs(n, 1).subs(A, A_sol))),
    }


def ode_residual_check(n_values) -> dict:
    """Numerically differentiate the closed form and check momentum balance."""
    from tensorsph.rheology import power_law_channel_residual

    height, G = 0.1, 9810.0
    out = {}
    for n in n_values:
        K = consistency_for_umax(height, G, n, 1.2)
        y = torch.linspace(0.02 * height, 0.98 * height, 41, dtype=torch.float64)
        # keep away from the centre line, where |u'|**(n-1) u' is not
        # differentiable for n != 1
        y = y[(y - height / 2).abs() > 0.02 * height]
        res = power_law_channel_residual(y, height, G, K, n)
        out[str(n)] = {
            "K": K,
            "max_abs_residual_over_G": float(res.abs().max()) / G,
        }
    return out


# --------------------------------------------------------------------------- #
# Case driver
# --------------------------------------------------------------------------- #


def build_config(dx: float, nu_max: float, device) -> Config:
    return Config(
        dx=dx,
        h_factor=1.3,
        rho0=1000.0,
        sound_speed=20.0,
        gamma=7.0,
        gravity=0.0,          # driving force applied along x by hand
        viscosity=nu_max,     # only used for the stability accounting here
        dt=1.0e-4,
        steps=0,
        output_every=1000,
        pool_width=0.2,
        pool_depth=0.1,
        boundary="wall",
        wall_layers=3,
        wall_friction=1.0,    # full fluid-wall coupling (mirror ghosts carry u=0)
        fail_fast=False,
        device=str(device),
    )


def run_case(cfg: Config, height: float, length: float, body_force: float,
             t_end: float, t_average: float, K: float, n: float,
             device, gamma_dot_floor: float | None = None,
             mu_max: float | None = None, closure: bool = True,
             nu_const: float | None = None, t_steady_check: float = 0.2) -> dict:
    """Integrate the channel to a steady state and bin the velocity profile.

    ``closure=True`` evaluates the power-law viscosity from the SPH strain-rate
    magnitude every step; ``closure=False`` runs the identical case with a
    constant kinematic viscosity ``nu_const`` (the falsification control: a
    solver that ignores the rheology can only produce the Newtonian parabola).
    """
    dx = cfg.dx
    rho0 = cfg.rho0
    if closure and nu_const is not None:
        raise ValueError("closure=True cannot be combined with nu_const")
    if not closure and nu_const is None:
        raise ValueError("closure=False needs nu_const")

    pos, mass = build_fluid(cfg, height, length, device)
    vel = torch.zeros_like(pos)
    n_fluid = pos.shape[0]
    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (height, -1)]

    nu_cap = cfg.viscosity
    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / nu_cap
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    avg_from = t_end - t_average

    acc_sum = torch.zeros_like(pos)
    gd_sum = torch.zeros_like(pos[:, 0])
    samples = 0
    trace = []
    n_ghost = 0
    capped_fraction = 0.0
    wall_clock = time.time()

    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)

        ghost_pos, ghost_vel, ghost_mass = plane_walls(
            pos_ext, vel_ext, mass_ext, planes, support, slip="no_slip")
        n_ghost = int(ghost_pos.shape[0])
        # The ghost layer is handed to ``rhs`` as the *wall* (so the wall blend
        # in cfg.wall_friction is what makes it no-slip), but the strain-rate
        # estimate needs it as a neighbour cloud -- the two orderings are the
        # same (fluid first, then ghosts), which is why one pair list serves
        # both.
        all_pos = torch.cat((pos_ext, ghost_pos), dim=0) if n_ghost else pos_ext
        all_vel = torch.cat((vel_ext, ghost_vel), dim=0) if n_ghost else vel_ext
        all_mass = (torch.cat((mass_ext, ghost_mass), dim=0)
                    if n_ghost else mass_ext)
        pairs = cell_linked_pairs(all_pos, support)
        pairs = drop_wall_wall_pairs(pairs, pos_ext.shape[0])

        nu_field = None
        if closure:
            rho_all = density_from_neighbor_pairs(all_pos, all_mass, h,
                                                  pairs=pairs)
            gd = shear_rate(all_pos, all_vel, all_mass, h, pairs=pairs,
                            rho=rho_all)
            mu = power_law_viscosity(gd, K, n,
                                     gamma_dot_floor=gamma_dot_floor,
                                     mu_max=mu_max)
            # Only the fluid particles carry an entry: ``rhs`` pads the wall
            # block itself and, for a fluid/wall pair, uses the fluid partner's
            # value (a mirror ghost is the image of that same particle).
            nu_field = mu[:pos_ext.shape[0]] / rho_all[:pos_ext.shape[0]]

        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                        pairs=pairs,
                        wall=ghost_pos if n_ghost else None,
                        wall_vel=ghost_vel if n_ghost else None,
                        dt=dt, nu_field=nu_field)
        acc = acc[:n_fluid].clone()
        acc[:, 0] += body_force

        t = step * dt
        if t >= avg_from:
            acc_sum += vel
            samples += 1
            if closure:
                gd_mean = gd[:n_fluid]
                gd_sum += gd_mean
                if mu_max is not None:
                    capped = float((gd_mean < (gamma_dot_floor or 0.0)).to(
                        pos.dtype).mean())
                    capped_fraction += capped

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            ux = vel[:, 0]
            trace.append({
                "t": t,
                "u_max": float(ux.max()),
                "max_mach": float(ux.abs().max()) / cfg.sound_speed,
                "wall_clock_s": time.time() - wall_clock,
            })

    u_mean = acc_sum / max(samples, 1)
    rows, profile = bin_rows(pos, u_mean, height, dx)

    tail = [p for p in trace if p["t"] >= t_end - t_steady_check]
    drift = 0.0
    if len(tail) >= 2 and abs(tail[0]["u_max"]) > 0:
        drift = abs(tail[-1]["u_max"] - tail[0]["u_max"]) / abs(tail[0]["u_max"])

    stride = max(1, n_fluid // 1500)
    return {
        "n_fluid": int(n_fluid),
        "n_ghost": n_ghost,
        "dt": dt,
        "steps": steps,
        "n_averaged_samples": samples,
        "rows": rows,
        "profile": profile,
        "field": {
            "x": pos[::stride, 0].tolist(),
            "y": pos[::stride, 1].tolist(),
            "u": u_mean[::stride, 0].tolist(),
        },
        "gamma_dot_mean": (gd_sum / max(samples, 1))[::stride].tolist()
        if closure else None,
        "capped_fraction": capped_fraction / max(samples, 1) if closure else None,
        "trace": trace,
        "steady_drift": drift,
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def evaluate(case: dict, height: float, G: float, K: float, n: float,
             rho0: float, resolution: int) -> dict:
    """Turn the binned profile into the graded quantities P1..P7."""
    profile = case["profile"]
    u_max_exact = power_law_channel_umax(height, G, K, n)
    rows = []
    for p in profile:
        u_ref = float(power_law_channel_profile(p["y"], height, G, K, n))
        rows.append({
            **p,
            "u_analytic": u_ref,
            "rel_error_over_umax": (p["u_mean"] - u_ref) / u_max_exact,
        })

    interior = [r for r in rows if 0.1 <= r["y_over_H"] <= 0.9]
    rms_interior = math.sqrt(
        sum(r["rel_error_over_umax"] ** 2 for r in interior) / len(interior))
    rms_all = math.sqrt(
        sum(r["rel_error_over_umax"] ** 2 for r in rows) / len(rows))

    u_max_meas = max(r["u_mean"] for r in rows)
    centre = min(rows, key=lambda r: abs(r["y_over_H"] - 0.5))

    # -- stress invariant: near-wall slope, tau = K |du/dy|**n ---------------
    near = rows[:3] if len(rows) >= 3 else rows
    a_mat = torch.tensor([[1.0, r["y"], r["y"] * r["y"]] for r in near],
                         dtype=torch.float64)
    b_vec = torch.tensor([r["u_mean"] for r in near], dtype=torch.float64)
    fit = torch.linalg.lstsq(a_mat, b_vec).solution
    du_wall = float(fit[1])                        # c1 = du/dy at y = 0
    tau_w_meas = K * abs(du_wall) ** n
    tau_w_exact = power_law_wall_shear_stress(height, G)

    # -- rheology identification: fit log gamma_dot vs log |1-2y/H| ----------
    band = [r for r in rows if 0.15 <= r["y_over_H"] <= 0.85]
    ident = float("nan")
    ident_r2 = float("nan")
    if len(band) >= 4:
        ys = [r["y"] for r in band]
        us = [r["u_mean"] for r in band]
        xs_log, gd_log = [], []
        for k in range(1, len(band) - 1):
            dudy = abs((us[k + 1] - us[k - 1]) / (ys[k + 1] - ys[k - 1]))
            s = abs(1.0 - 2.0 * ys[k] / height)
            if dudy > 0 and s > 0:
                xs_log.append(math.log(s))
                gd_log.append(math.log(dudy))
        if len(xs_log) >= 3:
            xm = sum(xs_log) / len(xs_log)
            ym = sum(gd_log) / len(gd_log)
            sxx = sum((x - xm) ** 2 for x in xs_log)
            sxy = sum((x - xm) * (y - ym) for x, y in zip(xs_log, gd_log))
            slope = sxy / sxx
            if abs(slope) > 1e-12:
                ident = 1.0 / slope
            syy = sum((y - ym) ** 2 for y in gd_log)
            r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else float("nan")
            ident_r2 = r2

    return {
        "resolution_H_over_dx": int(resolution),
        "dx": height / resolution,
        "n": n,
        "K": K,
        "G": G,
        "tau_w_exact": tau_w_exact,
        "gamma_dot_wall_exact": power_law_wall_shear_rate(height, G, K, n),
        "u_max_measured": u_max_meas,
        "u_max_exact": u_max_exact,
        "centre_line_offset_over_umax": centre["rel_error_over_umax"],
        "u_max_error_pct": 100.0 * (u_max_meas / u_max_exact - 1.0),
        "profile_rms_interior_pct": 100.0 * rms_interior,
        "profile_rms_all_rows_pct": 100.0 * rms_all,
        "max_row_error_pct": 100.0 * max(
            abs(r["rel_error_over_umax"]) for r in rows),
        "du_dy_wall_measured": du_wall,
        "tau_w_measured": tau_w_meas,
        "tau_w_ratio": tau_w_meas / tau_w_exact,
        "tau_w_error_pct": 100.0 * (tau_w_meas / tau_w_exact - 1.0),
        "n_identified": ident,
        "n_identified_error_pct": (100.0 * (ident / n - 1.0)
                                   if ident == ident else float("nan")),
        "n_identified_r2": ident_r2,
        "steady_drift": case["steady_drift"],
        "capped_fraction": case.get("capped_fraction"),
        "wall_clock_s": case["wall_clock_s"],
        "gates": {},
    }


def gates(results: list, control: dict | None) -> dict:
    """Evaluate the predeclared gates over a finished ladder."""
    fine = results[-1]
    rms = [r["profile_rms_interior_pct"] for r in results]
    a = [r["u_max_error_pct"] for r in results]
    out = {
        "P1_centre_line_5pct": {
            "value_pct": a[-1], "threshold_pct": 5.0,
            "pass": bool(a[-1] == a[-1] and abs(a[-1]) <= 5.0)},
        "P2_profile_rms_5pct": {
            "value_pct": rms[-1], "threshold_pct": 5.0,
            "pass": bool(rms[-1] <= 5.0)},
        "P3_convergence_monotone": {
            "rms_pct_ladder": rms,
            "pass": bool(all(rms[k] >= rms[k + 1] - 1e-12
                             for k in range(len(rms) - 1))),
        },
        "P6_stress_invariant_5pct": {
            "value_pct": fine["tau_w_error_pct"], "threshold_pct": 5.0,
            "pass": bool(abs(fine["tau_w_error_pct"]) <= 5.0)},
        "P7_rheology_identification_5pct": {
            "n_identified": fine["n_identified"],
            "error_pct": fine["n_identified_error_pct"],
            "threshold_pct": 5.0,
            "pass": bool(fine["n_identified_error_pct"] ==
                         fine["n_identified_error_pct"]
                         and abs(fine["n_identified_error_pct"]) <= 5.0),
        },
    }
    if control is not None:
        out["P4_closure_control"] = {
            "control_profile_rms_interior_pct":
                control["profile_rms_interior_pct"],
            "pass": bool(control["profile_rms_interior_pct"] > 3.0 * 5.0),
        }
    return out


def newtonian_consistency(regime: dict) -> dict:
    """P5: the closure must be exactly K at n = 1 (no power round trip)."""
    gd = torch.tensor([0.0, 1e-6, 1.0, 1e2, 1e4], dtype=torch.float64)
    mu = power_law_viscosity(gd, regime["K"], 1.0)
    return {
        "all_equal_K": bool(torch.all(mu == regime["K"]).item()),
        "max_abs_dev": float((mu - regime["K"]).abs().max()),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{CASE_KEY} power-law arm")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16, 24])
    ap.add_argument("--indices", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                    help="flow-behaviour indices n (all runs at the same u_max)")
    ap.add_argument("--height", type=float, default=0.1)
    ap.add_argument("--rho0", type=float, default=1000.0)
    ap.add_argument("--body-force", type=float, default=9.81,
                    help="per-mass acceleration g (m/s^2); G = rho0 g")
    ap.add_argument("--umax", type=float, default=1.2,
                    help="target centre-line speed, used to set K per n")
    ap.add_argument("--columns", type=int, default=8)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--t-average", type=float, default=0.3)
    ap.add_argument("--floor-fraction", type=float, default=0.05,
                    help="gamma_dot floor as a fraction of the wall shear rate "
                         "(bounds mu for n < 1)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(REPO / "benchmarks"
                                         / "case_poiseuille_powerlaw" / "out"))
    args = ap.parse_args(argv)

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.quick:
        args.resolutions = [8]
        args.indices = [0.5, 1.0]
        args.t_end = 0.4
        args.t_average = 0.1

    height = args.height
    rho0 = args.rho0
    G = rho0 * args.body_force
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "case": CASE_KEY,
        "arm": "power_law_plane_channel",
        "geometry": {"height_m": height, "rho0": rho0,
                     "body_force_per_mass": args.body_force,
                     "G_pa_per_m": G,
                     "columns": args.columns},
        "symbolic": symbolic_reference(),
        "ode_residual": ode_residual_check(args.indices),
        "references": [
            "Bird, Armstrong & Hassager, Dynamics of Polymeric Liquids I, ch. 4",
            "Skelland, Non-Newtonian Flow and Heat Transfer (channel solution)",
        ],
    }

    regimes = []
    for n in args.indices:
        K = consistency_for_umax(height, G, n, args.umax)
        gd_w = power_law_wall_shear_rate(height, G, K, n)
        floor = args.floor_fraction * gd_w
        mu_at_floor = (float(power_law_viscosity(torch.tensor(floor), K, n,
                                                 gamma_dot_floor=floor))
                       if n < 1.0 else None)
        regimes.append({
            "n": n, "K": K, "gamma_dot_wall": gd_w,
            "gamma_dot_floor": floor if n < 1.0 else None,
            "mu_max": mu_at_floor,
            "mu_wall": K * gd_w ** (n - 1.0),
            "nu_wall": K * gd_w ** (n - 1.0) / rho0,
            "nu_cap": ((mu_at_floor / rho0) if n < 1.0 else
                       K * gd_w ** (n - 1.0) / rho0),
            "u_max_exact": power_law_channel_umax(height, G, K, n),
            "tau_w_exact": power_law_wall_shear_stress(height, G),
            "newtonian_consistency": newtonian_consistency(
                {"K": K}) if n == 1.0 else None,
        })
        print(f"[poiseuille-powerlaw] n={n:g} K={K:.4g} "
              f"nu_wall={regimes[-1]['nu_wall']:.4g} "
              f"nu_cap={regimes[-1]['nu_cap']:.4g}", flush=True)

    results = []
    for regime in regimes:
        n, K = regime["n"], regime["K"]
        nu_cap = regime["nu_cap"]
        for r in args.resolutions:
            dx = height / r
            length = args.columns * dx
            cfg = build_config(dx, nu_cap, device)
            print(f"[poiseuille-powerlaw] n={n:g} H/dx={r} ...", flush=True)
            case = run_case(cfg, height, length, args.body_force,
                            args.t_end, args.t_average, K, n, device,
                            gamma_dot_floor=regime["gamma_dot_floor"],
                            mu_max=regime["mu_max"])
            res = evaluate(case, height, G, K, n, rho0, r)
            res["field"] = case["field"]
            res["gamma_dot_mean"] = case["gamma_dot_mean"]
            res["n_ghost"] = case["n_ghost"]
            res["dt"] = case["dt"]
            res["steps"] = case["steps"]
            print(f"[poiseuille-powerlaw]   u_max {res['u_max_error_pct']:+.2f}%"
                  f"  rms {res['profile_rms_interior_pct']:.2f}%"
                  f"  tau {res['tau_w_error_pct']:+.2f}%"
                  f"  n_id {res['n_identified']:.3f}"
                  f"  drift {res['steady_drift']:.2e}"
                  f"  ({res['wall_clock_s']:.1f}s)", flush=True)
            results.append(res)
        ladder = [x for x in results if x["n"] == n]
        regime["ladder"] = ladder

    # Control: the shear-thinning regime run with a *constant* viscosity. A
    # solver that ignores the closure can only produce the Newtonian parabola,
    # so this is what P2 has to be able to reject.
    control = None
    thinning = [r for r in regimes if r["n"] < 1.0]
    if thinning and not args.quick:
        regime = thinning[0]
        n, K = regime["n"], regime["K"]
        dx = height / args.resolutions[-1]
        length = args.columns * dx
        cfg = build_config(dx, regime["nu_wall"], device)
        print(f"[poiseuille-powerlaw] control n={n:g} closure OFF "
              f"(nu={regime['nu_wall']:.4g}) ...", flush=True)
        case = run_case(cfg, height, length, args.body_force, args.t_end,
                        args.t_average, K, n, device, closure=False,
                        nu_const=regime["nu_wall"])
        control = evaluate(case, height, G, K, n, rho0, args.resolutions[-1])
        control["note"] = ("same geometry and driving force as n<1, viscosity "
                           "frozen at its wall value -- a Newtonian solver")
        print("[poiseuille-powerlaw]   control rms "
              f"{control['profile_rms_interior_pct']:.2f}% "
              f"(must be >> 5%)", flush=True)

    control_owner = thinning[0] if thinning else None
    for regime in regimes:
        regime["gates"] = gates(regime["ladder"],
                                control if regime is control_owner else None)

    payload = {
        **provenance,
        "config": {k: getattr(args, k) for k in
                   ("resolutions", "indices", "t_end", "t_average",
                    "floor_fraction", "columns", "umax")},
        "regimes": [{k: v for k, v in r.items() if k != "ladder"}
                    for r in regimes],
        "ladders": {str(r["n"]): r["ladder"] for r in regimes},
        "control": control,
        "gates": {str(r["n"]): r["gates"] for r in regimes},
        "all_gates_pass": all(g["pass"] for r in regimes
                              for g in r["gates"].values()),
        "generated_by": "benchmarks/case_poiseuille_powerlaw/run.py",
    }
    path = outdir / "results.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"[poiseuille-powerlaw] wrote {path}")
    print(f"[poiseuille-powerlaw] all gates pass: {payload['all_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())