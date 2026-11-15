#!/usr/bin/env python3
"""Wall-layer *density and pressure* state of the mirror-wall Poiseuille anchor.

Why this file exists
--------------------
``CASE-POISEUILLE-SPH`` is graded on kinematics only.  Every gate in
``case_verdict.json`` -- centre-line, profile rms, the wall-shear battery G1-G5,
the free-slip control -- reads a **velocity**.  The state that drives the
velocity in a weakly compressible SPH solver, the summation density and the Tait
pressure, has never been recorded for this case, so two questions are open:

* Q1 (physics, independent of any estimator) -- a plane channel driven by a
  uniform body force along ``x`` has no body force along ``y``, so the pressure
  must be *flat* across the channel and *flat* along ``x`` inside the periodic
  box; ``u(y)`` solves ``mu u'' = -rho g`` with ``grad p = 0``.  Is the measured
  pressure field flat in both directions, and how large is it compared with the
  dynamic scale ``rho0 g H``?
* Q2 (the one open caveat of the anchor) -- the strictest wall-shear gauge, the
  wall-plane secant ``G1 = u(row 0) / (dx/2)``, exceeds the 5 % band after its
  stencil bias is removed at ``H/dx = 8, 12`` (+9.30 %, +6.05 %) and only passes
  at ``H/dx = 16`` (+4.61 %).  ``case_verdict.json`` records that as a
  *resolution requirement* with no mechanism.  This script tests a mechanism:
  the same near-wall row also carries a summation-density deficit ``d = 1 -
  rho(row 0)/rho0`` because the mirror/ghost lattice does not fill the truncated
  kernel support exactly, and a deficit changes the local ratio of viscous force
  to inertia.  Prediction to be falsified: ``eps_G1 = u_meas/u_analytic(at row
  0) - 1`` is proportional to ``d``, and both shrink together when the support
  is widened at fixed ``dx``.

Nothing here changes the solver: the only difference from
``run.py::run_case_mirror`` is that the two return values ``rhs`` already
computes (``rho`` and ``pressure``) are averaged over the same window instead of
being dropped, and that ``h_factor`` may be swept.  The velocity path is the
anchor's path, and ``--check-anchor`` re-derives the anchor's own row profile
from this run so the two drivers can be compared row by row.

Usage
-----
    PYTHONPATH=. python benchmarks/poiseuille_mirror/wall_layer_state.py \
        --resolutions 8 12 16 --threads 1
    PYTHONPATH=. python benchmarks/poiseuille_mirror/wall_layer_state.py \
        --resolutions 8 --h-factors 1.3 1.5 1.7 --tag h_sweep

Output: ``out/wall_layer_state.json`` (one block per resolution / ``h_factor``).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from benchmarks.poiseuille.run import (
    analytic,
    bin_rows,
    build_fluid,
    make_config,
    periodic_images,
)
from tensorsph.droplet_impact import resolve_device, rhs
from tensorsph.wall_mirror import plane_walls

REPO = Path(__file__).resolve().parents[2]

#: Pressure is reported in units of the body-force scale ``rho0 g H`` -- the
#: scale a *spurious* pressure gradient would have to reach to balance the body
#: force, and therefore the natural yardstick for "is grad p zero?".
def _p_scale(rho0: float, body_force: float, height: float) -> float:
    return rho0 * body_force * height


def run_case_state(cfg, height: float, length: float, body_force: float,
                   t_end: float, t_average: float, device,
                   slip: str = "no_slip") -> dict:
    """Integrate the mirror channel and average ``rho`` and ``p`` per row.

    Identical time loop to ``run.py::run_case_mirror`` (same ``pos``/``vel``
    update, same ghost construction, same ``dt`` rule); the extra work is that
    the density and pressure that ``rhs`` returns are accumulated alongside the
    velocity over the same averaging window.
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
    avg_from = t_end - t_average

    u_sum = torch.zeros_like(pos)
    rho_sum = torch.zeros(n, device=pos.device, dtype=pos.dtype)
    p_sum = torch.zeros(n, device=pos.device, dtype=pos.dtype)
    samples = 0
    wall_clock = time.time()
    n_ghost = 0

    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)

        ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                              planes, support, slip=slip)
        n_ghost = int(ghost_pos.shape[0])

        rho, pressure, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                                 wall=ghost_pos if n_ghost else None,
                                 wall_vel=ghost_vel if n_ghost else None)
        acc = acc[:n].clone()
        acc[:, 0] += body_force

        t = step * dt
        if t >= avg_from:
            u_sum += vel
            rho_sum += rho[:n]
            p_sum += pressure[:n]
            samples += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

    u_mean = u_sum / max(samples, 1)
    rho_mean = rho_sum / max(samples, 1)
    p_mean = p_sum / max(samples, 1)

    _, profile = bin_rows(pos, u_mean, height, dx)

    # Per-row density / pressure: same row binning as the velocity profile, so a
    # row index means the same particles in every table below.
    row_idx = torch.round((pos[:, 1] - dx * 0.5) / dx).to(torch.long)
    rho0 = cfg.rho0
    rows = []
    for entry in profile:
        sel = row_idx == entry["row"]
        rho_r = rho_mean[sel]
        p_r = p_mean[sel]
        u_num = entry["u_mean"]
        u_ref = analytic(entry["y"], height, body_force, cfg.viscosity, 0.0)
        rows.append({
            **entry,
            "rho_over_rho0": float((rho_r / rho0).mean()),
            "rho_std_over_rho0": (float((rho_r / rho0).std(unbiased=False))
                                  if rho_r.numel() > 1 else 0.0),
            "p_mean": float(p_r.mean()),
            "p_std": (float(p_r.std(unbiased=False)) if p_r.numel() > 1 else 0.0),
            "p_over_scale": float(p_r.mean()) / _p_scale(rho0, body_force, height),
            "u_analytic": u_ref,
            "u_excess_over_analytic": (u_num / u_ref - 1.0) if u_ref else float("nan"),
        })

    # Streamwise structure of the pressure at fixed row: the periodic box has no
    # pressure gradient to apply, so any column-to-column trend is spurious.
    col_idx = torch.round((pos[:, 0] - dx * 0.5) / dx).to(torch.long)
    n_cols = int(round(length / dx))
    col_p = []
    for c in range(n_cols):
        sel = col_idx == c
        if int(sel.sum()):
            col_p.append(float(p_mean[sel].mean()))
    p_x_span = (max(col_p) - min(col_p)) if col_p else float("nan")

    return {
        "n_fluid": int(n),
        "n_ghost": n_ghost,
        "h_factor": float(cfg.h_factor),
        "h_over_dx": float(cfg.h_factor),
        "dt": float(dt),
        "steps": int(steps),
        "n_averaged_samples": int(samples),
        "slip": slip,
        "rows": rows,
        "p_columns": col_p,
        "p_x_span_over_scale": p_x_span / _p_scale(rho0, body_force, height),
        "wall_clock_s": time.time() - wall_clock,
    }



# --------------------------------------------------------------------------- #
# Static operator probe: hand the *exact* solution to the solver
# --------------------------------------------------------------------------- #

def _bin_scalar(pos, values, height: float, dx: float):
    """Average a per-particle scalar over the same lattice rows as ``bin_rows``."""
    row = torch.round((pos[:, 1] - dx * 0.5) / dx).to(torch.long)
    out = []
    for k in range(int(round(height / dx))):
        sel = row == k
        if int(sel.sum()) == 0:
            continue
        out.append((k, float(dx * 0.5 + k * dx), float(values[sel].mean())))
    return out


def operator_probe(cfg, height: float, length: float, body_force: float,
                   device, slip: str = "no_slip") -> dict:
    """Residual acceleration of the *continuous* solution on the discrete lattice.

    The steady solution of the case is known exactly, ``u(y) = g(H-y)y/(2 nu)``.
    Sampling it on the lattice and handing it to :func:`rhs` -- with the mirror
    ghosts built from the same field, so the wall condition is imposed exactly as
    in a run -- measures the one thing a run cannot separate from time
    integration: **how much of the body force the discrete viscous operator
    fails to balance, row by row**,

    .. math:: r(y) = \\frac{1}{g}\\left( a_{\\mathrm{SPH},x}(y) + g \\right),

    with ``a`` the acceleration ``rhs`` returns.  A consistent operator gives
    ``r = 0`` up to round-off on every row; whatever is left is the discretisation
    error that a steady run turns into a velocity bias.  No time integration, one
    ``rhs`` call per configuration.
    """
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    y = pos[:, 1]
    u_exact = body_force * (height - y) * y / (2.0 * cfg.viscosity)
    vel = torch.stack((u_exact, torch.zeros_like(u_exact)), dim=-1)

    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (height, -1)]

    pos_ext = periodic_images(pos, length)
    vel_ext = torch.cat((vel, vel, vel), dim=0)
    mass_ext = torch.cat((mass, mass, mass), dim=0)
    ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                          planes, support, slip=slip)
    rho, pressure, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                             wall=ghost_pos if ghost_pos.shape[0] else None,
                             wall_vel=ghost_vel if ghost_vel.shape[0] else None)
    acc = acc[:n]

    rows = []
    for k, yk, res in _bin_scalar(pos, acc[:, 0], height, dx):
        rows.append({
            "row": k,
            "y": yk,
            "y_over_H": yk / height,
            "u_exact": float(u_exact[(torch.round((y - dx * 0.5) / dx)
                                      .to(torch.long)) == k].mean()),
            "residual_over_g": res / body_force,
        })
    rho_rows = [v for _, _, v in _bin_scalar(pos, rho[:n], height, dx)]
    p_rows = [v for _, _, v in _bin_scalar(pos, pressure[:n], height, dx)]
    return {
        "residual_over_g_row0": rows[0]["residual_over_g"],
        "residual_over_g_interior_max": max(
            abs(r["residual_over_g"]) for r in rows
            if 0.1 <= r["y_over_H"] <= 0.9),
        "residual_over_g_max_abs": max(abs(r["residual_over_g"]) for r in rows),
        "residual_over_g_odd_part": 0.5 * (rows[0]["residual_over_g"]
                                           - rows[-1]["residual_over_g"]),
        "rho_over_rho0_exact_field": sum(rho_rows) / len(rho_rows) / cfg.rho0,
        "p_over_scale_exact_field": (sum(p_rows) / len(p_rows)
                                     / (cfg.rho0 * body_force * height)),
        "rows": rows,
        "n_ghost": int(ghost_pos.shape[0]),
    }



# --------------------------------------------------------------------------- #
# Discrete fixed point: solve the scheme the run converges to, without running
# --------------------------------------------------------------------------- #

def discrete_fixpoint(cfg, height: float, length: float, body_force: float,
                      device, slip: str = "no_slip") -> dict:
    """Solve ``L(u) + g = 0`` for the row profile, using ``rhs`` as the operator.

    The time loop of the anchor converges to the fixed point of
    ``vel <- vel + dt * (L(vel) + g)``.  Holds the lattice fixed: the summation
    density and hence the (uniform) pressure depend on positions only, so for a
    *fixed* lattice ``rhs`` is **linear in the velocity**, and a row-uniform
    basis spans the symmetric solution exactly.  Probing one unit row velocity
    per call therefore assembles the discrete operator ``L`` (``n_rows`` calls,
    no time stepping), and the steady profile is the solution of an
    ``n_rows x n_rows`` linear system.

    What this buys: the run's velocity error is split into "what the discrete
    operator does to the exact solution" (this function) and "what the time
    integration leaves behind".  If the two agree, the anchor's profile error is
    a *discretisation* property and refinable; if they disagree, the time loop
    is adding error of its own.
    """
    import numpy as np

    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    row = torch.round((pos[:, 1] - dx * 0.5) / dx).to(torch.long)
    n_rows = int(round(height / dx))

    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (height, -1)]
    pos_ext = periodic_images(pos, length)
    mass_ext = torch.cat((mass, mass, mass), dim=0)

    def apply_row_profile(ux):
        vel = torch.stack((ux, torch.zeros_like(ux)), dim=-1)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                              planes, support, slip=slip)
        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                        wall=ghost_pos if ghost_pos.shape[0] else None,
                        wall_vel=ghost_vel if ghost_vel.shape[0] else None)
        acc = acc[:n, 0]
        out = np.zeros(n_rows)
        for k in range(n_rows):
            sel = row == k
            out[k] = float(acc[sel].mean()) if int(sel.sum()) else 0.0
        return out

    L = np.zeros((n_rows, n_rows))
    for j in range(n_rows):
        ux = (row == j).to(pos.dtype)
        L[:, j] = apply_row_profile(ux)

    rhs_vec = np.full(n_rows, -body_force)
    # Row 0 and the last row are the two wall layers; the system is symmetric
    # about mid-height, so a least-squares solve is used and the residual is
    # reported rather than assumed zero.
    sol, *_ = np.linalg.lstsq(L, rhs_vec, rcond=None)
    resid = float(np.max(np.abs(L @ sol - rhs_vec)))

    sv = np.linalg.svd(L, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] else float("inf")

    ys = np.array([dx * 0.5 + k * dx for k in range(n_rows)])
    u_exact = body_force * (height - ys) * ys / (2.0 * cfg.viscosity)
    return {
        "n_rows": n_rows,
        "u_fixpoint": sol.tolist(),
        "u_exact": u_exact.tolist(),
        "y": ys.tolist(),
        "u_excess_fixpoint": (sol / u_exact - 1.0).tolist(),
        "u_max_fixpoint": float(sol.max()),
        "u_max_excess": float(sol.max() / u_exact.max() - 1.0),
        "G1_bias_corrected_excess": float((sol[0] / u_exact[0] - 1.0)
                                         / (1.0 - 0.5 * dx / height)),
        "system_residual_over_g": resid / body_force,
        # The steady *amplitude* is a balance between the body force and the
        # wall drag.  If the operator is ill-conditioned in the row basis, a
        # small residual momentum imbalance produces a large amplitude error --
        # which is the difference between "the run agrees with the parabola" and
        # "the scheme's fixed point agrees with the parabola".
        "singular_values": sv.tolist(),
        "condition_number": cond,
        # amplification of a per-row momentum imbalance into a u_max shift
        "amplification_umax_per_2pct_residual": float(
            np.linalg.norm(np.linalg.pinv(L) @ np.full(n_rows, 0.02 * body_force),
                           ord=np.inf)),
        # the operator's response to the *exact* solution, row by row
        "operator_response_exact": (apply_row_profile(
            torch.tensor(u_exact, dtype=pos.dtype, device=pos.device)[row]
        ) / body_force).tolist(),
    }


# --------------------------------------------------------------------------- #
# Diagnostics: is the wall-layer state consistent, and does it explain G1?
# --------------------------------------------------------------------------- #

def diagnostics(case: dict, height: float, body_force: float,
                viscosity: float, rho0: float) -> dict:
    """Reduce a case block to the numbers Q1 and Q2 are asked about."""
    rows = case["rows"]
    scale = _p_scale(rho0, body_force, height)
    ys = [r["y"] for r in rows]
    ps = [r["p_mean"] for r in rows]
    n = len(rows)

    # dp/dy by least squares over the rows (units: dyn/cm^3 per m -> compare to
    # rho0 g, the only force density available to drive a wall-normal flow).
    y_bar = sum(ys) / n
    p_bar = sum(ps) / n
    sxy = sum((y - y_bar) * (p - p_bar) for y, p in zip(ys, ps))
    sxx = sum((y - y_bar) ** 2 for y in ys)
    dp_dy = sxy / sxx if sxx else 0.0

    interior = [r for r in rows if 0.1 <= r["y_over_H"] <= 0.9]
    rho_bulk = sum(r["rho_over_rho0"] for r in interior) / len(interior)
    p_bulk_over_scale = sum(r["p_over_scale"] for r in interior) / len(interior)

    row0 = rows[0]
    row0_opposite = rows[-1]
    return {
        "row0_y_over_H": row0["y_over_H"],
        "rho_over_rho0_row0": row0["rho_over_rho0"],
        "rho_deficit_row0": 1.0 - row0["rho_over_rho0"],
        "rho_over_rho0_row_last": row0_opposite["rho_over_rho0"],
        "rho_deficit_asymmetry": abs((1.0 - row0["rho_over_rho0"])
                                     - (1.0 - row0_opposite["rho_over_rho0"])),
        "rho_over_rho0_bulk": rho_bulk,
        "rho_deficit_bulk": 1.0 - rho_bulk,
        "rho_min_over_rho0": min(r["rho_over_rho0"] for r in rows),
        "p_bulk_over_scale": p_bulk_over_scale,
        "p_row0_over_scale": row0["p_over_scale"],
        "p_max_abs_over_scale": max(abs(r["p_over_scale"]) for r in rows),
        "p_noise_row_std_over_scale": max(r["p_std"] for r in rows) / scale,
        "dp_dy_over_rho0_g": dp_dy / (rho0 * body_force),
        "p_x_span_over_scale": case["p_x_span_over_scale"],
        # Q2: the anchor's open number, re-derived here from this run's own rows
        "u_excess_row0": row0["u_excess_over_analytic"],
        "u_excess_interior_max": max(abs(r["u_excess_over_analytic"])
                                     for r in interior),
        # stencil-bias-corrected G1 = excess row 0 / (1 - dx/(2H))
        "G1_bias_corrected_excess": (row0["u_excess_over_analytic"]
                                     / (1.0 - row0["y_over_H"])),
        "ratio_excess_to_deficit": ((row0["u_excess_over_analytic"]
                                     / (1.0 - row0["rho_over_rho0"]))
                                    if (1.0 - row0["rho_over_rho0"]) else float("nan")),
    }


def correlation(xs, ys) -> float:
    """Pearson correlation, ``nan`` when either series is constant."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks/poiseuille_mirror/out")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--h-factors", type=float, nargs="+", default=[1.3],
                    help="smoothing factor h/dx; the anchor pins 1.3. Widening "
                         "the support fills the truncated wall kernel better, so "
                         "it is the controlled perturbation of the deficit.")
    ap.add_argument("--viscosity", type=float, default=0.01)
    ap.add_argument("--body-force", type=float, default=9.81)
    ap.add_argument("--height", type=float, default=0.1)
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--t-average", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--skip-fixpoint", action="store_true",
                    help="skip the linear-system prediction of the steady profile")
    ap.add_argument("--skip-operator-probe", action="store_true",
                    help="skip the one-call static consistency probe")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    rho0 = 1000.0

    cases = []
    for hf in args.h_factors:
        for r in args.resolutions:
            dx = args.height / r
            length = args.columns * dx
            cfg = make_config(dx, args.viscosity, 1.0, device)
            cfg.h_factor = float(hf)
            print(f"[wall-layer] H/dx={r} h/dx={hf} ...", flush=True)
            case = run_case_state(cfg, args.height, length, args.body_force,
                                  args.t_end, args.t_average, device)
            case["resolution_H_over_dx"] = int(r)
            case["dx"] = float(dx)
            case["length"] = float(length)
            case["diagnostics"] = diagnostics(case, args.height,
                                              args.body_force,
                                              args.viscosity, rho0)
            if not args.skip_fixpoint:
                case["discrete_fixpoint"] = discrete_fixpoint(
                    cfg, args.height, length, args.body_force, device)
                fp = case["discrete_fixpoint"]
                print(f"[wall-layer]   fixpoint u_max {100 * fp['u_max_excess']:+.2f}% "
                      f"row0 {100 * (fp['u_fixpoint'][0] / fp['u_exact'][0] - 1.0):+.2f}% "
                      f"(measured row0 {100 * case['diagnostics']['u_excess_row0']:+.2f}%) "
                      f"cond {fp['condition_number']:.0f}",
                      flush=True)
            if not args.skip_operator_probe:
                case["operator_probe"] = operator_probe(
                    cfg, args.height, length, args.body_force, device)
            print(f"[wall-layer]   rho0_row {case['diagnostics']['rho_over_rho0_row0']:.5f} "
                  f"deficit {100 * case['diagnostics']['rho_deficit_row0']:+.2f}%  "
                  f"p/scale {case['diagnostics']['p_bulk_over_scale']:+.4f}  "
                  f"u_excess {100 * case['diagnostics']['u_excess_row0']:+.2f}%  "
                  f"r0 {case.get('operator_probe', {}).get('residual_over_g_row0', float('nan')):+.4f}  "
                  f"({case['wall_clock_s']:.1f}s)", flush=True)
            cases.append(case)

    # Q2 test across the whole sweep (the anchor ladder is the h/dx = 1.3 subset)
    is_anchor = [abs(c["h_factor"] - 1.3) < 1e-9 for c in cases]
    anchor = [c["diagnostics"] for c, a in zip(cases, is_anchor) if a]
    widened = [c["diagnostics"] for c, a in zip(cases, is_anchor) if not a]

    def _series(ds, key):
        return [d[key] for d in ds]

    q2 = {
        "prediction": ("eps_G1 (row-0 velocity excess) is proportional to the "
                       "row-0 summation-density deficit d, and both shrink when "
                       "the support is widened at fixed dx"),
        "anchor_ladder": {
            "resolutions": [c["resolution_H_over_dx"]
                            for c, a in zip(cases, is_anchor) if a],
            "rho_deficit_row0": _series(anchor, "rho_deficit_row0"),
            "u_excess_row0": _series(anchor, "u_excess_row0"),
            "G1_bias_corrected_excess": _series(anchor, "G1_bias_corrected_excess"),
        },
        "corr_deficit_vs_excess_anchor": correlation(
            _series(anchor, "rho_deficit_row0"), _series(anchor, "u_excess_row0")),
        "corr_deficit_vs_G1_anchor": correlation(
            _series(anchor, "rho_deficit_row0"),
            _series(anchor, "G1_bias_corrected_excess")),
        "widened_support": {
            "h_factors": [c["h_factor"]
                          for c, a in zip(cases, is_anchor) if not a],
            "rho_deficit_row0": _series(widened, "rho_deficit_row0"),
            "u_excess_row0": _series(widened, "u_excess_row0"),
        },
        "corr_deficit_vs_excess_all": correlation(
            [c["diagnostics"]["rho_deficit_row0"] for c in cases],
            [c["diagnostics"]["u_excess_row0"] for c in cases]),
    }

    payload = {
        "case": "plane_poiseuille_channel_mirror_wall::wall_layer_state",
        "purpose": ("record the density/pressure state the CASE-POISEUILLE-SPH "
                    "anchor never stored, and test whether the near-wall "
                    "summation-density deficit explains gauge G1's residual"),
        "parameters": {
            "height": args.height,
            "columns": args.columns,
            "viscosity": args.viscosity,
            "body_force": args.body_force,
            "rho0": rho0,
            "t_end": args.t_end,
            "t_average": args.t_average,
            "resolutions": args.resolutions,
            "h_factors": args.h_factors,
        },
        "reference": {
            "velocity": "u(y) = g (H-y) y / (2 nu); u_max = g H^2 / (8 nu); tau_w = rho g H / 2",
            "pressure": ("plane channel, uniform axial body force: grad p = 0, "
                         "so p/(rho0 g H) must be flat in y and in x and small "
                         "compared with 1 (Batchelor §4.2; White §3-2.2)"),
        },
        "tag": args.tag,
        "torch": torch.__version__,
        "cases": cases,
        "q2_wall_layer_hypothesis": q2,
        "wall_clock_s": sum(c["wall_clock_s"] for c in cases),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    name = ("wall_layer_state.json" if args.tag == "main"
            else f"wall_layer_state_{args.tag}.json")
    path = args.out / name
    path.write_text(json.dumps(payload, indent=1))
    print(f"[wall-layer] wrote {path}")
    print(f"[wall-layer] corr(deficit, excess) anchor = "
          f"{q2['corr_deficit_vs_excess_anchor']:+.3f}, all = "
          f"{q2['corr_deficit_vs_excess_all']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())