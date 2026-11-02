#!/usr/bin/env python3
"""Convergence-order benchmark for TensorSPH.

Runs the hydrostatic and dam-break cases at multiple resolutions, measures
the L2 error against analytical references, and computes the observed
convergence order.

Usage:
    python benchmarks/convergence/run.py --json-out artifacts/convergence_report.json

Output:
    JSON report with errors at each resolution and the observed convergence
    order for each case.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from tensorsph.dam_break import DamBreakConfig, run_case
from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    make_particles,
    rhs,
    wall_lattice,
)


def hydrostatic_l2_error(dx, pool_width=0.6, pool_depth=0.2, device="cpu"):
    """Compute L2 pressure error with hydrostatic initialization."""
    cfg = Config(dx=dx, pool_width=pool_width, pool_depth=pool_depth,
                 device=device, boundary="wall", hydrostatic_init=True)
    device_t = torch.device(device)
    pos, mass = make_particles(cfg, device_t)
    keep = pos[:, 1] <= cfg.pool_depth
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    n_particles = pos.shape[0]
    wall = wall_lattice(cfg, device_t)
    vel = torch.zeros_like(pos)

    t0 = time.perf_counter()
    _, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall)
    elapsed = time.perf_counter() - t0

    if not torch.isfinite(acc).all():
        return {"dx": dx, "error": float("nan"), "status": "non-finite",
                "n_particles": n_particles}

    surface_threshold = cfg.pool_depth * 0.9
    interior = pos[:, 1] < surface_threshold
    if not interior.any():
        return {"dx": dx, "error": float("nan"), "status": "no_interior",
                "n_particles": n_particles}

    y_interior = pos[interior, 1]
    p_sim = pressure[interior]
    p_ref = cfg.rho0 * cfg.gravity * (cfg.pool_depth - y_interior)

    valid = p_ref > 1.0
    if not valid.any():
        return {"dx": dx, "error": float("nan"), "status": "no_valid_ref",
                "n_particles": n_particles}

    rel_error = (p_sim[valid] - p_ref[valid]) / p_ref[valid]
    l2 = float(torch.sqrt((rel_error ** 2).mean()))
    l_inf = float(rel_error.abs().max())

    return {
        "dx": dx,
        "n_particles": n_particles,
        "l2_error": l2,
        "l_inf_error": l_inf,
        "elapsed_seconds": elapsed,
        "status": "pass",
    }


def dam_break_l2_error(dx, sim_time=0.15, device="cpu"):
    """Run dam-break and return error vs Ritter reference."""
    cfg = DamBreakConfig(
        dx=dx, pool_width=1.0, column_width=0.1, tank_height=0.4,
        sound_speed=25.0, device=device,
    )
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(sim_time / cfg.dt))
    cfg.output_every = max(1, int(round(0.005 / cfg.dt)))

    t0 = time.perf_counter()
    run = run_case(cfg)
    elapsed = time.perf_counter() - t0

    scale = cfg.column_width
    tau = [t * math.sqrt(2.0 * cfg.gravity / scale) for t in run.times]
    z_over_a = [z / scale for z in run.front]
    ritter = [2.0 * t for t in tau]

    numerator = sum((z - r) ** 2 for z, r in zip(z_over_a, ritter))
    denominator = sum(r ** 2 for r in ritter)
    l2 = math.sqrt(numerator / denominator) if denominator > 0 else float("nan")

    return {
        "dx": dx,
        "n_particles": run.initial_mass / (cfg.rho0 * dx * dx),
        "l2_error": l2,
        "elapsed_seconds": elapsed,
        "status": "pass",
        "max_speed": run.max_speed,
        "max_compression": run.max_density_compression,
        "escaped": run.escaped,
    }


def convergence_order(dx_values, errors):
    """Least-squares slope of log(error) vs log(dx)."""
    if len(dx_values) < 2:
        return float("nan")
    log_dx = [math.log(d) for d in dx_values]
    log_err = [math.log(e) for e in errors]
    n = len(log_dx)
    mean_x = sum(log_dx) / n
    mean_y = sum(log_err) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(log_dx, log_err))
    den = sum((a - mean_x) ** 2 for a in log_dx)
    if den == 0:
        return float("nan")
    return num / den


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--hydrostatic-dx", nargs="+", type=float,
                        default=[0.05, 0.025, 0.0125])
    parser.add_argument("--dam-break-dx", nargs="+", type=float,
                        default=[0.04, 0.02, 0.01])
    args = parser.parse_args()

    report = {
        "benchmark": "convergence_order",
        "device": args.device,
        "hydrostatic": {"resolutions": [], "convergence_order": None},
        "dam_break": {"resolutions": [], "convergence_order": None},
    }

    # Hydrostatic convergence
    print("=== Hydrostatic Convergence ===")
    hydro_results = []
    for dx in args.hydrostatic_dx:
        print(f"  dx={dx}...", end=" ", flush=True)
        result = hydrostatic_l2_error(dx, device=args.device)
        hydro_results.append(result)
        print(f"error={result.get('l2_error', 'nan'):.4f} "
              f"({result['n_particles']} particles, {result.get('elapsed_seconds', 0):.3f}s)")
    report["hydrostatic"]["resolutions"] = hydro_results

    hydro_errors = [r["l2_error"] for r in hydro_results if r.get("status") == "pass"]
    hydro_dx = [r["dx"] for r in hydro_results if r.get("status") == "pass"]
    if len(hydro_errors) >= 2:
        order = convergence_order(hydro_dx, hydro_errors)
        report["hydrostatic"]["convergence_order"] = order
        print(f"  Observed order: {order:.3f}")

    # Dam-break convergence
    print("\n=== Dam-Break Convergence ===")
    db_results = []
    for dx in args.dam_break_dx:
        print(f"  dx={dx}...", end=" ", flush=True)
        result = dam_break_l2_error(dx, sim_time=0.15, device=args.device)
        db_results.append(result)
        print(f"error={result.get('l2_error', 'nan'):.4f} "
              f"({result['n_particles']:.0f} particles, {result.get('elapsed_seconds', 0):.1f}s)")
    report["dam_break"]["resolutions"] = db_results

    db_errors = [r["l2_error"] for r in db_results if r.get("status") == "pass"]
    db_dx = [r["dx"] for r in db_results if r.get("status") == "pass"]
    if len(db_errors) >= 2:
        order = convergence_order(db_dx, db_errors)
        report["dam_break"]["convergence_order"] = order
        print(f"  Observed order: {order:.3f}")

    # Summary
    print("\n=== Summary ===")
    if report["hydrostatic"]["convergence_order"] is not None:
        h_order = report["hydrostatic"]["convergence_order"]
        print(f"Hydrostatic pressure: order = {h_order:.3f} "
              f"({'PASS' if 0.8 <= h_order <= 2.5 else 'FAIL'})")
    if report["dam_break"]["convergence_order"] is not None:
        d_order = report["dam_break"]["convergence_order"]
        print(f"Dam-break front:      order = {d_order:.3f} "
              f"({'PASS' if 0.2 <= d_order <= 2.0 else 'FAIL'})")

    if args.json_out:
        import os
        directory = os.path.dirname(args.json_out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.json_out}")


if __name__ == "__main__":
    main()
