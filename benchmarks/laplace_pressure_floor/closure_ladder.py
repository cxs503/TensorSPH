"""Ladder for the free-surface pressure closure (the open P0 of m2971).

``docs/verification_laplace_pressure_floor.md`` §7 named the free-surface
pressure closure as the one remaining gate on the Young-Laplace check: the Tait
law is normalised at ``rho0`` and clamped at ``p >= 0``, the whole free-surface
shell of a lattice drop has a density deficit, so that shell carries ``p = 0``
and the interior can only hold the fraction of ``sigma/R`` that the shell does
not take.  ``Config.free_surface_pressure`` closes the shell with the Adami,
Ihmsen & Adams (2012) §3.3 shell-pressure extrapolation; this script measures
what it buys, on the four acceptance points (dx x c0) of the P0 criterion and
as a function of the number of sweeps.

Points are independent processes: ``--jobs`` runs them concurrently, which is
the only reason this fits in one mission (each point advances a drop for 1.5
capillary times).

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/closure_ladder.py
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/closure_ladder.py --jobs 10
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/closure_ladder.py --quick
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "closure_ladder.json"

# The P0 acceptance points, exactly as written in
# docs/verification_laplace_pressure_floor.md §7: c0 = 20 and 5 at dx = 2 mm and
# 1.5 mm, all four inside 15 %.
ACCEPTANCE_POINTS = [(0.002, 20.0), (0.002, 5.0), (0.0015, 20.0), (0.0015, 5.0)]
PASSES = [0, 1, 2, 3, 6]   # 0 = closure off (the previous best configuration)

_spec = importlib.util.spec_from_file_location("lpf", HERE / "run.py")
lpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lpf)


def one_point(args):
    dx, c0, passes = args
    torch.set_num_threads(int(os.environ.get("LADDER_THREADS", "4")))
    t0 = time.time()
    run = lpf.relax(dx=dx, c0=c0,
                    free_surface_pressure=passes > 0,
                    free_surface_pressure_passes=max(1, passes))
    run["wall_clock_s"] = time.time() - t0
    err = run["absolute_time_average_tail"]["rel_error_pct"]
    print(f"dx={dx} c0={c0:4.1f} passes={passes} "
          f"tail={err:+7.2f}% rho_ratio={run['density']['ratio_compression']:.3f} "
          f"vmax/vref={run['spurious_current']['vmax_over_vref_final']:.3f} "
          f"{run['wall_clock_s']:.0f}s", flush=True)
    return run


def build_report(runs):
    baseline = {(r["dx_m"], r["sound_speed"]): r for r in runs
                if not r["free_surface_pressure"]}
    ladder = []
    for r in sorted(runs, key=lambda q: (q["dx_m"], q["sound_speed"],
                                        q["free_surface_pressure_passes"])):
        base = baseline[(r["dx_m"], r["sound_speed"])]
        err = r["absolute_time_average_tail"]["rel_error_pct"]
        ladder.append({
            "dx_m": r["dx_m"], "sound_speed": r["sound_speed"],
            "passes": r["free_surface_pressure_passes"],
            "n_particles": r["n_particles"], "steps": r["steps"],
            "delta_p_Pa": r["absolute_time_average_tail"]["delta_p_Pa"],
            "rel_error_pct": err,
            "rel_error_final_snapshot_pct":
                r["definition_absolute"]["rel_error_pct"],
            "within_3pct": abs(err) < 3.0,
            "within_15pct": abs(err) < 15.0,
            # Error reduction in percentage points (sign-free: the baseline
            # residual is negative, i.e. the drop under-pressures).
            "error_reduction_pt": (abs(base["absolute_time_average_tail"]
                                      ["rel_error_pct"]) - abs(err)),
            # Interior compression divided by what sigma/R needs, read from the
            # time-averaged pressure (the snapshot value breathes with the drop's
            # fundamental mode and is not a stable diagnostic).
            "compression_ratio": 1.0 + err / 100.0,
            "compression_ratio_snapshot": r["density"]["ratio_compression"],
            "rho_core_over_rho0": r["density"]["rho_core_over_rho0"],
            "vmax_over_vref": r["spurious_current"]["vmax_over_vref_final"],
            "finite": r["finite"],
            "spurious_current_ok": r["spurious_current"]["passes_adami_0p1"],
            "pressure_profile": r["pressure_profile"],
            "trace": r["trace"],
        })

    best = {}
    for row in ladder:
        if row["passes"] == 0:
            continue
        key = (row["dx_m"], row["sound_speed"])
        cur = best.get(key)
        # Best = smallest |error| that keeps the spurious-current criterion.
        if row["spurious_current_ok"] and (
                cur is None or abs(row["rel_error_pct"]) < abs(cur["rel_error_pct"])):
            best[key] = row

    acceptance = {}
    for passes in sorted({row["passes"] for row in ladder}):
        rows = [row for row in ladder if row["passes"] == passes]
        at_points = {(row["dx_m"], row["sound_speed"]): row for row in rows}
        ok = all(key in at_points and at_points[key]["within_15pct"]
                 for key in ACCEPTANCE_POINTS)
        acceptance[str(passes)] = {
            "all_four_points_within_15pct": ok,
            "worst_abs_error_pct": max(abs(at_points[k]["rel_error_pct"])
                                       for k in ACCEPTANCE_POINTS
                                       if k in at_points),
            "points": {f"{k[0]}_{k[1]}": at_points[k]["rel_error_pct"]
                       for k in ACCEPTANCE_POINTS if k in at_points},
        }
    return {
        "generated_by": "benchmarks/laplace_pressure_floor/closure_ladder.py",
        "reference": {"delta_p_Pa": lpf.DP_ANALYTIC, "sigma_N_per_m": lpf.SIGMA,
                      "radius_m": lpf.RADIUS, "rho0": lpf.RHO0,
                      "t_sigma_s": lpf.T_SIGMA},
        "acceptance_points": [list(k) for k in ACCEPTANCE_POINTS],
        "passes_tested": PASSES,
        "total_wall_clock_s": sum(r["wall_clock_s"] for r in runs),
        "ladder": ladder,
        "best_per_point": {f"{k[0]}_{k[1]}": v["passes"] for k, v in best.items()},
        "acceptance": acceptance,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--quick", action="store_true",
                    help="only dx=2 mm, c0=20 (fast smoke run)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    points = [(0.002, 20.0)] if args.quick else ACCEPTANCE_POINTS
    tasks = [(dx, c0, p) for (dx, c0) in points for p in PASSES]
    t0 = time.time()
    if args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            runs = list(pool.map(one_point, tasks))
    else:
        runs = [one_point(t) for t in tasks]
    report = build_report(runs)
    report["wall_clock_s"] = time.time() - t0
    report["jobs"] = args.jobs
    if not args.no_write:
        OUT.write_text(json.dumps(report, indent=1))
        print(f"wrote {OUT.relative_to(HERE.parents[1])} "
              f"({report['wall_clock_s']:.0f}s wall, {args.jobs} jobs)")
    for passes, row in report["acceptance"].items():
        print(f"  passes={passes:>2}: within_15pct={row['all_four_points_within_15pct']}"
              f"  worst={row['worst_abs_error_pct']:.2f}%")


if __name__ == "__main__":
    main()