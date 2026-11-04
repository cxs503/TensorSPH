"""Verification of the free-surface pressure closure on the Young-Laplace law.

Why this benchmark exists
-------------------------
``benchmarks/laplace_pressure_floor`` (m2971) decomposed the WITHHELD Laplace
check into four error sources and named exactly one gate item:

    C3  free-surface pressure closure.  ``tait_pressure`` clamps p >= 0 and the
        outer shell of a drop carries a 12-34 % sum-density deficit, so every
        particle there sits at p = 0.  The interface cannot carry ``sigma/R`` and
        the interior settles at 0.57 +- 0.06 of the analytic jump (residual
        -34 ... -38 %).

That mission stopped at the diagnosis and asked for the shell-pressure
extrapolation of Adami, Hu & Adams (2012) eq. (11) -- the one this solver
already applies to wall particles -- to be applied to free-surface particles as
well (Ihmsen et al. 2014 eq. 11).  This benchmark measures whether that fixes it,
with the same pre-registered read-out as before:

    p_abs(t) = mean of p over the core (r < 0.5 R),  Tait normalised at rho0 so
    the vacuum is the p = 0 reference;  score = time average over the last 40 %
    of the window ("absolute_time_average_tail" in the m2971 report).

What is recorded
----------------
1. ``matrix``       dx = 0.002 / 0.0015 m  x  c0 = 20 / 5 m/s, closure off and
                    on (2 Jacobi sweeps).  This is the m2971 acceptance matrix.
2. ``passes``       sweep count ladder 1..4 at dx = 0.002, c0 = 20: how many
                    sweeps the one-sided extrapolation needs, and how much
                    oscillation each one injects.
3. ``long_window`` same case, 3 capillary times instead of 1.5: the drop
                    breathes, so the tail mean at 1.5 t_sigma carries a phase
                    error that a longer average should remove.
4. ``profile``      radial pressure profile, baseline vs closure.  A Laplace
                    drop must have a *uniform* interior pressure; the flatness
                    of p(r < 0.8 R) is the shape test the scalar jump cannot see.

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_fs_pressure/run.py
    PYTHONPATH=src python benchmarks/laplace_fs_pressure/run.py --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLOOR = HERE.parent / "laplace_pressure_floor"


def _load_floor():
    spec = importlib.util.spec_from_file_location("laplace_floor_run", FLOOR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floor = _load_floor()
DP = floor.DP_ANALYTIC
T_SIGMA = floor.T_SIGMA


def window_stats(trace, frac=0.4):
    """Mean / envelope / std of the core pressure over the last ``frac``."""
    n = max(1, int(frac * len(trace["absolute"])))
    vals = trace["absolute"][-n:]
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "n_samples": n,
        "mean_Pa": mean,
        "mean_over_analytic": mean / DP,
        "rel_error_pct": 100.0 * (mean - DP) / DP,
        "min_Pa": min(vals),
        "max_Pa": max(vals),
        "std_Pa": math.sqrt(var),
        "rel_std_pct": 100.0 * math.sqrt(var) / DP,
    }


def interior_flatness(pressure_profile, r_max=0.80):
    """Max deviation of the interior bins from their own mean, in % of sigma/R."""
    bins = [row for row in pressure_profile if row["r_over_R_mid"] <= r_max]
    if not bins:
        return float("nan")
    mean = sum(row["mean"] for row in bins) / len(bins)
    if mean == 0.0:
        return float("inf")
    return 100.0 * max(abs(row["mean"] - mean) for row in bins) / abs(mean)


def summarise(label, run, closure):
    stats = window_stats(run["trace"])
    row = {
        "label": label,
        "dx_m": run["dx_m"],
        "sound_speed": run["sound_speed"],
        "free_surface_passes": run["free_surface_passes"],
        "wall_clock_agnostic_steps": run["steps"],
        "tail40": stats,
        "snapshot_absolute_rel_error_pct": run["definition_absolute"]["rel_error_pct"],
        "bandA_rel_error_pct": run["definition_bandA"]["rel_error_pct"],
        "spurious_current_vmax_over_vref": run["spurious_current"]["vmax_over_vref_final"],
        "spurious_current_passes_adami_0p1": run["spurious_current"]["passes_adami_0p1"],
        "density": run["density"],
        "interior_flatness_pct": interior_flatness(run["pressure_profile"]),
        "rho_core_over_rho0": run["density"]["rho_core_over_rho0"],
        "finite": run["finite"],
        "within_15pct": abs(stats["rel_error_pct"]) < 15.0,
        "within_3pct": abs(stats["rel_error_pct"]) < 3.0,
        "pressure_profile": run["pressure_profile"],
        "trace": run["trace"],
    }
    del closure
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="dx = 0.002 only, 2 points per closure state")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dxs = [0.002] if args.quick else [0.002, 0.0015]
    c0s = [20.0, 5.0] if args.quick else [20.0, 5.0]

    report = {
        "benchmark": "laplace_fs_pressure",
        "reference": {
            "sigma_N_per_m": floor.SIGMA,
            "radius_m": floor.RADIUS,
            "delta_p_Pa": DP,
            "t_sigma_s": T_SIGMA,
            "readout": "core p (r < 0.5 R), Tait normalised at rho0 (vacuum = 0), "
                       "time-averaged over the last 40 % of the window",
            "closure": "p_i = max(0, sum_j (p_j + rho_j g.(r_i - r_j)) W_ij / sum_j W_ij) "
                       "for rho/rho0 < 0.99, fluid neighbours only; "
                       "Adami, Hu & Adams (2012) eq. (11), used for free-surface "
                       "particles by Ihmsen et al. (2014) eq. (11)",
        },
        "matrix": [],
        "passes": [],
        "long_window": {},
        "controls": {},
    }

    for dx in dxs:
        for c0 in c0s:
            for passes in (0, 2):
                label = f"dx={dx:g}_c0={c0:g}_passes={passes}"
                run = floor.relax(dx, c0, free_surface_passes=passes)
                row = summarise(label, run, passes > 0)
                report["matrix"].append(row)
                print(f"{label:32s} tail40 {row['tail40']['rel_error_pct']:+7.1f} %  "
                      f"flat {row['interior_flatness_pct']:6.1f} %  "
                      f"vmax {row['spurious_current_vmax_over_vref']:.3f}", flush=True)

    if not args.quick:
        for passes in (1, 3, 4):
            run = floor.relax(0.002, 20.0, free_surface_passes=passes)
            row = summarise(f"passes={passes}", run, True)
            row.pop("pressure_profile", None)
            row.pop("trace", None)
            report["passes"].append(row)
            print(f"passes={passes} tail40 {row['tail40']['rel_error_pct']:+7.1f} %  "
                  f"env [{row['tail40']['min_Pa']:.0f}, {row['tail40']['max_Pa']:.0f}] Pa",
                  flush=True)

        run = floor.relax(0.002, 20.0, free_surface_passes=3,
                          t_end_over_t_sigma=3.0, max_steps=8000)
        row = summarise("dx=0.002_c0=20_passes=3_tend=3", run, True)
        row["t_end_over_t_sigma"] = 3.0
        report["long_window"] = row
        print(f"long window (3 t_sigma): tail40 {row['tail40']['rel_error_pct']:+7.1f} %  "
              f"flat {row['interior_flatness_pct']:6.1f} %", flush=True)

        # Negative control: the same extrapolation applied to *every* particle
        # (threshold >= 1) is an over-smoothing, not a closure -- it must lose.
        run = floor.relax(0.002, 20.0, free_surface_passes=2,
                          fs_rho_threshold=1.05)
        row = summarise("control_threshold_1.05", run, True)
        row.pop("pressure_profile", None)
        row.pop("trace", None)
        report["controls"]["threshold_1p05"] = row
        print(f"control threshold 1.05: tail40 {row['tail40']['rel_error_pct']:+7.1f} %",
              flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()