"""B2 dam-break: front-trajectory ladder measured against *both* reference tables.

`benchmarks/dam_break/resolution_ladder.json` measured the Martin & Moyce error
on a three-rung refinement ladder and found it resolution independent (0.5988 /
0.5774 / 0.5728 in ``Z/a``).  `reference_audit.py` shows why that residual
cannot be read as a solver error: the bundled reference is a piecewise-linear
sketch whose spread against an independent transcription reaches 0.85 in
``Z/a``, 17x its own ±0.05 claim.

This script closes the loop by re-measuring the *same* simulations against both
references with everything else held fixed, so the reference choice is the only
difference:

* ``bundled``    - ``martin_moyce_1952.json`` (the table the gate has used);
* ``crosscheck`` - ``martin_moyce_1952_crosscheck.json`` (independent transcription);
* ``ritter``     - analytic shallow-water front, reported, never gated.

The output is machine readable and feeds
``docs/benchmarks/dam-break-reference-audit.md``; no number in that document is
typed by hand.

Usage::

    python benchmarks/dam_break/reference_ladder.py --sizes 0.04,0.02 --seconds 0.5 \
        --json-out benchmarks/dam_break/reference_ladder.json --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402

try:  # importable both as a script and as benchmarks.dam_break.reference_ladder
    from .reference_audit import _interp, load_bundled, load_crosscheck  # noqa: E402
except ImportError:  # pragma: no cover - direct script execution
    from reference_audit import _interp, load_bundled, load_crosscheck  # noqa: E402


def error_vs(tau: list[float], za: list[float], ref: dict) -> dict:
    """MAE / RMSE / max error of a simulated curve against a reference table."""
    errors = []
    rows = []
    for tau_ref, za_ref in zip(ref["tau"], ref["za"]):
        if not tau or tau_ref < tau[0] - 1e-12 or tau_ref > tau[-1] + 1e-12:
            continue
        za_sim = _interp(tau, za, tau_ref)
        if za_sim is None:
            continue
        errors.append(za_sim - za_ref)
        rows.append({"tau": tau_ref, "ref": za_ref, "sim": za_sim,
                     "error": za_sim - za_ref})
    if not errors:
        return {"n_points": 0, "mae": None, "rmse": None, "max_abs": None}
    abs_errors = [abs(e) for e in errors]
    return {
        "n_points": len(errors),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "max_abs": max(abs_errors),
        "mean_signed": sum(errors) / len(errors),
        "within_claimed_uncertainty":
            sum(1 for e in abs_errors if e <= ref["claimed_uncertainty"]) / len(abs_errors),
        "rows": rows,
    }


def run_one(dx: float, args, refs: dict) -> dict:
    cfg = DamBreakConfig(
        dx=dx,
        pool_width=args.tank_width,
        column_width=args.column_width,
        sound_speed=args.sound_speed,
        max_courant=args.max_courant,
        device=args.device,
        wall_friction=args.wall_friction,
        density_diffusion=args.density_diffusion,
        adaptive_dt=args.adaptive_dt,
    )
    cfg.column_height = 2.0 * args.column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(args.seconds / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))

    t0 = time.time()
    run = run_case(cfg)
    wall = time.time() - t0

    a = cfg.column_width
    sim_tau = [t * math.sqrt(2.0 * cfg.gravity / a) for t in run.times]
    sim_za = [(f - a) / a for f in run.front]

    ritter = {
        "n_points": len(sim_tau),
        "mae": None,
    }
    ritter_errors = [abs(za - tau) / tau for tau, za in zip(sim_tau, sim_za) if tau > 0.01]
    if ritter_errors:
        ritter["mae"] = sum(ritter_errors) / len(ritter_errors)

    record = {
        "dx": dx,
        "particles": int(run.final_positions.shape[0]),
        "dt": cfg.dt,
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "tau_max": sim_tau[-1] if sim_tau else None,
        "wall_clock_s": wall,
        "max_speed": run.max_speed,
        "mach_number": run.max_speed / cfg.sound_speed,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "sim_tau": sim_tau,
        "sim_za": sim_za,
        "ritter_mean_relative_error": ritter["mae"],
    }
    for name, ref in refs.items():
        record[f"error_vs_{name}"] = error_vs(sim_tau, sim_za, ref)
    return record


def gate_table(results: list[dict], ref_name: str, limit: float) -> dict:
    rows = []
    for r in results:
        err = r[f"error_vs_{ref_name}"]
        rows.append({
            "dx": r["dx"],
            "particles": r["particles"],
            "mae": err["mae"],
            "rmse": err["rmse"],
            "within": err["within_claimed_uncertainty"],
            "result": "pass" if (err["mae"] is not None and err["mae"] <= limit) else "fail",
        })
    return {
        "reference": ref_name,
        "metric": f"mean absolute error in Z/a vs {ref_name}",
        "limit": limit,
        "rows": rows,
        "finest": rows[-1] if rows else None,
        "all_pass": all(r["result"] == "pass" for r in rows) if rows else False,
    }


def observed_order(results: list[dict], ref_name: str) -> dict:
    """Slope of log(MAE) against log(dx): is the residual converging at all?"""
    pairs = [
        (r["dx"], r[f"error_vs_{ref_name}"]["mae"], r[f"error_vs_{ref_name}"]["rmse"])
        for r in results
        if r[f"error_vs_{ref_name}"]["mae"] is not None
    ]
    out = {"pairs": [{"dx": dx, "mae": mae, "rmse": rmse} for dx, mae, rmse in pairs],
           "order_mae": [], "order_rmse": []}
    for (dx0, mae0, rmse0), (dx1, mae1, rmse1) in zip(pairs, pairs[1:]):
        out["order_mae"].append(math.log(mae1 / mae0) / math.log(dx1 / dx0))
        out["order_rmse"].append(math.log(rmse1 / rmse0) / math.log(dx1 / dx0))
    return out


def make_plot(results: list[dict], refs: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    colors = {0.04: "#1f77b4", 0.03: "#9467bd", 0.02: "#ff7f0e", 0.015: "#8c564b",
              0.01: "#2ca02c", 0.005: "#d62728"}
    for r in results:
        ax.plot(r["sim_tau"], r["sim_za"], "-", lw=2,
                color=colors.get(round(r["dx"], 4), "#333333"),
                label=f"TensorSPH dx={r['dx']:.3f} m (N={r['particles']})", zorder=3)

    b = refs["bundled"]
    c = refs["crosscheck"]
    ax.errorbar(b["tau"], b["za"], yerr=b["claimed_uncertainty"], fmt="ks",
                ms=5, capsize=3, label="bundled digitisation (claimed ±0.05)", zorder=5)
    ax.plot(c["tau"], c["za"], "o-", color="#d62728", ms=4, lw=1.2, alpha=0.8,
            label="independent transcription (cross-check)", zorder=4)
    # Shade the region where the two references disagree: that band is the
    # reference uncertainty the gate was treating as ±0.05.
    grid = [0.01 * i for i in range(int(c["tau"][-1] / 0.01) + 1)]
    lo = [_interp(c["tau"], c["za"], t) for t in grid]
    hi = [_interp(b["tau"], b["za"], t) for t in grid]
    ax.fill_between(grid, lo, hi, color="0.75", alpha=0.35,
                    label="cross-source spread = reference uncertainty", zorder=0)

    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=13)
    ax.set_ylabel(r"$Z/a$", fontsize=13)
    ax.set_title("B2 dam-break front: solver curves vs the two reference transcriptions",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="0.04,0.02")
    p.add_argument("--tank-width", type=float, default=2.0)
    p.add_argument("--column-width", type=float, default=0.2)
    p.add_argument("--sound-speed", type=float, default=25.0)
    p.add_argument("--density-diffusion", type=float, default=0.0)
    p.add_argument("--wall-friction", type=float, default=0.0)
    p.add_argument("--max-courant", type=float, default=0.3)
    p.add_argument("--adaptive-dt", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seconds", type=float, default=0.5)
    p.add_argument("--limit", type=float, default=0.15,
                   help="gate limit on the mean absolute error in Z/a")
    p.add_argument("--json-out", default=str(HERE / "reference_ladder.json"))
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-out", default=str(HERE / "reference_ladder.png"))
    args = p.parse_args()

    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    refs = {"bundled": load_bundled(), "crosscheck": load_crosscheck()}

    results = []
    for dx in sizes:
        rec = run_one(dx, args, refs)
        results.append(rec)
        print(f"dx={dx:.3f} N={rec['particles']:5d} wall={rec['wall_clock_s']:7.1f}s "
              f"MAE_bundled={rec['error_vs_bundled']['mae']:.4f} "
              f"MAE_crosscheck={rec['error_vs_crosscheck']['mae']:.4f}", flush=True)

    report = {
        "benchmark": "B2_dam_break_reference_ladder",
        "simulation_params": {
            "tank_width": args.tank_width,
            "column_width": args.column_width,
            "column_height": 2.0 * args.column_width,
            "sound_speed": args.sound_speed,
            "wall_friction": args.wall_friction,
            "density_diffusion": args.density_diffusion,
            "adaptive_dt": args.adaptive_dt,
            "seconds": args.seconds,
            "device": args.device,
        },
        "references": {
            "bundled": {"path": refs["bundled"]["path"],
                        "claimed_uncertainty": refs["bundled"]["claimed_uncertainty"]},
            "crosscheck": {"path": refs["crosscheck"]["path"],
                           "claimed_uncertainty": refs["crosscheck"]["claimed_uncertainty"]},
        },
        "results": results,
        "gates": {
            "bundled": gate_table(results, "bundled", args.limit),
            "crosscheck": gate_table(results, "crosscheck", args.limit),
        },
        "convergence": {
            "bundled": observed_order(results, "bundled"),
            "crosscheck": observed_order(results, "crosscheck"),
        },
        "plot_path": args.plot_out if args.plot else None,
    }
    Path(args.json_out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args.json_out}")
    for name, g in report["gates"].items():
        finest = g["finest"]
        print(f"  gate[{name}]: finest dx={finest['dx']:.3f} MAE={finest['mae']:.4f} "
              f"limit={g['limit']:.2f} -> {finest['result'].upper()}")
    if args.plot:
        make_plot(results, refs, args.plot_out)
        print(f"wrote {args.plot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
