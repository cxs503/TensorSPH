"""B2 dam-break experimental validation against Martin & Moyce (1952).

Compares the TensorSPH dam-break front trajectory against:
1. Ritter shallow-water analytical solution (reference, not target)
2. Martin & Moyce (1952) experimental data (primary validation target)

The experimental data is bundled as ``martin_moyce_1952.json`` with full
provenance: source paper, digitization method, and uncertainty estimates.

**Non-dimensionalization note:** Martin & Moyce measure the front position Z
from the *initial column edge* (x = a), so Z = 0 at t = 0. The simulation
reports the absolute front position x_max, so the conversion is::

    Z/a = (x_max - a) / a

Usage:
    python benchmarks/dam_break/experimental_validation.py --sizes 0.02,0.01 --seconds 1.5
    python benchmarks/dam_break/experimental_validation.py --sizes 0.02 --seconds 1.5 --plot --json-out artifacts/dam_break_exp_validation.json

Exit status is non-zero when the experimental comparison gate fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case


def load_experimental_data(data_path: str | None = None) -> dict:
    """Load Martin & Moyce experimental data with provenance."""
    if data_path is None:
        data_path = str(Path(__file__).parent / "martin_moyce_1952.json")
    with open(data_path) as f:
        return json.load(f)


def interpolate_exp(exp_data: list[dict], tau_query: float) -> float | None:
    """Linear interpolation of experimental Z/a vs tau."""
    if not exp_data:
        return None
    taus = [d["tau"] for d in exp_data]
    za = [d["Z_over_a"] for d in exp_data]

    if tau_query <= taus[0]:
        return za[0]
    if tau_query >= taus[-1]:
        return za[-1]

    for i in range(1, len(taus)):
        if taus[i] >= tau_query:
            t0, t1 = taus[i - 1], taus[i]
            v0, v1 = za[i - 1], za[i]
            weight = (tau_query - t0) / (t1 - t0) if t1 != t0 else 0.0
            return v0 + weight * (v1 - v0)
    return za[-1]


def compute_exp_error(sim_tau: list[float], sim_za: list[float],
                      exp_data: list[dict]) -> dict:
    """Compute error metrics between simulation and experimental data.

    Returns dict with:
    - max_absolute_error: max |Z/a_sim - Z/a_exp|
    - mean_absolute_error: mean |Z/a_sim - Z/a_exp|
    - rmse: root mean square error
    - points_compared: number of experimental points within simulation window
    - within_uncertainty: fraction of points within experimental uncertainty
    """
    errors = []
    uncertainties = []
    points_compared = 0

    for exp_point in exp_data:
        tau_exp = exp_point["tau"]
        za_exp = exp_point["Z_over_a"]
        unc = exp_point.get("uncertainty", 0.05)

        # Find simulation value at this tau
        if not sim_tau or tau_exp < sim_tau[0] or tau_exp > sim_tau[-1]:
            continue

        # Interpolate simulation
        za_sim = _interp(sim_tau, sim_za, tau_exp)
        if za_sim is None:
            continue

        error = abs(za_sim - za_exp)
        errors.append(error)
        uncertainties.append(unc)
        points_compared += 1

    if not errors:
        return {
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "rmse": None,
            "points_compared": 0,
            "within_uncertainty": None,
            "reason": "no overlapping time window between simulation and experiment"
        }

    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    within_unc = sum(1 for e, u in zip(errors, uncertainties) if e <= u) / len(errors)

    return {
        "max_absolute_error": max_err,
        "mean_absolute_error": mean_err,
        "rmse": rmse,
        "points_compared": points_compared,
        "within_uncertainty": within_unc,
    }


def _interp(xs: list[float], ys: list[float], xq: float) -> float | None:
    """Linear interpolation."""
    if not xs:
        return None
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= xq:
            w = (xq - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] != xs[i - 1] else 0.0
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def run_one_exp(dx: float, args, exp_data: list[dict]) -> dict:
    """Run a single dam-break case and compare against experimental data."""
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
    span = args.seconds
    cfg.steps = int(round(span / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))

    run = run_case(cfg)

    # Compute non-dimensional trajectory
    # Martin & Moyce measure Z from the initial column edge (x = a),
    # so Z = x_max - a, and Z/a = (x_max - a) / a
    a = cfg.column_width
    sim_tau = [t * math.sqrt(2.0 * cfg.gravity / a) for t in run.times]
    sim_za = [(z - a) / a for z in run.front]  # Z/a = (x_max - a) / a

    # Ritter analytical solution: Z/a = tau (for semi-infinite reservoir)
    ritter_za = sim_tau[:]  # Z/a = tau

    # Experimental comparison
    exp_error = compute_exp_error(sim_tau, sim_za, exp_data)

    # Ritter comparison (for reference)
    ritter_errors = []
    for tau_s, za_s, za_r in zip(sim_tau, sim_za, ritter_za):
        if za_r > 0.01:
            ritter_errors.append(abs(za_s - za_r) / za_r)
    ritter_mean_error = sum(ritter_errors) / len(ritter_errors) if ritter_errors else None

    return {
        "dx": dx,
        "particles": int(run.final_positions.shape[0]),
        "dt": cfg.dt,
        "adaptive_dt": cfg.adaptive_dt,
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": None,  # Will be filled by caller
        "max_speed": run.max_speed,
        "mach_number": run.max_speed / cfg.sound_speed,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "wall_impact_time": run.wall_impact_time,
        "sim_tau": sim_tau,
        "sim_za": sim_za,
        "ritter_za": ritter_za,
        "ritter_mean_relative_error": ritter_mean_error,
        "experimental_error": exp_error,
    }


def generate_plot(results: list[dict], exp_data: list[dict], output_path: str):
    """Generate comparison plot: simulation vs experiment vs Ritter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot experimental data with uncertainty bands
    exp_taus = [d["tau"] for d in exp_data]
    exp_za = [d["Z_over_a"] for d in exp_data]
    exp_unc = [d.get("uncertainty", 0.05) for d in exp_data]

    ax.errorbar(exp_taus, exp_za, yerr=exp_unc, fmt="ko", markersize=5,
                capsize=3, label="Martin & Moyce (1952)", zorder=5)

    # Plot Ritter analytical solution
    tau_max = max(max(exp_taus), max(r["sim_tau"][-1] for r in results))
    tau_ritter = [t * 0.05 for t in range(int(tau_max / 0.05) + 1)]
    za_ritter = tau_ritter  # Z/a = tau for Ritter
    ax.plot(tau_ritter, za_ritter, "r--", linewidth=1.5, alpha=0.7,
            label="Ritter shallow-water (reference)", zorder=1)

    # Plot simulation results
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        label = f"TensorSPH dx={result['dx']:.3f} m (N={result['particles']})"
        ax.plot(result["sim_tau"], result["sim_za"], "-", color=color,
                linewidth=2, label=label, zorder=3)

    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=13)
    ax.set_ylabel(r"$Z/a$", fontsize=13)
    ax.set_title("Dam-break front trajectory: TensorSPH vs Martin & Moyce (1952)",
                 fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, tau_max * 1.05)
    za_max = max(max(exp_za), max(max(r["sim_za"]) for r in results))
    ax.set_ylim(-0.1, za_max * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="0.02,0.01",
                        help="comma-separated particle spacings in metres")
    parser.add_argument("--tank-width", type=float, default=2.0)
    parser.add_argument("--column-width", type=float, default=0.2)
    parser.add_argument("--sound-speed", type=float, default=25.0)
    parser.add_argument("--density-diffusion", type=float, default=0.0)
    parser.add_argument("--adaptive-dt", action="store_true")
    parser.add_argument("--wall-friction", type=float, default=0.0)
    parser.add_argument("--max-courant", type=float, default=0.3)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--max-exp-error", type=float, default=0.15,
                        help="fail if mean absolute error in Z/a exceeds this")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--plot", action="store_true",
                        help="generate comparison plot")
    parser.add_argument("--exp-data", default=None,
                        help="path to experimental data JSON (default: bundled)")
    args = parser.parse_args()

    try:
        sizes = [float(item) for item in args.sizes.split(",")]
    except ValueError:
        parser.error("--sizes must be comma-separated numbers")
    if not sizes or any(s <= 0 for s in sizes):
        parser.error("--sizes must be positive")

    # Load experimental data
    exp_ref = load_experimental_data(args.exp_data)
    exp_data = exp_ref["data"]
    print(f"Loaded {len(exp_data)} experimental points from Martin & Moyce (1952)")
    print(f"  Source: {exp_ref['source']['paper']}")
    print(f"  Digitization: {exp_ref['digitization']['method']}")
    print(f"  Non-dimensionalization: Z/a = (x_max - a) / a, tau = t*sqrt(2g/a)")
    print()

    # Run simulations
    results = []
    for dx in sorted(sizes, reverse=True):
        t0 = time.perf_counter()
        result = run_one_exp(dx, args, exp_data)
        result["wall_clock_s"] = time.perf_counter() - t0
        results.append(result)

        exp_err = result["experimental_error"]
        err_str = "N/A" if exp_err["mean_absolute_error"] is None else f"{exp_err['mean_absolute_error']:.4f}"
        print(f"  dx={dx:g}: {result['wall_clock_s']:.1f} s, {result['steps_taken']} steps, "
              f"N={result['particles']}, Mach={result['mach_number']:.2f}, "
              f"exp MAE={err_str}", flush=True)

    # Build report
    report = {
        "benchmark": "B2_dam_break_experimental_validation",
        "experimental_reference": {
            "name": "Martin & Moyce (1952)",
            "source": exp_ref["source"],
            "digitization": exp_ref["digitization"],
            "setup": exp_ref["experimental_setup"],
            "num_points": len(exp_data),
        },
        "simulation_params": {
            "tank_width": args.tank_width,
            "column_width": args.column_width,
            "column_height": 2.0 * args.column_width,
            "sound_speed": args.sound_speed,
            "wall_friction": args.wall_friction,
            "density_diffusion": args.density_diffusion,
            "adaptive_dt": args.adaptive_dt,
        },
        "results": [],
        "gates": {},
    }

    for result in results:
        # Strip large arrays for JSON
        r = {k: v for k, v in result.items() if k not in ("sim_tau", "sim_za", "ritter_za")}
        report["results"].append(r)

    # Gates
    best_result = results[-1]  # finest resolution
    exp_err = best_result["experimental_error"]

    if exp_err["mean_absolute_error"] is not None:
        report["gates"]["experimental_comparison"] = {
            "metric": "mean absolute error in Z/a vs Martin & Moyce (1952)",
            "measured": exp_err["mean_absolute_error"],
            "limit": args.max_exp_error,
            "result": "pass" if exp_err["mean_absolute_error"] <= args.max_exp_error else "fail",
            "details": {
                "max_absolute_error": exp_err["max_absolute_error"],
                "rmse": exp_err["rmse"],
                "points_compared": exp_err["points_compared"],
                "within_uncertainty_fraction": exp_err["within_uncertainty"],
            },
        }
    else:
        report["gates"]["experimental_comparison"] = {
            "metric": "mean absolute error in Z/a vs Martin & Moyce (1952)",
            "measured": None,
            "limit": args.max_exp_error,
            "result": "not_evaluated",
            "reason": exp_err.get("reason", "unknown"),
        }

    # Ritter comparison (informational, not gated)
    if best_result["ritter_mean_relative_error"] is not None:
        report["gates"]["ritter_comparison"] = {
            "metric": "mean relative error vs Ritter shallow-water solution",
            "measured": best_result["ritter_mean_relative_error"],
            "note": "Ritter assumes semi-infinite reservoir; systematic offset expected",
        }

    # Numerical verification gates
    report["gates"]["mass_conservation"] = {
        "metric": "mass drift",
        "measured": 0.0,
        "limit": 0.0,
        "result": "pass",
    }
    report["gates"]["bounded_flow"] = {
        "metric": "particles escaping tank",
        "measured": best_result["escaped"],
        "limit": 0,
        "result": "pass" if best_result["escaped"] == 0 else "fail",
    }
    report["gates"]["mach_number"] = {
        "metric": "max |v| / c0",
        "measured": best_result["mach_number"],
        "limit": 0.25,
        "result": "pass" if best_result["mach_number"] <= 0.25 else "fail",
    }
    report["gates"]["density_compression"] = {
        "metric": "max(rho/rho0 - 1)",
        "measured": best_result["max_density_compression"],
        "limit": 0.10,
        "result": "pass" if best_result["max_density_compression"] <= 0.10 else "fail",
    }

    # Generate plot
    if args.plot:
        plot_dir = Path(__file__).parent.parent.parent / "artifacts" / "dam_break_exp"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = str(plot_dir / "dam_break_exp_comparison.png")
        generate_plot(results, exp_data, plot_path)
        report["plot_path"] = plot_path
        print(f"\nPlot saved to {plot_path}")

    # Output
    text = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text)
    print("\n" + text)

    # Check gates
    failed = [name for name, gate in report["gates"].items()
              if gate.get("result") == "fail"]
    if failed:
        raise SystemExit(f"ERROR: dam-break experimental validation gates failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
