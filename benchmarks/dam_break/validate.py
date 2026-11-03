"""Dam-break validation against Martin & Moyce (1952) experimental data.

This script runs the TensorSPH dam-break benchmark at multiple resolutions,
compares the surge-front trajectory against the classical Martin & Moyce (1952)
experimental data, and generates a validation report with comparison plots.

Usage:
    python benchmarks/dam_break/validate.py --sizes 0.04,0.02,0.01 --seconds 1.2
    python benchmarks/dam_break/validate.py --quick  # single coarse run

The validation gates are:
1. Mass conservation: |m(t_end) - m(0)| < 1e-10 kg
2. No particle escape: escaped == 0
3. Bounded speed: max|v| < 10 m/s
4. Density compression: max(rho/rho0 - 1) < 0.10
5. Experimental agreement: relative L2 error of Z/a vs Martin & Moyce < 15%

Note: The Martin & Moyce experiment used a semi-infinite tank, while our
simulation uses a finite tank (2.0 m). The comparison is valid only up to
the time when the surge front reaches the far wall.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time


from tensorsph.dam_break import DamBreakConfig, run_case


def load_experimental_data(path: str) -> list[dict]:
    """Load Martin & Moyce (1952) experimental data."""
    with open(path) as f:
        data = json.load(f)
    return data["data"]


def interpolate(times, values, query):
    """Linear interpolation."""
    if query <= times[0]:
        return values[0]
    if query >= times[-1]:
        return values[-1]
    for i in range(1, len(times)):
        if times[i] >= query:
            t0, t1 = times[i - 1], times[i]
            v0, v1 = values[i - 1], values[i]
            w = 0.0 if t1 == t0 else (query - t0) / (t1 - t0)
            return v0 + w * (v1 - v0)
    return values[-1]


def compute_relative_l2(sph_samples, exp_data, max_tau=None):
    """Compute relative L2 error between SPH and experimental Z/a vs tau.

    Only compares up to max_tau (to avoid wall reflection effects).
    """
    sph_taus = [s["tau"] for s in sph_samples]
    sph_z = [s["front_over_a"] for s in sph_samples]

    # Filter experimental data by max_tau
    if max_tau is not None:
        exp_filtered = [d for d in exp_data if d["tau"] <= max_tau]
    else:
        exp_filtered = list(exp_data)

    if not exp_filtered or not sph_taus:
        return None, 0

    exp_taus = [d["tau"] for d in exp_filtered]
    exp_z = [d["Z_over_a"] for d in exp_filtered]

    numerator = 0.0
    denominator = 0.0
    n_points = 0

    for tau, z_exp in zip(exp_taus, exp_z):
        z_sph = interpolate(sph_taus, sph_z, tau)
        numerator += (z_sph - z_exp) ** 2
        denominator += z_exp ** 2
        n_points += 1

    if denominator <= 0 or n_points == 0:
        return None, n_points

    return math.sqrt(numerator / denominator), n_points


def run_one(dx, args, seconds=None):
    """Run a single dam-break case."""
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
    span = args.seconds if seconds is None else seconds
    cfg.steps = int(round(span / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))

    started = time.perf_counter()
    run = run_case(cfg)
    elapsed = time.perf_counter() - started

    samples = []
    for t, z, ke, pe in zip(run.times, run.front, run.kinetic_energy, run.potential_energy):
        samples.append({
            "time": t,
            "tau": t * math.sqrt(2.0 * cfg.gravity / cfg.column_width),
            "front": z,
            "front_over_a": z / cfg.column_width,
            "kinetic_energy": ke,
            "potential_energy": pe,
        })

    return {
        "dx": dx,
        "dt": cfg.dt,
        "steps": cfg.steps,
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "elapsed_seconds": elapsed,
        "particles": int(run.final_positions.shape[0]),
        "mass_kg": run.mass,
        "mass_drift_kg": run.mass_drift,
        "max_speed": run.max_speed,
        "mach_number": run.max_speed / cfg.sound_speed,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "wall_impact_time": run.wall_impact_time,
        "samples": samples,
    }


def generate_comparison_plot(results, exp_data, output_path):
    """Generate Z/a vs tau comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot experimental data
    exp_taus = [d["tau"] for d in exp_data]
    exp_z = [d["Z_over_a"] for d in exp_data]
    ax.plot(exp_taus, exp_z, "ko", markersize=6, label="Martin & Moyce (1952)", zorder=5)

    # Plot SPH results
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, result in enumerate(results):
        taus = [s["tau"] for s in result["samples"]]
        z = [s["front_over_a"] for s in result["samples"]]
        color = colors[i % len(colors)]
        label = f"SPH dx={result['dx']:.3f} m (N={result['particles']})"
        ax.plot(taus, z, "-", color=color, linewidth=2, label=label)

    # Plot Ritter solution (reference only)
    ritter_taus = [t * 0.1 for t in range(0, 61)]
    ritter_z = [2.0 * tau for tau in ritter_taus]
    ax.plot(ritter_taus, ritter_z, "--", color="gray", linewidth=1, label="Ritter (shallow-water)")

    ax.set_xlabel(r"$\tau = t \sqrt{2g/a}$", fontsize=14)
    ax.set_ylabel(r"$Z/a$", fontsize=14)
    ax.set_title("Dam-break surge front: TensorSPH vs Martin & Moyce (1952)", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="0.04,0.02,0.01",
                        help="comma-separated particle spacings")
    parser.add_argument("--tank-width", type=float, default=2.0)
    parser.add_argument("--column-width", type=float, default=0.2)
    parser.add_argument("--sound-speed", type=float, default=25.0)
    parser.add_argument("--density-diffusion", type=float, default=0.0)
    parser.add_argument("--adaptive-dt", action="store_true")
    parser.add_argument("--wall-friction", type=float, default=0.0)
    parser.add_argument("--max-courant", type=float, default=0.3)
    parser.add_argument("--seconds", type=float, default=1.2)
    parser.add_argument("--max-mass-drift", type=float, default=1e-10)
    parser.add_argument("--max-mach", type=float, default=0.25)
    parser.add_argument("--max-density-error", type=float, default=0.10)
    parser.add_argument("--max-exp-error", type=float, default=0.15,
                        help="max relative L2 error vs Martin & Moyce (default 15%)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true", help="single coarse run")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--exp-data", default=None,
                        help="path to Martin & Moyce experimental data JSON")
    args = parser.parse_args()

    # Load experimental data
    exp_data_path = args.exp_data or os.path.join(
    os.path.dirname(__file__), "martin_moyce_1952.json")
    exp_data = load_experimental_data(exp_data_path)
    print(f"Loaded {len(exp_data)} experimental data points from {exp_data_path}")

    # Run SPH simulations
    sizes = [float(s) for s in args.sizes.split(",")]
    if args.quick:
        sizes = [sizes[0]]

    results = []
    for dx in sorted(sizes, reverse=True):
        print(f"\nRunning dx={dx:.3f} m...", flush=True)
        result = run_one(dx, args)
        results.append(result)
        print(f"  N={result['particles']}, {result['steps_taken']} steps, "
              f"{result['elapsed_seconds']:.1f}s, Mach={result['mach_number']:.3f}, "
              f"compression={result['max_density_compression']:.4f}")

    # Compute experimental comparison
    exp_comparison = []
    for result in results:
        max_tau = None
        if result["wall_impact_time"] is not None:
            max_tau = result["wall_impact_time"] * math.sqrt(
                2.0 * 9.81 / args.column_width
            )
        error, n_points = compute_relative_l2(result["samples"], exp_data, max_tau)
        exp_comparison.append({
            "dx": result["dx"],
            "particles": result["particles"],
            "max_tau_compared": max_tau,
            "n_points": n_points,
            "relative_l2_error": error,
        })

    # Gates
    gates = {}
    for result in results:
        dx = result["dx"]
        gates[f"mass_conservation_dx{dx}"] = {
            "metric": "|m(t_end) - m(0)|",
            "measured": abs(result["mass_drift_kg"]),
            "limit": args.max_mass_drift,
            "result": "pass" if abs(result["mass_drift_kg"]) < args.max_mass_drift else "fail",
        }
        gates[f"no_escape_dx{dx}"] = {
            "metric": "escaped particles",
            "measured": result["escaped"],
            "limit": 0,
            "result": "pass" if result["escaped"] == 0 else "fail",
        }
        gates[f"bounded_speed_dx{dx}"] = {
            "metric": "max|v|",
            "measured": result["max_speed"],
            "limit": 10.0,
            "result": "pass" if result["max_speed"] < 10.0 else "fail",
        }
        gates[f"density_compression_dx{dx}"] = {
            "metric": "max(rho/rho0 - 1)",
            "measured": result["max_density_compression"],
            "limit": args.max_density_error,
            "result": ("pass"
           if result["max_density_compression"] < args.max_density_error
           else "fail"),
        }

    for comp in exp_comparison:
        dx = comp["dx"]
        if comp["relative_l2_error"] is not None:
            gates[f"experimental_agreement_dx{dx}"] = {
                "metric": "relative L2 error of Z/a vs Martin & Moyce (1952)",
                "measured": comp["relative_l2_error"],
                "limit": args.max_exp_error,
                "result": "pass" if comp["relative_l2_error"] <= args.max_exp_error else "fail",
            }
        else:
            gates[f"experimental_agreement_dx{dx}"] = {
                "metric": "relative L2 error of Z/a vs Martin & Moyce (1952)",
                "measured": None,
                "limit": args.max_exp_error,
                "result": "not_evaluated",
            }

    # Generate plot
    plot_path = None
    if args.plot_out:
        plot_path = generate_comparison_plot(results, exp_data, args.plot_out)
        print(f"\nPlot saved to {plot_path}")

    # Report
    report = {
        "benchmark": "B2_dam_break_validation",
        "reference": "Martin & Moyce (1952), Phil. Trans. R. Soc. Lond. A 244:312-324",
        "geometry": {
            "column_width_a_m": args.column_width,
            "column_height_m": 2.0 * args.column_width,
            "tank_width_m": args.tank_width,
        },
        "numerical_setup": {
            "sound_speed": args.sound_speed,
            "max_courant": args.max_courant,
            "wall_friction": args.wall_friction,
            "density_diffusion": args.density_diffusion,
            "adaptive_dt": args.adaptive_dt,
        },
        "results": results,
        "experimental_comparison": exp_comparison,
        "gates": gates,
        "notes": [
            "The Martin & Moyce experiment used a semi-infinite tank; our simulation uses "
            "a finite tank (2.0 m). Comparison is valid only up to wall impact time.",
            "The Ritter shallow-water solution (Z/a = 2*tau) assumes a semi-infinite reservoir "
            "and shallow water; it is plotted as a reference but not used as a pass criterion.",
            "SPH density compression is measured as max(rho/rho0 - 1) over every integration step.",
            "Particle masses are constants; mass drift should be exactly zero.",
        ],
    }

    text = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text)
    print("\n" + text)

    # Check gates
    failed = [name for name, gate in gates.items() if gate["result"] == "fail"]
    if failed:
        raise SystemExit(f"ERROR: validation gates failed: {', '.join(failed)}")
    else:
        print("\n✅ All validation gates passed.")


if __name__ == "__main__":
    main()
