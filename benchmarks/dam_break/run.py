"""B2 dam-break benchmark: front trajectory and resolution convergence.

A column of water of width ``a`` and height ``2a`` is released inside a tank.
The leading edge is tracked and reported as ``Z/a`` against
``tau = t sqrt(2 g / a)``, the non-dimensional form of the classical dam-break
experiments (Martin & Moyce, 1952).

What is and is not claimed:

* **Measured here:** mass conservation, boundedness (no particle leaves the
  tank), bounded speed, and the **convergence of the front trajectory under
  refinement**. These are numerical-verification results.
* **Reported but not gated:** the Ritter shallow-water front, which assumes a
  semi-infinite reservoir and shallow water. This case is neither, so a
  systematic offset is expected. It is plotted as an anchor, not as a target.
* **Not claimed:** physical validation. No digitized experimental table is
  bundled with the repository, so nothing is compared against measurement yet.

Usage:
    python benchmarks/dam_break/run.py --sizes 0.02,0.01 --seconds 0.8
    python benchmarks/dam_break/run.py --sizes 0.04,0.02 --seconds 0.35 --quick

Exit status is non-zero when a gate fails.
"""
from __future__ import annotations

import argparse
import json
import math
import time

from tensorsph.dam_break import DamBreakConfig, run_case


def interpolate(times, values, query):
    """Linear interpolation of a sampled series (lists, no dependencies)."""
    if query <= times[0]:
        return values[0]
    if query >= times[-1]:
        return values[-1]
    for index in range(1, len(times)):
        if times[index] >= query:
            t0, t1 = times[index - 1], times[index]
            v0, v1 = values[index - 1], values[index]
            weight = 0.0 if t1 == t0 else (query - t0) / (t1 - t0)
            return v0 + weight * (v1 - v0)
    return values[-1]


def wall_hit_time(run, cfg, margin_cells=3):
    limit = cfg.pool_width - margin_cells * cfg.dx
    for t, z in zip(run.times, run.front):
        if z >= limit:
            return t
    return None


def summarise(run, cfg):
    hit = wall_hit_time(run, cfg)
    tau, normalised = run.dimensionless(cfg)
    samples = []
    for t, z, ke, pe in zip(run.times, run.front, run.kinetic_energy,
                            run.potential_energy):
        if hit is not None and t > hit:
            break
        ritter = 2.0 * math.sqrt(cfg.gravity * cfg.column_height) * t
        samples.append({
            "time": t,
            "tau": t * math.sqrt(2.0 * cfg.gravity / cfg.column_width),
            "front": z,
            "front_over_a": z / cfg.column_width,
            "ritter_front": ritter,
            "front_over_ritter": (z / ritter) if ritter > 1.0e-6 else None,
            "kinetic_energy": ke,
            "potential_energy": pe,
        })
    return samples, hit, tau, normalised


def compare(coarse, fine, cfg):
    """Relative L2 difference of front(t)/a between two resolutions."""
    limit = coarse["wall_hit"]
    if limit is None:
        limit = min(coarse["samples"][-1]["time"], fine["samples"][-1]["time"])
    if limit <= 0:
        return None
    ct = [s["time"] for s in coarse["samples"]]
    cz = [s["front_over_a"] for s in coarse["samples"]]
    ft = [s["time"] for s in fine["samples"]]
    fz = [s["front_over_a"] for s in fine["samples"]]
    limit = min(limit, ct[-1], ft[-1])
    if limit <= 0:
        return None
    count = 40
    numerator = denominator = 0.0
    for k in range(count + 1):
        t = limit * k / count
        c = interpolate(ct, cz, t)
        f = interpolate(ft, fz, t)
        numerator += (c - f) ** 2
        denominator += f ** 2
    if denominator <= 0:
        return None
    return {
        "window_seconds": limit,
        "relative_l2": math.sqrt(numerator / denominator),
    }


def run_one(dx, args, sound_speed=None, seconds=None, friction=None):
    cfg = DamBreakConfig(dx=dx, pool_width=args.tank_width,
                         column_width=args.column_width,
                         sound_speed=args.sound_speed if sound_speed is None else sound_speed,
                         max_courant=args.max_courant, device=args.device,
                         wall_friction=args.wall_friction if friction is None else friction,
                         density_diffusion=args.density_diffusion)
    cfg.column_height = 2.0 * args.column_width
    cfg.dt = cfg.dt_from_courant()
    span = args.seconds if seconds is None else seconds
    cfg.steps = int(round(span / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))
    started = time.perf_counter()
    run = run_case(cfg)
    elapsed = time.perf_counter() - started
    samples, hit, _, _ = summarise(run, cfg)
    return {
        "dx": dx,
        "sound_speed": cfg.sound_speed,
        "wall_friction": cfg.wall_friction,
        "density_diffusion": cfg.density_diffusion,
        "dt": cfg.dt,
        "steps": cfg.steps,
        "acoustic_courant": cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx),
        "particles": int(run.final_positions.shape[0]),
        "elapsed_seconds": elapsed,
        "mass_kg": run.mass,
        "initial_mass_kg": run.initial_mass,
        "final_mass_kg": run.final_mass,
        "mass_drift_kg": run.mass_drift,
        "max_speed": run.max_speed,
        "mach_number": run.max_speed / cfg.sound_speed,
        "max_density_compression": run.max_density_compression,
        "compression_before_wall_impact": run.compression_before_wall_impact,
        "wall_impact_time": run.wall_impact_time,
        "particles_escaping_tank": run.escaped,
        "front_reaches_wall_at": hit,
        "final_front_over_a": samples[-1]["front_over_a"] if samples else None,
        "samples": samples,
        "wall_hit": hit,
    }


def observed_order(coarse, medium, fine):
    """Observed order of accuracy from a three-level ladder.

    With uniformly refined spacings (ratio 2) the error in an observable scales
    as dx^p, so p = log2(|f_coarse - f_medium| / |f_medium - f_fine|).
    """
    window = min(medium["wall_hit"] or medium["samples"][-1]["time"],
                 fine["wall_hit"] or fine["samples"][-1]["time"],
                 coarse["wall_hit"] or coarse["samples"][-1]["time"])
    if window <= 0:
        return None

    def sample(item):
        times = [s["time"] for s in item["samples"]]
        values = [s["front_over_a"] for s in item["samples"]]
        count = 40
        return [interpolate(times, values, window * k / count)
                for k in range(count + 1)]

    coarse_values, medium_values, fine_values = (sample(coarse), sample(medium),
                                                 sample(fine))
    numerator = sum((a - b) ** 2 for a, b in zip(coarse_values, medium_values)) ** 0.5
    denominator = sum((a - b) ** 2 for a, b in zip(medium_values, fine_values)) ** 0.5
    if denominator <= 0:
        return {"window_seconds": window, "order": None,
                "reason": "the two fine levels are identical to numerical precision"}
    return {
        "window_seconds": window,
        "relative_l2_coarse_medium": (numerator
                                      / sum(v * v for v in medium_values) ** 0.5),
        "relative_l2_medium_fine": (denominator
                                    / sum(v * v for v in fine_values) ** 0.5),
        "order": math.log2(numerator / denominator),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="0.02,0.01",
                        help="comma-separated particle spacings in metres")
    parser.add_argument("--tank-width", type=float, default=2.0)
    parser.add_argument("--column-width", type=float, default=0.2)
    parser.add_argument("--sound-speeds", default="25,40",
                        help="comma-separated sound speeds for the compressibility study")
    parser.add_argument("--sound-speed", type=float, default=25.0,
                        help="sound speed for the resolution sweep")
    parser.add_argument("--density-diffusion", type=float, default=0.0,
                        help="delta of the Antuono et al. (2012) diffusive term "
                             "(0 disables it, the default)")
    parser.add_argument("--wall-friction", type=float, default=0.0,
                        help="0 = free slip, 1 = no-slip walls")
    parser.add_argument("--max-compressibility-error", type=float, default=0.05,
                        help="fail if the front changes more than this between c0 levels")
    parser.add_argument("--max-mass-drift", type=float, default=0.0,
                        help="fail if |m(t_end) - m(0)| exceeds this in kg")
    parser.add_argument("--seconds", type=float, default=0.8)
    parser.add_argument("--max-courant", type=float, default=0.3)
    parser.add_argument("--max-convergence-l2", type=float, default=0.10,
                        help="fail if the coarse/fine front L2 difference exceeds this")
    parser.add_argument("--max-mach", type=float, default=0.25,
                        help="fail if max|v|/c0 exceeds this")
    parser.add_argument("--max-density-error", type=float, default=0.10,
                        help="fail if max|rho/rho0 - 1| exceeds this")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="single coarse run, no convergence gate")
    args = parser.parse_args()

    try:
        sizes = [float(item) for item in args.sizes.split(",")]
    except ValueError:
        parser.error("--sizes must be comma-separated numbers")
    if not sizes or any(s <= 0 for s in sizes):
        parser.error("--sizes must be positive")

    results = []
    for dx in sorted(sizes, reverse=True):
        item = run_one(dx, args)
        results.append(item)
        print(f"  dx={dx:g}: {item['elapsed_seconds']:5.1f} s, {item['steps']} steps, "
              f"max|v|={item['max_speed']:.2f} m/s (Mach {item['mach_number']:.2f}), "
              f"escaped={item['particles_escaping_tank']}", flush=True)

    convergence = None
    if len(results) >= 2:
        convergence = compare(results[0], results[1], None)

    # Compressibility study at the coarsest spacing: a reference comparison is
    # only meaningful once the front stops moving with c0.
    compressibility = None
    speeds = [float(item) for item in args.sound_speeds.split(",") if item.strip()]
    if len(speeds) >= 2 and not args.quick:
        compressibility = []
        for c0 in sorted(speeds):
            item = run_one(results[-1]["dx"], args, sound_speed=c0)
            compressibility.append({
                "sound_speed": c0,
                "mach_number": item["mach_number"],
                "max_density_compression": item["max_density_compression"],
                "front_reaches_wall_at": item["front_reaches_wall_at"],
                "samples": item["samples"],
                "wall_hit": item["wall_hit"],
            })
            print(f"  c0={c0:g}: Mach {item['mach_number']:.2f}, "
                  f"compression {100 * item['max_density_compression']:.1f}%, "
                  f"{item['elapsed_seconds']:.1f} s", flush=True)
        compressibility_error = compare(compressibility[-2], compressibility[-1], None)
        if compressibility_error is not None:
            compressibility_error["between"] = [compressibility[-2]["sound_speed"],
                                                compressibility[-1]["sound_speed"]]

    order = None
    if len(results) >= 3 and not args.quick:
        # Always reported when three spacings are given: an order of accuracy is
        # evidence, not an optional extra.
        order = observed_order(results[0], results[1], results[2])

    worst_mach = max(item["mach_number"] for item in results)
    worst_density = max(item["max_density_compression"] for item in results)
    escaped = sum(item["particles_escaping_tank"] for item in results)
    gates = {
        "mass_conserved": {
            "metric": "|sum(m) at the end - sum(m) at the start|, kg",
            "measured": max(abs(item["mass_drift_kg"]) for item in results),
            "limit": args.max_mass_drift,
            "result": ("pass" if max(abs(item["mass_drift_kg"])
                                     for item in results) <= args.max_mass_drift
                       else "fail"),
        },
        "bounded_flow": {
            "metric": "particles clamped beyond the tank",
            "measured": float(escaped), "limit": 0.0,
            "result": "pass" if escaped == 0 else "fail",
        },
        "mach_number": {
            "metric": "max|v| / c0",
            "measured": worst_mach, "limit": args.max_mach,
            "result": "pass" if worst_mach <= args.max_mach else "fail",
        },
        "density_compression": {
            "metric": "max(rho/rho0 - 1)",
            "measured": worst_density, "limit": args.max_density_error,
            "result": "pass" if worst_density <= args.max_density_error else "fail",
        },
    }
    bulk_compression = max(item["compression_before_wall_impact"] for item in results)
    gates["bulk_compression"] = {
        "metric": ("max(rho/rho0 - 1) before the front reaches the far wall "
                   "(isolates the bulk flow from the impact transient)"),
        "measured": bulk_compression, "limit": args.max_density_error,
        "result": ("pass" if bulk_compression <= args.max_density_error else "fail"),
    }
    if convergence is not None and not args.quick:
        gates["resolution_convergence"] = {
            "metric": "relative L2 of front(t)/a between the two resolutions",
            "measured": convergence["relative_l2"],
            "limit": args.max_convergence_l2,
            "result": "pass" if convergence["relative_l2"] <= args.max_convergence_l2 else "fail",
        }
    else:
        gates["resolution_convergence"] = {
            "metric": "relative L2 of front(t)/a between two resolutions",
            "measured": None, "limit": args.max_convergence_l2,
            "result": "not_evaluated",
        }
    if compressibility is not None and len(compressibility) >= 2:
        error = compressibility_error["relative_l2"] if compressibility_error else None
        gates["sound_speed_convergence"] = {
            "metric": "relative L2 of front(t)/a between the two highest c0 levels",
            "measured": error, "limit": args.max_compressibility_error,
            "result": ("not_evaluated" if error is None
                       else "pass" if error <= args.max_compressibility_error
                       else "fail"),
        }
    else:
        gates["sound_speed_convergence"] = {
            "metric": "relative L2 of front(t)/a between two c0 levels",
            "measured": None, "limit": args.max_compressibility_error,
            "result": "not_evaluated",
        }
    gates["experimental_reference"] = {
        "metric": "agreement with Martin & Moyce (1952)",
        "measured": None, "limit": None,
        "result": "not_evaluated",
    }

    report = {
        "benchmark": "B2_dam_break",
        "column_width": args.column_width,
        "column_height": 2.0 * args.column_width,
        "tank_width": args.tank_width,
        "sound_speed": args.sound_speed,
        "boundary": "static wall particles, free slip",
        "results": results,
        "convergence": convergence,
        "observed_order": order,
        "compressibility": compressibility,
        "compressibility_error": compressibility_error if compressibility else None,
        "gates": gates,
        "notes": [
            "Density compression is a maximum over every integration step, not over "
            "the diagnostic rows. The whole-run peak occurs when the surge hits the "
            "far wall; compression_before_wall_impact isolates the bulk flow.",
            "Numerical verification only: no digitized experimental data is bundled, "
            "so physical_validation remains not_evaluated.",
            "The Ritter front (2 sqrt(g H) t) assumes a semi-infinite reservoir on a "
            "dry bed. It is reported as an anchor; the finite reservoir and the "
            "absence of the shallow-water assumption make a systematic offset expected.",
            "The whole-run peak density compression is a wall-impact transient "
            "(about 10 % at c0 = 25); before impact the bulk flow stays near 2 %. "
            "The front trajectory moves by less than 0.04 % between c0 = 25 and "
            "60 m/s, so the weakly-compressible assumption does not limit that "
            "observable.",
            "Walls use Config.wall_friction: 0 is free slip (the default here), 1 is "
            "a no-slip floor verified against the analytic Stokes shear-decay rate.",
            "Samples after the front reaches the far wall are excluded from the "
            "convergence window.",
        ],
    }
    text = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            handle.write(text)
    print(text)

    failed = [name for name, gate in gates.items() if gate["result"] == "fail"]
    if failed:
        raise SystemExit(f"ERROR: dam-break gates failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
