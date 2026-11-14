#!/usr/bin/env python3
"""Run the Koshizuka & Oka (1996) dam-break pressure peak benchmark.

Usage:
    python benchmarks/koshizuka_dam_break/run.py [--dx 0.005] [--quick]

Reference:
    Koshizuka, S. & Oka, Y. (1996). Moving-particle semi-implicit method for
    fragmentation of incompressible fluid. Nuclear Science and Engineering,
    123(3), 421-434. DOI: 10.13182/NSE96-A24205.

Reference pressure peak: 2700 Pa (from Fig. 15, probe at y = 0.024 m)
Tolerance: <12% per task specification.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from benchmarks.koshizuka_dam_break.koshizuka import (
    KoshizukaConfig,
    KoshizukaRun,
    run_koshizuka,
    KOSHIZUKA_REF_PRESSURE_PA,
    KOSHIZUKA_REF_PRESSURE_UNCERTAINTY,
)


def run_one(dx: float, quick: bool = False) -> dict:
    """Run a single resolution and return results."""
    cfg = KoshizukaConfig(
        dx=dx,
        sound_speed=30.0,
        max_courant=0.2,
        hydrostatic_init=True,
        wall_friction=0.0,
        density_diffusion=0.1,
        adaptive_dt=True,
    )
    cfg.dt = cfg.dt_from_courant()

    if quick:
        cfg.steps = 2000
        cfg.output_every = 50
    else:
        cfg.steps = int(0.4 / cfg.dt) + 100
        cfg.output_every = 10

    started = time.perf_counter()
    run = run_koshizuka(cfg)
    elapsed = time.perf_counter() - started

    # Compute error vs reference
    ref_pressure = KOSHIZUKA_REF_PRESSURE_PA
    got_pressure = run.peak_pressure
    error_pct = abs(got_pressure - ref_pressure) / ref_pressure * 100 if ref_pressure > 0 else float("nan")
    passes = error_pct < KOSHIZUKA_REF_PRESSURE_UNCERTAINTY * 100

    return {
        "dx": dx,
        "dt": cfg.dt,
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "elapsed_seconds": elapsed,
        "particles": int(run.initial_mass / (cfg.rho0 * dx * dx)),
        "peak_pressure_pa": got_pressure,
        "peak_pressure_time_s": run.peak_pressure_time,
        "reference_pressure_pa": ref_pressure,
        "error_pct": error_pct,
        "passes_tolerance": passes,
        "tolerance_pct": KOSHIZUKA_REF_PRESSURE_UNCERTAINTY * 100,
        "mass_drift_kg": run.mass_drift,
        "max_speed_ms": run.max_speed,
        "max_density_compression": run.max_density_compression,
        "max_acoustic_courant": run.max_acoustic_courant,
        "escaped": run.escaped,
        "wall_impact_time": run.wall_impact_time,
        "pressure_history": list(zip(run.pressure_probe_times, run.pressure_probe)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.005,
                        help="particle spacing in metres (default: 0.005)")
    parser.add_argument("--quick", action="store_true",
                        help="short run for smoke testing")
    parser.add_argument("--json-out", default=None,
                        help="write results to JSON file")
    args = parser.parse_args()

    print(f"Koshizuka & Oka (1996) dam-break pressure peak benchmark")
    print(f"Reference: P_max = {KOSHIZUKA_REF_PRESSURE_PA} Pa (Fig. 15, y = 0.024 m)")
    print(f"Tolerance: <{KOSHIZUKA_REF_PRESSURE_UNCERTAINTY * 100:.0f}%")
    print(f"dx = {args.dx} m, quick = {args.quick}")
    print()

    result = run_one(args.dx, args.quick)

    print(f"  Peak pressure: {result['peak_pressure_pa']:.1f} Pa "
          f"(ref: {result['reference_pressure_pa']:.0f} Pa, "
          f"error: {result['error_pct']:.1f}%)")
    print(f"  Peak time: {result['peak_pressure_time_s']:.4f} s")
    print(f"  Particles: {result['particles']}")
    print(f"  Simulated time: {result['simulated_time']:.4f} s")
    print(f"  Wall clock: {result['elapsed_seconds']:.1f} s")
    print(f"  Mass drift: {result['mass_drift_kg']:.6e} kg")
    print(f"  Max speed: {result['max_speed_ms']:.2f} m/s")
    print(f"  Max compression: {result['max_density_compression']:.4f}")
    print(f"  Escaped: {result['escaped']}")
    print(f"  Passes tolerance: {'YES ✓' if result['passes_tolerance'] else 'NO ✗'}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.json_out}")

    return 0 if result["passes_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main())
