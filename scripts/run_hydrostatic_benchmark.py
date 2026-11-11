#!/usr/bin/env python
"""Standalone hydrostatic pressure profile benchmark.

Runs the SPH hydrostatic case and produces a self-contained JSON report
with ref-vs-got comparison, suitable for maturity-gate evidence.

Usage:
    python scripts/run_hydrostatic_benchmark.py --tier smoke
    python scripts/run_hydrostatic_benchmark.py --tier full
    python scripts/run_hydrostatic_benchmark.py --dx 0.02 --seconds 2.0

Output: artifacts/hydrostatic/report.json (and figures/)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.verification.hydrostatic import (
    HydrostaticCase,
    run_hydrostatic_case,
    zero_density_control,
    gate,
)
from tensorsph.vv.hydrostatic import run_hydrostatic, HydrostaticConfig


TIERS = {
    "smoke": dict(dx=0.05, tank_width=0.2, depth=0.2, seconds=0.2, min_particles=16),
    "full": dict(dx=0.02, tank_width=1.0, depth=0.4, seconds=2.0, min_particles=1000),
}


def run_benchmark(tier: str = "smoke", dx: float | None = None,
                  seconds: float | None = None, out_dir: str | None = None) -> dict:
    """Run the hydrostatic benchmark and return the report dict."""
    if tier in TIERS and dx is None:
        params = TIERS[tier]
    else:
        params = dict(
            dx=dx or 0.02,
            tank_width=1.0,
            depth=0.4,
            seconds=seconds or 2.0,
            min_particles=1000,
        )

    case = HydrostaticCase(**params)
    started = time.perf_counter()
    result = run_hydrostatic_case(case)
    elapsed = time.perf_counter() - started
    result["wall_seconds"] = round(elapsed, 2)

    # Zero density control
    control = zero_density_control()

    # Gate
    gate_result = gate(result, limit_pct=2.0 if tier == "full" else 5.0)

    # Reference solution
    ref = {
        "kind": "analytic",
        "solution": "p(y) = rho0 * g * (H - y)",
        "rho0": 1000.0,
        "g": 9.81,
        "H": case.depth,
        "source": "Hydrostatic equilibrium of incompressible fluid under gravity",
    }

    report = {
        "benchmark": "hydrostatic_pressure_profile",
        "tier": tier,
        "case": {
            "dx": case.dx,
            "tank_width": case.tank_width,
            "depth": case.depth,
            "seconds": case.seconds,
            "min_particles": case.min_particles,
            "hydrostatic_init": True,
        },
        "reference": ref,
        "result": result,
        "zero_density_control": control,
        "gate": gate_result,
        "repro_cmd": (
            f"cd {REPO} && PYTHONPATH=src python scripts/run_hydrostatic_benchmark.py "
            f"--tier {tier}"
        ),
        "impl_files": [
            "src/tensorsph/vv/hydrostatic.py",
            "src/tensorsph/verification/hydrostatic.py",
            "benchmarks/hydrostatic/run.py",
            "scripts/run_hydrostatic_benchmark.py",
        ],
        "entry": "python -m tensorsph.verification --tier full --out runs/hydrostatic_full",
    }

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--dx", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--out", default="artifacts/hydrostatic")
    args = parser.parse_args()

    report = run_benchmark(tier=args.tier, dx=args.dx, seconds=args.seconds, out_dir=args.out)

    # Print summary
    print(f"\n=== Hydrostatic Benchmark Summary ===")
    print(f"Tier: {report['tier']}")
    print(f"Particles: {report['result']['measured']['n_particles']}")
    print(f"Max interior band error: {report['result']['measured']['max_band_error_pct']:.2f}%")
    print(f"Gate limit: {report['gate']['limit_pct']}%")
    print(f"Gate: {'PASS' if report['gate']['pass'] else 'FAIL'}")
    print(f"Zero density control: {'PASS' if report['zero_density_control']['pass'] else 'FAIL'}")
    print(f"Wall time: {report['result']['wall_seconds']:.1f}s")

    return 0 if report['gate']['pass'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
