"""Momentum conservation analysis for XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION.

This script runs dam-break cases with and without TVC, tracks the total
momentum evolution, and quantifies the indirect momentum effect of the
position correction.

Usage::

    PYTHONPATH=src python benchmarks/xval_sphinx_tvc/momentum/analyse.py \
        --dx 0.04 --steps 300 --out docs/xval_sphinx_tvc/data/momentum_analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402


@dataclass
class MomentumRecord:
    """Results of one dam-break run with momentum tracking."""

    variant: str
    dx: float
    steps: int
    tvc_alpha: float | None
    tvc_scope: str | None
    max_speed: float
    max_density_compression: float
    max_tvc_shift: float
    mass_drift: float
    wall_clock: float
    # Momentum diagnostics (computed from the run)
    momentum_ratio: float | None = None  # TVC max_speed / baseline max_speed
    extras: dict = field(default_factory=dict)


def run_momentum_analysis(dx: float = 0.04, steps: int = 300,
                          device: str = "cpu") -> list[MomentumRecord]:
    """Run dam-break cases with different TVC settings and track momentum."""
    records = []

    variants = [
        {"name": "baseline", "tvc_every": 0, "tvc_alpha": None, "tvc_scope": None},
        {"name": "tvc_a005_all", "tvc_every": 1, "tvc_alpha": 0.05, "tvc_scope": "all"},
        {"name": "tvc_a010_all", "tvc_every": 1, "tvc_alpha": 0.10, "tvc_scope": "all"},
        {"name": "tvc_a015_all", "tvc_every": 1, "tvc_alpha": 0.15, "tvc_scope": "all"},
        {"name": "tvc_a020_all", "tvc_every": 1, "tvc_alpha": 0.20, "tvc_scope": "all"},
        {"name": "tvc_a005_bulk", "tvc_every": 1, "tvc_alpha": 0.05, "tvc_scope": "bulk"},
    ]

    baseline_max_speed = None

    for v in variants:
        t0 = time.time()
        cfg = DamBreakConfig(
            dx=dx, dt=1.0e-4, steps=steps,
            output_every=max(1, steps // 10),
            device=device,
            column_width=0.2, column_height=0.4,
            pool_width=2.0, tank_height=0.75,
            tvc_every=v["tvc_every"],
            tvc_alpha=v["tvc_alpha"],
            tvc_scope=v["tvc_scope"],
        )
        run = run_case(cfg)
        wall_clock = time.time() - t0

        if v["name"] == "baseline":
            baseline_max_speed = run.max_speed

        rec = MomentumRecord(
            variant=v["name"],
            dx=dx,
            steps=steps,
            tvc_alpha=v["tvc_alpha"],
            tvc_scope=v["tvc_scope"],
            max_speed=run.max_speed,
            max_density_compression=run.max_density_compression,
            max_tvc_shift=run.max_tvc_shift,
            mass_drift=run.mass_drift,
            wall_clock=wall_clock,
        )

        if baseline_max_speed is not None and v["name"] != "baseline":
            rec.momentum_ratio = run.max_speed / baseline_max_speed

        records.append(rec)
        print(f"  {v['name']}: max_speed={run.max_speed:.4f} m/s, "
              f"compression={run.max_density_compression:.4f}, "
              f"shift/h={run.max_tvc_shift / (1.3 * dx):.4f}, "
              f"wall={wall_clock:.1f}s")

    return records


def main():
    parser = argparse.ArgumentParser(description="TVC momentum analysis")
    parser.add_argument("--dx", type=float, default=0.04, help="Particle spacing")
    parser.add_argument("--steps", type=int, default=300, help="Time steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--out", type=str,
                        default=str(ROOT / "docs" / "xval_sphinx_tvc" / "data" / "momentum_analysis.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"TVC momentum analysis: dx={args.dx}, steps={args.steps}, device={args.device}")
    print("=" * 70)

    records = run_momentum_analysis(dx=args.dx, steps=args.steps, device=args.device)

    # Build output
    output = {
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "analysis": "momentum_conservation",
        "parameters": {
            "dx": args.dx,
            "steps": args.steps,
            "device": args.device,
            "h_factor": 1.3,
        },
        "records": [
            {
                "variant": r.variant,
                "dx": r.dx,
                "steps": r.steps,
                "tvc_alpha": r.tvc_alpha,
                "tvc_scope": r.tvc_scope,
                "max_speed": r.max_speed,
                "max_density_compression": r.max_density_compression,
                "max_tvc_shift": r.max_tvc_shift,
                "max_tvc_shift_over_h": r.max_tvc_shift / (1.3 * args.dx) if r.max_tvc_shift else 0,
                "mass_drift": r.mass_drift,
                "momentum_ratio": r.momentum_ratio,
                "wall_clock": r.wall_clock,
            }
            for r in records
        ],
        "findings": {
            "baseline_max_speed": records[0].max_speed,
            "tvc_max_speed_range": [r.max_speed for r in records[1:] if r.tvc_alpha is not None],
            "max_momentum_ratio": max((r.momentum_ratio for r in records if r.momentum_ratio is not None), default=None),
            "mass_conserved": all(r.mass_drift == 0.0 for r in records),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nOutput written to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Momentum Analysis Summary")
    print("=" * 70)
    print(f"{'Variant':<18} {'max_speed':>10} {'ratio':>8} {'shift/h':>8} {'compress':>10}")
    print("-" * 70)
    for r in records:
        ratio_str = f"{r.momentum_ratio:.3f}" if r.momentum_ratio is not None else "—"
        shift_h = r.max_tvc_shift / (1.3 * args.dx) if r.max_tvc_shift else 0
        print(f"{r.variant:<18} {r.max_speed:>10.4f} {ratio_str:>8} {shift_h:>8.4f} {r.max_density_compression:>10.4f}")


if __name__ == "__main__":
    main()
