"""Dump field data for the wave-tank figures.

The benchmark driver stores numbers, not fields: a full particle cloud per
snapshot would make the JSON evidence files useless. This script runs two short
cases and writes ``.npz`` files that ``make_figures.py`` turns into the field and
profile panels of the report.

    python benchmarks/wave_tank/viz_case.py --case regular
    python benchmarks/wave_tank/viz_case.py --case solitary
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorsph.wave_tank import (                                                # noqa: E402
    WaveTankConfig,
    run_tank,
    solitary_beta,
    solitary_crest_offset,
    solitary_elevation,
)

DEPTH = 0.5


def regular_case() -> WaveTankConfig:
    wavelength = 2.0 * math.pi * DEPTH / 3.0
    cfg = WaveTankConfig(
        depth=DEPTH, dx=0.02, length=2.4 * wavelength, kh=3.0, wave_height=0.05,
        max_courant=0.45, maker="relaxation", ramp_periods=0.5, viscosity=1.0e-4,
        probe_offsets=(0.4, 0.7, 1.0, 1.3), sponge_wavelengths=0.9,
        output_every=5,
    )
    cfg.end_time = 2.4 * cfg.period()
    cfg.steps = int(math.ceil(cfg.end_time / cfg.dt))
    return cfg


def solitary_case() -> WaveTankConfig:
    cfg = WaveTankConfig(
        depth=DEPTH, dx=0.02, length=5.0, kh=3.0, wave_height=0.1,
        wave_kind="solitary", maker="none", max_courant=0.45, output_every=10,
        viscosity=1.0e-4,
        probe_offsets=(1.5, 3.5), sponge_wavelengths=0.25,
    )
    cfg.end_time = 1.6
    cfg.steps = int(math.ceil(cfg.end_time / cfg.dt))
    return cfg


def dump(cfg: WaveTankConfig, name: str, fractions=(0.55, 0.75, 0.95)) -> dict:
    run = run_tank(cfg, snapshot_times=fractions)
    payload = {
        "times": np.array(run.times),
        "probe_x": np.array(run.probe_x),
        "probes": np.array(run.probes),
        "paddle": np.array(run.paddle_track),
        "depth": cfg.depth, "dx": cfg.dx, "length": cfg.length,
        "kh": cfg.kh, "wave_height": cfg.wave_height,
        "omega": cfg.omega(), "period": cfg.period(), "wavelength": cfg.wavelength(),
        "stroke": cfg.stroke(), "celerity": cfg.celerity(),
        "sponge_start": cfg.sponge_start(),
        "mass_error": run.mass_error, "volume_error": run.volume_error,
        "max_compression": run.max_compression,
        "max_courant": run.max_courant, "steps": run.steps_taken,
        "wall_seconds": run.wall_seconds,
    }
    for index, snapshot in enumerate(run.snapshots):
        t, centres, eta, pos, vel = snapshot
        payload[f"eta_{index}"] = np.array(eta)
        payload[f"centres_{index}"] = np.array(centres)
        payload[f"x_{index}"] = np.array(pos[:, 0])
        payload[f"y_{index}"] = np.array(pos[:, 1])
        speed = torch.linalg.vector_norm(vel, dim=-1)
        payload[f"speed_{index}"] = np.array(speed)
        payload[f"t_{index}"] = t
    if cfg.wave_kind == "solitary":
        crest = solitary_crest_offset(cfg)
        payload["crest0"] = crest
        payload["beta"] = solitary_beta(cfg.wave_height, cfg.depth)
        grid = torch.linspace(0.0, cfg.length, 601, dtype=torch.float64)
        payload["analytic_x"] = np.array(grid)
        payload["analytic_x0"] = np.array(
            solitary_elevation(grid, cfg.wave_height, cfg.depth, crest))
    path = f"runs/wave_tank/viz_{name}.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"{name}: {path}  steps={run.steps_taken} wall={run.wall_seconds:.0f}s "
          f"volume_error={run.volume_error:.2e}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("regular", "solitary"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = regular_case() if args.case == "regular" else solitary_case()
    cfg.validate()
    dump(cfg, args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
