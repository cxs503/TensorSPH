"""Extend the Kleefsman obstacle case past the reference peak (m21324).

Why this script exists
----------------------
``reference.json`` records the *frozen* acceptance protocol (slab gauge, 0.08 m
aperture, station 0.05 m upstream of the box face, 30 ms symmetric peak filter,
0.02 m arrival threshold, frozen by m20773).  Every committed run of the case
however stops at ``t_end = 1.0 .. 1.2 s``, while the experimental H2 maximum
(0.370 m) is reached at ``t = 1.703 s``.  The arrival attribution of m21173
therefore had to record ``peak_comparable = false``: the peak half of the case
acceptance could not be evaluated at all, because each stored maximum was the
maximum of a *truncated* record.

This driver runs the case at the frozen gauge protocol with a horizon that
covers the whole experimental record up to its peak and the relaxation after
it (``t_end = 2.6 s``), writes ``results/peak_horizon/<tag>.json`` and
``results/peak_horizon/<tag>_time_series.csv`` (committed paths), and leaves the
scoring to :mod:`acceptance_frozen` so that the verdict is reproducible from the
committed CSV alone.

Usage (one grid per invocation, chunked through ``--budget`` if the shell caps a
command)::

    python benchmarks/kleefsman_obstacle/peak_horizon.py --dx 0.02 --t-end 2.6
    python benchmarks/kleefsman_obstacle/peak_horizon.py --dx 0.01 --t-end 2.6

The resumed trajectory is bit-identical to an uninterrupted one (the mechanism
is the one already pinned by ``tests/test_kleefsman_obstacle.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle import (  # noqa: E402
    KleefsmanConfig,
    run_case,
    validate_case,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "peak_horizon"

# ---- frozen acceptance protocol (reference.json -> metric_definition.frozen_primary)
FROZEN_APERTURE_M = 0.08
FROZEN_STATION_OFFSET_M = (0.05, 0.248)
FROZEN_PEAK_WINDOW_S = 0.03
FROZEN_ARRIVAL_THRESHOLD_M = 0.02


def frozen_config(dx: float, t_end: float, sound_speed: float, geom: dict) -> KleefsmanConfig:
    cfg = KleefsmanConfig(
        dx=dx, t_end=t_end, device="cpu", sound_speed=sound_speed,
        tank_length=geom["tank_length"], tank_height=geom["tank_height"],
        column_length=geom["column_length"], column_height=geom["column_height"],
        obstacle_x=geom["obstacle_upstream_face_x"],
        obstacle_length=geom["obstacle_length"], obstacle_height=geom["obstacle_height"],
        pool_width=geom["tank_length"], pool_depth=geom["tank_height"],
        gauge_aperture=FROZEN_APERTURE_M,
        stations_offset=FROZEN_STATION_OFFSET_M,
        arrival_threshold=FROZEN_ARRIVAL_THRESHOLD_M,
    )
    # dt must satisfy the acoustic Courant limit of *this* c0 (see run.py).
    cfg.dt = min(2.0e-4, cfg.dt_acoustic())
    validate_case(cfg)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--t-end", type=float, default=2.6)
    ap.add_argument("--budget", type=float, default=0.0,
                    help="wall-clock budget [s] per invocation; 0 = run to completion")
    ap.add_argument("--output-every", type=int, default=20)
    ap.add_argument("--sound-speed", type=float, default=20.0)
    ap.add_argument("--out-dir", default="",
                    help="output directory name under results/ (default: peak_horizon)")
    args = ap.parse_args()

    global OUT
    if args.out_dir:
        OUT = OUT.parent / args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads((HERE / "reference.json").read_text())
    geom = ref["geometry_m"]
    tag = f"dx{args.dx:g}"
    cfg = frozen_config(args.dx, args.t_end, args.sound_speed, geom)
    cfg.output_every = args.output_every

    state_path = OUT / f"{tag}_state.pt"
    state = torch.load(state_path, weights_only=False) if state_path.exists() else None
    t0 = time.time()
    run = run_case(cfg, resume=state,
                   time_budget=(args.budget if args.budget > 0 else None))
    wall = time.time() - t0
    torch.save(run.state, state_path)
    print(json.dumps({"tag": tag, "finished": run.finished,
                      "elapsed_sim": round(run.simulated_time, 4),
                      "chunk_wall_s": round(wall, 1)}), flush=True)
    if not run.finished:
        return 0

    series_keys = sorted(run.series)
    with (OUT / f"{tag}_time_series.csv").open("w") as fh:
        fh.write("t_s,height_m,front_x_m,front_x_robust_m,excluded,column_level_m,"
                 + ",".join(k.replace("@", "_") for k in series_keys) + "\n")
        cols = run.column if len(run.column) == len(run.times) else [float("nan")] * len(run.times)
        robust = (run.front_robust if len(run.front_robust) == len(run.times)
                  else [float("nan")] * len(run.times))
        excl = (run.excluded if len(run.excluded) == len(run.times)
                else [float("nan")] * len(run.times))
        for idx, (t, hh, fr, rb, ex, cl) in enumerate(
                zip(run.times, run.height, run.front, robust, excl, cols)):
            row = [f"{t:.6f}", f"{hh:.6f}", f"{fr:.6f}", f"{rb:.6f}", f"{ex:.0f}", f"{cl:.6f}"]
            for k in series_keys:
                v = run.series[k]
                row.append(f"{v[idx]:.6f}" if idx < len(v) else "nan")
            fh.write(",".join(row) + "\n")

    row = {
        "tag": tag, "dx": args.dx, "t_end": args.t_end,
        "sound_speed": args.sound_speed, "dt": cfg.dt,
        "simulated_time_s": run.simulated_time, "steps": run.steps_taken,
        "particles_fluid": math.floor(cfg.column_length / cfg.dx)
        * math.floor(cfg.column_height / cfg.dx),
        "gauge_aperture_m": cfg.resolve_gauge_aperture(),
        "stations_offset_m": list(cfg.stations_offset),
        "peak_window_s": FROZEN_PEAK_WINDOW_S,
        "arrival_threshold_m": cfg.arrival_threshold,
        "raw_probe_peak_m": run.peak_height, "raw_probe_peak_time_s": run.peak_time,
        "raw_probe_arrival_s": run.arrival_time,
        "gauges": {k: dict(v, station_x_m=float(k.split("@x")[1]),
                           operator=k.split("@")[0])
                   for k, v in run.metrics.items()},
        "max_density_compression": run.max_density_compression,
        "max_speed": run.max_speed, "mass_error_rel": run.mass_error,
        "boundary_escapes": run.escaped,
        "front_metric": "front_x_m = legacy bare max (saturates after a clamp); "
                        "front_x_robust_m = max excluding wall-crossed particles",
        "wall_clock_s": round(wall, 1),
        "time_series_csv": f"{tag}_time_series.csv",
    }
    (OUT / f"{tag}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())