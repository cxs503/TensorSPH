"""Long-horizon (t = 6 s) runner for the Kleefsman case, resumable in chunks.

Why this exists
---------------
Every committed Kleefsman run in this repo stopped at ``t_end <= 2.6 s``, so the
late-time landmarks that Kleefsman et al. (2005) state *in words* -- the return
of the surge from the back wall (about 1.8 s), the second impact of the return
wave on the box (about 4.7 s) and the second meeting of that wave with probe H2
(about 5 s) -- had never been tested against our solver.  Those statements are
text, not a raster curve, so they are checkable without any digitisation.

``chunked.py`` also runs in chunks but only persists the legacy probe columns,
while the landmark audit needs the slab gauges (``slab@x<S>``) at several
stations.  This driver therefore reuses the same ``run_case(resume=...,
time_budget=...)`` mechanism and writes the *full* gauge table to a committed
path (``results/long_horizon/``); the multi-megabyte solver state goes to the
git-ignored ``artifacts/`` tree.

Usage (from the repo root)::

    python benchmarks/kleefsman_obstacle/long_horizon.py --dx 0.04 --t-end 5.6 --budget 95
    python benchmarks/kleefsman_obstacle/long_horizon.py --dx 0.04 --t-end 5.6 --budget 95

The second call resumes; repeat until the process prints ``"finished": true``.
A finished tag is skipped on re-invocation, so the loop is idempotent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle import KleefsmanConfig, run_case  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "long_horizon"
STATE = HERE / "artifacts" / "long_horizon"


def build_config(dx: float, t_end: float, geom: dict, sound_speed: float,
                 stations: tuple[float, ...], aperture: float) -> KleefsmanConfig:
    cfg = KleefsmanConfig(
        dx=dx, t_end=t_end, device="cpu", sound_speed=sound_speed,
        tank_length=geom["tank_length"], tank_height=geom["tank_height"],
        column_length=geom["column_length"], column_height=geom["column_height"],
        obstacle_x=geom["obstacle_upstream_face_x"],
        obstacle_length=geom["obstacle_length"], obstacle_height=geom["obstacle_height"],
        pool_width=geom["tank_length"], pool_depth=geom["tank_height"],
        gauge_aperture=aperture, stations_offset=stations,
    )
    cfg.dt = min(2.0e-4, cfg.dt_acoustic())
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--t-end", type=float, default=5.6)
    ap.add_argument("--budget", type=float, default=95.0,
                    help="wall-clock seconds per invocation (shell caps a command)")
    ap.add_argument("--sound-speed", type=float, default=20.0)
    ap.add_argument("--gauge-aperture", type=float, default=0.08)
    ap.add_argument("--stations", type=float, nargs="*", default=(0.0, 0.05, 0.248),
                    help="gauge stations as distances UPSTREAM of the box face [m]")
    ap.add_argument("--output-every", type=int, default=40)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    ref = json.loads((HERE / "reference.json").read_text())
    tag = args.tag or f"dx{args.dx:g}"
    done = RES / f"{tag}.json"
    if done.exists():
        prev = json.loads(done.read_text())
        if prev.get("finished"):
            print(json.dumps({"tag": tag, "already_finished": True}))
            return 0

    cfg = build_config(args.dx, args.t_end, ref["geometry_m"], args.sound_speed,
                       tuple(args.stations), args.gauge_aperture)
    cfg.output_every = args.output_every
    state_path = STATE / f"{tag}_state.pt"
    state = torch.load(state_path, weights_only=False) if state_path.exists() else None

    t0 = time.time()
    run = run_case(cfg, resume=state, time_budget=args.budget)
    torch.save(run.state, state_path)
    chunk = round(time.time() - t0, 1)
    if not run.finished:
        print(json.dumps({"tag": tag, "finished": False,
                          "simulated_time_s": round(run.simulated_time, 4),
                          "steps": run.steps_taken, "chunk_wall_s": chunk}), flush=True)
        return 0

    keys = sorted(run.series)
    metric = {
        "tag": tag, "dx": cfg.dx, "t_end": cfg.t_end, "finished": True,
        "sound_speed": cfg.sound_speed, "dt": cfg.dt,
        "dt_min": run.dt_min, "dt_max": run.dt_max,
        "simulated_time_s": run.simulated_time, "steps": run.steps_taken,
        "output_every": cfg.output_every,
        "gauge_aperture_m": cfg.resolve_gauge_aperture(),
        "stations_offset_m": list(cfg.stations_offset),
        "station_x_m": list(cfg.station_x()),
        "obstacle_x_m": cfg.obstacle_x,
        "particles_fluid": int(math.floor(cfg.column_length / cfg.dx)
                               * math.floor(cfg.column_height / cfg.dx)),
        "max_density_compression": run.max_density_compression,
        "max_speed": run.max_speed, "mass_error_rel": run.mass_error,
        "boundary_escapes": run.escaped,
        "legacy_probe": {"x_m": cfg.probe_x(), "peak_m": run.peak_height,
                         "peak_time_s": run.peak_time, "arrival_time_s": run.arrival_time},
        "gauges": {k: dict(v, station_x_m=float(k.split("@x")[1]),
                           operator=k.split("@")[0]) for k, v in run.metrics.items()},
        "wall_clock_s": chunk,
    }
    done.write_text(json.dumps(metric, indent=2))
    with (RES / f"{tag}_time_series.csv").open("w") as fh:
        fh.write("t_s,height_m,front_x_m,front_robust_x_m,column_level_m,"
                 + ",".join(k.replace("@", "_") for k in keys) + "\n")
        cols = run.column if len(run.column) == len(run.times) else [float("nan")] * len(run.times)
        frb = (run.front_robust if len(run.front_robust) == len(run.times)
               else [float("nan")] * len(run.times))
        for idx, (t, hh, fr, fb, cl) in enumerate(zip(run.times, run.height, run.front,
                                                     frb, cols)):
            row = [f"{t:.6f}", f"{hh:.6f}", f"{fr:.6f}", f"{fb:.6f}", f"{cl:.6f}"]
            for k in keys:
                v = run.series[k]
                row.append(f"{v[idx]:.6f}" if idx < len(v) else "nan")
            fh.write(",".join(row) + "\n")
    print(json.dumps(metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())