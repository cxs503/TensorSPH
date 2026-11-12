"""Chunked runner for the Kleefsman case (shell caps a command at ~120 s).

Each invocation advances the case for at most ``--budget`` seconds of wall clock,
persists the solver state, and exits; repeat until it reports ``finished``::

    python benchmarks/kleefsman_obstacle/chunked.py --dx 0.01 --budget 95
    python benchmarks/kleefsman_obstacle/chunked.py --dx 0.01 --budget 95   # resume

The resumed trajectory is bit-identical to an uninterrupted one (see
``tests/test_kleefsman_obstacle.py::test_chunked_run_matches_a_single_run``).
When the run finishes, ``results/<tag>.json`` and ``results/<tag>_time_series.csv``
are written (a committed path, unlike ``artifacts/``).
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

from tensorsph.kleefsman_obstacle import (  # noqa: E402
    KleefsmanConfig,
    run_case,
    validate_case,
)

HERE = Path(__file__).resolve().parent


def driver_config(dx: float, sound_speed: float, t_end: float, geom: dict) -> KleefsmanConfig:
    """Case config with a Courant-consistent dt for *this* sound speed.

    ``dt`` is the smaller of the historical 2e-4 s cap and the acoustic limit
    ``max_courant * h / c0``.  Hard-coding the 20 m/s of the default scan meant
    that any c0 > ~78 m/s silently stepped past the acoustic Courant limit at
    dx = 0.04 m (same defect class as the one fixed in ``run.py``).
    """
    cfg = KleefsmanConfig(
        dx=dx, t_end=t_end, device="cpu",
        tank_length=geom["tank_length"], tank_height=geom["tank_height"],
        column_length=geom["column_length"], column_height=geom["column_height"],
        obstacle_x=geom["obstacle_upstream_face_x"],
        obstacle_length=geom["obstacle_length"], obstacle_height=geom["obstacle_height"],
        pool_width=geom["tank_length"], pool_depth=geom["tank_height"],
        sound_speed=sound_speed)
    cfg.dt = min(2.0e-4, cfg.dt_acoustic())
    validate_case(cfg)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--t-end", type=float, default=1.2)
    ap.add_argument("--budget", type=float, default=95.0)
    ap.add_argument("--output-every", type=int, default=40)
    ap.add_argument("--sound-speed", type=float, default=20.0,
                    help="artificial sound speed c0 [m/s]; dt follows the CFL of *this* c0")
    args = ap.parse_args()

    ref = json.loads((HERE / "reference.json").read_text())
    g = ref["geometry_m"]
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    tag = f"dx{args.dx:g}"
    cfg = driver_config(args.dx, args.sound_speed, args.t_end, g)
    cfg.output_every = args.output_every

    state_path = res / f"{tag}_state.pt"
    state = torch.load(state_path, weights_only=False) if state_path.exists() else None
    t0 = time.time()
    run = run_case(cfg, resume=state, time_budget=args.budget)
    torch.save(run.state, state_path)
    print(json.dumps({"tag": tag, "finished": run.finished, "elapsed_sim": run.simulated_time,
                      "chunk_wall_s": round(time.time() - t0, 1),
                      "last_height_m": run.height[-1] if run.height else None}), flush=True)
    if not run.finished:
        return 0

    row = {"tag": tag, "dx": args.dx, "t_end": args.t_end,
           "particles_fluid": math.floor(cfg.column_length / args.dx)
           * math.floor(cfg.column_height / args.dx),
           "steps": run.steps_taken, "peak_height_m": run.peak_height,
           "peak_time_s": run.peak_time, "arrival_time_s": run.arrival_time,
           "mass_error_rel": run.mass_error, "boundary_escapes": run.escaped,
           "max_density_compression": run.max_density_compression,
           "max_speed": run.max_speed,
           "height_capped_by_tank": bool(run.peak_height >= 0.995 * cfg.tank_height)}
    (res / f"{tag}.json").write_text(json.dumps(row, indent=2))
    with (res / f"{tag}_time_series.csv").open("w") as fh:
        fh.write("t_s,height_m,front_x_m\n")
        for t, hh, fr in zip(run.times, run.height, run.front):
            fh.write(f"{t:.6f},{hh:.6f},{fr:.6f}\n")
    study = res / "resolution_study.json"
    merged = json.loads(study.read_text()) if study.exists() else {"t_end": args.t_end, "runs": []}
    merged["runs"] = sorted([r for r in merged["runs"] if r["dx"] != args.dx] + [row],
                            key=lambda r: -r["dx"])
    study.write_text(json.dumps(merged, indent=2))
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())