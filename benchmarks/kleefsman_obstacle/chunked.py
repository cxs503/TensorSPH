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

from tensorsph.kleefsman_obstacle import KleefsmanConfig, run_case  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--t-end", type=float, default=1.2)
    ap.add_argument("--budget", type=float, default=95.0)
    ap.add_argument("--output-every", type=int, default=40)
    args = ap.parse_args()

    ref = json.loads((HERE / "reference.json").read_text())
    g = ref["geometry_m"]
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    tag = f"dx{args.dx:g}"
    cfg = KleefsmanConfig(
        dx=args.dx, t_end=args.t_end, device="cpu", output_every=args.output_every,
        tank_length=g["tank_length"], tank_height=g["tank_height"],
        column_length=g["column_length"], column_height=g["column_height"],
        obstacle_x=g["obstacle_upstream_face_x"],
        obstacle_length=g["obstacle_length"], obstacle_height=g["obstacle_height"],
        pool_width=g["tank_length"], pool_depth=g["tank_height"],
        dt=min(2.0e-4, 0.3 * 1.3 * args.dx / 20.0))

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