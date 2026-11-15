"""Driver for the 3-D Kleefsman obstacle case (TensorSPH).

Runs one grid of :mod:`tensorsph.kleefsman_obstacle_3d` to ``--t-end`` and writes
the gauge time series plus the scored verdict to committed paths:

    results/<tag>_time_series.csv   t, slab, height, front_x, bypass
    results/<tag>.json              run summary + verdict vs Kleefsman Fig. 19

Example (one grid per invocation, because a grid is minutes to hours of wall
clock on the cluster)::

    python benchmarks/kleefsman_obstacle_3d/run.py --dx 0.04 --t-end 2.6 \
        --wall-layers 3 --tag dx0.04
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle_3d import (  # noqa: E402
    Kleefsman3DConfig,
    run_case,
    score,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--t-end", type=float, default=2.6)
    ap.add_argument("--dt", type=float, default=None,
                    help="default: the acoustic Courant limit for this dx")
    ap.add_argument("--sound-speed", type=float, default=20.0)
    ap.add_argument("--wall-layers", type=int, default=3)
    ap.add_argument("--obstacle-z", type=float, default=0.0,
                    help="spanwise origin of the box (0 = flush with a side wall)")
    ap.add_argument("--output-every", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    tag = args.tag or f"dx{args.dx:g}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Kleefsman3DConfig(
        dx=args.dx,
        t_end=args.t_end,
        sound_speed=args.sound_speed,
        wall_layers=args.wall_layers,
        obstacle_z=args.obstacle_z,
        output_every=args.output_every,
    )
    if args.dt is not None:
        cfg.dt = args.dt
    else:
        cfg.dt = 0.9 * cfg.dt_from_courant()

    print(f"[{tag}] dx={cfg.dx:g} t_end={cfg.t_end:g} dt={cfg.dt:.3e} "
          f"c0={cfg.sound_speed:g} wall_layers={cfg.wall_layers}", flush=True)

    def progress(t: float, h: float) -> None:
        print(f"  t={t:7.3f} slab={h:6.3f} m", flush=True)

    t0 = time.time()
    run = run_case(cfg, device="cpu", progress=progress)
    verdict = score(run)
    verdict["tag"] = tag
    verdict["host"] = {
        "threads": torch.get_num_threads(),
        "wall_clock_s": round(time.time() - t0, 1),
    }

    csv_path = out_dir / f"{tag}_time_series.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_s", "slab_height_m", "height_operator_m",
                         "front_x_m", "n_bypass"])
        for row in zip(run.times, run.slab, run.height, run.front, run.bypass):
            writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])

    json_path = out_dir / f"{tag}.json"
    json_path.write_text(json.dumps(verdict, indent=1) + "\n")

    print(f"[{tag}] verdict={verdict['verdict']} "
          f"peak={verdict['run']['peak_height_m']:.4f} m "
          f"@ {verdict['run']['peak_time_s']:.4f} s, "
          f"arrival={verdict['run']['arrival_time_s']}, "
          f"wall={verdict['run']['wall_clock_s']:.0f} s", flush=True)
    for row in verdict["rows"]:
        got = row["got"]
        print(f"  {row['metric']:14s} ref={row['ref']:.4f} got="
              f"{'None' if got is None else f'{got:.4f}'} "
              f"rel={row['rel_error']*100:.1f}% pass={row['pass']}", flush=True)
    print(f"  wrote {csv_path} and {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())