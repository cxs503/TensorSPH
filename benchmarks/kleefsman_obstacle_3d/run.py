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
    ap.add_argument("--checkpoint", default=None,
                    help="state file used to continue a run that was killed; "
                         "written every --checkpoint-every steps and removed "
                         "once t_end is reached")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing checkpoint and start at t = 0")
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

    # The series is streamed to disk sample by sample.  These grids are
    # half-hour cluster jobs and the production rungs have already been killed
    # mid-run by a glibc heap abort ("free(): corrupted unsorted chunks"), which
    # discarded every sample because the whole series used to be written only
    # after the loop returned.
    csv_path = out_dir / f"{tag}_time_series.csv"
    resuming = bool(args.checkpoint and Path(args.checkpoint).exists()
                    and not args.no_resume)
    # a resumed run appends: the samples already in the checkpoint are not
    # re-emitted, so the file stays one monotone series
    csv_fh = csv_path.open("a" if resuming else "w", newline="")
    if resuming:
        print(f"[{tag}] resuming from {args.checkpoint}", flush=True)
    writer = csv.writer(csv_fh)
    writer.writerow(["t_s", "slab_height_m", "height_operator_m",
                     "front_x_m", "n_bypass"])
    csv_fh.flush()

    def on_sample(_i: int, row: dict) -> None:
        writer.writerow([f"{r:.6f}" if isinstance(r, float) else r
                         for r in (row["t_s"], row["slab_height_m"],
                                   row["height_operator_m"], row["front_x_m"],
                                   row["n_bypass"])])
        csv_fh.flush()

    # fingerprint written before the run, so a run that never returns still
    # leaves the protocol it was run under (the ladder refuses to compare a
    # series whose protocol cannot be established).
    meta_path = out_dir / f"{tag}_meta.json"
    meta_path.write_text(json.dumps({
        "tag": tag, "dx": cfg.dx, "t_end": cfg.t_end, "dt": None,
        "sound_speed": cfg.sound_speed, "wall_layers": cfg.wall_layers,
        "obstacle_z": cfg.obstacle_z, "station_offset": cfg.station_offset,
        "gauge_aperture": cfg.gauge_aperture,
        "arrival_threshold": cfg.arrival_threshold,
        "output_every": cfg.output_every,
        "threads": torch.get_num_threads(),
        "complete": False,
    }, indent=1) + "\n")

    t0 = time.time()
    try:
        run = run_case(cfg, device="cpu", progress=progress, on_sample=on_sample,
                       checkpoint_path=args.checkpoint,
                       checkpoint_every=args.checkpoint_every,
                       resume=not args.no_resume)
    except BaseException as exc:            # noqa: BLE001 - want the evidence
        csv_fh.close()
        meta = json.loads(meta_path.read_text())
        meta.update({"complete": False, "aborted": True,
                     "error": f"{type(exc).__name__}: {exc}",
                     "wall_clock_s": round(time.time() - t0, 1)})
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
        print(f"[{tag}] ABORTED after {meta['wall_clock_s']:.0f} s: {meta['error']}",
              flush=True)
        print(f"[{tag}] partial series kept in {csv_path}", flush=True)
        raise
    csv_fh.close()

    verdict = score(run)
    verdict["tag"] = tag
    verdict["complete"] = True
    verdict["host"] = {
        "threads": torch.get_num_threads(),
        "wall_clock_s": round(time.time() - t0, 1),
    }

    json_path = out_dir / f"{tag}.json"
    json_path.write_text(json.dumps(verdict, indent=1) + "\n")

    meta = json.loads(meta_path.read_text())
    meta.update({"complete": True, "aborted": False, "dt": cfg.dt,
                 "steps": run.steps, "simulated_time_s": run.simulated_time,
                 "wall_clock_s": round(time.time() - t0, 1)})
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")

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