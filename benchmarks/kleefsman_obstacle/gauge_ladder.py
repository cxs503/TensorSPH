"""Resolution ladder for the Kleefsman dam-break-with-obstacle gauge metrics.

Runs the case on a fixed physical gauge set (three operators, two stations,
one dx-independent slab aperture) on every grid of the ladder and writes one
``metrics.json`` per grid under ``results/gauge_ladder/``.  ``merge`` then folds
the per-grid files into a single ``ladder.json`` so the analysis script
(``gauge_convergence.py``) has no ad-hoc file discovery of its own.

Why a ladder at all: the case-pool acceptance is "probe water-level peak and
arrival time vs Kleefsman et al. (2005), tolerance < 10 %".  A percentage
against an experiment is only falsifiable if the *numerical* metric has an
uncertainty band narrower than 10 %; this script produces the ladder that band
is read from.

Usage (cluster shard = one grid per shard)::

    python benchmarks/kleefsman_obstacle/gauge_ladder.py --shard 0 --shards 3
    python benchmarks/kleefsman_obstacle/gauge_ladder.py --merge
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "gauge_ladder"
DEFAULT_GRIDS = (0.04, 0.02, 0.01)


def run_grid(dx: float, t_end: float, aperture: float, threads: int) -> dict:
    tag = f"dx{dx:g}"
    outdir = RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)
    import os
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    cmd = [sys.executable, str(HERE / "run.py"), "--dx", repr(dx),
           "--t-end", repr(t_end), "--gauge-aperture", repr(aperture),
           "--tag", tag, "--outdir", str(outdir)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(HERE.parents[1]))
    return json.loads((outdir / "metrics.json").read_text())


def merge() -> int:
    rows = []
    for path in sorted(RESULTS.glob("dx*/metrics.json")):
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda r: r["dx"])
    ladder = {
        "case": "Kleefsman (2005) dam break with obstacle -- gauge-metric ladder",
        "grids_dx": [r["dx"] for r in rows],
        "protocol": {
            "t_end": sorted({r["t_end"] for r in rows}),
            "sound_speed": sorted({r.get("sound_speed") for r in rows}),
            "gauge_aperture_m": sorted({r.get("gauge_aperture_m") for r in rows}),
            "stations_offset_m": sorted({tuple(r.get("stations_offset_m") or ()) for r in rows}),
            "arrival_threshold": sorted({r.get("arrival_threshold") for r in rows}),
        },
        "rows": rows,
    }
    out = RESULTS / "ladder.json"
    out.write_text(json.dumps(ladder, indent=2))
    print("wrote", out, "grids:", ladder["grids_dx"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--grids", type=float, nargs="*", default=list(DEFAULT_GRIDS))
    ap.add_argument("--t-end", type=float, default=1.2)
    ap.add_argument("--aperture", type=float, default=0.08,
                    help="slab-gauge FULL width [m]; must be >= 2 dx on the coarsest grid")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge()
    mine = [g for k, g in enumerate(args.grids) if k % args.shards == args.shard]
    if not mine:
        print(f"shard {args.shard}/{args.shards}: nothing to do")
        return 0
    for dx in mine:
        run_grid(dx, args.t_end, args.aperture, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())