"""Thread-count scan: is the ``free(): corrupted unsorted chunks`` abort a
function of the *thread count* rather than of the grid resolution?

The ladder's committed readings were all produced at ``--threads 4`` (the
``ensemble_peak`` default).  Every abort so far happened at ``--threads >= 8``
(``cj3803`` dx=0.5 mm @16, ``cj4103`` dx=1 mm @8, ``cj4106`` dx=2 mm @8).  The
two hypotheses are therefore confounded in the existing evidence, and the
previous reading ("the finest grids are memory-fragile") assumed the resolution
axis without ever holding the thread count fixed.

One shard = one (grid, thread-count) cell.  Each shard runs the *same* estimator
as the production ladder, prints a compact digest (the cluster log is tail
truncated at 8 kB, so the full JSON stays on disk) and, on a clean exit, the
pooled readings are written to ``docs/artifacts/<out>/``.

Usage (cluster, one shard per cell)::

    python benchmarks/case_sph_wedge_entry_2d/thread_scan.py --shard 0 --of 6 \
        --out docs/artifacts/case_sph_wedge_entry_2d_thread_scan
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ensemble_peak as ep  # noqa: E402

# (dx [m], threads) cells.  Thread counts 4 (= committed default) and 8/16 (=
# the aborting counts) are both present, at two resolutions, so the two axes can
# be separated instead of assumed.
CELLS = (
    (2.0e-3, 4),
    (2.0e-3, 8),
    (2.0e-3, 16),
    (1.0e-3, 4),
    (1.0e-3, 8),
    (1.0e-3, 16),
)

# Fine-grid cells: the two grids whose shards had never been read back
# (cj4103 dx=1 mm, cj3803 dx=0.5 mm) at the aborting and the committed thread
# counts, so the fine end of the ladder is either filled or attributed.
CELLS_FINE = (
    (1.0e-3, 8),
    (1.0e-3, 16),
    (0.5e-3, 8),
    (0.5e-3, 32),
)

BIN_BAND = "6,8,10,12,16,20,30,40,50,60,80,100,120,160,180"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=None)
    ap.add_argument("--frames", type=int, default=61)
    # Single-cell override: lets an expensive grid (dx = 0.5 mm) be submitted as
    # its own shard without editing CELLS.
    ap.add_argument("--cells", choices=("coarse", "fine"), default="coarse",
                    help="which (grid, threads) cell table to shard over")
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=HERE.parents[1] / "docs/artifacts/case_sph_wedge_entry_2d_thread_scan")
    args = ap.parse_args(argv)
    table = CELLS_FINE if args.cells == "fine" else CELLS
    dx, threads = table[args.shard % len(table)]
    if args.dx is not None:
        dx = args.dx
    if args.threads is not None:
        threads = args.threads
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = ep.main(["--dx", repr(dx), "--frames", str(args.frames),
                  "--threads", str(threads), "--bin-grid", BIN_BAND,
                  "--out", str(out)])
    # Compact digest from the artefact the run just wrote (no recomputation).
    tag = f"dx{dx * 1e3:g}mm_t{threads}"
    artefact = out / f"ensemble_dx{dx * 1e3:g}mm_c0v20.json"
    if artefact.exists():
        d = json.loads(artefact.read_text())
        band = d["by_frames"][str(args.frames)]["resolution_band"]
        fp = d["footprint_peak"]
        digest = {
            "cell": tag, "dx_mm": dx * 1e3, "threads": threads,
            "steps": d.get("steps"), "wall_s": round(d.get("wall_seconds", 0.0), 1),
            "band_peak_median": band["peak_median"],
            "band_spread_rel": band["spread_rel"],
            "jackknife_rel": d["by_frames"][str(args.frames)]["jackknife_rel"],
            "max_rho_ratio": max(s.get("max_rho_ratio", 0.0) for s in d["frame_scan"]),
            "footprint_U": {f: fp["bootstrap"][f]["rel_half_width"]
                            for f in fp["bootstrap"]},
        }
        print("DIGEST " + json.dumps(digest, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())