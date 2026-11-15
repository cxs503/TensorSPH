"""Shard driver for the wedge-entry peak-pressure ladder (m21677).

One shard = one grid of the dyadic ladder, run with the extended bin band so the
*matched* bandwidth (one bin per surface cell) and the finite-area-averaged
(transducer footprint) readings are recorded alongside the historical
``BIN_GRID`` rows.  The extended band contains every original ``BIN_GRID``
value, so the shard also re-derives the committed rows bit-for-bit -- a
determinism check on the whole chain.

Usage::

    python benchmarks/case_sph_wedge_entry_2d/run_ladder_shard.py \
        --shard 0 --of 4 --threads 8 --out docs/artifacts/<dir>

``--shard i`` maps to ``dx = 4 mm / 2**i`` (``i`` taken modulo the ladder size),
so a sharded cluster submission needs no per-shard edit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ensemble_peak as ep  # noqa: E402

LADDER_DX = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)
# original BIN_GRID first (determinism check), then the matched-bandwidth and
# footprint resolutions the fine grids need (up to 180 bins = 1/3 cell at dx0.5)
BIN_BAND = (6, 8, 10, 12, 16, 20, 30, 40, 50, 60, 80, 100, 120, 160, 180)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=len(LADDER_DX))
    ap.add_argument("--velocity", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--c0-factor", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    dx = LADDER_DX[args.shard % len(LADDER_DX)]
    band = ",".join(str(b) for b in BIN_BAND)
    return ep.main(["--dx", repr(dx), "--velocity", repr(args.velocity),
                    "--frames", str(args.frames),
                    "--c0-factor", repr(args.c0_factor),
                    "--threads", str(args.threads),
                    "--bin-grid", band, "--out", str(args.out)])


if __name__ == "__main__":
    raise SystemExit(main())