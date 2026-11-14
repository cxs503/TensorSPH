"""Run one matrix cell and emit the record to stdout as base64.

``run_cell.py`` writes its record to ``--out``.  On the cluster the record lives
in the node's scratch copy of the staged work tree, and ``run_compute fetch``
only guarantees the job log, so the record itself can be lost (this is exactly
what made mission 20569 an honest FAIL: cells 0-3 ran with ``rc=0`` but no
artefact came back).

This wrapper makes the record travel with the log: after the cell finishes it
prints

    >>>BEGIN_RECORD_B64 <label> <sha256> <bytes>
    <one base64 line>
    >>>END_RECORD_B64 <label>

so the caller can rebuild the file byte-exactly and verify it with the printed
digest.  Nothing else changes: the cell is ``run_cell.main`` verbatim.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/run_cell_dump.py 0 \
        --out runs/xval/cell_0.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_cell  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=sorted(run_cell.CELLS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    label = run_cell.CELLS[args.index][0]
    rc = run_cell.main([str(args.index), "--out", args.out])

    path = Path(args.out)
    if path.exists():
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        print(f">>>BEGIN_RECORD_B64 {label} {digest} {len(blob)}", flush=True)
        print(base64.b64encode(blob).decode("ascii"), flush=True)
        print(f">>>END_RECORD_B64 {label}", flush=True)
    else:
        print(f">>>RECORD_MISSING {label} {args.out}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())