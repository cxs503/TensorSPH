"""Compact digest of a wedge-entry ensemble artefact (m21953, m22255).

Why this exists
---------------
A cluster run writes its artefacts into the *executed* checkout, and the
compute-gate's ``fetch`` returns the tarball that was staged *before* the run
(m21005 §③).  Every verification round that needs a new grid therefore has to
get its numbers out of the job's standard output instead of out of a fetched
file.  This tool prints, as one JSON object on stdout, exactly the fields that
``footprint_ledger.py`` consumes from an ``ensemble_<tag>_c0v20.json`` -- so a
run's read-out can be transcribed back with ``--restore`` without hand copying
a single number.

Why ``--compact`` (m22255)
--------------------------
The job log keeps only the **last ~8000 characters** of stdout
(``/srv/tensor/agent/cluster/jobs/<jid>.log`` -> ``output_tail``, truncated to
8000 chars).  The full digest of a ``dx = 2 mm`` artefact is 19 455 chars, so a
*successful* run could still lose its reading: in cj3802 (``dx = 1 mm``, rc 0)
the tail was cut exactly through the ``footprint_peak`` block, which is why the
grid was recorded as ``MISSING`` although the run had finished.  ``--compact``
keeps every field the ledger reads (``config``, ``t_end``, ``C``,
``footprint_peak``) and replaces the two bulky audit blocks (61 per-frame scan
rows, 16-entry bin sweep) by their summaries, so the read-out fits the tail.
The losslessness of the compact form *for the ledger* is asserted by
``tests/test_wedge_artifact_digest.py``.

Usage::

    # inside the job: artefact -> one JSON object on stdout (fits the log tail)
    python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py --compact \
        docs/artifacts/<dir>/ensemble_dx1mm_c0v20.json

    # on the writing side: digest -> reconstructed minimal artefact
    python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py \
        digest.json --restore docs/artifacts/<dir>/ensemble_dx1mm_c0v20.json

The digest carries ``provenance`` (host name, source path, sha256 of the source
file) so the restored artefact is traceable to the run that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

# fields the ledger reads (footprint_ledger.grid_record) + audit context
KEEP_TOP = ("case", "script", "t_end", "C", "q", "c0", "steps", "wall_seconds",
            "frame_scan", "matched_bandwidth", "config", "design",
            "wagner_cp_max", "wagner_cp_centre", "wagner_tau_peak")

# what a job log keeps of stdout (cluster/jobs/<jid>.log -> output_tail)
STDOUT_TAIL_CHARS = 8000

_FRAME_SUMMARY = ("max_rho_ratio",)
_SWEEP_SUMMARY = ("peak", "n_bins_matched", "n_cells", "tau_at_peak",
                  "samples_at_peak")


def _max_finite(values) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float)) and v == v]
    return max(vals) if vals else None


def digest(art: dict, src: Path | None = None, compact: bool = False) -> dict:
    """Digest of ``art``; ``compact`` fits the job-stdout tail (see module doc)."""
    out = {k: art.get(k) for k in KEEP_TOP if k in art}
    fp = art.get("footprint_peak") or {}
    out["footprint_peak"] = {"footprints_cells": fp.get("footprints_cells"),
                             "rows": fp.get("rows"),
                             "bootstrap": fp.get("bootstrap")}
    if compact:
        fs = art.get("frame_scan")
        if isinstance(fs, list) and fs:
            out["frame_scan"] = {
                "n_frames": len(fs),
                "max_rho_ratio": _max_finite(r.get(k) for r in fs
                                             for k in _FRAME_SUMMARY),
                "per_frame_rows_dropped": len(fs),
            }
        mb = art.get("matched_bandwidth") or {}
        if mb:
            out["matched_bandwidth"] = {k: mb.get(k)
                                        for k in _SWEEP_SUMMARY if k in mb}
            out["matched_bandwidth"]["bandwidth_sweep_dropped"] = len(
                mb.get("bandwidth_sweep") or [])
        out.pop("design", None)
        out["digest_mode"] = "compact"
    out["provenance"] = {"host": platform.node(),
                         "source": str(src) if src else None,
                         "digest_sha256": (hashlib.sha256(src.read_bytes()).hexdigest()
                                           if src and src.exists() else None)}
    return out


def restore(dig: dict, dst: Path) -> None:
    """Write a minimal artefact with the fields the ledger consumes."""
    art = {k: v for k, v in dig.items() if k != "provenance"}
    art["restored_from_digest"] = dig.get("provenance", {})
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="artefact json, or digest json with --restore")
    ap.add_argument("--restore", type=Path, default=None,
                    help="write a minimal artefact for footprint_ledger.py to this path")
    ap.add_argument("--compact", action="store_true",
                    help=f"shrink the read-out below the {STDOUT_TAIL_CHARS}-char "
                         "job-stdout tail (drops per-frame rows / bin sweep)")
    args = ap.parse_args(argv)
    doc = json.loads(args.source.read_text())
    if args.restore is not None:
        restore(doc, args.restore)
        print(f"restored {args.restore}")
        return 0
    text = json.dumps(digest(doc, args.source, compact=args.compact),
                      sort_keys=True, separators=(",", ":"))
    if args.compact and len(text) > STDOUT_TAIL_CHARS:
        print(f"[digest] WARNING compact digest is {len(text)} chars > tail "
              f"{STDOUT_TAIL_CHARS}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())