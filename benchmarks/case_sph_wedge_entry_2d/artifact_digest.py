"""Compact digest of a wedge-entry ensemble artefact (m21953).

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

Usage::

    # inside the job: artefact -> one compact JSON line set on stdout
    python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py \
        docs/artifacts/<dir>/ensemble_dx1mm_c0v20.json

    # on the writing side: digest -> reconstructed minimal artefact
    python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py \
        digest.json --restore docs/artifacts/<dir>/ensemble_dx1mm_c0v20.json

The digest carries ``provenance`` (job id, wall clock, steps, host name) so the
restored artefact is traceable to the run that produced it, and
``digest_sha256`` of the source file so a fetched-and-restored artefact can be
told apart from one written directly by the solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

# fields the ledger reads (footprint_ledger.grid_record) + audit context
KEEP_TOP = ("case", "script", "t_end", "C", "q", "c0", "steps", "wall_seconds",
            "frame_scan", "matched_bandwidth", "config", "design",
            "wagner_cp_max", "wagner_cp_centre", "wagner_tau_peak")


def digest(art: dict, src: Path | None = None) -> dict:
    out = {k: art.get(k) for k in KEEP_TOP if k in art}
    fp = art.get("footprint_peak") or {}
    out["footprint_peak"] = {"footprints_cells": fp.get("footprints_cells"),
                             "rows": fp.get("rows"),
                             "bootstrap": fp.get("bootstrap")}
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
    args = ap.parse_args(argv)
    doc = json.loads(args.source.read_text())
    if args.restore is not None:
        restore(doc, args.restore)
        print(f"restored {args.restore}")
        return 0
    print(json.dumps(digest(doc, args.source), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())