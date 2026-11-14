"""Provenance manifest for the method-selection decision cells.

Why this exists
---------------
The matrix rows of anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` are graded
from JSON records under ``runs/xval/``.  Two missions in a row could not say
*which job produced which record*: one found ``rc=0`` cells whose artefacts never
came back (m20569), the next found cells that died with a glibc heap error before
writing anything (m20783).  A record that appears in the working tree with no job
identity attached is not evidence a reader can reproduce.

This module writes a small manifest that binds every matrix record to

* the cell label it belongs to (``run_cell.CELLS``, single source of truth),
* the sha256 and byte length of the file that was actually graded,
* the compute job id that produced it, and
* how the bytes were recovered (``log_base64`` channel, ``fetch_tar`` channel,
  or ``missing``).

The manifest is *derived*, never hand written: digests are recomputed from the
working tree every time it is built, so it cannot drift from the files.  A
record that is absent is listed as ``missing`` instead of being dropped from the
manifest -- the gap is evidence too.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/record_provenance.py \
        --out docs/reports/xval_method_selection_records_manifest.json \
        --job 0=cj3094 --job 1=cj3095 --job 2=cj3096 --job 3=cj3097

Exit code is 0 when the manifest was written, 1 when a ``--job`` spec names a
cell that does not exist (a typo must not silently produce a shorter manifest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import run_cell  # noqa: E402  (stdlib only: no torch/matplotlib at import)

ANCHOR = "XVAL-SPH-METHOD-SELECTION-2026-11-12"
DEFAULT_OUT = REPO / "docs" / "reports" / "xval_method_selection_records_manifest.json"
RECOVERY_MODES = ("log_base64", "fetch_tar", "local", "missing")


def digest(path: Path) -> dict:
    """sha256 + byte length of ``path``, or a ``missing`` stub."""
    if not path.exists():
        return {"present": False, "sha256": None, "bytes": None}
    blob = path.read_bytes()
    return {"present": True, "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob)}


def parse_job_specs(specs, cells) -> dict:
    """``["0=cj3094", ...]`` -> ``{0: "cj3094"}``; unknown index is an error."""
    jobs: dict[int, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--job needs INDEX=cjN, got {spec!r}")
        index_text, jid = spec.split("=", 1)
        try:
            index = int(index_text)
        except ValueError:
            raise SystemExit(f"--job index must be an integer, got {index_text!r}")
        if index not in cells:
            raise SystemExit(f"--job index {index} is not a cell "
                             f"(known: {sorted(cells)})")
        if not jid.startswith("cj"):
            raise SystemExit(f"--job id must look like cjN, got {jid!r}")
        jobs[index] = jid
    return jobs


def display_path(path: Path) -> str:
    """Repository-relative when the file lives in the tree, absolute otherwise."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def build(record_dir: Path, jobs: dict, recovery: dict | None = None,
          cells=None, mission: int | None = None) -> dict:
    """Assemble the manifest dict for every cell in ``cells`` (default run_cell)."""
    cells = run_cell.CELLS if cells is None else cells
    recovery = recovery or {}
    rows = []
    for index in sorted(cells):
        label = cells[index][0]
        path = Path(record_dir) / f"cell_{index}.json"
        info = digest(path)
        how = recovery.get(index, "fetch_tar" if info["present"] else "missing")
        if not info["present"] and how != "missing":
            # The claimed recovery channel contradicted the working tree.
            raise SystemExit(f"cell {index} ({label}): recovery={how!r} but "
                             f"{path} is absent")
        if how not in RECOVERY_MODES:
            raise SystemExit(f"cell {index}: unknown recovery mode {how!r}")
        rows.append({
            "index": index,
            "label": label,
            "path": display_path(path),
            "job": jobs.get(index),
            "recovery": how,
            **info,
        })
    return {
        "anchor": ANCHOR,
        "mission": mission,
        "generated_by": "benchmarks/xval_method_selection/record_provenance.py",
        "cells_total": len(rows),
        "cells_present": sum(1 for r in rows if r["present"]),
        "records": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-dir", default=str(REPO / "runs" / "xval"),
                        help="directory holding cell_<i>.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--job", action="append", default=[], metavar="INDEX=cjN",
                        help="compute job that produced cell INDEX (repeatable)")
    parser.add_argument("--recovery", action="append", default=[],
                        metavar="INDEX=log_base64|fetch_tar|local",
                        help="how the bytes came back (repeatable)")
    parser.add_argument("--mission", type=int, default=None)
    args = parser.parse_args(argv)

    jobs = parse_job_specs(args.job, run_cell.CELLS)
    recovery: dict[int, str] = {}
    for spec in args.recovery:
        if "=" not in spec:
            raise SystemExit(f"--recovery needs INDEX=MODE, got {spec!r}")
        index_text, how = spec.split("=", 1)
        recovery[int(index_text)] = how

    manifest = build(Path(args.record_dir), jobs, recovery, mission=args.mission)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out}: {manifest['cells_present']}/{manifest['cells_total']} "
          f"records present")
    for row in manifest["records"]:
        print(f"  cell {row['index']} {row['label']:<14} job={row['job']} "
              f"recovery={row['recovery']:<10} "
              f"sha256={(row['sha256'] or '-')[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())