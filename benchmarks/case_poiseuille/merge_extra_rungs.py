"""Merge split extra-rung outputs into ``out/backflow_level_target.json``.

The original single job (cj4578) crashed mid-run (glibc heap corruption on the
node) after H/dx = 24, so the remaining work was re-submitted as two isolated
jobs:

* cj4840: ``--rungs 32 48``            -> ``out/extra_rungs_32_48.json``
* cj4841: ``--rungs 24`` (+ long-time) -> ``out/extra_rungs_24_longtime.json``

This tool stitches the two payloads back into the single target schema that
``backflow_level.py`` and ``make_backflow_level_doc.py`` consume:

    runs               = [rung 24, rung 32, rung 48]  (sorted by resolution)
    attribution        = per-run decompositions, same order
    longtime           = long-time control decompositions (t_end = 2.0, 4.0)
    longtime_runs_meta = wall-clock / window metadata for the control

Every field is copied verbatim from the job payloads; nothing is re-derived
here.  Usage::

    python3 benchmarks/case_poiseuille/merge_extra_rungs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PARTS = (OUT / "extra_rungs_24_longtime.json",
         OUT / "extra_rungs_32_48.json")
TARGET = OUT / "backflow_level_target.json"


def main() -> int:
    parts = []
    for p in PARTS:
        if not p.exists():
            print(f"missing {p} -- fetch the cluster job output first")
            return 2
        parts.append(json.loads(p.read_text()))

    runs, attribution, longtime, meta = [], [], [], []
    base = parts[0]
    for part in parts:
        runs.extend(part.get("runs", []))
        attribution.extend(part.get("attribution", []))
        longtime.extend(part.get("longtime", []))
        meta.extend(part.get("longtime_runs_meta", []))

    order = {r["resolution_H_over_dx"]: i for i, r in enumerate(runs)}
    runs.sort(key=lambda r: r["resolution_H_over_dx"])
    attribution.sort(key=lambda d: order[d["resolution_H_over_dx"]])

    payload = {
        "case": base["case"],
        "anchor": base["anchor"],
        "study": base["study"],
        "purpose": base["purpose"],
        "parameters": base["parameters"],
        "runs": runs,
        "attribution": attribution,
        "longtime": longtime,
        "longtime_runs_meta": meta,
        "device": base["device"],
        "torch": base["torch"],
        "python": base["python"],
        "wall_clock_s": sum(p.get("wall_clock_s", 0.0) for p in parts),
        "generated_by": "benchmarks/case_poiseuille/merge_extra_rungs.py "
                        "(parts: cj4840 + cj4841; original cj4578 crashed after "
                        "H/dx=24 with 'free(): corrupted unsorted chunks')",
    }
    TARGET.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"wrote {TARGET}: rungs "
          f"{[r['resolution_H_over_dx'] for r in runs]}, "
          f"longtime t_end={[m['t_end'] for m in meta]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
