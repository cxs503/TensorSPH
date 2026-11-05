# ruff: noqa: E501  (report/prose templates carry long lines by design)
"""Non-lossy three-way resolution of the ``convergence/*.json`` conflicts.

Context
-------
PR #103 (branch ``agent/m3163``) and ``main`` both re-ran the same three
Young-Laplace relaxation artifacts (``relax_dx0.0015/0.0020/0.0025.json``), so
git cannot textually merge them: ``setup.steps``, ``physical_time_s`` and every
element of the ``history`` arrays differ.  Two independent runs of the same
experiment are *not* reconcilable value by value -- but neither may be dropped.

Resolution rule (semantic union, no mechanical overlay)
-------------------------------------------------------
Each file is a single-run record, so the union is expressed structurally:

* the ``agent/m3163`` run stays the **canonical record at the top level** --
  it is the one that matches the merged ``benchmarks/laplace_law/run.py``
  (``duration_t_sigma`` protocol: every rung read at the same capillary time
  ``t = 1.0 t_sigma``) and it is the schema superset (``field`` snapshot plus
  the dual read-out keys) that ``tests/test_laplace_resolution_ladder.py`` and
  ``benchmarks/laplace_law/make_ladder_doc.py`` require;
* the ``main`` run (fixed 900 steps -> window drifts with ``dx``:
  0.79 / 1.05 / 1.32 ``t_sigma``) is kept **inside the same JSON** under
  ``alternate_runs``, every key verbatim, plus a ``merge_resolution`` block
  that states which run is canonical and why.

Nothing is inferred or re-typed by hand: the two sides are read straight out of
git (merge stages 2 and 3, i.e. the exact blobs of ``origin/main`` and
``origin/agent/m3163``) and copied structurally.

The archived byte-exact copies under ``docs/reports/m3853-merge/main-900step/``
are written by ``make_artifacts.sh`` as a regeneration-proof backstop: running
``run_ladder.py`` later rewrites the artifacts and would drop
``alternate_runs``, but the archive cannot be overwritten by a benchmark run.

Usage (must be run with the merge in progress, i.e. the three files conflicted)::

    python docs/reports/m3853-merge/make_union.py
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

CASES = ("0015", "0020", "0025")
REL = "benchmarks/laplace_law/convergence/relax_dx0.{case}.json"

CANONICAL_REF = "ba07a10"      # origin/main at merge time
BRANCH_REF = "b7af006"         # origin/agent/m3163 at merge time
CANONICAL_SIDE = "agent/m3163"
ALTERNATE_SIDE = "main"

RECORD_KEYS = ("setup", "reference", "laplace", "spurious_current",
               "curvature_final", "history")


def stage(path: str, number: int) -> dict:
    """Read a merge stage blob (2 = ours/main, 3 = theirs/agent/m3163)."""
    out = subprocess.run(["git", "show", f":{number}:{path}"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def build(canonical: dict, alternate: dict, path: str) -> dict:
    merged = copy.deepcopy(canonical)
    alt = {"label": "main-900step-fixed-step-window",
           "source_ref": f"main@{CANONICAL_REF}",
           "protocol": (f"fixed step count (steps={alternate['setup']['steps']}); "
                        "the physical window therefore drifts with dx "
                        f"(t/t_sigma = {alternate['setup']['t_over_t_sigma']:.3f} here)"),
           "retained_because": ("the run is real data from the other side of the merge; "
                                "it is not reconcilable value by value with the "
                                "fixed-window run, so it is kept whole instead of dropped")}
    for key in RECORD_KEYS:
        alt[key] = copy.deepcopy(alternate[key])
    merged["alternate_runs"] = [alt]
    merged["merge_resolution"] = {
        "file": path,
        "canonical_side": CANONICAL_SIDE,
        "canonical_ref": BRANCH_REF,
        "alternate_side": ALTERNATE_SIDE,
        "alternate_ref": CANONICAL_REF,
        "rule": ("semantic union: the fixed-window run (identical t/t_sigma on every "
                 "rung, dual read-out, field snapshot) is the top-level record because "
                 "it is the one the merged run.py produces and the ladder tests require; "
                 "the fixed-step main run is preserved verbatim in alternate_runs and "
                 "byte-exactly under docs/reports/m3853-merge/main-900step/"),
        "detail": "docs/reports/m3853-merge/merge-conflict-resolution.md",
    }
    return merged


def main() -> None:
    for case in CASES:
        path = REL.format(case=case)
        ours = stage(path, 2)        # main
        theirs = stage(path, 3)      # agent/m3163
        merged = build(theirs, ours, path)
        Path(path).write_text(json.dumps(merged, indent=2) + "\n")
        print(f"union written: {path}  (canonical steps={merged['setup']['steps']}, "
              f"alternate steps={merged['alternate_runs'][0]['setup']['steps']})")


if __name__ == "__main__":
    main()
