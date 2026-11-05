#!/usr/bin/env python3
"""Verify the m3793 conflict resolution for the three laplace-law convergence artefacts.

The merge of ``agent/m3163`` into ``main`` conflicted only on the three
per-resolution result files under ``benchmarks/laplace_law/convergence/``.
Both sides had re-run the *same* benchmark code with a different physical
window, so no textual union of scalars exists; the resolution keeps the
``agent/m3163`` run (fixed capillary window, dual read-out) as the canonical
rung and preserves the ``main`` (m3312, 900 step) run byte-exactly under
``docs/reports/m3793-merge/``.

This script proves the resolution is non-lossy at the JSON-key level and that
the kept run is a strict extension of the dropped one:

  * resolved file == stage 2 (``agent/m3163``) byte for byte;
  * every key of the ``main`` document still exists in the resolved document;
  * same ``dt_s`` / ``t_sigma`` (same physical set-up, same integrator);
  * resolved rung read at the fixed capillary window ``t = 1.0 t_sigma``;
  * the live ``t_over_t_sigma > 0.5`` assertion of ``tests/test_laplace_law.py``
    still holds;
  * ``main``'s ``history.t`` samples lie on the same ``dt`` grid.

The two coarser ``main`` rungs actually integrated *longer* than 1.0 t_sigma
(a fixed 900 steps means 1.05 and 1.32 t_sigma at dx = 2.0 mm and 2.5 mm);
that information is kept verbatim in the archived copies and is the very
reason the ladder was rebuilt on a fixed window.

Run from the repository root::

    python docs/reports/m3793-merge/verify_merge.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CASES = ("0015", "0020", "0025")
BRANCH_TIP = "1ba56a5"      # agent/m3163 at the time of the merge
MAIN_TIP = "origin/main"    # main at the time of the merge
BENCH = "benchmarks/laplace_law/convergence"
PRESERVED = Path("docs/reports/m3793-merge")


def git_show(rev_path: str) -> str:
    out = subprocess.run(["git", "show", rev_path], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git show {rev_path} failed: {out.stderr.strip()}")
    return out.stdout


def key_paths(obj, prefix=""):
    """Flatten a JSON document into the set of its key paths."""
    paths = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= key_paths(value, path)
    elif isinstance(obj, list):
        for item in obj:
            paths |= key_paths(item, f"{prefix}[]")
    return paths


def main() -> int:
    failures: list[str] = []
    rows: list[dict] = []

    for case in CASES:
        rel = f"{BENCH}/relax_dx0.{case}.json"
        ours = git_show(f"{BRANCH_TIP}:{rel}")             # agent/m3163 tip
        theirs_text = git_show(f"{MAIN_TIP}:{rel}")        # main tip
        resolved_text = Path(rel).read_text()

        if resolved_text != ours:
            failures.append(f"{rel}: resolved file is not byte-identical to the kept side")

        kept = json.loads(resolved_text)
        dropped = json.loads(theirs_text)
        preserved_text = (PRESERVED / f"relax_dx0.{case}.main-m3312-900step.json").read_text()
        if json.loads(preserved_text) != dropped:
            failures.append(f"{rel}: preserved copy under docs/ does not round-trip")

        missing = key_paths(dropped) - key_paths(kept)
        if missing:
            failures.append(f"{rel}: keys lost from the main side: {sorted(missing)[:5]}")

        same_dt = kept["setup"]["dt_s"] == dropped["setup"]["dt_s"]
        t_sigma_kept = kept["setup"]["physical_time_s"] / kept["setup"]["t_over_t_sigma"]
        t_sigma_dropped = dropped["setup"]["physical_time_s"] / dropped["setup"]["t_over_t_sigma"]
        same_t_sigma = abs(t_sigma_kept - t_sigma_dropped) < 1e-9

        # The ladder invariant introduced by agent/m3163: every rung is read at
        # the *same* capillary time.  main used a fixed step count, so its
        # physical window drifts with dx (0.79 / 1.05 / 1.32 t_sigma) and the
        # rungs are not comparable.
        window_is_fixed = abs(kept["setup"]["t_over_t_sigma"] - 1.0) < 1e-3
        # The live assertion in tests/test_laplace_law.py must still hold.
        passes_live_criterion = kept["setup"]["t_over_t_sigma"] > 0.5

        t_dropped = dropped["history"]["t"]
        step = kept["setup"]["dt_s"]

        def on_grid(t: float) -> bool:
            ratio = t / step
            return abs(ratio - round(ratio)) <= 1e-6

        grid_ok = all(on_grid(t) for t in t_dropped)

        for name, ok in (
            ("same dt_s", same_dt),
            ("same t_sigma", same_t_sigma),
            ("kept window fixed at 1.0 t_sigma", window_is_fixed),
            ("kept rung passes the live t > 0.5 criterion", passes_live_criterion),
            ("dropped history on the same dt grid", grid_ok),
        ):
            if not ok:
                failures.append(f"{rel}: check '{name}' failed")

        rows.append({
            "case": case,
            "kept_steps": kept["setup"]["steps"],
            "kept_t": kept["setup"]["t_over_t_sigma"],
            "dropped_steps": dropped["setup"]["steps"],
            "dropped_t": dropped["setup"]["t_over_t_sigma"],
            "same_dt": same_dt,
            "same_t_sigma": same_t_sigma,
            "keys_added": len(key_paths(kept) - key_paths(dropped)),
            "dropped_longer": dropped["setup"]["t_over_t_sigma"] > kept["setup"]["t_over_t_sigma"],
        })

    print(f"{'case':6} {'kept steps':>10} {'kept t/t_s':>11} {'main steps':>10} "
          f"{'main t/t_s':>10} {'same dt':>8} {'same t_s':>9} {'keys+':>6} "
          f"{'main window longer':>18}")
    for r in rows:
        print(f"{r['case']:6} {r['kept_steps']:>10d} {r['kept_t']:>11.4f} "
              f"{r['dropped_steps']:>10d} {r['dropped_t']:>10.4f} "
              f"{str(r['same_dt']):>8} {str(r['same_t_sigma']):>9} "
              f"{r['keys_added']:>6d} {str(r['dropped_longer']):>18}")

    if failures:
        print("\nFAIL")
        for item in failures:
            print(" -", item)
        return 1
    print("\nPASS: resolution is non-lossy; kept rungs share the fixed 1.0 t_sigma window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
