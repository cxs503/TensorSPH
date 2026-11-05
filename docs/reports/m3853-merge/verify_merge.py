# ruff: noqa: E501  (report/prose templates carry long lines by design)
"""Machine proof that the TensorSPH#103 merge is non-lossy.

The three conflicted files (``benchmarks/laplace_law/convergence/relax_dx*.json``)
are *single-run snapshots* of the same experiment produced by two different
protocols, so no value-by-value textual union exists:

* ``main`` (m3312 chain) re-ran them with a **fixed step count** (steps = 900);
  because ``dt ~ dx / c0`` the physical window drifts with the resolution
  (0.79 / 1.05 / 1.32 ``t_sigma``);
* ``agent/m3163`` (PR #103) re-ran them with a **fixed capillary window**
  (``t = 1.0 t_sigma`` on every rung) and added the dual read-out plus the
  final field snapshot -- that run is the one the merged
  ``benchmarks/laplace_law/run.py`` produces.

The resolution therefore keeps the m3163 run as the *canonical* top-level
record and carries the main run **inside the same JSON** under
``alternate_runs`` (every key verbatim), plus byte-exact archived copies under
``docs/reports/m3853-merge/main-900step/`` as a regeneration-proof backstop.

This script asserts, for every conflicted file:

 1. every key path of the canonical (``agent/m3163``) document still exists in
    the merged document with an identical value  -> nothing of that side lost;
 2. every key path of the ``main`` document exists either at the top level or
    under ``alternate_runs[]`` with an identical value -> nothing of that side
    lost (this is the check a mechanical ``checkout --ours`` would fail);
 3. ``alternate_runs[0]`` deep-equals the ``main`` document field by field;
 4. the archived copies under ``docs/`` round-trip to the ``main`` document;
 5. the physical set-up is unchanged (same ``dt_s``, same ``t_sigma``);
 6. the canonical rungs sit on the fixed window ``t = 1.0 t_sigma`` and still
    satisfy the live criterion of ``tests/test_laplace_law.py`` (``t > 0.5``);
 7. the ``main`` history samples lie on the canonical ``dt`` grid (same
    integrator, same lattice -- the two runs are comparable, not unrelated);
 8. the merged document still carries what
    ``tests/test_laplace_resolution_ladder.py`` requires (dual read-out keys and
    a field snapshot matching ``setup.n_particles``).

Run from the repository root::

    python docs/reports/m3853-merge/verify_merge.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CASES = ("0015", "0020", "0025")
REL = "benchmarks/laplace_law/convergence/relax_dx0.{case}.json"

# tips at merge time (recorded so the proof can be replayed later)
BASE_REF = "184fef9"            # merge-base
MAIN_REF = "ba07a10"            # origin/main
BRANCH_REF = "b7af006"          # origin/agent/m3163 (PR #103)
PRESERVED = Path("docs/reports/m3853-merge/main-900step")

LADDER_KEYS = ("p_center_absolute_Pa", "rel_error_absolute_pct",
               "rel_error_absolute_tailavg_pct",
               "compression_ratio_measured_over_required",
               "rho_core_rel_std", "rel_error_from_mean_core_density_pct")


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


def lookup(obj, path: str):
    """Resolve a flattened key path back into the document (lists -> item 0)."""
    cur = obj
    for part in path.split("."):
        if part.endswith("[]"):
            part = part[:-2]
            cur = cur[part][0]
        else:
            cur = cur[part]
    return cur


def main() -> int:
    failures: list[str] = []
    rows: list[dict] = []

    for case in CASES:
        rel = REL.format(case=case)
        canonical = json.loads(git_show(f"{BRANCH_REF}:{rel}"))
        alternate = json.loads(git_show(f"{MAIN_REF}:{rel}"))
        merged = json.loads(Path(rel).read_text())

        # (1) canonical side: identical values for every key it had
        for path in sorted(key_paths(canonical)):
            if path.startswith("merge_resolution") or path.startswith("alternate_runs"):
                continue
            try:
                same = lookup(merged, path) == lookup(canonical, path)
            except (KeyError, IndexError):
                same = False
            if not same:
                failures.append(f"{rel}: canonical key changed/lost: {path}")

        # (2) alternate side: every key survives somewhere, with the same value
        lost = sorted(p for p in key_paths(alternate)
                      if p not in key_paths(merged)
                      and f"alternate_runs[].{p}" not in key_paths(merged))
        if lost:
            failures.append(f"{rel}: main-side keys lost: {lost[:5]}")
        for path in sorted(key_paths(alternate)):
            if f"alternate_runs[].{path}" not in key_paths(merged):
                failures.append(f"{rel}: main-side key not carried verbatim: {path}")

        # (3) alternate record is the main document, field for field
        alt = merged["alternate_runs"][0]
        for key, value in alternate.items():
            if alt.get(key) != value:
                failures.append(f"{rel}: alternate_runs[{key}] differs from main")

        # (4) archived backstop round-trips
        preserved = json.loads(
            (PRESERVED / f"relax_dx0.{case}.json").read_text())
        if preserved != alternate:
            failures.append(f"{rel}: archived main copy does not round-trip")

        # (5) same physical set-up
        same_dt = merged["setup"]["dt_s"] == alternate["setup"]["dt_s"]
        t_sig_kept = merged["setup"]["physical_time_s"] / merged["setup"]["t_over_t_sigma"]
        t_sig_alt = alternate["setup"]["physical_time_s"] / alternate["setup"]["t_over_t_sigma"]
        same_t_sigma = abs(t_sig_kept - t_sig_alt) < 1e-9
        if not (same_dt and same_t_sigma):
            failures.append(f"{rel}: physical set-up changed (dt={same_dt}, t_sigma={same_t_sigma})")

        # (6) canonical rung on the fixed window + live criterion
        window_fixed = abs(merged["setup"]["t_over_t_sigma"] - 1.0) < 1e-3
        live_ok = merged["setup"]["t_over_t_sigma"] > 0.5
        if not window_fixed:
            failures.append(f"{rel}: canonical window is not 1.0 t_sigma")
        if not live_ok:
            failures.append(f"{rel}: t_over_t_sigma <= 0.5 (tests/test_laplace_law.py would be red)")

        # (7) alternate history on the same dt grid
        step = merged["setup"]["dt_s"]

        def on_grid(t: float) -> bool:
            ratio = t / step
            return abs(ratio - round(ratio)) <= 1e-6

        grid_ok = all(on_grid(t) for t in alternate["history"]["t"])
        if not grid_ok:
            failures.append(f"{rel}: main history samples off the canonical dt grid")

        # (8) ladder contract still satisfied
        missing_ladder = [k for k in LADDER_KEYS if k not in merged["laplace"]]
        if missing_ladder:
            failures.append(f"{rel}: ladder read-out keys missing: {missing_ladder}")
        n = merged["setup"]["n_particles"]
        field = merged.get("field", {})
        if len(field.get("x_m", [])) != n or len(field.get("pressure_Pa", [])) != n:
            failures.append(f"{rel}: field snapshot does not match n_particles={n}")

        rows.append({
            "case": case,
            "kept_steps": merged["setup"]["steps"],
            "kept_t": merged["setup"]["t_over_t_sigma"],
            "alt_steps": alternate["setup"]["steps"],
            "alt_t": alternate["setup"]["t_over_t_sigma"],
            "keys_canonical": len(key_paths(canonical)),
            "keys_alternate": len(key_paths(alternate)),
            "keys_merged": len(key_paths(merged)),
        })

    print(f"{'case':<6}{'kept steps':>11}{'kept t/t_s':>12}{'main steps':>11}"
          f"{'main t/t_s':>11}{'keys m3163':>12}{'keys main':>11}{'keys merged':>12}")
    for r in rows:
        print(f"{r['case']:<6}{r['kept_steps']:>11d}{r['kept_t']:>12.4f}"
              f"{r['alt_steps']:>11d}{r['alt_t']:>11.4f}"
              f"{r['keys_canonical']:>12d}{r['keys_alternate']:>11d}"
              f"{r['keys_merged']:>12d}")

    print(f"\nrefs: merge-base {BASE_REF}, main {MAIN_REF}, agent/m3163 {BRANCH_REF}")
    if failures:
        print("\nFAIL")
        for item in failures:
            print(" -", item)
        return 1
    print("\nPASS: both sides retained (canonical + alternate_runs + archive); "
          "kept rungs share the fixed 1.0 t_sigma window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
