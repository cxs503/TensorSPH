# PR #85 heal: semantic union of the two B2 dam-break reference audits

**Repository:** `TensorSPH` · **PR:** #85 · **Branch:** `agent/m2905` (healed on `agent/m3849`)
**Merge base:** `5800532` (`Merge pull request #53 from cxs503/agent/m2588`)
**Old branch tip:** `b53f29e1ee7152a0ff1226c2001f8034b72a1642` (`origin/agent/m2905`)
**Merged tip (new):** `0918dfe` — parents `b53f29e` (`agent/m2905`) + `ba07a10` (`origin/main`, `Merge #104 (m3204)`)

The PR was `mergeable=false`. The cause is not a textual clash: **both sides added an
audit of the same bundled experimental reference at the same four paths**, each with its
own JSON schema, its own generator and its own regression guards. `git` reported four
`add/add` conflicts; a mechanical "take one side" resolution would have deleted one whole
audit together with its tests.

## 1. The two audits are not two versions of one artifact

| | `main` (theirs) | `agent/m2905` (ours) |
|---|---|---|
| Question asked | is the bundled table structurally a sketch, is the Ritter reference implemented at `c=2`, and what front coefficient does the solver actually produce? | where did the bundled table come from, how large is its uncertainty, and can the `physical validation fails` verdict be upheld? |
| JSON top-level keys | `geometry`, `table_audit`, `ritter`, `table_slope`, `window_tau`, `runs`, `solver`, `figures` | `tables`, `grid_stats`, `cross_source_spread`, `gates`, `verdict` |
| Module API | `audit_table`, `ritter_profile`, `ritter_checks`, `fit_window`, `front_coefficient`, `run_rung`, `targets`, `make_figures`, `main` | `_interp`, `load_bundled`, `load_crosscheck`, `_claimed_uncertainty`, `grid_stats`, `cross_source_spread`, `evaluate`, `main` |
| Extra inputs | solver runs (`tensorsph.dam_break.run_case`) | `martin_moyce_1952_crosscheck.json`, `reference_ladder.json` |
| Artifact | `benchmarks/dam_break/reference_audit.json` (1.7 k lines, solver-measured) | `benchmarks/dam_break/reference_audit.json` (234 lines, no simulation) |

Only one name clashes functionally (`main()`), and both write the **same** JSON path with
**different** schemas. Two exact-duplicate artifact paths cannot both survive, so the union
is done by namespace, not by file merge.

## 2. Resolution rule

`main` keeps the canonical paths: `make_audit_doc.py`, the generated document, the
regression guards and `benchmarks/dam_break/experimental_validation.py` on `main` all point
at `benchmarks/dam_break/reference_audit.{py,json}`. `agent/m2905`'s audit — which is new
content that nothing on `main` refers to — moves to an explicit `crosscheck_audit` family.

| `agent/m2905` path (old) | merged path (new) | edit beyond the move |
|---|---|---|
| `benchmarks/dam_break/reference_audit.py` | `benchmarks/dam_break/crosscheck_audit.py` | path constants only (`--json-out` default, docstring) |
| `benchmarks/dam_break/reference_audit.json` | `benchmarks/dam_break/crosscheck_audit.json` | none (content byte-identical) |
| `benchmarks/dam_break/make_reference_audit_doc.py` | `benchmarks/dam_break/make_crosscheck_audit_doc.py` | path constants + `F541`/`W292` lint |
| `docs/benchmarks/dam-break-reference-audit.md` | `docs/benchmarks/dam-break-crosscheck-audit.md` | regenerated from the renamed artifact |
| `tests/test_dam_break_reference_audit.py` | `tests/test_dam_break_crosscheck_audit.py` | import + artifact path only |
| `benchmarks/dam_break/reference_ladder.py` | *(kept)* | follows the import (`from .crosscheck_audit import ...`) |

Two incidental edits, both required by the repo's `ruff` gate (E/F/W) which `main` passes
and the merged tree must keep passing:

* `.scratch/probe.py` — `F821` undefined `cfg_dt` in a dead branch, `E401`/`F401` unused
  imports; rewritten into the same minimal config probe.
* `make_crosscheck_audit_doc.py` — `F541` f-string without placeholders, `W292` missing
  trailing newline (also on `crosscheck_audit.py`, `reference_ladder.py`, and the renamed
  test module).

Nothing else was touched: **all 162 paths that `main` changed are byte-identical to
`main`**, and all 10 paths that `agent/m2905` changed are present verbatim (4 with path
constants rewritten, 2 with the lint edits above).

## 3. No-silent-loss evidence (machine-checked)

`artifacts/pr85_heal/union_audit.py` walks every path either side changed relative to the
merge base and looks for that side's blob in the merged tree (same path, documented path,
or identical modulo the documented path rewrites / lint edits). Output
`artifacts/pr85_heal/union_audit.json`:

```
main: 162 paths -> {'kept verbatim at same path': 162}
agent/m2905: 10 paths -> {'kept verbatim at same path': 3,
                          'kept verbatim at documented path': 1,
                          'kept, path constants rewritten only': 4,
                          'kept, intentional edit (lint)': 2}
unexplained: none
```

Reproduction: `python3 artifacts/pr85_heal/resolve_union.py` then
`python3 artifacts/pr85_heal/union_audit.py`. `resolve_union.py` reconstructs the four
resolved files from the two merge stages, so the resolution is auditable rather than
hand-copied.

## 4. Test evidence (merged vs `main` baseline, same machine, same command)

`python3 -m pytest tests -q --continue-on-collection-errors -p no:cacheprovider`

| tree | passed | failed | xfailed | collection error | wall |
|---|---|---|---|---|---|
| merged `agent/m3849` @ `0918dfe` | **362** | 3 | 8 | 1 | 310 s |
| `origin/main` @ `ba07a10` (baseline) | **351** | 3 | 8 | 1 | 312 s |

The three failures and the one collection error are **identical on both trees**
(`test_dam_break_lattice_clearance::test_non_integer_full_width_column_stays_clear_of_the_wall`,
`test_oscillating_droplet::test_recorded_mode2_frequency_deficiency_is_reproduced`,
`test_oscillating_droplet::test_oscillation_frequency_gate_is_not_met`, and
`ModuleNotFoundError: tensorsph.lattice_consistency` in `test_kernel_sum_consistency.py`) —
pre-existing on `main`, not introduced here. The **+11 passing tests** are exactly
`agent/m2905`'s audit guards, now under their new module name.

Targeted checks on the two audit families, each in isolation:

* `pytest tests/test_dam_break_crosscheck_audit.py -q` → **11 passed** (m2905 side, renamed)
* `pytest tests/test_dam_break_reference_audit.py -q` → **12 passed** (main side, untouched)
* `python3 -m ruff check .` → **All checks passed** (baseline `main`: clean)
* `python3 benchmarks/dam_break/crosscheck_audit.py --json-out /tmp/xcheck.json` reproduces
  the committed `crosscheck_audit.json` numerically; the only difference is the recorded
  absolute table path (`.../work/m2905/...` vs `.../work/m3849/...`), which is pre-existing
  in the m2905 artifact and left untouched.
* `python3 benchmarks/dam_break/make_crosscheck_audit_doc.py` regenerates the committed
  document with no diff (`git status` clean).

## 5. Limits of this heal

* The branch is **not pushed** from the agent sandbox (network write policy); the healed
  tree lives on `agent/m3849` @ `0918dfe`, whose first parent is `b53f29e`
  (`agent/m2905`). Merging that commit into `main` is a fast-forward-free but trivially
  clean merge — `main` is an ancestor of it.
* `main`'s own pre-existing holes were **not** repaired (missing
  `tensorsph.lattice_consistency`, the three red tests). They are out of scope for a merge
  heal and are recorded here so they are not mistaken for merge damage.
* The committed `crosscheck_audit.json` still stores an absolute path from the
  `agent/m2905` worktree; normalising it would change an artifact the m2905 guards pin, so
  it was left verbatim.