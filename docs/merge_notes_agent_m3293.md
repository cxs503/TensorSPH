# Merge record — `agent/m3293` × `main` (PR #118 un-dirtying)

The free-surface pressure closure work on this branch was developed against merge
base `184fef9`. By the time it was reviewed, `main` had advanced 20 commits, so
GitHub reported `mergeable=false`. This note records *how* the tree was
re-unified, and what was checked so that the resolution is auditable rather
than a mechanical overlay.

## 1. Refs

| ref | sha |
|---|---|
| merge base (`origin/main` ∩ branch at fork time) | `184fef9` |
| branch tip before this merge (`origin/agent/m3293`) | `83ea9d0` |
| target `origin/main` at merge time | `ba07a10` |
| merge commit (this branch) | `e6d8936` |

## 2. Conflict inventory

`git merge origin/main` produced exactly **one** content conflict:

    tests/test_laplace_pressure_floor.py

Everything else auto-merged. The two files both sides had touched:

| file | main's change | branch's change | outcome |
|---|---|---|---|
| `tests/test_laplace_pressure_floor.py` | import-block lint fix + trailing newline | new P0 acceptance gate, closure-attribution test, compression-restoration test | **conflict** → union |
| `benchmarks/laplace_pressure_floor/run.py` | trailing newline | `closure_runs` grid, `closure_acceptance()`, closure traces in figures | auto, verified |
| `src/tensorsph/droplet_impact.py` | none | `free_surface_pressure_closure()` + `Config.free_surface_pressure` | auto, verified |

### Resolution of the single conflict

Both sides touched the end of the file, but main's side of the hunk is
**empty** — main only added the missing final newline there (the branch's
original text had `\ No newline at end of file`). The resolution therefore:

* keeps the branch's block verbatim (the `within_15pct` verdict assertion,
  `test_closure_is_what_closes_the_gap`, `test_closure_restores_the_required_compression`);
* keeps main's import-block lint fix and trailing newline, which had merged
  cleanly above the conflict;
* leaves no conflict marker anywhere in the tree.

Nothing was dropped by choosing a side: the two sides are additive here.

## 3. Silent-loss audit

Two-directional diff after staging the merge, looking specifically for
deletions that an overlay-style resolution would hide:

| check | command | result |
|---|---|---|
| does the merge delete anything main has? | `git diff --cached --diff-filter=D --name-only origin/main` | **empty** |
| does the merge delete anything the branch has? | `git diff --cached --diff-filter=D --name-only 83ea9d0` | **empty** |
| merged tree vs `main` | `git diff --cached --stat origin/main` | 7 files, +2894/−19 — exactly this branch's own contribution |
| merged tree vs branch tip | `git diff --cached --stat 83ea9d0` | 56 files — exactly main's 20 commits |

The `−19` lines in the first row are this branch's own removals of the old
`xfail` marker and its docstring paragraph, not main content.

## 4. Lint gate

The merge exposed that two of the branch's new files were never green under
`ruff check .` (the job CI runs before pytest), which would have kept the PR
red even with the conflict resolved:

| file | code | fix |
|---|---|---|
| `benchmarks/laplace_pressure_floor/make_closure_doc.py` | E501 (104 > 100) | shortened a markdown table header |
| | F841 (`comp` unused) | removed the unused comprehension |
| | W292 | added the final newline |
| `tests/test_free_surface_pressure.py` | F401 (`cubic_spline_kernel` unused) | dropped from the import list |
| | F841 (`mass`, `h` unused) ×2 | renamed to `_mass` / `_h` (values still computed, semantics unchanged) |

No physics, thresholds or recorded numbers were touched: the fixes are a
header string, a dead list, an unused import and two dummy-variable renames.
`python -m ruff check .` → **All checks passed**.

## 5. Test evidence (this host, python 3.11.6, torch 2.7.1)

    tests/test_laplace_pressure_floor.py + tests/test_free_surface_pressure.py
        19 passed
    full suite (python -m pytest tests -q)
        374 passed, 7 xfailed, 2 failed, 469 s

The two "failures" are `XPASS(strict)`: `test_recorded_mode2_frequency_deficiency_is_reproduced`
and `test_oscillation_frequency_gate_is_not_met` carry strict `xfail` markers
whose own reason string records the measurement as architecture-fragile
(x86_64 +8.64 % vs loong64 −31.78 % on identical code). This host satisfies
their gate, so strict xfail reports them as unexpected passes. They are
untouched by this branch, and are not exercised by the closure code.

Measured parity instead of assertion: a throwaway worktree checked out at
`ba07a10` (pure `origin/main`) was run on the same host with the same command
and selector --

    # merged tree (this branch)
    failed: test_recorded_mode2_frequency_deficiency_is_reproduced,
            test_oscillation_frequency_gate_is_not_met
    # origin/main worktree, -k 'mode2_frequency_deficiency or frequency_gate_is_not_met'
    2 failed, 10 deselected in 225.06 s

with the identical `XPASS(strict)` reason string on both sides. The branch's own
test count is therefore not below the `main` baseline: same 2 environment-
dependent unexpected passes, everything else green.

## 6. Traps worth remembering

* **The worktree is not the package that gets imported.** `python -c "import tensorsph"`
  in this workspace resolved to `/root/tp-agent/work/m3482/src/tensorsph` — a
  *different* agent's worktree, via a stale editable install. Test runs must be
  launched with `PYTHONPATH=$PWD/src`, otherwise the suite silently grades
  someone else's code.
* **The 200 KB artifact cap is a content filter, not an error.** The branch's
  local commit carried a 273 KB `laplace_pressure_floor.png`; the platform
  dropped it when maturing the branch (`1 个文件被门禁滤除`), so the PR head never
  contained it. That is why the delivered tree has no figure binary under
  `benchmarks/laplace_pressure_floor/`: the plot is reproducible with
  `python benchmarks/laplace_pressure_floor/run.py` (its `make_figures`).

## 7. Scope

This merge changes no numerics. The closure feature, its acceptance grid and
`closure_acceptance.within_15pct = true` (worst case −7.17 % at dx = 2 mm,
c0 = 20 m/s) are exactly what the branch recorded in
`benchmarks/laplace_pressure_floor/report.json`; nothing was re-run to make a
number look better. No parameter is claimed to be re-validated here.