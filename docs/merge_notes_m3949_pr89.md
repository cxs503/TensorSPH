# PR #89 merge notes (m3949): agent/m2923 -> main

**Scope.** Heal PR #89 (`agent/m2923`, last authored commit `19b0d00`)
against `main` (`ba07a10`) whose three-way merge was reported dirty
(`mergeable=false`).  The merge is a *semantic union*: no line authored on either side is
dropped silently, and every file touched by m2923 is accounted for below.

**Merge base.** `5800532`

## 1. Conflict inventory (1 file)

| file | conflict kind | resolution |
|---|---|---|
| `docs/verification_surface_tension_consistent.md` | both sides appended a trailing `## 7.` section (m2592 curvature-bias fix on main vs. m2923 pressure-jump erratum) | **kept both**, renumbered the incoming section to `## 8. 勘误（m2923 复核，2026-11-03）：压差判据与归因的修正`; sections 1-6 untouched |

No other file needed manual resolution: m2923's five *new* files
(`benchmarks/laplace_pressure_jump/{run.py,make_doc.py,validation_report.json}`,
`docs/verification_laplace_pressure_jump.md`, `tests/test_laplace_pressure_jump.py`)
are add-only and do not collide with main's additions
(`docs/verification_laplace_pressure_floor.md`, `tests/test_laplace_pressure_floor.py`, ...).

## 2. Silent-loss repair (the reason the branch was red)

`tests/test_laplace_pressure_jump.py::test_figure_exists` asserts that
`benchmarks/laplace_pressure_jump/laplace_pressure_jump.png` exists and is >10 kB.
The figure was present in m2923's authoring commit `2f2f238` (147 301 B) but was **dropped by the
deliverable gate** when the branch was re-committed as `19b0d00`
"matured mission deliverables（1 个文件被门禁滤除）", leaving the test red on the branch.
Recovered byte-exactly from `2f2f238` (`git hash-object` =
`6b752dc9e3dafb41b8a1019493d6fd50ff4e8c69`)
instead of regenerating it, so the doc's embedded figure and the test stay consistent.

## 3. Style gate

`python3 -m ruff check .` (repo pins `E,F,W`, line-length 100) was **clean on main** and
**10 errors on the merged tree**, all inside m2923's new files (`make_doc.py` 3x E501 + W292,
`run.py` W292, `tests/test_laplace_pressure_jump.py` W292).  Fixed by re-wrapping the three
long lines (string-concatenation only, no semantic change) and adding the missing trailing
newlines: `ruff check .` now reports *All checks passed*.

## 4. Test evidence (per-file, `-m 'not slow'`)

Method: each `tests/test_*.py` run in its own pytest process (`runs/run_suite.py`, 8-way
parallel), results captured as JSON and aggregated here programmatically - no hand-copied numbers.

| tree | files | passed | failed | xfailed |
|---|---|---|---|---|
| `main` @ ba07a10 (baseline) | 36 | 346 | 0 | 8 |
| merged (this branch) | 37 | 360 | 0 | 9 |

Merged tree is a strict superset (the extra file is m2923's `test_laplace_pressure_jump.py`,
14 passed + 1 xfailed) and no baseline test regressed.

Per-file results on the merged tree:

| file | rc | pytest summary |
|---|---|---|
| `test_adaptive_dt.py` | 0 | 9 passed, 2 deselected, 2 warnings |
| `test_boundary.py` | 0 | 7 passed, 2 deselected, 2 warnings |
| `test_capillary_wave.py` | 0 | 10 passed, 2 warnings |
| `test_conservation.py` | 0 | 9 passed, 2 warnings |
| `test_contact_angle.py` | 0 | 10 passed, 2 warnings |
| `test_convergence.py` | 0 | 7 passed, 2 warnings |
| `test_csf_consistent.py` | 0 | 20 passed, 2 warnings |
| `test_csf_curvature_audit.py` | 0 | 6 passed, 2 warnings |
| `test_csf_curvature_fit.py` | 0 | 10 passed, 2 warnings |
| `test_dam_break.py` | 0 | 14 passed, 1 deselected, 2 warnings |
| `test_dam_break_experimental.py` | 0 | 14 passed, 2 warnings |
| `test_dam_break_lattice_clearance.py` | 0 | 2 passed, 2 warnings |
| `test_dam_break_reference_audit.py` | 0 | 12 passed, 1 warning |
| `test_dam_break_resolution_ladder.py` | 0 | 7 passed, 1 warning |
| `test_dam_break_validation.py` | 0 | 9 passed, 2 warnings |
| `test_droplet_impact.py` | 0 | 23 passed, 2 warnings |
| `test_free_surface.py` | 0 | 8 passed, 2 warnings |
| `test_kernel_sum_consistency.py` | 0 | 11 passed, 2 warnings |
| `test_laplace_droplet.py` | 0 | 11 passed, 2 warnings |
| `test_laplace_law.py` | 0 | 9 passed, 1 xfailed, 2 warnings |
| `test_laplace_plateau.py` | 0 | 9 passed, 2 warnings |
| `test_laplace_pressure.py` | 0 | 8 passed, 2 warnings |
| `test_laplace_pressure_floor.py` | 0 | 8 passed, 1 xfailed, 2 warnings |
| `test_laplace_pressure_jump.py` | 0 | 14 passed, 1 xfailed, 1 warning |
| `test_laplace_validation.py` | 0 | 7 passed, 1 xfailed, 2 warnings |
| `test_laplace_verification.py` | 0 | 2 passed, 3 xfailed, 2 warnings |
| `test_lattice_wall_clearance.py` | 0 | 5 passed, 2 warnings |
| `test_neighbor_search.py` | 0 | 23 passed, 2 warnings |
| `test_pair_based_rhs.py` | 0 | 13 passed, 2 warnings |
| `test_paper_laplace_uncertainty.py` | 0 | 10 passed, 2 warnings |
| `test_poiseuille.py` | 0 | 5 passed, 2 deselected, 2 warnings |
| `test_ritter.py` | 0 | 11 passed, 3 warnings |
| `test_ritter_reference.py` | 0 | 8 passed, 2 warnings |
| `test_surface_tension.py` | 0 | 15 passed, 2 warnings |
| `test_surface_tension_laplace.py` | 0 | 5 passed, 2 xfailed, 2 warnings |
| `test_wall_pressure.py` | 0 | 3 passed, 2 warnings |
| `test_young_laplace.py` | 0 | 6 passed, 2 warnings |

**One inherited red, not a regression.** `tests/test_oscillating_droplet.py` is byte-identical to
main (`git diff origin/main -- tests/test_oscillating_droplet.py benchmarks/oscillating_droplet`
is empty), and its mode-2 pin is marked `xfail(strict)` with the reason "architecture-fragile
(x86_64 CI +8.64 % vs loong64 -31.78 %)".  On this x86_64 host it XPASSes, which `strict=True`
reports as a failure - the same verdict the file produces on `main`, so the merge neither
introduces nor hides it.  It is out of scope for a merge-heal and is reported, not patched.

## 5. How to reproduce

```bash
git worktree add ../m3949 origin/main && cd ../m3949
git merge --no-ff agent/m2923            # 1 conflict: the trailing section above
python3 -m ruff check .                  # All checks passed
python3 -m pytest -q -m 'not slow' -n 16 # merged tree
```
