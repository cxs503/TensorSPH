# m25393 — resolution of the reopened TensorSPH#1046 (m22109 regrade deliverables)

**Case anchor:** `CASE-SPH-WEDGE-ENTRY-2D` (2-D wedge water-entry slamming pressure)
**Mission:** 25393 (reopened TensorSPH#1046, 管到底 round 1)
**Branch:** `agent/m25393` (rebased onto main @ `20f34a84`)
**Date:** 2026-11-19

---

## ① What this mission asked for

The reopened mission restates the **m22109 matured mission deliverables** — four files:

| # | file | lines in PR #1046 |
|---|---|---|
| 1 | `benchmarks/case_sph_wedge_entry_2d/run.py` | +22/-3 |
| 2 | `docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json` | +100/-0 |
| 3 | `docs/figures/case_sph_wedge_entry_2d_regrade_ladder.png` | +0/-0 (binary) |
| 4 | `tests/test_case_sph_wedge_entry_2d_literature_regrade.py` | +22/-1 |

## ② Finding: PR #1046 is a byte-for-byte duplicate of the already-merged PR #863

Both PRs were opened from the **same branch at the same commit**:

| | PR #863 | PR #1046 |
|---|---|---|
| head | `agent/m22109` @ `9b30f5e551` | `agent/m22109` @ `9b30f5e551` |
| base | `main` | `main` |
| created | 2026-11-15T16:56:39Z | 2026-11-16T21:20:09Z |
| outcome | **merged 2026-11-15T16:57:07Z** (28 s) | closed 2026-11-16T21:21:06Z (57 s, unmerged) |
| files | run.py +22/-3 · regrade.json +100/-0 · ladder.png +0/-0 · test +22/-1 | **identical** |

The merge commit on main (`31246d9f`, "Merge #863 (m22109) via merge-master",
2026-11-16 00:57 +0800) carries exactly the same four files with the same line
counts:

```
commit 31246d9f  "Merge #863 (m22109) via merge-master"
 benchmarks/case_sph_wedge_entry_2d/run.py          |  25 +++++-
 .../case_sph_wedge_entry_2d_regrade/regrade.json   | 100 +++++++++++++++++++++
 .../case_sph_wedge_entry_2d_regrade_ladder.png     | Bin 99908 -> 108019 bytes
 ...t_case_sph_wedge_entry_2d_literature_regrade.py |  23 ++++-
 4 files changed, 144 insertions(+), 4 deletions(-)
```

The m22109 branch was pushed twice by the auto-pusher (tp_pusher): the first push
became #863 and was auto-merged by the merge governor in 28 s; the second push
became #1046 and was closed as a duplicate. **Re-landing #1046 verbatim would
therefore be an empty diff against main** (or, if the original `run.py`/test
versions were restored, a *regression* of the m23321/m23975/m25023 fixes that
landed on top — see ③).

## ③ File-by-file status on current main (`c5dccec0`)

| file | PR #1046 head | current main | verdict |
|---|---|---|---|
| `regrade.json` | 13531 B | **byte-identical** (`diff` empty) | in main |
| `regrade_ladder.png` | md5 `a4d0e17d…c54c1` | **byte-identical** (same md5) | in main |
| `run.py` | hardcoded `"status": "REAL"` | evolved superset: status inherited from `refs.json` (m23321), `sys.path` bootstrap (m23975), grading against the mission-designated Z&F reference (m25023) | superseded (improved) |
| test file | asserted `zf["status"] == "REAL"` | evolved: asserts the inherited `WITHHELD` + `provenance_tier` (m23321) | superseded (improved) |

The two "superseded" files are *strict improvements* made by later missions:

* **m23321** (`ff43c483`): the hardcoded `"REAL"` contradicted `refs.json`'s own
  note (the primary article is unread, so the Z&F entry is `WITHHELD` with the
  value tracked as a secondary).  Restoring `"REAL"` would re-introduce that
  defect.
* **m23975** (`dd2f35f5`): the bare `from literature_refs import …` added by
  m22109 broke full-suite collection; the `sys.path` bootstrap in `run.py` is the
  fix.
* **m25023** (`7fa0eec8`): `cp_max_reference` switched from the Wagner anchor
  (`C²+1 = 8.402`) to the mission-designated Z&F COM value (`6.92`, q units) —
  the whole point of the regrade.

## ④ Work actually done on this branch

The deliverables are in main, but two real defects in the regrade deliverable
family were found and closed:

1. **The PR #1046 test was not full-suite-safe.**
   `test_driver_refs_derived_wires_the_mission_designated_reference` (the test
   PR #1046 added, in its m23321-evolved form) did a bare `import run as R`.
   Under a full-suite run, `sys.modules['run']` may already hold *another*
   case's `run.py` (e.g. `scripts/xval_projection_laplace/run.py`, imported at
   top level by `tests/test_xval_projection_laplace.py`), so the test silently
   graded the wrong driver and failed with
   `AttributeError: module 'run' has no attribute 'refs_derived'`.
   Reproduced deterministically:
   `pytest tests/test_xval_projection_laplace.py tests/test_case_sph_wedge_entry_2d_literature_regrade.py`
   → 1 failed on clean main; **0 failed after the fix** (the driver is now loaded
   by file location under the unique name `case_wedge_run`, the same pattern
   `tests/test_case_sph_wedge_entry_2d.py` already uses).

2. **The regrade doc's test count was stale (17 vs 18).**
   `make_literature_regrade_doc.py` hardcoded "17 项测试" / "17 passed", but the
   regrade test file gained its 18th test when the driver-wiring test was
   re-landed in evolved form.  The count is now read from the test file
   (`n_tests()`), never typed; the doc was regenerated (only the two count
   strings changed; every other number is injected from `regrade.json` and is
   unchanged).

## ⑤ Verification actually run on this branch

```
$ python -m pytest tests/test_case_sph_wedge_entry_2d_literature_regrade.py -q
18 passed

$ python -m pytest tests/test_xval_projection_laplace.py \
      tests/test_case_sph_wedge_entry_2d_literature_regrade.py -q   # order-dependent repro
25 passed, 2 skipped          # was: 1 failed on clean main

$ python -m pytest tests/ -k "wedge or regrade" -q --continue-on-collection-errors
373 passed, 10 skipped, 1 failed   # the 1 failure is pre-existing, see ⑥

$ python -m pytest tests/ -q --continue-on-collection-errors        # full suite, cluster cj5617
(see ⑥)
```

## ⑥ Pre-existing failures on clean main (not caused by this mission)

| failing item | cause | related to m22109? |
|---|---|---|
| `tests/test_case_sph_wedge_entry_2d.py::test_grade_passes_on_exact_reference_field` | m25023 switched the grader's `cp_max_reference` from the Wagner anchor (8.402) to the mission-designated Z&F COM value (6.92); the test's synthetic field is the exact *Wagner* profile, whose peak is +21.4 % above the Z&F value, so the band-peak verdict is now FAIL. The test's premise (Wagner profile = the reference field) predates the reference switch. | no — different file, different functional point (main-driver grading) |
| 30 collection errors (`test_case_poiseuille*`, `test_couette*`, …) | this machine's `site-packages` contains an unrelated `benchmarks` package that shadows the repo's `benchmarks/` directory (`ModuleNotFoundError: No module named 'benchmarks.case_poiseuille'`). Environmental, not a repo defect. | no |

Both fail identically on a clean `main` (verified by stashing this branch's
changes and re-running).  Neither touches the four m22109 files.

## ⑦ Error judgement (explicit)

| key quantity | reference | measured | rel. error | < 3 %? |
|---|---|---|---|---|
| in-repo Wagner profile vs published eq. (10) | `wagner_eq10_over_q` | `wagner_p_over_q` | 7.1e-15 (machine precision) | **yes** |
| Z&F composite (COM) vs exact similarity (EIM) | 6.92 (q) | 6.90 (q) | 0.29 % | **yes** |
| regrade.json vs PR #1046 head | 13531 B | 13531 B | 0 (byte-identical) | **yes** |
| ladder figure vs PR #1046 head | md5 `a4d0e17d…` | md5 `a4d0e17d…` | 0 (byte-identical) | **yes** |
| SPH `dx=1 mm` band peak vs Z&F COM | 6.92 | 6.805 | −1.66 % | **yes** (and < 15 % gate → PASS) |
| SPH `dx=0.5 mm` band peak vs Z&F COM | 6.92 | 8.984 | +29.83 % | **no** (estimator uncertainty 20.9 % > 15 % gate → UNSCORED, honest) |

The case's production verdict remains **PARTIAL_SINGLE_GRID** (one judgeable
grid inside the gate; the closure rule asks for ≥ 2 with the same sign).  That
is the honest state of the physics; this mission closes the *bookkeeping* debt
(the duplicate PR), not the physics.

## ⑧ Decision

* The m22109 deliverables **are in main** (PR #863, 2026-11-15T16:57Z) and
  **pass**.  PR #1046 was a duplicate push of the same branch/commit and was
  correctly closed.
* This PR is the 管到底 closure: it records the duplicate-PR finding with
  commit-level evidence and closes the two real (small) defects in the regrade
  deliverable family (full-suite test isolation; stale test count in the doc).
* No solver, no core module, no hot file is touched.  Changed files:
  `tests/test_case_sph_wedge_entry_2d_literature_regrade.py`,
  `benchmarks/case_sph_wedge_entry_2d/make_literature_regrade_doc.py`,
  `docs/case_sph_wedge_entry_2d_literature_regrade.md`, and this document.
