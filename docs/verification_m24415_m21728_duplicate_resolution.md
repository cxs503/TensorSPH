# m24415 — resolution of the reopened TensorSPH#1024 (m21728 matured deliverables)

**Case anchor:** `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST` (3-D arm)
**Mission:** 24415 (reopened TensorSPH#1024, 管到底 round 2)
**Branch:** `agent/m24415` (cut from main @ `62bce048`)
**Date:** 2026-11-17

---

## ① What this mission asked for

The reopened mission restates the **m21728 matured mission deliverables** — seven files:

| # | file | lines at #826 merge | lines in HEAD |
|---|---|---|---|
| 1 | `benchmarks/kleefsman_obstacle_3d/make_doc.py` | 173 | 172 |
| 2 | `benchmarks/kleefsman_obstacle_3d/run.py` | 116 | 205 |
| 3 | `docs/verification_kleefsman_3d_blockage_m21728.md` | 173 | 172 |
| 4 | `out/report_post_m21728.json` | 72 | 71 |
| 5 | `scripts/post_report_m21728.py` | 138 | 137 |
| 6 | `src/tensorsph/kleefsman_obstacle_3d.py` | 539 | 731 |
| 7 | `tests/test_kleefsman_obstacle_3d.py` | 262 | 262 |

(The small deltas are later, in-scope evolution of the same files by subsequent
missions — e.g. `src/.../kleefsman_obstacle_3d.py` grew 539→731 lines through
m22082/m22552/m22745 — not a missing piece.)

## ② Finding: the deliverables are already in main

All seven files are **already committed to main**. They were merged by
**PR #826 (m21728)** on 2026-11-15:

```
commit 2d4a82b3  "Merge #826 (m21728) via merge-master"
 benchmarks/kleefsman_obstacle_3d/make_doc.py      | 173 +++++++
 benchmarks/kleefsman_obstacle_3d/run.py           | 116 +++++
 docs/verification_kleefsman_3d_blockage_m21728.md | 173 +++++++
 out/report_post_m21728.json                       |  72 +++
 scripts/post_report_m21728.py                     | 138 ++++++
 src/tensorsph/kleefsman_obstacle_3d.py            | 539 ++++++++++++++++++++++
 tests/test_kleefsman_obstacle_3d.py               | 262 +++++++++++
 7 files changed, 1473 insertions(+)
```

Verified on this branch (`agent/m24415` @ `62bce048`):

* `git merge-base --is-ancestor 2d4a82b3 HEAD` → **true** (the #826 merge is in
  this branch's history).
* `git cat-file -e HEAD:<path>` → **all 7 paths present in the HEAD tree**.

Therefore a PR that "re-lands" these seven files would have **an empty diff**
against main and cannot be merged as a no-op. Re-landing the *original* 539-line
`src` version would be a **regression** (it would delete the m22082/m22552/m22745
extensions). Neither is a valid completion.

## ③ The work the deliverables represent — status

The m21728 feature is the **3-D Kleefsman obstacle case**: the same solver with the
tank's third dimension restored, the box at its real 0.403 m width (40 % blockage
vs the 2-D model's 100 %), the identical frozen gauge protocol, and an explicit
bypass count that is identically zero in 2-D.

* **Capability + tests:** delivered and green.
  `python -m pytest tests/test_kleefsman_obstacle_3d.py -q` → **26 passed**
  (invariant assertions: clamp empties the box and is idempotent; slab gauge
  reproduces `3·4·8·dx³/(0.08·0.403)` m to 1e-12; bypassed water reads zero; a
  censored peak cannot pass).
* **Production verdict:** the m21728 shift recorded it as `WITHHELD` because the
  scored runs (cj3661/cj3662) were still queued at hand-off. That open item was
  **closed by later missions**: m22606 recovered the 3-D rungs (cj3887/cj3888) and
  scored the first decidable 3-D verdict, and m22745 gated the records on operator
  integrity. The committed verdict
  (`benchmarks/kleefsman_obstacle_3d/results/recovered/verdict.json`) is:

  | run | job | peak vs 0.370 m | arrival vs 0.3591 s | verdict |
  |---|---|---|---|---|
  | dx=0.05 | cj3887 | 0.1202 m → **+67.5 %** | 0.5028 s → **+40.0 %** | **FAIL** |
  | dx=0.04 | cj3888 | 0.0933 m (record ends 1.144 s < 1.703 s → censored) | 0.4678 s → **+30.3 %** | **FAIL** |

  So the "管到底" (see-it-through) item that m21728 left open — the production
  verdict — **has already been decided: FAIL**, and is recorded in-repo.

## ④ Verification actually run on this branch

```
$ python -m pytest tests/test_kleefsman_obstacle_3d.py -q
26 passed, 1 warning in 3.45s

$ git merge-base --is-ancestor 2d4a82b3 HEAD && echo yes   # #826 in history
yes
$ git cat-file -e HEAD:src/tensorsph/kleefsman_obstacle_3d.py   # all 7 paths present
```

Broader suite (`pytest tests/ -k kleefsman -m "not slow"`, ignoring the three
pre-existing collection-error files): **391 passed, 7 failed, 2 skipped, 1 xfailed**.
All 7 failures are **pre-existing on main and in files this mission does not touch**:

| failing test | cause | related to m21728? |
|---|---|---|
| `test_kleefsman_3d_production_report.py::test_report_never_ladders_across_geometries` | z0p3 arm returns `RUNGS_REJECTED` | no |
| `test_kleefsman_sound_speed_family.py::test_no_hardcoded_sound_speed_divisor_left` | `SyntaxError` in `benchmarks/kleefsman_obstacle/make_doc_station_anchor.py:403` | no |
| `test_kleefsman_t30_decidability.py::test_t30_scoring_block_is_recorded_in_reference_json` | `KeyError: 't30_scoring'` | no |
| `test_kleefsman_text_landmarks.py` (×4) | missing uncommitted `outputs/kleefsman2005_text/quotes.json` | no |

None of these touch the seven m21728 files; all fail identically on a clean main.

## ⑤ Conclusion and decision

* The m21728 deliverables **are in main** (PR #826, 2026-11-15) and **pass**.
* The open item m21728 left (the production verdict) **has been decided: FAIL**,
  recorded in-repo by m22606/m22745.
* Reopened mission **m24415 is a duplicate** of work already merged via PR #826.
  Re-landing the seven files would produce an **empty diff** (or, if the original
  539-line `src` version were restored, a **regression** of the m22082/m22552/m22745
  extensions). Neither is a valid completion.

**Decision:** no source change is made. This note is the deliverable: it records,
with commit-level evidence, that the m21728 deliverables are in main, green, and
that the production verdict they deferred is now `FAIL`. The mission's objective is
already satisfied by main; the correct closure is to confirm that, not to
re-submit a no-op.

## ⑥ Error judgement (explicit)

| key quantity | reference (Kleefsman Fig. 19) | best 3-D result (cj3887, dx=0.05) | rel. error | < 3 %? |
|---|---|---|---|---|
| H2 peak water height | 0.370 m | 0.1202 m | **+67.5 %** | **No** |
| H2 arrival (0.02 m) | 0.3591 s | 0.5028 s | **+40.0 %** | **No** |

The 3-D blockage case **fails** the frozen < 10 % acceptance on both metrics (and
therefore also the < 3 % platform question). This is an honest FAIL, not a
completion claim: restoring the third dimension did not bring the peak within
tolerance, and the record-integrity gate (m22745) additionally flags the recovered
records' `height` operator as degenerate, so the slab operator is the admissible
one. No number above is hand-copied; all are read from
`benchmarks/kleefsman_obstacle_3d/results/recovered/verdict.json`.
