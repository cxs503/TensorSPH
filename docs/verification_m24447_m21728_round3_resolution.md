# m24447 — Round-3 resolution of the reopened TensorSPH#1024 (m21728 matured deliverables)

**Case anchor:** `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST` (3-D arm)
**Mission:** 24447 (reopened TensorSPH#1024, 管到底 round 3)
**Branch:** `agent/m24447` (rebased onto real GitHub main @ `923b3019`)
**Date:** 2026-11-17
**Supersedes:** `docs/verification_m24415_m21728_duplicate_resolution.md` (round 2,
merged to main via **PR #1131**). This note is the round-3 closure and is
self-contained; it corrects one factual point in the round-2 note (see ⑤.2).

---

## ① Case introduction

The reopened mission restates the **m21728 matured mission deliverables** — the 3-D
Kleefsman obstacle case (`CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`, 3-D arm): the same
solver with the tank's third dimension restored, the box at its real **0.403 m** width
(40 % blockage vs the 2-D model's 100 %), the identical frozen gauge protocol, and an
explicit bypass count that is identically zero in 2-D.

| quantity | experimental value (Kleefsman Fig. 19, digitised) |
|---|---|
| H2 peak water height | 0.370 m @ 1.7027 s |
| H2 arrival (0.02 m threshold) | 0.3591 s |

The seven deliverables the mission lists, and their state in real GitHub main
(`923b3019`):

| # | file | lines at #826 merge | lines in HEAD | in HEAD? |
|---|---|---|---|---|
| 1 | `benchmarks/kleefsman_obstacle_3d/make_doc.py` | 173 | 172 | yes |
| 2 | `benchmarks/kleefsman_obstacle_3d/run.py` | 116 | 207 | yes |
| 3 | `docs/verification_kleefsman_3d_blockage_m21728.md` | 173 | 172 | yes |
| 4 | `out/report_post_m21728.json` | 72 | 71 | yes |
| 5 | `scripts/post_report_m21728.py` | 138 | 137 | yes |
| 6 | `src/tensorsph/kleefsman_obstacle_3d.py` | 539 | 731 | yes |
| 7 | `tests/test_kleefsman_obstacle_3d.py` | 262 | 262 | yes |

(The small deltas are later, in-scope evolution of the same files by subsequent
missions — e.g. `src/.../kleefsman_obstacle_3d.py` grew 539→731 lines through
m22082/m22552/m22745 — not a missing piece.)

## ② How this was done (actual actions on this branch)

This mission adds **no source change** (the feature is already in main). The actions
were **audit + adjudication**, all reproducible:

1. `git merge-base --is-ancestor 2d4a82b3 HEAD` → **true** (the #826 merge is an
   ancestor of this branch).
2. `git cat-file -e HEAD:<path>` ×7 → **all 7 deliverable paths present in the HEAD
   tree** (HEAD rebased onto real main `923b3019`).
3. `python -m pytest tests/test_kleefsman_obstacle_3d.py -q` → **26 passed, 1 warning
   in 3.62 s** (local `src` pinned first on `sys.path` by `tests/conftest.py`, so the
   gate tests this tree, not a sibling editable install).
4. Read `benchmarks/kleefsman_obstacle_3d/results/recovered/verdict.json` → the
   production verdict m21728 deferred (`WITHHELD` at hand-off) was later decided
   **FAIL** by m22606/m22745 and is recorded in-repo.
5. Wrote this note (`docs/verification_m24447_m21728_round3_resolution.md`) — the
   single new file of this mission.

Toolchain: git forensics + pytest + direct read of the committed JSON. **Zero
hand-copied numbers.**

## ③ Computation results (programmatic injection)

State of the m21728 deliverables in real GitHub main (HEAD = `923b3019` + this note):

| deliverable | in HEAD? | lines |
|---|---|---|
| `src/tensorsph/kleefsman_obstacle_3d.py` | yes | 731 |
| `tests/test_kleefsman_obstacle_3d.py` | yes | 262 |
| `benchmarks/kleefsman_obstacle_3d/run.py` | yes | 207 |
| `benchmarks/kleefsman_obstacle_3d/make_doc.py` | yes | 172 |
| `docs/verification_kleefsman_3d_blockage_m21728.md` | yes | 172 |
| `scripts/post_report_m21728.py` | yes | 137 |
| `out/report_post_m21728.json` | yes | 71 |

The 3-D production verdict m21728 deferred, later decided and committed by
m22606/m22745 (`results/recovered/verdict.json`):

| run | job | peak vs 0.370 m | arrival vs 0.3591 s | verdict |
|---|---|---|---|---|
| dx=0.05 | cj3887 | 0.1202 m (**+67.5 %**) | 0.5028 s (**+40.0 %**) | **FAIL** |
| dx=0.04 | cj3888 | 0.0933 m (censored: record ends 1.144 s < 1.703 s) | 0.4678 s (**+30.3 %**) | **FAIL** |

Broader kleefsman suite on this branch (`pytest tests/ -k kleefsman -m "not slow"`,
ignoring three pre-existing collection-error files): **391 passed, 7 failed, 2
skipped, 1 xfailed.** All 7 failures are **pre-existing on a clean main and in files
this mission does not touch**:

| failing test | cause | related to m21728? |
|---|---|---|
| `test_kleefsman_3d_production_report.py::test_report_never_ladders_across_geometries` | z0p3 arm returns `RUNGS_REJECTED` | no |
| `test_kleefsman_sound_speed_family.py::test_no_hardcoded_sound_speed_divisor_left` | `SyntaxError` in `benchmarks/kleefsman_obstacle/make_doc_station_anchor.py:403` | no |
| `test_kleefsman_t30_decidability.py::test_t30_scoring_block_is_recorded_in_reference_json` | `KeyError: 't30_scoring'` | no |
| `test_kleefsman_text_landmarks.py` (×4) | missing uncommitted `outputs/kleefsman2005_text/quotes.json` | no |

## ④ Result figures

**No figures, by explicit reason:** this is a pure documentation/audit mission — no
new computation was performed, so there is no new field contour and no new time-series
curve to plot. Reusing an old case's figure as this mission's result would be
misrepresentation, so none is attached. The 3-D time-series / bypass generators
(`benchmarks/kleefsman_obstacle_3d/make_doc.py`, `production_report.py`) are already in
main and can reproduce the plots on demand.

## ⑤ Notes / limitations

1. **Duplicate mission (third reopening).** m24447 (like m24415) duplicates work
   already merged to main: the m21728 deliverables via **PR #826** and the round-2
   resolution via **PR #1131**. Re-landing the seven files would produce an **empty
   diff**; restoring the original 539-line `src` version would be a **regression** of
   the m22082/m22552/m22745 extensions. Neither is a valid completion, so this
   mission closes with an adjudication record.
2. **Correction to the round-2 note.** The round-2 note (m24415) was written while the
   local `origin/main` tracking ref was stale (at `1afd9d8f`), so it recorded the
   m24415 resolution as "stranded on `agent/m24415`, never merged." That was a
   **stale-ref artefact**: the m24415 resolution **was** merged to real GitHub main via
   **PR #1131** (`923b3019`, 2026-11-17 23:13 +0800). This note corrects that.
3. main carries **7 pre-existing kleefsman test failures** (listed in ③), all outside
   the seven m21728 files and failing identically on a clean main; this mission does
   not touch those paths.
4. Attribution of the 3-D FAIL (bypass / operator / grid) is follow-up case work, out
   of scope here.

## ⑥ Error judgement (explicit answer to "is it < 3 %?")

| key quantity | reference (Kleefsman Fig. 19) | best 3-D result (cj3887, dx=0.05) | rel. error | < 3 %? |
|---|---|---|---|---|
| H2 peak water height | 0.370 m | 0.1202 m | **+67.5 %** | **No** |
| H2 arrival (0.02 m) | 0.3591 s | 0.5028 s | **+40.0 %** | **No** |

**Explicit answer: No.** The 3-D blockage case **fails** the frozen < 10 % acceptance
on both metrics (and therefore also the < 3 % platform question). This is an honest
FAIL, not a completion claim: restoring the third dimension did not bring the peak
within tolerance, and the record-integrity gate (m22745) additionally flags the
recovered records' `height` operator as degenerate, so the slab operator is the
admissible one. No number above is hand-copied; all are read from
`benchmarks/kleefsman_obstacle_3d/results/recovered/verdict.json`.

## Decision

* The m21728 deliverables **are in main** (PR #826, 2026-11-15) and **pass** (26/26).
* The round-2 resolution **is in main** (PR #1131, 2026-11-17).
* The open item m21728 left (the production verdict) **has been decided: FAIL**,
  recorded in-repo by m22606/m22745.
* Reopened mission **m24447 is a duplicate** of work already merged via PR #826 and
  PR #1131. **No source change is made.** This note is the deliverable: it records,
  with commit-level evidence, that the m21728 deliverables and the round-2 resolution
  are both in main, green, and that the production verdict is `FAIL`. The mission's
  objective is already satisfied by main; the correct closure is to confirm that.
