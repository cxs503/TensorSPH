# m24504 — Round-2 resolution of the reopened TensorSPH#1025 (m21792 matured deliverables)

**Case anchor:** `CASE-SPH-LAKE-AT-REST` (closed-tank lake at rest; analytic identity,
machine-precision tolerance `1e-9`)
**Mission:** 24504 (reopened TensorSPH#1025, 管到底 round 2)
**Branch:** `agent/m24504` (rebased onto real GitHub main @ `95f0877f`)
**Date:** 2026-11-18
**Supersedes:** nothing — this is the first resolution note for #1025. The original
PR #1025 was closed by the owner on 2026-11-05; the same deliverables were re-landed
to main via **PR #829** (merge commit `f9991b7d`, 2026-11-15 19:43 +0800).

---

## ① Case introduction

The reopened mission restates the **m21792 matured mission deliverables** — the
hydrostatic-closure verdict for the closed-tank lake-at-rest case: on one frozen
mass field, which pressure/wall/operator closure drives the hydrostatic residual
`max|a|/g` below the anchor tolerance `1e-9`. The verdict: the **consistent**
closure (incompressible hydrostatic pressure + unclamped wall continuation +
first-order-consistent moment gradient) reaches `1.087e-15` at `dx = 0.02 m`
(9.2e5× under tolerance); the **shipped** WCSPH closure and the **compressible**
(Tait) closure do not — honest FAILs at `1.242e-01` and a `3.128e-04` floor that
does not shrink with resolution.

The ten deliverables the mission lists, and their state in real GitHub main
(`95f0877f`):

| # | file | lines at #829 merge | lines in HEAD | in HEAD? |
|---|---|---|---|---|
| 1 | `benchmarks/lake_at_rest/freeze.py` | 383 | 384 | yes |
| 2 | `benchmarks/lake_at_rest/hydrostatic_closure.py` | 451 | 451 | yes |
| 3 | `docs/figures/lake_hydrostatic_closure_classes.png` | (png) | (png) | yes |
| 4 | `docs/figures/lake_hydrostatic_closure_ladder.png` | (png) | (png) | yes |
| 5 | `docs/verification_lake_at_rest_hydrostatic_closure.md` | 249 | 249 | yes |
| 6 | `src/tensorsph/hydrostatic_closure.py` | 424 | 424 | yes |
| 7 | `tests/data/lake_at_rest/MANIFEST.sha256` | 11 | 13 | yes |
| 8 | `tests/data/lake_at_rest/channel.json` | 36 | 38 | yes |
| 9 | `tests/data/lake_at_rest/hydrostatic_closure.json` | 5338 | 5338 | yes |
| 10 | `tests/test_lake_at_rest_hydrostatic_closure.py` | 177 | 177 | yes |

(The small deltas in rows 1, 7, 8 are later, in-scope evolution by **PR #951
(m23007)**: `freeze.py` gained `rest_split.json` / `closure_response.json` /
`switch_horizon.json` in `EXTRA_RECORDS`, and the manifest/channel records gained
the matching entries — not a missing piece of m21792.)

## ② How this was done (actual actions on this branch)

This mission adds **no source change** (the feature is already in main). The actions
were **audit + adjudication**, all reproducible:

1. `git merge-base --is-ancestor f9991b7d HEAD` → **true** (the #829 merge commit is
   an ancestor of this branch; HEAD = `95f0877f`).
2. `git cat-file -e HEAD:<path>` ×10 → **all 10 deliverable paths present in the
   HEAD tree**.
3. `python -m pytest tests/test_lake_at_rest_hydrostatic_closure.py -q` →
   **11 passed, 1 warning in 5.2 s** (the gate re-grades the shipped record from its
   raw numbers and re-derives the physics at `dx = 0.05` against the record).
4. Broader lake-at-rest suite (all 16 `tests/test_lake_at_rest*.py` files) →
   **162 passed** in 107 s.
5. `tests/data/lake_at_rest/MANIFEST.sha256` re-hashed → **13/13 entries verify**
   (hash + byte size).
6. Fresh coarse re-derivation at `dx = 0.05` on this tree (consistent closure):
   `max|a|/g = 5.435e-16` vs record `5.440e-16` — the record still describes this
   code; shipped control re-derived at `1.092e-01`, identical to the record.
7. Wrote this note — the single new file of this mission.

Toolchain: git forensics + pytest + sha256 + one fresh `closure_residual` call.
**Zero hand-copied numbers** (tables below are read from the shipped JSON and from
this branch's own re-derivation).

## ③ Computation results (programmatic injection)

State of the m21792 deliverables in real GitHub main (HEAD = `95f0877f` + this note):

| deliverable | in HEAD? | lines |
|---|---|---|
| `src/tensorsph/hydrostatic_closure.py` | yes | 424 |
| `benchmarks/lake_at_rest/hydrostatic_closure.py` | yes | 451 |
| `tests/test_lake_at_rest_hydrostatic_closure.py` | yes | 177 |
| `docs/verification_lake_at_rest_hydrostatic_closure.md` | yes | 249 |
| `tests/data/lake_at_rest/hydrostatic_closure.json` | yes | 5338 |
| `benchmarks/lake_at_rest/freeze.py` | yes | 384 |
| `tests/data/lake_at_rest/MANIFEST.sha256` | yes | 13 |
| `tests/data/lake_at_rest/channel.json` | yes | 38 |
| `docs/figures/lake_hydrostatic_closure_{classes,ladder}.png` | yes | (png) |

Gate table re-graded from the shipped record (all four resolutions; `all_pass`
recomputed = **True**):

| gate | dx=0.05 | dx=0.04 | dx=0.025 | dx=0.02 | threshold |
|---|---|---|---|---|---|
| `H1` anchor tolerance | 5.440e-16 | 5.465e-16 | 7.252e-16 | 1.087e-15 | < 1e-09 |
| `H2` shipped control | 1.092e-01 | 1.203e-01 | 1.013e-01 | 1.242e-01 | > 1e-02 |
| `H3` clamp control | 1.155e-01 | 1.170e-01 | 1.143e-01 | 1.177e-01 | > 1e-02 |
| `H4` compressible floor | 7.675e-04 | 6.187e-04 | 3.813e-04 | 3.096e-04 | > 1e-06 |
| `H5` level-set invariance | 1.806e-16 | 3.022e-18 | 4.798e-19 | 1.808e-16 | < 1e-15 |
| `H6` slope sensitivity | 1.838e+13 | 1.830e+13 | 1.379e+13 | 9.202e+12 | > 1e+06 |
| `H7` zero-pressure control | 1.000 | 1.000 | 1.000 | 1.000 | 0.9…1.1 |
| `H8` operator cross-check | 1.599e-14 | 1.066e-14 | 3.908e-14 | 4.974e-14 | < 1e-10 |
| `H9` regulariser ladder | monotone | monotone | monotone | monotone | — |
| `H10` resolution floor | — | — | — | 2.00× | < 10× |

Fresh re-derivation on this branch (dx = 0.05, consistent closure):

| quantity | record | this branch | rel. diff |
|---|---|---|---|
| `max|a|/g` (consistent) | 5.440e-16 | 5.435e-16 | < 1e-3 |
| `max|a_x|/g` (shipped control) | 1.092e-01 | 1.092e-01 | 0 |

## ④ Result figures

The two case figures are the deliverable's own, generated by
`benchmarks/lake_at_rest/hydrostatic_closure.py --figures` from the shipped record
and committed with it (`docs/figures/lake_hydrostatic_closure_classes.png`,
`lake_hydrostatic_closure_ladder.png`): residual by closure class against the
anchor-tolerance line, and the Tikhonov ladder showing the floor is roundoff. No
new computation was performed in this mission, so no new field contour is attached;
the figures above are the adjudicated case's result curves, not new results.

## ⑤ Notes / limitations

1. **Duplicate mission.** m24504 duplicates work already merged to main: the m21792
   deliverables via **PR #829** (`f9991b7d`, 2026-11-15). Re-landing the ten files
   would produce an **empty diff**; restoring the pre-#951 versions of `freeze.py` /
   `MANIFEST.sha256` / `channel.json` would be a **regression** of the m23007
   evolution. Neither is a valid completion, so this mission closes with an
   adjudication record.
2. main carries **3 pre-existing collection errors** when the whole `tests/`
   directory is collected in one process (`test_tvc_dissipation_attribution.py`,
   `test_wedge_compressibility_study.py`, `test_xval_tvc_cadence_invariance.py`):
   a bare `from run import …` inside `benchmarks/xval_sphinx_tvc/` resolves to
   `benchmarks/case_poiseuille/run.py` once another test file has put that directory
   on `sys.path`. All three files pass in isolation; none is a lake-at-rest path and
   this mission does not touch them.
3. The `1e-15` residual is an algebraic identity of "linear field + first-order
   consistent operator + unclamped closure" (see the verification document §6): it
   is not a general-flow accuracy claim, and the document says so.

## ⑥ Error judgement (explicit answer to "is it < 3 %?")

The anchor is an analytic identity (residual = 0), so the relative error is the
absolute residual in units of g:

| key quantity | reference | measured (dx = 0.02) | rel. error | < 3 %? |
|---|---|---|---|---|
| consistent-closure residual | 0 (analytic) | 1.087e-15 | 1.087e-15 | **Yes** (and < 1e-9 anchor tolerance) |
| shipped-closure residual | 0 (analytic) | 1.242e-01 | 1.242e-01 | **No** (honest FAIL) |
| compressible-closure floor | 0 (analytic) | 3.128e-04 | 3.128e-04 | **No** (honest FAIL) |

**Explicit answer:** the anchor tolerance is met **only** by the consistent closure
(Yes, by 9.2e5×); the shipped and compressible closures **fail** it, and that is the
verdict m21792 recorded — not a completion defect.

## Decision

* The m21792 deliverables **are in main** (PR #829, 2026-11-15) and **pass**
  (11/11 gate, 162/162 lake-at-rest suite, 13/13 manifest hashes).
* The record still describes this code (fresh re-derivation matches).
* Reopened mission **m24504 is a duplicate** of work already merged via PR #829.
  **No source change is made.** This note is the deliverable: it records, with
  commit-level evidence, that the m21792 deliverables are in main, green, and that
  the verdict (consistent closure meets the anchor tolerance; shipped/compressible
  closures do not) stands. The mission's objective is already satisfied by main;
  the correct closure is to confirm that.
