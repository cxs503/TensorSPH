# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE — the two-point "decay order" defect and the per-level completion tool

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission m22485 ·
acceptance = owner 15 % gate on the Young-Laplace jump.

## ① What was wrong (defect, not a new number)

`docs/xval_projection_laplace_solenoid_ladder.md` draws a **negative** conclusion about the
anchor's open question (is the residual spurious current of the dynamic projection arm a
lattice artefact that refines away, or a formulation defect?). It rests on
`results_dynamic_ladder/diagnose_solenoid.json`, whose `levels` list holds **only**
`R/dx` = 8 and 12 — a **two-point** least-squares slope — while the page quotes it as a
"fitted **decay order**" of `-1.311` and sets `H1_artifact_hypothesis_supported = false`.
A two-point slope is not a fit, and the two points differ by a factor 1.5 in `R/dx` with the
liquid particle count crossing shell boundaries (`n_liquid` 232 → 520), so the sign of that
slope is not evidence that the mechanism fails to converge. The claim was **overstated
relative to its record**, not wrong in arithmetic.

`diagnose_csf_solenoid.py` does measure all four levels, but in one process that writes its
JSON only at the end: the shipped record is what survived that process being stopped. There
was no way to add the two missing levels without re-paying for the two already measured, and
no way to keep partial work when a long run dies.

## ② What this unit ships

| artifact | what |
|---|---|
| `benchmarks/laplace_sph/complete_solenoid_ladder.py` | runs **only the missing** levels, writes one record per level (`results_dynamic_ladder/solenoid_levels/Rdx<N>.json`), merges through `diagnose_csf_solenoid.summarise` (the authoritative gate logic), then regenerates the figure and the page |
| `tests/test_complete_solenoid_ladder.py` | 3 offline gates: an already-recorded level is **never** re-measured; a new level is persisted per level and merged; an empty level list only re-merges |
| queued jobs | `cj4193` = `--levels 16,24` (8 cores, `run_compute`), `cj4155`…`cj4160` = the 7-job dynamic marching ladder, all `queued` at hand-off |

No number on the page or in the JSON is computed by the new tool: it calls the same
`measure_level` / `summarise` / `make_figure` / `make_doc` functions as the original
diagnostic, so a gate cannot be relaxed by adding a code path.

## ③ Verification actually run (this unit)

* `pytest tests/test_complete_solenoid_ladder.py -q` → **3 passed** (synthetic records; no
  particle solve).
* Merge path replayed on the shipped record:
  `PYTHONPATH=src python benchmarks/laplace_sph/complete_solenoid_ladder.py --levels "" --threads 1 --no-doc`
  → `levels [8, 12] not measured [16, 24]`, gate verdicts identical to the shipped JSON, and
  `git diff` on `diagnose_solenoid.json` **empty** (byte-identical recompute — the merge path
  cannot silently move a number).
* Replayed gate verdicts (unchanged, honest-FAIL kept): `adjoint_exact` 1.7e-15 ✓,
  `reconstruction_exact` 7.1e-18 ✓, `solenoidal_decay_order_min` −1.311 ✗,
  `solenoidal_fraction_finest` 0.267 ✗, `consistency_with_march` 5.13 ✗.

## ④ What is **not** done (honest scope)

* `R/dx` = 16 and 24 are **still unmeasured**: `cj4193` was submitted with 38 jobs ahead of it
  and had not started when this unit closed, so the decay order is still a two-point slope.
  The page `xval_projection_laplace_solenoid_ladder.md` therefore still carries the
  under-evidenced sentence; **it is flagged here, not repaired.**
* The dynamic marching ladder is still **WITHHELD**: 6 of its 7 jobs (`cj4155`…`cj4160`) were
  `queued` for the whole unit (≥ 52 min, queue depth ~55 in flight) and never dispatched, so
  no new `dp` or spurious-current number exists. `docs/xval_projection_laplace_dynamic_ladder_debt.md`
  remains accurate as written and was not touched.
* No Young-Laplace `dp` value was produced by this unit, so the anchor's 15 % acceptance is
  **not** re-adjudicated here.

## ⑤ Error statement (`< 3 %`?)

The anchor's key quantity is `Δp` against `σ/R = 3.6000 Pa`. This unit produced **no** new
`Δp`; the only measured dynamic value in the tree remains `+1.17 %` at `R/dx` = 8
(`results_dynamic_ladder/dx8_nu01.json`), i.e. **< 3 % at that single, coarsest level** — which
is not a convergence statement and is not claimed as one. Question not answerable from this
unit's evidence: whether the ladder as a whole is < 3 %. Reason: compute never dispatched.

## ⑥ Reproduce

```bash
# extend the one-shot solenoidal diagnostic to the missing levels (queued: cj4193)
PYTHONPATH=src python benchmarks/laplace_sph/complete_solenoid_ladder.py --levels 16,24 --threads 8
# merge only (recompute the record/page from whatever levels exist)
PYTHONPATH=src python benchmarks/laplace_sph/complete_solenoid_ladder.py --levels "" --no-doc
# the tool's own gates
python -m pytest tests/test_complete_solenoid_ladder.py -q
```