# B2 dam-break: is the experimental reference able to carry a verdict?

**Generated file** -- every number below is injected from `benchmarks/dam_break/reference_crosscheck.json`, `benchmarks/dam_break/reference_ladder.json` and `benchmarks/dam_break/reference_audit.json` by `benchmarks/dam_break/make_reference_audit_doc.py`. Do not edit by hand.

## Why this audit exists

`docs/benchmarks/dam-break-resolution-ladder.md` measured the Martin & Moyce error of the B2 dam-break case on a refinement ladder and found it *resolution independent* (mean absolute error 0.5988 / 0.5774 / 0.5728 in `Z/a` for `dx` = 0.04 / 0.02 / 0.01 m). An error that survives refinement is either a structural reference problem or a structural solver problem. The ladder could not separate the two, and the case was reported as `physical validation fails`.

This audit settles the reference half of that question by measurement, with no simulation involved: it compares the bundled table against a second, independent transcription of the same published figure. If those two disagree by more than the gate limit, then the gate cannot arbitrate anything and the verdict must change from `fail` to `withheld`.

## The two references

| table | file | points | claimed uncertainty | role |
|---|---|---|---|---|
| bundled | `benchmarks/dam_break/martin_moyce_1952.json` | 20 | +-0.05 | the table the B2 gate has used |
| cross-check | `benchmarks/dam_break/martin_moyce_1952_crosscheck.json` | 14 | +-0.05 | independent transcription, cross-check only, never a gate |

Neither table has been verified against the paywalled 1952 original; both are transcriptions of the figure-level data that circulates in the open literature. Their spread is therefore a **lower bound** on the reference uncertainty.

## Gate 1 -- is the bundled table a point-by-point digitisation?

| property | bundled | cross-check | what a figure reading must look like |
|---|---|---|---|
| points | 20 | 14 | -- |
| abscissa step | 0.2260 (spread 8.9e-16) | -- | irregular, figure pixels |
| abscissa grid arithmetic | True | False | no |
| ordinate quantum | 0.050 | 0.010 | finer than the claimed uncertainty |
| distinct finite-difference slopes | 4 | 12 | many |

**Result: `FAIL`** (limit: fail if the tau steps are uniform AND the ordinate quantum is > 0.01 AND the slope levels are <= 6). The bundled table is a piecewise-linear sketch: every tau step is 0.226 (spread 8.9e-16), every ordinate is a multiple of 0.05, and the finite-difference slope takes only 4 discrete values. A point-by-point reading off a figure does not produce that.

## Gate 2 -- does the reference contain the acceleration phase?

| table | slope at first interval / peak slope |
|---|---|
| bundled | 0.800 |
| cross-check | 0.146 |

**Result: `FAIL`** (limit: first-interval slope must be <= 0.5 x the peak slope; a larger ratio means an impulsive start). The bundled table already reaches 80 % of its peak front speed within the first tau interval after release, i.e. it describes an impulsive start. The independent transcription reaches 15 % there and then grows to a steady front speed, which is the shape a gate-release experiment has.

## Gate 3 -- is the claimed uncertainty consistent with the cross-source spread?

| quantity | value |
|---|---|
| overlap in `tau` | 0.00 - 2.96 |
| samples on a 0.01 grid | 297 |
| max abs difference in `Z/a` | **0.8506** at `tau` = 1.36 |
| rms difference | 0.5728 |
| mean difference | 0.5062 |
| ratio to the claimed +-0.05 | **17.0x** |

**Result: `FAIL`** (limit: <= 3.0). Two independent transcriptions of the same figure differ by up to 0.851 in Z/a (rms 0.573) at tau = 1.36, 17.0x the claimed +-0.05. Both derive from the same paywalled original, so this spread is a lower bound on the reference uncertainty.

## The same simulations, scored against each reference

Each row re-introduces nothing but the reference: `dx` = 0.04, 0.02 m, `c0` = 25.0 m/s, Courant 0.3, `wall_friction` = 0.0, `density_diffusion` = 0.0, t = 0.5 s, column 0.2 x 0.4 m in a 2.0 m tank.

| dx (m) | particles | wall clock (s) | MAE vs bundled | MAE vs cross-check | MAE vs Ritter |
|---|---|---|---|---|---|
| 0.040 | 50 | 9.2 | 0.5988 | 0.1302 | 2.1516 |
| 0.020 | 200 | 97.7 | 0.5770 | 0.0438 | 2.0096 |

| gate reading | value |
|---|---|
| gate limit on MAE in `Z/a` | 0.15 |
| cross-source spread | 0.8506 = **5.7x** the gate limit |
| MAE shift caused by changing only the reference (finest dx) | 0.5332 |
| gate result vs bundled / vs cross-check | FAIL / PASS |

The two references differ by more than the gate limit itself, so a gate reading 'MAE > 0.15' cannot be attributed to the solver: changing only the reference moves the MAE by 0.533, a shift of the same order as the failure margin. The B2 physical-validation verdict therefore stays withheld until a scatter-bearing, citable reference is installed.

## What the companion audit adds

`reference_audit.py` re-derives the analytic reference and pins the dimensionless front-speed coefficient. The Ritter identity `Z/a = 2 tau` holds to a relative residual of 0.0e+00 and the shipped helper matches to 0.0e+00. The bundled table itself has `c = d(Z/a)/d tau` = 0.7348 (36.7 % of Ritter's 2), so it encodes a front that is far slower than any shallow-water limit -- one more sign that it is not a reading of the published points.

Table-integrity grade recorded there: `WITHHELD`.

## Verdict

`physical_validation` for the B2 case: **`withheld`**. The reference cannot separate a solver error from a digitisation error of the same size (0.85 in Z/a here), so no pass/fail verdict on physical validation of the B2 case can be issued against it. A statement that physical validation *fails* over-attributes to the solver what the reference does not support.

What this does *not* say: that the solver is accurate. The SPH front is still far from the analytic Ritter limit (`MAE` of order 2 in `Z/a`), which is expected for a finite reservoir at these resolutions and is not a validation statement either way. The point is narrower and checkable: the experimental table in the repository cannot support a pass/fail verdict against any solver, so the honest verdict is `withheld`.

What would change it: a scatter-bearing, citable digitisation of the 1952 figure (or the paper's own tabulated values). Re-run `reference_crosscheck.py`; gate 1 and gate 3 would then pass and the verdict would flip to `evaluable` automatically. The tests pin the current state in *both* directions, so the flip cannot happen silently.

## Reproduction

```
python benchmarks/dam_break/reference_crosscheck.py --plot
python benchmarks/dam_break/reference_ladder.py --sizes 0.04,0.02 --plot
python benchmarks/dam_break/make_reference_audit_doc.py
python -m pytest tests/test_dam_break_reference_crosscheck.py -q
```

| file | role |
|---|---|
| `benchmarks/dam_break/reference_crosscheck.py` | structural audit + cross-source spread, no solver |
| `benchmarks/dam_break/reference_crosscheck.json` | its output |
| `benchmarks/dam_break/reference_ladder.py` | same simulations vs three references |
| `benchmarks/dam_break/reference_ladder.json` | its output |
| `benchmarks/dam_break/martin_moyce_1952_crosscheck.json` | the independent transcription |
| `benchmarks/dam_break/reference_audit.py` | companion audit (table integrity, Ritter identity, friction sweep) |
| `tests/test_dam_break_reference_crosscheck.py` | guards on every finding above |
