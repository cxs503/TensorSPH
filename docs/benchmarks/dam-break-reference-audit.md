# B2 dam-break: what the reference audit changed

**Generated** by `benchmarks/dam_break/make_audit_doc.py` from `benchmarks/dam_break/reference_audit.json` (2026-11-03). Do not hand-edit.

## 1. Summary

The dam-break case (B2) is the only place in TensorSPH where the solver is compared against an experiment, and that comparison is where the suite reports its worst numbers. The audit asked two prior questions: *is the validation target real*, and *is the analytic reference implemented correctly*. Both answers came back no.

| Question | Answer | Grade |
|---|---|---|
| Is the bundled Martin & Moyce table a digitisation? | No: exact arithmetic abscissa, exact 0.05 ordinate lattice at all 19 points | WITHHELD |
| Is the Ritter reference correct in the harness? | No: `Z/a = tau` was used instead of `Z/a = 2 tau` (factor 2 low) | fixed |
| Is the frame bookkeeping correct? | No: `front/ritter` mixed the lab frame with the self-similar frame | fixed |
| Can the solver be held to a reference-free number? | Yes: `c = d(Z/a)/d tau`, Ritter = 2 | REAL (numerical) |

## 2. Table provenance

* abscissa: `tau = 0.226 k`, max residual to that lattice **4.4e-16**;
* ordinates: every `Z/a` is an exact multiple of **0.05**, residual **7.1e-15**;
* second-difference RMS **0.0236** against the ±0.05 uncertainty the JSON claims for itself -- only ~2.1x below it, so the residual *magnitude* is not the argument. The argument is the exactness: a digitisation has scatter of the order of the figure resolution and never lands on an exact 0.05 lattice at every one of 19 points.

Whatever generated these 20 points, it was not a digitising tablet. The table is therefore *not* usable as a validation target; `physical_validation` for B2 is **WITHHELD**, not fail, because there is nothing trustworthy to fail against. The values are plausible in magnitude, so the table is kept as a plot reference and flagged in `martin_moyce_1952.json` itself.

## 3. Analytic reference

Ritter's dry-bed solution for a reservoir of depth `H`, `c = sqrt(gH)`:

```
h(x,t) = (2c - x/t)^2 / (9g)   on  -ct <= x <= 2ct
u(x,t) = (2/3)(c + x/t)
x_front(t) = 2ct
```

The measurements in `tests/test_ritter_reference.py` (all at machine precision):

* `h(-ct) = H`, `h(0) = 4H/9`, `h(2ct) = 0`;
* Riemann invariant `u + 2 sqrt(gh) = 2c` identically across the fan;
* self-similarity `h(Lx, Lt) = h(x,t)` to 1e-13;
* water ahead of the dam integrates to exactly `(4/27) H x_front`.

Non-dimensionalised with `tau = t sqrt(2g/a)` and `H = 2a` the front is **`Z/a = 2 tau`** (residual 0.0e+00). The harness used `Z/a = tau`, so every reported Ritter error in `experimental_validation.py` was computed against a reference half the right size. `run.py` had a second, independent frame error: it divided the lab-frame front `x_max` by the self-similar `Z`, which is off by the column width in every sample.

## 4. Reference-free measurement

With the table unusable, the case is held to a dimensionless, reference-free number instead: `c = d(Z/a)/d tau`, fitted over `tau` in [0.5, 4.0] to skip the startup transient. Ritter gives `c = 2` and the bundled table gives `c = 0.7348`.

| dx (m) | particles | wall_friction | steps | c | c/2 | max compression | Mach | escaped |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 50 | 0 | 801 | 1.2016 | 60.1% | 0.023 | 0.132 | 0 |
| 0.04 | 50 | 0.25 | 801 | 1.1099 | 55.5% | 0.032 | 0.110 | 0 |
| 0.04 | 50 | 0.5 | 801 | 1.0580 | 52.9% | 0.022 | 0.104 | 0 |
| 0.04 | 50 | 0.75 | 801 | 1.0123 | 50.6% | 0.028 | 0.098 | 0 |
| 0.04 | 50 | 1 | 801 | 0.9682 | 48.4% | 0.026 | 0.094 | 0 |
| 0.02 | 200 | 0 | 1603 | 1.2468 | 62.3% | 0.027 | 0.133 | 0 |
| 0.02 | 200 | 1 | 1603 | 0.8959 | 44.8% | 0.018 | 0.094 | 0 |

Findings:

1. At free slip (`wall_friction = 0`) the front runs at **60%–48% of Ritter** (`c = 1.2016` at dx = 0.04, `c = 1.2468` at dx = 0.02). Ritter over-predicts because it assumes a semi-infinite shallow reservoir; the collapse here is deep (`H/a = 2`) and finite.
2. Imposing a no-slip floor lowers the coefficient monotonically (`c = 1.2016` -> `0.9682` at dx = 0.04), i.e. the floor boundary condition carries ~19% of the front speed. The effect *grows* with resolution (`c = 0.9682` -> `0.8959` at dx = 0.02): the no-slip blend is not yet resolution-converged, so no friction calibration should be published from dx = 0.04 m.
3. The bundled table's `c = 0.7348` is **below** the no-slip run at both resolutions, so bed friction alone cannot explain the residual -- but since the table is unverified, the residual is not evidence against the solver either. Grade the physics claim WITHHELD, not FAIL.

## 5. Limits of this audit

* The friction sweep is at `dx = 0.04 m` (50 particles) with only the two end points repeated at `dx = 0.02 m` (200 particles); the load on the host blocked a full 5 x 2 sweep. Nothing here claims resolution convergence for `wall_friction > 0`.
* No citable digitisation of Martin & Moyce (1952) was installed. Until one is, `physical_validation` for B2 must stay `not_evaluated`, and the `--max-exp-error` gate in `experimental_validation.py` must not be quoted as a pass/fail verdict.
* The three-step path back to a real physical claim: (a) digitise Figure 4 of the cited paper (or an open-access reproduction) with a scatter-bearing table and cite the figure number; (b) check the `H/a = 2` geometry against the paper's set-up; (c) then, and only then, quote an error against the 15 %% platform bar.

## 6. Artifacts

| File | What |
|---|---|
| `benchmarks/dam_break/reference_audit.json` | all numbers above |
| `benchmarks/dam_break/reference_audit.py` | regenerates the JSON and figures |
| `benchmarks/dam_break/make_audit_doc.py` | renders this document |
| `benchmarks/dam_break/dam_break_references.png` | trajectories vs both references |
| `benchmarks/dam_break/dam_break_table_integrity.png` | lattice / quantum residuals |
| `benchmarks/dam_break/dam_break_friction_sweep.png` | `c` vs floor friction |
| `tests/test_ritter_reference.py` | analytic verification of the reference |
| `tests/test_dam_break_reference_audit.py` | guards on the findings above |

