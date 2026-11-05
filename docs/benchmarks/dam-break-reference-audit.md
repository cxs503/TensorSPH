# B2 dam-break: what the reference audit changed

**Generated** by `benchmarks/dam_break/make_audit_doc.py` from `benchmarks/dam_break/reference_audit.json` (2026-11-05). Do not hand-edit.

## 1. Summary

The dam-break case (B2) is the only place in TensorSPH where the solver is compared against an experiment, and that comparison is where the suite reports its worst numbers. The audit asked two prior questions: *is the validation target real*, and *is the analytic reference implemented correctly*. Both answers came back no.

| Question | Answer | Grade |
|---|---|---|
| Is the bundled Martin & Moyce table a digitisation? | No: exact arithmetic abscissa, exact 0.05 ordinate lattice at all 20 points | WITHHELD |
| Do two independent transcriptions of the same figure agree? | No: spread 0.851 in `Z/a` = 17.0x the claimed +-0.05 | reference uncertainty is a lower bound |
| Is the Ritter reference correct in the harness? | No: `Z/a = tau` was used instead of `Z/a = 2 tau` (factor 2 low) | fixed |
| Is the frame bookkeeping correct? | No: `front/ritter` mixed the lab frame with the self-similar frame | fixed |
| Can the solver be held to a reference-free number? | Yes: `c = d(Z/a)/d tau`, Ritter = 2 | REAL (numerical) |
| Can B2 carry a physical-validation verdict? | No: **withheld** | see section 5 |

## 2. Table provenance

* abscissa: `tau = 0.226 k`, max residual to that lattice **4.4e-16**;
* ordinates: every `Z/a` is an exact multiple of **0.05**, residual **7.1e-15**;
* second-difference RMS **0.0236** against the +-0.05 uncertainty the JSON claims for itself -- only ~2.1x below it, so the residual structure is the lattice, not noise;
* verdict: **synthetic-looking: abscissa is an exact arithmetic sequence and every ordinate is an exact multiple of 0.05, i.e. the table is generated rather than digitised**; grade **WITHHELD**, usable as a validation target: **False**.

## 3. Structure and kinematics of the two transcriptions

Both tables are read in one frame: `Z/a` is measured from the column edge, and the independent transcription bundles `X = x_front/a`, so `Z/a = X - 1`. The conversion puts the front at `Z/a = 0` when `tau = 0` in both tables (bundled +0.000, cross-check +0.000). No number here touches the solver; they are properties of the reference.

| property | bundled | cross-check |
|---|---|---|
| source | `benchmarks/dam_break/martin_moyce_1952.json` | `benchmarks/dam_break/martin_moyce_1952_crosscheck.json` |
| points | 20 | 14 |
| `tau` range | 0.000 - 4.294 | 0.000 - 2.960 |
| abscissa arithmetic | True | False |
| `tau` step spread | 8.9e-16 | 0.310 |
| ordinate quantum | 0.050 | 0.010 |
| slope levels | 4 | 12 |
| first-interval slope / peak | 0.800 | 0.146 |

The cross-check has the shape a gate-release experiment must have: the front leaves rest at 15% of its peak speed and accelerates by 6.8x. The bundled table is already at 80% of its peak in the first interval after release, i.e. it describes an impulsive start.

## 4. Cross-source spread = a lower bound on the uncertainty

Over the common window `tau` in [0.00, 2.938] (14 points):

| statistic | value |
|---|---|
| max abs difference in `Z/a` | 0.851 |
| rms difference | 0.556 |
| mean signed difference | 0.474 |
| claimed uncertainty (both tables) | +-0.050 |
| ratio max difference / claimed | 17.0x |

| `tau` | bundled `Z/a` | cross-check `Z/a` | difference |
|---|---|---|---|
| 0.000 | 0.000 | 0.000 | +0.000 |
| 0.226 | 0.200 | 0.061 | +0.139 |
| 0.452 | 0.450 | 0.122 | +0.328 |
| 0.678 | 0.700 | 0.185 | +0.515 |
| 0.904 | 0.950 | 0.268 | +0.682 |
| 1.130 | 1.200 | 0.404 | +0.796 |
| 1.356 | 1.450 | 0.599 | +0.851 |
| 1.582 | 1.650 | 0.837 | +0.813 |
| 1.808 | 1.850 | 1.096 | +0.754 |
| 2.034 | 2.050 | 1.394 | +0.656 |
| 2.260 | 2.200 | 1.670 | +0.530 |
| 2.486 | 2.350 | 1.983 | +0.367 |
| 2.712 | 2.500 | 2.308 | +0.192 |
| 2.938 | 2.650 | 2.634 | +0.016 |

Both tables come from the same paywalled original, so this spread is a *lower bound*: two honest readers of the same figure already disagree by up to 0.85 in `Z/a`, 17.0x the quoted error bar.

## 5. Gates on the bundled reference

| gate | metric | measured | limit | result |
|---|---|---|---|---|
| `bundled_is_pointwise` | arithmetic tau grid AND coarse ordinate quantum AND <= 6 slope levels | tau_grid_is_arithmetic=True, tau_step_spread=8.881784197001252e-16, ordinate_quantum=0.05, n_slope_levels=4 | tau steps must vary (figure pixels), quantum <= 0.01, > 6 slope levels | **fail** |
| `acceleration_phase_present` | slope at the first interval / peak slope | 0.8000 | <= 0.5 (a front released from rest starts well below its later speed) | **fail** |
| `claimed_uncertainty_consistent` | max |bundled - crosscheck| / claimed uncertainty | 17.0183 | <= 3.0 | **fail** |

**Verdict: `physical_validation = withheld`.** The reference cannot separate a solver error from a digitisation error of the same size (0.5-0.9 in Z/a), so no pass/fail verdict on physical validation of the B2 case can be issued against it. The earlier 'physical validation fails' statement over-attributes to the solver what the reference does not support.

## 6. Analytic reference

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

## 7. Reference-free measurement

With the table unusable, the case is held to a dimensionless, reference-free number instead: `c = d(Z/a)/d tau`, fitted over `tau` in [0.5, 4.0] to skip the startup transient. Ritter gives `c = 2` and the bundled table gives `c = 0.7348`.

Solver configuration (single source: `DamBreakConfig`): `c0` = 25 m/s, max Courant 0.3, simulated 0.5 s, adaptive `dt` False, density diffusion 0, viscosity 0.01, `h` factor 1.3.

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

1. At free slip (`wall_friction = 0`) the front runs at **60%-48% of Ritter** (`c = 1.2016` at dx = 0.04, `c = 1.2468` at dx = 0.02). Ritter over-predicts because it assumes a semi-infinite shallow reservoir; the collapse here is deep (`H/a = 2`) and finite.
2. Imposing a no-slip floor lowers the coefficient monotonically (`c = 1.2016` -> `0.9682` at dx = 0.04), i.e. the floor boundary condition carries ~19% of the front speed. The effect *grows* with resolution (`c = 0.9682` -> `0.8959` at dx = 0.02): the no-slip blend is not yet resolution-converged, so no friction calibration should be published from dx = 0.04 m.
3. The bundled table's `c = 0.7348` is **below** the no-slip run at both resolutions, so bed friction alone cannot explain the residual -- but since the table is unverified, the residual is not evidence against the solver either. Grade the physics claim WITHHELD, not FAIL.

## 8. The same simulations against both references

Setup (identical for every rung and for both references): `a` = 0.2 m, column height = 0.4 m (2:1), tank = 2 m, `c0` = 25 m/s, wall friction 0, density diffusion 0, simulated time 0.5 s.

| `dx` (m) | particles | steps | wall clock (s) | MAE vs bundled | RMSE vs bundled | MAE vs cross-check | RMSE vs cross-check |
|---|---|---|---|---|---|---|---|
| 0.040 | 50 | 801 | 14.2 | 0.5988 | 0.6766 | 0.1302 | 0.1373 |
| 0.020 | 200 | 1603 | 112.0 | 0.5774 | 0.6817 | 0.0438 | 0.0541 |
| 0.010 | 800 | 3205 | 553.5 | 0.5728 | 0.6994 | 0.1163 | 0.1234 |

Gates (limit on the mean absolute error in `Z/a` = 0.15):

| reference | finest `dx` | MAE | result | all rungs pass |
|---|---|---|---|---|
| bundled | 0.010 | 0.5728 | **fail** | False |
| crosscheck | 0.010 | 0.1163 | **pass** | True |

Observed convergence order of the error (slope of log MAE against log `dx`, so a value near 0 means the residual does not respond to refinement):

| reference | order from MAE | order from RMSE |
|---|---|---|
| bundled | +0.052, +0.012 | -0.011, -0.037 |
| crosscheck | +1.573, -1.410 | +1.345, -1.191 |

### The cross-check pass is not a convergence proof

The cross-check error is *not* monotone in `dx`: it reaches 0.0438 at `dx` = 0.020 m and then rises to 0.1302 at `dx` = 0.040 m, so the observed order above changes sign. Every rung stays inside the 0.15 limit, but the honest reading is "within the reference uncertainty at all three rungs", not "converging to the experiment". A 50-800 particle 2-D run cannot settle which of the two tables is closer to the 1952 apparatus; it can only show that the misfit against the bundled table is the same size as the disagreement between the two transcriptions.

| reference | points | claimed uncertainty | path |
|---|---|---|---|
| bundled | 20 | +-0.05 | `benchmarks/dam_break/martin_moyce_1952.json` |
| crosscheck | 14 | +-0.05 | `benchmarks/dam_break/martin_moyce_1952_crosscheck.json` |

## 9. What this changes and what it does not

1. **The B2 status changes from `fails` to `withheld`.** The old verdict read a 5.51 `Z/a` misfit as a solver error. Against a reference whose two transcriptions differ by up to 0.85, that attribution is not supported. This is a correction of the *verdict*, not of the numbers: the solver's misfit against the bundled table is unchanged.

2. **The bundled table stays in the repository and stays the default for the B2 gate**, so the historical series is comparable; the audit only records that its verdicts cannot be read as validation of the solver.
3. **The two remaining candidates are untouched here**: the 2-D cross-section versus the 3-D experiment (side-wall drag), and the front definition (`x_max` is the leading particle, which at `dx` = 0.04 m is a one-particle-thick sheet above the floor). Neither can be settled from the tables alone; a 3-D run and a bulk-front estimator are the next increments.
4. **What would upgrade this audit**: access to the archived 1952 figure at its native resolution, or experimental data from an open dataset with per-point error bars. Until then any B2 validation claim should quote the reference spread alongside the solver error.

## 10. Limits of this audit

* The friction sweep is at `dx = 0.04 m` (50 particles) with only the two end points repeated at `dx = 0.02 m` (200 particles); the load on the host blocked a full 5 x 2 sweep. Nothing here claims resolution convergence for `wall_friction > 0`.
* `accounting for the reference`: no citable digitisation of Martin & Moyce (1952) is installed, and the two tables in the repository are independent *transcriptions*, not two measurements. Their spread bounds the reading error from below only.
* The three-step path back to a real physical claim: (a) digitise Figure 4 of the cited paper (or an open-access reproduction) with a scatter-bearing table and cite the figure number; (b) check the `H/a = 2` geometry against the paper's set-up; (c) then, and only then, quote an error against the 15 %% platform bar.

## 11. Artifacts

| File | What |
|---|---|
| `benchmarks/dam_break/reference_audit.json` | all numbers above (both layers) |
| `benchmarks/dam_break/reference_audit.py` | regenerates the JSON, both layers and the figures |
| `benchmarks/dam_break/reference_ladder.json` | the resolution ladder against both references |
| `benchmarks/dam_break/reference_ladder.py` | regenerates the ladder |
| `benchmarks/dam_break/make_audit_doc.py` | renders this document |
| `benchmarks/dam_break/dam_break_references.png` | trajectories vs both references |
| `benchmarks/dam_break/dam_break_table_integrity.png` | lattice / quantum residuals |
| `benchmarks/dam_break/dam_break_friction_sweep.png` | `c` vs floor friction |
| `benchmarks/dam_break/reference_ladder.png` | front trajectory ladder vs both references |
| `tests/test_ritter_reference.py` | analytic verification of the reference |
| `tests/test_dam_break_reference_audit.py` | guards on every finding above, both layers |

## 12. Reproduce

```bash
python benchmarks/dam_break/reference_audit.py --skip-runs   # layers A+B, no simulation
python benchmarks/dam_break/reference_audit.py \
    --sizes 0.04 --frictions 0,0.25,0.5,0.75,1 --plot          # + sweep
python benchmarks/dam_break/reference_ladder.py --sizes 0.04,0.02,0.01 \
    --seconds 0.5 --json-out benchmarks/dam_break/reference_ladder.json --plot
python benchmarks/dam_break/make_audit_doc.py
pytest tests/test_dam_break_reference_audit.py -q
```

