# Benchmark: root cause of the B2 experimental residual (dam break)

**Generated file.** Every number below is injected from
`outputs/dam_break_exp_bias/bias_diagnosis.json` by `benchmarks/dam_break/make_bias_doc.py`.
Do not edit by hand; re-run the two scripts instead.

## 1. Why this study exists

`benchmarks/dam_break/resolution_ladder.json` measured the Martin & Moyce (1952)
comparison on a three-rung ladder and found the mean absolute error in `Z/a`
**flat** across `dx` = 0.04 / 0.02 / 0.01 m while the RMSE *grew*. A residual
that does not respond to refinement is not numerical diffusion of the front, so
refining further could never close the gate. This study decomposes that residual
into measurable components instead of assuming a cause.

The three candidates tested here are the ones the benchmark document listed as
open work: startup lag of the released column, the settled front *speed*, and the
wall treatment (implementation-sequence item 6 of `docs/algorithm-gap-analysis.md`,
"decide the default `wall_friction` from that comparison"). A fourth question is
asked of the reference table itself, because a gate is only as good as its
reference.

## 2. Method

| item | setting |
|---|---|
| case | 2:1 dam break, column 0.2 × 0.4 m in a 2.0 × 0.75 m tank, dry bed |
| resolutions | `dx` = 0.02 m, `dx` = 0.04 m |
| wall treatment | `wall_friction` = 0 (free slip), `wall_friction` = 1 (no slip) |
| fluid | `rho0` = 1000 kg/m³, `c0` = 25 m/s, `gamma` = 7, acoustic Courant 0.3 |
| window | 0.5 s (~ 52 samples at 10 ms), `tau` up to ~4.95 |
| reference | `benchmarks/dam_break/martin_moyce_1952.json` (20 points) |
| tool | `benchmarks/dam_break/exp_bias_diagnosis.py` |

Diagnostics computed from each trajectory:

1. **startup lag** `Δtau` — first time the front has advanced 0.1 in `Z/a` past
   its initial value (the initial value is not 0: the leading particle centre
   sits half a cell inside the column, so `Z/a = -0.5 dx/a`);
2. **late-time slope** d(Z/a)/d(tau), ordinary least squares over `tau >= 2`;
3. **reference MAE**, raw and after removing the best-fit startup lag (grid of
   0.02 in `tau`), so the lag and the slope error are not confounded;
4. **provenance audit** of the bundled table: abscissa uniformity, ordinate
   quantisation, second-difference scatter versus the stated uncertainty.

## 3. Results

### 3.1 Front speed is the dominant defect, not resolution

| `dx` (m) | `wall_friction` | particles | startup lag `Δtau` | late slope | slope / reference | slope / Ritter |
|---|---|---|---|---|---|---|
| 0.02 | 0 | 200 | 0.49 | 1.526 | 2.92 | 0.76 |
| 0.02 | 1 | 200 | 0.59 | 0.878 | 1.68 | 0.44 |
| 0.04 | 0 | 50 | 0.59 | 1.471 | 2.81 | 0.74 |
| 0.04 | 1 | 50 | 0.69 | 1.001 | 1.91 | 0.50 |
| — | reference table | 20 | — | 0.523 | 1.00 | 0.26 |
| — | Ritter (2:1 column, dry bed) | — | — | 2.000 | — | 1.00 |

### 3.2 The gate metric hides two opposite-sign errors

| `dx` (m) | `wall_friction` | MAE raw | best `Δtau` | MAE after shift | gate limit | passes raw | passes shifted |
|---|---|---|---|---|---|---|---|
| 0.02 | 0 | 0.577 | 0.12 | 0.572 | 0.15 | no | no |
| 0.02 | 1 | 0.565 | 0.90 | 0.158 | 0.15 | no | no |
| 0.04 | 0 | 0.599 | 0.66 | 0.555 | 0.15 | no | no |
| 0.04 | 1 | 0.595 | 0.90 | 0.227 | 0.15 | no | no |

### 3.3 Corrected Ritter anchor

The harness compared the simulation against `Z/a = tau`, which corresponds to a
reservoir half as deep as the one that is actually simulated. For a 2:1 column on
a dry bed the Ritter leading edge is `x_f = 2 sqrt(g H) t`, and measuring `Z` from
the initial column edge gives

```
Z/a = (2 sqrt(g*2a) t - a)/a = 2 tau - 1.
```

`experimental_validation.py` now derives this from `tensorsph.dam_break.ritter_front`
through the new `ritter_za()` helper, and `benchmarks/dam_break/ritter_limit.py`
uses the same expression. Mean relative deviation of the simulation from the
corrected Ritter curve, over the part of the trajectory where `Z/a_Ritter > 0.2`:

| `dx` (m) | `wall_friction` | mean relative error vs corrected Ritter |
|---|---|---|
| 0.02 | 0 | 0.478 |
| 0.02 | 1 | 0.621 |
| 0.04 | 0 | 0.542 |
| 0.04 | 1 | 0.639 |

These numbers supersede the `ritter_mean_relative_error` values stored in
`resolution_ladder.json` (0.411 / 0.340 / …, computed against `Z/a = tau`).

### 3.4 Reference-table provenance audit

| quantity | measured |
|---|---|
| points | 20 |
| `tau` range | 0.000 – 4.294 |
| `tau` spacing | 0.226000 ± 4.4e-16 |
| ordinate quantum | 0.05 |
| stated uncertainty | ±0.05 |
| mean abs second difference | 0.0111 (22 % of the stated uncertainty) |
| late slope of the table | 0.523 |

**Verdict:** provenance not independently verifiable: the bundled table is internally inconsistent with the documented digitisation method.

Red flags:

- abscissae are exactly uniform (spacing 0.226, spread 4.4e-16): manual figure digitisation does not produce a uniform grid.
- every Z/a lies on a 0.050 grid, which is also the stated uncertainty -- the ordinates are quantised read-off steps, not measured values with scatter.

What this does *and does not* mean. It does **not** show that the numbers are
wrong: the trend of the table (front advancing at ~1.0 `tau` early, decelerating
to ~0.5 `tau` late) is qualitatively what a collapsing column does, and it lies
below the Ritter line as expected. What it does show is that the table cannot be
the manual figure digitisation its metadata claims, so the repository has no
traceable path from the published figure to these twenty numbers. A gate whose
reference cannot be traced cannot support the words "physical validation".

## 4. Error judgement

| question | answer |
|---|---|
| experimental gate (MAE in `Z/a` <= 0.15), best configuration | 0.158 -> **fail** |
| experimental gate, as shipped (free slip, no shift) | 0.577 -> **fail** |
| front speed within 15 % of the reference | **no** (best ratio 1.68) |
| resolution convergence of the gate metric | **no** — the residual is flat over
  a factor-4 refinement in `dx`, i.e. no observed order to extrapolate |
| platform accuracy standard (> 15 % error unacceptable) | **not met** for the
  physical front speed at any measured configuration |
| is the physical-validation claim supported | **no** — the reference provenance
  audit flags the table, so B2 stays `WITHHELD`, not `pass` |

## 5. Notes, pitfalls, scope

1. **A single scalar MAE hides the physics.** Free slip and no slip give almost
   the same raw MAE (0.599 vs 0.595 at `dx` = 0.04), yet their late-time front
   speeds differ by 47 %. The front is too slow early (startup lag) and too fast
   late (slope error), and the two errors cancel in the mean absolute value. Any
   gate built on one number over the whole window can therefore be satisfied by
   cancellation; the lag and the slope have to be reported separately, which is
   what this artifact does.

2. **The startup lag is partly numerical.** It shrinks with refinement
   (`wall_friction` = 0: 0.59 -> 0.49 from `dx` = 0.04 to 0.02), so part of it is
   the released column taking time to resolve into a bore, and part is the
   half-cell initial offset in the front definition (`Z/a` starts at
   `-0.5 dx/a`). Removing it by a fitted shift is a diagnostic, not a fix.

3. **No slip is the physically consistent choice for this comparison, not
   necessarily the solver default.** The experiment had a no-slip bed; with
   `wall_friction` = 1.0 the late slope drops from 1.47–1.53 to 0.88–1.00 and the
   lag-corrected MAE from 0.55–0.57 to 0.16–0.23, i.e. to the edge of the 0.15
   gate. That is evidence for using no slip in the B2 experimental gate; it is
   not evidence about the other benchmark cases, which were not re-measured here.

4. **2-D vs 3-D.** The reference is a three-dimensional experiment with sidewall
   and bottom friction; this is a 2-D cross-section. Even with a no-slip bed the
   two configurations are not the same flow, so a residual of a few per cent is
   the floor, not zero.

5. **The comparison window is honest but short.** 0.5 s reaches `tau` ~ 4.95, past
   the last reference point (`tau` = 4.29); the interpolation never extrapolates, and all 20 reference points are compared.

## 6. What to do next

1. Trace the twenty reference ordinates back to the published figure, or replace
   the table with one whose provenance is verifiable; until then B2's
   `physical_validation` status is `WITHHELD`.
2. Re-run the gate with `wall_friction` = 1.0 in `experimental_validation.py`
   (the flag already exists) and re-baseline the threshold on the corrected
   Ritter anchor.
3. Report lag and late-time slope as separate gates instead of one windowed MAE.
4. Only then consider a finer ladder: refinement cannot fix a slope error, so it
   should not be spent before the wall treatment is settled.

## Artifacts

- `outputs/dam_break_exp_bias/bias_diagnosis.json` — all metrics above
- `outputs/dam_break_exp_bias/bias_diagnosis.png` — trajectory overlay + residual bars
- `benchmarks/dam_break/exp_bias_diagnosis.py` — measurement + audit tool
- `benchmarks/dam_break/make_bias_doc.py` — this document's generator
- `tests/test_dam_break_exp_bias.py` — pins the audit and the diagnostics
