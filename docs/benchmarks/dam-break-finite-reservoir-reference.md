# B2 dam break: a finite-reservoir reference instead of Ritter

**Repository**: TensorSPH ｜ **Module**: `src/tensorsph/shallow_water.py` ｜
**Artifact**: `benchmarks/dam_break/shallow_water_reference.json` ｜
**Tests**: `tests/test_shallow_water_reference.py`

**Verdict first.** B2's front has been reported at a systematic −27 %…−41 %
against Ritter's `Z/a = 2 tau` with the caveat "Ritter assumes a semi-infinite
reservoir, so agreement is not expected". That caveat was never quantified, so
the deficit could not be attributed. This work quantifies it: a numerically
verified shallow-water solution of the **same finite column** is
`Z/a = 1.70 tau` (grid-converged to 0.010 over the three grids,
estimator-bias-corrected ≈ 1.78), i.e. the geometry accounts for only
about 16
percentage points, not the whole deficit. **The SPH front really is slow; the
frame convention is exonerated.**

## ① Case, geometry and parameters

Same case as B2 and as the classical experiment's non-dimensionalisation
(`tau = t sqrt(2 g / a)`, `Z/a` measured from the initial column face):

| parameter | value |
|---|---|
| column width `a` | 0.2 m |
| column height `H` | 0.4 m |
| aspect ratio `H/a` | 2 (the classical 2:1 column) |
| reference tank width | 3.0 m (wider than B2's 2.0 m so the front never reaches a wall inside the window) |
| gravity | 9.81 m/s² |
| initial velocity | 0, dry bed downstream, frictionless walls |
| reported window | `tau <= 4.4` |
| `tau` at which the rarefaction reaches the back wall | 1.00 |

## ② Method

1. **A reference for the same idealisation.** `tensorsph.shallow_water` integrates
   the depth-averaged equations with HLL fluxes (Einfeldt wave speeds), MUSCL
   reconstruction with a minmod limiter and the Hancock half-step predictor,
   reflective walls (`0.5 g h^2` momentum flux, zero mass flux), positivity
   clamping and a dry-state guard. CFL 0.4, `dt` recomputed every step.
2. **Verify the generator before quoting it.** The same code is run in the
   *semi-infinite* regime (dam at `x = 3.0 m`, windows
   `t = 0.1, 0.2 s`) where Ritter is exact, and
   scored against the exact solution **with the same leading-edge estimator on the
   same grid**, so the estimator's bias is measured rather than hidden.
3. **Only then** the finite column is integrated at three resolutions and the
   `Z/a` slope over `tau >= 2` is taken.
4. **Attribution.** The archived SPH samples (`benchmarks/dam_break/run.py`,
   `dx = 0.02 m`) are compared with the reference and with Ritter in one window.

The leading-edge estimator is a depth threshold (`h > level * H`). It is a
*measurement instrument*, so it was calibrated: the depth-threshold crossing on a
first/second-order scheme under-reads the travel by a few percent, and the
calibration table below is what fixes that number.

## ③ The generator, verified against the exact solution

`L1` is the depth-profile error over the whole domain; the three error columns are
the leading-edge error at `h/H = 0.01, 0.001, 0.0001`, always as a fraction of
the travel distance from the dam (the lab frame is not used: comparing absolute
`x` would dilute a front error by the dam offset, which is exactly the trap the
`ritter_front` docstring warns about).

| order | cells | t (s) | L1 | edge @0.01 | edge @0.001 | edge @0.0001 | mass error |
|---|---|---|---|---|---|---|---|
| 1 | 400 | 0.10 | 0.45 % | -9.3 % | -8.2 % | -3.9 % | 1e-13 |
| 1 | 400 | 0.20 | 0.59 % | -11.2 % | -12.1 % | -9.7 % | 3e-13 |
| 1 | 800 | 0.10 | 0.29 % | -11.2 % | -12.1 % | -9.7 % | 1e-13 |
| 1 | 800 | 0.20 | 0.38 % | -8.9 % | -12.9 % | -11.6 % | 3e-13 |
| 1 | 1600 | 0.10 | 0.19 % | -8.9 % | -12.9 % | -11.6 % | 2e-13 |
| 1 | 1600 | 0.20 | 0.24 % | -6.7 % | -11.5 % | -11.6 % | 4e-13 |
| 2 | 400 | 0.10 | 0.38 % | +0.0 % | +0.0 % | +3.9 % | 7e-14 |
| 2 | 400 | 0.20 | 0.51 % | -4.5 % | -6.1 % | -3.9 % | 3e-13 |
| 2 | 800 | 0.10 | 0.26 % | -4.5 % | -6.1 % | -3.9 % | 1e-13 |
| 2 | 800 | 0.20 | 0.34 % | -4.5 % | -9.0 % | -7.7 % | 3e-13 |
| 2 | 1600 | 0.10 | 0.17 % | -4.5 % | -9.0 % | -7.7 % | 2e-13 |
| 2 | 1600 | 0.20 | 0.22 % | -3.9 % | -8.5 % | -8.7 % | 4e-13 |

Reading: second order is worth about 6 points of leading-edge accuracy at the
same cost; the `0.01` level at second order is the least biased and the most
stable across grids (−4.5 %…−3.9 %), so that is the level used for the reference,
with the residual carried as a ±5 % uncertainty on the reference's slope. Mass
conservation is round-off (a reflective wall cannot leak), `h` at the dam section
converges to the classical `4H/9` from below, and the Riemann invariant
`u + 2 sqrt(g h) = 2 c0` is preserved across the fan except at the smeared
initial discontinuity, where its worst relative deviation is 19 % on the 800-cell
grid (pinned as a regression guard, not quoted as an accuracy claim).

## ④ The finite-reservoir result

`front_coefficients` is the least-squares slope of `Z/a` against `tau` over
`tau >= 2`, i.e. after the column has finished collapsing:

| cells | steps | mass error | slope @0.01 | slope @0.001 |
|---|---|---|---|---|
| 800 | 1023 | 1e-11 | 1.694 | 1.776 |
| 1600 | 2106 | 2e-11 | 1.704 | 1.815 |
| 3200 | 4314 | 2e-11 | 1.701 | 1.849 |

Ritter's slope is 2 by construction. The reference slope is
**1.701 ± 0.010** (grid spread) at the calibrated level, i.e.
-15.0 % against Ritter; the bias-corrected value is
1.78. The `0.001` column is reported for transparency: it is the same
profile read with a smaller threshold, and it sits +9 % higher,
which is the size of the estimator's level dependence.

![finite-reservoir reference](shallow_water_reference.png)

## ⑤ Attribution of the SPH deficit

The SPH run at `dx = 0.02 m` gives, over
`tau` in [2.08, 4.25]
(23 samples):

| quantity | value |
|---|---|
| SPH vs Ritter, mean relative difference in `Z/a` | -53.7 % |
| finite-reservoir reference vs Ritter, same window | -15.6 % |
| SPH vs finite-reservoir reference | -45.1 % |
| share of the SPH deviation explained by resorting to Ritter | 29 % |
| SPH front slope `Z/a` per `tau` (window) | 1.473 |
| same slope in units of Ritter's 2 | 0.737 |

(The whole window from `tau` = 1.09 gives
-59.3 % — larger in
magnitude because `tau < 1` is still the collapse transient, where the SPH front
has barely started to move. Averaging that into the window would flatter or
punish the solver for the wrong reason, so it is reported separately.)


**Answer to the question the repository could not answer before: no,
Ritter's inapplicability does not explain the B2 front deficit.** Retaining the
reservoir's finite extent removes 15.6
percentage points of the 53.7-point
SPH deficit — about 29 % of it. The remaining
45.1 points are not a frame
convention, not a non-dimensionalisation slip and not a Ritter artefact; they are
a property of the 2-D solver (or of the deep-column physics that shallow water
cannot represent). One hypothesis is now dead, and the experiment — still
`WITHHELD` because the bundled table is synthetic — remains the only arbiter of
which of the two it is.

## ⑥ Error verdict, explicitly

| claim | reference | error | within 3 %? |
|---|---|---|---|
| shallow-water generator reproduces Ritter's leading edge | exact Ritter, estimator-matched | +4.5 % (worst, order 2) | no — 4.5 %, reported as the reference's uncertainty |
| shallow-water generator reproduces Ritter's profile | exact Ritter | 0.51 % L1 | yes |
| finite-reservoir slope is grid-converged | self-convergence of three grids | 0.010 in slope (0.6 %) | yes |
| SPH front vs finite-reservoir reference | this reference | -45.1 % | **no** — and that is the finding, not a defect of the reference |
| SPH front vs experiment | Martin & Moyce (1952) | not evaluated | no citable table: the bundled one is synthetic (`reference_audit.json`, WITHHELD) |

The last row is the only one that can settle whether the remaining deficit is a
solver defect or deep-column physics; this work does not claim to settle it.

## ⑦ Notes, traps and scope

* **`H/a = 2` is outside shallow water.** This reference is *not* a physical model
  of B2 and must never be quoted as validation against the 1952 experiment. Its
  only claim is the elimination of the "Ritter does not apply" hypothesis.
* **Frame traps.** `Z` is measured from the initial column face, while the SPH
  `front` is a lab-frame `x_max`. Subtracting the column width once is required;
  forgetting it costs a whole column width in every sample, and measuring the
  error against absolute `x` hides 12 points of front error behind the dam offset
  (this was measured, not guessed).
* **Estimator bias is real.** First order at `dx = 0.0075 m` under-reads the
  travel by 12 %; a *smaller* threshold is not automatically better (the numerical
  tip has an exponential tail while the exact tip is quadratic). The threshold has
  to be calibrated on the exact solution, and the calibration must use the travel
  frame.
* **Startup transient.** Before `tau ≈ 0.5` the SPH column has not collapsed and
  its front has barely moved; relative errors measured there are dominated by the
  transient. The attribution window therefore starts at `tau = 2`.
* **Reflective-wall flux.** Zeroing the whole boundary flux (instead of only its
  mass component) removes the wall pressure `0.5 g h^2` and lets the tank leak
  momentum; a mirrored ghost state instead advects `h u^2` through a solid wall.
  Both are silent at the level of the front trajectory.

## ⑧ Reproduce

```bash
PYTHONPATH=src python benchmarks/dam_break/shallow_water_reference.py --plot
PYTHONPATH=src python benchmarks/dam_break/make_sw_reference_doc.py
PYTHONPATH=src python -m pytest tests/test_shallow_water_reference.py -q
```
