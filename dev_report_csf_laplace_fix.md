# Dev report: repairing the CSF surface-tension model (Laplace law) and restoring the contact-angle BC

**Repository** TensorSPH · **Mission** m2454 (self-start, applications seat) · **Branch** agent/m2454
**Verdict**: the Laplace defect documented by m2142 is repaired to **within the 15 % gate at
every ladder level** (worst level 2.4 % → 12.0 %, was 51.6 % → 252.0 %); one of the four
benchmark gates (`force_direction`) still fails on 3 % of the interface particles at the
finest level. The merged branch also did not import at all; that is fixed.

Primary artefacts: `src/tensorsph/surface_tension.py`,
`benchmarks/surface_tension/laplace.py`, `artifacts/surface_tension_laplace.json`,
`artifacts/surface_tension_laplace.png`, `tests/test_interface_fields.py`,
`tests/test_surface_tension_laplace.py`, `tests/test_contact_angle.py`,
`docs/benchmarks/surface-tension-laplace.md`.

---

## 0. Two findings before the physics

1. **The merged branch did not import.** `src/tensorsph/__init__.py` exported
   `apply_contact_angle_boundary` from `tensorsph.surface_tension`, but that function did not
   exist anywhere in the git history of the module (`git log -S` is empty), so
   `import tensorsph` raised `ImportError` and *every* test file that imports the package
   failed at collection. `tests/test_contact_angle.py` had been written as the specification of
   that missing function. It is now implemented (`apply_contact_angle_boundary`, local wall
   frame by PCA + magnitude-preserving rotation of the interface normal onto the prescribed
   angle, distance-weighted blending, `compute_surface_tension_force(..., wall=...)`), and its
   11 tests pass.
2. **The `honest FAIL` of m2142 was correct and its follow-up was never dispatched.** The
   benchmark gate and the strict-xfail tests were still in place; this mission executed the
   fix the report asked for.

## 1. Case (physics, geometry, parameters)

A 2-D circular water droplet at rest, radius `R = 0.1 m`, `sigma = 0.072 N/m`,
`rho0 = 1000 kg/m^3`, `h/dx = 1.3`, uniform density, `R/dx = 16, 32, 64, 96`
(N = 812 … 28 968 particles). The exact reference is the 2-D Laplace law `dp = sigma/R`
= **0.7200 Pa** at every level. Measured quantity: the normal integral of the CSF force
density over the fluid side of the interface,
`dp = sum rho (-a . n_exact) dx^2 / (2 pi r_c)` with the force-weighted centroid radius
`r_c` (geometry-exact for a thin shell). The estimator's own error is bounded by the
synthetic-field gauge self-test: **<= 0.13 %** at every level, so the numbers below are
model error, not measurement error.

## 2. Method

The 0.8.0 model built its interface from a neighbour-count heuristic
(`c_i = count_i / mean(count)`, surface detector `count < 0.7 mean`) and took
`delta_s = |grad c|` unnormalised, with all derivatives at the *particle* smoothing length
`h = 1.3 dx`. Four independent defects were identified and repaired; all interface fields now
come from one function, `interface_fields()`, which the force, the tests and the benchmark
share (so a measurement can no longer describe a different model than the one shipped):

| # | Defect in 0.8.0 | Repair |
|---|---|---|
| 1 | colour field = raw neighbour count (Poisson noise `sqrt(n)` into `grad c`) | colour = kernel completeness `c_i = sum_j V_j W_ij`: 1 in the interior, ~1/2 on a planar free surface, a smooth volumetric measure |
| 2 | derivatives at `h = 1.3 dx`; the interface shell is only two particles wide, so `kappa = div n` is noise of zero mean | one smoothing length `h_c = 2 h` for the whole interface (`Config.surface_tension_h_factor`) |
| 3 | difference-form gradient used unrenormalised at a truncated support; measured on a half plane it is **3.3x too small at the surface row** | first-order consistency (Cummins & Rudman 1999): divide by the 2x2 matrix `A_i = -sum_j V_j grad W_ij (x_i-x_j)`, singular values clamped to [0.3, 3] |
| 4 | `delta_s = |grad c|` (one-sided integral 0.28 instead of 1) | `delta_s = 2 |grad c|`; the continuum identity for a plane is `int_fluid |grad c| dxi = 1/2`, but only the fluid half is discretised, so the pressure jump `int_fluid rho a.(-n) dxi = sigma kappa` needs the factor 2 |
| — | surface-particle selector (recall 0.82–1.00) | dropped: `delta_s` already vanishes in the interior, so the force is applied everywhere and no detector can bias it |

Curvature keeps the difference-form divergence `div n = sum_j V_j (n_j - n_i).grad W_ij`
(consistent for linear fields) with the scalar support factor `-2 / sum_j V_j (x_i-x_j).grad W`
(= 1 for a full kernel), clamped to [0.5, 2]. The unnormalised-divergence operator was *not*
the 0.8.0 defect; the noisy normal was.

Tool chain: `python benchmarks/surface_tension/laplace.py --radii-dx 16,32,64,96`
(CPU, float64, ~2 min total; the ladder is the acceptance harness).

## 3. Results (all numbers read from `artifacts/surface_tension_laplace.json`)

| R/dx | N | dp [Pa] | theory | ratio | rel. error | kappa R (median) | inward fraction | int delta_s |
|---|---|---|---|---|---|---|---|---|
| 16 | 812 | 0.7370 | 0.7200 | 1.024 | **2.4 %** | 1.367 | 1.00 | 0.74 |
| 32 | 3209 | 0.6659 | 0.7200 | 0.925 | **7.5 %** | 1.236 | 1.00 | 0.75 |
| 64 | 12853 | 0.6423 | 0.7200 | 0.892 | **10.8 %** | 1.092 | 1.00 | 0.75 |
| 96 | 28917 | 0.6336 | 0.7200 | 0.880 | **12.0 %** | 1.111 | 0.97 | 0.75 |

Same ladder before the repair (m2142, unchanged model):

| R/dx | dp [Pa] | rel. error | kappa R (median) | int delta_s |
|---|---|---|---|---|
| 16 | 0.3486 | 51.6 % | -0.210 | 0.253 |
| 32 | 0.7729 | 7.4 % (false PASS) | -0.139 | 0.273 |
| 64 | 1.6559 | 130.0 % | -0.156 | 0.281 |
| 96 | 2.5343 | 252.0 % | -0.256 | 0.284 |

Gates: `gauge_self_test` PASS (<=0.13 %), `laplace_error_finest` **PASS** (12.0 % <= 15 %),
`curvature_sign_and_scale` **PASS** (|1.111-1| = 0.111 <= 0.15), `force_direction`
**FAIL** (0.97 < 1.00). Verdict therefore remains FAIL, on one gate.

## 4. Figures

* `artifacts/surface_tension_laplace.png` - the ladder: measured dp vs sigma/R, the
  curvature ratio kappa R, and the delta-function integral, all vs R/dx (generated by the
  benchmark, no hand-copied numbers).

## 5. Notes, pitfalls, scope

* Pitfall (kept from m2142, still true): pointwise `|a|` is **not** comparable with
  `sigma/(rho R)`; the CSF force is a volume density carrying `delta_s ~ 1/dx`. Only the
  normal integral is physical. The shipped `static_droplet` benchmark still does the
  pointwise comparison and therefore cannot see any of this.
* Pitfall: the difference-form gradient is exact for linear fields **only over a complete
  support**. At a free surface it is 3.3x too small (measured on a half plane, `h_c = 2h`).
  This was the single largest contributor to the 0.8.0 amplitude deficit and is invisible to
  any test that does not put particles next to a boundary - which is why it is now pinned by
  `tests/test_interface_fields.py`.
* Pitfall: the remaining error is **not** convergent. The observed order between the two
  finest levels is -0.26, i.e. the error approaches ~12 % from below. It is the residual of
  two partially cancelling one-sided errors: `int delta_s = 0.75` instead of 1 (the outermost
  half-spacing of fluid is not sampled, and no geometry-free kernel identity restores it) and
  `kappa R = 1.11` instead of 1. Forcing `int delta_s = 1` alone would overshoot by ~11 %.
  A resolution-independent 12 % is *better* than a diverging model but it is still not a
  converged verification, and it must not be quoted as one.
* Scope: 2-D, single phase, static geometry, uniform density, default parameters. Capillary
  wave dispersion and Lamb frequencies are still not measured.
* `force_direction` fails on 3 % of interface particles at `R/dx = 96`, where the local
  divergence of the normal field is slightly negative. Normal-field smoothing (1-3 passes)
  raises the inward fraction to 0.973 but does not reach 1.00; the gate is an absolute
  requirement, so it stays failing and is reported as such.
* The strict-xfail test for the curvature at `R/dx = 32` (24 % high) remains: it now encodes
  a *residual bias*, not a sign error, and still fails the suite the day it starts passing.

## 6. Error judgement (explicit)

| quantity | measured | reference | rel. error | <3 %? |
|---|---|---|---|---|
| dp at R/dx = 16 | 0.7370 Pa | 0.7200 Pa | 2.4 % | **yes** |
| dp at R/dx = 32 | 0.6659 Pa | 0.7200 Pa | 7.5 % | no |
| dp at R/dx = 64 | 0.6423 Pa | 0.7200 Pa | 10.8 % | no |
| dp at R/dx = 96 | 0.6336 Pa | 0.7200 Pa | 12.0 % | no |
| kappa R at R/dx = 96 | 1.111 | 1 | 11.1 % | no |
| kappa R at R/dx = 32 | 1.236 | 1 | 23.6 % | no |
| gauge (estimator) | <= 0.0013 | 0 | <= 0.13 % | yes |

**Answer to "is the key quantity within 3 %?"**: yes at the coarsest level (2.4 %), no at the
three finer levels (7.5 %, 10.8 %, 12.0 %). The 15 % acceptance gate is met at all four
levels; the 3 % criterion is not met except at `R/dx = 16`, and the error does not decrease
with refinement, so this is a bounded but *not* converged agreement.

## 7. What is not done

* The 12 % amplitude residual (one-sided delta sampling) is diagnosed and quantified, not
  removed; a consistent one-sided delta normalisation is the next mission.
* `force_direction` at the finest level (3 % of particles) - see above.
* The README's surface-tension entry was not rewritten in this mission; the corrected claim
  lives in `docs/benchmarks/surface-tension-laplace.md` and this report.
* Three-dimensional and multi-phase extensions are untouched.
