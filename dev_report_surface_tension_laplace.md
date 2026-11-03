# Dev report: Laplace-law verification of the CSF surface-tension model

**Repository** TensorSPH · **Mission** m2142 (self-start, models seat) · **Verdict: honest FAIL of the merged feature's physical claim** (benchmark gate fails by design; test-suite is green because the defect is encoded as a strict xfail).

Primary artefacts: `benchmarks/surface_tension/laplace.py`,
`benchmarks/surface_tension/laplace_ladder.json`,
`benchmarks/surface_tension/laplace_ladder.png`,
`docs/benchmarks/surface-tension-laplace.md`,
`tests/test_surface_tension_laplace.py`.

---

## 1. Case introduction (physics, geometry, parameters)

Surface tension entered TensorSPH via PR #13 (`agent/m277`, CSF model of
Brackbill, Kothe & Zemach 1992) and is declared in the README as implemented.
Its only validation was `benchmarks/static_droplet`, which checks that the force
points inward and that its pointwise magnitude is "reasonable" - neither is a
physical law.

The exact, experiment-free reference for a curved interface is the 2-D Laplace
law `Delta p = sigma / R`. Case: a circular water droplet at rest, radius
`R = 0.1 m`, `sigma = 0.072 N/m` (water-air, 20 C),
`rho0 = 1000.0 kg/m^3`, module defaults `h/dx = 1.3`,
`0.7 * mean` neighbour-count surface threshold, uniform density (the most
favourable case for a density-ratio colour function). Target jump at every
resolution: **0.7200 Pa**.

## 2. Method

* **What must be measured.** The CSF force is a volume density
  `f = -sigma kappa n |grad c|`; the Laplace law is its integral across the
  interface, `integral f.(-n) dxi = sigma kappa`. Comparing a *pointwise*
  acceleration with `sigma/(rho R)` is dimensionally wrong because the pointwise
  field carries `|grad c| ~ 1/dx`. (The shipped static-droplet report does
  exactly that, which is why it cannot see this defect.)
* **Estimator.** For a radial field, `integral f dr = (1/2pi) integral f dA / r`,
  evaluated with the force-weighted centroid radius `r_c`:
  `Delta p = sum rho (-a.n) dx^2 / (2 pi r_c)`. `n` is the exact geometric
  normal; `a` is the module output. `r_c` removes an O(t/R) bias that otherwise
  reaches 3.1% at the coarse level.
* **Gauge.** The same estimator applied to a synthetic smooth bump of known
  normal integral returns `sigma/R` to within
  0.1%, so measurement error is
  negligible compared with the model error.
* **Ablation.** (i) exact `kappa=1/R` with the model's `|grad c|`; (ii) the
  model's `kappa` with an exact normalised one-sided delta. This localises the
  defect.
* **Ladder.** `R/dx = 16, 32, 64, 96` (N = 797 ... 28917 particles), full
  benchmark wall clock 14.2 s on CPU; no tuning of the model.

## 3. Results

### 3.1 Resolution ladder (all numbers from `laplace_ladder.json`)

| R/dx | N | Delta p [Pa] | theory [Pa] | ratio | rel. error | gauge err | kappaR (median) | kappaR (force-weighted) | int delta_s |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 797 | 0.3486 | 0.7200 | 0.484 | 51.6% | 0.1% | -0.210 | -0.000 | 0.253 |
| 32 | 3209 | 0.7729 | 0.7200 | 1.074 | 7.4% | 0.1% | -0.139 | -0.000 | 0.273 |
| 64 | 12853 | 1.6559 | 0.7200 | 2.300 | 130.0% | 0.1% | -0.156 | 0.000 | 0.281 |
| 96 | 28917 | 2.5343 | 0.7200 | 3.520 | 252.0% | 0.0% | -0.256 | 0.000 | 0.284 |

Observed order of the Laplace error: **-1.63** (negative:
the error grows with refinement).

### 3.2 Ablation

| R/dx | Delta p, exact kappa [Pa] | ratio | Delta p, exact delta [Pa] | Delta p, full model [Pa] | ratio |
|---|---|---|---|---|---|
| 16 | 0.1953 | 0.271 | -0.00000 | 0.3486 | 0.484 |
| 32 | 0.2032 | 0.282 | -0.00000 | 0.7729 | 1.074 |
| 64 | 0.2056 | 0.286 | 0.00000 | 1.6559 | 2.300 |
| 96 | 0.2069 | 0.287 | 0.00000 | 2.5343 | 3.520 |

Interpretation:

1. **Unnormalised delta.** With exact curvature the jump converges to
   0.207 Pa = 0.29 x theory,
   exactly the one-sided delta integral 0.253 -> 0.284
   (should be 1). The force is ~3.6x too weak even with perfect curvature.
2. **Zero-mean curvature.** With an exact normalised delta, the model curvature
   yields 1.2e-13 Pa at every level - zero. The
   force-weighted `kappa R` is 0.000 vs exact +1,
   while 50.0% of interface particles have
   `kappa > 0`: the curvature field is noise centred on zero, not `1/R`.
3. **Correlated-noise product.** In the full model both noisy fields come from
   the same neighbour-count colour function, so their product injects a spurious
   inward force that grows with resolution (0.48 -> 3.52 x
   theory). The apparent inward pull is not curvature.

### 3.3 Why a ladder is mandatory

At `R/dx = 32` the error is only 7.4% and would pass a 15% gate;
one refinement later it is 130.0%. A single-resolution check would have produced a
false PASS - the exact failure mode this mission class is meant to catch.

## 4. Error judgement

| gate | measured | limit | result |
|---|---|---|---|
| `gauge_self_test` | 0.001337 | 0.05 | PASS |
| `laplace_error_finest` | 2.52 | 0.15 | FAIL |
| `curvature_sign_and_scale` | -0.2559 | 1 | FAIL |
| `force_direction` | 0.9348 | 1 | FAIL |

Explicit answer: **the key quantity is not within 3%, nor within 15%.** Best
level 7.4%, finest level 252.0%; the curvature is
force-weightedly zero instead of +1 and the delta integral is
0.284 instead of 1. The error does not decrease with
refinement (order -1.63), so it is a discretisation defect, not a
resolution shortfall.

## 5. Notes, pitfalls and scope

* Pitfall 1: pointwise acceleration is not a pressure jump (the shipped
  report's `expected_acceleration_scale` is `sigma/(rho R)`, off by the
  `|grad c|` factor). Any future surface-tension test must integrate.
* Pitfall 2: with a one-sided free surface, particles exist only for `r <= R`,
  so a symmetric delta profile is not representable and a naive Gaussian gauge
  under-reports by ~2x. The gauge here uses a bump on the fluid side, and its
  integral is exact by construction before it is applied to the model.
* Pitfall 3: `R/dx = 8` was excluded - the interface width `2h/R` there is
  16.2% of the radius, violating the thin-interface
  assumption; the numbers at that level are not comparable.
* Scope: 2-D, single phase, static geometry, uniform density, defaults only.
  Capillary-wave dispersion and Lamb oscillation frequencies are **not**
  measured; they are uninformative until the defect is fixed.
* This is a *verification* result only: the solver remains stable with
  `sigma > 0`, and no existing benchmark verdict is changed. What changes is
  that the README no longer implies the model is physically validated.

## 6. Follow-up

A native mission should be dispatched for the fix (normalise the surface delta;
replace the curvature estimator with a consistent one; accept only against this
ladder with a non-negative observed order). The strict-xfail tests in
`tests/test_surface_tension_laplace.py` will fail loudly the moment the model
starts passing, forcing this report and the docs to be updated rather than
leaving a stale claim.
