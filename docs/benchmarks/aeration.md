# Free-surface air entrainment, bubble transport and aerated drag

Campaign node **P165W2-N04** (TensorSPH).  Module: `src/tensorsph/aeration`.
Report generated from `benchmarks/aeration/report.json` (2026-11-10T14:31:07+00:00, torch 2.7.1).

## 1. Case introduction

A free surface that breaks entrains air.  The chain implemented here has
three links, each an independent closure with its own anchor:

| link | quantity | closure in this module | anchor |
|---|---|---|---|
| inception | `K = V sin t / sqrt(g d cos t)` | Chanson (1997), `C_crit = 1.0` |
| | | | energy balance `C_crit = sqrt(2c) = 1.000` |
| transport | rise velocity, void fraction | Stokes / rigid-sphere drag balance /
| | | Mendelson (1967) wave theory; advective-diffusion `C(y,t)` |
| | | `C = 1 - erf(y/(Y90/sqrt2))` |
| resistance | friction reduction `R(alpha)` | Chisholm multiplier vs mixture log law |
| | | mutual cross-check |

Physical parameters (all declared, none fitted): water/air at 20 C,
`rho_l = 998.2 kg/m3`, `mu_l = 1.002e-3 Pa s`, `sigma = 0.0728 N/m`, `rho_g = 1.204 kg/m3`.
Hydraulic-jump configuration: inflow depth 0.1 m, `Fr_1 = 4 .. 8`, station
`x = 3.0 d_2`, roller diffusivity `D = 0.050 m2/s`, bubble radius `R = 0.15 mm`.

## 2. Method

* **Inception.** The criterion is evaluated in closed form and, independently,
  by bisecting the sign change of the entrainment-flux closure, so the algebraic
  and the operational onset cannot drift apart.
* **Void fraction.** Finite-volume integration of
  `dC/dt = D d2C/dy2 + w dC/dy` on `y in [0, 2 m]` with a Dirichlet surface
  (`C = 1`), zero far field and upwinded buoyant drift; `w` from the
  rigid-sphere drag balance of a 0.15 mm bubble.
* **Bubbles.** Terminal rise velocity from the drag balance with
  Schiller-Naumann drag (rigid sphere) below 0.2 mm, Mendelson wave theory
  above 0.5 mm, linear blend in between; Lagrangian clouds integrate drift +
  turbulent random walk (same `D`) and are mapped to the Eulerian void
  fraction by the SPH summation `alpha_i = sum_j V_j W(r_ij, h)`.
* **Resistance.** Lockhart-Martinelli parameter with the turbulent-turbulent
  correlation, Chisholm multiplier with `C = 20`, and a log-law friction factor
  `sqrt(8/f) = (1/kappa)(ln(Re sqrt(f/8)) - 1.5) + B` for the mixture; both
  routes report a reduction rate referenced to clear water at the same
  mixture volumetric flux.
* Toolchain: `torch` (double precision), `pytest`, `matplotlib`; all runs on CPU.

## 3. Results

### 3.1 Criteria

| id | measured | threshold | comparison | pass |
|---|---|---|---|---|
| C1 | 0.0000 | 0.10 | <= | PASS |
| C2 | 0.0954 | 0.15 | <= | PASS |
| C3 | 0.1310 | 0.20 | <= | PASS |
| C4 | 0.0000 | 0.10 | <= | PASS |
| C5 | 0.0037 | 0.05 | <= | PASS |
| C6 | 31 tests | 15 | >= | PASS |

All criteria passed: **True**.

### 3.2 Inception (C1)

Energy-balance constant `sqrt(2c)` with cavity fraction `c = 0.50` gives 1.000000 against the empirical 1.000000: relative error 0.00%.
The bisection onset detector reproduces the algebraic inception velocity to 0.00%.

| slope [deg] | inception velocity [m/s] | inception Froude [-] |
|---|---|---|
| 5.0 | 8.0204 | 11.452 |
| 10.0 | 4.0024 | 5.715 |
| 15.0 | 2.6595 | 3.797 |
| 30.0 | 1.3035 | 1.861 |
| 45.0 | 0.8329 | 1.189 |

(reference depth 0.050 m; the inception Froude
number is strongly slope dependent, so no single value is quoted.)

### 3.3 Aeration layer vs the erf anchor (C2)

| Fr1 | d2 [m] | U2 [m/s] | t [s] | Y90 [m] | L2 vs erf, drift free | L2 vs erf, with drift | L2 vs closed form |
|---|---|---|---|---|---|---|---|
| 4.0 | 0.518 | 0.765 | 2.031 | 0.901 | 0.00% | 6.49% | 0.08% |
| 5.0 | 0.659 | 0.752 | 2.630 | 1.026 | 0.00% | 7.38% | 0.08% |
| 6.0 | 0.800 | 0.743 | 3.231 | 1.137 | 0.00% | 8.16% | 0.08% |
| 8.0 | 1.082 | 0.732 | 4.436 | 1.332 | 0.15% | 9.54% | 0.12% |

Worst L2 against the erf anchor: **9.54%** (threshold 15.00%).

### 3.4 Bubble rise velocity (C4)

| R [mm] | model [m/s] | Mendelson [m/s] | rigid sphere [m/s] | error |
|---|---|---|---|---|
| 0.50 | 0.5491 | 0.5491 | 0.1122 | 0.00% |
| 0.91 | 0.4222 | 0.4222 | 0.1918 | 0.00% |
| 1.32 | 0.3695 | 0.3695 | 0.2624 | 0.00% |
| 1.73 | 0.3440 | 0.3440 | 0.3277 | 0.00% |
| 2.14 | 0.3320 | 0.3320 | 0.3894 | 0.00% |
| 2.55 | 0.3275 | 0.3275 | 0.4483 | 0.00% |
| 2.95 | 0.3276 | 0.3276 | 0.5051 | 0.00% |
| 3.36 | 0.3307 | 0.3307 | 0.5600 | 0.00% |
| 3.77 | 0.3357 | 0.3357 | 0.6134 | 0.00% |
| 4.18 | 0.3419 | 0.3419 | 0.6655 | 0.00% |
| 4.59 | 0.3491 | 0.3491 | 0.7165 | 0.00% |
| 5.00 | 0.3568 | 0.3568 | 0.7664 | 0.00% |

Stokes-limit check of the drag-balance solver at `R = 1.0e-05 m`: 2.161431e-04 m/s vs analytic 2.169113e-04 m/s, error 0.35%.
At the hand-over radius `R = 5.0e-04 m` the rigid-sphere branch
gives 0.1122 m/s against Mendelson's 0.5491 m/s (ratio 4.89), which is why
the rigid-sphere closure is abandoned there.

### 3.5 Aerated friction reduction (C3)

| alpha | X | Phi_l^2 | R (Chisholm) | R (log law) | rel. deviation | in band |
|---|---|---|---|---|---|---|
| 0.02 | 729.7 | 1.0274 | 0.0102 | 0.0169 | 66.04% | no |
| 0.05 | 311.1 | 1.0643 | 0.0318 | 0.0424 | 33.24% | no |
| 0.08 | 198.0 | 1.1010 | 0.0559 | 0.0679 | 21.45% | no |
| 0.10 | 158.8 | 1.1260 | 0.0729 | 0.0850 | 16.67% | no |
| 0.12 | 132.1 | 1.1515 | 0.0904 | 0.1022 | 13.10% | yes |
| 0.16 | 97.8 | 1.2047 | 0.1265 | 0.1367 | 8.02% | yes |
| 0.20 | 76.5 | 1.2615 | 0.1640 | 0.1714 | 4.52% | yes |
| 0.25 | 59.1 | 1.3388 | 0.2122 | 0.2152 | 1.40% | yes |
| 0.30 | 47.1 | 1.4249 | 0.2616 | 0.2594 | 0.84% | yes |
| 0.35 | 38.4 | 1.5220 | 0.3119 | 0.3040 | 2.51% | no |

Worst deviation inside the predeclared band `alpha in [0.12, 0.30]`: **13.10%** at `alpha = 0.12`.
Outside the band (dilute end) the relative deviation reaches 66.04% at `alpha = 0.02`, while the absolute gap there is only 0.0067; both rates collapse to zero, so a relative
criterion is meaningless below the band.

### 3.6 Air-volume conservation (C5)

Lagrangian cloud (4000 bubbles, drift 0.0293 m/s, `D = 0.050 m2/s`) over four characteristic times of 2.0 s: relative volume drift 0.00% (reflecting free surface).
With the escaping-surface bookkeeping, in-domain plus escaped closes to 1.49e-08 (escaped fraction 83.38%).
SPH map: prescribed void fraction 0.0500, kernel-summation mean 0.05019, relative error 0.37%.

### 3.7 Figures

* `benchmarks/aeration/aeration_profile_and_rise_velocity.png`
* `benchmarks/aeration/drag_reduction.png`
* `benchmarks/aeration/inception_froude.png`

## 4. Error assessment: is every key quantity below 3%?

| quantity | reference | error | < 3%? |
|---|---|---|---|
| FV profile with drift | closed-form advective-diffusion solution | 0.08% .. 0.12% | yes |
| FV profile, drift free | erf anchor | 0.15% | yes |
| FV profile with drift | erf anchor (drift-free idealisation) | 9.54% | **no** (physical drift, see below) |
| drag-balance rise velocity | Stokes limit | 0.35% | yes |
| rise velocity in band | Mendelson formula | 0.00% | yes (closure identity) |
| inception constant | energy balance vs empirical | 0.00% | yes |
| onset velocity | algebraic inception | 0.00% | yes |
| air volume, Lagrangian | initial volume | 0.00% | yes |
| air volume, SPH map | prescribed void fraction | 0.37% | yes |
| friction reduction | Chisholm multiplier | 13.10% | **no** (cross-closure spread, declared) |

Two entries answer **no** and are not papered over:

1. The drift-carrying profile departs from the *drift-free* erf anchor by up to
   9.54%.  That departure is the physics of
   buoyancy, not solver error: the same profile matches the closed-form
   drift solution to better than 0.1%.  A drift-free anchor is therefore the
   correct reference only while `w t << Y_90`, which is exactly the
   configuration used here (0.06 m against 0.9 m).
2. The two friction-reduction closures differ by up to
   13.10% inside the bubbly band.  They are
   independent engineering closures, not measurements, so a 13% spread is the
   honest state of knowledge; the 20% campaign threshold is met.

## 5. Pitfalls, limitations, range of validity

* **No experimental table is bundled.**  C1 compares the criterion constant with
  its own energy-balance derivation and C4 compares the model with the
  Mendelson formula it is built on.  Neither is independent experimental
  validation; both are implementation/consistency checks, and they are labelled
  as such rather than dressed up as validation.
* **Mendelson's range.**  The wave-theory formula is reliable for bubbles of
  roughly 1-20 mm diameter; at `R = 0.5 mm` it returns 0.55 m/s where
  measurements give roughly 0.1 m/s.  The campaign criterion is defined against
  the formula, so the model matches it, but the *formula* is outside its
  validity range at the small-radius end.
* **Rigid-sphere branch.**  Below 0.2 mm the interface is immobile and the drag
  balance is right; at the 0.5 mm hand-over it is already 4.9x below the
  mobile-interface value.  The blend in between is an interpolation, not a
  measured transition.
* **Single surface source.**  Only bubbles entrained at the free surface are
  followed; bubble breakup/coalescence, dissolution and the two-way coupling
  back into the liquid momentum equation are not modelled, so the void fraction
  does not react on the flow.  This is a one-way (dilute) closure.
* **`gamma = 0`.**  The Wood (1991) von-Karman modification is implemented but
  its coefficient is left at zero: no drag-reduction dataset is bundled, and
  fitting it to the Chisholm answer would make the C3 cross-check circular.

## 6. Reproduction

```bash
python benchmarks/aeration/run.py        # criteria + figures + report.json
python benchmarks/aeration/make_doc.py  # regenerate this document
python -m pytest tests/test_aeration.py -q
```

Test suite: `31 passed, 1 warning in 12.91s`.

