# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE — the spurious-current source, one-shot

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · case `CASE-LAPLACE-SPH-DYNAMIC-SOLENOID`.

`docs/verification_laplace_sph_isph_dynamic.md` marches the adjoint-consistent projection at **one** resolution (`R/dx = 8`) and records `max|u|/v_ref = 0.1791` against Adami et al. (2012)'s `< 0.1` line. One resolution cannot tell a **lattice artefact** from a **formulation defect**.

## ① Case and parameters

| quantity | value |
|---|---|
| configuration | 2-D circular drop, equal-density carrier, hydrostatic |
| radius `R` | 20.0 mm |
| surface tension `sigma` | 0.0720 N/m |
| density `rho0` | 1000 kg/m^3 |
| analytic reference | `dp = sigma/R` = 3.6000 Pa |
| velocity scale `v_ref` | 0.0600 m/s |
| kernel / curvature | cubic spline `h = 1.3 dx`; mls (div n of the twice-smoothed normal) |
| resolutions | `R/dx` in [8, 12, 16, 24] |
| split | helmholtz_split, A = D D^T PCG rel_tol 1e-10 |

## ② Method

A projection step removes exactly the component of the velocity its own gradient `G = -D^T` can represent, so the `D`-divergence of the *residual* acceleration is re-injected every step — that is the steady source of a spurious current. `helmholtz_split` separates the static CSF traction `a = (sigma/rho) kappa grad c` into `G q` (annihilated by the projection) and `a_sol` (kept), using the same operator the projection uses. One Poisson solve per level replaces one per time step: the mechanism is measured in seconds instead of hours, at all four resolutions.

Predictor: `max|a_sol| t_ref / v_ref` with `t_ref = sqrt(rho R^3/sigma)` — an order-of-magnitude estimate only, compared with the recorded march value at `R/dx = 8` under a factor-3 gate.

## ③ Results

| `R/dx` | particles | `||a||` [m/s²] | `||a_sol||` [m/s²] | `||Gq||` [m/s²] | solenoidal fraction | irrotationality | adjoint defect | recon defect | CG its | `u_pred/v_ref` |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 780 | 6.119 | 0.9607 | 6.044 | 0.1570 | 0.1570 | 4.0e-16 | 3.1e-18 | 1133 | 0.919 |
| 12 | 1733 | 11.61 | 3.102 | 11.19 | 0.2671 | 0.2671 | 1.7e-15 | 7.1e-18 | 1786 | 2.067 |
| 16 | 3090 | 18.87 | 6.969 | 17.53 | 0.3694 | 0.3694 | 9.6e-16 | 1.2e-17 | 2578 | 5.354 |
| 24 | 6853 | 38.19 | 18.34 | 33.5 | 0.4801 | 0.4801 | 2.8e-16 | 1.8e-17 | 3832 | 12.529 |

Fitted **decay orders** (positive = the quantity falls as `R/dx` grows): solenoidal fraction -1.024, solenoidal amplitude -2.692; the raw slopes are +1.024 and +2.692.

### Gate verdicts (declared values copied verbatim into the JSON)

| gate | measured | declared | verdict |
|---|---|---|---|
| `adjoint_exact` | 1.733e-15 | <= 1e-10 | **PASS** |
| `reconstruction_exact` | 1.778e-17 | <= 1e-06 | **PASS** |
| `solenoidal_decay_order_min` | -1.024 | >= 0.5 | **FAIL** |
| `solenoidal_fraction_finest` | 0.4801 | <= 0.1 | **FAIL** |
| `consistency_with_march` | 5.133 | <= 3 | **FAIL** |

## ④ Figure

`results_dynamic_ladder/laplace_sph_solenoid_ladder.png` — (a) solenoidal fraction vs `R/dx` (log-log) with the fitted order; (b) the absolute split `||a_sol||` vs `||Gq||`; (c) the predicted spurious current against the recorded march value and the Adami line.

## ⑤ Notes, limits, what this does **not** say

* This measures the **source term**, not the current itself. It says whether the residual driving the drop is a discretisation artefact; it does not replace a marched ladder, and `run_dynamic_ladder.py` (pre-registered gates, same anchor) is the marching counterpart.
* The predictor `max|a_sol| t_ref / v_ref` ignores viscous damping and the projection's partial removal per step, so only its **trend and order of magnitude** are claimed (factor-3 gate).
* Equal-density carrier: this is `sigma/R` physics only, not a two-phase flow model. Capillary waves and droplet oscillation are not covered.
* Curvature is the shipped `mls` estimator. Its own noise on the `delta_s` shell is a documented, resolution-independent bias on *pointwise* curvature (`docs/xval_projection_laplace_curvature_variation.md`); the solenoidal fraction measured here is a global norm and is not the same instrument.

## ⑥ Error statement

* The physics reference here is **zero** (an equilibrated drop must not accelerate solenoidal-ly). The claim is a **decay order**, not a percentage: measured -1.024 against a declared minimum of 0.50 — FAIL.
* **< 3 % ?** Not applicable as a percentage — but the equivalent statement is that the finest level's solenoidal fraction is 0.4801 (declared gate 0.10) and the recorded spurious current at `R/dx = 8` is 0.179 against the 0.1 line, i.e. the same defect at 0.1570. Operator identities are exact to round-off (adjoint 1.7e-15, reconstruction 1.8e-17).

## ⑦ Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_sph/diagnose_csf_solenoid.py --render
```

| artifact | what |
|---|---|
| `benchmarks/laplace_sph/diagnose_csf_solenoid.py` | the one-shot diagnostic + renderer |
| `benchmarks/laplace_sph/results_dynamic_ladder/diagnose_solenoid.json` | every number on this page |
| `benchmarks/laplace_sph/results_dynamic_ladder/laplace_sph_solenoid_ladder.png` | figure |
| `tests/test_xval_projection_laplace_solenoid.py` | gates on the record |

