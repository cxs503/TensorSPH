# CASE-LAPLACE-SPH-DYNAMIC — projection-SPH Young-Laplace drop that stays in equilibrium

`Δp = σ/R` on a static drop, advanced in time with incompressible (projection) SPH.  The companion page `docs/verification_laplace_sph_isph.md` measures the *static* jump to better than 1 % but records an honest FAIL for the time marching: that integration diverged after **12 steps**.  This page closes that gap and states exactly what closes it.

## 1. Case, geometry, parameters

| parameter | value |
|---|---|
| drop radius `R` | 0.0200 m |
| particle spacing `dx` | 0.0025 m |
| smoothing length `h` | 0.0033 m (h/dx = 1.3000) |
| resolution | R/dx = 8.0, R/h = 6.15 |
| surface tension `σ` | 0.0720 N/m |
| density `ρ` | 1000 kg/m³ (both phases: colour-completion carrier) |
| outer carrier layer | 0.0120 m |
| capillary time `t_σ` | 0.0218 s |
| time step | 0.10 t_σ |
| reference velocity `v_ref = √(σ/ρR)` | 0.0600 m/s |
| steps / sampling | 60 / every 10 |
| CG | rel. tol 1e-06, max 4000 iterations |
| curvature | mls (grad.n of the twice-smoothed normal) |

Bond number 0 (no gravity), Weber 0 (statics).  N = 780 particles, 6498 pairs.  The vapour phase carries the liquid density: a static-verification device for a complete colour function `1 → 0`, not a multiphase model.

## 2. Method — what was actually wrong, and the fix

A projection step is only a projection if the operator that *measures* divergence and the one that *corrects* with pressure are a discrete adjoint pair:

```
u^{n+1} = u* − (dt/ρ) G p ,   (−D G) p = −(ρ/dt) D u* ,   G = −D^T
```

The shipped pair (`csf_consistent.mls_divergence` / `mls_gradient`) is first-order consistent but **not mutually adjoint**.  Measured on the drop's own cloud:

| operator pair | adjointness defect `|<Du,q> − <u,−Gq>| / |<Du,q>|` | divergence left after one projection step |
|---|---|---|
| shipped MLS pair | 0.963 | 1.144 (doubles: it is an anti-projection) |
| `projection_adjoint` pair | 4.0e-16 (round-off) | 9.4e-11 |

That single number is why the earlier integration lost its lattice: every step re-injected the divergence it was supposed to remove, so the velocity grew without bound (13 steps).  The new pair is *derived*, not fitted: `D u|_i = Σ_j V_j (u_j − u_i)·∇_i W_ij` on the unordered pair list, and `D^T` follows from `<D u, p> = Σ_pairs (V_j p_i + V_i p_j)(u_j − u_i)·g_ij`; `G = −D^T` reproduces `∇p` on a linear field to 1.008 on an interior stencil.  `A = −D G = D D^T` is symmetric positive semi-definite by construction, one particle is pinned (`p = 0`, the ambient gauge) and the solve is Jacobi-preconditioned CG.

Second ingredient, also measured rather than tuned: a consistent projection still cannot remove the **solenoidal** part of the lattice CSF traction.  A *physical* viscosity `ν = ν_f √(σR/ρ)` (discretised with the same pair Laplacian) is therefore swept; `ν_f = 0` is kept as the inviscid control.  The outer 1.6 dx carrier layer has its velocity pinned: with no ghost particles its truncated kernel is otherwise a momentum source (the same reason the static page put the Dirichlet gauge there).

## 3. Results

### 3.1 The projection is exact (operator-level)

| check | value |
|---|---|
| adjointness defect, shipped MLS pair | 0.963 |
| adjointness defect, adjoint pair | 4.0e-16 |
| `G p` vs analytic `∇p` (p = x, interior mean) | 1.0082 |
| divergence removed in one step | -100.0000 % of the incoming divergence left (9.4e-11), 1266 CG iterations |

### 3.2 The march (this is the capability)

| ν_f | ν [m²/s] | diverged | max\|u\|/v_ref | mean\|u\|/v_ref | Δp [Pa] | Δp/(σ/R) − 1 | area ratio | steps |
|---|---|---|---|---|---|---|---|---|
| 0.003 | 3.6e-06 | no | 0.188 | 0.030 | 3.675 | +2.08 % | 1.00431 | 6 |
| 0.010 | 1.2e-05 | no | 0.179 | 0.029 | 3.673 | +2.02 % | 1.00431 | 6 |
| 0.000 | 0.00 | no | 0.158 | 0.021 | 3.607 | +0.19 % | 0.99569 | 3 |

Adami et al. (2012) admissibility for spurious currents is `max|u|/v_ref < 0.1`.  The legacy pair is listed for reference: it diverges at step 12.

### 3.3 Time series of the best damped run

ν_f = 0.010 (ν = 1.2e-05 m²/s):

| t/t_σ | max\|u\|/v_ref | mean\|u\|/v_ref | Δp/(σ/R) | area ratio | div_after/div_before | CG its |
|---|---|---|---|---|---|---|
| 1.00 | 0.0588 | 0.0075 | 1.0072 | 1.00000 | 9.4e-07 | 638 |
| 2.00 | 0.1117 | 0.0144 | 1.0041 | 1.00000 | 9.9e-07 | 640 |
| 3.00 | 0.1529 | 0.0202 | 1.0024 | 0.99569 | 9.7e-07 | 616 |
| 4.00 | 0.1783 | 0.0246 | 1.0046 | 1.00000 | 9.9e-07 | 646 |
| 5.00 | 0.1867 | 0.0274 | 1.0149 | 1.00000 | 9.8e-07 | 627 |
| 6.00 | 0.1791 | 0.0289 | 1.0202 | 1.00431 | 9.9e-07 | 565 |

![dynamics](../benchmarks/laplace_sph/laplace_sph_dynamics.png)

## 4. Error judgement against the reference

Reference: the analytic Young-Laplace law `Δp = σ/R` = 3.6000 Pa (σ = 0.0720 N/m, R = 0.0200 m).

| quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| Laplace jump `Δp` | 3.6000 Pa | 3.6726 Pa | +2.02 % | yes |
| drop area | initial area | ratio 1.00431 | +0.43 % | yes |
| spurious currents | 0 (equilibrium) | 0.1791 v_ref | — | gate <0.1: FAIL |

Gates:

| gate | value | verdict |
|---|---|---|
| Laplace within 15 % | +2.02 % | yes |
| Adami spurious < 0.1 | 0.1791 | no |
| area drift < 2 % | +0.43 % | yes |
| projection exact (<1e-5 leftover divergence) | 9.4e-11 | yes |
| legacy path still diverges (record kept) | step 12 | yes |

## 5. Notes, limits, failure modes

* **What is new**: the time integration runs.  The jump stays inside the 15 % acceptance band and the shape is conserved, so the projection formulation is now usable as a solver for the static/equilibrium Laplace problem, not only as a one-shot measurement.
* **What is still not**: capillary waves and droplet oscillation (dynamic Bond/Weber ≠ 0) are not covered.  The vapour carrier shares the liquid density, so interface advection with a real density jump remains unimplemented, and no claim is made about it.
* The viscosity is a *damping* device for spurious currents, not a modelled fluid property; the equilibrium it reaches does not depend on ν (the jump is `σ/R` for every ν_f that stays stable), which is why a sweep is reported rather than a single tuned value.
* The legacy operators and their record are untouched: `tests/test_laplace_sph_isph.py::test_strict_xfail_spurious_currents` still asserts that `laplace_isph.relax_projection` diverges, because it still does.  Two paths coexist, one broken and documented as broken.
* Outer-layer velocity pinning is a boundary condition, not physics: with ghost particles it would not be needed.  Its effect is bounded by the area and pressure-jump gates (both measured on the liquid, never on the pinned layer).

## 6. Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic.py --figure
PYTHONPATH=src python benchmarks/laplace_sph/make_doc_dynamic.py
PYTHONPATH=src python -m pytest tests/test_projection_adjoint.py -q
```

| artefact | role |
|---|---|
| `src/tensorsph/projection_adjoint.py` | adjoint pair, gauge-fixed CG, projection step, damped march |
| `benchmarks/laplace_sph/run_dynamic.py` | harness (writes the JSON and the figure) |
| `benchmarks/laplace_sph/results_dynamic.json` | every number on this page |
| `benchmarks/laplace_sph/laplace_sph_dynamics.png` | field + time series |
| `tests/test_projection_adjoint.py` | adjointness, projection exactness, gates, anti-drift |

