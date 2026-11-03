# Verification of the viscous-transport operator against the exact Stokes decay

**What**: `tensorsph.droplet_impact.rhs` -- the regularised Morris-type laminar viscous term, `VISCOUS_LAPLACIAN_FACTOR * nu * sum_j (m_j/rho_j) * ReLU(-(r_ij . grad W_ij)) / (|r_ij|^2 + 0.01 h^2)`, together with the `Config.wall_friction` no-slip condition that lets the same term act on fluid-wall pairs.

**Why**: before this, the operator was pinned only on one frozen lattice at one resolution, and one flow reproduced the Stokes decay rate to a ratio of 0.84 at dx = 0.02 -- a 16 % error, outside the 15 % band, with no information about what happens under refinement. Any Reynolds number quoted from this solver depends on that ratio.

**Headline**: the effective viscosity is *low*, the deficit is first order -- -35.0 % at 5 rows across the channel, -10.6 % at 20 -- so a case configured at `nu` runs at roughly `nu/0.89`. The 3 % standard is met nowhere in the usable ladder; the 15 % standard is met only from dx = 0.01 m upwards.

## 1. Case and exact solution

| item | value |
|---|---|
| channel height `H` | 0.2 m |
| slab width | 0.6 m |
| viscosity `nu` | 0.05 m^2/s |
| initial condition | `u_x = U0 sin(pi y / H), u_y = 0, no body force` |
| exact velocity | `u_x(y,t) = U0 sin(k y) exp(-nu k^2 t), k = pi/H` |
| exact modal amplitude | `A(t) = U0 exp(-nu k^2 t)` |
| exact energy | `E_k(t) = E_k(0) exp(-2 nu k^2 t)` |
| wave number `k = pi/H` | 15.7080 1/m |
| boundaries | three layers of wall particles above and below the slab, wall_friction = 1 (no slip at both plates); the lateral ends are an open boundary and no metric is taken near them |
| measured band in x | 0.35 W .. 0.65 W |
| integrator | explicit Euler, vel += dt*acc; pos += dt*vel, apply_boundary after each drift, Verlet pair cache |
| end time | 0.25 s |
| acoustic Courant, fixed across the ladder | 0.29 |
| environment | Python 3.11.6, torch 2.7.1, cpu |

The exact solution is the first viscous eigenmode of a slab with no slip at both plates. It is a solution of the full Navier-Stokes equations here, not a simplification: the field is x-independent, so the convective term vanishes, the pressure is uniform, and the momentum equation reduces to the diffusion equation.

Two estimators of the effective viscosity are fitted, because they fail differently. `nu_modal` is the rate of the modal amplitude `sum m u_x sin(k y) / sum m sin^2(k y)`, which projects out every motion that is not the exact mode; `nu_E` is the rate of the kinetic energy, the estimator the previous test used, which counts all of it.

## 2. Domain choice (it was measured, not assumed)

Both alternatives were run and archived at dx = 0.02 m, same initial condition, same window:

| boundary variant | `nu_modal/nu` | peak `|u_y|/U0` | peak surface rise (m) | verdict |
|---|---|---|---|---|
| floor | 0.289 | 1.9e-02 | 2.13e-03 | rejected |
| box | 2.594 | 7.4e-02 | 1.45e-04 | rejected |
| lid (used) | 0.802 | 2.6e-03 | 4.23e-08 | used |

With no lid the fitted rate moves by a factor 0.36 of the lidded value, i.e. the measured decay is about 2.8x slower: the free surface rises, the field stops being the eigenmode, and the number says more about the surface than about the viscosity. With side walls the factor is 3.24 (3.2x faster), because an x-independent velocity field cannot satisfy no-slip at `x = 0` and `x = L`: the initial condition is incompatible with the boundary, and what is fitted is the rate of the adjustment. Only the two-plate lid has both the exact solution and the boundary it assumes. The lateral ends therefore stay open, every metric is taken in a central band, and the residual effect of the open ends is what excludes the finest run below.

## 3. Resolution ladder

| dx (m) | rows | particles | steps | dt (s) | `nu_modal/nu` | `nu_E/nu` | peak `|u_y|/U0` (band) | peak compression | peak Mach | idealisation | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 40 | 4800 | 2653 | 9.43e-05 | **1.1808** | 0.9466 | 2.35e-02 | 4.28 % | 0.047 | **violated** | 61.0 |
| 0.01 | 20 | 1200 | 1326 | 1.89e-04 | **0.8945** | 0.8969 | 3.96e-03 | 0.37 % | 0.008 | ok | 14.7 |
| 0.02 | 10 | 300 | 663 | 3.77e-04 | **0.8019** | 0.8070 | 2.59e-03 | 0.00 % | 0.005 | ok | 4.3 |
| 0.04 | 5 | 75 | 332 | 7.54e-04 | **0.6503** | 0.6612 | 2.75e-03 | 0.00 % | 0.005 | ok | 1.4 |

The idealisation gate requires, in every run used for the convergence claim: `peak_vertical_velocity` <= 0.002, `peak_density_compression` <= 0.01, `peak_mach` <= 0.1, `estimator_agreement` <= 0.1.

Excluded runs, with the reason:

* dx = 0.005 m: peak `|u_y|` in the band 2.4e-03 m/s (2.4e-02 U0) but 3.6e-01 m/s globally, peak compression 4.28 %, and the two estimators disagree by 0.234. Failed gates: flow_is_horizontal, flow_is_incompressible, estimators_agree. The open lateral boundary develops a disturbance at this spacing, which is the same direction as the instability already recorded at dx = 0.006 in `backlog_update_m281.md`.

Profile error against the exact mode, finest usable run:

| t (s) | L2 relative | L-infinity / `U0` |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| 0.0207 | 0.0543 | 0.0583 |
| 0.0415 | 0.0756 | 0.0493 |
| 0.0622 | 0.0997 | 0.0405 |
| 0.0829 | 0.1263 | 0.0336 |
| 0.104 | 0.1165 | 0.0235 |
| 0.124 | 0.1377 | 0.0213 |

The table stops when the exact mode has fallen below 20 % of `U0`: after that the relative error is dominated by motion the exact solution does not contain, and it is not a statement about the viscosity. At t = 0.25 s that residual is 2.1 % of `U0` while the exact mode is down to 4.6 % -- a background motion, not a growing instability, and the reason the modal estimator, which projects it out, is preferred here.

## 4. Controls

| control | measurement | expected | verdict |
|---|---|---|---|
| half the viscosity at dx = 0.02 | rate ratio 0.5114 | 0.5 | pass |
| half the time step at dx = 0.02 | rate ratio 0.9995 | 1.0 | pass |
| `sum m u_x`, inviscid, no walls | relative drift 6.22e-08 | round-off | pass |
| `sum m u_x`, viscous, no walls | relative drift 1.29e-02 | 0 | **not conserved** -- see 7.2 |
| `sum m u_x`, with walls | relative drift 9.12e-01 | not zero: the fluid momentum is transferred to the static walls as the mode decays | expected, not a defect |

The time-step control is what makes the ladder interpretable: `dt` is refined together with `dx` (the acoustic Courant is held fixed), so without it the measured error could be temporal error in disguise. It is not -- halving the step moves the rate by 0.05 %.

## 5. Where the deficit comes from

`nu_inst` is the rate the operator imposes on the exact mode *before any motion*: one right-hand-side evaluation at `t = 0`, projected on the mode. It separates the operator's transfer function from everything that happens during the integration.

| dx (m) | flow `nu_modal/nu` | `nu_inst/nu`, no walls | `nu_inst/nu`, with walls | flow minus frozen (no walls) |
|---|---|---|---|---|
| 0.005 | 1.1808 | 0.9846 | 0.9713 | +0.1963 |
| 0.01 | 0.8945 | 0.9659 | 0.9413 | -0.0714 |
| 0.02 | 0.8019 | 0.9132 | 0.8774 | -0.1113 |
| 0.04 | 0.6503 | 0.7294 | 0.7407 | -0.0792 |

Two separate losses are visible. The instantaneous one is the regularised Laplacian under-resolving the mode, and it converges at better than first order. The remaining loss survives that convergence and is the flow-level part: the no-slip closure and the deformed near-wall profile. Fixing only the Laplacian would therefore not close the gap, which is why the recommendation below names both.

## 6. Convergence of the viscosity error

| dx (m) | `|nu_modal/nu - 1|` | `|nu_E/nu - 1|` |
|---|---|---|
| 0.01 | 0.1055 | 0.1031 |
| 0.02 | 0.1981 | 0.1930 |
| 0.04 | 0.3497 | 0.3388 |

Observed order over the three usable resolutions: **0.86** (modal), 0.86 (energy). A first-order error in a nominally second-order operator is the expected signature of the blended, non-symmetric fluid-wall term.

## 7. Error judgement

| quantity | reference | measured | error | below 3 %? | below 15 %? |
|---|---|---|---|---|---|
| `nu_modal` at dx = 0.01 (20 rows) | 0.05 m^2/s | 0.04472 | -10.6 % | no | yes |
| `nu_modal` at dx = 0.02 (10 rows) | 0.05 m^2/s | 0.04009 | -19.8 % | no | no |
| `nu_modal` at dx = 0.04 (5 rows) | 0.05 m^2/s | 0.03251 | -35.0 % | no | no |

**Is the key quantity below 3 %?** No. The best usable point, dx = 0.01 m (20 rows across the channel), is off by -10.6 %. Below 15 % only from 20 rows upwards. Both answers are measured against the nominal viscosity the case was configured with, with no fitted factor anywhere in the loop: `VISCOUS_LAPLACIAN_FACTOR` is the shipped constant 2.

## 8. Findings

### 8.1 The solver's viscosity is systematically low, first order

The deficit is -35.0 % at 5 rows and -10.6 % at 20. At the resolutions the shipped 2-D cases use, a flow configured at `nu` feels roughly 10-20 % less viscosity, i.e. sits at a Reynolds number 10-25 % too high. Any Re quoted from this solver at those resolutions must carry the correction from this artifact or be re-measured.

### 8.2 The viscous form does not conserve momentum when rho varies

With no walls and no viscosity the pressure term alone keeps `sum m u_x` to 6.2e-08 (round-off), which is the invariant the pair form is supposed to have. Switching the viscosity on leaves a drift of 1.3e-02: the term `m_j/rho_j` is antisymmetric only when `rho_i = rho_j`, so the viscous form injects net momentum wherever the density is not uniform. This is a property of the shipped discretisation, measured here for the first time; it is small in this incompressible case and would not be small in a strongly compressed one.

### 8.3 The previously published no-slip figure mixed two errors

The 0.84-0.91 ratio that stood for the no-slip condition was measured on a slab with a free surface on top. On that class of geometry the flat-lid assumption of the exact solution does not hold -- the archived `floor` variant above, same idea with a full-sine initial condition, moves the fitted rate by a factor 0.36. So the old figure was not a measurement of the viscosity error alone, and the lidded number here is the one that is:

| dx (m) | old geometry (free surface) | lidded, this work |
|---|---|---|
| 0.02 | 0.84 (`tests/test_boundary.py`) | 0.802 |

### 8.4 Recommendation (recorded, not implemented)

1. Quote `nu_eff/nu(dx)` from this artifact in the case definition of every viscous run, or
2. replace the regularised Laplacian with a first-order-consistent one (`tensorsph.csf_consistent` is the precedent already in the tree), and
3. give the no-slip closure the same treatment, since the flow-level residual in section 5 survives the operator's own convergence.

The new P1 row in `docs/algorithm-gap-analysis.md` carries the exit criterion: a usable ladder whose worst `|nu_modal/nu - 1|` is below 3 %.

## 9. Limitations

1. Two dimensions, one viscosity, one geometry, one mode: the deficit is measured for the eigenmode this lattice supports at `k = pi/H`, not for a general turbulent flow.
2. The lateral ends are an artificial open boundary; every metric is taken at least 0.21 m from them, and the residual effect is the reason the dx = 0.005 m run is excluded rather than reported as a data point.
3. The no-slip condition is a blended fluid-wall viscous term with static wall particles, not a geometric constraint; the ladder measures the interior operator and the wall closure together, and section 5 is what separates them.
4. Only the acoustic Courant is held fixed; the viscous and gravity limits are far from binding in this case, so nothing is claimed about them.
5. `nu_E` and `nu_modal` are two readings of the same run, not two independent experiments.

## 10. Reproducing

```bash
python benchmarks/viscous_transport/run.py --resolutions 0.04,0.02,0.01,0.005
python benchmarks/viscous_transport/make_doc.py
pytest tests/test_viscous_transport.py -q
```

Archived run: 109 s wall clock on CPU, 6375 particles over the ladder.

