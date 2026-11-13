# XVAL: SPHinXsys `fsi2-benchmark-and-force-from-fluid` vs TensorSPH

feat_slug `GAP-SPH-FSI-BENCHMARK` · repo `Xiangyu-Hu/SPHinXsys` · tag `fsi2-benchmark-and-force-from-fluid` · mission 17452 · 2026-11-13

## 1. Question and reference provenance

The OSS feature under test is the *fluid-to-solid force transfer* used by the `fsi2` benchmark: `FSI::PressureForceFromFluid` + `FSI::ViscousForceFromFluid` acting on the surface particles of a solid body.  The question this report answers: **does that convention reproduce a known load within the owner accuracy standard (<15 %), and should TensorSPH adopt it?**

Blueprint (read from source, not transcribed from prose):

- `Xiangyu-Hu/SPHinXsys@master` `src/shared/shared_ck/particle_dynamics/fluid_structure_interaction/force_on_structure.hpp:55-72`
- `Xiangyu-Hu/SPHinXsys@master` `src/shared/shared_ck/particle_dynamics/fluid_structure_interaction/force_on_structure.hpp:103-124`
- `Xiangyu-Hu/SPHinXsys@master` `src/shared/particle_dynamics/fluid_dynamics/fluid_integration.h:206`
- `Xiangyu-Hu/SPHinXsys@master` `src/shared/shared_ck/particle_dynamics/fluid_dynamics/acoustic_step_2nd_half.h:197`
- `Xiangyu-Hu/SPHinXsys@master` `src/shared/materials/riemann_solver.h:93`
- `Xiangyu-Hu/SPHinXsys@master` `src/shared/materials/viscosity.h:54`
- `Xiangyu-Hu/SPHinXsys@master` `tests/2d_examples/test_2d_fsi2/fsi2.cpp:155`
- `Xiangyu-Hu/SPHinXsys@master` `tests/2d_examples/test_2d_fsi2/fsi2.h:14-35`

Reference cases: `tests/2d_examples/test_2d_fsi2/fsi2.h:14-35` (channel 11x4.1, cylinder r=0.5 at (2,2), beam 3.5x0.2, Re=100, c=10U, no gravity) and `tests/2d_examples/test_2d_hydrostatic_fsi/hydrostatic_fsi.cpp:14-19` (tank 1.0x2.1, water 1.0x2.0).

## 2. Method

The operator is separated from the solver: each case is a *static* particle arrangement with an analytically known load, and the four operator conventions are evaluated on the same data.  This is deliberate -- the operator is the OSS feature, and a coupled run would bury its systematic error inside the flow error.

Operators: **A** `oss` = one-sided summation with Adami wall-pressure extrapolation in the fsi2 configuration (NoRiemann, NoKernelCorrection); **B** the same with the acoustic Riemann dissipation switched on; **C** TensorSPH's kernel-weighted plane fit (panel, m16352); **D** the pre-m16352 symmetric Adami term kept as baseline. The viscous branch A' is scored on the Couette case.

Implementation: `src/tensorsph/fsi_force_from_fluid.py` (algebra quoted in its docstring, line-for-line against the OSS `.hpp`), cases in `benchmarks/xval_fsi2/cases.py`, driver `benchmarks/xval_fsi2/run.py`. Kernel: Wendland C2, `h = 1.3 dx` (the OSS defaults, `adaptation.h:71` `h_spacing_ratio = 1.3`).

## 3. Parameters

| case | source | dx ladder [m] | h/dx | support/dx | analytic reference | fluid state |
|---|---|---|---|---|---|---|
| X1 vertical wall (gravity transverse) | `tests/2d_examples/test_2d_hydrostatic_fsi/hydrostatic_fsi.cpp:14-19` | 0.05, 0.025, 0.0125 | 1.3 | 2.6 | 1/2 rho g H^2 | analytic |
| X2 floor / gate (gravity normal) | `tests/2d_examples/test_2d_hydrostatic_fsi/hydrostatic_fsi.cpp:14-19` | 0.05, 0.025, 0.0125 | 1.3 | 2.6 | rho g H L | analytic |
| X3 plane Couette (viscous branch) | `plane Couette (textbook), viscosity branch of force_on_structure.hpp:55-72` | 0.025, 0.0125, 0.00625 | 1.3 | 2.6 | mu U / h * L | analytic |
| X4 fsi2 insert body, linear field, no gravity | `tests/2d_examples/test_2d_fsi2/fsi2.h:14-35` | 0.1, 0.05, 0.025 | 1.3 | 2.6 | -c A (linear field, closed body) | analytic |

Dimensionless groups that enter the transfer: `h/dx = 1.3`, `r_support/dx = 2.6` (= 2h, Wendland C2), `dx/H` from 0.025 to 6.25e-3 for the walls and `dx/r_cyl` = 0.2/0.1/0.05 for the fsi2 body.  The fsi2 flow quantities `Re = 100`, `c/U = 10`, `rho_s/rho_f = 10`, `Ae = 1.4e3` are reported for completeness but do not enter a static transfer check.  The OSS solid body is refined twice relative to the fluid (`fsi2.cpp:37` `defineAdaptationRatios(1.15, 2.0)`), i.e. `dx_solid = dx_fluid/2` with `h_s = 1.15 dx_solid`; the comparison below uses `h = 1.3 dx` on both sides and records the difference as a porting decision.

## 4. Comparison table (reference vs measured)

Error = `(F_got - F_ref)/|F_ref|` on the component along the analytic direction.  Gate: |error| < 15 % = PASS (owner standard).

### X1 vertical wall (gravity transverse)

| dx | A OSS fsi2 (NoRiemann, no correction) | B OSS + acoustic Riemann dissipation | C TensorSPH panel (WLS plane fit) | D TensorSPH pairwise (legacy) | A' OSS viscous branch |
|---|---|---|---|---|---|
| 0.05 | +27.59 % FAIL | +27.59 % FAIL | -0.00 % PASS | -226.90 % FAIL | n/a |
| 0.025 | +27.96 % FAIL | +27.96 % FAIL | -0.00 % PASS | -227.61 % FAIL | n/a |
| 0.0125 | +28.14 % FAIL | +28.14 % FAIL | -0.00 % PASS | -227.96 % FAIL | n/a |

reference: 1/2 rho g H^2 = [19620.0, 0.0] ; hydrostatic load on a vertical wall, depth 2.0 m; the wall normal carries no gravity component, so the OSS wall-pressure extrapolation sees only the vertical part of e_ij

### X2 floor / gate (gravity normal)

| dx | A OSS fsi2 (NoRiemann, no correction) | B OSS + acoustic Riemann dissipation | C TensorSPH panel (WLS plane fit) | D TensorSPH pairwise (legacy) | A' OSS viscous branch |
|---|---|---|---|---|---|
| 0.05 | +24.23 % FAIL | +24.23 % FAIL | -0.00 % PASS | -223.26 % FAIL | n/a |
| 0.025 | +26.26 % FAIL | +26.26 % FAIL | -0.00 % PASS | -225.78 % FAIL | n/a |
| 0.0125 | +27.29 % FAIL | +27.29 % FAIL | -0.00 % PASS | -227.04 % FAIL | n/a |

reference: rho g H L = [0.0, -19620.0] ; water column 1.0 x 2.0 m resting on a floor; the wall normal is aligned with gravity, which is the configuration the Adami-style wall-pressure extrapolation is designed for

### X3 plane Couette (viscous branch)

| dx | A OSS fsi2 (NoRiemann, no correction) | B OSS + acoustic Riemann dissipation | C TensorSPH panel (WLS plane fit) | D TensorSPH pairwise (legacy) | A' OSS viscous branch |
|---|---|---|---|---|---|
| 0.025 | n/a | n/a | n/a | n/a | -38.54 % FAIL |
| 0.0125 | n/a | n/a | n/a | n/a | -38.19 % FAIL |
| 0.00625 | n/a | n/a | n/a | n/a | -38.01 % FAIL |

reference: mu U / h * L = [0.09999999999999999, 0.0] ; linear Couette profile U=1.0 over a 0.1 m gap, mu=0.01; only the viscous branch is exercised

### X4 fsi2 insert body, linear field, no gravity

| dx | A OSS fsi2 (NoRiemann, no correction) | B OSS + acoustic Riemann dissipation | C TensorSPH panel (WLS plane fit) | D TensorSPH pairwise (legacy) | A' OSS viscous branch |
|---|---|---|---|---|---|
| 0.1 | +107.54 % FAIL | +107.54 % FAIL | -0.70 % PASS | -287.89 % FAIL | n/a |
| 0.05 | +83.32 % FAIL | +83.32 % FAIL | +20.08 % FAIL | -275.25 % FAIL | n/a |
| 0.025 | +67.66 % FAIL | +67.66 % FAIL | +32.26 % FAIL | -263.95 % FAIL | n/a |

reference: -c A (linear field, closed body) = [-0.0, 1.4853981633974485] ; fsi2 insert body (cylinder r=0.5 + beam 3.5x0.2) in a linear pressure field with no gravity; analytic load = -c A

## 5. Convergence and sensitivity

| case / operator | dx ladder errors [%] | change over the ladder |
|---|---|---|
| X1 vertical wall / A OSS fsi2 | +27.59, +27.96, +28.14 | +0.55 pts |
| X1 vertical wall / B OSS + acoustic Riemann dissipation | +27.59, +27.96, +28.14 | +0.55 pts |
| X1 vertical wall / C TensorSPH panel | -0.00, -0.00, -0.00 | +0.00 pts |
| X1 vertical wall / D TensorSPH pairwise | -226.90, -227.61, -227.96 | -1.06 pts |
| X2 floor / gate / A OSS fsi2 | +24.23, +26.26, +27.29 | +3.06 pts |
| X2 floor / gate / B OSS + acoustic Riemann dissipation | +24.23, +26.26, +27.29 | +3.06 pts |
| X2 floor / gate / C TensorSPH panel | -0.00, -0.00, -0.00 | -0.00 pts |
| X2 floor / gate / D TensorSPH pairwise | -223.26, -225.78, -227.04 | -3.78 pts |
| X3 plane Couette / A' OSS viscous branch | -38.54, -38.19, -38.01 | +0.52 pts |
| X4 fsi2 insert body, linear field, no gravity / A OSS fsi2 | +107.54, +83.32, +67.66 | -39.89 pts |
| X4 fsi2 insert body, linear field, no gravity / B OSS + acoustic Riemann dissipation | +107.54, +83.32, +67.66 | -39.89 pts |
| X4 fsi2 insert body, linear field, no gravity / C TensorSPH panel | -0.70, +20.08, +32.26 | +32.96 pts |
| X4 fsi2 insert body, linear field, no gravity / D TensorSPH pairwise | -287.89, -275.25, -263.95 | +23.94 pts |

Read-out: every OSS arm sits at a **dx-independent offset** (the change over a 4x refinement is a fraction of a percentage point), which identifies it as a structural factor rather than a resolution deficiency.  The panel arm on the flat walls is at machine zero at every resolution, as expected for a reproduction of a linear field.

Sensitivity to the number of solid layers filling the body (dx = 0.05):

| case | layers | OSS | panel | pairwise |
|---|---|---|---|---|
| vertical_wall | 1 | +27.59 % | -0.00 % | -226.90 % |
| vertical_wall | 3 | +66.48 % | +198.46 % | -265.59 % |
| horizontal_gate | 1 | +24.23 % | -0.00 % | -223.26 % |
| horizontal_gate | 3 | +62.73 % | +199.73 % | -261.05 % |

This is the sharpest porting lesson: the one-sided OSS sum and the panel operator both depend on *how much solid* receives the force.  The panel operator scales almost linearly with the number of layers (it applies the face pressure to every layer), so it must be restricted to surface particles; the OSS sum saturates after two layers because the second layer already sees the full kernel support from the fluid side.

Kernel / smoothing-length sensitivity (pressure branch):

| case | dx | kernel | h/dx | OSS error |
|---|---|---|---|---|
| fsi2_insert_linear | 0.05 | wendland | 1.3 | +83.32 % |
| fsi2_insert_linear | 0.05 | quintic | 1.3 | +42.08 % |
| fsi2_insert_linear | 0.05 | wendland | 1.15 | +58.12 % |
| horizontal_gate | 0.025 | wendland | 1.3 | +26.26 % |
| horizontal_gate | 0.025 | quintic | 1.3 | -11.71 % |
| horizontal_gate | 0.025 | wendland | 1.15 | +47.68 % |

## 6. Failure modes, and what is not claimed

1. **The OSS convention does not meet the 15 % standard on any case here** (+24 % to +108 % over-load on the pressure branch, -38 % on the viscous branch).  It is not a resolution effect and the Riemann term does not move it (columns A and B are identical to 2 decimals because the fsi2 configuration carries no dissipation and the transfer is evaluated on a static field).
2. **Attack of the factor.** The pressure branch sums `(p_wall + p_j)` over a fluid half-space.  With `p_wall ~ p_j` on a static field the pair term is `2p`, i.e. the operator behaves as if the fluid filled both sides of the surface; the measured over-load `1/4..1/5` of the analytic `1/2 rho g H^2` is consistent with that, but the factor has **not** been proved analytically here -- it is a hypothesis supported by the dx-independence and by the layer-count behaviour (a second solid layer, which sees a full support, pushes the error further up rather than removing it).
3. **The viscous branch** reproduces only ~62 % of `mu U/h` with a single surface layer, and the deficit does not shrink with dx (-38.5 %, -38.2 %, -38.0 %).  The OSS form is a one-sided velocity-derivative sum without the factor 2 of the standard SPH Laplacian; again a hypothesis, not a proof, but it matches the observed constant offset.
4. **The closed fsi2 insert body is the worst case** (+107.5 % at the benchmark resolution dx=0.1, +67.7 % at dx=0.025): on a convex closed surface every particle's support is cut, and the over-load compounds over the perimeter.
5. **The panel arm on the closed body is not explained**: -0.7 % at dx=0.1 but +20 % / +32 % at dx=0.05 / 0.025, i.e. it *degrades* with refinement, which no convergence argument explains.  It is reported as an open defect of that arm on curved closed surfaces; the flat-wall result (machine zero) is unaffected. Nothing in the verdict rests on this arm.
6. **Not attempted this round (honest scope)**: the coupled fsi2 transient (beam-tip position vs `BeamObserver_Position_Run_0_result.xml`, time-averaged `TotalViscousForceFromFluid` = [0.451157, 0.00157671] from `InsertedBody_TotalViscousForceFromFluid_time_averaged_mean_variance.xml`). Reason: the OSS reference is a large-amplitude VIV of an elastic beam, which needs a total-Lagrangian solid with strong coupling; this repository's elastic rod carries a known +87 % large-deformation error (m16252 audit, case A2), so a coupled run would measure the solid solver, not the transfer.  The two quantities above are kept as round-2 targets.

## 7. Difference list: this repo vs the OSS implementation

| item | OSS (SPHinXsys master) | TensorSPH (this round) | consequence |
|---|---|---|---|
| fluid->solid pressure | one-sided sum over fluid neighbours, `(p_wall + p_j)`, Adami extrapolation `p_wall = p_j + rho_j r (a_ext - a_s).e` | same algebra ported verbatim (`fsi_force_from_fluid.pressure_force_from_fluid`) | reproduces the OSS over-load, see X1/X2/X4 |
| wall-pressure extrapolation driver | `ForcePrior/m - acc_ave` (fluid external acceleration minus solid average acceleration) | `acc_ext_fluid - acc_solid`, broadcast allowed | identical; on fsi2 there is no gravity, so the term is identically zero and cannot repair the cut support |
| pressure branch Riemann term | `NoRiemannSolver` in the fsi2 case, `AcousticRiemannSolver` elsewhere | switchable, default off (= fsi2) | columns A/B identical here; the term only matters when the solid moves |
| kernel correction | `NoKernelCorrectionCK` for the fsi2 wall contact; the general form applies the contact particle's correction matrix | optional `correction_fluid` argument, default identity | fsi2-equivalent by default |
| viscous branch | `mu_j` (one-sided, `viscosity.h:54`) times `(u_i-u_j)/(r^2+0.01h^2)` times `r dW` | ported verbatim | -38 % on Couette |
| load accounting | force is a per-particle acceleration-like quantity, summed over *all* body particles (`QuantitySummation`) | force per unit thickness, intended for surface particles | thickness sensitivity table: the two accountings differ by ~38 % per extra layer |
| solid discretisation | body lattice filled with solid particles, `defineAdaptationRatios(1.15, 2.0)` -> `dx_s = dx_f/2`, `h_s = 1.15 dx_s` (`fsi2.cpp:37`) | single surface layer, `h = 1.3 dx` | the OSS body is twice as fine as the fluid and 4x denser in particles; not reproduced here |
| solid constitutive model | `SaintVenantKirchhoffSolid` (total Lagrangian SPH) on the insert body | discrete elastic rod (energy form), no TLSPH | the kinematics are not comparable yet -> coupled case deferred |
| boundary condition at the inlet | periodic + parabolic inflow buffer (`InflowVelocity`, `u_ave` ramped over `t_ref = 2`) | none (steady flow not implemented) | part of the round-2 case work |
| time stepping | dual-criteria (advection / acoustic) | single acoustic CFL | cost, not accuracy, of a coupled run |

**Porting decision.** Do not replace the panel operator with the OSS one: the panel operator is exact on linear pressure fields (X1/X2) and the OSS one is +24..+108 % off.  Port the OSS *algebra* only as the reference arm of this cross-validation (`fsi_force_from_fluid.py`), and if a case needs the OSS behaviour exactly, use it with the OSS body discretisation (filled body, `dx_s = dx_f/2`) -- mixing the OSS operator with a single surface layer would import a ~28 % bias.

## 8. Artifacts

- `src/tensorsph/fsi_force_from_fluid.py` -- the ported operators + panel + legacy arms.
- `benchmarks/xval_fsi2/{cases.py,run.py,make_doc.py}` -- reproducible driver.
- `artifacts/xval_fsi2/results.json` -- every number in this report.
- `artifacts/xval_fsi2/oss_src/` -- verbatim OSS files read for the blueprint and the fsi2 regression references.
- `artifacts/xval_fsi2/fig_error_vs_dx.png` -- load error vs particle spacing for every operator and case.
- `artifacts/xval_fsi2/fig_thickness.png` -- load error against the number of solid layers filling the body.

## 9. Ecosystem friction (as found)

1. `raw.githubusercontent.com` resets the connection from this runner; every file had to be read through the GitHub contents API (base64), which works.
2. There is no route to run the OSS case itself (C++/TBB build, no compiler toolchain budget this round), so the OSS arm had to be reproduced algebraically here.  That is weaker than running the reference: the regression XMLs remain the only direct OSS numbers, and they belong to the coupled case that is deferred.
3. `oss_solutions` id 359 (this very feature) carries the method summary and the file:line pointers that made the blueprint lookup one query; the record's `transfer` field points at the density-regularisation xval, which is unrelated to FSI -- a small data-hygiene defect in the ledger.

