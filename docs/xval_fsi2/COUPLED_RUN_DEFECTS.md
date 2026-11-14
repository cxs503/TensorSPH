# Coupled FSI2 driver: defect ledger and added-mass verdict

**Anchor**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID` (mission 20552, branch `agent/m20552`)
**Module**: `src/tensorsph/fsi2_turek_hron.py` · **Tests**: `tests/test_fsi2_coupled_fixing.py` (11 tests)
**Experiment**: `benchmarks/xval_fsi2/run_coupling_stability.py` → `assets/coupling_stability.json`

## 1. Why this round exists

Every earlier round on this anchor verified *pieces*: V1-V4 geometry/stiffness checks,
the force-from-fluid operators on static particle sets, the similarity groups, and the
verdict that the archived SPHinXsys records are not the benchmark point. None of them
integrated the coupled system. The first coupled integration (`step_fsi2` / `run_fsi2`,
present in the module since m19450) explodes **on the first step**:

| step-0 quantity | before | physical scale |
|---|---|---|
| max fluid speed | 3777 m/s | 0.3 m/s |
| max density | 3494 kg/m3 | 1000 kg/m3 |
| max pressure | 8.2e8 Pa | ~1e2 Pa |
| beam load (net) | 2.1e8 N/m | ~6 N/m |

## 2. Defect ledger (all reproduced, all fixed except D4)

| id | defect | evidence | fix |
|---|---|---|---|
| D1 | `interpolation_matrix_beam` mapped stations onto the **node index space [0,1]** instead of `[0, n_nodes-1]`, so all 17 boundary particles were squeezed between node 0 and node 1 (12 mm of a 350 mm beam, 0.04 dx apart). Overlapping wall particles inflated the local density and pressure as above. | `pos[:,0].max()-min` = 0.012 m before, 0.350 m after | station `xi = k (n_nodes-1)/(n_b-1)` |
| D2 | the beam was a **single row of boundary particles on the centreline**, so both faces saw the same row; their loads averaged into one normal and the net transverse load collapsed. The benchmark beam is a plate of thickness 20 mm. | two-face force decomposition test | `beam_boundary_geometry`: one row per face at +-t/2 with its own outward normal, both lumped to the rod nodes |
| D3 | fluid particles were initialised **on the beam surface** (`> t/2` strict), i.e. zero-distance wall neighbours. t/2 = 0.01 m is a multiple of every dyadic dx, so the coincidence is generic, not a corner case. | min fluid-beam distance 0.0 before, >= 1.5 dx after | exclusion band `|dy| <= t/2 + dx/2 (+1e-9)`, x-range widened by dx/2 |
| D4 | the channel is treated as a **closed tank** (positions clamped at both ends): no inflow, no outflow, so the benchmark's through-flow cannot exist. | `min x` grows monotonically, particles pile at x = L | `open_boundaries` (default on): inject an inlet column with the parabolic profile once the inlet has drifted one lattice spacing, delete particles at the outlet, impose the inlet velocity in a 2 dx buffer, cosine start-up ramp |
| D5 | load transfer used the **Wendland C2 kernel** while the solver uses a cubic spline (the +24..28 % overload of `PRESSURE_MOMENT_NORMALISATION.md`) | settled in that note | `boundary_pressure_from_fluid` now uses `di.cubic_spline_kernel` |

Invariants kept by the tests: a *uniform* pressure field on the symmetric beam produces a
net load < 1e-9 N/m (it was 2.1e8), each face carries `p * trib * M` individually, the two
rows stay exactly `t` apart along the normal, and the inlet column carries a non-uniform
(parabolic) velocity.

## 3. The blocking result: added-mass instability at the FSI2 density ratio

**Decision**: can the *loosely coupled* explicit driver (one fluid solve + one explicit rod
step per dt) produce the Turek-Hron FSI2 numbers?
**Alternatives**: loose; with the acoustic (Riemann) wall pressure `p* = p_f - rho c (v_w - v_f).n`;
with nodal-load under-relaxation alpha = 0.05.
**Manipulated variable**: solid/fluid density ratio (FSI2 is posed at **1**; the SPHinXsys
regression point is 10). Everything else fixed (dx = 20 mm, dt = 1.73e-4 s, c0 = 5 m/s,
U_mean = 0.2 m/s, Re = 100, Ae = 5600, cubic-spline kernel, same particle set).
**Metric (predeclared)**: `sigma = d ln|v_beam|max / dt` after step 20; divergent iff
`sigma > 0` **and** `|v_beam|max > 0.3 m/s` (the drag-acceleration scale is 0.03 m/s after
17 ms, so a run merely responding to the load must stay below it).

| loose | 1 | +1397.3 | 1.15e+00 | 3.53e+50 | DIVERGENT |
| loose | 2 | +143.1 | 3.84e-07 | 5.67e-02 | stable |
| loose | 10 | +149.9 | 8.48e-08 | 2.22e-02 | stable |
| impedance | 1 | +1395.8 | 1.15e+00 | 3.13e+50 | DIVERGENT |
| impedance | 2 | +213.2 | 2.48e-06 | 1.26e+02 | DIVERGENT |
| impedance | 10 | +182.2 | 2.33e-07 | 8.94e-01 | DIVERGENT |
| relaxed | 1 | +1397.3 | 1.15e+00 | 3.53e+50 | DIVERGENT |
| relaxed | 2 | +166.4 | 4.16e-08 | 4.27e-02 | stable |
| relaxed | 10 | +168.1 | 9.19e-09 | 1.09e-02 | stable |

Result: `rho_s/rho_f = 1` blows up in every variant (beam speed 1e50 m/s within 100 steps,
1.7e-2 s of physical time), while 2 and 10 stay bounded and simply build up the drag load.
Under-relaxation changes nothing (identical sigma) and the *correctly signed* Riemann
impedance makes the explicit coupling **worse** at ratio >= 2 (v_end 125 m/s at ratio 2).
This is the classical added-mass instability of partitioned explicit FSI in the limit
`m_added ~ m_structural`; no load filter can cure it, because there is no coupling
iteration to converge.

Figures: `assets/fig_beam_boundary_defect.png` (D1), `assets/fig_added_mass_divergence.png`
(beam speed, ratio 1 vs 10).

## 4. Error judgement against the benchmark

| quantity | benchmark | this round | <15 %? | <3 %? |
|---|---|---|---|---|
| tip y amplitude | 0.0082 m | **not run** (D4/D6 block the coupled run) | not assessable | not assessable |
| tip x amplitude | 0.0028 m | **not run** | not assessable | not assessable |
| drag / lift coefficient | 3.2 / 0.22 | **not run** | not assessable | not assessable |
| beam boundary stations | 0.35 m span | 0.350 m (+0.0 %) | yes | yes |
| net load, uniform pressure | 0 | 8e-14 N/m | yes | yes |
| subdivision limit | stable at ratio 1 | **FAIL** (DIVERGENT) | no | no |

**The anchor's <15 % acceptance is NOT met.** The coupled benchmark number is `WITHHELD`:
the driver cannot run the parameter point at all. This is an honest FAIL, recorded as such.

## 5. Next step (what the next round must build)

1. Implicit or added-mass-aware coupling: sub-iterate the fluid/structure pair inside each
   time step (Aitken/Anderson relaxation) or add the added-mass operator
   (Causin-Gerbeau-Nobile) to the rod's effective mass. Budget: 3-10 RHS evaluations per
   step at dx = 10 mm, ~1e4 fluid particles.
2. Only then measure the tip amplitudes and compare with `fsi2_reference.BENCHMARK_TIP`.
3. Drag/lift still need a primary source (the archived records publish no pressure force).
