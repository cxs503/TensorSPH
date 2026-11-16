# Coupled FSI2 driver: defect ledger and added-mass verdict

**Anchor**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID` (mission 20552, branch `agent/m20552`)
**Module**: `src/tensorsph/fsi2_turek_hron.py` · **Tests**: `tests/test_fsi2_coupled_fixing.py` (12 tests)
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

## 2. Defect ledger (all reproduced, all fixed; D4's injection *clock* was later replaced -- see §6)

| id | defect | evidence | fix |
|---|---|---|---|
| D1 | `interpolation_matrix_beam` mapped stations onto the **node index space [0,1]** instead of `[0, n_nodes-1]`, so all 17 boundary particles were squeezed between node 0 and node 1 (12 mm of a 350 mm beam, 0.04 dx apart). Overlapping wall particles inflated the local density and pressure as above. | `pos[:,0].max()-min` = 0.012 m before, 0.350 m after | station `xi = k (n_nodes-1)/(n_b-1)` |
| D2 | the beam was a **single row of boundary particles on the centreline**, so both faces saw the same row; their loads averaged into one normal and the net transverse load collapsed. The benchmark beam is a plate of thickness 20 mm. | two-face force decomposition test | `beam_boundary_geometry`: one row per face at +-t/2 with its own outward normal, both lumped to the rod nodes |
| D3 | fluid particles were initialised **on the beam surface** (`> t/2` strict), i.e. zero-distance wall neighbours. t/2 = 0.01 m is a multiple of every dyadic dx, so the coincidence is generic, not a corner case. | min fluid-beam distance 0.0 before, >= 1.5 dx after | exclusion band `|dy| <= t/2 + dx/2 (+1e-9)`, x-range widened by dx/2 |
| D4 | the channel is treated as a **closed tank** (positions clamped at both ends): no inflow, no outflow, so the benchmark's through-flow cannot exist. | `min x` grows monotonically, particles pile at x = L | `open_boundaries` (default on): inject an inlet column with the parabolic profile, delete particles at the outlet, impose the inlet velocity in a 2 dx buffer, cosine start-up ramp. The injection *clock* is per-streamline flux (`sum(u(y_j) dt) >= dx`); the first version of this fix injected on a geometric trigger (`min x > 1.5 dx`) and stalled -- see `INFLOW_CLOCK_STALL.md` and §6 |
| D5 | load transfer used the **Wendland C2 kernel** while the solver uses a cubic spline (the +24..28 % overload of `PRESSURE_MOMENT_NORMALISATION.md`) | settled in that note | `boundary_pressure_from_fluid` now uses `di.cubic_spline_kernel` |

Invariants kept by the tests: a *uniform* pressure field on the symmetric beam produces a
net load < 1e-9 N/m (it was 2.1e8), each face carries `p * trib * M` individually, the two
rows stay exactly `t` apart along the normal, and the inlet column carries a non-uniform
(parabolic) velocity.

## 3. The blocking result, *as recorded in m20552*: added-mass instability at the FSI2 density ratio

> **Superseded in part -- read §6 first.** The ratio-1 "divergence" below was measured with a
> rod sub-cycling count sized on the *axial* frequency; the binding mode was the highest
> *bending* mode, which was under-resolved (fixed in m20761,
> `STRONG_COUPLING_AND_ROD_STABILITY.md`). The quadrant of the table that says
> `rho_s/rho_f = 1 DIVERGENT` therefore measures a structural-integration defect, not
> added-mass instability, and the numbers are kept here only as the before-picture. §6
> re-runs the same predeclared experiment on the current tree.

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

## 6. Re-measurement on the current tree (m22712)

The m20552 record and the asset `assets/coupling_stability.json` had drifted apart
(`assets/` said `n_steps = 100`, the committed runner defaults to 600 and writes a
`resolution_check` block that the asset did not have), and both predate the m20761 rod
sub-cycling fix and the per-streamline inlet clock. The experiment is therefore re-run
from the runner, unchanged in its predeclared decision, metric and thresholds, and the
table below is **generated** by `benchmarks/xval_fsi2/run_coupling_stability.py`
(`table_markdown`) straight from `assets/coupling_stability.json` -- no number in this
section is typed by hand.

Two controls are added, both of which only *classify* the measurement, they do not move
the thresholds:

* a `dx = 12.5 mm` confirmation of the ratio-1 verdict, so a coarse-lattice artefact
  cannot masquerade as a stability result;
* the same ratio-1 point with `open_boundaries = False`, i.e. the old closed-tank channel,
  to show the verdict does not hinge on the through-flow treatment (D4).

`out/fsi2_added_mass.json` and `out/fsi2_impedance.json` are the m20552 one-off records
(`dt = 4.33e-5 s`, no open-boundary field, ad-hoc runners that are not in the tree). They
are kept as the before-picture and are superseded by the table below.

<!-- BEGIN GENERATED: coupling_stability -->
| variant | rho_s/rho_f | sigma [1/s] | |v_beam|max, step 20 [m/s] | |v_beam|max, end [m/s] | verdict |
|---|---|---|---|---|---|
| loose | 1 | +10.1 | 4.19e-04 | 1.84e-01 | STABLE |
| loose | 2 | +10.2 | 2.88e-04 | 1.33e-01 | STABLE |
| loose | 5 | +10.3 | 1.70e-04 | 8.74e-02 | STABLE |
| loose | 10 | +9.8 | 1.10e-04 | 4.16e-02 | STABLE |
| impedance | 1 | +17.6 | 4.19e-04 | 1.70e+01 | DIVERGENT |
| impedance | 2 | +8.3 | 2.88e-04 | 4.39e-02 | STABLE |
| impedance | 5 | +8.6 | 1.70e-04 | 3.14e-02 | STABLE |
| impedance | 10 | +9.4 | 1.10e-04 | 3.13e-02 | STABLE |
| relaxed | 1 | +14.0 | 2.03e-04 | 9.42e-01 | DIVERGENT |
| relaxed | 2 | +12.1 | 1.36e-04 | 2.05e-01 | STABLE |
| relaxed | 5 | +11.3 | 7.79e-05 | 6.98e-02 | STABLE |
| relaxed | 10 | +10.5 | 4.94e-05 | 2.83e-02 | STABLE |
| loose (open_boundaries=False) | 1 | +8.3 | 4.19e-04 | 6.44e-02 | STABLE |
| loose (dx=0.0125 m confirmation) | 1 | +42.0 | n/a | 1.71e-01 | STABLE |

Findings (derived from the rows above):
* `rho_s/rho_f = 1`: **DIVERGENT** in impedance, relaxed (end speeds: loose 1.84e-01 m/s, impedance 1.70e+01 m/s, relaxed 9.42e-01 m/s).
* `rho_s/rho_f = 2`: stable in every variant (end speeds: loose 1.33e-01 m/s, impedance 4.39e-02 m/s, relaxed 2.05e-01 m/s).
* `rho_s/rho_f = 5`: stable in every variant (end speeds: loose 8.74e-02 m/s, impedance 3.14e-02 m/s, relaxed 6.98e-02 m/s).
* `rho_s/rho_f = 10`: stable in every variant (end speeds: loose 4.16e-02 m/s, impedance 3.13e-02 m/s, relaxed 2.83e-02 m/s).

Generated by `benchmarks/xval_fsi2/run_coupling_stability.py` at 600 steps, dt = 1.040e-03 s (0.624 s of physical time), dx = 0.02 m, c0 = 5 m/s, U_mean = 0.2 m/s, auto rod sub-steps = 11.  Divergence threshold (predeclared): sigma > 0 and |v_beam|max > 0.3 m/s.
<!-- END GENERATED: coupling_stability -->

**Reading of the table** (qualitative; the numbers live in the generated block above):
the unmodified loose driver now stays below the predeclared divergence limit at *every*
density ratio including the benchmark's `rho_s/rho_f = 1`, so the m20552 blocker -- "the
coupled run cannot be started at the FSI2 point" -- no longer reproduces on the current
tree: the ratio-1 blow-up recorded in §3 was the axial-only rod sub-step count (m20761),
not the fluid coupling. What still fails at ratio 1 is the *modified* load transfer: the
correctly signed acoustic impedance arm and the 5 % load under-relaxation arm both cross
the limit there and are stable at ratio >= 2, i.e. the added-mass sensitivity the anchor
was about is real but confined to those variants (`STRONG_COUPLING_AND_ROD_STABILITY.md`,
`docs/fsi2_impedance_sign/01_report.md`). Neither control changes the loose ratio-1
verdict: not the finer `dx = 12.5 mm` lattice, not the closed-tank channel.

Two honest caveats. First, `sigma > 0` on every row means the beam speed is still
*rising* over the 0.624 s window (drag/VIV build-up, as the predeclared metric
anticipated), so "stable" here means "under the predeclared limit at the end of the
window", not "a converged VIV state". Second, the end speed of the loose ratio-1 arm is
an order of magnitude above the 0.03 m/s drag-acceleration scale quoted in the runner
docstring, so the margin to the limit is not large; a longer window would be the next
test of the same hypothesis.
