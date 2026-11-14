# FSI2 at the benchmark density ratio: the added-mass verdict was a rod sub-stepping defect

**Anchor**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID` · **Mission**: m20761 · **Branch**: `agent/m20761`
**Modules**: `src/tensorsph/fsi2_turek_hron.py` (criterion fix), `src/tensorsph/fsi2_strong_coupling.py` (new)
**Tests**: `tests/test_fsi2_strong_coupling.py` (15), `tests/test_fsi2_coupled_fixing.py` (verdict revised)
**Data**: `docs/xval_fsi2/assets/strong_coupling_data.json` · **Figure**: `docs/xval_fsi2/assets/fig_rod_substeps_and_coupling.png`

## 1. Case introduction

The Turek-Hron FSI2 benchmark (LNCSE 53, 2006) is a 2-D channel
(2.5 m x 0.41 m) with a parabolic inlet past a rigid cylinder (D = 0.1 m at
(0.2, 0.2) m) carrying an elastic beam (L = 0.35 m, t = 0.02 m,
E = 5.6 MPa, `rho_s = rho_f = 1000 kg/m^3`, Re = 100). The beam vibrates under
vortex shedding; the acceptance quantities are the beam-tip displacement
amplitudes (reference: tip x 0.0028 m, tip y 0.0082 m) and drag/lift.

The repository state inherited by this round was *blocking*: the coupled driver
was reported to diverge at the benchmark density ratio `rho_s/rho_f = 1`
(`docs/xval_fsi2/COUPLED_RUN_DEFECTS.md`, mission 20552: beam speed 3.5e50 m/s
inside 100 steps, "classical added-mass instability, no load filter can cure
it"). Therefore only the *static* operators had ever been verified (V1-V4,
force-from-fluid) and the coupled benchmark never run.

All runs below use the repository's similarity-corrected case
(`tensorsph.fsi2_similarity.similarity_corrected_case`, `U_mean = 0.2 m/s`,
`E = 2.24e5 Pa`) so that `Re = 100`, `Ae = 5600`, `rho_s/rho_f = 1` and
`L/t = 17.5` match the benchmark while the SPH Mach number stays small;
`dx = 20 mm`, `c0 = 5 m/s`, `dt = 1.04e-3 s`, 2559 fluid particles,
30 rod nodes, cubic-spline kernel.

## 2. What was done

1. **Isolated the rod from the coupling.** A clamped beam was perturbed by
   1e-5 m and integrated with *zero* load: no fluid, no coupling, only
   `beam_advance` and the sub-step count the case chooses for itself.
2. **Derived a criterion for the sub-step count.** `omega_max` is taken as the
   highest eigenfrequency of the discrete energy Hessian at the straight
   configuration, mass-normalised (`sqrt(lambda_max(H_free)/m_node)`), so both
   the stretch and the bending terms enter; the sub-step count is
   `ceil(dt*omega_max/rod_stability_target)` with `rod_stability_target = 0.25`.
   The old criterion used the **axial** frequency only.
3. **Re-ran the coupling experiment** at `rho_s/rho_f = 1` with the corrected
   sub-cycling (loose driver, impedance off/on).
4. **Built a strong-coupling stepper** (`step_fsi2_strong`) with two interface
   accelerators (Aitken dynamic relaxation, IQN-ILS) and validated the
   accelerators against a synthetic interface map with a known gain
   (1.5 / 4.35 / 20) — they must cancel it exactly. This was intended as the
   cure for the added-mass instability; after step 1-3 it is no longer needed
   to run the case, and is kept as a verified, tested tool.

## 3. Results

### 3.1 The rod integrator is what blows up at `rho_s/rho_f = 1`

Free beam, zero load, 200 steps, `dx = 20 mm`:

| density ratio | `omega_max` [rad/s] | `dt*omega_max` | auto sub-steps | energy ratio E/E0 |
|---|---|---|---|---|
| 1 | 2476.7 | **2.58** | 11 (new) | 0.773 |
| 2 | 1751.3 | 1.82 | 8 (new) | 0.520 |
| 10 | 783.2 | 0.81 | 4 (new) | 0.426 |

and at ratio 1 with the sub-step count forced (same case):

| sub-steps | E/E0 after 200 zero-load steps |
|---|---|
| 1 (auto before m20761) | **9.36e+249** |
| 2 | 7.29e-01 |
| 4 | 7.76e-01 |
| 8 | 7.74e-01 |
| 11 (auto now) | 7.73e-01 |

The empirical flip between `dt*omega_max = 1.82` (bounded) and `2.58` (bounded
growth of 1e249) brackets the symplectic-Euler limit `dt*omega_max < 2`, so the
derived criterion is not tuned: the analytic boundary falls inside the measured
interval. At ratio 1 the old criterion produced 1 sub-step where the bending
mode needs more than one; at ratios 2 and 10 the same old criterion happened to
be sufficient — **which is exactly the pattern that had been read as an
added-mass effect.**

### 3.2 Coupled run at the benchmark density ratio (`rho_s/rho_f = 1`)

Loose driver, 200 steps (0.21 s physical), `dx = 20 mm`:

| configuration | sub-steps | outcome | max\|v_beam\| at the end |
|---|---|---|---|
| axial-only sub-cycling (pre-m20761) | 1 | diverges, step 26 | 1.75e+03 m/s |
| bending-aware sub-cycling (m20761) | 11 | **bounded** | 9.0e-02 m/s |
| bending-aware + acoustic wall impedance | 11 | diverges, step 156 | 1.36e+03 m/s |

Figure: `assets/fig_rod_substeps_and_coupling.png`, panel (a) rod energy vs
sub-steps, panel (b) the three beam-speed histories.

A 600-step (0.62 s) confirmation of the surviving configuration gives a bounded,
physical beam response: max|v_beam| = 4e-2..1.2e-1 m/s, tip_y oscillating
between -1e-5 m and -1.8e-2 m, drag O(1) N/m, fluid max speed 0.18-0.60 m/s
(inlet `U_max = 0.3 m/s`), 2559 -> 2549 particles (open boundaries working).

**The acoustic (Riemann) wall-impedance term `p* = p_f + rho_0 c0 (v_w-v_f).n`
is the only remaining destabiliser** in this configuration: it is on by default
in `FSI2Case`, and it turns a bounded run into a blow-up at step 156 — the sign
found earlier at ratio 2 and 10 in `COUPLED_RUN_DEFECTS.md` §3, now isolated
from the rod defect.

### 3.3 Falsified claim

| claim | status |
|---|---|
| `COUPLED_RUN_DEFECTS.md` §3: "`rho_s/rho_f = 1` blows up in every variant; classical added-mass instability of the partitioned explicit FSI; no load filter can cure it" | **FALSIFIED as a statement about added mass.** The ratio-1 divergence was the rod sub-stepper (1e249 zero-load energy growth). With the bending-aware criterion the loose driver at ratio 1 stays bounded (9.0e-2 m/s vs 1.75e+03 m/s), and the old regression test that encoded the claim (`test_added_mass_limit_of_loose_coupling[1.0-True]`) now fails against the new code and has been revised with the measured numbers. |

## 4. Error judgement (predeclared gates: <15 % acceptable, <3 % reporting)

| quantity | reference | this round | rel. error | <15 % | <3 % |
|---|---|---|---|---|---|
| empirical stability boundary vs analytic `dt*omega_max < 2` | 2.0 (symplectic Euler) | flip bracketed by [1.82, 2.58] | boundary inside the bracket | yes | not resolvable (bracket, not a point) |
| free-beam energy drift at auto sub-steps, ratio 1 | <= E0 | 0.773 E0 | -22.7 % (numerical dissipation, bounded) | yes | n/a (dissipative, not an accuracy target) |
| FSI2 tip-y amplitude | 0.0082 m | **WITHHELD** — the surviving run covers 0.62 s (0.6 of a flapping period), not the periodic state | not assessable | **not assessable** | **not assessable** |
| FSI2 tip-x amplitude | 0.0028 m | WITHHELD, same reason | not assessable | **not assessable** | **not assessable** |
| drag mean / amplitude, lift amplitude | 3.2 / 0.96 / 0.22 | WITHHELD (transient only) | not assessable | **not assessable** | **not assessable** |

Honest statement: **the benchmark comparison was not achieved in this round.**
What was achieved is the removal of the blocker that made the comparison
impossible, and the identification of the remaining one (wall impedance). The
periodic-state run needs 2000-3000 steps (~1.2 s of beam flapping at
`f ~ 0.9 Hz` scaled) and is the round-2 target.

## 5. Notes and limitations

1. **The tolerance of the strong stepper matters more than the accelerator.**
   Gating the interface residual on `||r||/L_beam` accepts the first iterate
   whenever the per-step response is small, so the iteration silently
   degenerates into loose coupling — that is how a diverging run first appeared
   to "converge in 1 iteration" here. The gate is now the scale-free reduction
   `||r_k||/||r_0||` (default 1e-2..1e-3) plus a machine-level floor.
2. **IQN-ILS sign convention** was wrong in the first implementation
   (`Wq + r` instead of `Wq` for the Newton step). It is now pinned by a unit
   test on a synthetic map with gains 1.5 / 4.35 / 20, where both accelerators
   must cancel the residual to machine precision; Aitken is additionally tested
   to produce the negative relaxation factor the added-mass regime needs.
3. **Strong coupling did not converge on the real interface** to better than
   `~1e-2` relative reduction within 25 iterations: the load-transfer operator
   is not smooth in the interface position (moving the beam one particle radius
   changes which fluid particles enter the kernel), and the measured interface
   stiffness is acoustically dominated (`dF_y/ddelta ~ 8.4e4 N/m` vs the beam's
   `3EI/L^3 ~ 1e2 N/m`). This is a property of the boundary-pressure
   extrapolation, and is the natural next defect to attack.
4. The wall velocity handed to the fluid in the sub-iteration is the step-start
   beam velocity unless `wall_velocity_from_iterate=True` (Robin form), which is
   implemented and exercised but is not needed by the surviving configuration.
5. `max_rod_substeps` now raises instead of silently clamping when the derived
   sub-step count would exceed the cap.

## 6. Reproduction

```bash
PYTHONPATH=src python benchmarks/xval_fsi2/make_strong_coupling_assets.py   # figure + data
PYTHONPATH=src python benchmarks/xval_fsi2/run_strong_coupling.py --nst 100 --ratios 1
PYTHONPATH=src python -m pytest tests/test_fsi2_strong_coupling.py tests/test_fsi2_coupled_fixing.py -q
```

Test evidence: 63 passed in 44.8 s (`test_fsi2_strong_coupling.py`,
`test_fsi2_turek_hron.py`, `test_fsi2_coupled_fixing.py`,
`test_fsi2_similarity.py`, `test_fsi_force_from_fluid.py`).

## 7. Next round (queued, not done here)

1. Run the bending-aware, impedance-off FSI2 case to `t ~ 2.5 s`
   (2400 steps, ~25 min wall clock at 70 ms/step on one core; submit to the
   cluster queue, not the control plane) and extract tip amplitude + flapping
   frequency; compare with the Turek-Hron reference (0.0082 m / 0.0028 m) and
   with the archived SPHinXsys records under the 15 % / 3 % gates.
2. Diagnose the sign of the Riemann wall-pressure term
   (`boundary_pressure_from_fluid`) — it should damp a moving wall and instead
   destabilises every configuration tested so far.
3. Make the boundary-pressure extrapolation smooth in the interface position
   (panel/WLS fit instead of one-sided kernel sum) so that the strong iteration
   can reach 1e-4 reduction; then re-test strong coupling as the production
   driver.
## 8. Round 4 (same anchor): next-round item 2 is closed -- the impedance sign was inverted

`boundary_pressure_from_fluid` implemented `p* = p_f - rho c (u_wall - u_f).n`
while its own docstring specified `+`.  The heavy-wall limit of the linearised
Riemann problem (`Z_wall >> Z_fluid`) is `p* = p_f + Z (u_wall - u_f).n`, so the
shipped sign was an **anti-damping** term, which is exactly the observed
"destabilises every configuration" behaviour.  Fixed (default
`wall_impedance_sign = +1`, the old convention kept as an audit arm), guarded by
`tests/test_fsi2_impedance_sign.py` and an energy diagnostic on a common
trajectory: `Wdot = sum F_imp . v` is negative (dissipation) at all 19 sampled
states with the fix and exactly mirrored (injection) without it.  Coupled probe:
pre-fix sign reaches 1.5e3 m/s in 200 steps, fixed sign 0.35 m/s -- but the
`wall_impedance = False` arm grows too, so the *residual* excitation is not the
impedance term.  Details, tables and the honest error budget:
`docs/fsi2_impedance_sign/01_report.md`; data in
`docs/xval_fsi2/assets/impedance_{sign,energy}.json`.
