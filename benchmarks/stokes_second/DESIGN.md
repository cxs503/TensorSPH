# Stokes' second problem in TensorSPH — predeclared design

*Written before the production ladder was run (2026-11-12). Case-pool anchor
`CASE-SPH-STOKES-2ND-PROBLEM`. Every threshold below is fixed here, not after
looking at the numbers.*

## 1. Decision the experiment has to support

**Does TensorSPH reproduce the exponential amplitude decay of a viscous shear
wave driven by an oscillating no-slip wall, with a penetration depth `delta`
correct to 5 %?** If yes, the solver's viscous operator + *moving* wall
condition are qualified for wall-driven unsteady shear (oscillating wavemaker,
vibrating structure, ice-plate shear); if no, the failure localises to the wall
condition or the viscous operator and is booked as a negative result.

Alternatives that were rejected, and why:

| alternative | rejected because |
|---|---|
| ISPH projection channel (`tensorsph.isph_channel`) | its walls are **static** (`u = 0`); the case needs a prescribed moving wall. Pressure projection buys nothing here — the exact field is solenoidal, so pressure is uniform and drops out of the momentum equation. |
| `wall_friction` blend against static wall particles (the `benchmarks/couette` route) | measured in `docs/verification_poiseuille_channel.md` at 31–49 % wall slip / 50 % of the wall shear lost: a wall offset of ~0.9 dx would enter the *decay rate*, which is the graded quantity. |
| mirror (ghost) wall, static | this is the chosen route, with the wall's own velocity added (`plane_vel`), so the no-slip plane moves **on** `y = 0`. |
| 1-D reduced (pure diffusion) solver | would not exercise the 2-D SPH machinery at all; the case pool asks for the SPH solver, not a diffusion FD scheme. |

## 2. Hypothesis

H1: with `delta/dx >= 8` and a mirror-ghost no-slip wall moving at
`U0 cos(omega t)`, the row-wise amplitude `A(y)` matches `U0 exp(-y/delta)` with
relative L2 error < 5 % inside `y <= delta`, and the fitted decay rate matches
`1/delta` within 5 %.
H0: the discrete amplitude profile is a discretisation artefact (error > 5 %, or
a decay rate that does not follow `1/delta`).

## 3. Outcome metrics and unit of analysis

Unit of analysis = **one lattice row** (`y_k = dx/2 + k dx`), because the
reference is a function of `y` only. Row-mean `u_x(t)` is sampled during the
measurement window at 160 samples per period and fitted with
`A cos(omega t + phase) + c` (`tensorsph.stokes_second.harmonic_fit`); `A(y)` and
`phase(y)` are the fit outputs. `A/analytic` per row is the scalable primitive;
the gates below aggregate it.

* primary: `L2` over rows with `y <= delta` of `(A - A_ref)/A_ref`.
* secondary: worst single row in the same set; decay-rate slope
  `-d lnA/dy` vs `1/delta`; finite-depth error; phase error (diagnostic).

## 4. Predeclared thresholds and stopping rule

| gate | quantity | threshold |
|---|---|---|
| G1 | amplitude relative L2, rows with `y <= delta`, vs `U0 exp(-y/delta)` | ≤ 5 % |
| G2 | worst single row in the same set | ≤ 10 % |
| G3 | fitted decay rate vs `1/delta` (`delta_measured/delta - 1`) | ≤ 5 % |
| G4 | same L2 but vs the **finite-depth** exact solution (discretisation only) | ≤ 5 % |
| G5 | negative control: **free-slip** oscillating wall drives flow | `A_max/U0 < 5 %` |
| G6 | resolution: G1 must not grow from the coarsest to the finest mesh | `L2(finest) <= L2(coarsest) + 0.5 pp` |
| – | Stokes layer must contain ≥ 5 particle rows | `>= 5` (case-pool requirement) |

Stopping rule: the ladder is `delta/dx = 8, 12, 18`; it is **not** extended to
"make the number look better". If G1 fails at all three resolutions the run is
reported as an honest FAIL with the failing row and the error sign, and the
diagnosis (wall vs operator) is recorded in `negative_knowledge_ledger`.

**Ladder composition is now part of the gate.** The first delivery shipped a
`results_ladder.json` whose `resolutions = [8]` while still writing
`G6_not_growing: true` — a one-rung ladder cannot test convergence, so the gate
was vacuous. `run.py` now records `convergence.complete` (the required meshes
are exactly the ones run) and `all_gates` is false when a rung is missing.

Revision note (m16330, pre-run): this document said `8, 12, 16` while the code
said `8, 12, 18`. No `delta/dx = 16` run has ever been made and no `18` result
was seen before the decision, so aligning the text to the code changed nothing
after the fact; it removes an ambiguity, it does not move a threshold.

## 5. Variables, controls, confounders

* **Controlled**: `nu = 0.01 m^2/s`, `omega = 2 pi rad/s`, `U0 = 0.01 m/s`,
  `H = 6 delta`, `rho0 = 1000`, `c0 = 4 m/s` (Ma = 2.5e-3), `gamma = 7`,
  `h/dx = 1.3`, `wall_friction = 1`, columns in x = 6, initial state at rest,
  fixed time step `dt = min(0.3 h/c0, 0.125 h^2/nu)`.
* **Changed**: `dx` (ladder), and one at a time in the sensitivity block:
  `U0` (halved), `h/dx` (1.3 → 1.9), wall slip model (no-slip → free-slip),
  wall motion on/off.
* **Confounders identified and handled**:
  - *domain truncation* — a channel of depth `H` has the exact solution
    `Re{U0 sinh(m(H-y))/sinh(mH) e^{i omega t}}`, not the half-space one; the
    difference is reported (`truncation_bias_at_delta_pct`) and G4 removes it.
  - *start-up transient* — the rest initial state decays with the slowest
    channel mode, time constant `H^2/(pi^2 nu) = 1.16` periods; the measurement
    window is the **last two periods** of a five-period run, and the
    window-to-window change of the amplitude is reported as the residual
    transient (a drift number, not a gate).
  - *acoustic noise / rectified drift* — reported as `rectified_mean_max_over_U0`
    (row-mean DC velocity) and `max_mach`; a DC offset would fake a large
    amplitude at large `y`, so it is quoted next to every amplitude.
  - *x-noise* — the exact field is `x`-invariant, so the row average is the
    right estimator and the intra-row scatter is the noise floor; the
    amplitude is only quoted where `A > 1 % of U0` for relative errors.
  - *time-step artefacts* — `dt` is set by the acoustic or viscous limit
    (whichever binds) and recorded; it changes with `dx` along the ladder, so a
    G1 trend with resolution cannot be blamed on `dt` alone; the viscous limit
    binds at `delta/dx = 16`.

## 6. Repetition and uncertainty rationale

Each configuration is deterministic (fixed `dt`, lattice initial state, no
random numbers), so a single run is exact and reproducible bit for bit; the
relevant variability is *spatial* (row-to-row scatter, `fit_residual`) and
*temporal* (window drift), both reported per run. No repetition is needed to
estimate run-to-run noise because there is none; the reported uncertainty is
therefore the discretisation spread across the ladder plus the identified
confounders.

## 7. Environment and reproducibility

```
PYTHONPATH=.:src python benchmarks/stokes_second/run.py --ladder --c0 4 \
    --periods 6 --measure-periods 2 --out benchmarks/stokes_second/out
PYTHONPATH=.:src python benchmarks/stokes_second/run.py --sensitivity-only \
    --c0 4 --periods 6 --measure-periods 2 --out benchmarks/stokes_second/out
PYTHONPATH=.:src python benchmarks/stokes_second/make_doc.py
```

`results_ladder.json` records `torch`/`numpy` versions, thread count (1:
measured 6x faster than the default on these tiny tensors), device, `dt`,
`steps`, mesh, wall clock and the symbolic + numeric provenance of the reference;
the `provenance.case` block is filled from the **parsed arguments** (previous
delivery recorded the module constant `c0 = 10` while the run used `c0 = 4`).
`benchmark.json` is the machine-readable ref-vs-got table (one row per judged
quantity, tolerance, pass) and `MANIFEST.json` carries the SHA-256 of every
artefact, so a later edit of a result file is detectable. Figures are written to
`benchmarks/stokes_second/out/*.png`.

The run length is six periods (the first delivery used five); the measurement
window is the last two periods in both cases, so the longer run only reduces the
rest-to-periodic transient, it does not change a threshold.

Artefact size: the field of the space-time solution is persisted once, for the
finest mesh, decimated in time by 2 — the previous 4.2 MB / 10.4 MB result files
were dropped by the artefact filter, i.e. the numbers survived but the raw
evidence did not.

## 8. Interpretation limits

A pass qualifies the *viscous operator and the moving no-slip mirror wall for
smooth, wall-driven, solenoidal shear at Ma ≈ 2e-3*, on a regular lattice, at
`delta/dx >= 8`. It does **not** qualify: free-surface cases, high-Ma impact,
turbulent shear, pressure-driven flow with a wall-normal pressure gradient, or
a turbulent boundary layer. Correlation with the analytic curve is not evidence
of anything outside the tested envelope.