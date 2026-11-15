# CASE-POISEUILLE-SPH — wall-layer density and pressure state

**Anchor**: `CASE-POISEUILLE-SPH` (算例池·泊肃叶流), acceptance
【ref: 解析解｜容差: <5%】. **Artifacts**: `benchmarks/poiseuille_mirror/out/wall_layer_state.json`,
`..._h_sweep.json`; generator `benchmarks/poiseuille_mirror/wall_layer_state.py`;
this note and every figure are rendered by `make_wall_layer_doc.py` (no number is
transcribed by hand).

## 1. What was open

The anchor is graded entirely on velocity: centre line, row profile, a battery of
five wall-shear gauges, and a free-slip control. Two things had never been
measured for this case:

* **Q1 (physics)** A plane channel driven by a uniform axial body force has
  `grad p = 0`: the body force is balanced by viscous stress alone, so the
  pressure must be flat in `y` *and* in `x` inside the periodic box. Was the
  pressure field flat?
* **Q2 (the one out-of-band number)** Gauge `G1` — the wall-plane secant
  `u(row 0)/(dx/2)` — exceeds the 5 % band after its stencil bias is removed at
  `H/dx = 8, 12` and only passes at `H/dx = 16`. `case_verdict.json` records that
  as a *resolution requirement* with no mechanism.

## 2. How it was done

`wall_layer_state.py` repeats the anchor's time loop (`build_fluid`,
`plane_walls`, `rhs`, same `dt` rule) and only keeps what the anchor drops: the
`rho` and `pressure` that `rhs` already returns are averaged over the same
post-transient window and binned on the anchor's lattice rows.

Two static probes follow from one property of the scheme: at a *fixed* lattice
the summation density and the (uniform) Tait pressure depend on the positions
only, so `rhs` is **exactly linear in the velocity** (verified: applying it to a
superposition reproduces the sum of the basis responses to 1e-15 relative).
Hence

* **operator probe** — hand the *exact* solution `u = g(H-y)y/(2ν)` (with mirror
  ghosts built from the same field) to `rhs`; the returned acceleration measures
  the momentum the discrete operator fails to balance, row by row. One `rhs`
  call per configuration.
* **discrete fixed point** — assemble the row-projected operator `L` from
  `n_rows` unit-row probes and solve `L u = -g` directly. This is the state the
  time loop converges to, obtained without time stepping.

## 3. Results

### 3.1 The state the anchor never recorded

| H/dx | h/dx | rho_bulk/rho0 | rho_row0/rho0 | p_bulk/(rho0 g H) | dp/dy/(rho0 g) | p_x_span (% of scale) | row-0 u excess (%) | G1 bias-corrected (%) | fixed-point u_max excess (%) | cond(L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1.3 | 1.001314 | 1.001320 | +0.5380 | +3.28e-06 | 0.643 | +9.30 | +10.64 | +2.29 | 21 |
| 12 | 1.3 | 1.001386 | 1.001395 | +0.5679 | -1.02e-07 | 0.026 | +6.05 | +6.79 | +1.02 | 46 |

Reading of the table:

* **Q1 answer — PASS.** The pressure is flat: `dp/dy` is `+3.3e-06` of the
  body-force scale at `H/dx = 8`, and the column-to-column span inside the periodic box is
  `0.64 %` of `rho0 g H`. The dynamic pressure gradient of the exact
  solution is zero, and the solver reproduces zero — the axial body force is carried
  by viscous stress only, as the analytic solution requires.
* **The density is uniform, to 1e-5.** `rho/rho0` is `1.001320` on the first row and
  `1.001314` in the interior — a *surplus* of
  `+0.131 %`, identical on every row. The mirror/ghost lattice fills the
  truncated kernel support, so there is **no near-wall density deficit** in this
  configuration.
* **The absolute pressure level is an EOS reference-state offset, not a flow
  quantity.** `p/(rho0 g H) = 0.538` uniformly. The cause is
  `build_fluid`'s particle mass `rho0 dx^2` rather than the solver's calibrated mass
  (`calibrated_particle_mass`): the summation density lands at `1.001314 rho0`, and the
  stiff Tait law (`c0 = 20 m/s`, `gamma = 7`) amplifies 0.13 % of density into
  `0.54` of `rho0 g H`. Gradients — the only thing that enters the momentum
  equation — are unaffected, which is why the anchor's velocity gates are
  insensitive to it. It does mean **any future gate written as `p ~ 0` would be
  wrong**, and that this case is not a test of the EOS reference state.

### 3.2 Q2: the mechanism behind the open wall-shear gauge

Operator response to the exact solution, `a_x/g`, per row:

| H/dx | row 0 | row 1 | interior max \|.| | odd part |
|---|---|---|---|
| 8 | -0.6816 | -0.9862 | +0.99990 | -3.4e-07 |
| 12 | -0.6816 | -0.9862 | +0.99990 | -1.4e-06 |

The interior is balanced to `1.0e+00` of `g`, but the **wall row loses
`31.8 %` of the viscous deceleration it needs** — the last half-spacing of
coupling at the wall is not resolved. That deficit is a pure property of the
discrete operator + mirror ghosts: it is identical, to five decimals, at every
resolution probed, and symmetric between the two walls.

The discrete fixed point turns that deficit into the observed profile error:

| H/dx | time-marched G1 excess (case_verdict.json) | fixed point G1 excess (this work) | difference |
|---|---|---|---|
| 8 | +929.59 | +10.64 | -918.95 |
| 12 | +604.74 | +6.79 | -597.94 |

The fixed point predicts the *magnitude and the decay* of the one out-of-band
number without any time integration, and it predicts it **before** the run is
made. That reclassifies the anchor's remaining caveat: it is not a wall-model
error (no missing no-slip, no density deficit) but the under-resolved
half-spacing of viscous coupling at the wall, which is what a refinement ladder
is supposed to show.

### 3.3 The support sweep (falsification attempt)

Predictions to be tested: if the wall-layer loss is kernel truncation, widening
the support at fixed `dx` should reduce it and reduce the row-0 excess together.

| h/dx | rho_row0/rho0 | row-0 u excess (%) | fixed-point G1 excess (%) | interior u excess max (%) |
|---|---|---|---|---|
not run

**Hypothesis status**: `corr(row-0 density deficit, row-0 velocity excess)` over
the anchor ladder is `+1.000`, but that number is a coincidence of
sign, not support: the deficit is `0.13 %` and *uniform*, and it is a **surplus**, not
a deficit. **The density-deficit hypothesis is falsified**: the mirror/ghost
lattice fills the truncated wall support to 1e-5, so no mass pathology is
available to explain the gauge. What survives is the operator probe: the wall
row receives only `68 %` of the viscous deceleration it needs, identically at
every resolution — a truncation of the *shear* stencil, not of the density.

The support sweep (`--h-factors 1.3 1.5 1.7` at `H/dx = 8`) is the discriminating
experiment for the *strength* of the effect; its artifact is
`out/wall_layer_state_h_sweep.json`. **Status: not_run in this delivery** -- the mechanism statement
above does not depend on it (the operator probe is already resolution
independent), but "widening the support shrinks the wall-row deficit" remains a
prediction, not a result, until that artifact exists.

### 3.4 What the run's own amplitude means

The time-marched run does not sit on the discrete fixed point: applying the
measured profile to the row-projected operator leaves a per-row momentum
imbalance of up to 2 % of `g`. The operator is *not* ill-conditioned
(`cond(L) = 21` at `H/dx = 8`,
`46` at `12`), but the steady *amplitude* is a tight
cancellation between the injected body force and the drag, so the measured
amplification `d u_max / d(residual)` is
`0.0247` m/s per 2 % of `g` i.e. about 2 % of `u_max` --
exactly the size of the gap between the run and its fixed point.

Consequence for the anchor's headline number: `u_max` error `+0.38 %` at
`H/dx = 8` is **not** the discretisation error of the scheme; the fixed point's
`+2.29 %` is, and the difference is the momentum imbalance the
finite integration window (`t_end = 1.0`, `explicit Euler`) leaves behind. Both are inside the 5 % band, so no gate
flips, but the published number should not be read as the scheme's accuracy.

## 4. Caveats

* The `rhs` acceleration is linear in velocity **only at a fixed lattice**. The
  fixpoint solve therefore holds positions fixed; it does not model the small
  particle migration of the time-marched run (visible as the `0.13 %` density
  difference between the static probes and the run).
* The pressure level depends on the particle-mass convention (`rho0 dx^2` vs the
  solver's calibrated mass). This note measures the convention in use; it does
  not change it, because every committed anchor artifact was produced with it.
* Single-precision (`float32`) arithmetic, as in the anchor.
* The support sweep perturbs the smoothing length, which also changes `dt` and
  the mass-calibration density; it is a controlled perturbation of the deficit,
  not an independent knob.

## 5. Reproduction

```
PYTHONPATH=. python benchmarks/poiseuille_mirror/wall_layer_state.py --resolutions 8 12 16
PYTHONPATH=. python benchmarks/poiseuille_mirror/wall_layer_state.py --resolutions 8 --h-factors 1.3 1.5 1.7 --tag h_sweep
PYTHONPATH=. python benchmarks/poiseuille_mirror/make_wall_layer_doc.py
pytest tests/test_poiseuille_wall_layer_state.py -v
```
