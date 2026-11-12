# CASE-COUETTE-SPH — predeclared design (written before any run)

Plane Couette flow between a stationary wall and a wall sliding at `U` in a
2-D channel of height `H`, periodic along `x`. The exact steady solution of the
incompressible Navier–Stokes equations for this configuration is the linear
profile

    u(y) = U y / H,     du/dy = U / H,     tau = mu U / H,     Q = U H / 2

(Batchelor 1967, *An Introduction to Fluid Dynamics*, CUP, §4.2 — steady
unidirectional flow between parallel plates, one plate moving; the same
closed form is used as the `Ls = 0` limit of the Navier-slip Couette anchor in
`TensorLBM/benchmarks/ice_lid_navier_slip.py:anchor1_navier_slip_couette`).
The dimensionless groups are `Re = U H / nu`, `Ma = U / c0` and the resolution
`H / dx`; there is no free parameter in the reference, so nothing can be tuned
to the answer.

## Decision this benchmark has to support

Does the TensorSPH 2-D weakly-compressible solver with a **moving wall**
(`rhs(..., wall_vel=...)`, `wall_friction = 1`) reproduce the analytic Couette
solution within the pool tolerance of 5 % — and is that agreement caused by the
tangential wall shear rather than by the initial condition or a fitting
artefact?

## Alternatives

* **A (adopted).** Reuse the existing moving-wall input of
  `src/tensorsph/droplet_impact.py:rhs` (`wall_vel`, already exercised by the
  `wave_flume` wavemaker) with `wall_friction = 1`, i.e. no new physics, only a
  new *case*: a tangentially sliding wall instead of a normally translating
  paddle.
* **B (rejected).** Drive the same linear profile with a body force plus a
  counter-shear, or with an imposed momentum source. Rejected: it tests the
  operator, not the wall condition, and duplicates
  `benchmarks/poiseuille` which already covers body-force-driven shear.
* **C (rejected).** Initialise at the analytic linear profile and report the
  first step. Rejected: that is a restatement of the initial condition; it
  cannot fail, so it carries no evidence. The run therefore starts from rest
  (impulsive start of the moving wall) and is integrated to a steady state.

## Hypothesis

H1: the measured steady profile, the mass flux and the wall shear agree with
the analytic values within 5 % at `H/dx >= 16`, and the error decreases with
`H/dx`.

H0 (what would falsify H1): the measured profile is not linear / the flux or
wall shear misses the analytic value by more than 5 %, or the result is
insensitive to the wall condition (i.e. the free-slip control reproduces the
same profile), which would mean the agreement comes from something other than
the wall shear.

## Metrics (all machine-computed in `run.py`)

| id | metric | analytic target |
|----|--------|-----------------|
| M1 | interior L2 profile error `||u_meas-u_ana||_2/U` over rows `0.15 <= y/H <= 0.85` | 0 |
| M2 | mass flux ratio `Q_meas/(U H/2)`, `Q = sum_k u_k dx` | 1 |
| M3 | wall shear ratio `tau_meas/(mu U/H)`, `tau_meas = rho nu (u_1-u_0)/dx` (bottom and top) | 1 |
| M4 | linearity: max `|u_meas - line_fit| / U` over the interior | 0 |
| M5 | steady-state drift of `Q(t)` over the last 20 % of the run | < 1 % |
| M6 | startup decay rate of `Q_inf - Q(t)` vs `nu pi^2 / H^2` (supplementary) | 1 |

## Predeclared acceptance thresholds (fixed before the first run)

* **PASS** if, at the finest ladder resolution `H/dx = 24` (and at `H/dx = 16`),
  M1 < 5 %, `|M2 - 1| < 5 %`, `|M3 - 1| < 5 %` (bottom wall; the top wall is
  reported too), and M5 < 1 %.
* M4 is reported as evidence of linearity but is **not** gated: the wall layer
  has a finite slip length by construction (an SPH wall is not a sharp plane),
  so a few percent of near-wall deviation from the exact straight line is
  expected and is itself the measurement (the slip length is reported).
* **Negative control** (`wall_friction = 0`, free slip): PASS the control if
  M2 < 0.2, i.e. the flow is not driven. If the control also reaches the
  analytic flux, the whole measurement is void.
* **Resolution response**: M1 at `H/dx = 24` must not exceed M1 at
  `H/dx = 10` (monotone or at least non-diverging convergence).
* **honest FAIL** is recorded if any gated metric misses; the numbers are
  reported unchanged.

## Controlled vs changed variables

Held fixed across the ladder: `H = 0.1 m`, `L = 12 dx`, `rho0 = 1000`,
`c0 = 20 m/s`, `gamma = 7`, `h = 1.3 dx`, `wall_layers = 3`, `U = 0.2 m/s`,
`nu = 0.02 m^2/s`, `wall_friction = 1`, `dt = 0.3 h/c0` (acoustic limit,
fixed step → bit-reproducible), start from rest, `t_end = 1.2 H^2/nu`.

Changed one at a time (sensitivity, two independent axes plus convergence):

1. resolution `H/dx in {10, 16, 24}` (and `32` with `--extended`);
2. driving velocity `U in {0.1, 0.2, 0.4}` → `Re = UH/nu in {0.5, 1, 2}`,
   `Ma = U/c0 in {0.005, 0.01, 0.02}`: the measured slope must scale with `U`;
3. wall friction `f in {0, 0.5, 1}`: `f = 0` is the negative control, `f = 0.5`
   must sit between free slip and no slip (slip-length trend).

`nu` is held at `0.02` for the ladders because the analytic profile is
independent of `nu`: varying it would change `Re` without changing the target,
which is exactly the confusion the `U` sweep avoids.

## Confounders and how they are handled

* **Finite wall slip length.** An SPH wall with `wall_friction` is a force BC,
  not a Dirichlet BC; the effective no-slip plane sits a fraction of `dx`
  outside the last wall layer (measured 0.91 dx in the Poiseuille benchmark).
  Handled by *reporting* the slip length and the near-wall row error separately
  from the interior error, and by the resolution ladder.
* **Acoustic transients at the impulsive start.** Squeezed out by
  `t_end = 1.2 H^2/nu` and a 20 % averaging window; M5 is the gate that proves
  the state is steady.
* **Compressibility.** `Ma <= 0.02` at all points; the Tait pressure is a
  numerical device here, the analytic solution is incompressible.
* **2-D vs 3-D.** Plane Couette is invariant in `z`; the 2-D solver is the
  correct reduction, no `z` dynamics are suppressed that the reference has.
* **Fitting artefact.** The analytic reference is a closed form evaluated at
  the same row centres as the measurement — no fit is used for M1/M2/M3. The
  line fit is used only for M4 (linearity) and for the slip length.

## Environment / reproducibility

`PYTHONPATH=src python benchmarks/couette/run.py --out benchmarks/couette/out`
(CPU, `torch.set_num_threads(1)`, no randomness anywhere: lattice initial
state, fixed `dt`). All numbers land in `out/results.json` with the exact
parameters of every run; `docs/reports/m15412-couette-sph.md` is generated from
that file, so no number is copied by hand.