# Lake at rest: does the shipped WCSPH hold a still lake still?

Case-pool anchor `CASE-SPH-LAKE-AT-REST`. Reference: the analytic hydrostatic
identity (`u = 0`, flat free surface) — no experimental uncertainty, no fitted
constant. Artefact: `artifacts/lake_at_rest/rest_dx020.json`; code
`src/tensorsph/lake_at_rest.py`; runner `benchmarks/lake_at_rest/run.py`;
reproduce with

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py \
    --json artifacts/lake_at_rest/rest_dx020.json
```

## 1. Verdict

**FAIL** — the anchor's machine-precision tolerance is *not* met, and this
document says exactly where it breaks and where it does not.

* The **vertical** balance of the shipped operator is achievable to machine
  precision: `max|a_y|/g = 2.944e-13` after 15
  Gauss-Newton iterations (converged: yes).
* The **horizontal** balance is not. Side-wall truncation leaves
  `max|a_x|/g = 0.124` in the two
  smoothing-lengths-wide wall band, and no mass field that is mirror-symmetric
  about the tank centreline can cancel it (the reconstruction has one degree of
  freedom per particle and the horizontal force is odd under that reflection).
* Consequence, measured over 0.50 s: `max|v|/c0 =
  0.016`, surface departure `max|dy|/H =
  0.084`, density departure `max|drho|/rho0 =
  0.235`. The lake sloshes.

## 2. Case, method and numbers

Geometry: tank `0.4 x 0.3 m`, particle spacing
`dx = 0.02 m`, `h = 0.0260 m` (`h = 1.3 dx`),
`300` fluid particles on a square lattice, `198`
static wall particles in three layers, `3165` interacting pairs,
`rho0 = 1000 kg/m3`, `c0 = 20.0 m/s`, `gamma = 7.0`,
`g = 9.81 m/s2`. Acoustic Mach number `c0/sqrt(gH) =
11.66`.

Method, in the order the code runs it:

1. **Lattice and wall.** Square lattice from `pool_lattice`; the wall is built
   mirror-symmetric about `x = width/2` (see §4, finding F2).
2. **Analytic profile as a prior.** The summation density is linear in the
   particle masses, so "the mass field whose summation density is the exact
   Tait equilibrium profile `rho(y)`" is one direct linear solve, not a fixed
   point (the fixed point diverges at the free surface: rows alternate by ±30 %).
   This prior's residual, `max|a_y|/g =
   0.261`, is the floor a
   continuum reference can reach — the negative control that makes the
   reconstruction's machine-precision number meaningful.
3. **Discrete reconstruction.** Damped Gauss-Newton on `u = log(m/ m_unit)`,
   residual `a_y` of every fluid particle from the *shipped*
   `droplet_impact.rhs`, unknowns tied over mirror pairs, Tikhonov pull towards
   the prior to pick the physically meaningful member of a rank-deficient
   solution set, backtracking line search on the RMS merit.
4. **Rest test.** Symplectic Euler at `dt = 3.900e-04 s`
   (`Courant = 0.30`) for 1282 steps, positions free to move.

### Gate table (exact thresholds in `benchmarks/lake_at_rest/run.py`)

| gate | quantity | measured | threshold | verdict |
|---|---|---|---|---|
| `G1` | vertical equilibrium max|a_y|/g | 2.944e-13 | lt 1.0e-09 | PASS |
| `G2` | horizontal residual, side-wall band max|a_x|/g | 0.124 | lt 1.0e-09 | FAIL |
| `G3` | rest preservation max|v|/c0 | 0.016 | lt 1.0e-09 | FAIL |
| `G4` | surface invariance max|dy|/H | 0.084 | lt 1.0e-09 | FAIL |
| `G5` | density invariance max|drho|/rho0 | 0.235 | lt 1.0e-09 | FAIL |
| `G3i` | interior rest max|v|/c0 (>= 2h from side walls) | 0.012 | lt 1.0e-09 | FAIL |
| `G4i` | interior surface invariance max|dy|/H | 0.084 | lt 1.0e-09 | FAIL |
| `G6` | negative control uniform lattice mass max|a|/g | 1.000 | gt 5.0e-01 | PASS |
| `G7` | negative control analytic profile max|a_y|/g | 0.261 | gt 5.0e-02 | PASS |
| `G8` | linear response r(1e-4)/r(1e-3) | 0.100 | gt 5.0e-02 | PASS |
| `G8` | linear response r(1e-4)/r(1e-3) | 0.100 | lt 1.5e-01 | PASS |

### Negative controls and linear response

| control | `max\|a\|/g` | interior `max\|a_y\|/g` | `p_max` [Pa] | `rho/rho0` |
|---|---|---|---|---|
| analytic_hydrostatic_profile | 0.261 | 0.149 | 2854.899 | 1.0002..1.0070 |
| uniform_lattice_mass | 1.000 | 1.000 | 3.553e-10 | 0.7633..1.0000 |

Scaling the reconstructed mass field by `1 + eps` must scale the residual
linearly, which is what proves the near-zero residual of `G1` is a balance and
not a dead code path:

| eps | `max\|a_y\|/g` | per unit eps |
|---|---|---|
| 0.0001 | 0.085 | 848.316 |
| 0.001 | 0.850 | 849.848 |
| 0.01 | 8.654 | 865.395 |

### Reconstructed profile (row means)

| y [m] | particles | `rho/rho0` | p [Pa] | `m/m_unit` | mass spread in row |
|---|---|---|---|---|---|
| 0.010 | 20 | 1.007087 | 2895.7 | 0.4049 | 1.38e-02 |
| 0.030 | 20 | 1.006521 | 2660.2 | 0.4016 | 2.84e-03 |
| 0.050 | 20 | 1.006191 | 2522.8 | 0.4034 | 1.28e-02 |
| 0.070 | 20 | 1.005562 | 2262.1 | 0.4017 | 4.04e-03 |
| 0.090 | 20 | 1.005254 | 2134.8 | 0.4028 | 1.25e-02 |
| 0.110 | 20 | 1.004616 | 1872.2 | 0.4012 | 5.04e-03 |
| 0.130 | 20 | 1.004299 | 1741.8 | 0.4028 | 1.26e-02 |
| 0.150 | 20 | 1.003681 | 1488.7 | 0.3999 | 6.26e-03 |
| 0.170 | 20 | 1.003317 | 1340.0 | 0.4043 | 1.45e-02 |
| 0.190 | 20 | 1.002778 | 1120.5 | 0.3957 | 1.23e-02 |
| 0.210 | 20 | 1.002273 | 915.3 | 0.4117 | 2.26e-02 |
| 0.230 | 20 | 1.001968 | 791.7 | 0.3795 | 3.26e-02 |
| 0.250 | 20 | 1.001117 | 448.2 | 0.4431 | 4.98e-02 |
| 0.270 | 20 | 1.001355 | 544.2 | 0.3152 | 1.19e-01 |
| 0.290 | 20 | 0.944365 | 0.0 | 0.5734 | 9.40e-02 |

Physics cross-check on the interior band (`2h` from the bottom and from the
surface, 180 particles, `y` in
[0.070, 0.230] m):

| quantity | value |
|---|---|
| density vs exact Tait equilibrium, max rel. | 3.200e-04 |
| pressure vs exact Tait equilibrium, max rel. | 0.057 |
| pressure vs `rho0 g (H-y)`, max rel. | 0.058 |
| compressibility bias of the reference itself | 2.794e-03 |

That last row matters: the exact Tait equilibrium differs from
`rho0 g (H-y)` by 2.794e-03 of the bottom
pressure at `c0 = 20.0 m/s`, so part of the
"pressure vs `rho0 g (H-y)`" column is *modelling* bias, not discretisation —
the decomposition machinery is `tensorsph.eos_hydrostatics`.

## 3. Figures

* `docs/figures/lake_at_rest_residual.png` — Newton history with the two
  negative controls and the side-wall horizontal residual on the same axis.
* `docs/figures/lake_at_rest_profiles.png` — reconstructed pressure and density
  profiles against the two analytic references.
* `docs/figures/lake_at_rest_drift.png` — velocity / surface / density departure
  during the 0.50 s rest test, against the 1e-9 gate.

No field cloud map is shown because the case is one-dimensional by construction:
the state is a function of `y` only, and a contour plot of a stratified state
would show nothing a profile does not.

## 4. Findings

**F1 (blocking, the anchor's own verdict).** The shipped WCSPH cannot hold a
lake at rest to machine precision in a *finite* tank. The vertical balance is
reconstructible to `2.944e-13` but the side-wall band keeps
`0.124` of horizontal residual, which
drives the whole body: `max|v|/c0 = 0.016` within
0.50 s. The anchor's tolerance (machine precision) is therefore
**unattainable with the shipped scheme**; the honest statement is the pair of
numbers above.

**F2 (method, measured).** Mirror symmetry of the wall had to be *checked*, not
assumed: the shipped `wall_lattice` reaches the same column through two
arithmetic paths (bottom row and side wall) and the halves differ in the last
bits, so the unrounded lattice carries almost-duplicate columns and the mirror
map finds no partner for one of them — the mass field then silently loses its
symmetry and a spurious horizontal residual appears. After rounding to 1e-12 the
report gives 0 unpaired columns (shipped) and 0 (this case's
wall). The horizontal residual of F1 is therefore *not* a wall asymmetry
artefact; it survives a symmetric wall.

**F3 (negative result, documented so it is not retried blind).** Enabling the
shipped Adami free-surface pressure closure (`cfg.surface_pressure_closure`)
makes the reconstruction *stall*: the Newton history flattens at
`max|a_y|/g ≈ 2.0` after ten iterations, because the closure re-derives the
surface pressure from a threshold test on the summation density and the Tait
clamp is no longer the only non-smoothness. The default here is `False`.

**F4 (non-uniqueness).** The pair operator annihilates a constant pressure field
on a complete support, so the balance Jacobian is rank deficient; without the
Tikhonov pull the mass field wanders by up to 40 % between neighbouring rows.
The regularised solution is reported with its per-row mass spread
(table above) so the reader can see how much of the profile is datum and how
much is the chosen member of the solution set.

## 5. Not done (unknown, not "passed")

* The resolution ladder (`dx = 0.02` only) — the interior-band profile error has
  no measured convergence order yet, because a second resolution needs the
  300-particle Newton solve again at 4x the particle count.
* A per-particle (non-mirror) reconstruction minimising both components, which is
  the obvious way to attack F1: it turns the system into 2N equations and N
  unknowns, and the best achievable compromise is not known.
* Independent-implementation cross-check (ISPH / DFSPH on the same state).
* Longer runs (`t > 0.50 s`) and closed (periodic-x) alternatives
  to the side walls.
