# ISPH (projection) plane channel: GAP-SPH-ISPH-PROJECTION

**Anchor:** `GAP-SPH-ISPH-PROJECTION` (produced by the mechanical gap rule of
`FEAT-OSS-INVENTORY-SPH`, `docs/oss_inventory_sph/inventory.md` §3, 37 % of the
surveyed codes expose a projection/DFSPH route; recorded in
`docs/oss_inventory_sph/gaps.json`).
**Code:** `src/tensorsph/isph_channel.py`, `benchmarks/isph_channel/run.py`,
`tests/test_isph_channel.py`.
**Artifacts:** `runs/isph_channel/m16024/results.json` + four figures.

## 1. Case

Physics: incompressible flow of kinematic viscosity `nu` in a 2-D channel of
height `H`, no-slip walls at `y = 0` and `y = H`, periodic in `x`, driven by a
uniform body force `g_x` along `x`. Steady solution of
`0 = -dp/dx + mu u'' + rho g_x` with `p = p(x)` only:

    u(y) = g (y - a)(b - y) / (2 nu),  a = 0, b = H
    u_max = g H^2 / (8 nu),   tau_wall = rho g H / 2

| parameter | value |
|---|---|
| `H` | 0.1 m |
| `L = columns*dx` | 12 cells in `x` (periodic) |
| `H/dx` | 8, 12, 16 |
| `nu` | 0.01 m^2/s |
| `g_x` | 9.81 m/s^2 |
| `rho0` | 1000 kg/m^3 |
| kernel | cubic spline, `h = 1.9 dx` |
| wall | 5 static lattice layers (density support) + antisymmetric mirror images (viscosity) |
| `t_end` / averaging window | 1.0 s / last 0.1 s |
| `u_max` analytic | 1.22625 m/s |
| `tau_wall` analytic | 490.5 Pa |

## 2. How

The solver is a *projection* (incompressible) SPH scheme, not a weakly
compressible one — there is no Tait equation of state anywhere in this path and
therefore no acoustic time-step limit:

1. **prediction:** `u* = u + dt (nu lap(u) + g_x e_x)`. The viscous term is the
   Morris/Fox/Zhu pairwise Laplacian (`viscous_laplacian`), regularised by
   `eta = 0.01 h`.
2. **projection:** the pressure Poisson equation `lap(p) = (rho0/dt) div(u*)` is
   assembled as a sparse matrix over fluid-fluid pairs and solved with
   preconditioned conjugate gradients; the velocity is corrected with
   `u = u* - (dt/rho0) grad p`, using the pairwise difference form of the
   pressure gradient.
3. **advection:** `x += dt u` (semi-implicit Euler); `dt = min(0.25 h/|u|,
   0.125 h^2/nu)`.

Three discretisation decisions were needed and each is *measured*, not assumed
(`operator_signatures` in the artifact):

* the discrete Laplacian `L[p] = sum_j 2 V_j (g_ij.d_ij)/(|d_ij|^2+eta^2) (p_i-p_j)`
  approximates **+grad^2 p** here (measured 5.955 against 6.0 for `p = 3 y^2`,
  0.8 % error), so the Poisson equation is `L[p] = +(rho0/dt) div u*`;
* static wall particles must be **excluded** from the divergence sum. A wall
  particle carrying `u = 0` enters `(u_j - u_i) . grad W` as a tangential
  velocity jump and injected a spurious divergence of `-1.04` against an
  analytic `0.4` in a linear-field probe. Excluding them also makes the Poisson
  right-hand side compatible (`sum_i V_i div_i = 0` exactly);
* no-slip is imposed with **antisymmetric mirror images** (`u_ghost = -u_parent`
  at the mirrored position) instead of static wall particles at rest. A static
  wall one spacing away implements the Dirichlet condition only to first order;
  the repository had already recorded the consequence as
  `poiseuille.wall_shear_ratio = 0.5025` (WITHHELD) in
  `benchmarks/verification_ledger/ledger.json`. The antisymmetric image makes the
  wall plane itself the zero-velocity surface; the wall shear readout then lands
  within 1.2 %.

One negative result worth keeping: XSPH velocity smoothing (`eps = 0.05`), which
is present in many ISPH codes to regularise particle motion, acts as a *numerical
viscosity* here and biases the result by +7 % (`nu_eff/nu = 1.08` at
`H/dx = 12`); with `eps = 0` the measured `nu_eff/nu` is 0.997. It is therefore
off by default.

Toolchain: PyTorch 2.7.1 CPU (no GPU on the control node), scipy 1.16 sparse CG,
matplotlib for the figures. Command:

```
PYTHONPATH=src python benchmarks/isph_channel/run.py --out runs/isph_channel/m16024
```

## 3. Results (measured, machine-written into `results.json`)

| H/dx | u_max meas [m/s] | err vs `gH^2/8nu` | nu_eff/nu | tau_wall meas [Pa] | err vs `rho g H/2` | max `|rho/rho0-1|` | div rms ratio |
|---|---|---|---|---|---|---|---|
| 8 | 1.26073 | **+2.81 %** | 1.0115 | 484.9 | **-1.14 %** | 1.36e-04 | 1.01 |
| 12 | 1.24308 | **+1.37 %** | 0.9973 | 491.8 | **+0.27 %** | 3.05e-04 | 0.94 |
| 16 | 1.23688 | **+0.87 %** | 0.9978 | 491.6 | **+0.22 %** | 5.78e-04 | 0.95 |

(`u_max` is the maximum of the time-averaged row profile; `tau_wall` is
`rho nu du/dy` with `du/dy` taken from the quadratic least-squares fit of the
measured profile extrapolated to `y = 0`; `nu_eff/nu` is read from the curvature
of the same fit, `nu_eff = g/(2|u''|)`.)

Gate verdicts (`GAP-SPH-ISPH-PROJECTION`, `results[*].gates`):

| gate | limit | H/dx=8 | H/dx=12 | H/dx=16 | verdict |
|---|---|---|---|---|---|
| density fluctuation | <= 1e-3 | 1.36e-04 | 3.05e-04 | 5.78e-04 | **PASS at all three** |
| centre velocity | <= 3 % | +2.81 % | +1.37 % | +0.87 % | **PASS at all three** |
| wall-shear budget closure | <= 3 % | -1.14 % | +0.27 % | +0.22 % | **PASS at all three** |

Comparison with the recorded WCSPH state of the same case
(`benchmarks/verification_ledger/ledger.json`):

| quantity | WCSPH (ledger) | ISPH (this work, H/dx=16) |
|---|---|---|
| centre velocity | 1.51 % error | 0.87 % error |
| wall shear ratio to `rho g H/2` | 0.5025 (WITHHELD — two readings inside one artifact disagree) | 1.0022 (readout agrees with the centre velocity) |
| density fluctuation | ~2.3e-2 (quoted by the gap rule, `gaps.json`) | 5.78e-04 |

The density figure is the one that is *not* re-measured here: the committed
WCSPH artifact (`benchmarks/poiseuille/out/results.json`) stores fluid particles
only, so a like-for-like re-evaluation with the wall-completed estimator used
here was not obtained — see §5.

## 4. Figures

* `fig1_profile.png` — mean profile at three resolutions against the analytic
  parabola: the curves are visually indistinguishable above `y/H = 0.1`; the
  largest deviation sits in the first fluid row, which read **+21 %** before the
  antisymmetric mirror-image wall treatment and is inside 3 % after it.
* `fig2_gates.png` — centre-velocity and wall-shear error against resolution with
  the 3 % gate drawn.
* `fig3_density.png` — max summation-density fluctuation in time against the
  1e-3 gate.
* `fig4_divergence.png` — RMS divergence after/before the projection. The ratio
  is `~0.95`, not `~0`: with Neumann walls and a truncated support the discrete
  divergence has a remainder the Poisson problem cannot remove; it is reported
  instead of being hidden (see §5).

## 5. Notes, traps, limits

1. **Wall shear estimator.** The ledger's WCSPH wall shear used the raw
   two-row difference `(u_1-u_0)/dx`, which is a first-order estimate. Here the
   wall shear comes from the quadratic fit's wall slope, which is exact for a
   parabola. The two estimators are not interchangeable; the WCSPH number 0.50 is
   not a like-for-like comparison and the ISPH number should not be read as a
   7x accuracy claim on the drag.
2. **The density numbers are summation densities**, i.e. a diagnostic of particle
   order, because there is no equation of state in this path. They are
   *not* comparable with a WCSPH `rho/rho0` read from a Tait law.
3. **Boundary clamping.** Particles that cross the wall band during advection are
   clamped back to `y = dx/2` (424 events at `H/dx = 12` over 320 steps (584 at `H/dx = 16` over 568 steps),
   recorded as `n_boundary_clamped`). Those events are a modelling device, not
   physics: the Neumann pressure condition does not by itself guarantee
   no-penetration for a Lagrangian cloud. A ghost-fluid pressure treatment would
   remove the need for it.
4. **Divergence remainder.** `div_after/div_before ~ 0.95` means the projection
   removes only a small part of the RMS divergence. That is expected: the RMS is
   dominated by the truncation bias at the walls, which the projection cannot
   remove because the pressure operator is fluid-only. The velocity field is
   nevertheless incompressible to 1e-4 in density because the *flow* is a steady
   shear, not because the projection is perfect.
5. **Not measured / open:** (a) like-for-like WCSPH density fluctuation with the
   wall-completed estimator (§3); (b) 3-D extension — the operators here are
   hard-wired to the 2-D cubic-spline normalisation, as in the rest of the
   repository; (c) a free-surface ISPH case (the Dirichlet `p = 0` condition is
   implemented in the literature but not here).

## 6. Error judgement (required question: is every key quantity < 3 %?)

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| u_max, H/dx = 8 | `g H^2/(8 nu)` = 1.22625 m/s | +2.81 % | yes (marginal) |
| u_max, H/dx = 12 | same | +1.37 % | yes |
| u_max, H/dx = 16 | same | +0.87 % | yes |
| tau_wall, H/dx = 8 | `rho g H/2` = 490.5 Pa | -1.14 % | yes |
| tau_wall, H/dx = 12 | same | +0.27 % | yes |
| tau_wall, H/dx = 16 | same | +0.22 % | yes |
| nu_eff/nu, H/dx = 12 | 1 | 0.27 % | yes |
| density fluctuation | gate 1e-3 (not a physical reference) | 5.78e-04 at H/dx = 16 | yes (gate) |

Every gate passes, and the errors decrease with resolution for the centre
velocity. The honest caveats are the wall-shear *estimator* change (§5.1), the
clamping events (§5.3) and the un-repeated WCSPH density comparison (§5.5a).
