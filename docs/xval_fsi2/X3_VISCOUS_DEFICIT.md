# X3 — the viscous fluid→solid branch: 38 % deficit, its origin, and the repair

Anchor: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`
(Xiangyu-Hu/SPHinXsys `test_2d_fsi2` + `FSI::ForceFromFluid`).
Status: the *pressure* transfer was cross-validated in
`GAP-SPH-FSI-BENCHMARK_xval.md` (X1/X2/X4).  The *viscous* branch failed there
with a flat `-38 %` that did not improve under refinement (X3, -38.54 % at `dx=0.025`, -38.01 % at `dx=0.00625`) —
above the 15 % gate, so per the owner accuracy standard it was **not usable** and
the cause was recorded as a hypothesis only.  This document closes that item:
the deficit is derived in closed form, the hypothesis is confirmed in structure
(not in value), and a corrected arm is added that is exact on the plane case.

## 1. Case introduction

Physics: the load a viscous fluid exerts on a wetted wall is the tangential
traction `tau = mu du_t/dn`.  The transfer operator used by the reference code
(`force_on_structure.hpp:55-72`, `OneSideViscosity`) sums, per solid particle,

```
F_i = V_i sum_j mu_j (dW/dr) r_ij (u_i - u_j) / (r_ij^2 + 0.01 h^2) V_j
```

ported verbatim as `fsi_force_from_fluid.viscous_force_from_fluid`.  Because the
solid surface cuts the kernel support, that sum is a *one-sided* quadrature.

| item | value | source |
|---|---|---|
| X3 geometry | plane Couette, gap 0.1 m, length 1.0 m, lower wall at rest | textbook; viscosity branch of `force_on_structure.hpp:55-72` |
| X3 fluid state | analytic `u_x = U y / gap`, `U = 1.0 m/s` | analytic (no solver in the loop) |
| X3 viscosity | `mu = 0.01 Pa s`, `rho = 1` | reference `rho0_f = 1`, `Re`-free static check |
| X3 reference | `tau = mu U / gap`, `F_x = tau L` | `cases.py:couette_shear` |
| X5 geometry | closed disk `R = 0.5 m` at `(2, 2)`, beam removed | `fsi2.h:14-35` (fsi2 cylinder) |
| X5 field | `u_x = gamma y + beta y^2/2`, `gamma = beta = 1`, `mu = 0.01` | analytic |
| X5 reference | `F_x = mu beta A`, `A = pi R^2` (divergence theorem) | `cases.py:curved_disk_shear` |
| kernel / smoothing | Wendland C2, `h = 1.3 dx`, support `2h`, `eps = 0.01` | OSS defaults (`adaptation.h:71`), `riemann_solver.h`, `viscosity.h` |
| OSS blueprint | `Xiangyu-Hu/SPHinXsys@master` read 2026-11-13 | `artifacts/xval_fsi2/oss_src/` |

## 2. Method

**Step 1 — derive the deficit instead of fitting it.**  For a wall at `y = 0` and
a linear profile `u_x = gamma y`, the OSS sum over a surface layer of spacing `s`
reduces (continuum limit, half-space support, `dV = r dr dtheta`) to

```
F_x/L = -2 mu gamma s int_0^R (dW/dr) r / (1 + eps h^2/r^2) dr
```

For Wendland C2, `int_0^2 f'(q) q dq = [f q]_0^2 - int_0^2 f dq = -2/3` and
`alpha_2 = 7/(4 pi)`, so the unregularised ratio to the exact traction is

```
ratio = 7/(3 pi) * (s/h)          (closed form, kernel-dependent, dx-free)
```

This is implemented as `one_sided_viscous_truncation_factor(h, dim, kernel,
spacing)` (kernel quadrature, no fitted constant) and tested against the closed
form.  **It is a moment of the kernel, not a discretisation error** — which is
exactly why the measured deficit does not shrink with `dx`.

**Step 2 — repair.**  `wall_shear_from_gradient` reconstructs the velocity-gradient
tensor at the wall from the *same* one-sided neighbour set and evaluates the
traction directly:

```
G_km = sum_j V_j (u_j - u_i)_k (dW/dr) e_m      (raw gradient sum)
B_pm = -sum_j V_j (dW/dr) r_p r_m / r           (measured first moment)
a    = G B^-1,      F_i = A_i mu_i (a . n_i)
```

`B` is taken from the *actual truncated* neighbour set, so the correction absorbs
the cut support (and, for a curved wall, the local moment anisotropy).  Because
`G` is linear in `r_ij` and `B` is built from the same pairs, a linear velocity
field is reproduced **algebraically exactly** — the plane case is not merely
accurate, it is machine exact.  `mu_i` is the kernel-weighted mean of the fluid
viscosity, `A_i = V_i^((d-1)/d)` is the same tributary area convention as the panel
pressure arm.

**Step 3 — score on two geometries.**  X3 (plane, linear field) isolates the
truncation deficit.  X5 (closed disk, *quadratic* field) is the only non-flat wall
in the bundle: the linear part of the traction integrates to zero over a closed
surface and the quadratic part gives `mu beta A`, so curvature error is measurable
against a non-degenerate scalar.  Both arms run on identical particle data;
`vel_solid` carries the no-slip value of the same field (see pitfall 1).

Toolchain: torch float64 CPU, deterministic lattices, no solver in the loop;
driver `benchmarks/xval_fsi2/run.py`, figures `make_fig_viscous.py`, this text
`make_doc_viscous.py` (both read `results.json` — no hand-copied number).

## 3. Results

### X3 plane Couette (reference `mu U/h L`)

| `dx` [m] | OSS one-sided (ported) | gradient-corrected (new) | continuum prediction `7/(3 pi) dx/h` |
|---|---|---|---|
| 0.025 | -38.54 % | -0.00 % | 0.5550 |
| 0.0125 | -38.19 % | -0.00 % | 0.5550 |
| 0.00625 | -38.01 % | -0.00 % | 0.5550 |

Measured OSS ratio to the exact traction: `0.6146` (dx independent);
kernel-quadrature prediction with the lattice factor `s/h` and `eps = 0.01`:
`0.5550`; closed form `7/(3 pi) = 0.7427` without the
`s/h` and regularisation factors.  The residual offset of the lattice value above
the continuum integral (+10.7 %) is the near-wall shell of the
integral that carries no particle (first fluid layer at `dx/2`; shaded band in the
right panel of the figure) — the lattice quadrature is coarser than the continuum
there, and no attribution beyond that is claimed.

### X5 curved closed disk, quadratic shear (reference `mu beta A`)

| `dx` [m] | `R/dx` | OSS one-sided | gradient-corrected |
|---|---|---|---|
| 0.1 | 5 | -38.10 % | +14.95 % |
| 0.05 | 10 | -42.03 % | +8.52 % |
| 0.025 | 20 | -42.33 % | +4.41 % |
| 0.0125 | 40 | -45.76 % | +1.92 % |

The OSS branch is *worse* on the curved wall and keeps drifting with refinement
(−38 % → −46 %): a fixed truncation moment times a local traction that does not
average out.  The corrected arm converges first order (`h/R`), i.e. the residual
is the one-sided reconstruction's curvature bias, not a normalisation error.

### Regression: the pressure arms are untouched

| case | `dx` | OSS pressure | panel (WLS) |
|---|---|---|---|
| vertical_wall | 0.05 | +27.59 % | -0.00 % |
| vertical_wall | 0.025 | +27.96 % | -0.00 % |
| vertical_wall | 0.0125 | +28.14 % | -0.00 % |
| horizontal_gate | 0.05 | +24.23 % | -0.00 % |
| horizontal_gate | 0.025 | +26.26 % | -0.00 % |
| horizontal_gate | 0.0125 | +27.29 % | -0.00 % |
| fsi2_insert_linear | 0.1 | +107.54 % | -0.70 % |
| fsi2_insert_linear | 0.05 | +83.32 % | +20.08 % |
| fsi2_insert_linear | 0.025 | +67.66 % | +32.26 % |

Identical to the numbers in `GAP-SPH-FSI-BENCHMARK_xval.md`, so the new arm is
additive: nothing in the pressure path moved.

## 4. Figures

`artifacts/xval_fsi2/fig_viscous_repair.png` (generated from `results.json`)

Left: load error vs `dx` for both arms on both cases (log axis, finer to the
right), 15 % gate in red and 3 % target in orange.  Right: the cumulative
truncation integral `-2 s int (dW/dr) r dr` as a function of `r/h`, showing the
plateau at the closed form, the measured lattice ratio, and the empty shell
`r < dx/2`.

## 5. Pitfalls (recorded so the next mission does not repeat them)

1. **The reconstruction is only meaningful with no-slip-consistent data.**  It
   reads `u_fluid - u_solid`; a solid velocity that is not the field value at the
   wall (e.g. `vel_solid = 0` inside a moving field) returns nonsense — my first
   X5 attempt did exactly this and produced `+282 / +550 / +1124 %` that *grew*
   with refinement.  That is invalid input, not a case; now pinned by
   `test_gradient_corrected_arm_needs_no_slip_consistent_solid_velocity`.
2. **A closed-body reference needs a fully wetted surface.**  The fsi2 insert body
   is a cylinder with a beam attached: the rear half of the cylinder is an
   *interior* (dry) surface, and giving it a one-sided load produces a spurious net
   force.  Use the isolated disk (X5) for curvature; the insert body stays for the
   pressure/Archimedes check only.
3. **Watch the lattice factor in any continuum estimate.**  `int (dW/dr) r dr` is
   `1/h` in 2-D, so the traction ratio must carry `s/h`; omitting it gives a
   dimensionally wrong `2.33` instead of `0.571`.  A dimensional check of the
   prediction against the measured ratio is the cheapest trap detector here.
4. **Curvature error is first order** in this arm (`h/R`): at `R/dx = 5` it is
   still ~15 %.  Expect to pay one refinement level for curved bodies.

## 6. Error judgement against the 3 % / 15 % standard

| quantity | reference | error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| X3 viscous load, corrected arm | `mu U/h L` | 1.9e-12 (machine) | **yes** | yes |
| X3 viscous load, OSS arm | `mu U/h L` | -38.54 % … -38.01 % | no | no |
| X5 disk load, corrected, `dx = 0.025` | `mu beta A` | +4.41 % | no | yes |
| X5 disk load, corrected, `dx = 0.0125` | `mu beta A` | +1.92 % | **yes** | yes |
| X5 disk load, OSS, `dx = 0.1` | `mu beta A` | -38.10 % | no | no |

Explicit answer: **the plane (flat-wall) viscous transfer is now exact**, and the
curved case reaches < 3 % only at `R/dx = 40`.  At the fsi2 benchmark resolution
(`dx = 0.1`, `R/dx = 5`) the corrected arm is +14.95 %: inside the 15 % gate, **not** inside 3 %.
Nothing here reports the coupled fsi2 transient (beam-tip history vs
`BeamObserver_Position_Run_0_result.xml`, time-averaged
`TotalViscousForceFromFluid = [0.451157, 0.00157671]`); that remains the round-2
target and still needs a total-Lagrangian solid, as stated in the pressure report.

## 7. Disposition

- Keep the OSS algebra verbatim as the **reference arm** (it is what the upstream
  code does) and never use it as the production viscous transfer.
- Use `wall_shear_from_gradient` for the viscous branch on flat or mildly curved
  walls; pair it with `panel_pressure_force` for the pressure branch.
- Do **not** use it with a non-no-slip `vel_solid` (pitfall 1).
- Artifacts: `src/tensorsph/fsi_force_from_fluid.py`,
  `benchmarks/xval_fsi2/{cases.py,run.py,make_fig_viscous.py,make_doc_viscous.py}`,
  `artifacts/xval_fsi2/{results.json,fig_viscous_repair.png}`,
  `tests/test_fsi_force_from_fluid.py`.

