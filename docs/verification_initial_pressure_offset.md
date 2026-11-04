# Verification debt: the initial density offset of a summation-density lattice

**Scope**: TensorSPH, `src/tensorsph/initial_conditions.py` +
`benchmarks/init_conditioning/`. **Status**: closed for the bulk, open at the
free surface (see "What this does *not* fix").

**Numbers in this document are injected by `benchmarks/init_conditioning/make_doc.py`
from `report.json`; nothing is typed by hand.** Figure: `init_conditioning.png`.

## 1. The claim under test

`backlog_update_m2971.md` item C2/P1 states: every weakly compressible SPH case
in this solver starts with a spurious pressure because the particle mass is
`rho0 * V0` while the *discrete* kernel sum `C = sum_j W_ij V0` is not exactly
one. The phenomenon matters because `c0 = 20 m/s` turns a sub-percent density
offset into kilopascals, which is larger than the capillary pressure the
verification suite is trying to measure.

Acceptance criterion (from the same backlog item, machine-checkable):

```
max|p| / (rho0 c0^2) < 1e-4      after initialisation
```

## 2. Method

Pure `t = 0` evaluation, no time stepping: a static disc is placed on a
hexagonal and a square lattice (`rho0 = 1000 kg/m^3`,
`R = 0.02 m`, `sigma = 7.2 N/m` so that
`sigma/R = 360 Pa`, the geometry of
`benchmarks/laplace_pressure_floor`), the summation density is taken with the
solver's own `density_from_neighbor_pairs`, and the pressure with the solver's
own Tait law (clamped at `p >= 0`).

The offset `b` is measured only on the **fully supported** particles - those
whose neighbour count equals the maximum count in the configuration
(`initial_conditions.supported_mask`, tolerance 0). Particles next to the free
surface must be excluded: their kernel sum is legitimately below one, and
averaging them in biases the fit (measured: a tolerance of two neighbours left
the strictly interior at `+7e-4` in `rho/rho0` instead of zero).

Calibration is `m -> m / (1 + b)`, and it is **exact in one pass**: the
summation density is linear in the masses, so scaling all masses by `alpha`
scales all densities by `alpha`. Verified: after calibration the interior
satisfies `|rho/rho0 - 1| < 6.0e-14`.

## 3. Results

### 3.1 The offset is a lattice constant, not a discretization error

Hexagonal lattice, `h = 1.3 dx`, radius 0.02 m:

| lattice | h/dx | dx [m] | N | fully supported | b [%] | max|p| [Pa] | before | after |
|---|---|---|---|---|---|---|---|---|
| hex | 1.3 | 0.004 | 87 | 33 | +0.5884 | 2395.4 | 5.988e-03 | 9.1e-15 |
| hex | 1.3 | 0.002 | 363 | 237 | +0.5884 | 2395.4 | 5.988e-03 | 2.0e-14 |
| hex | 1.3 | 0.001 | 1449 | 1185 | +0.5884 | 2395.4 | 5.988e-03 | 6.0e-14 |

`b` is identical at every spacing to the printed precision
(`hex_bias_dx_independent = True`), so refining the
mesh does **not** remove it. This is the key qualitative point: the offset is a
property of the packing, not of the resolution.

### 3.2 Spurious pressure at the verification geometry

| c0 [m/s] | max|p| before [Pa] | normalised | max|p| after [Pa] | x sigma/R |
|---|---|---|---|---|
| 5 | 149.7 | 5.9884e-03 | 5.00e-10 | 0.42 |
| 10 | 598.8 | 5.9884e-03 | 2.00e-09 | 1.66 |
| 20 | 2395.4 | 5.9884e-03 | 7.99e-09 | 6.65 |
| 40 | 9581.4 | 5.9884e-03 | 3.20e-08 | 26.62 |

The absolute offset goes exactly as `c0^2` (the normalised criterion is
constant, `5.9884e-03` for every `c0`), so lowering the
sound speed shrinks the pressure but never the *relative* error. At
`c0 = 20 m/s` the offset is `6.65 x` the
`sigma/R` signal - the number quoted as 6.7x in the backlog is confirmed.

### 3.3 Lattice / smoothing-length dependence

| lattice | h/dx | b [%] | max|p|/(rho0 c0^2) | mass change [%] |
|---|---|---|---|---|
| hex | 1.2 | +0.3225 | 3.256e-03 | -0.321 |
| hex | 1.3 | +0.5884 | 5.988e-03 | -0.585 |
| hex | 1.4 | +0.4171 | 4.223e-03 | -0.415 |
| hex | 1.5 | +0.1718 | 1.727e-03 | -0.171 |
| square | 1.2 | -0.0243 | 0.000e+00 | +0.024 |
| square | 1.3 | -0.0053 | 0.000e+00 | +0.005 |
| square | 1.4 | +0.2079 | 2.092e-03 | -0.207 |
| square | 1.5 | +0.3440 | 3.476e-03 | -0.343 |

(square lattice at `h = 1.3 dx` is the one combination that happens to be
already balanced; it is also the packing used by `make_particles`, which is why
the droplet-impact production case never showed the effect.)

### 3.4 After calibration

* worst normalised offset over the whole sweep: `6.0e-14`
  against the `0.0001` limit (`all_pass_after =
  True`);
* cost: total mass moves by `-0.585 %` for the hex
  lattice - exact mass conservation against the analytic `rho0 * Area` is given
  up for a pressure-free initial state. This is the documented trade-off:
  the calibration preserves *density*, not *mass*.

### 3.5 What the calibration does **not** change

The merged CSF jump measurement (`benchmarks/laplace_droplet`, `dx =
0.01 m`) moves by `+0.79` percentage
points only (`dp_ratio` from `0.1069` to
`0.1149`; `N_delta` is unchanged at
`0.0000` because the colour field does not depend on
a uniform density scale). So the residual 16-21 % deficit of the force-integral
jump is **not** caused by C2 - attribution stays where `m2971` left it (the
free-surface pressure closure). What C2 does contaminate is every *pressure*
reading taken at or near `t = 0`.

## 4. API

```python
from tensorsph.initial_conditions import (
    TAIT_ACCEPTANCE_LIMIT, calibrate_lattice_mass, measure_density_bias,
    spurious_pressure, supported_mask,
)

bias = measure_density_bias(pos, mass, h, rho0)          # dict
mass, info = calibrate_lattice_mass(pos, mass, h, rho0)  # exact in one pass
spurious_pressure(pos, mass, h, rho0, c0, gamma)         # dict, 'passes_acceptance'
```

`Config.lattice_mass_calibration` (default `False`) wires the calibration into
`make_particles`. It is **off by default** on purpose: it shifts every published
number by `-b/(1+b)` and the production tank uses the square lattice, whose
offset is only `-0.005 %`. Turning it on is a re-baselining decision, not a bug
fix, and the pinned droplet-impact numbers would have to be re-measured.

## 5. What this does *not* fix

1. **Free-surface deficit.** The drop below `r/R ~ 0.8` in figure (b) is the
   kernel deficit of the surface layer (`rho/rho0` down to ~0.6). The Tait clamp
   turns it into `p = 0`, which is a first-order free-surface treatment, not a
   closure. That is item P0 of `backlog_update_m2971.md` and remains **open**.
2. **Non-uniform lattices.** A jittered or adaptively refined particle set has a
   *distribution* of kernel sums, not one constant; a scalar calibration then
   only removes the mean. `measure_density_bias` reports `rho_ratio_min/max` so
   the spread is visible.
3. **Walls.** `wall_mass` keeps `rho0 dx^2`; walls use the extrapolated pressure
   of Adami et al. (2012) and this benchmark does not measure them.

## 6. Reproduction

```bash
PYTHONPATH=src python benchmarks/init_conditioning/run.py     # ~21 s, 1 core
PYTHONPATH=src python benchmarks/init_conditioning/make_doc.py
PYTHONPATH=src python -m pytest tests/test_init_conditioning.py -q
```
