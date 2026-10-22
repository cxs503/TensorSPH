# Benchmark plan: 2D droplet impact on a water pool

## Purpose

Exercise the WCSPH implementation with a free-surface impact problem and create a
reproducible artifact bundle. The current implementation is a prototype, not a validated
reference solution.

## Default case

| quantity | value |
|---|---|
| particle spacing `dx` | 0.025 m |
| smoothing length `h = 1.3 dx` | 0.0325 m |
| pool | 1.0 x 0.35 m, 560 particles |
| drop | radius 0.10 m, centre y = 0.49 m, 48 particles |
| wall particles | 291 (3 layers, bottom and both sides) |
| `c0`, `gamma`, `rho0` | 20 m/s, 7, 1000 kg/m^3 |
| `dt` | 4e-4 s (acoustic Courant 0.246) |
| steps / simulated time | 750 / 0.30 s |
| wall-clock | ~14 s on 2 CPU cores |
| drop free-fall time | ~0.13 s (derived from the discretised arrangement) |

The run is long enough to contain the whole event of interest: the drop falls, contacts the
pool at roughly t = 0.13 s, and forms a crater with a rising rim by t = 0.30 s.

## Current software checks

- kernel values are finite, nonnegative, and compactly supported;
- kernel gradient is finite at zero separation;
- pressure is non-negative everywhere (no tensile instability);
- wall particles lie outside the fluid box and restore missing kernel support;
- a run writes particle snapshots, diagnostics, and a run manifest with acceptance results;
- runs are bit-for-bit reproducible for a fixed configuration.

These checks demonstrate software execution and numerical consistency. They do not establish
physical accuracy.

## Numerical acceptance criteria (enforced)

Recorded in `run_manifest.json` under `acceptance.criteria`:

| id | metric | limit | rationale |
|---|---|---|---|
| `finite_state` | all diagnostics finite | required | catches blow-up |
| `density_compression` | max(rho/rho0 - 1) | 5 % | WCSPH assumes weak compressibility; the pre-fix run reached +19 % |
| `mach_number` | max&#124;v&#124; / c0 | 0.10 | above this the weakly-compressible assumption fails; the pre-fix run reached 1.1 |
| `acoustic_courant` | max dt(c0+&#124;v&#124;)/h | 0.30 | explicit stability limit |
| `boundary_escape` | particles clamped beyond the box | 0 | with wall particles the fluid should never need the safety clamp |

Measured on the default case: 2.2 %, 0.075, 0.265, 0 respectively. `fail_fast` raises
`SimulationUnstableError` after the artifacts are written, so a failing run is still
inspectable.

`physical_validation` stays `not_evaluated` in every run: no published experiment is
compared against yet.

## Required next validation gates

1. **Impact trajectory:** report drop centre of mass, maximum spread, and free-surface elevation over time. These are diagnostics today, not yet pass/fail criteria.
2. **Conservation/stability:** monitor mass, momentum, kinetic energy, potential energy, and density extrema over the full event; define tolerances before accepting a run.
3. **Resolution study:** at least three particle spacings with convergence of selected observables.
4. **Reference comparison:** compare impact time, spread ratio, and shape evolution against a cited experimental or published numerical case with matching dimensionless parameters.

## Known limitations

- No surface-tension force. Free-surface pressure is clamped at zero, which suppresses tensile instability but is not a capillary model, so the rim and any droplet break-up are not quantitatively meaningful.
- Walls are static dummy particles with free-slip viscosity: no wall friction, no moving boundaries.
- The weakly-compressible formulation at `c0 = 20 m/s` corresponds to roughly 1 % density fluctuation at hydrostatic pressure; capillary-dominated features need a much smaller time step if they were ever added.
- Pair construction still copies coordinates to CPU and dominates the per-step cost.
- The current case uses a 2D cross-section and must not be interpreted as a 3D physical drop.

## Artifact contract

Each run preserves `run_manifest.json`, `diagnostics.csv`, particle snapshots, plots when
Matplotlib is available, the exact configuration, device, seed, and source revision (when
run from a git checkout). Future TensorPhysics/TensorPlatform adapters should register these
outputs rather than treating a successful process exit as scientific acceptance.

## Diagnostic fields

`diagnostics.csv` records total particle mass, both momentum components, density extrema and
mean, maximum relative density deviation from the reference density, pressure extrema,
maximum speed, maximum acceleration, Mach number, kinetic energy, the free-surface extent,
and the acoustic Courant number

`C = dt (c0 + |v|_max) / h`.

These feed the acceptance criteria above. In particular, a *low* Courant number does not by
itself indicate stability: the pre-fix run held C between 0.012 and 0.026 while the flow
disintegrated, because the instability was caused by tensile pressure rather than by the
time step.
