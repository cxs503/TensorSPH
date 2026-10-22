# Benchmark plan: 2D droplet impact on a water pool

## Purpose

Exercise the first TensorSPH WCSPH implementation with a free-surface impact problem and create a reproducible artifact bundle. The current implementation is a prototype, not a validated reference solution.

## Current software checks

- kernel values are finite, nonnegative, and compactly supported;
- kernel gradient is finite at zero separation;
- density, pressure, and acceleration are finite for the initial condition;
- a short CPU run writes particle snapshots, diagnostics, and a run manifest.

These checks demonstrate software execution only. They do not establish physical accuracy.

## Required next validation gates

1. **Static droplet / hydrostatic pool:** quantify density error and pressure profile; check sensitivity to particle spacing and smoothing length.
2. **Impact trajectory:** report drop center of mass, maximum spread, and free-surface elevation over time.
3. **Conservation/stability:** monitor mass, momentum, kinetic energy, potential energy, and density extrema; define tolerances before accepting a run.
4. **Resolution study:** compare at least three particle spacings and report convergence of selected observables.
5. **Reference comparison:** compare impact time, spread ratio, and shape evolution against a cited experimental or published numerical case with matching dimensionless parameters.

## Known limitations in the first prototype

- all-pairs interactions scale quadratically and are only suitable for small smoke tests;
- no surface-tension force is implemented;
- boundary treatment is a simple domain clamp, not a validated wall/free-surface boundary condition;
- the time step is fixed and not yet checked against an explicit stability criterion;
- the current case uses a 2D cross-section and must not be interpreted as a 3D physical drop.

## Artifact contract

Each run should preserve `run_manifest.json`, `diagnostics.csv`, particle snapshots, plots when Matplotlib is available, the exact configuration, device, seed, and source revision. Future TensorPhysics/TensorPlatform adapters should register these outputs rather than treating a successful process exit as scientific acceptance.

## Diagnostic fields added in the next increment

The run CSV now also records total particle mass, both momentum components, maximum relative density deviation from the reference density, maximum speed, maximum acceleration, and the acoustic Courant number

`C = dt (c_0 + |v|_{max}) / h`.

These are monitoring signals, not pass/fail criteria yet. In particular, a low Courant number does not by itself guarantee stability, and clamped boundaries can change momentum. A future validation stage must define case-specific thresholds and use physically appropriate wall treatment before calling this a validated benchmark.
