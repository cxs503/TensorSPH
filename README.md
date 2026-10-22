# TensorSPH

**PyTorch-native smoothed particle hydrodynamics (SPH) for agent-driven scientific computing.**

TensorSPH is the SPH solver project in the Tensor ecosystem. It is intended to provide reproducible, testable particle-based fluid simulations that can later be orchestrated by TensorPhysics and recorded by TensorPlatform.

## First milestone: 2D droplet impact on a water pool

The first implementation milestone is a small, CPU/GPU-capable weakly-compressible SPH (WCSPH) case built directly on PyTorch:

- reproducible initial particle arrangement and configuration;
- cubic-spline kernel and pressure/viscous force evaluation;
- gravity and simple solid-wall treatment;
- time integration with finite-value and stability checks;
- particle snapshots, diagnostic CSV, and plots;
- tests for kernel normalization trends, finite forces, and a short smoke run.

This first case is a **numerical prototype**, not yet a validated engineering solver. Surface tension, robust free-surface consistency, scalable neighbor search, and quantitative comparison against published droplet-impact data are subsequent validation milestones.

## Planned layout

- `src/tensorsph/`: reusable SPH operators and solver
- `examples/droplet_impact_2d/`: case configuration and runner
- `tests/`: operator and smoke tests
- `docs/benchmarks/`: benchmark definitions, metrics, and validation evidence

## Development principles

1. PyTorch tensors are the primary compute representation; use CUDA when available.
2. Keep case setup, numerical operators, time integration, and output separate.
3. Every benchmark records configuration, seed, software revision, diagnostics, and artifacts.
4. Distinguish software execution success from numerical verification and physical validation.
5. Integrate with TensorPhysics/TensorPlatform through explicit run metadata and artifacts, without making the solver depend on the platform.

## Status

Repository bootstrap. The first runnable WCSPH droplet-impact case is under development; no physical-validation claim is made at this stage.
