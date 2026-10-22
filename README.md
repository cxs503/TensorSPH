# TensorSPH

**PyTorch-native smoothed particle hydrodynamics (SPH) for agent-driven scientific computing.**

TensorSPH is the SPH solver project in the Tensor ecosystem. It is intended to provide reproducible, testable particle-based fluid simulations that can later be orchestrated by TensorPhysics and recorded by TensorPlatform.

## First milestone: 2D droplet impact on a water pool

The first implementation is a small, CPU/GPU-capable weakly-compressible SPH (WCSPH) prototype built directly on PyTorch:

- reproducible initial particle arrangement and configuration;
- cubic-spline kernel and pressure/viscous force evaluation;
- gravity and simple domain clamping (not yet validated wall boundary conditions);
- finite-state checks and diagnostic probes for mass, momentum, density, speed, acceleration, and acoustic Courant number;
- particle snapshots, diagnostic CSV, run manifest, and density plots;
- unit tests and a GitHub Actions workflow for smoke execution.

## Quick start

Install the package and test dependencies, then run the example:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python examples/droplet_impact_2d/run.py --steps 100 --output artifacts/droplet_impact_2d
```

A run writes `diagnostics.csv`, `run_manifest.json`, particle snapshot files, and (when Matplotlib is available) `particle_snapshots.png`.

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

## Current limitations

- **Prototype only:** no physical-validation claim is made.
- No surface-tension model; impact morphology should not be treated as a quantitatively accurate splash prediction.
- All-pairs interactions scale quadratically and are suitable only for small cases.
- Simple position clamping is not a validated wall or free-surface boundary treatment.
- Fixed time step is monitored but not automatically selected from a stability criterion.
- Published benchmark matching, resolution convergence, and acceptance thresholds remain to be implemented.

## Development roadmap

See the [algorithm gap analysis and validation roadmap](docs/algorithm-gap-analysis.md) for the prioritized gaps, benchmark ladder, acceptance contract, and proposed implementation sequence. For the current impact case, see the [droplet-impact benchmark plan](docs/benchmarks/droplet-impact-2d.md).
