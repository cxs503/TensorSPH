# TensorSPH

**PyTorch-native smoothed particle hydrodynamics (SPH) for agent-driven scientific computing.**

TensorSPH is the SPH solver project in the Tensor ecosystem. It is intended to provide reproducible, testable particle-based fluid simulations that can later be orchestrated by TensorPhysics and recorded by TensorPlatform.

## First milestone: 2D droplet impact on a water pool

The first implementation is a small, CPU/GPU-capable weakly-compressible SPH (WCSPH) prototype built directly on PyTorch:

- reproducible lattice initialisation of a water pool and a falling drop;
- cubic-spline kernel, density summation, Tait equation of state, pressure/viscous forces;
- **non-negative pressure** at the free surface (no tensile instability);
- **static wall particles** (Adami et al., 2012) on the bottom and both sides instead of coordinate clamping;
- **time step chosen from an acoustic Courant limit**, enforced at start-up;
- diagnostics for mass, momentum, density, speed, acceleration, Mach number, and Courant number;
- particle snapshots, diagnostic CSV, run manifest, and density plots;
- explicit acceptance gates that separate software execution, numerical verification, and physical validation;
- unit tests, a hydrostatic benchmark, and a GitHub Actions workflow.

## Quick start

Install the package and test dependencies, then run the example:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m tensorsph --output artifacts/droplet_impact_2d
# or, equivalently:
python examples/droplet_impact_2d/run.py --output artifacts/droplet_impact_2d
```

The default case runs 750 steps at `dt = 4e-4 s` (t = 0.3 s), which is long enough for the drop to free-fall, contact the pool at roughly t = 0.13 s, and form a crater and rim. A run writes `diagnostics.csv`, `run_manifest.json`, particle snapshot files, and (when Matplotlib is available) `particle_snapshots.png`.

The time step is not free: `Config.dt` must satisfy `dt * c0 / h <= max_courant`, and `run` raises `ValueError` with the largest allowed step if it does not. `suggested_dt(cfg)` returns that value.

## Planned layout

- `src/tensorsph/`: reusable SPH operators and solver (`python -m tensorsph` runs the case)
- `examples/droplet_impact_2d/`: case configuration and runner
- `tests/`: operator, boundary, artifact, and smoke tests
- `docs/benchmarks/`: benchmark definitions, metrics, and validation evidence
- `benchmarks/`: executable benchmark and comparison scripts

## Development principles

1. PyTorch tensors are the primary compute representation; use CUDA when available.
2. Keep case setup, numerical operators, time integration, and output separate.
3. Every benchmark records configuration, seed, software revision, diagnostics, and artifacts.
4. Distinguish software execution success from numerical verification and physical validation.
5. Integrate with TensorPhysics/TensorPlatform through explicit run metadata and artifacts, without making the solver depend on the platform.

## Verification status

| Gate | Status | Evidence |
|---|---|---|
| Software execution | passing | 49 unit tests, CI on Python 3.10 and 3.12, smoke run |
| Numerical verification: neighbour pairs | passing | exact match with brute force on deterministic 1-D/2-D/3-D clouds |
| Numerical verification: pair-based RHS | passing | matches the dense reference to `rtol=2e-10` in float64 |
| Numerical verification: hydrostatics (B1) | **passing, 0.63 % interior error** | [`docs/benchmarks/hydrostatic.md`](docs/benchmarks/hydrostatic.md) |
| Physical validation | **not evaluated** | no published experiment has been reproduced |

Passing a numerical gate is not a claim that a simulation is physically accurate.

## Current limitations

- **Prototype only:** no physical-validation claim is made.
- No surface-tension model. Free-surface pressure is clamped at zero, which removes tensile instability but is not a capillary model; impact morphology and any capillary-driven feature should not be treated as quantitative.
- Walls are three layers of static dummy particles with an extrapolated pressure field and free-slip viscosity. Hydrostatic accuracy is measured (0.63 % time-averaged interior error), but no dam-break or wave-impact validation has been done.
- The pool does not come to rest: with free-slip walls and no state relaxation there is a persistent residual motion of a few cm/s.
- Pair construction still copies coordinates to CPU (`positions.detach().cpu().tolist()`), so it dominates the per-step cost. Pair-based RHS is asymptotically cheaper than the dense reference, but no GPU speedup has been demonstrated.
- Resolution convergence and acceptance against published experiments remain to be implemented.
- `physical_validation` in `run_manifest.json` is always `not_evaluated`.

## Development roadmap

See the [algorithm gap analysis and validation roadmap](docs/algorithm-gap-analysis.md) for prioritized gaps and the benchmark ladder. Utilities:

```bash
python benchmarks/neighbor_search/compare.py --particles 1000 --dimension 2
python benchmarks/neighbor_search/compare_rhs.py --sizes 256,512,1024
python benchmarks/hydrostatic/run.py --seconds 4 --boundary wall
```

For the impact case, see the [droplet-impact benchmark plan](docs/benchmarks/droplet-impact-2d.md) and the [neighbor-search baseline](docs/benchmarks/neighbor-search.md).
