# TensorSPH

**PyTorch-native smoothed particle hydrodynamics (SPH) for agent-driven scientific computing.**

TensorSPH is the SPH solver project in the Tensor ecosystem. It is intended to provide reproducible, testable particle-based fluid simulations that can later be orchestrated by TensorPhysics and recorded by TensorPlatform.

## Implemented

Two weakly-compressible SPH cases share one set of verified operators:

| case | status |
|---|---|
| 2-D droplet impact on a pool | runs end to end; numerically bounded; **not** compared with experiment |
| 2-D dam break (B2) | front converges at observed order 0.96 and is insensitive to `c0`; **no** experimental gate |

Shared machinery:

- cubic-spline kernel, density summation, Tait equation of state, symmetric pressure forces, regularized laminar viscosity;
- **non-negative pressure** at free surfaces (no tensile instability);
- **static wall particles** (Adami et al., 2012) with extrapolated wall pressure and Tait-consistent wall density;
- **device-resident linked-cell neighbour search** plus a Verlet skin cache: no host round-trip, no Python loop over particles;
- **calibrated viscosity**: the discrete operator applies the kinematic viscosity it is given (verified against a known Laplacian);
- **free-slip or no-slip walls** (`Config.wall_friction`), verified against the analytic Stokes shear-decay rate;
- time step validated against an acoustic Courant limit, with the limit reported on failure;
- diagnostics for mass, momentum, density, speed, acceleration, Mach number, and Courant number;
- run manifest with separate `software_execution` / `numerical_verification` / `physical_validation` statuses and per-criterion acceptance results;
- particle snapshots, diagnostic CSV, density plots, and JSON benchmark reports.

## Quick start

```bash
python -m pip install -e ".[test]"
python -m pytest -q

# droplet impact: 750 steps, t = 0.3 s, ~5 s on two CPU cores
python -m tensorsph --output artifacts/droplet_impact_2d

# dam break: front trajectory and resolution convergence, ~40 s
python benchmarks/dam_break/run.py --sizes 0.02,0.01 --seconds 0.8

# hydrostatic verification of the wall model, ~60 s
python benchmarks/hydrostatic/run.py --seconds 4 --boundary wall
```

The default impact case is long enough for the drop to free-fall, contact the pool at roughly t = 0.13 s, and form a crater and rim. `Config.dt` must satisfy `dt * c0 / h <= max_courant`; `run` raises `ValueError` with the largest allowed step if it does not, and `suggested_dt(cfg)` returns that value.

## Verification status

| Gate | Status | Evidence |
|---|---|---|
| Software execution | passing | 67 unit tests, CI on Python 3.10 and 3.12, smoke run |
| Neighbour pairs | passing | exact match with brute force on deterministic 1-D/2-D/3-D clouds and with an independent host implementation |
| Pair-based RHS | passing | matches the dense reference to `rtol=2e-10` in float64 |
| B1 hydrostatics | **passing, 1.09 %** time-averaged interior error | [`docs/benchmarks/hydrostatic.md`](docs/benchmarks/hydrostatic.md) |
| Viscous operator | passing | applies the nominal `viscosity` to 0.1 % on a field with a known Laplacian |
| No-slip wall | passing | quarter-wave shear decays at 0.84x (dx 0.02) and 0.91x (dx 0.01) of the analytic Stokes rate |
| B2 dam-break convergence | **passing, observed order 0.96** (L2 2.58 % then 1.31 %) | [`docs/benchmarks/dam-break-2d.md`](docs/benchmarks/dam-break-2d.md) |
| B2 compressibility | passing | front moves 0.019 % of `a` between `c0` 40 and 60 m/s |
| B2 experimental comparison | **not evaluated** — no digitized reference bundled | gate named in the same document |
| Physical validation | **not evaluated** | no published experiment has been reproduced |

Passing a numerical gate is not a claim that a simulation is physically accurate.

## Current limitations

- **Prototype:** no physical-validation claim is made, and no benchmark is compared against measured data.
- No surface-tension model. Free-surface pressure is clamped at zero, which removes tensile instability but is not a capillary model.
- Walls are three layers of static dummy particles. Free slip is the default; `Config.wall_friction = 1.0` gives a verified no-slip condition but is not the default and has not been tuned.
- Density compression in the dam-break case reaches 9.7–9.9 % at `dx` = 0.04–0.02 against a 10 % limit (3.8 % at `dx` = 0.01), so coarse-grid results sit at the edge of the weakly-compressible assumption. The front itself is insensitive to `c0`, so this does not affect the reported convergence.
- A pool in a free-slip basin rings in a sloshing mode with a `H²/nu ≈ 12 s` damping time, so hydrostatic profiles are only stable when time-averaged.
- Resolution convergence is measured on a three-level ladder; the observed order is 0.96 for the dam-break front. No other observable has a convergence study.
- CUDA execution is exercised by construction but has never been benchmarked on a GPU in this repository.

## Layout

- `src/tensorsph/`: SPH operators, droplet-impact case (`python -m tensorsph`), dam-break case
- `examples/droplet_impact_2d/`: case runner
- `benchmarks/`: executable benchmark and comparison scripts (`neighbor_search`, `hydrostatic`, `dam_break`)
- `docs/benchmarks/`: benchmark definitions, metrics, and measured evidence
- `tests/`: operator, boundary, cache, case, artifact, and smoke tests

## Development principles

1. PyTorch tensors are the primary compute representation; use CUDA when available.
2. Keep case setup, numerical operators, time integration, and output separate.
3. Every benchmark records configuration, software revision, diagnostics, and artifacts.
4. Distinguish software execution success from numerical verification and physical validation.
5. A benchmark passes only on recorded thresholds; missing thresholds are `not_evaluated`, never silently treated as pass.
6. Integrate with TensorPhysics/TensorPlatform through explicit run metadata and artifacts, without making the solver depend on the platform.

## Roadmap

See the [algorithm gap analysis and validation roadmap](docs/algorithm-gap-analysis.md), the [engineering review](docs/review-2026-10.md), and the per-benchmark documents under `docs/benchmarks/`.
