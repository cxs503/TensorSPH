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
- **non-negative pressure** at free surfaces (no tensile instability), plus an optional diffusive density term (`Config.density_diffusion`, Antuono et al. 2012) measured against it in [`docs/benchmarks/free-surface-treatment.md`](docs/benchmarks/free-surface-treatment.md);
- **static wall particles** (Adami et al., 2012) with extrapolated wall pressure and Tait-consistent wall density;
- **device-resident linked-cell neighbour search** plus a Verlet skin cache: no host round-trip, no Python loop over particles;
- **calibrated viscosity**: the discrete operator applies the kinematic viscosity it is given (verified against a known Laplacian);
- **free-slip or no-slip walls** (`Config.wall_friction`), verified against the analytic Stokes shear-decay rate;
- time step validated against **three** limits — acoustic Courant, viscous (`0.125 h²/ν`) and gravity — with the binding limit reported on failure;
- stability gates judged on **every integration step**, not on the steps that happen to write a diagnostic row;
- wall-wall neighbour pairs pruned: they cannot reach any retained quantity (verified bit-exact);
- diagnostics for mass, momentum, density, speed, acceleration, Mach number, and Courant number;
- run manifest with separate `software_execution` / `numerical_verification` / `physical_validation` statuses, per-criterion acceptance results, and the population each criterion was measured over;
- particle snapshots, diagnostic CSV, density plots, and JSON benchmark reports.

## Quick start

```bash
python -m pip install -e ".[test]"   # torch, pytest, matplotlib, ruff
# a runtime-only install is `pip install .`; the figure needs `.[plot]`
python -m pytest -q

# droplet impact: 750 steps, t = 0.3 s, ~4.5 s on two CPU cores (3.5 s with --no-plot)
python -m tensorsph --output artifacts/droplet_impact_2d

# dam break: front trajectory, convergence and a compressibility sweep, ~70 s
python benchmarks/dam_break/run.py --sizes 0.02,0.01 --seconds 0.65

# hydrostatic verification of the wall model, ~60 s
python benchmarks/hydrostatic/run.py --seconds 4 --boundary wall

# free-surface treatments side by side, ~2 min
python benchmarks/free_surface/compare.py --deltas 0,0.1
```

The default impact case is long enough for the drop to free-fall, contact the pool at roughly t = 0.13 s, and form a crater and rim. `Config.dt` must satisfy the acoustic, viscous and gravity limits; `run` raises `ValueError` naming the binding limit and the largest allowed step if it does not, and `suggested_dt(cfg)` / `viscous_dt_limit(cfg)` / `gravity_dt_limit(cfg)` return the numbers without running anything.

## Verification status

| Gate | Status | Evidence |
|---|---|---|
| Software execution | passing | 90 unit tests, ruff, CI on Python 3.10 / 3.11 / 3.12, smoke run |
| Neighbour pairs | passing | exact match with brute force on deterministic 1-D/2-D/3-D clouds and with an independent host implementation |
| Pair-based RHS | passing | matches the dense reference to `rtol=2e-10` in float64 |
| B1 hydrostatics | **passing, 1.09 %** time-averaged interior error | [`docs/benchmarks/hydrostatic.md`](docs/benchmarks/hydrostatic.md) |
| Viscous operator | passing | applies the nominal `viscosity` to 0.1 % on a field with a known Laplacian |
| No-slip wall | passing | quarter-wave shear decays at 0.84x (dx 0.02) and 0.91x (dx 0.01) of the analytic Stokes rate |
| B2 dam-break convergence | **passing, observed order 0.96** (L2 2.58 % then 1.31 %) | [`docs/benchmarks/dam-break-2d.md`](docs/benchmarks/dam-break-2d.md) |
| B2 compressibility | passing | front moves 0.019 % of `a` between `c0` 40 and 60 m/s |
| B2 density compression | **failing, 10.0–10.3 %** against a 10 % limit | peak is a wall-impact transient; bulk flow stays at 2 %. See below |
| Free-surface treatment comparison | passing (numerics only) | clamp vs density diffusion on B1, B2 and clumping: interior error 1.086 % → 0.943 %, worst pair separation 0.052 → 0.089 `dx`, +30 % cost. Default unchanged |
| B2 experimental comparison | **not evaluated** — no digitized reference bundled | gate named in the same document |
| Physical validation | **not evaluated** | no published experiment has been reproduced |

Passing a numerical gate is not a claim that a simulation is physically accurate.

The B2 density gate is reported as failing on purpose. Until 0.5.0 it read 3.8 %
because the metric was sampled every 0.01 s; judged on every step it is 10.0 %
at `dx = 0.01`, and 2 % once the window stops before the surge hits the far
wall. The whole-run metric is left unchanged and failing rather than redefined.
[`docs/changes-0.5.0.md`](docs/changes-0.5.0.md) records the change and
[`docs/benchmarks/dam-break-2d.md`](docs/benchmarks/dam-break-2d.md) the
measurements.

The optional density-diffusion term added in 0.6.0 was measured against the
pressure clamp rather than substituted for it, and did not earn a default change:
it improves the worst pair separation by 71 % and costs 30 %, but the hydrostatic
interior error improves only within the spread of choosing a different band set
and the free-surface band gets worse. `Config.density_diffusion = 0.1` is
available for cases where near-coincident particles matter. The comparison, and
what would change the conclusion, is in
[`docs/benchmarks/free-surface-treatment.md`](docs/benchmarks/free-surface-treatment.md).

## Current limitations

- **Prototype:** no physical-validation claim is made, and no benchmark is compared against measured data.
- No surface-tension model. Free-surface pressure is clamped at zero, which removes tensile instability but is not a capillary model. An optional density-diffusion term is available and measured, but it smooths density rather than creating a capillary pressure jump.
- Walls are three layers of static dummy particles. Free slip is the default; `Config.wall_friction = 1.0` gives a verified no-slip condition but is not the default and has not been tuned.
- Density compression in the dam-break case peaks at 9.7–10.3 % across the three-level ladder, against a 10 % limit. The peak is the surge hitting the far wall (measured at t = 0.66 s, maximum 17 ms later); before impact the bulk stays at 1.8–2.4 %. `c0 = 40 m/s` brings the peak to 6.1 % without moving the front.
- A pool in a free-slip basin rings in a sloshing mode with a `H²/nu ≈ 12 s` damping time, so hydrostatic profiles are only stable when time-averaged.
- Resolution convergence is measured on a three-level ladder; the observed order is 0.96 for the dam-break front. No other observable has a convergence study.
- CUDA execution is exercised by construction but has never been benchmarked on a GPU in this repository.

## Layout

- `src/tensorsph/`: SPH operators, droplet-impact case (`python -m tensorsph`), dam-break case
- `examples/droplet_impact_2d/`: case runner
- `benchmarks/`: executable benchmark and comparison scripts (`neighbor_search`, `hydrostatic`, `dam_break`, `free_surface`)
- `docs/benchmarks/`: benchmark definitions, metrics, and measured evidence
- `docs/changes-0.6.0.md`: free-surface comparison and why the default did not change
- `docs/changes-0.5.0.md`: what changed in 0.5.0 and what it measured
- `docs/analysis-2026-10-23.md`: independent review of the 0.4.0 tree
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
