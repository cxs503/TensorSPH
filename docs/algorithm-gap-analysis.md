# TensorSPH algorithm gap analysis and validation roadmap

**Status:** engineering assessment of the current prototype, not a claim of validated SPH accuracy.
**Scope:** 2-D PyTorch WCSPH prototype (droplet impact and dam break) at version 0.6.0.

## Executive decision

Keep TensorSPH as a PyTorch-native solver whose numerical operators, case definitions, diagnostics, and run artifacts are explicit and machine-readable. Study mature projects for algorithm choices and validation methodology; do not copy their implementation wholesale.

Recommended references:

- [PySPH](https://github.com/pypr/pysph): equation-oriented architecture, neighbor search, integrators, wall conditions, and a catalog of reproducible examples. Its official project description notes Python-defined equations with performance-critical execution through Cython/OpenCL. PyPI currently lists BSD licensing; verify the exact source-file headers and bundled dependencies before reusing any code.
- [DualSPHysics](https://github.com/DualSPHysics/DualSPHysics): engineering-scale free-surface cases, GPU execution, and practical case setup. The repository identifies LGPL-2.1; review the applicable license obligations before any code reuse.
- [SPHinXsys](https://github.com/xiangyu-hu/sphinxsys): architecture for coupled multi-physics and fluid-structure interaction. The repository identifies Apache-2.0; verify dependency and file-level terms before reuse.
- Adami, Hu & Adams (2012), *A generalized wall boundary condition for smoothed particle hydrodynamics*, J. Comput. Phys. 231:7057–7075 — the wall-particle pressure extrapolation used here.
- Monaghan (2000), *SPH without a tensile instability*, J. Comput. Phys. 159:290–311 — the artificial-stress remedy for the free-surface problem documented below.

The first milestone is *verification and diagnostics*, not adding more visually impressive splash effects.

## Current implementation baseline

The solver has a 2-D cubic-spline kernel, density summation, Tait equation of state with
**non-negative pressure**, symmetric pressure-force form, regularized viscosity, gravity,
three layers of static wall particles with Adami pressure extrapolation, a fixed time step
that is validated against an acoustic Courant limit, and semi-implicit Euler integration.
It writes particle snapshots, a diagnostics CSV, a run manifest with separate statuses and
machine-readable acceptance criteria, and density plots.

CI passing on Python 3.10 and 3.12 confirms the covered software paths execute. It does
**not** establish conservation, convergence, or agreement with experiment.

## What changed in 0.2.0

The previous increment was described as "no scientifically valid run is currently
possible". The root cause was diagnosed and fixed:

| Defect | Effect before | Fix |
|---|---|---|
| Tait EOS applied to free-surface rarefaction | initial pressure of **-56 kPa** on 100 % of particles; initial acceleration 4400 m/s^2 (450 g), all of it spurious; particle clumping to 0.36 dx within 7 ms; kinetic energy 200x the available potential energy | `tait_pressure` clamps at zero pressure |
| `dt = 2e-5` with `c0 = 20`, `h = 0.0325` | acoustic Courant 0.012, i.e. 24x smaller than necessary; the default 500-step run covered 0.01 s, during which the drop fell 0.5 mm and never touched the pool | `dt` defaults to 4e-4 (Courant 0.25); `validate_config` rejects steps above `max_courant` |
| Coordinate clamping as the boundary | 30 % low hydrostatic pressure, coincident particles, an artificial ceiling at `pool_depth + 2*drop_radius` that started above the drop and amputated it | static wall particles; `boundary_box` derives the ceiling from the drop geometry |
| No numerical acceptance criteria | a diverged run reported success | `evaluate_acceptance` records five criteria; `fail_fast` raises `SimulationUnstableError` after artifacts are written |
| Manifest missing the documented contract | ran without revision, versions, or statuses | manifest records solver revision, environment, three separate statuses, and per-criterion measured value/threshold/result |

See [`docs/review-2026-10.md`](review-2026-10.md) for the measurements behind these claims.

## What changed in 0.6.0

| Item | Before | After |
|---|---|---|
| Free-surface remedies | one, first-order: clamp the Tait law at zero | a second, optional, literature-backed option (`Config.density_diffusion`, the Antuono-Colagrossi-Marrone term) with a measured comparison, its own step limit, and seven invariant tests |
| B1 diagnostics | time-averaged pressure profile only | also the temporal pressure fluctuation per band: a time average hides exactly the noise a free-surface treatment targets |
| Default behaviour | — | **unchanged**. The comparison did not justify switching; the numbers behind that decision are in [`benchmarks/free-surface-treatment.md`](benchmarks/free-surface-treatment.md) |
| Benchmarks | three scripts | `--density-diffusion` on the hydrostatic and dam-break scripts, plus `benchmarks/free_surface/compare.py` |

## What changed in 0.5.0

No new physics. Three gate defects, three interfaces that disagreed with their
own documentation, and a measured speed-up. Details and numbers:
[`changes-0.5.0.md`](changes-0.5.0.md).

| Item | Before | After |
|---|---|---|
| `mass_conserved` gate | compared a literal `0.0` with itself; could not fail | measures \|m(t_end) − m(0)\| from two tensor sums |
| stability gates | judged from diagnostics sampled every `output_every` steps (default 50); the impact case's real peak speed is 2x the sampled one | extrema accumulated every step; every criterion records its `population` |
| time-step validation | acoustic Courant only; `viscosity = 1` passed validation and diverged to 53 % compression | acoustic + viscous + gravity limits, each exposed as a function |
| dam-break `boundary` | accepted and silently ignored | rejected by `validate_case` |
| `DamBreakConfig(device="auto")` | `RuntimeError` from `torch.device` | `resolve_device`, like the impact case |
| `rhs` dimension | accepted 3-D input with a 2-D kernel normalisation | rejects anything but 2-D |
| dead CLI flags | `--include-host` (always true), `--observed-order` (never read) | `--skip-host`; the order is always reported for three spacings |
| pair list | 23 % wall-wall pairs carried through every sum | pruned at build time, verified bit-exact |
| force evaluation | 702 ATen calls, 60 mask-indexing calls | 515 and 17; 2.62 → 1.85 ms per step at 608 particles |
| figure | rendered unconditionally, ~1.5 s of a 5 s run | `run(..., plot=False)` / `--no-plot`; snapshots streamed to disk |
| packaging | `matplotlib` a hard dependency; no lint gate | `[plot]` extra + `RuntimeWarning`; ruff `E,F,W` in CI; Python 3.11 added to the matrix |

**Consequence for the gates:** judged on every step, B2's peer-run peak density
compression is 10.0–10.3 %, not the 3.8 % that the sampled metric reported. The
`density_compression` gate now reports **fail** at all three resolutions while
`bulk_compression` (the same metric before the surge reaches the far wall) is
2.4 %. The metric was not redefined to make the run pass.

## What changed in 0.3.0

| Item | Before | After |
|---|---|---|
| Neighbour search | coordinates copied to the host, Python loops over particles; **slower than brute force** at 1000 particles (0.85x) | vectorized and device-resident; 3.0x faster than brute force at 1000 particles, 5.3x at 5000, and 5x faster than the previous host implementation |
| Pair rebuilding | every step, 83 % of the step cost | Verlet skin cache (`Config.pair_skin_factor`, default 0.25 h): 11 steps per rebuild, 1.72x faster step, bit-identical results |
| Whole impact run (incl. artifacts and plot) | 14.2 s | **5.2 s** (2.8x) |
| Free-surface case | droplet impact only, a smoke example | dam-break case with a three-level ladder: **observed order 0.96**, front insensitive to `c0` |
| Initial equilibrium | uniform density, the pool starts in free fall | `Config.hydrostatic_init` for a discrete-equilibrium start (t=0 profile error 100 % -> 5 %); measured not to change the time-averaged B1 result, so it stays opt-in |
| B1 boundary comparison | one configuration | wall particles vs coordinate clamp measured on the same case: 1.09 % vs 26.7 % interior error, 0 vs 1955 escapes |

The honest headline for 0.3.0 is **performance and a second verification case**,
not new physics. The solver still has no surface tension, no wall friction, and
no comparison against measured data.

## Gap register and priority

| Priority | Gap | Current risk | Required work | Exit evidence |
|---|---|---|---|---|
| P0 | No scientific acceptance gates | A successful process can be mistaken for a valid simulation | **Done for numerical criteria, and as of 0.5.0 the criteria are measured over every step rather than over the printed rows.** Physical acceptance still absent | `run_manifest.json` separates `software_execution`, `numerical_verification`, `physical_validation`, and records each criterion's population |
| P0 | Boundary was a coordinate clamp | **Fixed.** Static wall particles with extrapolated pressure | Validate against a dynamic case (dam break), add wall-force diagnostics | B1 passes at 1.09 % interior error; B2 has no experimental comparison yet |
| P0 | No convergence study | Results may depend strongly on particle spacing and smoothing length | **Two-level convergence exists for B2.** A three-level ladder and the observed order of accuracy are still missing | Resolution table and plots; no unsupported convergence claim |
| P1 | Free-surface tensile instability | **Clamping still the default, now with a measured alternative.** A diffusive density term (Antuono et al. 2012) is implemented and compared against the clamp on B1, B2 and clumping: interior error 1.086 % → 0.943 %, front moves < 0.3 %, worst pair separation 0.052 → 0.089 `dx`, +30 % cost. Kept off by default because the hydrostatic gain is inside the band-selection spread and the fluctuation improves by only 3 % | A quiet free-surface case (oscillating drop in zero g) to measure the noise the term actually targets, then decide | [`benchmarks/free-surface-treatment.md`](benchmarks/free-surface-treatment.md) records the comparison, the invariants and the decision |
| P1 | All-pairs interactions, O(N^2) | **Fixed.** One shared linked-cell pair list feeds density, pressure, and viscosity | — | Pair-set equivalence test (done); timing report (done) |
| P1 | Pair construction copied coordinates to CPU | **Fixed in 0.3.0.** Vectorized, device-resident binning plus a Verlet skin cache: 5x faster than the host search, 1.72x per solver step after skin reuse, 2.8x for the whole impact run | CUDA timing on real hardware | `device_equals_brute_force` and the host cross-check both pass at 400–5000 particles |
| P0 | No dam-break reference comparison | **Still open, but the solver side is ready.** Convergence (order 0.96), compressibility (0.019 % between `c0` 40 and 60) and no-slip walls (0.91x of the analytic Stokes rate) are all verified; only traceable reference data is missing | Bundle a digitized Martin & Moyce table with provenance and an uncertainty per point | B2 experimental gate currently `not_evaluated` |
| P1 | Viscosity was half its nominal value | **Fixed in 0.4.0.** The Morris Laplacian was missing its factor 2; measured 0.500 applied/nominal at four resolutions | — | `test_viscosity_is_calibrated` holds the ratio to 1e-3 |
| P2 | Wall friction absent | **Fixed in 0.4.0.** `wall_friction` gives a no-slip floor, verified against the analytic Stokes decay | Tune nothing yet: the default stays free slip | Ratio 0.84 → 0.91 under refinement |
| P1 | No surface-tension model | Droplet shape and splash morphology are not physically credible for capillary-dominated cases | Select a literature-backed formulation; isolate curvature/interface terms and parameters | Static droplet Laplace-pressure test and oscillating-drop frequency/damping comparison |
| P1 | Fixed time step without adaptation | **Partially fixed.** As of 0.5.0 the step is validated at start-up against the acoustic, viscous and gravity limits, but it is still not adapted to the evolving maximum speed | Adaptive step with all chosen limits recorded | Unit tests for the time-step policy; Courant stays bounded in long runs |
| P0 | Gate integrity | **Fixed in 0.5.0.** A constant comparison, sampled-only extrema and a missing viscous CFL all let a failing run report success | Keep the `population` field honest whenever a new criterion is added | CI asserts the stability criteria are measured on `all_steps` |
| P1 | Persistent residual motion | The pool never comes to rest (a few cm/s after 4 s) because walls are free slip and there is no relaxation phase | Add wall friction or a documented relaxation procedure | Decaying kinetic-energy trace in B1 |
| P2 | Diagnostics are monitoring-only | **Partially fixed.** Numerical criteria are now pass/fail; physical criteria are absent | Add per-case acceptance definitions with tolerances justified by reference and resolution | JSON validation report with provenance |
| P2 | No benchmark provenance registry | Comparisons can silently use mismatched parameters | Store source, reference, dimensionality, units, parameters, resolution, and extraction method | Each reference comparison is traceable |
| P2 | Lattice initialisation is not in discrete equilibrium | **Investigated, not solved.** `hydrostatic_init` takes the t=0 profile error from 100 % to 5 %, but does not change the time-averaged B1 result, because the residual is a slowly damped sloshing mode rather than a startup transient | A damped relaxation phase, or wall friction, or a longer averaging window | Single-snapshot B1 error comparable to the time-averaged one |

## Benchmark ladder

Run the following in order. Do not use the droplet-impact case as the first proof of physical correctness.

### B0 — Operator and invariance checks

- Kernel support and symmetry; gradient antisymmetry under particle exchange.
- Finite values at zero separation and near support boundaries.
- Pressure is non-negative everywhere, and monotonic in density.
- Neighbor-search result equals brute-force pair enumeration on deterministic small point sets.
- Wall particles outside the kernel support do not change the solution at all.
- Repeated CPU runs with the same configuration reproduce diagnostics exactly.

**Gate:** deterministic unit tests pass; no non-finite values. These are software/numerical checks, not physical validation. **Status: passing.**

### B1 — Static fluid / hydrostatic equilibrium

- Static configuration with the selected wall treatment.
- Report the depth-averaged pressure profile against `rho0 g (H - y)`, the free-surface drop, and the residual velocity.
- Time-average to remove the acoustic oscillation caused by the non-equilibrium initial lattice; report the free-surface band separately.

**Gate:** mean absolute relative error of the interior bands <= 5 %, free-surface drop <= 5 % of depth. **Status: passing at 1.09 % / 3.5 %.** See [`docs/benchmarks/hydrostatic.md`](benchmarks/hydrostatic.md).

### B2 — Dam-break free-surface flow

- Case with the classical 1:2 column aspect ratio, reported as `Z/a` against `tau = t sqrt(2 g / a)`.
- Front trajectory across resolutions; Ritter shallow-water front as an anchor, not a target.
- Mass conservation, boundedness, Mach number, density compression.

**Gate:** front trajectory converges under refinement, and every other measured quantity is inside a recorded limit. **Status: passing at observed order 0.96 over `dx` = 0.04 / 0.02 / 0.01, with the front converged to 0.019 % in `c0`.** The experimental comparison is implemented as a named gate and is `not_evaluated` because no digitized reference table is bundled. See [`docs/benchmarks/dam-break-2d.md`](benchmarks/dam-break-2d.md).

### B3 — Surface-tension verification

After a surface-tension formulation is implemented:

- Static droplet: pressure jump versus curvature and surface tension (Laplace-law check).
- Oscillating droplet: frequency and damping versus a cited reference.
- Report sensitivity to resolution and interface treatment.

**Gate:** no claim of capillary fidelity until the quantitative comparison is documented. **Status: not started, no model.**

### B4 — Droplet impact on a pool

Only after B1–B3 are understood:

- Report center-of-mass trajectory, maximum spread ratio, free-surface elevation, density extrema, kinetic energy, and momentum.
- Compare shape evolution and selected observables with a cited case using matching dimensionless parameters.
- Record limitations of 2-D cross-section modelling explicitly; do not present it as a 3-D drop.

**Gate:** the reference, extraction method, tolerance, and resolution sensitivity are all recorded. **Status: the case runs and is bounded, but nothing has been compared against a reference.**

## Machine-readable run contract

Every run records:

- source revision, solver version, case identifier, config, device, Python/PyTorch versions;
- time step and the criterion that limits it, smoothing-length ratio, particle spacing, boundary model;
- diagnostics CSV, snapshots/plots;
- separate statuses: `software_execution`, `numerical_verification`, `physical_validation`;
- for every acceptance criterion: metric, measured value, threshold, units, and result.

Missing thresholds must be reported as `not_evaluated`, never silently treated as pass. The
`boundary_escape` criterion is `not_evaluated` when the case uses `boundary="clamp"`, because
in that mode exceeding the box *is* the boundary condition.

## Agent-facing workflow

The TensorPlatform agent should be able to:

1. select a benchmark case from a versioned manifest;
2. run a deterministic smoke test, then the requested scientific case;
3. check process exit and artifact completeness;
4. evaluate only criteria with explicit, versioned thresholds;
5. produce a report distinguishing failed execution, failed numerical checks, and unvalidated physics;
6. attach source revision, config, plots, CSV, and report to the run record.

The solver should remain usable from the command line and tests without TensorPlatform. Platform integration should orchestrate and record runs rather than becoming a hidden dependency of the numerical kernel.

## Suggested implementation sequence

1. ~~Non-negative pressure, time-step validation, and acceptance criteria.~~ Done in 0.2.0.
2. ~~Wall-boundary implementation and a hydrostatic verification case.~~ Done in 0.2.0.
3. ~~Device-resident neighbour search and pair-list reuse.~~ Done in 0.3.0.
4. ~~Dam-break case with resolution convergence.~~ Done in 0.3.0.
5. ~~Gate integrity, interface honesty, and the measured performance pass.~~ Done in 0.5.0.
6. Bundle a digitized experimental dam-break reference with provenance; decide the default `wall_friction` from that comparison.
7. Surface tension only with dedicated static and oscillating-drop tests.
8. Adaptive time step driven by the evolving maximum speed (all three limits already exist as functions).
9. Revisit droplet impact as a physical benchmark, not merely a smoke example.

Each step should be delivered as a reviewable PR with reproducible tests and artifacts. No benchmark should be marked validated without traceable reference data and documented acceptance criteria.
