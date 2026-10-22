# TensorSPH algorithm gap analysis and validation roadmap

**Status:** engineering assessment of the current prototype, not a claim of validated SPH accuracy.  
**Scope:** 2-D PyTorch WCSPH droplet-impact prototype on `main` after PR #2.

## Executive decision

Keep TensorSPH as a PyTorch-native solver whose numerical operators, case definitions, diagnostics, and run artifacts are explicit and machine-readable. Study mature projects for algorithm choices and validation methodology; do not copy their implementation wholesale.

Recommended references:

- [PySPH](https://github.com/pypr/pysph): equation-oriented architecture, neighbor search, integrators, wall conditions, and a catalog of reproducible examples. Its official project description notes Python-defined equations with performance-critical execution through Cython/OpenCL. PyPI currently lists BSD licensing; verify the exact source-file headers and bundled dependencies before reusing any code.
- [DualSPHysics](https://github.com/DualSPHysics/DualSPHysics): engineering-scale free-surface cases, GPU execution, and practical case setup. The repository identifies LGPL-2.1; review the applicable license obligations before any code reuse.
- [SPHinXsys](https://github.com/xiangyu-hu/sphinxsys): architecture for coupled multi-physics and fluid-structure interaction. The repository identifies Apache-2.0; verify dependency and file-level terms before reuse.

The first milestone should be *verification and diagnostics*, not adding more visually impressive splash effects.

## Current implementation baseline

The current solver has a 2-D cubic-spline kernel, density summation, Tait equation of state, symmetric pressure-force form, a regularized viscosity approximation, gravity, fixed-step semi-implicit Euler integration, and a simple coordinate clamp. It writes particle snapshots, a diagnostics CSV, a run manifest, and density plots.

CI passing on Python 3.10 and 3.12 confirms the covered software paths execute. It does **not** establish conservation, convergence, correct wall physics, surface-tension accuracy, or agreement with experiment.

## Gap register and priority

| Priority | Gap | Current risk | Required work | Exit evidence |
|---|---|---|---|---|
| P0 | No scientific acceptance gates | A successful process can be mistaken for a valid simulation | Separate software status, numerical verification, and physical validation in run metadata | Automated report labels each gate separately |
| P0 | Boundary is a coordinate clamp | Artificial momentum changes and unreliable near-wall density/pressure | Implement a documented wall method (e.g. a published dynamic/ghost boundary formulation) behind a boundary interface | Static hydrostatic and dam-break checks, wall penetration and force diagnostics |
| P0 | No convergence study | Results may depend strongly on particle spacing and smoothing length | Add a three-resolution protocol with fixed dimensionless inputs and comparison metrics | Resolution table and plots; no unsupported convergence claim |
| P1 | All-pairs interactions, O(N²) | Particle count becomes the immediate scaling limit | Introduce a neighbor-search interface; first implement/test a cell-linked list or spatial hash for compact support | Pair-set equivalence test against brute force; timing and memory scaling report |
| P1 | No surface-tension model | Droplet shape and splash morphology are not physically credible for capillary-dominated cases | Select a literature-backed formulation; isolate curvature/interface terms and parameters | Static droplet Laplace-pressure test and oscillating-drop frequency/damping comparison |
| P1 | Fixed time step without enforcement | Monitored Courant number may exceed a safe case-specific bound | Add explicit stability limits and fail-fast or adaptive-step policy, with all chosen limits recorded | Unit tests for time-step policy and stable reference runs |
| P1 | Integration and force models not independently benchmarked | Errors can be difficult to localize | Separate operators, boundary treatment, integrator, and case orchestration into testable interfaces | Operator-level tests plus a minimal integrator convergence test |
| P2 | Diagnostics are monitoring-only | No reliable machine-readable pass/fail result | Add per-case acceptance definitions with tolerances justified by reference and resolution | JSON validation report with measured value, limit, pass/fail, and provenance |
| P2 | No benchmark provenance registry | Comparisons can silently use mismatched parameters | Store source, reference, dimensionality, units, parameters, resolution, and extraction method | Each reference comparison is traceable and reproducible |

## Benchmark ladder

Run the following in order. Do not use the droplet-impact case as the first proof of physical correctness.

### B0 — Operator and invariance checks

- Kernel support and symmetry; gradient antisymmetry under particle exchange.
- Finite values at zero separation and near support boundaries.
- Neighbor-search result equals brute-force pair enumeration on deterministic small point sets.
- Force-pair symmetry and net internal force are checked where the selected discretization mathematically guarantees them.
- Repeated CPU runs with the same seed reproduce initial conditions and diagnostics within documented floating-point tolerances.

**Gate:** deterministic unit tests pass; no non-finite values. These are software/numerical checks, not physical validation.

### B1 — Static fluid / hydrostatic equilibrium

- Use a well-defined static configuration with the selected wall treatment.
- Report density error, pressure profile, wall penetration, and residual velocity over time.
- Sweep particle spacing and smoothing-length ratio independently.
- Avoid applying surface-tension acceptance criteria until a surface-tension model exists.

**Gate:** publish the measured error curves first; set thresholds only after the reference solution and expected discretization error are specified.

### B2 — Dam-break free-surface flow

- Select a published experiment/reference case and record geometry, fluid properties, gravity, and nondimensional parameters.
- Compare the leading-front position against time and compare selected wall-pressure histories where the reference provides them.
- Plot reference and TensorSPH curves together; include resolution and timestep sensitivity.
- Use physically appropriate walls, not coordinate clamping.

**Gate:** report errors and sensitivity. A visual resemblance alone is not acceptance.

### B3 — Surface-tension verification

After a surface-tension formulation is implemented:

- Static droplet: pressure jump versus curvature and surface tension (Laplace-law check).
- Oscillating droplet: frequency and damping versus a cited reference.
- Report sensitivity to resolution and interface treatment.

**Gate:** no claim of capillary fidelity until the quantitative comparison is documented.

### B4 — Droplet impact on a pool

Only after B1–B3 are understood:

- Report center-of-mass trajectory, maximum spread ratio, free-surface elevation, density extrema, kinetic energy, and momentum.
- Compare shape evolution and selected observables with a cited case using matching dimensionless parameters.
- Record limitations of 2-D cross-section modelling explicitly; do not present it as a 3-D drop.

**Gate:** the reference, extraction method, tolerance, and resolution sensitivity are all recorded.

## Machine-readable run contract

Every run should record:

- source revision, solver version, case identifier, config, random seed, device, Python/PyTorch versions;
- physical units, dimensionality, reference source, and parameter provenance;
- time step and its selection rule, smoothing-length ratio, particle spacing, and boundary model;
- diagnostics CSV, snapshots/plots, and execution logs;
- separate statuses: `software_execution`, `numerical_verification`, and `physical_validation`;
- for every acceptance criterion: metric, measured value, threshold, units, reference, and result.

Missing thresholds must be reported as `not_evaluated`, never silently treated as pass.

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

1. Keep PR #2's input validation and diagnostic probes as the baseline.
2. Add a neighbor-search abstraction and brute-force equivalence tests.
3. Add the first wall-boundary implementation and a hydrostatic/static verification case.
4. Add a dam-break case with reference data and comparison plots.
5. Add surface tension only with dedicated static and oscillating-drop tests.
6. Enforce time-step policy, then run resolution studies.
7. Revisit droplet impact as a physical benchmark, not merely a smoke example.

Each step should be delivered as a reviewable PR with reproducible tests and artifacts. No benchmark should be marked validated without traceable reference data and documented acceptance criteria.
