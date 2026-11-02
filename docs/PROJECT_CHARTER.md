# TensorSPH — smoothed particle hydrodynamics charter

**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission
Provide independently runnable, PyTorch-native SPH methods for particle-based fluid simulation and explicitly supported physics.

## Owns
- SPH kernels, particle neighborhoods, density/pressure, viscosity, boundaries, time integration and solver-local models.
- Particle diagnostics, stability limits, reproducible examples and SPH-specific verification.
- Method-specific comparisons and supported-use documentation.

## Does not own
- Global agent coordination, shared platform infrastructure, or all fluid physics across solvers.
- Assuming SPH and LBM/DG results are interchangeable without method-specific evidence.
- Using animation appearance alone as a correctness criterion.

## Integration
Offer parameterized standalone execution and structured particle/field outputs for TensorPhysics. Cross-solver comparison or coupling must specify mapping, units, time synchronization and conservation tests. Use TensorPlatform contracts for shared records, not private platform imports.

## Quality gates
Check conservation, hydrostatic equilibrium, canonical dam-break/droplet/impact cases as applicable, particle-resolution and time-step sensitivity, boundary artifacts, and reference agreement. Separate benchmark evidence from visual demos.

## Agent rules
Read this charter and README; identify ownership and affected contracts; prefer stable interfaces; add reproducible tests and benchmarks; report commands, failures, assumptions and limitations; never infer correctness from successful execution or attractive visuals.

## Maturity
Only implemented and measured capabilities should be advertised as available.
