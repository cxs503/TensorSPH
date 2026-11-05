# CSF Surface Tension: curvature_color_smoothing Parameter Exposure

**Mission**: m3665 (self-start, verifier seat)  
**Date**: 2026-11-04  
**Repository**: TensorSPH

## Summary

Added the `curvature_color_smoothing` parameter to `surface_tension_acceleration()`,
exposing a capability that already existed in `csf_fields()` but was not accessible
through the public API.

## Change

**File**: `src/tensorsph/csf_consistent.py`

Added `curvature_color_smoothing: int = 0` parameter to `surface_tension_acceleration()`
and passed it through to `csf_fields()`.

### Before
```python
def surface_tension_acceleration(
    pos, density, mass, pairs, h, sigma,
    rho0=None, wall=None, contact_angle=None,
    curvature_scheme="mls", curvature_stencil=4.0,
):
    ...
    state = csf_fields(pos, mass, pairs, h, rho0, wall=wall,
                       contact_angle=contact_angle,
                       curvature_scheme=curvature_scheme,
                       curvature_stencil=curvature_stencil)
```

### After
```python
def surface_tension_acceleration(
    pos, density, mass, pairs, h, sigma,
    rho0=None, wall=None, contact_angle=None,
    curvature_scheme="mls", curvature_stencil=4.0,
    curvature_color_smoothing=0,
):
    ...
    state = csf_fields(pos, mass, pairs, h, rho0, wall=wall,
                       contact_angle=contact_angle,
                       curvature_scheme=curvature_scheme,
                       curvature_stencil=curvature_stencil,
                       curvature_color_smoothing=curvature_color_smoothing)
```

## Why This Matters

The `curvature_color_smoothing` parameter (documented in `csf_fields`) computes
curvature from a Shepard-smoothed copy of the colour field while the force keeps
using the sharp field. As documented in `docs/verification_csf_curvature_bias.md`:

> `kappa = div n` amplifies the angular discretisation error of the normal by
> `~1/dx`, and on a one-sided free-surface support that error is concentrated
> in the inner shell, where `c` approaches 1 and the normal is barely defined:
> those particles carried 12-24 % of the force but a kappa of +95..+183 %.
> One smoothing pass on `c` removes that noise at the source and brings the
> delta_s-weighted curvature error from +22..+25 % to +7..+14 % for dx <= 2 mm.

Without this parameter exposed, users of `surface_tension_acceleration()` could
not benefit from this improvement.

## Verification

- All existing tests pass (23/23 in `test_surface_tension.py` + `test_laplace_pressure.py`)
- The parameter signature is correctly exposed:
  ```
  Parameters: ['pos', 'density', 'mass', 'pairs', 'h', 'sigma', 'rho0',
               'wall', 'contact_angle', 'curvature_scheme', 'curvature_stencil',
               'curvature_color_smoothing']
  curvature_color_smoothing default: 0
  ```
- Backward compatible: default value is 0, preserving existing behavior.

## Context: Laplace Law Verification Status

The CSF surface tension model still has a known Laplace law verification deficit
(see `docs/benchmarks/surface-tension-laplace.md`):

| Resolution | jump_ratio (mls, smooth=0) | delta_integral |
|------------|---------------------------|----------------|
| R/dx=16    | 0.820                     | 0.688          |

The delta function integral is 0.688 instead of 1.0, meaning the CSF force is
about 31% weaker than it should be. This is a fundamental issue with the
kernel-sum colour function in single-phase SPH (the colour doesn't reach 0
outside the fluid, so the total jump is ~0.3 instead of 1.0).

The `curvature_color_smoothing` parameter helps with curvature noise but does
not address the delta function normalization issue. A separate fix is needed
for the Laplace law to pass the 15% gate.

## Deliverables

- `src/tensorsph/csf_consistent.py` — added parameter
- `docs/verification_csf_color_smoothing.md` — this document
