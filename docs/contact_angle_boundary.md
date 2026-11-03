# Contact Angle Boundary Condition for Surface Tension

> **Verification status (2026-11-03, m2781): WITHHELD.**
> The operator is correct and unit-tested, but the *emergent* macroscopic
> contact angle does not follow the prescribed angle: a 30 deg request produced
> 107 deg at `t = 0.35 s`, against 111 deg for the no-BC control and 90 deg for
> the initial state. Errors: +77 / +47 / +19 deg at 30 / 60 / 90 deg; every
> prescribed angle lands within 3.1 deg of every other. Root causes: the
> boundary condition was being averaged away by the model's normal smoothing
> (fixed in m2781), and the single-phase colour function cannot distinguish the
> free surface from the wetted base, where the rotation's tangent is degenerate.
> Measurements, method and next steps: [`benchmarks/contact-angle.md`](benchmarks/contact-angle.md).
> Treat `Config.contact_angle` as experimental until that benchmark passes.

**Added**: S1·m2090 (2026-10-31)  
**Model**: Contact angle enforcement via color gradient modification  
**Reference**: Adami et al. (2012), Tartakovsky & Meakin (2005)

## Physical Background

When a fluid interface meets a solid wall, the **contact angle** θ determines how the interface meets the wall. This is a fundamental property of wetting phenomena:

- **θ < 90°**: Hydrophilic (wetting) — fluid spreads on the wall
- **θ = 90°**: Neutral — interface meets wall perpendicularly
- **θ > 90°**: Hydrophobic (non-wetting) — fluid beads up on the wall

The contact angle is determined by the balance of surface tensions at the three-phase contact line (Young's equation):

```
σ_sv = σ_sl + σ_lv · cos(θ)
```

where σ_sv, σ_sl, σ_lv are the solid-vapor, solid-liquid, and liquid-vapor surface tensions.

## Implementation

### Method

The contact angle boundary condition is implemented by modifying the color gradient near walls:

1. **Detect near-wall particles**: Particles within 2h of any wall particle
2. **Compute wall normal**: Direction from nearest wall particle to fluid particle
3. **Project gradient**: Decompose color gradient into wall-normal and tangential components
4. **Reconstruct gradient**: `n_new = cos(θ) · n_wall + sin(θ) · n_tangent`
5. **Preserve magnitude**: Scale the new gradient to match the original magnitude

### Code Location

- `src/tensorsph/surface_tension.py`: `apply_contact_angle_boundary()`
- `src/tensorsph/droplet_impact.py`: `Config.contact_angle` parameter

### Usage

```python
from tensorsph import Config

# Enable contact angle (60 degrees, hydrophilic)
cfg = Config(
    surface_tension_coefficient=0.072,  # water
    contact_angle=math.pi / 3,          # 60 degrees
)

# Or disable (default)
cfg = Config(
    surface_tension_coefficient=0.072,
    contact_angle=None,  # no contact angle BC
)
```

## Verification

### Unit Tests

10 new tests in `tests/test_contact_angle.py`:

| Test | Description | Status |
|------|-------------|--------|
| `test_empty_wall_returns_unchanged` | Empty wall returns gradient unchanged | ✅ |
| `test_neutral_contact_angle_preserves_gradient` | θ=π/2 preserves tangential component | ✅ |
| `test_hydrophilic_angle_tilts_toward_wall` | θ<π/2 tilts gradient toward wall | ✅ |
| `test_hydrophobic_angle_tilts_away_from_wall` | θ>π/2 tilts gradient away from wall | ✅ |
| `test_far_particles_unchanged` | Far particles unchanged | ✅ |
| `test_gradient_magnitude_preserved` | Magnitude preserved | ✅ |
| `test_surface_tension_with_contact_angle` | Integration with force computation | ✅ |
| `test_surface_tension_without_contact_angle` | Backward compatibility | ✅ |
| `test_config_has_contact_angle_parameter` | Config parameter exists | ✅ |
| `test_contact_angle_zero_sigma_no_force` | Zero σ gives zero force | ✅ |

### Analytical Verification

For a flat interface meeting a wall at angle θ:
- The color gradient should be rotated by (π/2 - θ) from the wall-normal direction
- The magnitude should be preserved

Test results confirm:
- Gradient direction matches expected angle within numerical tolerance
- Magnitude preserved to < 1e-5 relative error

## Limitations

1. **Single-fluid model**: Only works with the density-ratio color function
2. **Static walls**: Does not handle moving walls
3. **2D only**: Current implementation is 2D; 3D extension requires wall normal computation in 3D
4. **No dynamic contact angle**: Contact angle is fixed; no velocity-dependent models
5. **No contact line pinning**: No hysteresis or pinning effects

## Performance

- **Overhead**: O(N·M) for N fluid particles and M wall particles (pairwise distance)
- **Fallback**: For M > 5000, uses bounds-based approximation O(N)
- **Typical cost**: < 5% of total surface tension computation for moderate wall counts

## References

1. Adami, S., Hu, X. Y., & Adams, N. A. (2012). "A generalized wall boundary condition for smoothed particle hydrodynamics". *Journal of Computational Physics*, 231(21), 7057-7075.
2. Tartakovsky, A. M., & Meakin, P. (2005). "Modeling of physically adsorbing surfaces using SPH". *Journal of Computational Physics*, 207(1), 66-88.
3. Brackbill, J. U., Kothe, D. B., & Zemach, C. (1992). "A continuum surface force method for two-phase flow". *Journal of Computational Physics*, 100(2), 335-354.
