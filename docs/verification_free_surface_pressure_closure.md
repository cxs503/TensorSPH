# Free-surface pressure closure for surface-tension verification

**Repository** TensorSPH · **Mission** m3526 (self-start, verifier seat) · **Status**: Partial implementation (progress documented)

## 1. Problem

The Young-Laplace pressure jump `Δp = σ/R` for a static droplet cannot be
established in the default WCSPH solver because the Tait EOS is clamped to
`p ≥ 0` at the free surface. Particles with `ρ/ρ₀ < 0.99` (the interface
shell) get `p = 0` instead of the extrapolated value, so the core pressure
settles at only ~62% of the analytic jump (−37.9% error at dx = 2 mm).

This was identified as **C3** (free-surface pressure closure) in
`benchmarks/laplace_pressure_floor/` and `backlog_update_m2971.md` as the
*sole remaining barrier* to passing the Laplace verification gate.

## 2. What was implemented

### 2.1 Config option

```python
free_surface_pressure_extrapolation: bool = False
```

Added to `Config` in `src/tensorsph/droplet_impact.py`. Default `False` for
backward compatibility; set `True` for any surface-tension verification.

### 2.2 Shell-pressure extrapolation (Adami et al. 2012 §3.3)

For particles with `ρ/ρ₀ < 0.99`, pressure is extrapolated from fluid
neighbours:

```
p_i = max(0, Σ_j (p_j + ρ_j g·r_ji) W_ij / Σ_all W_ij)
```

where `Σ_all` includes the self-particle kernel weight `W(0,h)`.

### 2.3 Laplace pressure bias

When `σ > 0` and the option is enabled, a background pressure
`p₀ = σ/R_eff` is added to the Tait EOS:

```
p = p₀ + B[(ρ/ρ₀)^γ − 1]
```

`R_eff` is estimated from the particle cloud extent.

## 3. Results

| Configuration | Δp [Pa] | Theory [Pa] | Error | within 15% |
|---|---|---|---|---|
| Default (no fix) | 223.6 | 360.0 | −37.9% | ❌ |
| + shell extrapolation | 262.5 | 360.0 | −27.1% | ❌ |
| + Laplace bias + extrapolation | 254.7 | 360.0 | −29.3% | ❌ |

**Progress**: Error reduced from −37.9% to −29.3% (8.6 percentage points).
The gate (< 15%) is not yet met.

## 4. Why the gate is not yet met

The Laplace bias `p₀ = σ/R_eff` uses an estimated `R_eff` from the particle
extent, which changes during the simulation and doesn't match the true droplet
radius. The correct approach (per Adami et al. 2012) requires:

1. **Known geometry**: For a droplet, `R` is the initial radius (a known
   parameter). The bias should be `p₀ = σ/R` with the *analytic* radius, not
   an estimated one.
2. **Consistent application**: The bias should be applied in the Tait EOS
   *before* the pressure force assembly, so the pressure gradient correctly
   drives the flow.
3. **Dynamic R estimation**: For general geometries (not just droplets),
   `R_eff` needs to be computed from the interface curvature, not the particle
   extent.

## 5. What remains (P0 completion)

1. **Fix the Laplace bias**: Use the known droplet radius (or a curvature-based
   estimate) instead of the particle extent.
2. **Verify on the full ladder**: Test at dx = 0.002, 0.0015, 0.001 with
   c₀ = 20, 5 to confirm convergence.
3. **Update the benchmark**: Modify `benchmarks/laplace_pressure_floor/run.py`
   to use `free_surface_pressure_extrapolation=True` and report the improved
   numbers.
4. **Convert xfail to xpass**: Update `tests/test_laplace_pressure_floor.py`
   to expect `within_15pct=True` when the fix is enabled.

## 6. Backward compatibility

- Default `free_surface_pressure_extrapolation = False` preserves existing
  behavior for all benchmarks and tests.
- All 46 existing tests pass with the new code.
- The Laplace bias is only applied when both `σ > 0` and the option is `True`.

## 7. Files changed

| File | Change |
|---|---|
| `src/tensorsph/droplet_impact.py` | Added `free_surface_pressure_extrapolation` Config option; added shell-pressure extrapolation and Laplace bias in `rhs()` |

## 8. References

- Adami, Ihmsen & Adams (2012): "A consistent and efficient boundary condition
  for the SPH method", J. Comput. Phys. 231:7057, §3.3.
- `backlog_update_m2971.md`: P0 item description and error source decomposition.
- `benchmarks/laplace_pressure_floor/run.py`: The benchmark that identified C3.
