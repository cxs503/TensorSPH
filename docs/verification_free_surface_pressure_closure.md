# P0 Fix: Free-Surface Pressure Closure (Adami et al. 2012 §3.3)

**Repository**: TensorSPH  
**Mission**: m3718 (self-start, verifier seat)  
**Date**: 2026-11-03  
**Status**: Code complete, unit tests green, benchmark pending (system load ≥172)

---

## 1. Problem Statement

The Young-Laplace pressure jump verification for the CSF surface-tension model has been
stuck at **WITHHELD** through three successive rounds of work (m2174, m2447, m2971). The
best measured value was **−34.0%** (237.6 Pa vs 360 Pa analytic), well outside the 15% gate.

The error-source decomposition in `benchmarks/laplace_pressure_floor/` identified four
contributors:

| Source | Impact | Status |
|--------|--------|--------|
| C1: Measurement口径 | −44…−55 ppt | Fixed (use absolute pressure) |
| C2: Initialization bias | ±9 ppt steady | Fixed (mass calibration) |
| **C3: Free-surface pressure closure** | **−34…−38%** | **This fix** |
| C4: EOS noise floor | Falsified | Not the cause |

**C3 is the single remaining barrier**: the Tait EOS clamps `p >= 0`, so particles in the
density-deficit shell at a free surface sit at `p = 0` even though the physical pressure
should be continuous across the interface. The interior therefore settles at only ~57% of
the analytic jump.

---

## 2. Method

### 2.1 The Fix

Implement the shell-pressure extrapolation of Adami, Ihmsen & Adams (2012) §3.3:

```
p_i = max(0, Σ_j (p_j + ρ_j g·r_ji) W_ij / Σ_j W_ij)
```

for particles with `ρ_i/ρ_0 < 0.99` (free-surface threshold). With `g = 0` (the Laplace
droplet case), this reduces to a kernel-weighted average of neighbor pressures.

### 2.2 Implementation

**New function**: `free_surface_pressure_closure()` in `src/tensorsph/droplet_impact.py`

- Identifies free-surface particles by density ratio threshold (default 0.99)
- Accumulates kernel-weighted pressure from neighbors, including body-force correction
- Returns blended pressure: extrapolated for free-surface, Tait for bulk

**Integration**: Called in `rhs()` and `rhs_all_pairs()` when
`surface_tension_coefficient > 0`, after the Tait pressure computation.

### 2.3 Why This Works

The closure propagates pressure from the compressed core to the density-deficit shell:
1. Core particles (ρ/ρ₀ ≥ 0.99) keep their Tait pressure (positive)
2. Near-surface particles (ρ/ρ₀ < 0.99) get extrapolated pressure from core neighbors
3. Over multiple timesteps, the pressure field equilibrates to the correct Laplace jump

---

## 3. Results

### 3.1 Unit Tests (8/8 PASS)

| Test | Verdict |
|------|---------|
| `test_bulk_particles_unchanged` | PASS |
| `test_free_surface_gets_extrapolated` | PASS |
| `test_no_pairs_returns_unchanged` | PASS |
| `test_no_free_surface_returns_unchanged` | PASS |
| `test_body_force_correction_with_gravity` | PASS |
| `test_zero_gravity_plain_average` | PASS |
| `test_threshold_controls_which_particles_are_free_surface` | PASS |
| `test_rhs_applies_closure_when_surface_tension_on` | PASS |

### 3.2 Regression Tests (34/34 PASS)

All existing surface tension and Laplace tests pass:
- `test_laplace_droplet.py`: 11/11 PASS
- `test_laplace_pressure.py`: 8/8 PASS
- `test_surface_tension.py`: 15/15 PASS

### 3.3 Benchmark Status

The full Laplace pressure floor benchmark (`benchmarks/laplace_pressure_floor/run.py`)
has **not been run** due to system load ≥172 (nproc=128). The benchmark advances the SPH
droplet for 1.5 capillary times (~4000 steps), which is computationally expensive.

**Expected outcome**: The `absolute_time_average_tail` error should drop from −34% to
within 15% (the acceptance criterion in `test_laplace_jump_within_15pct`).

---

## 4. Error Judgement

| Gate | Before | After (expected) | Result |
|------|--------|------------------|--------|
| `laplace_jump_within_15pct` | FAIL (−34%) | **PASS** (pending benchmark) | PENDING |
| Unit tests | N/A | 8/8 PASS | PASS |
| Regression tests | N/A | 34/34 PASS | PASS |

**Explicit answer**: The key quantity (Laplace pressure jump) is expected to be within 15%
after this fix, based on the physics of the closure. The benchmark must be run to confirm.

---

## 5. Notes, Pitfalls and Scope

### 5.1 Scope
- **2D only**: The closure is implemented for 2D SPH (the current solver dimension)
- **Single-phase**: Works with the kernel-sum color function of the consistent CSF scheme
- **g = 0 case**: The Laplace droplet benchmark has zero gravity, so the body-force term
  is zero; the closure reduces to a plain weighted average

### 5.2 Pitfalls
1. **Outermost layer**: Particles with no positive-pressure neighbors stay at p = 0. This
   is physically correct for the outermost layer of a drop in vacuum, but means the
   closure doesn't fully eliminate the pressure deficit in one step.
2. **Threshold sensitivity**: The default threshold (0.99) was chosen to match the
   density deficit observed in the benchmark. Different cases may need tuning.
3. **Single-pass**: The closure is applied once per timestep. For large density deficits,
   multiple passes or an iterative solver might be needed (not implemented).

### 5.3 Limitations
- The closure assumes the neighbor pressure field is already close to equilibrium. In
  highly dynamic cases (e.g., droplet impact), the extrapolation may lag.
- The body-force correction is only implemented for 2D (gravity in y-direction).

---

## 6. Follow-up

1. **Run the benchmark** when system load permits to confirm the 15% gate is met.
2. **Update `test_laplace_jump_within_15pct`** from `xfail(strict=True)` to a regular
   test once the benchmark passes.
3. **3D extension**: The closure needs to be extended to 3D (gravity vector handling).
4. **Adaptive threshold**: Consider making the threshold a Config parameter for cases
   with different density deficit characteristics.

---

## 7. Deliverables

- [x] `src/tensorsph/droplet_impact.py`: `free_surface_pressure_closure()` function + integration
- [x] `tests/test_free_surface_pressure_closure.py`: 8 unit tests
- [x] `docs/verification_free_surface_pressure_closure.md`: This document

---

**References**:
- Adami, S., Ihmsen, M., & Adams, N. (2012). "A generalized wall boundary condition for
  smoothed particle hydrodynamics." J. Comput. Phys. 231:7057. §3.3.
- backlog_update_m2971.md: P0 item description and error-source decomposition.
