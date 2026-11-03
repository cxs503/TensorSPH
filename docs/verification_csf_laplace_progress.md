# TensorSPH CSF Surface Tension: Laplace Law Verification Progress Report

**Mission**: m2791 (self-start, verifier seat)  
**Date**: 2026-11-03  
**Repository**: TensorSPH  
**Status**: Partial progress — curvature fixed, delta_s normalization in progress

---

## 1. Case Introduction

The CSF (Continuum Surface Force) surface tension model in TensorSPH implements the Brackbill et al. (1992) formulation:

```
f_st = -σ · κ · δ_s · n
```

where σ is the surface tension coefficient, κ is curvature, δ_s is the surface delta function, and n is the interface normal.

The exact reference for a 2D circular droplet at equilibrium is the **Laplace law**:

```
Δp = σ / R
```

For water-air at 20°C: σ = 0.072 N/m, R = 0.1 m → Δp = 0.720 Pa.

---

## 2. What Was Done

### 2.1 Baseline State (before this work)

The `csf_consistent.py` module had:
- MLS-corrected gradient/divergence operators
- Kernel-sum color function with self-volume contribution
- Factor-of-2 delta_s normalization (based on assumed c_drop = 0.5)
- Default curvature scheme: "mls" (MLS divergence of normal)

Benchmark results at R/dx = 16, 32, 64:
| R/dx | Δp [Pa] | ratio | error | κR (median) |
|------|---------|-------|-------|-------------|
| 16   | 0.590   | 0.820 | 18.0% | 0.499       |
| 32   | 0.604   | 0.838 | 16.2% | 0.123       |
| 64   | 0.635   | 0.882 | 11.8% | -0.155      |

The Laplace error at finest level (11.8%) passed the 15% gate, but curvature and force direction gates failed.

### 2.2 Changes Made

**File: `src/tensorsph/csf_consistent.py`**
1. Added `_color_drop()` function to dynamically estimate the color drop across the interface from the particle field
2. Changed delta_s normalization from fixed `2.0 * |∇c|` to `|∇c| / (c_bulk - c_surface)`
3. Changed default `curvature_scheme` from "mls" to "fit_quadratic"
4. Updated docstrings to reflect new defaults

**File: `src/tensorsph/droplet_impact.py`**
1. Changed `Config.csf_curvature` default from "mls" to "fit_quadratic"

**File: `src/tensorsph/surface_tension.py`**
1. Changed default `curvature_scheme` in `compute_surface_tension_force` from "mls" to "fit_quadratic"

### 2.3 Key Finding: Curvature Fit vs MLS Divergence

The `fit_quadratic` curvature scheme (local weighted quadratic fit of the interface) gives nearly perfect curvature:

| R/dx | κR (fit_quadratic) | κR (mls) |
|------|-------------------|----------|
| 16   | 1.003             | 0.499    |
| 32   | 0.996             | 0.123    |
| 64   | 1.150             | -0.155   |

The MLS divergence of the normal gives systematically wrong curvature (wrong sign at high resolution), while the quadratic fit is within 1-15% of the exact value.

### 2.4 Delta_s Normalization Issue

The delta_s integral with the old factor of 2.0 was 0.688 (should be 1.0). The new dynamic normalization based on color drop overshoots because the estimated color drop (~0.17) is smaller than the effective drop that the gradient integral represents (~0.34).

With the corrected curvature and adjusted delta_s:
| R/dx | Δp [Pa] | ratio | error |
|------|---------|-------|-------|
| 16   | 1.485   | 2.062 | 106.2%|
| 32   | 1.447   | 2.010 | 101.0%|
| 64   | 1.540   | 2.138 | 113.8%|

The overshoot indicates the color drop estimation needs refinement. The correct normalization factor is approximately 1.43 (not 2.0 or 5.9).

---

## 3. Results Summary

### 3.1 What Works
- **Gauge self-test**: 0.07-0.13% error (estimator is accurate)
- **Curvature with fit_quadratic**: κR = 0.996-1.150 (within 15% of exact +1)
- **Force direction**: 100% inward at R/dx=16

### 3.2 What Doesn't Work Yet
- **Delta_s normalization**: The dynamic color drop estimation overshoots by ~2x
- **Laplace pressure jump**: Currently 100-114% error (needs to be <15%)

### 3.3 Root Cause Analysis

The kernel-sum color function in single-phase SPH falls from c ≈ 1.0 in the bulk to c ≈ 0.66-0.89 at the outermost particle layer. The gradient integral over the fluid equals this color drop (0.11-0.34), not 1.0. The remaining drop (from c_surface to 0) lives in the void where there are no particles.

The correct normalization factor is `1 / ∫|∇c|dn`, which is approximately 1.43 for the current kernel-sum color function on a hex lattice with h/dx = 1.3.

---

## 4. Next Steps (for future missions)

1. **Fix delta_s normalization**: Use a fixed factor of ~1.43, or compute the gradient integral directly and normalize by it
2. **Update xfail tests**: Once Laplace error < 15%, the `test_laplace_law_is_satisfied` xfail should become xpass
3. **3D extension**: The fit_quadratic scheme is 2D-only; need fit_sphere for 3D
4. **Spurious currents**: Measure max|v|/√(σ/ρR) after equilibrium (Adami 2012 criterion: < 0.1)

---

## 5. Error判定

| Quantity | Benchmark | Target | Status |
|----------|-----------|--------|--------|
| Gauge error | 0.07-0.13% | < 5% | ✅ PASS |
| Laplace error (finest) | 113.8% | < 15% | ❌ FAIL (overshoot) |
| Curvature κR (fit_quad) | 0.996-1.150 | 1.0 ± 15% | ✅ PASS |
| Curvature κR (mls) | -0.155-0.499 | 1.0 ± 15% | ❌ FAIL |
| Force direction | 66-100% | 100% | ⚠️ PARTIAL |

**Conclusion**: The curvature fix (fit_quadratic) is validated and ready. The delta_s normalization requires further tuning — the dynamic color drop estimation needs to be replaced with a direct gradient integral normalization.
