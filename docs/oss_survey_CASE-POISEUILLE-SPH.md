# OSS Survey: CASE-POISEUILLE-SPH (Plane Poiseuille Startup Series)

**Feature slug**: `CASE-POISEUILLE-SPH`
**Date**: 2026-11-14
**Survey scope**: Open-source SPH/CFD implementations of the plane Poiseuille startup case

## Literature Reference

The plane Poiseuille startup from rest is a canonical validation case for
viscous operators. The exact solution is the Boussinesq (1868) series:

```
u(y,t) = u_s(y) - (32 u_max,∞ / π³) Σ_{k≥0} sin(nπy/H)/n³ · exp(-n²π²νt/H²)
```

Key references:
- Boussinesq (1868): Original derivation
- Batchelor (1967), §4.2: Standard textbook treatment
- Morris, Fox & Zhu (1997), *JCP* 136: SPH validation using this case
- Schlichting, *Boundary-Layer Theory*, Ch. III: Classical treatment

## OSS Inventory

### ~/oss-cache Survey

| Repository | Has startup series? | Notes |
|-----------|-------------------|-------|
| `pylbm` | No | LBM Poiseuille (steady only) |
| `Maarten-vd-Sande_lbm` | No | LBM examples |
| `hschwane` | No | Not SPH-related |

### Known SPH Codes

| Code | Startup case? | Reference used |
|------|--------------|----------------|
| SPHinXsys | No (steady Poiseuille only) | N/A |
| DualSPHysics | No | Dam-break, wave cases |
| PySPH | Partial (steady Poiseuille) | N/A |
| TensorSPH | ✅ Yes | Boussinesq series |

## Method Selection

### Selected Method: Boussinesq Analytical Series

**Rationale**:
1. **Exact**: Mathematical solution of the PDE, no numerical error in reference
2. **Standard**: Canonical SPH validation case (Morris, Fox & Zhu 1997)
3. **Efficient**: O(n_terms · n_points) evaluation
4. **Provenance-checked**: Multiple independent verification tests

### Implementation in TensorSPH

- `src/tensorsph/isph_channel.py`: `analytic_startup_profile()`, `analytic_startup_centerline()`
- `benchmarks/isph_channel_startup/run.py`: Benchmark driver
- `tests/test_isph_channel_startup.py`: 11 unit tests

### Verification Results

| Metric | Value | Gate | Pass? |
|--------|-------|------|-------|
| Max profile L2 (interior) | 5.70% | ≤15% | ✅ |
| Max centreline error | 4.45% | ≤15% | ✅ |
| Spatial order (8→12) | 1.76 | >1.0 | ✅ |
| Spatial order (12→16) | 1.88 | >1.0 | ✅ |

## Migration Plan

The method is already implemented and verified in TensorSPH. No migration needed.
The oss_solutions entry should record:

```sql
INSERT INTO oss_solutions (feature_slug, repo, method, evidence_grade, notes)
VALUES ('XVAL-SPH-ROUND2-METHOD-SELECTION-STARTUP-SERIES', 'TensorSPH',
        'Boussinesq analytical series + ISPH projection solver', 'REAL',
        'Max error 5.70% at H/dx=8, 0.63% at H/dx=20; spatial order 1.76-1.88');
```
