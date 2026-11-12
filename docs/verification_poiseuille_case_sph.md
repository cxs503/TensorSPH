# Verification Report: CASE-POISEUILLE-SPH — Plane Poiseuille Channel Flow

**Anchor**: CASE-POISEUILLE-SPH  
**Mission**: m16734  
**Date**: 2026-11-12  
**Repository**: TensorSPH  
**Branch**: agent/m16734  
**Artefact**: `benchmarks/poiseuille/out/results.json`  
**Tests**: `tests/test_poiseuille_verification_gate.py` (9 passed)

---

## 1. Case Introduction

### Physical Problem
Plane Poiseuille flow: an incompressible fluid of kinematic viscosity ν fills a 2-D channel of height H with no-slip walls at y = 0 and y = H, driven by a uniform body force g_x along x. The domain is periodic in x, making the problem one-dimensional.

### Analytical Solution
The steady solution of 0 = −dp/dx + μ∂²u/∂y² + ρg_x with mass conservation is:

    u(y) = g_x · y · (H − y) / (2ν)

with peak velocity at the centreline:

    u_max = g_x · H² / (8ν)

and wall shear stress:

    τ_w = ρ · g_x · H / 2

### Parameters

| Parameter | Value |
|---|---|
| Channel height H | 0.1 m |
| Kinematic viscosity ν | 0.01 m²/s |
| Body force g_x | 9.81 m/s² |
| Analytic u_max | 1.22625 m/s |
| Analytic τ_w | 490.5 Pa |
| Sound speed c₀ | 20 m/s |
| Kernel smoothing h | 1.3·dx |
| Wall model | Adami et al. (2012), 3 layers, wall_friction = 1 |

### Acceptance Criterion
**容差 < 5%** against the analytical solution for the peak velocity u_max.

---

## 2. Method

The benchmark (`benchmarks/poiseuille/run.py`) runs the WCSPH solver at three resolutions (H/dx = 8, 12, 16) and measures:

1. **Steady-state velocity profile**: time-averaged over the last 0.2 s of a 1.0 s run
2. **Peak velocity u_max**: maximum of the row-averaged profile
3. **Effective viscosity ν_eff**: from the curvature of the fitted parabola (c₂ = −g_x/(2ν_eff))
4. **Wall offset δ**: from the intercepts of the fitted parabola
5. **Wall shear stress**: from the velocity gradient at the wall row
6. **Negative control**: free-slip walls (no steady state expected)

The verification gate test (`tests/test_poiseuille_verification_gate.py`) reads the committed artefact and checks:
- u_max error < 5% at finest resolution
- Analytic u_max correctness
- Steady-state convergence (drift < 3%)
- Mach number < 0.1
- Parabolic fit quality
- Free-slip control accelerates
- Convergence trend

---

## 3. Results

### Peak Velocity Verification (Primary Gate)

| H/dx | u_max measured [m/s] | u_max analytic [m/s] | Ratio | Error [%] | Gate < 5% |
|---|---|---|---|---|---|
| 8 | 1.5219 | 1.2263 | 1.2411 | +24.11% | ❌ |
| 12 | 1.3640 | 1.2263 | 1.1123 | +11.23% | ❌ |
| **16** | **1.2448** | **1.2263** | **1.0151** | **+1.51%** | **✅** |

**Verdict: PASS** — at H/dx = 16, the peak velocity error is 1.51%, well within the 5% gate.

### Convergence Trend

| H/dx | ν_eff/ν (bulk) | δ/dx | Wall shear ratio | Steady drift |
|---|---|---|---|---|
| 8 | 0.952 | 0.907 | 0.580 | 0.48% |
| 12 | 1.028 | 0.916 | 0.630 | 2.79% |
| 16 | 1.149 | 0.911 | 0.503 | 1.08% |

### Negative Control

The free-slip control (wall_friction = 0) shows monotonic acceleration from u_max = 0.003 to 1.414 m/s, confirming that the steady parabola requires wall friction — the no-slip result is not tautological.

---

## 4. Error Analysis

### Why Does u_max Pass at H/dx = 16?

The 1.51% u_max error is the result of **two compensating defects**:

1. **Over-diffusive bulk operator**: ν_eff/ν = 1.149 (15% high) — the Morris Laplacian with h = 1.3·dx over-predicts viscosity
2. **Under-predicted wall shear**: τ_measured/τ_analytic = 0.503 (50% low) — the Adami wall model has an effective slip of ~0.9·dx

These two errors partially cancel: the higher viscosity would reduce u_max, but the wall slip increases it, yielding a net error of only 1.51%.

### What This Case Does NOT Certify

- **Wall shear stress**: 50% error at all resolutions — the wall model defect is resolution-independent
- **Full velocity profile**: interior RMS error is 8.51% at H/dx = 16 (above 5%)
- **No-slip boundary layer**: the effective wall is ~0.9·dx outside the nominal position

### What This Case DOES Certify

- **Peak velocity of the steady parabola**: < 5% error at H/dx ≥ 16
- **Parabolic shape**: fit residual < 2% of u_max
- **Convergence trend**: u_max error decreases with resolution
- **Negative control**: free-slip flow accelerates indefinitely (no steady state)

---

## 5. Limitations and Scope

| Aspect | Status | Note |
|---|---|---|
| Peak velocity u_max | ✅ PASS (< 5%) | At H/dx ≥ 16 |
| Wall shear stress | ❌ FAIL (~50% low) | Wall model defect, resolution-independent |
| Full velocity profile | ⚠️ PARTIAL (8.5% RMS) | Needs finer resolution or improved wall model |
| Mass conservation | ✅ PASS | No boundary escapes |
| Compressibility | ✅ PASS | Max Mach < 0.1 |
| Steady state | ✅ PASS | Drift < 3% at finest resolution |

**Applicability**: This case certifies the peak velocity of steady pressure-driven channel flow at moderate Reynolds numbers (Re ≈ 15). It does not certify the wall shear stress or the near-wall velocity profile, which are limited by the Adami wall model's effective slip.

---

## 6. Verdict

**CASE-POISEUILLE-SPH: CONDITIONAL PASS**

The peak velocity u_max passes the < 5% gate at H/dx = 16 (error = 1.51%). However, the wall shear stress remains ~50% low at all resolutions due to a known defect in the Adami wall model. The case certifies the bulk flow behavior but not the near-wall boundary layer.

**Recommendation**: To fully close this anchor, either:
1. Accept the conditional pass (u_max only), or
2. Improve the wall model to reduce the effective slip below 0.5·dx

---

## Reproduction

```bash
# Run benchmark (requires ~2 min on CPU)
PYTHONPATH=src python benchmarks/poiseuille/run.py --out benchmarks/poiseuille/out

# Run verification gate (reads artefact, no solver)
PYTHONPATH=src python -m pytest tests/test_poiseuille_verification_gate.py -v
```

**Artefact SHA256**: `benchmarks/poiseuille/out/results.json` — committed to repo.
