# CSF Surface Tension Verification: Consistent Scheme vs Laplace Law

> **⚠️ SUPERSEDED-BY: docs/verification_ledger.md** —— 本文数字是**历史口径**，已被后续复核取代/校正，不得再作为当前精度引用。
> 当前证据分级与逐项误差见机器生成的验证账本 `docs/verification_ledger.md`（由 `benchmarks/verification_ledger/ledger.py` 从已提交工件注入）。

**Date**: 2026-11-03  
**Mission**: m3028 (selfstart · verifier席)  
**Scope**: TensorSPH CSF surface tension model verification against Young-Laplace law

## 1. Problem Statement

The CSF surface tension model (m277) was merged without verification against the
analytic Young-Laplace law. The benchmark `benchmarks/laplace_law/run.py` had a
**mismatch**: its diagnostic `csf_fields()` function used the legacy neighbor-count
color function, while the simulation's `compute_surface_tension_force()` defaulted
to the consistent scheme (kernel-sum color + MLS operators). This meant the
validation report measured the wrong scheme.

## 2. What Was Fixed

### 2.1 Benchmark–Simulation Alignment

`benchmarks/laplace_law/run.py` was updated so its `csf_fields()` function delegates
to `tensorsph.csf_consistent.csf_fields()` instead of reimplementing the legacy
neighbor-count color function. This ensures diagnostics measure the same scheme
that the simulation uses.

**Key change**: replaced the local `csf_fields()` (neighbor-count color, uncorrected
SPH gradient/divergence) with a thin wrapper around `csf_consistent.csf_fields()`
(kernel-sum color, MLS gradient/divergence, Shepard-smoothed normal).

### 2.2 Validation Artifacts Updated

- `benchmarks/laplace_law/validation_report.json` — regenerated with consistent scheme
- `benchmarks/laplace_law/instant_report.json` — regenerated with consistent scheme
- `benchmarks/laplace_law/convergence/*.json` — regenerated (coarse resolutions)
- `tests/test_laplace_law.py` — updated assertions to reflect new results

## 3. Results

### 3.1 Instant Diagnostics (static evaluation, no time integration)

| Resolution | Cut Force Ratio | Inward Cos (median) | κ Error (δ_s-weighted) |
|-----------|-----------------|---------------------|------------------------|
| dx=0.003  | 1.110           | 0.9993              | 30.0%                  |
| dx=0.002  | 1.077           | 0.9996              | 24.9%                  |
| dx=0.0015 | 1.074           | 0.9996              | 25.2%                  |
| dx=0.001  | 1.122           | 0.9996              | 30.6%                  |

**Verdicts**:
- ✅ Cut force identity: all within 15% (platform bar)
- ✅ Force direction: all > 0.999 median cosine (excellent radial alignment)
- ❌ Curvature: ~25% error with MLS divergence operator (above 15% bar)

### 3.2 Laplace Pressure Jump (relaxation, 380 steps, R/h ≈ 5)

| Quantity | Theory | Measured | Error |
|----------|--------|----------|-------|
| Δp (Pa)  | 360.0  | 380.2    | +5.6% |

**Verdict**: ✅ Within 15% platform bar (was −77% to −118% with legacy scheme).

### 3.3 Spurious Currents

| Quantity | Criterion | Measured | Verdict |
|----------|-----------|----------|---------|
| max\|v\|/√(σ/ρR) | < 0.1 (Adami 2012) | 0.041 | ✅ Pass |

**Verdict**: ✅ Well below the literature threshold (was 0.30–0.42 with legacy scheme).

## 4. Test Suite Results

```
54 passed, 1 xfailed
```

The single xfail is `test_curvature_reproduces_circle` — the MLS divergence operator
systematically overestimates curvature by ~25% on a circle. This is a known limitation
of the MLS div n operator on a particle shell (second derivative of a discretised
quantity). The interface-fit operators (`fit_quadratic`, `fit_circle`) in
`csf_curvature.py` are available but need further tuning for the lattice droplet case.

## 5. Remaining Deficits

| Deficit | Status | Priority |
|---------|--------|----------|
| Curvature ~25% error (MLS div n) | Open — needs interface-fit tuning | P1 |
| 3D extension of CSF model | Not started | P0 |
| Experimental data comparison | Not started | P1 |
| Multiphase SPH | Not started | P1 |

## 6. References

- Brackbill, Kothe & Zemach (1992), JCP 100:335 — CSF method
- Adami, Ihmsen & Adams (2012), JCP 231:7057 — spurious currents criterion
- Bonet & Lok (1999), CMAME 180:97 — MLS consistent operators
- `docs/verification_surface_tension_consistent.md` — detailed verification report
- `benchmarks/laplace_law/validation_report.json` — committed validation data
