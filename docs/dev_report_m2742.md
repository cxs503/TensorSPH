# TensorSPH m2742: Laplace-law benchmark scheme alignment

**Mission**: m2742 (self-start, verifier seat)  
**Date**: 2026-11-03  
**Repository**: TensorSPH  
**Verdict**: PASS — benchmark now scores the consistent CSF scheme

## 1. What was done

The `benchmarks/laplace_law/run.py` benchmark was scoring curvature, delta_s,
and cut-force using the **legacy** neighbour-count colour function, while the
acceleration field came from the **consistent** scheme (kernel-sum colour + MLS
operators). This inconsistency meant:

- `test_curvature_reproduces_circle` tested the legacy scheme's curvature
  (3-12x error), not the consistent scheme's (~25-30% error).
- The xfail reason was stale ("3-12x" instead of "~25-30%").
- The benchmark's `csf_fields` function duplicated logic that already exists
  in `tensorsph.csf_consistent.csf_fields`.

**Changes**:
1. `benchmarks/laplace_law/run.py`: replaced the inline legacy `csf_fields`
   with an import of `csf_fields_consistent` from `tensorsph.csf_consistent`.
   Both `instant_diagnostics` and `relax` now use the consistent scheme for
   all field scoring.
2. `tests/test_laplace_law.py`: updated the module docstring and the xfail
   reason for `test_curvature_reproduces_circle` to reflect the consistent
   scheme's actual error (~25-30%, not 3-12x).

## 2. Verification

### 2.1 Test results

```
tests/test_laplace_law.py: 9 passed, 1 xfailed
tests/test_csf_consistent.py: 18 passed
tests/test_csf_curvature_fit.py: 11 passed
tests/test_surface_tension.py: 16 passed
Total: 45 passed, 1 xfailed
```

### 2.2 Instant diagnostics (consistent scheme, R = 0.02 m, sigma = 7.2 N/m)

| dx [m] | N | kappa err | F_cut/(2σ) | cos_med |
|---|---|---|---|---|
| 0.003 | 161 | +30.0% | 1.110 | +0.999 |
| 0.002 | 362 | +24.9% | 1.077 | +1.000 |
| 0.0015 | 642 | +25.2% | 1.074 | +1.000 |
| 0.001 | 1450 | +30.6% | 1.122 | +1.000 |

These match the numbers in `docs/verification_surface_tension_consistent.md`
section 3.1, confirming the benchmark now scores the same scheme.

### 2.3 Error judgement

| Key quantity | Benchmark | Consistent scheme | < 15%? |
|---|---|---|---|
| Cut-force F_cut | 2σ = 14.4 N/m | 1.074-1.122 × theory | Yes |
| Curvature κ | 50 1/m | +24.9-30.6% | No |
| Force direction | cos = 1.0 | 0.999-1.000 | Yes |
| Laplace Δp | 360 Pa | passes at R/h~5 | Yes (coarse) |

## 3. What was NOT done

- The curvature error (~25-30%) was **not** fixed — this requires a second-order
  curvature estimator or a different discretisation, which is a P0 physics task
  beyond the scope of this benchmark alignment.
- The `benchmarks/csf_consistent/run.py` was not modified (it already uses the
  consistent scheme correctly).
- No `src/` files were changed — this is a benchmark/test alignment only.

## 4. Files changed

- `benchmarks/laplace_law/run.py` — replaced legacy csf_fields with consistent scheme
- `tests/test_laplace_law.py` — updated docstring and xfail reason
- `benchmarks/laplace_law/instant_report.json` — regenerated with consistent scheme data
