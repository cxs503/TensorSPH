# Audit Report: m17598 XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE

**Auditor**: board-audit (m17635)
**Date**: 2026-11-13
**Target**: m17598 — Cross-validation of SPHinXsys SYCL execution policy port to TensorSPH

---

## 1. Audit Scope

This is a **交差核验单** (completion verification), not a re-do. The mission m17598 reported `succeeded` status. This audit verifies:
1. The three-piece acceptance set exists (run_artifacts, benchmark ref vs got, repro_cmd)
2. The got values are within tolerance per the benchmark criteria
3. The report has all six required sections
4. The maturity level is correctly assigned

## 2. Evidence Verification

### 2.1 Implementation Files (✓ REAL)
- `src/tensorsph/execution_policy.py`: ~200 lines of real code implementing:
  - `ExecutionInstance` singleton with deleted copy/assign (analogue of OSS `= delete`)
  - `get_uniform_nd_range()` — byte-for-byte replica of OSS conditional round-up
  - `iterate_blocks()` — yields padded `WorkBlock` with explicit tail mask
  - `work_item_counts()` — launch bookkeeping
- `src/tensorsph/wcsph/swept.py`: Real computation code:
  - `summation_density_vectorised()` — reference arm using `index_add`
  - `summation_density_swept()` — ported arm with padded masked sweep
  - `summation_density_naive()` — independent scalar loop (third opinion)
  - `build_csr()` — symmetric ragged adjacency builder

### 2.2 Entry Point (✓ REAL)
- `benchmarks/xval_sycl_execution_policy/run.py`: ~600 lines, full measurement harness
  - Predeclares criteria before run (GATE_OWNER=0.15, GATE_TARGET=1e-12, GATE_CONTROL=0.15)
  - Sweeps local_size ∈ {1,3,7,8,16,64,128,256,1024} and dx ∈ {0.03, 0.02, 0.0125}
  - Writes results.json, provenance.json, summary.json, 4 figures

### 2.3 Run Artifacts (✓ REAL)
| Artifact | Size | Content |
|---|---|---|
| results.json | 28KB | Full benchmark data: xval_table, local_size_sweep, particle_sweep, naive_crosscheck |
| provenance.json | 5.7KB | OSS checkout info, 4 files with sha256, 11 facts verified |
| summary.json | 305B | Gate pass summary |
| fig_error_vs_workgroup_size.png | 71KB | Error vs work-group size plot |
| fig_padding_cost.png | 41KB | Idle fraction plot |
| fig_density_field.png | 138KB | Density field + deviation plot |
| fig_wallclock.png | 49KB | Wall clock comparison |

### 2.4 Tests (✓ REAL, 117/117 PASS)
- `tests/test_execution_policy.py`: 117 assertions covering:
  - Singleton/non-copyability invariants
  - OSS round-up formula vs independent ceil re-derivation (16 parameter combos)
  - Block coverage (every index exactly once)
  - Invariance across 11 work-group sizes
  - Negative control (unmasked tail breaks result)
  - Diagnostics reporting
- Verified: `pytest tests/test_execution_policy.py -v` → **117 passed in 16.52s**

### 2.5 Benchmark Results (✓ PASS)
| Quantity | Ref | Got | Max Rel Err | <15% | <1e-12 |
|---|---|---|---|---|---|
| summation density, worst work-group | vectorised index_add | ported sweep | **0.0** | PASS | PASS |
| Tait pressure, worst work-group | Tait on vectorised density | ported sweep | **0.0** | PASS | PASS |
| summation density vs scalar loop | independent scalar loop | ported sweep | **6.4e-16** | PASS | PASS |
| negative control (mask removed) | vectorised index_add | ported sweep | **635.4** | FAIL ✓ | FAIL ✓ |

- `all_gates_pass: true`
- `worst_density_rel_err: 0.0` (exact match, float64 round-off)
- `worst_control_rel_err: 635.4` (negative control correctly exceeds 15% gate)
- OSS facts: **11/11 verified** in SPHinXsys checkout (commit 5241a890)

### 2.6 Repro Command (✓ EXECUTABLE)
- `python benchmarks/xval_sycl_execution_policy/run.py` — verified working, produces consistent results
- Quick run confirmed: `all_gates_pass: true`, `worst_density_rel_err: 0.0`

### 2.7 Report Structure (✓ 7 sections, exceeds 6-element requirement)
1. Problem, reference and provenance
2. Method and implementation
3. Parameters (complete, incl. dimensionless groups)
4. Cross-validation table (ref vs got, gate check)
5. Sensitivity and convergence
6. Differences, port trade-offs, failure modes, friction
7. Reproduce

## 3. Maturity Level Assessment

| Level | Criterion | Met? |
|---|---|---|
| L0 | Document only | N/A (has real code) |
| L1 | Math/unit verification | ✓ (117 tests with analytical checks) |
| L2 | Independent function (input→compute→output) | ✓ (real code in package) |
| L3 | System integration | ✓ (integrated into tensorsph package, called from benchmarks+tests) |
| L4 | Benchmark verification (complete case + error analysis + reproducible) | ✓ (full xval with ref/got comparison, predeclared thresholds, 4 figures, repro_cmd verified) |
| L5 | Research engineering | ✗ (architectural equivalence test, not a full physics result; OSS side read-not-run) |

**Verdict: L4** — Complete benchmark verification with error analysis, reproducibility, and figures. The comparison is against a well-defined reference with predeclared thresholds, and the negative control validates the test isn't vacuous.

## 4. Limitations & Caveats

1. **OSS side read-not-run**: No SYCL device available on loong64 host; comparison is source-to-implementation, not run-to-run.
2. **Static snapshot only**: No time integration, no wall pressure model, no free-surface closure. This is a numerical equivalence test, not a hydrostatic result.
3. **tensor_benchmark_cases**: No entry for this anchor in the benchmark cases table (same situation as other missions — the table has limited coverage).
4. **No tensor_reports entry**: Mission 17598 has no report in the tensor_reports table (API returned 404).

## 5. Disposition

**CONFIRMED** — The mission's claims are substantiated:
- All three acceptance pieces exist and are real
- Benchmark numbers are within tolerance (0.0 error for the ported sweep)
- Negative control correctly fails (635× error without mask)
- 117 tests pass
- Repro command verified
- Report has all required sections

**Level: L4** (Benchmark verification)

## 6. Verdict Inserted

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorSPH', 'm17598 XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE', 17635, 4, '<evidence JSON>', 'board-audit');
```
Row ID: 47
