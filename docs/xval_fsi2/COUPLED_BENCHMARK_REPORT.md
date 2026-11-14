# Coupled FSI2 Benchmark: TensorSPH vs SPHinXsys Reference

**Anchor**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`
**Mission**: m20858
**Date**: 2026-11-15
**Revision**: agent/m20858

## 1. Case Introduction

### Physical Background
The Turek-Hron FSI2 benchmark (Turek & Hron, LNCSE 53, 2006) is the canonical
test for fluid-structure interaction solvers. A 2-D channel (2.5 × 0.41 m)
carries laminar flow (Re=100) past a rigid cylinder (D=0.1 m) with a flexible
elastic beam (L=0.35 m, t=0.02 m) attached downstream. The beam undergoes
vortex-induced vibration (VIV); engineering quantities of interest are drag/lift
coefficients and beam tip displacement amplitudes.

### Reference Values
| Quantity | Turek-Hron [m] | SPHinXsys OSS [m] | OSS rel. error |
|---|---|---|---|
| tip_x_amplitude | 0.0028 | 0.0156 | +456 % |
| tip_y_amplitude | 0.0082 | 0.0797 | +872 % |

The SPHinXsys `test_2d_fsi2` record is **not** the Turek-Hron benchmark case
(see `docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md`): the OSS case has
`Ae = 1.4e3` vs the benchmark's `5.6e3`, and `rho_s/rho_f = 10` vs `1`.

### This Benchmark's Question
Can TensorSPH's strong-coupling FSI2 driver reproduce the **SPHinXsys archived
record** (not the Turek-Hron paper values) within the owner accuracy standard
(<15 %)? This is a code-to-code comparison at matched parameters.

## 2. Methodology

### Design
- **Similarity-corrected case**: `U_mean = 0.2 m/s`, `Re = 100`, `Ae = 5600`,
  `rho_s/rho_f = 1` (the benchmark point, not the OSS regression point)
- **Coupling schemes**: loose (baseline), strong-IQN-ILS, strong-Aitken
- **Metric**: beam-tip displacement half-range amplitudes after warmup
- **Gate**: |rel_error| < 15 % (owner standard), < 3 % (reporting target)
- **Reference**: SPHinXsys archived `BeamObserver_Position_Run_0_result.xml`
  (converted to physical units by `tensorsph.fsi2_reference`)

### Implementation
- Driver: `benchmarks/xval_fsi2/run_coupled_benchmark.py`
- Strong coupling: `src/tensorsph/fsi2_strong_coupling.py` (IQN-ILS, Aitken)
- Reference parsing: `src/tensorsph/fsi2_reference.py`
- Tests: `tests/test_fsi2_coupled_benchmark.py` (9 tests)

## 3. Computational Results (Probe: dx=0.02 m, 40 steps)

### Loose Coupling at rho_s/rho_f = 1
| Metric | Value |
|---|---|
| tip_y_amplitude | 0.5609 m |
| tip_x_amplitude | 0.0000 m |
| vs OSS reference | +604 % (FAIL) |
| vs Turek-Hron | +6740 % (FAIL) |
| finite | Yes |
| wall clock | 10.2 s |

The loose coupling produces a large displacement (0.56 m) that is physically
unreasonable for this case. The beam is being driven by an unbalanced load
because the loose coupling cannot converge the interface at ratio 1.

### Strong Coupling (IQN-ILS) at rho_s/rho_f = 1
| Metric | Value |
|---|---|
| tip_y_amplitude | ~0.0000 m (40 steps, flow still ramping) |
| finite | Yes |
| coupling converged | Yes (1 iter/step) |
| wall clock | >120 s (timeout at 40 steps) |

The strong coupling stays finite and converges, but 40 steps (t ≈ 0.04 s) is
insufficient for the flow to develop. The inlet ramp time is 1.0 s, so the
parabolic profile is still building. The beam response is expected to grow
over hundreds to thousands of steps.

### Reference Comparison
| Source | tip_x_amp [m] | tip_y_amp [m] |
|---|---|---|
| Turek-Hron benchmark | 0.0028 | 0.0082 |
| SPHinXsys OSS (run 0) | 0.0156 | 0.0797 |
| TensorSPH probe (loose, 40 steps) | ~0.0 | 0.5609 |
| TensorSPH probe (strong, 40 steps) | ~0.0 | ~0.0 |

## 4. Error Judgment

| Comparison | tip_x error | tip_y error | <15 %? |
|---|---|---|---|
| OSS vs Turek-Hron | +456 % | +872 % | **FAIL** |
| TensorSPH loose vs OSS | N/A (divergent behavior) | +604 % | **FAIL** |
| TensorSPH strong vs OSS | N/A (insufficient time) | N/A | **NOT EVALUATED** |

**Verdict**: The probe confirms that:
1. The OSS record is **not** the Turek-Hron benchmark (previously established)
2. The loose coupling at ratio 1 produces unphysical displacements
3. The strong coupling stays finite but needs long-time integration to develop
   a meaningful response

The full benchmark (dx = 0.005 m, t_end = 10 s, ~10,000 steps) must be run on
the cluster to produce a definitive comparison.

## 5. Notes and Limitations

### What Was Done
- ✅ Coupled benchmark infrastructure (script + tests)
- ✅ Short probe run (dx = 0.02 m, 40 steps)
- ✅ All 9 tests pass
- ✅ Reference archive integrity verified

### What Was Not Done
- ❌ Full long-time integration (t_end = 10 s) — requires cluster submission
- ❌ dx = 0.005 m production run — computationally expensive (~20k fluid particles)
- ❌ Comparison against Turek-Hron paper values — requires the full run first

### Known Issues
1. The strong coupling is slow (~0.5 s/step at dx=0.02 m with 2559 fluid particles)
2. The inlet ramp time (1.0 s) means ~1000 steps are needed before the flow is
   fully developed
3. The beam response at the benchmark point is small (mm scale), requiring
   long integration to accumulate measurable amplitudes

### Recommendations
1. Submit the full benchmark to the cluster queue (dx = 0.005 m, t_end = 10 s)
2. Consider using a coarser dx for the initial comparison (dx = 0.01 m)
3. The OSS record should not be used as the acceptance reference — it is a
   different parameter point

## 6. Artifacts

| Artifact | Path |
|---|---|
| Benchmark script | `benchmarks/xval_fsi2/run_coupled_benchmark.py` |
| Probe results | `artifacts/xval_fsi2/coupled_benchmark_probe.json` |
| Tests | `tests/test_fsi2_coupled_benchmark.py` (9 tests, all pass) |
| Reference archive | `benchmarks/xval_fsi2/reference/oss/` |
| Reference parser | `src/tensorsph/fsi2_reference.py` |
| Strong coupling | `src/tensorsph/fsi2_strong_coupling.py` |
| Similarity audit | `src/tensorsph/fsi2_similarity.py` |
| GAP analysis | `docs/xval_fsi2/GAP-SPH-FSI-BENCHMARK_xval.md` |
