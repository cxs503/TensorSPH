# CSF Curvature Verification: fit_quadratic gap on hex lattice

**Mission**: m3432 (self-start, verifier seat)  
**Date**: 2026-11-03  
**Repository**: TensorSPH  
**Verdict**: Benchmark-test mismatch fixed; fit_quadratic curvature gap exposed

---

## 1. What was done

### 1.1 Benchmark fix (`benchmarks/laplace_law/run.py`)

The `csf_fields` wrapper in the Laplace-law benchmark was calling
`csf_fields_consistent(pos, mass, pairs, h, rho0)` with the default
`curvature_scheme="mls"`, while the solver's `Config` defaults to
`csf_curvature="fit_quadratic"`. This meant the benchmark was scoring the
MLS operator (~42 % curvature error) instead of the solver's actual default.

**Fix**: The wrapper now passes `curvature_scheme=getattr(cfg, "csf_curvature", "mls")`
and rebuilds the pair list with the wider stencil (4h) that the fit operator
requires, exactly as `surface_tension_acceleration` does.

### 1.2 Test update (`tests/test_laplace_law.py`)

The `test_curvature_reproduces_circle` test was `xfail(strict=True)` with the
reason "MLS div n on a circle is ~25 % high". After the benchmark fix, the
test now uses the solver's default `fit_quadratic` scheme. However, the
measured error is ~37 % (not the <1 % claimed in the docstring), so the xfail
is retained with an updated reason documenting the actual gap.

---

## 2. Curvature error decomposition

On the verification lattice (R = 0.02 m, dx = 0.004 m, R/dx = 5, hex packing):

| Scheme | δ_s-weighted κ [1/m] | Exact [1/m] | Rel. error |
|--------|---------------------|-------------|------------|
| MLS div n | 71.21 | 50.00 | **+42.4 %** |
| fit_quadratic (2h pairs) | 36.60 | 50.00 | **−26.8 %** |
| fit_quadratic (4h pairs) | 31.56 | 50.00 | **−36.9 %** |

The fit_quadratic scheme is the solver's default (`Config.csf_curvature`) but
its measured error on the hex lattice is ~37 %, not the <1 % claimed in the
`csf_curvature.py` docstring. The docstring references
`docs/verification_surface_tension_curvature.md` which claims <1 % error,
but that claim is not borne out on the verification lattice used by the
benchmark.

### 2.1 Why the wider stencil makes it worse

The fit_quadratic operator fits a quadratic `v = c + b·u + a·u²` to interface
particles in the local (tangent, normal) frame. With the 2h kernel-support
pairs, the stencil contains only 3-4 "staircase" points per particle on a
lattice, giving −26.8 % error. With the 4h stencil (as required by the
operator), more particles are included but they are further from the interface
and carry lower `δ_s` weights, which may be degrading the fit quality on this
particular lattice geometry.

---

## 3. What still passes

After the benchmark fix, the following tests pass (9/10, 1 xfail):

| Test | Status | Notes |
|------|--------|-------|
| `test_instant_diagnostics_structure` | PASS | Structure check |
| `test_csf_force_is_net_inward` | PASS | Cut-force direction |
| `test_surface_detector_covers_the_interface` | PASS | Surface detection |
| `test_gamma_zero_disables_surface_tension` | PASS | σ = 0 no-op |
| `test_curvature_reproduces_circle` | **XFAIL** | ~37 % error (see §2) |
| `test_csf_force_direction_is_radial` | PASS | Radial alignment |
| `test_cut_force_identity` | PASS | Within 15 % |
| `test_laplace_pressure_jump` | PASS | Within 15 % |
| `test_committed_verdicts_are_self_consistent` | PASS | JSON consistency |
| `test_spurious_current_is_below_the_literature_threshold` | PASS | Adami criterion |

---

## 4. Follow-up

1. **P0 (curvature)**: Investigate why `fit_quadratic` gives ~37 % error on
   the hex lattice instead of the claimed <1 %. Possible causes:
   - The quadratic fit may be sensitive to the lattice staircase geometry
   - The `δ_s` weighting may not be optimal for this lattice
   - The stencil radius (4h) may need tuning

2. **P1 (free-surface pressure closure)**: The Laplace pressure jump passes
   at 15 % but the residual ~34 % is due to free-surface pressure closure
   (Tait pressure clamped at p ≥ 0). This is the P0 item in
   `backlog_update_m2971.md`.

3. **Documentation**: The `csf_curvature.py` docstring claims <1 % error for
   `fit_quadratic` on a circle. This claim should be updated to reflect the
   actual measured error on the verification lattice, or the conditions under
   which <1 % is achieved should be documented.

---

## 5. Files changed

| File | Change |
|------|--------|
| `benchmarks/laplace_law/run.py` | `csf_fields` wrapper now passes `curvature_scheme` from config and rebuilds pairs with 4h stencil for fit operators |
| `tests/test_laplace_law.py` | Updated docstring and xfail reason for `test_curvature_reproduces_circle` |
| `docs/verification_csf_curvature_m3432.md` | This report |
