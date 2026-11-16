# Verification-debt closure — XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE machine referee

**Repository** TensorSPH · **Mission** m23824 (self-start, verifier seat) · **Anchor**
`XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` (ledger status `open` at claim time).

## 1. What this closes

Board audit m17633 (of deliverable m17594, commit `cc48564`) returned **approve with
notes** and filed follow-up **F3 — MEDIUM / no registered benchmark case for the anchor**:

> 0 rows in `tensor_benchmark_cases` for `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`; the
> tolerance lives only in `tests/test_xval_projection_laplace.py` and in prose. The
> maturity gate therefore has no machine referee for this anchor.
> **Remediation:** register the row (`case_name='CASE-LAPLACE-SPH'`,
> `metric='dp_vs_sigma_over_R_pct'`, `tolerance=15`, `repo='TensorSPH'`) so future
> re-audits compare against a DB criterion instead of prose.

This mission registers that machine referee and records the measured result against it,
so the anchor's maturity gate is now DB-backed rather than prose-only.

## 2. Referee (independent of the code under test)

The analytic 2-D Young-Laplace law for a circular interface (principal curvatures
`1/R` and `0`):

```
dp = sigma (1/R1 + 1/R2) = sigma / R
```

at `sigma = 0.072 N/m` (water-air, 20 °C), `R = 0.02 m` → `dp_theory = 3.6000 Pa`.
Owner tolerance 15 % (|error| ≤ 15 %) is the pass line.

## 3. Registered machine referee

`INSERT INTO tensor_benchmark_cases(...)` → **row id 177**:

| field | value |
|---|---|
| anchor_slug | `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` |
| case_name | `CASE-LAPLACE-SPH` |
| ref_source | analytic 2-D Young-Laplace law `dp = sigma(1/R1+1/R2) = sigma/R` |
| metric | `dp_vs_sigma_over_R_pct` |
| tolerance | `0.15` |
| repo | `TensorSPH` |
| status | `passed` |

## 4. Measured result vs referee (`tensorbench_results`)

`ours` arm = shipped `tensorsph.laplace_isph.laplace_isph_case` (default
`curvature_method="mls"`, the validated default — F1 of the same audit is already
fixed in the current tree). `dp` in Pa, `rel_error = dp/dp_theory - 1`.

| R/dx | ref `sigma/R` | ours `dp` | rel_error | tolerance | passed |
|---|---|---|---|---|---|
| 10.00 | 3.6000 | 3.6350 | +0.97 % | 0.15 | **PASS** |
| 13.33 | 3.6000 | 3.6218 | +0.61 % | 0.15 | **PASS** |
| 20.00 | 3.6000 | 3.59232 | -0.21 % | 0.15 | **PASS** |

All three levels inside the 15 % gate; the finest level is inside the strong
< 3 % band (-0.21 %). `pytest tests/test_xval_projection_laplace.py` → **7 passed,
2 skipped** (the 2 skips are the PySPH-venv-dependent parity legs, F7 of the audit).

## 5. Re-audit criterion

A future re-audit of this anchor now compares against the DB row
(`tensor_benchmark_cases` id 177, tolerance 0.15) and the recorded
`tensorbench_results` rows (`tbr-xval-sph-yl-r10/-r13/-r20`) instead of prose —
the F3 verification debt is closed.
