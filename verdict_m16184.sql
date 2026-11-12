-- board-audit verdict for m16184 (CASE-SPH-SOD-SHOCK-COMPRESSIBLE), auditor m16254
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorSPH',
 'm16184 CASE-SPH-SOD-SHOCK-COMPRESSIBLE',
 16254,
 4,
 '{
  "impl_files": [
    "src/tensorsph/compressible/sod.py:116 SodSPHConfig",
    "src/tensorsph/compressible/sod.py:162 exact_star_state",
    "src/tensorsph/compressible/sod.py:194 exact_wave_positions",
    "src/tensorsph/compressible/sod.py:338 run_sod_sph",
    "src/tensorsph/compressible/sod.py:446 evaluate_metrics",
    "src/tensorsph/compressible/__init__.py (public API exports)"
  ],
  "entry": [
    "scripts/run_case_compressible_sod.py (one-command repro, 42s on audit host)",
    "benchmarks/case_sod_shock_compressible/run.py (case-pool entry, prints ref-vs-got table)"
  ],
  "run_artifacts": [
    "runs/compressible_sod/production.json",
    "runs/compressible_sod/reference.json",
    "runs/compressible_sod/studies.json",
    "artifacts/compressible_sod/sod_profiles.png",
    "artifacts/compressible_sod/contact_zoom.png",
    "artifacts/compressible_sod/studies.png",
    "docs/validation/compressible_sod/REPORT.md"
  ],
  "tests": [
    "tests/test_compressible_sod.py (17 passed in 12.9s, re-run by auditor 2026-11-12; conftest.py pins local src)"
  ],
  "integration_callers": [
    "tensorsph.compressible subpackage (importable public API)",
    "benchmarks/case_sod_shock_compressible/run.py",
    "scripts/run_case_compressible_sod.py"
  ],
  "benchmark": {
    "registered_criterion": "tensor_case_pool id=123 case_code=SPH-SOD-SHOCK-COMPRESSIBLE tol_policy=<5% | metric: density/pressure plateau values and shock position vs Sod exact solution (t=0.2); mission goal identical",
    "note": "tensor_benchmark_cases has NO row for anchor_slug=CASE-SPH-SOD-SHOCK-COMPRESSIBLE (0 rows, auditor-verified 2026-11-12); criterion registered in tensor_case_pool + mission goal instead. Bookkeeping gap, not a functional one.",
    "ref": "Sod 1978 JCP 27(1):1-31 doi:10.1016/0021-9991(78)90023-2; star values p*=0.30313 u*=0.92745 rho*_L=0.42632 rho*_R=0.26557 (Toro 1999 table 4.1 cross-check, dev <0.05%)",
    "got_vs_ref": "11/11 gated metrics within tolerance: max|rel err|=0.78% (p*_L) < 3% < 5%; L1(rho)=1.14%; contact pressure jump 0.70% of p*; mass err 0.0; energy change -1.5e-6",
    "pass": true
  },
  "repro_cmd": "python scripts/run_case_compressible_sod.py (auditor re-ran: physics bit-identical to committed artifacts; only wall_s/timestamp differ; 3 PNGs byte-identical)",
  "external_anchor": "Sod 1978 doi:10.1016/0021-9991(78)90023-2; Inutsuka 2002 JCP 179:238 doi:10.1006/jcph.2002.7064",
  "commit": "b6deb05 (PR #537 merged)",
  "report": "tensor_reports id=15299 (six sections ①-⑥ verified present; numbers match JSON artifacts)",
  "audit_checks": {
    "run_artifacts_exist": true,
    "ref_vs_got_within_tolerance": true,
    "repro_cmd_executed": true,
    "tests_passed": true,
    "report_six_sections": true,
    "determinism": "bit-identical physics across runs (wall_s/timestamp only)"
  }
 }'::jsonb,
 'board-audit'
)
RETURNING id, level, gate_state;
