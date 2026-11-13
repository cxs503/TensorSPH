-- Verdict for m17598: XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE
-- Audit by: board-audit (m17635)
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorSPH',
  'm17598 XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE',
  17635,
  4,
  '{"impl_files": ["src/tensorsph/execution_policy.py (ExecutionInstance singleton, get_uniform_nd_range, iterate_blocks, WorkBlock, work_item_counts)", "src/tensorsph/wcsph/swept.py (summation_density_vectorised, summation_density_swept, summation_density_naive, build_csr)"], "entry": ["benchmarks/xval_sycl_execution_policy/run.py"], "run_artifacts": ["artifacts/xval_execution_policy/results.json", "artifacts/xval_execution_policy/provenance.json", "artifacts/xval_execution_policy/summary.json", "artifacts/xval_execution_policy/fig_error_vs_workgroup_size.png", "artifacts/xval_execution_policy/fig_padding_cost.png", "artifacts/xval_execution_policy/fig_density_field.png", "artifacts/xval_execution_policy/fig_wallclock.png"], "tests": ["tests/test_execution_policy.py (117 tests, all PASS in 16.52s)"], "integration_callers": ["benchmarks/xval_sycl_execution_policy/run.py imports tensorsph.execution_policy, tensorsph.wcsph.swept, tensorsph.neighbor_search, tensorsph.wcsph.cases", "tests/test_execution_policy.py imports tensorsph.execution_policy, tensorsph.wcsph.swept"], "benchmark": {"xval_table": "4 comparisons: summation density max_rel_err=0.0 PASS, Tait pressure 0.0 PASS, vs scalar loop 6.4e-16 PASS, negative control 635.4 FAIL-as-expected", "criteria": "owner_gate=0.15, engineering_target=1e-12, negative_control_must_exceed=0.15", "all_gates_pass": true, "worst_density_rel_err": 0.0, "worst_control_rel_err": 635.44622901483, "oss_facts_verified": "11/11 facts found in SPHinXsys checkout commit 5241a890"}, "repro_cmd": "python benchmarks/xval_sycl_execution_policy/run.py (verified: produces results.json with all_gates_pass=true)", "external_anchor": "Xiangyu-Hu/SPHinXsys commit 5241a890; 4 OSS files verified by sha256: execution_policy.h, implementation_sycl.h, algorithm_primitive_sycl.h, device_copyable_variable.h"}',
  'board-audit'
);
