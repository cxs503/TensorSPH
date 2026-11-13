-- board-audit verdict for m17020 (REPRO-SPH-MONAGHAN1994), auditor m17046
-- claimed level = 2  (increment only: post-processing code + recomputable artefact + tests;
--   case-level L4 is carried by m16873 + m16936 and is explicitly NOT credited here)
-- machine gate: expect gate_level=2 (L2 keys resolve; L3 denied: no integration caller,
--   increment absent from mainline; L4 denied: the "error budget" is not a valid ref-vs-got
--   comparison -- components sum to 27x the total and Ritter runs 2.1x past sqrt(2))
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorSPH',
 'm17020 REPRO-SPH-MONAGHAN1994',
 17046,
 2,
$json${
 "increment_credited": false,
 "scope_of_credit": "m17020's increment only (error-budget post-processing, make_artifacts section, 9 tests). The case-level capability (L4) is carried by m16873 (benchmarks/repro_monaghan_1994) + m16936 (audit/repair) and is NOT credited to m17020; this audit re-verified it but does not bank it twice.",
 "level_rationale": "L2 = real implementation + recomputable artefact + tests. L3 denied: error_budget.py has no caller outside its own test and is not on mainline. L4 denied: the increment's self-described 'error budget / Ritter comparison' is not a valid ref-vs-got comparison (F1: components sum to 27.04x the total it claims to decompose and a negative residual is attributed to discretization; F2: Ritter window reaches tau=2.967, 2.1x past its own stated validity limit tau<sqrt2).",
 "impl_files": [
  "/root/tp-agent/work/m17020/benchmarks/repro_monaghan_1994/error_budget.py",
  "src/tensorsph/dam_break.py"
 ],
 "impl_symbols": {
  "/root/tp-agent/work/m17020/benchmarks/repro_monaghan_1994/error_budget.py": "branch agent/m17020 c670396, 257 lines, NOT on mainline at audit time; 64-88 ritter_comparison (docstring says tau<sqrt2, loop uses WINDOW 0.5-3.0), 113-165 error_budget (non-closing decomposition), 155-160 the f-string that emits the -27.48% residual sentence, 232-257 main",
  "src/tensorsph/dam_break.py": "43 DamBreakConfig, 133 front_position, 183 ritter_front_over_a (Z/a=2tau), 240 run_case"
 },
 "entry": [
  "benchmarks/repro_monaghan_1994/run.py",
  "/root/tp-agent/work/m17020/benchmarks/repro_monaghan_1994/error_budget.py"
 ],
 "run_artifacts": [
  "benchmarks/repro_monaghan_1994/results/summary.json",
  "benchmarks/repro_monaghan_1994/results/trajectory_dx0.01_c025.json",
  "benchmarks/repro_monaghan_1994/results/trajectory_dx0.02_c025.json",
  "benchmarks/repro_monaghan_1994/results/trajectory_dx0.04_c025.json",
  "benchmarks/repro_monaghan_1994/results/refs_verified.json",
  "/root/tp-agent/work/m17020/benchmarks/repro_monaghan_1994/results/error_budget.json",
  "docs/board_audit/m17046/independent_recompute.json",
  "docs/board_audit/m17046/fig1_audit_m17046.png"
 ],
 "tests": [
  "tests/test_repro_monaghan_1994.py (16 tests, mainline)",
  "tests/test_repro_monaghan1994_evidence_contract.py (6 tests, mainline)",
  "tests/test_repro_monaghan1994_convergence_decomposition.py (8 tests, mainline)",
  "/root/tp-agent/work/m17020/tests/test_repro_monaghan_1994_error_budget.py (9 tests, branch only; auditor re-ran: 9 passed in 0.08 s -- they assert consistency, not additivity/validity, so they stay green while F1/F2 hold)",
  "tests/test_audit_m17046_repro_monaghan_rederivation.py (new audit guard: 6 passed / 3 skipped on the clean tree; reverse control with m17020's artefacts + document in place: 3 failed)"
 ],
 "integration_callers": [
  "src/tensorsph/verification/dam_break_2d.py (imports tensorsph.dam_break)",
  "src/tensorsph/vv/dambreak.py",
  "src/tensorsph/ice_splash.py",
  "src/tensorsph/sloshing.py"
 ],
 "integration_note": "callers above are the solver-library callers of the dam-break case (case-level integration from m16873). m17020's own script has NO caller besides its test and is not on mainline -- hence L3 not claimed.",
 "benchmark": {
  "registered_criterion": "tensor_benchmark_cases has NO row for anchor_slug=REPRO-SPH-MONAGHAN1994 (0 rows, auditor psql 2026-11-13; 29 anchors / 32 rows exist in total). Referee actually used = mission acceptance 'quantitative_targets 逐项容差' + pre-registered protocol in run.py:54-57 / summary.json:protocol.",
  "ref": "Martin & Moyce (1952) independent digitisation (benchmarks/dam_break/martin_moyce_1952_crosscheck.json, doi:10.1098/rsta.1952.0006); bundled table benchmarks/dam_break/martin_moyce_1952.json is graded WITHHELD",
  "got_vs_ref": "dx=0.01 m vs crosscheck over tau in [0.5,3.0]: MAE/amplitude = 4.376% (MAE 0.116836 in Z/a, amplitude 2.67), bias +0.11451 (+4.29% amplitude), max|delta| 0.16537 -> PASS at the pre-registered 15% tolerance; the 3% paper line answer is NO (4.376 > 3)",
  "ladder": "MAE/amplitude 4.666% / 1.650% / 4.376% at dx = 0.04/0.02/0.01 m; non-monotone because the bias crosses zero (-4.67% -> +1.20% -> +4.29% of amplitude); gate resolution_trend_vs_independent = fail, reported as fail and not laundered",
  "reference_pair": "crosscheck vs bundled max spread 0.85055 in Z/a (mean 0.57417) = 17.01x the claimed +-0.05 -> gate reference_self_consistency = fail; bundled number (17.271%) reported but never used as a pass source",
  "compressibility": "max rho/rho0-1 = 0.019360 (dx=0.01) vs limit 0.10 -> PASS",
  "auditor_recompute": "all 6 stored comparisons (3 dx x 2 references) rederived to 0.0 absolute difference with an independent bisection-interpolation implementation that does not import run.py:compare/interp; run.py --stage accept rebuilds results/summary.json byte-identically",
  "increment_claim_invalid": "m17020's 'error budget' does not close: 0.318558 (reference spread) + 0.845170 (Ritter deviation) + 0.019360 (compression) = 1.183088 = 27.04x the total 0.043759 it claims to decompose, and the printed residual (-27.48%) is attributed to discretization/boundary effects (F1). Ritter Z/a=2*tau reaches 5.933 at tau=2.967 against a front that saturates near 2.6 -> that deviation measures shallow-water breakdown (F2)."
 },
 "repro_cmd": "PYTHONPATH=.:src python benchmarks/repro_monaghan_1994/run.py --stage accept   # executed by the auditor on this tree: prints trajectory_vs_independent pass (0.04376<=0.15), compressibility pass (0.01936<=0.10), reference_self_consistency fail (17.01>2), resolution_trend fail, and rebuilds results/summary.json byte-identically. The increment's own entry: /root/tp-agent/work/m17020 python benchmarks/repro_monaghan_1994/error_budget.py (deterministic, output byte-identical on re-run) -- branch only.",
 "external_anchor": [
  "benchmarks/repro_monaghan_1994/results/refs_verified.json"
 ],
 "external_anchor_detail": "Monaghan (1994) J. Comput. Phys. 110:399-406 doi:10.1006/jcph.1994.1034 resolved by Crossref rule; OpenAlex oa_status=closed, no deposited abstract -> the paper's own tables/figures are WITHHELD (T4) and no transcribed value was used. Graded reference is the 1952 experiment via a digitisation shared with Lethe; its two transcriptions disagree by 17.01x their claimed uncertainty, so external validation of the anchor itself is NOT established (this is why the case is judged L4, not L5).",
 "audit_checks": {
  "tests_passed": true,
  "repro_cmd_executed": true,
  "run_artifacts_exist": true,
  "independent_recompute": true,
  "report_six_sections": true,
  "benchmark_registered_row": false,
  "increment_additive": false,
  "increment_landed_on_mainline": false,
  "reference_self_consistency": false,
  "key_quantity_error_pct": 4.376,
  "key_quantity_under_3pct": false,
  "reverse_control": "3 guards fail when m17020's artefacts/document are present; 6 passed / 3 skipped when absent"
 },
 "audit_findings": {
  "F1_BLOCKER_non_closing_budget": "components sum to 27.04x the total and the generated sentence 'residual (-27.48%) is attributable to discretization and boundary condition effects' attributes a negative error to physics; emitted by error_budget.py:155-160 into results/error_budget.json and docs/repro_monaghan_1994.md section 5b",
  "F2_BLOCKER_ritter_out_of_validity": "ritter_comparison window ends at tau=2.967 while its own docstring restricts Ritter to tau<sqrt(2)=1.4142; inside validity mean|delta| is still 57.61% of amplitude, so Ritter is not a reference for H/a=2 and 'SPH is more realistic' is unsupported",
  "F3_MAJOR_not_landed": "c670396 exists only on branch agent/m17020 (git log --all); the mainline tree has no error_budget.py / error_budget.json / error-budget test; feature_ledger REPRO-SPH-MONAGHAN1994 is still status=open with evidence NULL and mission_id NULL",
  "F4_MAJOR_tests_assert_consistency_not_truth": "the 9 new tests compare stored numbers with summary.json and thresholds like mae_Za>0; none constrains additivity or reference validity, so they stay green under F1/F2",
  "F5_INFO_bookkeeping": "tensor_benchmark_cases has no row for this anchor (same gap class as m16254/m16353 audits)",
  "F6_INFO_case_is_real": "independent rederivation to 0.0, byte-identical summary rebuild, trajectory metadata 800 particles / 2179 steps / 299.4 s / escaped=0, solver used by 4 library call sites"
 },
 "disposition": "request changes. Blockers F1 (relabel the three quantities as separate diagnostics with their own denominators, state non-additivity explicitly, delete the negative-residual sentence) and F2 (restrict Ritter to tau<=sqrt2 or mark the region as shallow-water breakdown) must be repaired before c670396 is merged; add the two invariant tests; F3 is the landing precondition. Do not merge c670396 as-is.",
 "artifacts": [
  "docs/board_audit/m17046/AUDIT_m17020.md",
  "docs/board_audit/m17046/fig1_audit_m17046.png",
  "docs/board_audit/m17046/verdict_m17046.sql",
  "docs/board_audit/m17046/independent_recompute.py",
  "docs/board_audit/m17046/independent_recompute.json",
  "docs/board_audit/m17046/make_figure.py",
  "tests/test_audit_m17046_repro_monaghan_rederivation.py"
 ],
 "verdict_text": "m17020 = L2: a real, deterministic, recomputable post-processing increment, but its headline 'error budget decomposition' is arithmetically non-closing (27x) with a physics-attributed negative residual, its Ritter comparison runs past validity, it is not on mainline, and its tests cannot catch either defect. Case-level REPRO-SPH-MONAGHAN1994 stays L4 on the strength of m16873 + m16936 (4.376% <= 15% pass, 3% line answered NO), not L5 while the anchor's own values are WITHHELD and the reference pair fails self-consistency."
}$json$::jsonb,
 'board-audit'
);