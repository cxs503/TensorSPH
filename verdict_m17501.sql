-- Verdict for m17501 XVAL-SPH-METHOD-SELECTION-2026-11-12
-- Audited by: board-audit (m17589)
-- Date: 2026-11-13
-- Level: L2 (独立功能实现)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorSPH',
    'm17501 XVAL-SPH-METHOD-SELECTION-2026-11-12',
    17589,
    2,
    '{"impl_files":["benchmarks/xval_method_selection/grade_matrix.py:350L","benchmarks/xval_method_selection/bl_route_check.py:173L","benchmarks/xval_method_selection/publish_report.py:368L","tests/test_xval_method_selection_matrix.py:122L","src/tensorsph/wave_flume.py:1208L(PRE-EXISTING)","src/tensorsph/wave_flume_grade.py:441L(PRE-EXISTING)"],"entry":"benchmarks/xval_method_selection/grade_matrix.py main()","run_artifacts":["runs/local/wf_D13_settle_d010.json(175KB,EXISTS but report says missing)","6 of 7 records MISSING"],"tests":"test_matrix_reports_the_known_height_offset_and_the_gap:PASS;test_report_artifact_matches_a_fresh_regrade:FAIL(stale report)","integration_callers":"none - standalone benchmark driver","benchmark":{"D13_run":{"worst_error":"+97.44%","tolerance":"15%","verdict":"FAIL","max_v_c0":4.479,"stability":"UNSTABLE"}},"repro_cmd":"cd /root/tp-agent/work/m17501 && PYTHONPATH=src python3 benchmarks/xval_method_selection/grade_matrix.py --run-dir runs/local","external_anchor":"Dean & Dalrymple 1991 eq.6.38; Putnam & Johnson 1949; Mei 1983 §4.2; oss_solutions id=408/355/356/358","critical_findings":["Report JSON STALE (claims all missing but 1 exists)","Single run NUMERICALLY UNSTABLE (v/c0=4.48, err=97%)","All deliverables UNTRACKED (git ??)","grade_matrix.py does not check stability criteria"]}',
    'board-audit'
);
