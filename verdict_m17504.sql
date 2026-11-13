-- Board audit verdict for m17504 XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION
-- Auditor: board-audit (m17629), 2026-11-13
-- Level: L4 (基准验证)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorSPH',
    'm17504 XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION',
    17629,
    4,
    '{"impl_files":["src/tensorsph/sphinxys_css.py (405 lines)"],"entry":"benchmarks/surface_tension/sphinxys_xval.py (277 lines)","run_artifacts":"benchmarks/surface_tension/out_sphinxys_xval/xval_ladder.json + xval_ladder.png","tests":"tests/test_sphinxys_css.py (7 passed)","integration_callers":["tests/test_sphinxys_css.py","scripts/post_report_m17504.py"],"benchmark":{"stress_reading":"pass (max |err|=3.95%, -1.65% at R/dx=64, converging)","flux_reading":"fail (max |err|=99.70%, honest FAIL)","delta_integral":"0.9835 at R/dx=64 (1000:1), 0.4894 at 1:1","gate_criterion":"< 15% vs sigma/R","resolution_ladder":[16,32,64],"sensitivity_cases":5},"repro_cmd":"cd /root/tp-agent/work/m17504 && PYTHONPATH=src python3 benchmarks/surface_tension/sphinxys_xval.py","external_anchor":"Xiangyu-Hu/SPHinXsys master, 20 files archived with sha256","level_reason":"L4: complete benchmark with resolution ladder, error analysis vs analytical sigma/R, reproducible tests, honest FAIL documented. Not L5 because flux reading fails and model not integrated into production solver path.","notes":["Stress reading passes gate (<15%) and shows first-order convergence","Flux reading is honest FAIL: symmetric (S_i+S_j) flux cancels curvature force","Upstream SPHinXsys does not publish Laplace numerical values; analytical sigma/R used as reference","Report published as tensor report #16471 with figure","Raw upstream sources archived with sha256 for zero-hand-copy verification"]}',
    'board-audit'
);
