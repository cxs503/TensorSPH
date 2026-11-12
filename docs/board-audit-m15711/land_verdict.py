#!/usr/bin/env python3
"""m15711 交差核验：把 CASE-COUETTE-SPH / m15412 的裁定写入 tensor_maturity_verdicts。

声明级别 = 3（系统集成）；L4 基准验证拒绝（预声明门 r16 steady_drift 未过 + 锚点未登记）。
evidence 只放真实指针（文件/行号/工件路径/测试名/重跑结论），由 maturity_gate 机器复核。
幂等：同 (repo, scope, verdict_by) 覆盖，并重置 gate_state 待裁。
"""
import json
import subprocess

REPO = "TensorSPH"
SCOPE = "m15412 CASE-COUETTE-SPH"
MISSION = 15711
LEVEL = 3
BY = "board-audit"

EVIDENCE = {
    "impl_files": [
        "benchmarks/couette/run.py",        # 算例实现（真调生产求解器）
        "benchmarks/couette/make_doc.py",   # 报告程序化生成器
        "tests/test_couette.py",            # 9 个测试（解析恒等/负控/真跑）
        "benchmarks/couette/DESIGN.md",     # 预声明判据（先于运行，随交付入库）
    ],
    "entry": "benchmarks/couette/run.py",
    "entry_invocation": "python benchmarks/couette/run.py --out <dir> --resolutions 10 16 24 "
                        "--speeds 0.4 --viscosities 0.04 --frictions 0.0（实测 4m20s 跑完 6 个 case，"
                        "确定性复现交付工件）",
    "run_artifacts": [
        "runs/couette_repro/results.json",                 # m15711 独立重跑，与交付 0 差异字段
        "runs/couette_repro/couette_profiles.png",
        "runs/couette_repro/couette_field.png",
        "runs/couette_repro/couette_convergence.png",
        "runs/couette_repro/couette_friction_sweep.png",
        "docs/board-audit-m15711/check_results.json",       # 三件套机器复核结果
        "docs/board-audit-m15711/FINDINGS.md",              # 审计报告（含未做项）
        "docs/board-audit-m15711/fig_couette_audit.png",
    ],
    "tests": [
        "tests/test_couette.py::test_analytic_profile_is_the_couette_line",
        "tests/test_couette.py::test_analytic_scalars_match_their_independent_formulas",
        "tests/test_couette.py::test_analysis_is_exact_on_the_analytic_profile",
        "tests/test_couette.py::test_flux_is_invariant_under_a_symmetric_wall_slip",
        "tests/test_couette.py::test_half_the_analytic_slope_is_reported_as_a_fifty_percent_error",
        "tests/test_couette.py::test_sliding_wall_carries_u_on_top_and_zero_on_bottom",
        "tests/test_couette.py::test_sliding_wall_drives_the_fluid",
        "tests/test_couette.py::test_free_slip_wall_does_not_drive_the_fluid",
        "tests/test_couette.py::test_no_slip_run_is_two_dimensional_and_steady_in_the_mean",
        "run: PYTHONPATH=src pytest tests/test_couette.py -q -> 9 passed in 120.13s (audit m15711)",
    ],
    "integration_callers": [
        "from benchmarks.couette.run import",   # tests/test_couette.py 消费算例分析面；CI tests.yml 跑 pytest -q
        "tensorsph.droplet_impact.rhs:benchmarks/couette/run.py:222 "
        "(rhs(..., wall=wall, wall_vel=wall_vel) —— 驱动生产求解器移动壁/no-slip(wall_friction) 通路)",
    ],
    "benchmark": {
        "registry": "tensor_benchmark_cases WHERE anchor_slug='CASE-COUETTE-SPH' -> 0 rows "
                    "(表内总行 22) => 平台基准裁判无行可裁，判据回落到仓内预声明 DESIGN.md",
        "predeclared_thresholds": {"interior_l2": 0.05, "flux_rel": 0.05,
                                   "wall_shear_rel": 0.05, "steady_drift": 0.01,
                                   "control_flux": 0.2},
        "pass_rule": "H/dx=16 与 24 同时全过四项门 + 负控 + 分辨率响应",
        "ref": "u(y)=U y/H, du/dy=U/H, tau=rho nu U/H, Q=U H/2 (Batchelor 1967 §4.2)",
        "got_ladder_r16": {"H_over_dx": 16, "Re": 1.0, "Ma": 0.01,
                           "interior_l2": 0.029328, "flux_ratio": 1.044155,
                           "wall_shear_ratio_bottom": 0.995219, "steady_drift": 0.015471},
        "got_ladder_r24": {"H_over_dx": 24, "interior_l2": 0.029480, "flux_ratio": 1.043700,
                           "wall_shear_ratio_bottom": 1.016000, "steady_drift": 0.008150},
        "error_vs_15pct_standard": "剖面 L2 2.93%、通量 4.42%、壁剪切 0.48% —— 均 ≪15%；"
                                   "失分项是自设稳态漂移门(1.55%≥1%)，非精度门",
        "case_verdict": "FAIL (ladder-r16: gate 'steady_drift' missed)",
        "reproduction": "交付 results.json 与 m15711 独立重跑规范化后 diff 行数=0（确定性）；"
                        "make_doc.py 由重跑工件再生 md 与交付 md 逐字节相同（5491 B）=零手抄",
        "negative_control": {"flux_ratio": 0.136343, "gate": 0.2, "pass": True},
        "self_report_vs_audit": "原单 headline/result/报告三处均自报 FAIL，与审计一致 —— 无虚标",
    },
    "repro_cmd": "PYTHONPATH=src python benchmarks/couette/run.py --out runs/couette_repro "
                 "--resolutions 10 16 24 --speeds 0.4 --viscosities 0.04 --frictions 0.0 && "
                 "PYTHONPATH=src python benchmarks/couette/make_doc.py "
                 "--results runs/couette_repro/results.json --md /tmp/regen.md && "
                 "PYTHONPATH=src python -m pytest tests/test_couette.py -q && "
                 "python3 docs/board-audit-m15711/verify_audit.py",
    "external_anchor": "Batchelor (1967) An Introduction to Fluid Dynamics, CUP §4.2 "
                       "(steady unidirectional flow between parallel plates, one moving) —— "
                       "闭式无自由参数，同解亦为 TensorLBM/benchmarks/ice_lid_navier_slip.py 的 Ls->0 极限；"
                       "注：本项为文献字符串非文件指针，故 L5 不予声明",
    "gaps_preventing_L4": [
        "F2 预声明门 r16 steady_drift=1.55% ≥ 1% -> 定常能力仅在 H/dx=24 成立",
        "F1 锚点 CASE-COUETTE-SPH 未登记 tensor_benchmark_cases（算例池供给未闭环）",
        "F3 交付 run 工件 runs/ 未入库（复核靠重跑补证）",
        "F4 壁剪切比 M3 由首两行有限差分导出，与 M1 同源，不构成独立证据",
    ],
}
SCOPE_SQL = SCOPE.replace("'", "''")
ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    f"VALUES ('{REPO}', '{SCOPE_SQL}', {MISSION}, {LEVEL}, '{ev}'::jsonb, '{BY}') "
    f"ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level={LEVEL}, mission_id={MISSION}, "
    f"evidence='{ev}'::jsonb, gate_state='pending', gate_level=NULL, gate_note=NULL"
)
r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-c", sql],
                   capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-500:])
r2 = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA",
                     "-F", "|", "-c",
                     "SELECT id, repo, scope, mission_id, level, verdict_by, "
                     "coalesce(gate_state,'pending'), (SELECT count(*) FROM "
                     f"jsonb_object_keys(evidence)) FROM tensor_maturity_verdicts "
                     f"WHERE scope='{SCOPE_SQL}' AND verdict_by='{BY}'"],
                    capture_output=True, text=True)
print("ROW:", r2.stdout.strip(), r2.stderr[-300:])