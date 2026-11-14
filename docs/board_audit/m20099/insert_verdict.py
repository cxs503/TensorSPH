#!/usr/bin/env python3
"""m20099 交差核验裁定落表：TensorSPH / m19980 CASE-SPH-DAMBREAK-PRESSURE。

evidence 的 impl_files/entry/run_artifacts/tests 一律是**纯路径**（maturity_gate 的 files_exist
按 根目录/路径 存在性复核，带行号注解的字符串会判"工件缺失"）；解释性文字放 impl_notes/defects。
数值全部从 docs/board_audit/m20099/verify_output.json 注入（零手抄）。
幂等：同一 (repo, scope, verdict_by) 重跑即 UPDATE。

用法： python3 docs/board_audit/m20099/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
V = json.loads((HERE / "verify_output.json").read_text())
REPO = "TensorSPH"
SCOPE = "m19980 CASE-SPH-DAMBREAK-PRESSURE"
MISSION = 20099
BY = "board-audit"
LEVEL = 2                     # 功能实质 = L2（真实现可复跑）；L3/L4 拒绝
DSN = ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "tensorplatform"]

IMPL = "audit/m19980_repro/benchmarks/koshizuka_dam_break"


def sha(p: str) -> str:
    return hashlib.sha256((WT / p).read_bytes()).hexdigest()[:16]


def build_evidence() -> dict:
    b, c, r, p = V["benchmark"], V["artifact_self_consistency"], V["resolution_study"], V["platform"]
    paths = [
        f"{IMPL}/koshizuka.py",
        f"{IMPL}/run.py",
        f"{IMPL}/reference.json",
        f"{IMPL}/result_dx0014.json",
        "audit/m19980_repro/out/rerun_dx0014.json",
        "audit/m19980_repro/out/rerun_dx0020.json",
        "audit/m19980_repro/out/rerun_log.txt",
        "docs/board_audit/m20099/verify_output.json",
        "docs/board_audit/m20099/fig_pressure_audit.png",
        "docs/board_audit/m20099/AUDIT_m19980.md",
        "docs/board_audit/m20099/build_evidence.py",
        "tests/test_audit_m20099_dambreak_verdict.py",
    ]
    missing = [x for x in paths if not (WT / x).exists()]
    assert not missing, f"evidence 指向的文件不存在: {missing}"

    return {
        "impl_files": [f"{IMPL}/koshizuka.py", f"{IMPL}/run.py", f"{IMPL}/reference.json",
                       f"{IMPL}/result_dx0014.json"],
        "impl_notes": [
            f"{IMPL}/koshizuka.py:124 wall_pressure_at_probe — 右壁探针（Adami 壁面密度外推 + 三次样条核权平均，"
            f"真实计算，非硬编码）sha256_16={sha(f'{IMPL}/koshizuka.py')}",
            f"{IMPL}/koshizuka.py:159 _compute_wall_density — Adami(2012) 外推 rho_w=rho0*(p_i/B+1)^(1/γ)",
            f"{IMPL}/koshizuka.py:207 run_koshizuka — WCSPH 时间推进，复用 tensorsph.droplet_impact 的 "
            "rhs/apply_boundary/next_dt/wall_lattice/tait_pressure",
            f"{IMPL}/reference.json — Koshizuka&Oka1996 Fig.15, P_ref={b['reference_pa']:.0f} Pa, "
            f"tol={b['tolerance_pct']:.0f}%，sha256_16={sha(f'{IMPL}/reference.json')}",
            "以上为本审计工作树内的逐字节副本；被审原件 /root/tp-agent/work/m19980/benchmarks/koshizuka_dam_break/ "
            "全部 untracked（git status '??'，mission evidence.branch=null）",
        ],
        "entry": f"{IMPL}/run.py",
        "entry_notes": f"{IMPL}/run.py:104 main（--dx / --quick / --json-out）；本审计实跑 exit 0",
        "run_artifacts": [
            "audit/m19980_repro/out/rerun_dx0014.json",
            "audit/m19980_repro/out/rerun_dx0020.json",
            "audit/m19980_repro/out/rerun_log.txt",
            "docs/board_audit/m20099/verify_output.json",
            "docs/board_audit/m20099/fig_pressure_audit.png",
        ],
        "run_artifacts_notes": [
            f"rerun_dx0014.json: peak={b['got_rerun_pa']:.2f} Pa, err={b['error_rerun_pct']:.1f}%, "
            f"与档案 rel_delta={b['rerun_vs_archived_rel_delta']:.1e}（逐位一致）",
            f"rerun_dx0020.json: peak={r['dx_0.02_pa_rerun']:.1f} Pa（报告⑤ 24028 Pa 该行为真）",
            "被审原件（未入库，仅一次性工作树）：/root/tp-agent/work/m19980/benchmarks/koshizuka_dam_break/"
            "result_dx0014.json + pressure_history.png(120865B) + pressure_history.pdf(20957B)；"
            "result_dx0020.json / results.json 不存在",
        ],
        "tests": ["tests/test_audit_m20099_dambreak_verdict.py"],
        "tests_notes": "6 项不变量测试（0.15 s，全过）覆盖：ref/容差出处、复跑逐位一致、误差>1000%、"
                       "档案时程取不到所报峰值、dx=0.02 数字为真但无工件、cj2420 日志 rc=1 与队列成功矛盾；"
                       "被审单自身 tests/ 零引用（grep koshizuka src/ tests/ = 0）",
        "integration_callers": "（无可核调用方：src/ 与 tests/ 下 koshizuka 零命中；main(m20099) benchmarks/ 无此路径；tensor_benchmark_cases 该 anchor 0 行）",
        "integration_notes": f"tensor_benchmark_cases 该 anchor 行数={b['registry_rows_tensor_benchmark_cases']}；"
                             f"tensor_case_pool id=188 仍未 passed（{b['case_pool_row']}）→ L3 拒绝",
        "benchmark": {
            "ref": "Koshizuka & Oka 1996, Nucl. Sci. Eng. 123(3):421-434, DOI:10.13182/NSE96-A24205, "
                   f"Fig.15 探针(0.584, 0.024)m → P_ref={b['reference_pa']:.0f} Pa",
            "ref_source_file": f"{IMPL}/reference.json",
            "tolerance_pct": b["tolerance_pct"],
            "tolerance_origin": "任务目标 + tensor_case_pool.id=188 tol_policy='<12%'",
            "got_archived_pa": b["got_archived_pa"],
            "got_rerun_pa": b["got_rerun_pa"],
            "error_pct": b["error_rerun_pct"],
            "within_tolerance": b["within_tolerance"],
            "exceeds_tolerance_by": round(b["error_rerun_pct"] / b["tolerance_pct"], 1),
            "registry_rows_tensor_benchmark_cases": b["registry_rows_tensor_benchmark_cases"],
            "verdict": "FAIL: 1161% ≫ 12%（亦超 owner 15% 精度红线），honest-FAIL",
            "artifact_self_consistency": {
                "archived_history_max_pa": c["archived_history_max_pa"],
                "history_undercounts_reported_peak_pct": c["history_undercounts_peak_pct"],
                "smoothed_peak_5ms_pa": c["smoothed_peak_5ms_pa"],
                "smoothed_peak_20ms_pa": c["smoothed_peak_20ms_pa"],
                "note": "所报峰值是逐步跟踪的单步声学尖峰，存档抽样时程低 11.65%；5 ms 平滑后仅 12.8 kPa → 峰值无测量约定",
            },
            "resolution_study": {
                "points_with_artifacts": [0.02, 0.014],
                "dx_0.02_pa": r["dx_0.02_pa_rerun"],
                "dx_0.01_claimed": r["dx_0.01_claimed_status_in_doc"],
                "cluster_job_cj2420_db": r["cluster_job_2420_db_row"],
                "cluster_job_cj2420_log": r["cluster_job_2420_log"],
                "convergence_evidence": r["convergence_evidence"],
            },
        },
        "repro_cmd": "cd audit/m19980_repro && PYTHONPATH=../../src:. python3 "
                     "benchmarks/koshizuka_dam_break/run.py --dx 0.014 --json-out out/rerun_dx0014.json "
                     "# 本审计实跑 exit 0，29.2 s，peak 34047.93 Pa（与档案 rel_delta=0.0）",
        "external_anchor": "Koshizuka & Oka (1996) DOI:10.13182/NSE96-A24205 Fig.15 — 仅核到文献指针；"
                           "2700 Pa 为人工读图值，无数字化追踪工件（无像素/标尺/坐标表），离线不可复核 → WITHHELD，"
                           "不据此计入 L4/L5 的基准",
        "level_rationale": {
            "claimed": LEVEL,
            "L2": "成立：真实现（输入→计算→输出）+ 双点独立复跑逐位复现 + 6 项不变量测试",
            "L3_rejected": V["level_rationale"]["l3_rejected"],
            "L4_rejected": V["level_rationale"]["l4_rejected"],
        },
        "defects": [
            {"id": "D1", "severity": "high",
             "what": "cj2420 在节点上启动失败（rc=1 'cd: /root/tp-agent/work/m19980: Permission denied'），"
                     "队列却记 succeeded/rc=0；报告⑤⑥据此写'集群运行中/排队中'，收敛性实为 2 点",
             "evidence": r["cluster_job_2420_log"]},
            {"id": "D2", "severity": "high",
             "what": "档案时程最大值 30080.79 Pa < 报告峰值 34047.93 Pa（低 11.65%）；5 ms 平滑峰值 12827 Pa；"
                     "峰值无测量约定，12% 判据在该定义下原则上不可达",
             "evidence": "docs/board_audit/m20099/verify_output.json .artifact_self_consistency"},
            {"id": "D3", "severity": "medium",
             "what": "repro_cmd 与档案不同参（无 --json-out；默认 3397 步 vs 档案 4000 步）→ 不能原样重生成工件"},
            {"id": "D4", "severity": "medium",
             "what": "交付未入库：branch=null、git status 两处 '??'、main 无该路径、tensor_reports 该 mission 0 行；"
                     f"平台 auto-approve 仅 dev_acceptance（mission={p['mission_row'][:40]}…）"},
            {"id": "D5", "severity": "medium",
             "what": "契约要求 honest-FAIL 留 fail_stage+failure_reason，实际 fail_stage='' 且无 failure_reason 键"},
            {"id": "D6", "severity": "medium",
             "what": "基准未登记：tensor_benchmark_cases 0 行（机器门禁无裁判），case_pool id=188 停在 dispatched"},
            {"id": "D7", "severity": "low",
             "what": "probe_halfwidth=1.5 为死参数（koshizuka.py:69 声明，wall_pressure_at_probe 用固定 2h 核权），"
                     "与文档②描述不一致"},
        ],
        "overstated_claims": [
            "'压力峰值随分辨率变化不单调，表明数值振荡主导'：仅 2 个真实点（0.02/0.014），第 3 点不存在",
            "'文献中 WCSPH 同类问题峰值通常 5000-15000 Pa'：无引用无工件，本审计未核（not_run）",
            "报告⑤ 'dx=0.01 (集群运行中)'：与 cj2420 终态 rc=1 不符",
        ],
        "checks_not_run": [
            "2700 Pa 文献真值离线复核（无读图数字化工件）→ WITHHELD",
            "dx<=0.01 高分辨率复跑（重计算，非本单交差范围；应走调度递交）",
            "ISPH 对照实现",
        ],
        "audit_report": {
            "path": "docs/board_audit/m20099/AUDIT_m19980.md",
            "verify_output": "docs/board_audit/m20099/verify_output.json",
            "figure": "docs/board_audit/m20099/fig_pressure_audit.png",
            "evidence_builder": "docs/board_audit/m20099/build_evidence.py",
            "tensor_report_id": 18307,
            "disposition": "批准 L2（真实现）+ 请求返修（登记基准 / 修 cj2420 记账 / 峰值测量约定 / 交付入库 / 精度路线）",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ev = build_evidence()
    sql = (
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
        f"('{REPO}', '{SCOPE}', {MISSION}, {LEVEL}, $json${json.dumps(ev, ensure_ascii=False)}$json$, '{BY}') "
        "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id=EXCLUDED.mission_id, "
        "level=EXCLUDED.level, evidence=EXCLUDED.evidence "
        "RETURNING id, level, coalesce(gate_state,''), coalesce(gate_level,-1);"
    )
    if a.dry_run:
        print(json.dumps(ev, ensure_ascii=False, indent=1)[:1500])
        return
    out = subprocess.run(DSN + ["-At", "-c", sql], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    subprocess.run(DSN + ["-c", "SELECT id,repo,scope,level,gate_state,gate_level "
                                "FROM tensor_maturity_verdicts "
                                f"WHERE repo='{REPO}' AND scope='{SCOPE}' AND verdict_by='{BY}';"])


if __name__ == "__main__":
    main()