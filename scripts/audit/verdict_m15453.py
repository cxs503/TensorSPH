#!/usr/bin/env python3
"""Write the board-audit verdict of mission m15328 (CASE-OSC-DROP) into the platform.

Numbers are pulled from ``runs/m15453_audit/audit_summary.json`` (produced by
``scripts/audit/audit_m15453_osc_drop.py``) so that nothing is retyped by hand.

    python scripts/audit/verdict_m15453.py            # dry run, prints the row
    python scripts/audit/verdict_m15453.py --write    # INSERT ... ON CONFLICT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "runs/m15453_audit/audit_summary.json"

REPO_SLUG = "TensorSPH"
SCOPE = "m15328 CASE-OSC-DROP"
MISSION = 15453
LEVEL = 3
VERDICT_BY = "board-audit"


def build_evidence(s: dict) -> dict:
    lamb = s["lamb_table"]
    worst = max(lamb, key=lambda r: abs(r["err_2d_pct"]))
    gates = s["gates_recomputed"]
    repro = s["repro"]
    delta = s["delta_measure"]
    return {
        "anchor_slug": "CASE-OSC-DROP",
        "audited_commit": s["audited_commit"],
        "reproduced_on_commit": s["reproduced_commit"],
        # NOTE: the maturity gate treats every list entry as a *path* (exists-and-non-dir),
        # so paths stay clean here; the ":line" detail lives in "impl_lines"/"entry_cmd".
        "impl_files": [
            "src/tensorsph/surface_tension_dispersion.py",
            "src/tensorsph/csf_consistent.py",
            "src/tensorsph/csf_curvature.py",
            "src/tensorsph/surface_tension.py",
        ],
        "impl_lines": [
            "src/tensorsph/surface_tension_dispersion.py:1-966 —— hex_lattice:139 / drop_geometry:205（2-D 极坐标 R(1+ε cos nθ)）"
            " / csf_state:254 / drop_modal_stiffness:341 / lamb_case:454 / curvature_audit:563 / delta_measure_audit:712"
            " / capillary_dt_limit:755 / run_validation:790 / main:921；门常数 TOL_WAVE_PCT=5.0 / TOL_LAMB_PCT=3.0 /"
            " MIN_ORDER=2.0 在 :129-133；ref 公式在 :484（ω²=n(n²−1)σ/(ρR³)）",
            "src/tensorsph/csf_consistent.py:1-539 —— mls_gradient:123 / mls_divergence:147 / csf_fields:262"
            " / surface_tension_acceleration:383（被基准测的算子本体）",
            "src/tensorsph/csf_curvature.py:78 fit_curvature（fit_quadratic / fit_circle / mls 三方案）",
            "src/tensorsph/surface_tension.py:174 compute_surface_tension_force / :231 surface_tension_acceleration"
            " / :292-315 capillary_dt_limit（求解器侧封装）",
        ],
        "entry": "scripts/run_surface_tension_dispersion.py",
        "entry_cmd": "PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --modes 2,3,4 --rdx 16,24,32 "
                     "--out-dir runs/surface_tension_dispersion（等价模块入口 python -m tensorsph.surface_tension_dispersion，"
                     "--help 已实测）；审计人重跑用 --out-dir runs/m15453_audit",
        "run_artifacts": [
            "runs/m15453_audit/report.json",
            "runs/m15453_audit/benchmark.json",
            "runs/m15453_audit/meta.json",
            "runs/m15453_audit/summary.md",
            "runs/m15453_audit/figures/capillary_wave_error.png",
            "runs/m15453_audit/figures/lamb_stiffness.png",
            "runs/m15453_audit/figures/curvature_audit.png",
            "runs/m15453_audit/figures/delta_measure.png",
            "runs/m15453_audit/audit_summary.json",
            "runs/m15453_audit/_source_m15328/report.json",
            "runs/m15453_audit/_source_m15328/benchmark.json",
            "runs/m15453_audit/_source_m15328/meta.json",
            "runs/m15453_audit/_source_m15328/summary.md",
            "docs/board_audit/M15453_CASE_OSC_DROP_BOARD_AUDIT.md",
            "docs/board_audit/m15453/fig1_omega_vs_mode.png",
            "docs/board_audit/m15453/fig2_error_vs_resolution.png",
            "docs/board_audit/m15453/fig3_diagnostics.png",
        ],
        "run_artifacts_note": "被审交付件：runs/surface_tension_dispersion/{report.json,benchmark.json,meta.json,summary.md,figures/*4}"
                              " 存在于 commit b5280f37b50eab41096511e15fcdc2bcc1ff7d27（agent/m15328），哈希与 meta 自校验相符"
                              "（report sha256 b4dedf4f…a0a7e0 / benchmark 3e533e13…213e0c）；该 commit 不是审计基线 25fc915 的祖先，"
                              "故 trunk 上缺失（findings.F2），快照复制在上列 _source_m15328/ 内。"
                              "审计人独立重跑件（runs/m15453_audit/）12/12 行 ω 与四门与交付逐位相同",
        "tests": [
            "tests/test_surface_tension_dispersion.py（含 test_committed_campaign_artifacts_satisfy_the_reported_invariants:237-262，"
            "断言工件存在 + status=='FAIL' + 三门失败 + option_screen 无通过组合）",
            "tests/test_csf_dispersion.py（算子级色散/δ_s 守卫，见 :1-30 文档串）",
            "tests/test_oscillating_droplet_lamb.py（解析分支 1e-12 :51-79；2-D/3-D 比 sqrt(8/6) :66；阻尼拟合 :89；"
            "噪声负对照 :98；RECORDED_MODE2_ERR_PCT[loongarch64]=-61.69 且断言 |err|>15 :180-190 —— 测试钉住的是已知失败）",
            "审计人实跑：pytest tests/test_oscillating_droplet_lamb.py tests/test_surface_tension_dispersion.py -q "
            "→ 1 failed（test_committed_campaign_artifacts…，因 trunk 缺 runs/ 工件）, 30 passed, 51.68 s",
        ],
        "integration_callers": [
            "src/tensorsph/surface_tension.py:174 compute_surface_tension_force → :231 surface_tension_acceleration"
            "（被基准测的同一 CSF 算子）",
            "src/tensorsph/droplet_impact.py:929-934 主求解器每步调用 compute_surface_tension_force（cfg.surface_tension_coefficient>0 时）",
            "src/tensorsph/threed/csf3.py:56 fit_curvature + src/tensorsph/threed/liquid_column.py:720 "
            "surface_tension_acceleration3（3-D 路径）",
            "src/tensorsph/threed/solver.py:40,65-81 acoustic_dt 用 capillary_dt_limit 作第二稳定限；"
            "src/tensorsph/surface_tension.py:292-315 capillary_dt_limit(cfg)",
        ],
        "benchmark": {
            "registered_case": "none —— SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                               "WHERE anchor_slug='CASE-OSC-DROP' → 0 行（全表 16 锚点，无 SPH 锚点）："
                               "3 % 容差为被审单自定（surface_tension_dispersion.py:131）",
            "reference": "2-D（柱状）Rayleigh–Lamb ω²=n(n²−1)σ/(ρR³)，σ=0.072 N/m，ρ0=1000，R=0.02 m，ε=0.10；"
                         "审计人独立重推：对 r=R(1+ε cos nθ) 数值取精确曲率投影得前因子 3.008/7.867/14.396 vs 解析 3/8/15"
                         "（+0.27/−1.67/−4.03 %，O(ε²) 修正），与 n(n²−1) 支自洽；3-D Rayleigh 支 n(n−1)(n+2) 为另一支",
            "got": {f"n={r['mode']} R/dx={r['radius_over_dx']}": {
                "omega_measured": round(r["omega_measured"], 6),
                "err_2d_pct": round(r["err_2d_pct"], 4),
                "err_3d_pct": round(r["err_3d_pct"], 4)} for r in lamb},
            "worst_lamb": {"mode": worst["mode"], "radius_over_dx": worst["radius_over_dx"],
                           "err_2d_pct": round(worst["err_2d_pct"], 4), "tolerance_pct": 3.0},
            "gates_recomputed_by_auditor": gates,
            "gates_committed": s["gates_committed"],
            "gates_agree_with_committed": s["gates_agree_with_committed"],
            "gates_disagreement_note": "C1 提交 198.421 % 只取最细网格行；审计按全 ladder 最坏行 338.815 %（两值都 FAIL）",
            "convergence_orders": s["convergence_order_committed"],
            "convergence_required": s["convergence_required"],
            "option_screen": s["option_screen"],
            "delta_measure": {"lamb_surface_measure_ratio": [delta["lamb_surface_measure_ratio_min"],
                                                             delta["lamb_surface_measure_ratio_max"]],
                              "target": 1.0, "deficit_pct": round(delta["deficit_pct"], 2)},
            "degenerate_gate_C4": s["degenerate_gate_c4"]["why"],
            "status": s["status_committed"],
            "conclusion": "FAIL 保持：n=2 −4.85 %、n=3 −9.48 %、n=4 −19.32 % 均 > 3 % 容差；"
                          "收敛阶 −4.143 < 2.0；换 3-D 支更差（−17.6/−19.0/−26.4 %）→ L4 不成立",
        },
        "repro_cmd": "PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --modes 2,3,4 "
                     "--rdx 16,24,32 --out-dir runs/m15453_audit  # 审计人实跑 61.7 s，12/12 行 ω 误差与交付逐位相同"
                     "（max abs diff 0.0 个百分点），四门逐位相同",
        "repro_verdict": {
            "bitwise_identical": repro["bitwise_identical"],
            "rows_common": repro["rows_common"],
            "max_abs_diff_pct_points": repro["max_abs_diff_pct_points"],
            "recorded_meta_command_executable_as_written": False,
            "why": "meta.json.command 缺 PYTHONPATH=src → 两处检出均 ModuleNotFoundError（环境内有 m14539 stale "
                   "editable 安装抢包）；report.json.repro_cmd 是另一条 python -m 形式 → 两条串不自洽（findings.F3）",
        },
        "external_anchor": "Rayleigh (1879) / Lamb, Hydrodynamics Art. 275 的 3-D 球滴 ω²=n(n−1)(n+2)σ/(ρR³)；"
                           "本算例为 2-D 柱状类比 ω²=n(n²−1)σ/(ρR³)（审计人独立重推几何曲率 + Bernoulli 得同式）。"
                           "有外部真值，但未在 tensor_benchmark_cases 登记（findings.F1）",
        "level": LEVEL,
        "level_rationale": "L2 满足：966 行真实现（格点几何→CSF 曲率/δ_s→模态投影→ω 数值），输入→计算→输出闭合。"
                           "L3 满足：同一 CSF 算子经 surface_tension.py:174/231 进入主求解器 droplet_impact.py:929-934 与 "
                           "threed/liquid_column.py:720，capillary_dt_limit 被 threed/solver.py:81 采用。"
                           "L4 不成立：基准误差 9.5–19.3 % ≫ 3 %、收敛阶 −4.14 < 2、裁判未登记、ε=0.10 无极限外推。"
                           "L5 不成立。百页报告不改变级别；级别=功能实质（集成成立、验证未成立）。",
        "findings": {
            "F2_high_not_on_trunk": "b5280f3 非 25fc915 祖先（merge-base --is-ancestor=no），trunk 无 runs/surface_tension_dispersion/ → "
                                    "tests/test_surface_tension_dispersion.py::test_committed_campaign_artifacts… 在 trunk 红（实跑 1 failed/30 passed）",
            "F1_medium_referee_absent": "tensor_benchmark_cases 无 CASE-OSC-DROP 行（0/16 含 SPH 锚点）；3 % 容差自定 → 门无法机器复核",
            "F3_medium_repro_cmd": "meta.json 命令原样不可执行（缺 PYTHONPATH=src，ModuleNotFoundError 两处复现）；report.json 另记一条 python -m 串",
            "F4_low_finest_grid_gate": "surface_tension_dispersion.py:867-874 门值只取最细网格：C1=198.42 % 而最坏行 338.81 %（都 FAIL）",
            "F5_low_tolerance_mismatch": "报告写「容差 <5 %（任务令）」，代码门 TOL_LAMB_PCT=3.0；两阈值下 n≥3 均 FAIL",
            "F6_low_vacuous_gate": "C4 σ 敏感性 max_spread_pct=0.0 为代数恒等式（牵引力∝σ、ω=sqrt(ratio)·ω_theory ⇒ ω/√σ 不变）",
            "F7_low_citation": "2-D 支挂 3-D 引文（Rayleigh 1879 / Lamb Art. 275）；若按 3-D 支，误差 −17.6/−19.0/−26.4 %",
            "F8_low_finite_amplitude": "ε=0.10 固定、无 ε→0 外推；同谐波 O(ε²) 修正 0.27/1.67/4.03 %（n=2/3/4）已达 3 % 门量级",
            "F9_info_diagnostics": "δ_s 积分 0.702–0.734（缺 28 %）、local_range 波上过冲 1.651、ppw=32 曲率符号未恢复"
                                   "（−0.044 vs +0.126 1/m，平界面噪声 RMS 0.405，SNR 0.31）、4 种开关组合最好 lamb 4.85 %/wave 93.21 % 全不通过 → FAIL 不可调参掩盖",
            "F10_info_green_tests_not_evidence": "test_oscillating_droplet_lamb.py:180-190 把 mode-2 误差钉在 −61.69 % 并断言 |err|>15 %："
                                                 "绿测试断言的是已知不达标；另一条动力学路线 benchmarks/oscillating_droplet/ 同样未达标",
        },
        "disposition": "approve with findings —— L3 confirmed, L4 NOT_REACHED（honest FAIL 保持，账不消）；F2/F1 需返修后复验",
        "checks_not_run": [
            "未审计 benchmarks/oscillating_droplet/ 动力学路线的全量 ladder（只核 RECORDED_MODE2_ERR_PCT 与测试断言）",
            "未跑 3-D 球滴（threed/csf3.py）频率算例；3-D 支仅做解析对照",
            "未做外部求解器（OpenFOAM/Basilisk）交叉验证（本机无）",
            "未核 GPU 路径与 ppw>48 收敛",
            "未判定「2-D 类比是否满足 BP 锚点意图」（领域口径问题，属锚点 owner）",
        ],
        "audit_artifacts": [
            "docs/board_audit/M15453_CASE_OSC_DROP_BOARD_AUDIT.md",
            "scripts/audit/audit_m15453_osc_drop.py",
            "runs/m15453_audit/audit_summary.json",
            "runs/m15453_audit/_source_m15328/（被审交付件快照）",
            "docs/board_audit/m15453/fig1_omega_vs_mode.png",
            "docs/board_audit/m15453/fig2_error_vs_resolution.png",
            "docs/board_audit/m15453/fig3_diagnostics.png",
        ],
        "reproduces_on_trunk": False,
    }


def main(argv: list[str]) -> int:
    s = json.loads(SUMMARY.read_text())
    ev = build_evidence(s)
    write = "--write" in argv
    if not write:
        print(json.dumps({"repo": REPO_SLUG, "scope": SCOPE, "mission_id": MISSION,
                          "level": LEVEL, "verdict_by": VERDICT_BY,
                          "evidence_keys": sorted(ev)}, indent=1))
        print("(dry run; pass --write to insert)")
        return 0
    # no psycopg2 in this image -> hand the statement to psql
    import subprocess, tempfile
    payload = json.dumps(ev, ensure_ascii=False).replace("'", "''")
    stmt = f"""INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('{REPO_SLUG}', '{SCOPE}', {MISSION}, {LEVEL}, '{payload}'::jsonb, '{VERDICT_BY}')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
      evidence = EXCLUDED.evidence, gate_state = 'pending';
SELECT id, repo, scope, mission_id, level, gate_state, verdict_by FROM tensor_maturity_verdicts
 WHERE repo = '{REPO_SLUG}' AND scope = '{SCOPE}' AND verdict_by = '{VERDICT_BY}';"""
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as fh:
        fh.write(stmt)
        path = fh.name
    subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-f", path], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))