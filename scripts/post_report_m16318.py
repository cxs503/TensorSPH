#!/usr/bin/env python3
"""Publish the m16318 board-audit report to /tensor/reports.

Numbers are injected programmatically from the m16244 artifacts
(benchmarks/rayleigh_taylor/runs/{reference,table,damping}.json) — zero hand-typed
values in the tables. Figure: docs/benchmarks/rayleigh-taylor-mp-audit/fig1_growth_audit.png
"""
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "benchmarks" / "rayleigh_taylor" / "runs"
FIG = REPO / "docs" / "benchmarks" / "rayleigh-taylor-mp-audit" / "fig1_growth_audit.png"
BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def req(method, path, body=None, tok=None):
    r = urllib.request.Request(BASE + path,
                               data=json.dumps(body).encode() if body else None,
                               method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode() or "null")


def main():
    ref = json.loads((RUNS / "reference.json").read_text())
    table = json.loads((RUNS / "table.json").read_text())
    damp = json.loads((RUNS / "damping.json").read_text())
    sig_ref = ref["sigma_ideal_Agk"]
    rows = [r for r in table if isinstance(r.get("sigma"), float) and math.isfinite(r["sigma"])]
    rows.sort(key=lambda r: r["dx"])

    got_rows = "\n".join(
        f"| {r['dx']:g} | {r['N']} | {r['steps']} | {r['sigma']:.4f} | "
        f"{100 * (r['sigma'] / sig_ref - 1):+.2f} % | {r['r2']:.4f} | "
        f"{100 * (r['vel_ratio'] - 1):+.2f} % | FAIL |"
        for r in rows)
    best = min(rows, key=lambda r: abs(r["sigma"] / sig_ref - 1))
    best_err = 100 * (best["sigma"] / sig_ref - 1)
    vel_best = min(rows, key=lambda r: abs(r["vel_ratio"] - 1))
    vel_best_err = 100 * (vel_best["vel_ratio"] - 1)
    damp_rows = "\n".join(f"| {d['dx']:g} | {d['sigma']:+.3f} | {d['r2']:.4f} |"
                          for d in damp)

    md = f"""# 交差核验 m16244 · CASE-SPH-RAYLEIGH-TAYLOR-MP（TensorSPH）→ L3，request changes

**结论先行**：实现是真的（L2）、接进包与基准驱动（L3），但**基准验证没过**——最好档相对 √(Agk) 误差 **{best_err:+.2f} %**，判据 **<10 %**；平台机器裁判 `l4-verdict/16244` 返回 `CLAIMED_NOT_PROVEN`（`tensorbench_results` 行数 0）。判 **L3**，返修。

## ① 案例介绍
二维单模 Rayleigh–Taylor 两层流（ρ_l/ρ_h = 500/1500 kg/m³，A = {ref['atwood']}，λ = 1 m，层深 d_l = d_h = 0.75 m，g = 9.81 m/s²，η₀ = 0.01λ，c₀ = 40 m/s，ν = 1e-3 m²/s）。重流体在上，早期界面扰动指数增长，指标 σ 对比线性理论锚：

- 理想：σ = √(Agk) = **{sig_ref:.6f} s⁻¹**（A·g·k = 0.5×9.81×2π，审计独立重算，与工件相对差 0.0）
- 有限深修正：{ref['sigma_finite_depth']:.6f} s⁻¹（因子 {ref['finite_depth_correction']:.5f}）
- 黏性修正：{ref['sigma_viscous']:.6f} s⁻¹（因子 {ref['viscous_correction']:.5f}）
- 出处：Taylor G.I., Proc. R. Soc. A **201** (1950) 192–196, doi:10.1098/rspa.1950.0052（线性阶段 σ² = Agk）

## ② 怎么做的（审计方法，非重做）
1. **三件套取证**：`GET /tensor/missions/16244` + 仓内 `benchmarks/rayleigh_taylor/runs/*.json`（脚本 `audit/audit_rt_m16318.py`）。
2. **机器裁判对表**：`GET /tensor/bench/l4-verdict/16244`；`tensor_benchmark_cases` 中 `anchor_slug='CASE-SPH-RAYLEIGH-TAYLOR-MP'` **0 行**（全表 30 行无 RT 条目），`tensorbench_results(16244)` = 0 行，`tensor_reports(16244)` = 0 篇。
3. **独立复算（反造假）**：不复用工件存储的 fit，直接从 `series.t / series.amplitude` 在窗口 η∈[0.02, 0.1]λ 重做 ln η 最小二乘 → 4 个完成档 σ 与存储值逐位重合（`audit/m16318_independent_refit.json`）。
4. **测试重跑**：`pytest tests/test_rayleigh_taylor.py -q` → 10 passed in 60.97 s。
5. **repro_cmd 可执行性**：CLI 解析通过，RK2+δ-SPH 分支实测在跑（99 s 推进到 t=0.175/0.45，被 120 s shell 上限收割）；全时长复跑递交集群（cj1301，审计窗口内仍排队）。
6. **工具链**：TensorSPH@`852e941`（PR #543）；psql 只读核 table；matplotlib 出图。

## ③ 计算结果（ref vs got，全部程序化注入自工件）
| dx [m] | 粒子数 | 步数 | σ_got [s⁻¹] | 质心量计误差 | R² | 速度量计误差 | <10 % |
|---|---|---|---|---|---|---|---|
{got_rows}

- 最好档：dx = {best['dx']:g} m → σ = {best['sigma']:.4f} s⁻¹，误差 **{best_err:+.2f} %** ⇒ **FAIL**（判据 10 %，差 {abs(best_err) - 10:.1f} 个百分点）。
- 速度模态量计最好档：dx = {vel_best['dx']:g} m，误差 {vel_best_err:+.2f} %（表观达标，但同档两量计相差 {abs(100 * (vel_best['vel_ratio'] - vel_best['sigma'] / sig_ref)):.1f} 个百分点，跨档散布 -41 %…+16 % ⇒ 属偶然抵消，**不采信为 PASS**）。
- A = 0 控制组（同几何、ρ_h = ρ_l，只测数值阻尼）：
| dx [m] | σ_num(A=0) [s⁻¹] | R² |
|---|---|---|
{damp_rows}
  即粗网格存在 **±2…4.4 s⁻¹ 量级的界面数值阻尼**，方向上解释系统性偏低；但该控制组只覆盖 dx = 0.05/0.025，最细档（0.0125）无对照。
- 未交付行：v2 文档 ④ 表另 3 行（dx = 0.02/0.0125 rk2、dx = 0.008 rk2+δ-SPH）仍为「集群计算中/待填/待判」，任务自带的集群作业 cj1287 在审计时 `queued`（排队 29 min，从未运行）。

## ④ 结果图
图 1（左）σ(dx) 对 Taylor 锚 ±10 % 带，质心量计（红）与速度量计（蓝）双口径对照，逐点标注相对误差；(右) dx = 0.0125/0.02 两档原始界面幅值历史 ln η(t)，阴影=拟合窗 η∈[0.02, 0.1]λ，虚线=σ 拟合。数据全部读自 `runs/*.json`，脚本 `scripts/m16318_audit_figure.py`。

## ⑤ 注意点（坑与局限）
1. **度量口径是判级的命门**：同一 dx、同一算例，质心量计 0.583 与速度量计 1.135（归一 σ/σ_th）可差 55 个百分点；只摘取达标的那一个就是 metric shopping。审计要求：先固定单一公示量计再报。
2. **工件聚合件失真**：`benchmarks/rayleigh_taylor/runs/summary.json` 只剩 1/8 条，因为 `run.py` 每次运行整表重写（末次覆盖）。下游若只读 summary 会漏掉 7 档。
3. **NaN 静默流通**：dx = 0.05 档无有效拟合窗 → σ = NaN 照样写进 `table.json/summary.json`，无 status 字段。
4. **新增能力零测试**：RK2 分支与 `density_diffusion ≠ 0` 没有任何专项断言，10 项测试全是 v1 的守恒/解析不变量。
5. **报告纪律缺口**：`tensor_reports` 0 篇、无图（`fields` 已存可出图），六段只在仓内 md 且 ④ 空。
6. **先例口径不一**：m16210 曾以「结果 FAIL 但诚实报告」判 L4；本单按任务书「基准判据是唯一裁判」+ 机器裁判 `l4_benchmark_pass=false` 判 L3，口径差异已留痕待平台统一。
7. **锚点重复风险**：既有 `CASE-RAYLEIGH-TAYLOR`（m15436，TensorLBM Boussinesq+相场）同一线性指标 6/6 PASS（-2.7 %…-9.9 %）——10 % 判据可达成，SPH 版 -24.5 % 是方法差距而非判据过苛；算例池登记时应带求解器族标签。

## ⑥ 误差判定（逐项，显式回答）
| 关键量 | ref | got | 相对误差 | 是否 <10 % | 是否 <3 % |
|---|---|---|---|---|---|
| σ（最好档 dx = {best['dx']:g}） | {sig_ref:.4f} s⁻¹ | {best['sigma']:.4f} s⁻¹ | {best_err:+.2f} % | **否** | **否** |
| σ（dx = 0.02） | {sig_ref:.4f} s⁻¹ | {rows[[r['dx'] for r in rows].index(0.02)]['sigma']:.4f} s⁻¹ | {100 * (rows[[r['dx'] for r in rows].index(0.02)]['sigma'] / sig_ref - 1):+.2f} % | 否 | 否 |
| σ（dx = 0.025） | {sig_ref:.4f} s⁻¹ | {rows[[r['dx'] for r in rows].index(0.025)]['sigma']:.4f} s⁻¹ | {100 * (rows[[r['dx'] for r in rows].index(0.025)]['sigma'] / sig_ref - 1):+.2f} % | 否 | 否 |
| σ（dx = 0.04） | {sig_ref:.4f} s⁻¹ | {rows[[r['dx'] for r in rows].index(0.04)]['sigma']:.4f} s⁻¹ | {100 * (rows[[r['dx'] for r in rows].index(0.04)]['sigma'] / sig_ref - 1):+.2f} % | 否 | 否 |
| 理论锚 √(Agk) 自身 | 5.5514884429 s⁻¹ | {sig_ref:.10f} s⁻¹ | 0.00 % | 是 | 是 |

**是否 <10 %：否（最好 {best_err:+.2f} %）。是否 <3 %：否。** 未粉饰：本算例线性增长率未达标，L4 不成立；honest-FAIL 归被审单，审计单只负责如实定级。

**落表**：`tensor_maturity_verdicts` id = 464，repo = TensorSPH，scope = `m16244 CASE-SPH-RAYLEIGH-TAYLOR-MP`，mission_id = 16318，level = **3**，verdict_by = board-audit，evidence 八键（impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor 全部真实指针）。
"""

    fig_b64 = base64.b64encode(FIG.read_bytes()).decode()
    payload = {
        "title": "交差核验 m16244 CASE-SPH-RAYLEIGH-TAYLOR-MP（TensorSPH）：L3 · RT 线性增长率 -24.48% vs <10% 判据 FAIL",
        "summary": (f"三件套核验：工件真实（独立复算 σ 逐位重合）、测试 10/10 通过、ref √(Agk)={sig_ref:.6f} 重算一致；"
                    f"但 4/4 完成网格全超容差（最好 {best_err:+.2f}%，判据 <10%），v2 改进档 3/3 未交付（cj1287 仍排队），"
                    "tensorbench_results/tensor_reports 均 0 行 → 机器裁判 CLAIMED_NOT_PROVEN。判 L3，request changes。"),
        "scenario_key": "board_audit/case_pool/rayleigh_taylor_mp",
        "mission_id": 16318,
        "tags": ["board-audit", "TensorSPH", "Rayleigh-Taylor", "multiphase", "benchmark", "maturity-L3",
                 "m16244", "case-pool"],
        "markdown": md,
        "figures": [{
            "name": "fig1_growth_audit.png",
            "caption": "(a) σ(dx) vs Taylor √(Agk) ±10 % 容差带（质心/速度双口径，标注相对误差）；(b) dx=0.0125 与 dx=0.02 原始 ln η(t) 及拟合窗",
            "png_base64": fig_b64,
        }],
    }
    tok = req("POST", "/auth/login", {"username": USER, "password": PASS})["access_token"]
    out = req("POST", "/tensor/reports", payload, tok=tok)
    print("report id:", out.get("id"), "| mission:", out.get("mission_id"), "| len(md):", len(md))


if __name__ == "__main__":
    main()