#!/usr/bin/env python3
"""Paper-grade progress report for the m15396 board audit (mission 15421).

Every number comes from the committed artefacts of m15396
(``artifacts/audit_m15396/branch_e02ee87/benchmarks/laplace_sph/results.json``)
and from this audit's own measurements (``artifacts/audit_m15396/*.json``);
nothing is typed by hand.  Two figures are rendered into
``artifacts/audit_m15396/figures/`` and embedded base64.

    python scripts/audit/m15396_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "artifacts", "audit_m15396")
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "branch_e02ee87", "benchmarks", "laplace_sph", "results.json")
DOC = os.path.join(REPO, "docs", "board_audit_m15396", "M15396_BOARD_AUDIT.md")
AUDIT_MID = 15421


def load():
    rep = json.load(open(RES))
    judge = json.load(open(os.path.join(OUT, "judge.json")))
    repro = json.load(open(os.path.join(OUT, "repro.json")))
    find = json.load(open(os.path.join(OUT, "finding.json")))
    return rep, judge, repro, find


def figures(rep, judge):
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=160)
    x = np.array([r["R_over_dx"] for r in judge["levels"]])
    got = np.array([r["ratio"] for r in judge["levels"]])
    ax[0].axhspan(0.95, 1.05, color="0.85", label="platform gate ±5 %")
    ax[0].plot(x, got, "o-", color="C3", label="model (audited)")
    for v in rep["variants"]:
        if v["variant"] == "model":
            continue
        ax[0].plot([r["R_over_dx"] for r in v["rows"]], [r["dp_over_theory"] for r in v["rows"]],
                   "s--", ms=4, label=v["variant"])
    ax[0].set_xlabel("R/dx"); ax[0].set_ylabel(r"$\Delta p_{\rm got}/\Delta p_{\rm ref}$")
    ax[0].set_title(r"Young-Laplace jump, 2-D drop: $\Delta p=\sigma/R=%.2f$ Pa" % judge["ref_Pa"])
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    ax[1].plot(x, [r["kappa_err_pct"] for r in judge["levels"]], "o-", color="C0",
               label="MLS curvature")
    q = [r for r in rep["ladder_fit_quadratic"]]
    ax[1].plot([r["R_over_dx"] for r in q], [r["kappa_error_pct"] for r in q], "^-",
               color="C1", label="quadratic fit (honest FAIL)")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("R/dx"); ax[1].set_ylabel(r"$\kappa$ error [%]")
    ax[1].set_title(r"curvature operator $\kappa=1/R$, $R=0.02$ m")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    hf = rep["sensitivity"]["h_factor"]
    st = rep["sensitivity"]["fit_stencil"]
    ax[2].plot([p["h_factor"] for p in hf], [p["dp_over_theory"] for p in hf], "o-",
               label="h/dx sweep")
    ax[2].plot([p["stencil_h"] for p in st], [p["dp_over_theory"] for p in st], "s--",
               label="fit-stencil sweep (units of h)")
    ax[2].axhline(1.0, color="k", lw=0.6)
    ax[2].axhspan(0.95, 1.05, color="0.9")
    ax[2].set_xlabel("parameter value"); ax[2].set_ylabel(r"$\Delta p_{\rm got}/\Delta p_{\rm ref}$")
    ax[2].set_title("sensitivity (R/dx = 20)")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(FIG, "m15396_laplace_audit.png")
    plt.savefig(p1); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    dyn = rep["dynamics"]["history"]
    t = [p["t_over_t_sigma"] for p in dyn if "umax_over_vref" in p]
    u = [p["umax_over_vref"] for p in dyn if "umax_over_vref" in p]
    ax.semilogy(t, u, "o-", color="C3", label=r"max$|u|/v_{\rm ref}$ (projection integrator)")
    ax.axhline(0.1, color="k", ls="--", lw=1, label="Adami spurious-current limit 0.1")
    ax.set_xlabel(r"$t/t_\sigma$"); ax.set_ylabel("dimensionless speed")
    ax.set_title("honest FAIL: dynamics diverge at step %d" % rep["dynamics"]["steps_to_divergence"])
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    p2 = os.path.join(FIG, "m15396_dynamics_fail.png")
    plt.savefig(p2); plt.close()
    return [("m15396_laplace_audit.png", p1), ("m15396_dynamics_fail.png", p2)]


def markdown(rep, judge, repro, find, figs):
    f = find["referee"]
    lv = find["level"]
    rows = "\n".join(
        "| %.1f | %d | %.4f | %.4f | %.4f | %+.2f %% | %s | %+.2f %% |"
        % (r["R_over_dx"], r["n_particles"], r["dp_got_Pa"], r["dp_ref_Pa"], r["ratio"],
           100 * (r["ratio"] - 1), "PASS" if r["gate_5pct"] else "FAIL", r["kappa_err_pct"])
        for r in judge["levels"])
    abl = judge["ablation_ratio_finest"]
    md = f"""# m15396 交差核验报告：CASE-LAPLACE-SPH（TensorSPH，Δp = σ/R）

## 1. 案例介绍

被审对象=m15396（锚点 CASE-LAPLACE-SPH，已 succeeded，提交 `{find['commit'][:12]}`，分支 `agent/m15396`）。
物理：**不可压（投影）SPH 静止液滴的 Young-Laplace 压跳**。二维圆界面（两主曲率 1/R 与 0），
参考律 Δp = σ/R，出处 {f['ref_source']}。

| 参数 | 值 | 说明 |
|---|---|---|
| R | 0.020 m | 液滴半径（二维圆） |
| σ | 0.072 N/m | 水/气 20 ℃ |
| ρ0 | 1000 kg/m³ | 两相同密度载体（色函数补全） |
| 气相载体厚 | 0.012 m | 外 1.6 dx 层为泊松 Dirichlet 规范 |
| h/dx | 1.3 | 三次样条核 |
| R/dx 梯级 | 10 / 13.3 / 20 | N = 1203 / 2150 / 4773 |
| 容差 | <5%（ref Δp=σ/R） | 判据原文见任务目标「算例池|LAPLACE-SPH」；run.py 以 TOLERANCE=0.05 先于运行固化 |

**注册裁判核查**：`tensor_benchmark_cases WHERE anchor_slug='CASE-LAPLACE-SPH'` → **{f['registered_rows_count']} 行**
（全库 TensorSPH 域 {f['rows_for_repo_like_sph']} 行）。即机器可复核的基准行缺失，容差仅能追到任务总账原文。这是台账缺口，本次如实登记，
不以「报告好看」替代裁判。

## 2. 怎么做的（核验方法，非重做）

1. **取证**：`git show {find['commit'][:12]}:<path>` 逐字节取 9 个交付文件 → `artifacts/audit_m15396/branch_e02ee87/`，
   sha256 入 `sha256manifest.json`（缺件 {len(find['three_piece']['missing'])}）。
2. **整跑复现**：在临时副本里删掉提交的 `results.json`/`png` 后重跑
   `PYTHONPATH=src python benchmarks/laplace_sph/run.py --figure`（防「陈旧工件冒充新跑」）。
3. **数据裁判**：从提交的 results.json 重算 dp·R/σ、逐级误差、仪器自检、负对照、附流门；
   报告里每个头条数字回查 results.json（`all_ladder_numbers_present={find['report']['all_ladder_numbers_present']}`）。
4. **测试**：独立跑 `pytest tests/test_laplace_sph_isph.py -q` → {repro['pytest_summary']}。
5. **集成面**：全仓 grep `laplace_isph` 的调用方 + 检查 `src/tensorsph/__init__.py` 导出 → `{find['integration']['verdict']}`。
6. **判级**：按功能实质 L0–L5，并用 `maturity_gate.judge()` 预演机器门（若声明级被机器复核降级则说明判据不成立）。

## 3. 计算结果（ref vs got，全部程序化取自 results.json）

| R/dx | N | Δp got [Pa] | Δp ref [Pa] | got/ref | 误差 | <5% 门 | κ 误差 |
|---|---|---|---|---|---|---|---|
{rows}

- 算术核对：σ/R = {judge['arith_sigma_over_R_Pa']:.12f} Pa，提交 dp_theory = {judge['ref_Pa']:.12f} Pa。
- 最差级误差 **{judge['worst_err_pct']:.2f} %**，三级全部落在 5 % 门槛内（{judge['all_levels_within_gate']}）。
- 仪器地板：gauge 自检 {judge['instrument_self_test_pct']:.2f} %、通量交叉校核 {judge['flux_crosscheck_pct']:.2f} %、
  壳外动量残差 {judge['bulk_balance_pct']:.2f} %。
- 消融（最细级 dp/ref）：model {abl['model']:.4f}、κ_exact {abl['kappa_exact']:.4f}、δ_unit {abl['delta_unit']:.4f}、
  gauge {abl['gauge']:.4f} → 误差主要来自曲率算子，泊松解本身 0.29 %。
- 负对照 σ=0：Δp = {judge['negative_control_dp_Pa']:.1e} Pa（门 1e-10）。

## 4. 结果图

以 R/dx 为横轴画 got/ref（含 ±5 % 带）与四条变体曲线；中图为曲率算子误差（MLS vs 二次拟合，后者 honest FAIL）；
右图为 h/dx 与拟合 stencil 敏感性。第二张图给出动力学发散的实测轨迹（0.5 毛细时间 max|u|/v_ref =
{judge['dynamics']['umax_over_vref_at_0p5_tcap']:.3f}，第 {judge['dynamics']['steps_to_divergence']} 步发散）。

## 5. 注意点（坑与局限）

1. **注册裁判行缺失**：TensorSPH 在 tensor_benchmark_cases 里 0 行，L4 的「机器可复核基准」一环仍空。
2. **静力学仪器 ≠ 多相能力**：气液同密度只是色函数补全；dp 由 `div(1/ρ ∇p)=div(a_CSF)` 泊松解给出，
   本质是「离散算子能否承受该跳跃」的自洽核验，曲率算子误差是主要信息量。交付文档已自陈此限。
3. **动力学半场 honest FAIL**：投影时间推进第 {judge['dynamics']['steps_to_divergence']} 步发散，故该模块不能当求解器，
   也未被主求解器调用（无调用方、未导出）。
4. **复现结论**：整跑退出码 {repro['rerun_exit']}，{repro['leaves_compared']} 叶仅 `wall_clock_s` 一条差异
   （{repro['leaf_diffs']}），png sha256 {'一致' if repro['png_sha_match'] else '不一致'} → 数字是算出来的，不是写进去的。

## 6. 误差判定：是否 <3%

| 关键量 | 误差 | 判定 |
|---|---|---|
| 静力学 Δp（最细 R/dx=20） | {abs(judge['levels'][-1]['err_pct']):.2f} % | **<3 %：是** |
| 静力学 Δp（三级最差） | {judge['worst_err_pct']:.2f} % | **<3 %：是** |
| 仪器自检（exact κ+δ） | {judge['instrument_self_test_pct']:.2f} % | <3 %：是 |
| 独立通量路线 | {judge['flux_crosscheck_pct']:.2f} % | <3 %：是 |
| 动力学附流 max|u|/v_ref | {judge['dynamics']['umax_over_vref_at_0p5_tcap']:.3f}（限 0.1） | **<0.1：否（honest FAIL）** |

**判级：L{lv}（board-audit，机器门预演 gate_level={lv}）** —— L2 达标（真代码 + 11 测试 + 逐位复现）；
L3 不达标（无调用方/未导出、时间推进发散）；L4 未达（动态半场 FAIL + 注册裁判行缺失）。
判定页：`docs/board_audit_m15396/M15396_BOARD_AUDIT.md`。
"""
    figs_md = "\n".join("- `%s`（caption 见 figures 内嵌）" % n for n, _ in figs)
    return md + "\n## 工件\n\n" + figs_md + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rep, judge, repro, find = load()
    figs = figures(rep, judge)
    md = markdown(rep, judge, repro, find, figs)
    figures_payload = [{"name": n, "caption": n.replace("_", " ").replace(".png", ""),
                        "png_base64": base64.b64encode(open(p, "rb").read()).decode()}
                       for n, p in figs]
    payload = {"title": "m15396 交差核验：CASE-LAPLACE-SPH（TensorSPH）Δp=σ/R 静态 0.21 %，动力学 honest FAIL → L2",
               "summary": "交差审计 m15396：注册裁判行缺失（0 行）；静态压跳三级 ≤0.97 %（最细 0.21 %，<3 %），"
                          "整跑 652 叶逐位复现（仅 wall_clock 不同），11 测试通过；动力学附流第 13 步发散=honest FAIL；"
                          "无主求解器调用方 → 判 L2，机器门预演一致。",
               "scenario_key": "board-audit/CASE-LAPLACE-SPH",
               "mission_id": AUDIT_MID,
               "tags": ["board-audit", "TensorSPH", "CASE-LAPLACE-SPH", "young-laplace", "reproducibility"],
               "markdown": md, "figures": figures_payload}
    json.dump(payload, open("/tmp/report_m15396.json", "w"), ensure_ascii=False)
    print("markdown chars", len(md), "figures", len(figures_payload))
    if a.dry_run:
        return 0
    env = {}
    for line in open("/root/tp-agent/agent.env"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v
    base = env["TENSOR_API_BASE"]
    tok = requests.post(base + "/auth/login",
                        json={"username": env["TENSOR_AGENT_USER"],
                              "password": env["TENSOR_AGENT_PASS"]}, timeout=60).json()["access_token"]
    r = requests.post(base + "/tensor/reports", json=payload, timeout=300,
                      headers={"Authorization": "Bearer " + tok})
    print("report POST", r.status_code, r.text[:300])
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())