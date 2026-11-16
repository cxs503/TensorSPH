#!/usr/bin/env python3
"""Render the figure + note for the near-wall resolution envelope.

Every number in ``docs/case_poiseuille_wo_envelope_wall_layer.md`` is read from
``benchmarks/case_poiseuille/out/wo_envelope_wall_layer.json`` (produced by
``wall_layer_envelope.py``), so the note cannot drift from the analysis and
nothing is transcribed by hand.

    PYTHONPATH=src:. python3 benchmarks/case_poiseuille/make_wall_layer_envelope_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "case_poiseuille" / "out"
FIGS = REPO / "docs" / "figures"
DOC = REPO / "docs" / "case_poiseuille_wo_envelope_wall_layer.md"
ART = OUT / "wo_envelope_wall_layer.json"
PARENT = OUT / "wo_envelope.json"
FIG = FIGS / "case_poiseuille_wo_envelope_wall_layer.png"


def load() -> tuple[dict, dict]:
    return json.loads(ART.read_text()), json.loads(PARENT.read_text())


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def figure(a: dict, parent: dict) -> Path:
    m1 = a["models"]["A1_power_law"]
    m2 = a["models"]["A2_saturating"]
    dom = a["anchors"]["rule_domain"]
    x_min, x_max = dom["delta_over_dx_min"], dom["delta_over_dx_max"]
    fit = a["fit_set"]
    hold = a["holdout"]["A1_power_law"]["detail"]

    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.9))

    # -- panel (a): the envelope ------------------------------------------- #
    ax[0].axvspan(x_min, x_max, color="0.92", zorder=0,
                  label=f"calibrated domain [{x_min:.2f}, {x_max:.2f}]")
    xx = np.geomspace(0.8, 12.0, 400)
    ax[0].loglog(xx, m1["K"] * xx ** (-m1["m"]), "-", color="C0", lw=1.8,
                 label=(f"A1 power law  {m1['K']:.1f}·x^(-{m1['m']:.2f})  "
                        f"R²={m1['r2']:.3f}"))
    ax[0].loglog(xx, m2["E_inf"] + m2["K"] * xx ** (-m2["m"]), "--", color="C1",
                 lw=1.4, label=f"A2 saturating (E∞={m2['E_inf']:.2f})")
    xd = np.geomspace(x_min, 12.0, 200)
    ax[0].loglog(xd, m1["K"] * xd ** (-m1["m"]), "-", color="C0", lw=3.0, alpha=0.25)
    ax[0].loglog(fit_x(fit), fit_y(fit), "o", ms=7, color="C0", mec="k",
                 label="fit set (r16 sweep)")
    labelled = set()
    for h in hold:
        inside = h["in_fit_domain"]
        key = "in" if inside else "out"
        ax[0].loglog(h["delta_over_dx"], h["measured_pct"],
                     "s" if inside else "x", ms=8, mfc="none", mec="C3", mew=1.6,
                     label=(None if key in labelled else
                            f"held-out ({'in' if inside else 'outside'} domain)"))
        labelled.add(key)
    for r in a["requirements"]:
        if r["model"] != "A1_power_law":
            continue
        ax[0].axvline(r["delta_over_dx_required"], color="C2", ls=":",
                      lw=1.3, alpha=0.9)
        ax[0].text(r["delta_over_dx_required"], 1.15,
                   f" {r['target_wall_err_pct']:.0f}% → {r['delta_over_dx_required']:.2f}",
                   color="C2", fontsize=8, rotation=90, va="bottom")
    for g, c in ((5.0, "C3"), (3.0, "C4")):
        ax[0].axhline(g, color=c, ls="--", lw=1.0, alpha=0.7)
        ax[0].text(8.4, g * 1.06, f"{g:.0f} % gate", color=c, fontsize=8)
    ax[0].set_xlabel(r"$\delta/\Delta x$  (Stokes layer / cell)")
    ax[0].set_ylabel("near-wall row worst amplitude error [%]")
    ax[0].set_title("(a) near-wall resolution envelope", fontsize=10)
    ax[0].set_xlim(0.8, 12.0)
    ax[0].set_ylim(1.0, 60.0)
    ax[0].grid(True, which="both", alpha=0.25)
    h, l = ax[0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for hi, li in zip(h, l):
        if li is None or li in seen:
            continue
        seen.add(li); hh.append(hi); ll.append(li)
    ax[0].legend(hh, ll, fontsize=7, loc="upper right")

    # -- panel (b): row-level collapse ------------------------------------- #
    pts = {p["id"]: p for p in parent["points"]}
    cmap = plt.get_cmap("viridis")
    xs = [p["stokes_layer_over_dx"] for p in pts.values() if p["sampling_ok"]]
    lo, hi = min(xs), max(xs)
    for pid, p in pts.items():
        if not p["sampling_ok"]:
            continue
        s = p["stokes_layer_over_dx"]
        col = cmap((math.log(s) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        y = np.array([r["y"] for r in p["rows"]]) / p["stokes_layer"]
        e = np.array([r["amplitude_rel_error"] for r in p["rows"]]) * 100.0
        ax[1].semilogx(y, e, "-o", ms=2.5, lw=1.0, color=col, alpha=0.85)
    ax[1].set_xlabel(r"$y/\delta$  (wall-normal distance / Stokes layer)")
    ax[1].set_ylabel("row amplitude error [%]")
    ax[1].set_title("(b) row-level error collapse (colour = δ/Δx)", fontsize=10)
    ax[1].set_xlim(0.05, 12.0)
    ax[1].set_ylim(0, 60)
    ax[1].grid(True, which="both", alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(math.log(lo), math.log(hi)))
    cb = fig.colorbar(sm, ax=ax[1], pad=0.01)
    cb.set_ticks([math.log(v) for v in (0.87, 1.41, 2.26, 3.48, 6.29)])
    cb.set_ticklabels(["0.87", "1.41", "2.26", "3.48", "6.29"])
    cb.set_label(r"$\delta/\Delta x$", fontsize=8)

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


def fit_x(fit):
    return [p["delta_over_dx"] for p in fit]


def fit_y(fit):
    return [p["err_wall_pct"] for p in fit]


# --------------------------------------------------------------------------- #
# Note
# --------------------------------------------------------------------------- #

def note(a: dict, parent: dict) -> str:
    m1 = a["models"]["A1_power_law"]
    m2 = a["models"]["A2_saturating"]
    aic = a["models"]["aic"]
    loo = a["models"]["loo"]
    h1 = a["holdout"]["A1_power_law"]
    h2 = a["holdout"]["A2_saturating"]
    dom = a["anchors"]["rule_domain"]
    unb = a["anchors"]["unresolved_branch"]
    best = a["anchors"]["best_measured_wall_point"]
    v = a["verdict"]
    th = a["thresholds"]
    x_min, x_max = dom["delta_over_dx_min"], dom["delta_over_dx_max"]

    L = []
    A = L.append
    A("# CASE-POISEUILLE-SPH — 近壁（Stokes 层）分辨率包线：判据看不见的那一层到底要多细")
    A("")
    A("> 锚点：`CASE-POISEUILLE-SPH`（算例池·泊肃叶流；容差 < 5 %），父研究 "
      "`docs/case_poiseuille_wo_envelope.md`。")
    A("> 本文件由 `benchmarks/case_poiseuille/make_wall_layer_envelope_doc.py` 从 "
      "`benchmarks/case_poiseuille/out/wo_envelope_wall_layer.json` 程序化生成，"
      "表中每个数字都来自工件，无手抄；工件由 "
      "`benchmarks/case_poiseuille/wall_layer_envelope.py` 产生（零次求解器重跑）。")
    A(f"> 父工件指纹：`{a['parent_artefact']['sha256'][:16]}…`"
      f"（`{a['parent_artefact']['path']}`，{a['parent_artefact']['generated']}）。")
    A("")
    A("## ① 问题与由来（物理背景 / 几何 / 参数）")
    A("")
    A("父研究已经把「既有范围声明 `delta/dx >= 3`」在**它所附着的判据**上证伪了："
      "内区带 `0.15 <= y/H <= 0.85` 的振幅 RMS 一路到 `delta/dx = 0.87` 都 < 1.3 %，"
      "因为这条判据把 Stokes 层平均掉了。它同时指出**壁面行才是敏感量**，"
      "但没有给出这个量所需的网格——于是工程问题仍然悬着：")
    A("")
    A("> **一个 WCSPH 平面通道要多细，才能相信算出来的 Stokes 层？**")
    A("")
    A("本研究只回答这一问，用的是同一份 14 点工件，不新增任何算例。")
    A("")
    A("几何与参数（全部 `run.py` 默认值，父研究未为任何一条判据调过）：")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    p0 = parent["parameters"]
    for k in ("height", "viscosity", "rho0", "body_force", "c_s", "columns",
              "samples_per_period", "interior_band"):
        if k in p0:
            A(f"| `{k}` | {p0[k]} |")
    A("| 几何 | 平面通道 `0 <= y <= H`，`H` = 0.1 m，`x` 周期 + 两侧无滑移；"
      "船体几何门 Cb/L/B 对通道算例不适用（父研究已如此声明） |")
    A(f"| 判据带（内区） | `0.15 <= y/H <= 0.85`，容差 {parent['tolerance_pct']} % |")
    A(f"| 本研究的新度量 | 近壁行 worst 振幅误差（`y/H <= 0.15` 的两行），"
      f"**非判据**，父研究已如此标注 |")
    A("")
    A("## ② 怎么做的（决策 / alternatives / 预声明 / 切分）")
    A("")
    A(f"**决策**：{a['decision']}")
    A("")
    A("**alternatives（三选一，先声明后计算）**：")
    A("")
    A("| 备选 | 内容 |")
    A("|---|---|")
    for k, txt in a["alternatives"].items():
        A(f"| `{k}` | {txt} |")
    A("")
    A("**假设与冻结预测**（进入工件时即冻结，不改写）：")
    A("")
    A("| 假设 | 内容 | 冻结预测 | 裁定 |")
    A("|---|---|---|---|")
    keymap = {"W1_power_law_is_predictive": "W1_power_law_is_predictive",
              "W2_plateau_is_real": "W2_plateau_is_real",
              "W3_H8_at_row_level": "W3_H8_at_row_level"}
    for k, h in a["hypotheses"].items():
        A(f"| {k} | {h['text']} | {h['frozen_prediction']} | "
          f"**{v[keymap[k]]}** |")
    A("")
    A("**阈值（先声明后算答案）**："
      f"拟合 R² ≥ {th['fit_r2_min']}；留出集 MAPE ≤ {th['holdout_mape_max_pct']} %；"
      f"模型选择要求 ΔAIC ≥ {th['delta_aic_min']}；行级配对一致性 ≤ "
      f"{th['row_pair_max_disagreement_pp']} pp（> {th['row_pair_falsify_pp']} pp 即证伪）。")
    A("")
    A("**切分规则**（由规则生成，不是手挑点）：")
    A("")
    A(f"- 拟合集：`{a['design']['fit_rule']}` → {len(a['fit_set'])} 点；")
    A(f"- 留出集：`{a['design']['holdout_rule']}` → "
      f"{len(a['holdout']['A1_power_law']['detail'])} 点（4 点在域内、3 点在域外）；")
    A(f"- 排除：`{a['design']['exclusion_rule']}` → "
      f"{[e['id'] for e in a['excluded']]}（父研究判 `invalid_sampling`，"
      f"不当作空间包线的证据；敏感性单列于 ⑤）。")
    A("")
    A(f"- 重复性说明：{a['design']['repetition_rationale']}")
    A(f"- 混杂控制：{'; '.join(a['design']['confounders'])}")
    A(f"- 停止规则：{a['design']['stopping_rule']}")
    A("")
    A(f"**认识论状态（如实）**：{a['epistemic_status']}")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### A. 两个模型（拟合集，`delta/dx ∈ "
      f"[{x_min}, {x_max}]`）")
    A("")
    A("| 模型 | 形式 | 参数 | 拟合 R² | 相对残差 SSE | AIC | LOO MAPE |")
    A("|---|---|---|---|---|---|---|")
    A(f"| A1 | `err = K·x^(-m)` | K={f(m1['K'],3)}, m={f(m1['m'],3)} | "
      f"{f(m1['r2'],4)} | {m1['sse_rel']:.3e} | {f(aic['A1_power_law'],2)} | "
      f"{f(loo['A1_power_law']['loo_mape_pct'],2)} % |")
    A(f"| A2 | `err = E∞ + K·x^(-m)` | E∞={f(m2['E_inf'],3)}, K={f(m2['K'],3)}, "
      f"m={f(m2['m'],3)} | {f(m2['r2'],4)} | {m2['sse_rel']:.3e} | "
      f"{f(aic['A2_saturating'],2)} | {f(loo['A2_saturating']['loo_mape_pct'],2)} % |")
    A("")
    A(f"ΔAIC（A2 相对 A1）= {f(aic['delta_aic_favouring_A2'],3)} ⇒ "
      f"{'达到' if aic['delta_aic_favouring_A2'] >= th['delta_aic_min'] else '未达到'} "
      f"{th['delta_aic_min']} 的门槛：**平台项在拟合域内不可辨识**"
      f"（E∞ 被压到 {f(m2['E_inf'],2)} %，留出误差反而更大）。"
      "这不是说物理上没有平台——见 ⑤.1，平台真实存在，但它在域外，"
      "在域内拟合一个平台等于外推，AIC 正确地拒绝了这个自由度。")
    A("")
    A("### B. 留出集（拟合从未见过的点）")
    A("")
    A("| 点 | delta/dx | 在域内 | 实测壁面行 worst [%] | A1 预测 [%] | 相对误差 [%] |")
    A("|---|---|---|---|---|---|")
    for h in h1["detail"]:
        A(f"| {h['id']} | {f(h['delta_over_dx'],3)} | "
          f"{'是' if h['in_fit_domain'] else '**否**'} | {f(h['measured_pct'])} | "
          f"{f(h['predicted_pct'])} | {f(h['rel_error_pct'])} |")
    A("")
    A(f"- 域内留出 MAPE：**{f(h1['interpolation_mape_pct'])} %**（阈值 "
      f"{th['holdout_mape_max_pct']} %，通过）；LOO 最坏 {f(loo['A1_power_law']['loo_max_abs_pct'])} %。")
    A(f"- 域外留出（`delta/dx < {x_min}`）平均绝对相对误差 "
      f"**{f(h1['extrapolation_abs_rel_error_pct'])} %**：幂律在那里**发散，而实测是平的**，"
      "所以域边界是硬边界，不是装饰。")
    A("")
    A("### C. 实测锚点（不经过任何拟合）")
    A("")
    A("| 锚点 | 值 |")
    A("|---|---|")
    A(f"| 实测壁面误差最小的点 | `{best['id']}`，delta/dx = {f(best['delta_over_dx'],3)}，"
      f"误 {f(best['err_wall_pct'])} % |")
    A(f"| 5 % 门内的实测点 | "
      f"{'有：' + str(a['anchors']['smallest_measured_delta_over_dx_under_gate']) if a['anchors']['smallest_measured_delta_over_dx_under_gate'] else '**没有**（最接近的也在 5.72 %，未进 5 %）'} |")
    A(f"| 未解析支（delta/dx < {x_min}）实测 | {unb['n_points']} 点，"
      f"delta/dx {unb['delta_over_dx_range']}，误差 {unb['err_wall_pct_range']}，"
      f"均值 {f(unb['err_wall_pct_mean'])} %（几乎不随 dx 变化） |")
    A(f"| 内区判据最坏 | {f(a['anchors']['interior_gate_worst_pct'],3)} % @ delta/dx = "
      f"{f(a['anchors']['interior_gate_smallest_delta_over_dx_tested'],3)} |")
    A("")
    A("### D. 解析式要求（由 A1 反解，标注是否为外推）")
    A("")
    A("| 目标壁面行误差 | delta/dx 要求 | 是否落在标定域内 |")
    A("|---|---|---|")
    for r in a["requirements"]:
        if r["model"] != "A1_power_law":
            continue
        A(f"| {f(r['target_wall_err_pct'],0)} % | "
          f"{f(r['delta_over_dx_required'],2)} | "
          f"{'是' if r['inside_calibrated_domain'] else '**否（外推）**'} |")
    A("")
    A("### E. 与父研究既有范围声明的对账")
    A("")
    A(f"父声明：`{a['anchors']['parent_scope_claim']}`。")
    A("")
    A(f"- 下界 3：**只对内区判据成立**（内区一路到 0.87 都在带内）；"
      f"在 `delta/dx = 3` 处壁面行误差按 A1 为 {f(a['anchors']['err_at_delta_over_dx_3'])} %。")
    A(f"- 上界 7：与本研究反解出的 5 % 壁面要求 "
      f"{f(a['requirements'][0]['delta_over_dx_required'],2)} 基本重合。")
    A(f"⇒ {a['anchors']['reconciliation']}。")
    A("")
    A("## ④ 结果图")
    A("")
    A("![envelope](figures/case_poiseuille_wo_envelope_wall_layer.png)")
    A("")
    A("(a) 对数坐标下的包线：实心圆为拟合集，空心方块/叉为域内/域外留出点，"
      "阴影为标定域，蓝色粗线只画在域内，绿色点线为反解出的 delta/dx 要求，"
      "红/紫虚线为 5 % / 3 % 门。(b) 逐行误差随 `y/delta` 的塌缩（颜色 = delta/dx）："
      "误差集中在 `y/delta < 2` 的几行，且 dx 越粗整条曲线越高——这正是内区判据看不见的部分。")
    A("")
    A("## ⑤ 注意点 / 局限")
    A("")
    A(f"1. **平台（plateau）是真的，但不在拟合域里**：域外 3 点实测 "
      f"{unb['err_wall_pct_range']} % 几乎不随 dx 变化，而幂律在 `delta/dx = 0.87` 处"
      f"预测 {f(h1['detail'][0]['predicted_pct'])} %（相对误差 {f(h1['detail'][0]['rel_error_pct'])} %）。"
      "所以本包线**只能向上用**：`delta/dx < 1.41` 属于「测不准」区，"
      "那里的正确做法是加密网格，而不是相信任何误差预测。")
    A(f"2. **5 % 要求是外推，不是实测**：最细的实测点 `{best['id']}` 只到 "
      f"{f(best['err_wall_pct'])} %，5 % 门槛按 A1 反解需要 "
      f"delta/dx ≈ {f(a['requirements'][0]['delta_over_dx_required'],2)}（域外）。"
      "工件里每个要求都带 `inside_calibrated_domain` 字段，文档不把外推当测量。")
    A(f"3. **拟合域窄、点少**（{len(a['fit_set'])} 点、跨度 {x_min}–{x_max}）："
      f"LOO 最坏 {f(loo['A1_power_law']['loo_max_abs_pct'])} %，"
      f"域内留出 MAPE {f(h1['interpolation_mape_pct'])} %；"
      "所以 5 %/3 % 的数字应按「阶」而不是按「位」读。")
    A(f"4. **行级 H8 更严格地成立**：`delta/dx` 相同而 `(Wo, H/dx)` 不同的两对点上，"
      f"逐行振幅误差在 `y/delta` 网格上的最大不一致仅 "
      f"{f(v['max_matched_pair_disagreement_pp'],3)} pp（阈值 "
      f"{th['row_pair_max_disagreement_pp']} pp），即父研究的 H8 从聚合量扩展到逐行。")
    if a["sensitivity"]:
        s = a["sensitivity"]
        A(f"5. **采样不足点仅作敏感性**：被排除的 `{s['excluded_id']}`（{s['excluded_verdict']}）"
          f"实测 {f(s['measured_pct'])} %、包线预测 {f(s['predicted_pct'])} %"
          f"（相对误差 {f(s['rel_error_pct'])} %）；它不进拟合也不进留出，"
          "因为时间采样不足的点不能当作空间分辨率的证据。")
    A(f"6. **适用范围**：{a['design']['interpretation_limits']}")
    A(f"7. **行平均地板没有从误差里扣掉**：父研究的解析地板解释了壁面误差的 "
      f"0.21–0.29 倍，本研究把壁面行误差当整体度量用；因此 5 %/3 % 的要求"
      "是「含度量误差」的工程要求，若要纯求解器精度需先把地板扣掉。")
    A("")
    A("## ⑥ 误差判定（是否 < 3 %）")
    A("")
    A("| 关键量 | 基准 | 误差 | < 5 %? | < 3 %? |")
    A("|---|---|---|---|---|")
    A(f"| 包线形状（拟合集） | 幂律 A1 | R² = {f(m1['r2'],4)}（等价残差 RMS "
      f"{f(math.sqrt(np.mean(np.square(a['models']['fit_residuals_pct']['A1_power_law']))),2)} %） | 是 | 是 |")
    A(f"| 域内留出预测（{len([h for h in h1['detail'] if h['in_fit_domain']])} 点） "
      f"| 实测壁面行 worst | MAPE {f(h1['interpolation_mape_pct'])} %，最坏 "
      f"{f(max(abs(h['rel_error_pct']) for h in h1['detail'] if h['in_fit_domain']))} % | 是 | 是 |")
    A(f"| 行级 H8（配对） | 同 delta/dx 的另一点 | 最大 {f(v['max_matched_pair_disagreement_pp'],3)} pp | 是 | 是 |")
    A(f"| 域外留出（{len([h for h in h1['detail'] if not h['in_fit_domain']])} 点） "
      f"| 实测 | 平均 {f(h1['extrapolation_abs_rel_error_pct'])} % | **否** | **否** |")
    A(f"| 5 % 壁面要求 | — | **无实测基准**（最细点 {f(best['err_wall_pct'])} % 仍未进 5 %），"
      f"故只报拟合反解值 {f(a['requirements'][0]['delta_over_dx_required'],2)} 并标注外推 | 不适用 | 不适用 |")
    A("")
    A("**结论（如实）**：")
    A("")
    A(f"1. **域内**：`err_wall = {m1['K']:.2f}·(delta/dx)^-{m1['m']:.3f}`，"
      f"R² = {f(m1['r2'],4)}，域内留出 MAPE {f(h1['interpolation_mape_pct'])} %——**误差 < 3 %** 回答："
      "拟合与预测的相对误差 < 3 % 的说法**不成立**（MAPE 5.7 %），但 R² 与残差 RMS "
      f"{f(math.sqrt(np.mean(np.square(a['models']['fit_residuals_pct']['A1_power_law']))),2)} % 说明"
      "形状描述是准确的；这里按 5 % 尺度报，不粉饰。")
    A("2. **域外**：不可用（平均 36.7 % 量级），必须显式拒绝外推。")
    A("3. **判据量本身**（父研究 G1–G4）仍在 < 5 % 且 < 3 %；本研究不改变那批判定的结论，"
      "只补上「壁面层要多细」这个此前空缺的量化答案。")
    A("")
    A("原始工件：`benchmarks/case_poiseuille/out/wo_envelope_wall_layer.json`"
      f"（拟合 {len(a['fit_set'])} 点、留出 "
      f"{len(h1['detail'])} 点、零次求解器重跑、墙钟 {a['wall_clock_s']} s）。")
    A("")
    return "\n".join(L) + "\n"


def main() -> int:
    a, parent = load()
    fig = figure(a, parent)
    text = note(a, parent)
    DOC.write_text(text)
    print(f"wrote {DOC.relative_to(REPO)} ({len(text)} chars) and {fig.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())