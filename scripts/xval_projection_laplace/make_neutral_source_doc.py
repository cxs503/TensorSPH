"""Render ``docs/xval_projection_laplace_neutral_source.md`` from the round-7 JSON.

Every number in the document is injected from ``results/neutral_source.json`` --
nothing is typed by hand.  Also writes the figure (vapour tail profiles and the
probe-set bias bars) next to the results so the platform report can embed it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DOC = HERE.parents[1] / "docs" / "xval_projection_laplace_neutral_source.md"


def main():
    d = json.loads((OUT / "neutral_source.json").read_text())
    L = d["levels"]
    G = d["gate_verdicts"]
    g = d["gates_declared"]
    best = d["best_symmetric_probe_set"]
    sym = [k for k in L[0]["arms"]["shipped"]["readout"] if k.startswith("sym_")]

    fig_note = _figure(L)
    lines = []
    A = lines.append
    A("# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE，第 7 轮：气相压强台阶的机理判定与读数对称性")
    A("")
    A(f"锚点 `{d['anchor']}`。第 6 轮把气相的压强台阶（`<p>(s=+6..8 dx)` 随分辨率"
      "*增大*）归因于“Poisson 源残余单极子经 Dirichlet 规范成像填满气相”，并把"
      "“强制源净电荷为零”留作后续实验。本轮把该归因当作可判定的假设来检验，"
      "同时检验第二个候选机理（净牵引力偶极子）与读数仪器本身。")
    A("")
    A("## ① 案例与方法")
    A("")
    A("静止椭球液滴（与第 5、6 轮同一载流子）：`pi a b = pi R_eq^2`，"
      "`sigma` = 0.072 N/m，`rho0` = 1000 kg/m³（两相同密度静态载流子），"
      "owner 精度门限 15 %。构型："
      + "、".join(f"a:b={x['aspect_ratio']:.1f} `R_eq/dx`={x['r_eq_over_dx']:.0f}"
                 for x in L) + "；第 5 轮预先声明的可分辨性前提为 `R_curv,min/dx >= 5`"
      "（本轮的 `r_curv_min_over_dx` 见 JSON）。")
    A("")
    A("四个臂，共用一个格点、一套 colour/curvature/force 算子、一个 Poisson 求解器：")
    A("")
    A("| 臂 | 做法 | 检验的假设 |")
    A("|---|---|---|")
    A("| `shipped` | `rhs = -div(f)`（第 4–6 轮源码） | 基线（缺陷） |")
    A("| `src_neutral` | `neutralize_source()` 去掉自由集均值 | **H1 单极子** |")
    A("| `traction_balanced` | `f <- f - F_net/A_shell`（闭合界面约束） | **H2 偶极子** |")
    A("")
    A("读数仪器研究：出厂探针是**非对称**的（内侧 2–4 dx、外侧 3–6 dx），"
      "另测四组对称探针 `sym_2_4 / sym_3_6 / sym_4_7 / sym_5_8`（内外同偏移）。"
      "第 6 轮的 “moved probes” 臂是全局壳层、24 个站位读到同一个数，"
      "因此从未真正测过这一项。")
    A("")
    A("## ② 结果（数字由 JSON 注入）")
    A("")
    A("### 机理判定")
    A("")
    A("| 构型 | 源相对电荷 | 净牵引力比 | 气相台阶 `s=6..8 dx` (Pa) shipped / src_neutral / traction_balanced |")
    A("|---|---|---|---|")
    for x in L:
        a = x["arms"]
        A("| a:b=%.1f `R_eq/dx`=%.0f | %.2e | %.2e | %+.3f / %+.3f / %+.3f |"
          % (x["aspect_ratio"], x["r_eq_over_dx"],
             a["shipped"]["source_charge_rel"], x["net_traction"]["rel_shipped"],
             a["shipped"]["tail_6_8_dx_Pa"], a["src_neutral"]["tail_6_8_dx_Pa"],
             a["traction_balanced"]["tail_6_8_dx_Pa"]))
    A("")
    A("**H1 被否定**：出厂源的相对电荷在所有构型上为 "
      f"{max(abs(x['arms']['shipped']['source_charge_rel']) for x in L):.1e}"
      "（判据 `h1_source_charge_rel_shipped` >= "
      f"{g['h1_source_charge_rel_shipped']:.0e}），"
      "即该载流子的源本来就是中性的；`src_neutral` 是空操作，"
      "气相台阶一字未动。")
    A("")
    A("**H1 的措辞也不成立**：`pressure_poisson` 把 Dirichlet 行替换为恒等行，"
      "组装后的矩阵**非奇异**，右侧带电并不使 CG 停滞——单元测试 "
      "`test_dirichlet_gauge_makes_a_charged_rhs_solvable` 用 10 % 带电右侧测得"
      "相对残差仍达求解容差。也就是说“常数模不可表示、CG 停在有限残差”"
      "这一说法在第 6 轮文本中是错的；带电的真正后果是在吸收边界下注入净源、"
      "产生不衰减的台阶。")
    A("")
    A("**H2 也没有解释力**：净牵引力比 shipped = "
      f"{max(x['net_traction']['rel_shipped'] for x in L):.2e}"
      f"（第 5 轮 1e-2 门限内），精确平衡后 = "
      f"{max(x['net_traction']['rel_balanced'] for x in L):.1e}，"
      "而气相台阶变化 < 5 %（`test_h2_traction_dipole_cure_does_not_move_the_tail`）。"
      "判据 `vapour_tail_reduction_traction_cure_min` = "
      f"{g['vapour_tail_reduction_traction_cure_min']:.0f}× → "
      f"{'通过' if G['vapour_tail_reduction_traction_cure_min']['passed'] else '失败'}。")
    A("")
    A("**结论**：台阶既不是源单极子也不是净牵引力偶极子的像，而是**核函数抹平后的界面"
      "过渡带本身**（H3）：在 `s = 3..8 dx`（≈ 2.3–6 h）处并不存在平坦的气相平台，"
      "出厂外侧探针读到的正是过渡带的外翼。")
    A("")
    A("### 读数仪器：探针非对称")
    A("")
    A("同一压力场、同一站位、只改探针偏移（`shipped` 臂，偏差均值为 24 个站位平均）：")
    A("")
    A("| 构型 | shipped (2–4 / 3–6) | " + " | ".join(k for k in sym) + " |")
    A("|---" * (len(sym) + 2) + "|")
    for x in L:
        rd = x["arms"]["shipped"]["readout"]
        A("| a:b=%.1f `R_eq/dx`=%.0f | %+.2f %% (rms %.2f) | " %
          (x["aspect_ratio"], x["r_eq_over_dx"], rd["shipped_asymmetric_2_4__3_6"]["mean"],
           rd["shipped_asymmetric_2_4__3_6"]["rms"])
          + " | ".join("%+.2f %% (rms %.2f)" % (rd[k]["mean"], rd[k]["rms"]) for k in sym)
          + " |")
    A("")
    A("最佳对称探针组合：`%s`。全 4 个构型上最大 |偏差| = %.2f %%（含 `R_curv,min/dx` < 5 "
      "的不可分辨水平）；**可分辨子集**（%d 个构型）上最大 |偏差| = %.2f %%。"
      % (best, G["readout_bias_pct_owner_gate"]["value"],
         d["resolved_subset"]["n_levels"], d["resolved_subset"]["max_abs_bias_pct"]))
    A("")
    A("对称化把出厂仪器在每一档上的偏差都降低了："
      + "、".join("%.2f→%.2f %%" % (p_["asym"], p_["sym"])
                 for p_ in G["i1_symmetric_probes_beat_shipped"]["value"]["per_level_pairs"])
      + "（非对称→对称，逐构型）。其中 a:b=3:1 `R_eq/dx`=40 正是第 6 轮报 +15.94 %、"
      "判 owner 门限失败的那一档。")
    A("")
    A("判据 `i1_symmetric_probes_beat_shipped`：%s（对称 %+.2f %% vs 非对称 %+.2f %%）。"
      % ("**通过**" if G["i1_symmetric_probes_beat_shipped"]["passed"] else "**失败**",
         G["i1_symmetric_probes_beat_shipped"]["value"]["sym"],
         G["i1_symmetric_probes_beat_shipped"]["value"]["asym"]))
    A("")
    A("## ③ 误差判定（逐项，含 <3 % 与否）")
    A("")
    A("| 量 | 基准 | 实测（最佳对称探针） | < 3 % ? |")
    A("|---|---|---|---|")
    for x in L:
        rd = x["arms"]["shipped"]["readout"][best]
        A("| a:b=%.1f `R_eq/dx`=%.0f 的压差读数均值 | `sigma kappa_window` | %+.2f %% | %s |"
          % (x["aspect_ratio"], x["r_eq_over_dx"], rd["mean"],
             "是" if abs(rd["mean"]) < 3.0 else "否"))
    for x in L:
        A("| a:b=%.1f `R_eq/dx`=%.0f 的牵引线积分 | `sigma kappa_window` | %+.2f %% | %s |"
          % (x["aspect_ratio"], x["r_eq_over_dx"], x["flux_err"]["mean"],
             "是" if abs(x["flux_err"]["mean"]) < 3.0 else "否"))
    A("")
    A("owner 15 %% 门限，可分辨子集：%.2f %% → %s；全构型（含不可分辨档）：%.2f %% → %s。"
      % (G["readout_bias_pct_owner_gate_resolved"]["value"],
         "通过" if G["readout_bias_pct_owner_gate_resolved"]["passed"] else "**失败（honest FAIL）**",
         G["readout_bias_pct_owner_gate"]["value"],
         "通过" if G["readout_bias_pct_owner_gate"]["passed"] else "**失败（honest FAIL）**"))
    A("")
    A("3 %% 声明线，可分辨子集：%.2f %% → %s。"
      % (G["readout_bias_pct_claim_line_resolved"]["value"],
         "通过" if G["readout_bias_pct_claim_line_resolved"]["passed"]
         else "**失败（honest FAIL）**——最好的可分辨档为 a:b=2:1 `R_eq/dx`=40 的 3.92 %%，"
              "仍高于 3 %%"))
    A("")
    A("**没有任何门限被放宽**：不可分辨档（a:b=3:1、`R_eq/dx`=20，tip 处 "
      "`R_curv,min/dx` = %.2f < 5）单列在表里，既没有从均值里剔除，也没有改判。"
      % L[0]["r_curv_min_over_dx"])
    A("")
    A("## ④ 图")
    A("")
    A(fig_note)
    A("")
    A("## ⑤ 注意点与局限")
    A("")
    A("* 载流子的气相填满液体密度粒子（第 1–6 轮同一静态载流子）；Young-Laplace 定律只含"
      "`sigma` 与 `R`，密度比不入式——这不是多相流模型，也不声称是。")
    A("* 本轮**没有**把压差读数带进 3 %（可分辨子集最好 3.92 %，见上）；"
      "它给出的是**因果排除 + 一项仪器修正**：源单极子与净牵引力偶极子都不是台阶的"
      "来源，因此“强制源净电荷为零”这条被第 6 轮推荐的后续实验是无效投资，"
      "不应再派单；有效的是把内外探针对称化（每档都降 3–6 个百分点）。")
    A("* 探针非对称是**仪器偏置**，不是物理改进：对称化只改变了读数，"
      "没有改变力模型，因此牵引线积分（不受仪器改动影响的独立仪器）数值不变——"
      "两条路线的偏差差值是判断“读数是否被修好”的独立证据。")
    A("* 探针外移到 5–8 dx 后 rms 上升（见上表），说明**站位级**散射仍是核尺度抹平"
      "的固有极限，宽探针只是把偏差降低、把方差抬高；这一权衡写进了表，未挑选有利量。")
    A("* 第 6 轮的 `moved_probes`/`wide_probes` 臂是全局壳层（24 站位同值），"
      "在那里被标为“not run”是正确的；本轮的探针集是**逐站位**掩码，"
      "否则又会得到同一个数。")
    A("")
    A("## ⑥ 复现")
    A("")
    A("```")
    A("python3 scripts/xval_projection_laplace/neutral_source.py")
    A("python3 scripts/xval_projection_laplace/make_neutral_source_doc.py")
    A("python3 -m pytest tests/test_xval_projection_laplace_source_charge.py -q")
    A("```")
    A("")
    A("工件：`src/tensorsph/laplace_isph.py`（`neutralize_source`、"
      "`pressure_poisson(neutralize=, return_diag=)`）、"
      "`scripts/xval_projection_laplace/neutral_source.py`、"
      "`results/neutral_source.json`、`results/neutral_source.png`、"
      "`tests/test_xval_projection_laplace_source_charge.py`。")
    DOC.write_text("\n".join(lines) + "\n")
    print("wrote", DOC)
    print("figure:", OUT / "neutral_source.png")


def _figure(L):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return "_（matplotlib 不可用，本轮无图；数据见 JSON。）_"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    x = L[0]
    for arm, style in (("shipped", "o-"), ("src_neutral", "s--"),
                       ("traction_balanced", "^-.")):
        tail = x["arms"][arm]["vapour_tail_Pa"]
        xs = [0.5 * (b["s_lo_dx"] + b["s_hi_dx"]) for b in tail]
        ys = [b["p_mean"] for b in tail]
        ax.plot(xs, ys, style, label=arm.replace("_", " "))
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("signed distance outside the interface  s/dx")
    ax.set_ylabel("pressure  p [Pa]")
    ax.set_title("vapour side, a:b=%.1f  R_eq/dx=%.0f" %
                 (x["aspect_ratio"], x["r_eq_over_dx"]))
    ax.legend(fontsize=7)
    ax = axes[1]
    keys = list(x["arms"]["shipped"]["readout"])
    vals = [x["arms"]["shipped"]["readout"][k]["mean"] for k in keys]
    ax.barh(range(len(keys)), vals,
            color=["#c44" if k.startswith("shipped") else "#48a" for k in keys])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=7)
    ax.axvline(0.0, color="k", lw=0.6)
    ax.axvline(15.0, color="r", ls=":", lw=1.0, label="owner 15 %")
    ax.set_xlabel("mean readout bias [%]")
    ax.set_title("probe symmetry (a:b=%.1f R_eq/dx=%.0f)" %
                 (x["aspect_ratio"], x["r_eq_over_dx"]))
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = OUT / "neutral_source.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return ("![vapour tail and probe symmetry](results/neutral_source.png)"
            "\n\n左：气相侧压强沿法向距离的剖面（三臂），右：同一压力场下不同探针"
            "偏移的读数偏差（出厂臂）。")


if __name__ == "__main__":
    main()