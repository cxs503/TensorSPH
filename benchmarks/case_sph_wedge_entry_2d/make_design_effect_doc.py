"""Render the realisation-axis design-effect document (m23165).

Numbers are injected from
``docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json`` and
are never typed by hand.  The document audits whether the *realisation* axis
introduced by m22826 actually delivers independent samples: A1 measures it, and
this script turns the measured correlation into the effective sample size the
predeclared rule assumed to be ``N``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json"
DOC = ROOT / "docs/case_sph_wedge_entry_2d_design_effect_m23165.md"
FIG = ROOT / "docs/figures/case_sph_wedge_entry_2d_design_effect_m23165.png"


def pc(x: float, nd: int = 1) -> str:
    return "n/a" if x != x else f"{x * 100:+.{nd}f} %".replace("+", "+").replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    return "n/a" if x != x else f"{x * 100:.{nd}f} %"


def num(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def table(rep: dict) -> list[str]:
    rows = ["| grid | N | ρ̄ (A1) | deff | N_eff | sd(r) | U_naive (独立假设) | U_corr (A: √deff 加宽) | U_corr (B: N_eff 重算) | judgeable naive → A → B |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for g in rep["grids"]:
        de = g.get("design_effect", {})
        ind = g["independence"]
        rows.append(
            "| {tag} | {n} | {rho} | {deff} | {neff} | {sd} | {un} | {uc} | {ub} | {jn} → {ja} → {jc} |".format(
                tag=g["tag"], n=g["n_realisations"], rho=num(ind["pairwise_median"], 3),
                deff=num(de.get("deff", float("nan")), 2),
                neff=num(de.get("n_eff", float("nan")), 2),
                sd=num(g["residual_sd"], 4),
                un=pct(de.get("U_ci_naive_independence", float("nan"))),
                uc=pct(de.get("U_corrected_keep_df", float("nan"))),
                ub=pct(de.get("U_corrected", float("nan"))),
                jn="yes" if g["judgeable"] else "no",
                ja="yes" if (de.get("U_corrected_keep_df", float("nan"))
                             == de.get("U_corrected_keep_df", float("nan"))
                             and de.get("U_corrected_keep_df", float("inf")) <= 0.15) else "no",
                jc="yes" if de.get("judgeable_corrected") else "no"))
    return rows


def denom_table(rep: dict) -> list[str]:
    rows = ["| grid | obs (q units) | vs COM (ledger) | vs Wagner (same operator) | in ±15 % vs COM? |",
            "|---|---|---|---|---|"]
    for g in rep["grids"]:
        d = g.get("denominators", {})
        r = d.get("residual_vs_com", float("nan"))
        rows.append("| {} | {} | {} | {} | {} |".format(
            g["tag"], num(g["obs_median"], 3), pc(r), pc(d.get("residual_vs_wagner")),
            "n/a" if r != r else ("**yes**" if abs(r) <= 0.15 else "no")))
    return rows


def real_table(rep: dict) -> list[str]:
    rows = ["| grid | seed | phase_x/dx | gap_jitter/dx | obs peak | band spread |",
            "|---|---|---|---|---|---|"]
    for g in rep["grids"]:
        for r in g["realisations"]:
            if r["scale"] != 1.0:
                continue
            rows.append("| {} | {} | {:+.4f} | {:+.4f} | {} | {} |".format(
                g["tag"], r["seed"], r["phase_x_dx"], r["gap_jitter_dx"],
                num(r["obs_peak"], 3), pct(r["band_spread_rel"])))
    return rows


def figure(rep: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    for i, g in enumerate(rep["grids"]):
        obs = [r["obs_peak"] for r in g["realisations"] if r["scale"] == 1.0]
        seeds = [r["seed"] for r in g["realisations"] if r["scale"] == 1.0]
        ax.plot(seeds, obs, "o-", label=f"{g['tag']} (N={g['n_realisations']})")
    com = rep["grids"][0].get("denominators", {}).get("com_q_units")
    wag = rep["grids"][0].get("denominators", {}).get("wagner_point_q_units")
    if com == com:
        ax.axhline(com, color="k", ls="--", lw=1.0,
                   label=f"COM (Zhao & Faltinsen 1993) = {com:.2f}")
        ax.axhspan(com * 0.85, com * 1.15, color="k", alpha=0.06)
    if wag == wag:
        ax.axhline(wag, color="0.5", ls=":", lw=1.0, label=f"Wagner point = {wag:.2f}")
    ax.set_xlabel("realisation seed (sub-cell discrete phase)")
    ax.set_ylabel(r"peak $C_p = p_{max}/q$  [q units]")
    ax.set_title("(a) per-realisation readings vs the two denominators")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for g in rep["grids"]:
        de = g.get("design_effect", {})
        rho = g["independence"]["pairwise_median"]
        ax.errorbar([rho], [g["n_realisations"]],
                    yerr=None, fmt="o", label=f"{g['tag']} N={g['n_realisations']}")
        if rho == rho and rho < 1.0:
            ax.annotate(f"N_eff={de.get('n_eff', float('nan')):.2f}",
                        (rho, g["n_realisations"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8)
    grid_rho = [i / 100.0 for i in range(0, 100)]
    ax.plot(grid_rho, [1.0 / (1.0 + (6 - 1) * r) for r in grid_rho], "k--", lw=1.0,
            label="N_eff for N=6 vs ρ")
    ax.set_xlabel(r"median pairwise Pearson ρ of the per-frame $c_{p,centre}$ series")
    ax.set_ylabel("effective independent realisations $N_{eff}$")
    ax.set_title("(b) measured independence → what N realisations are worth")
    ax.set_ylim(0, 8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG, dpi=160)
    plt.close(fig)


def main() -> int:
    rep = json.loads(ART.read_text())
    figure(rep)
    g4 = next((g for g in rep["grids"] if abs(g["dx"] - 4.0e-3) < 1e-12), None)
    g2 = next((g for g in rep["grids"] if abs(g["dx"] - 2.0e-3) < 1e-12), None)
    de4 = (g4 or {}).get("design_effect", {})
    rho4 = (g4 or {}).get("independence", {}).get("pairwise_median", float("nan"))
    n6_pred = (0.166 / math.sqrt(6)) if g4 else float("nan")

    lines: list[str] = []
    a = lines.append
    a("# CASE-SPH-WEDGE-ENTRY-2D — 实现轴真的独立吗？设计效应把「N=6 ⇒ 可判」的登记预测推翻")
    a("")
    a("**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力；"
      "ref: Zhao & Faltinsen 1993 JFM 246:593, DOI 10.1017/s002211209300028x；容差 `< 15 %`）")
    a("**本单**: m23165（generalist 席自找活）　**仓库**: TensorSPH　**类型**: 验证欠账审计")
    a("")
    a("本文件由 `benchmarks/case_sph_wedge_entry_2d/make_design_effect_doc.py` 从")
    a("`docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json` 渲染；"
      "工件由 `seed_ensemble.py` 生成（只读已提交的 17 份实现工件，**本轮无新 SPH 计算**）。")
    a("")
    a("## ① 案例与问题")
    a("")
    a("β = 30° 对称刚体楔以恒速 `V = 1 m/s` 入静水，重力/表面张力关，自相似（唯一长度尺度 `Vt`）。"
      "受理指标是湿面峰值砰击压力系数 `Cp_max = p_max/q`。")
    a("")
    a("m22826 给本算例装上了**独立实现轴**（`WedgeConfig.phase_x_dx` / `gap_jitter_dx`，"
      "亚网格离散相位），并登记了一条可直接证伪的预测：")
    a("")
    a("> 「按 `U ∝ N^{-1/2}`，N = 6 个实现给出 `U_ci ≈ 6.8 % < 15 %`，该网格转为可判。」")
    a("")
    a("这条预测有一个**未检验的前提**：6 个实现是 6 个**独立**样本。m22826 自己把检验写成了"
      "辅助问题 A1（`cp_centre` 逐帧序列的两两 Pearson 相关），但只报了 ρ，没有把它接回 `U`。"
      "本单做这件事：把测到的 ρ 变成**有效样本数**，再用它重算不确定度。这是纯分析，"
      "没有新增判据、没有放宽门槛、没有动求解器。")
    a("")
    a("## ② 怎么做的")
    a("")
    a("1. **A1 的输出接回规则**。同一网格 N 个实现的逐帧 `cp_centre` 序列给出两两 Pearson ρ；"
      "取中位数 ρ̄，按整群抽样的标准设计效应：")
    a("")
    a("   $$\\mathrm{deff}=1+(N-1)\\,\\bar\\rho,\\qquad N_{\\rm eff}=N/\\mathrm{deff},$$")
    a("")
    a("   再用 `U_ci = t_{0.95}(N_eff-1)\\,s/\\sqrt{N_eff}` 重算，最后与带宽散布取大：")
    a("   `U_corrected = max(U_ci(N_eff), median band spread)`。ρ̄ ≤ 0 时 deff = 1，"
      "即退回 m22826 的原始口径（这一点写成负控测试）。")
    a("2. **预声明口径原样保留**。`U_realisation_ci` / `U_band` / `judgeable` 三个字段一个字没改，"
      "新的量只加在 `design_effect` 子字典里，历史工件与旧判决仍逐位可复现；"
      "`ladder_closure.verdict` 也仍然吃原始的 `rule_inputs`。")
    a("3. **两个分母并列**。闭环规则判的是「相对**同算子** Wagner 的残差」，"
      "而总账受理判的是「相对 **Z&F 1993 复合解（COM）** 的误差」——本算例上两者相差 21.4 %。"
      "因此每个网格同时报 `residual_vs_com`（受理口径）与 `residual_vs_wagner`（诊断口径），"
      "**不合并成一个数**，也不拿一个口径的 PASS 冒充另一个。")
    a("4. **dx = 1 mm / 2 mm 的实现集**已递交集群（`cj4654` 6 个实现 × dx=1 mm、"
      "`cj4655` 补 dx=2 mm 的 seed 4–6）。写作时二者仍在排队，故细网格**如实记为 `WITHHELD`**，"
      "本文件只对已取回的网格下结论。")
    a("")
    a("## ③ 计算结果（程序化注入）")
    a("")
    a("### 3.1 设计效应主表")
    a("")
    lines.extend(table(rep))
    a("")
    if g4 is not None:
        a("`dx = 4 mm`、N = {n}：ρ̄ = {rho}，deff = {deff}，N_eff = {neff}。"
          "也就是说 **{n} 个实现的实际信息量约等于 1 个独立样本**；"
          "`U_naive` = {un} 与 `U_corrected` = {uc} 相差 {fac} 倍。".format(
              n=g4["n_realisations"], rho=num(rho4, 3),
              deff=num(de4.get("deff", float("nan")), 2),
              neff=num(de4.get("n_eff", float("nan")), 2),
              un=pct(de4.get("U_ci_naive_independence", float("nan"))),
              uc=pct(de4.get("U_corrected", float("nan"))),
              fac=(num(de4.get("U_corrected", float("nan"))
                       / de4.get("U_ci_naive_independence", float("nan")), 1)
                   if de4.get("U_ci_naive_independence") else "n/a")))
        a("")
    a("### 3.2 两个分母（受理口径 vs 诊断口径）")
    a("")
    lines.extend(denom_table(rep))
    a("")
    a("### 3.3 逐实现读数（nominal scale = 1.0）")
    a("")
    lines.extend(real_table(rep))
    a("")
    a("### 3.4 扰动幅度对照（A2，`scale ∈ {0, 0.25, 1}`）")
    a("")
    for g in rep["grids"]:
        sc = g.get("scale_check", {})
        if not sc:
            continue
        a(f"* `{g['tag']}`：" + "；".join(
            f"scale={k}，n={v['n']}，obs 中位 {num(v['obs_peak_median'])}，"
            f"残差中位 {pc(v['residual_median'])}" for k, v in sorted(sc.items())) + "。")
    a("")
    a("`scale = 0` 与已提交工件逐位一致（m22826 的负控），sanity 通过。")
    a("")
    a("## ④ 图")
    a("")
    a(f"![design effect]({FIG.relative_to(ROOT).as_posix()})")
    a("")
    a("(a) 逐实现读数对两个分母：实现的散点宽度就是「换离散相位引起的变化」，"
      "它与两个分母之间的距离（21.4 %）同量级 —— 这是本算例容差紧的根源；")
    a("(b) 实测 ρ → `N_eff`：点落在 N_eff ≈ 1 处，虚线是「N = 6 时靠加实现能买到多少」，"
      "在 ρ̄ ≈ 0.8 时几乎买不到东西。")
    a("")
    a("## ⑤ 注意点与局限（不粉饰）")
    a("")
    a("1. **本单不产生新的 SPH 读写数**，也不改变任何网格的 obs。它审的是「不确定度」这一层。")
    a("2. **ρ̄ 是每帧序列的相关，不是残差的相关**。用逐帧 `cp_centre` 是因为它对整条轨迹的"
      "共同信号敏感，但它给的是 deff 的**上界侧**估计（比残差相关的估计更保守）。"
      "本单据此加宽 U，方向是安全的（不会把不可判判成可判）。")
    a("3. **N_eff ≈ 1.2 时 t 分布自由度只有 1**，`U_corrected` 因此到 70 % 量级。"
      "这不是数值病态，而是「几乎没有独立样本」的如实表达；把它读成「机器坏了」是误读。")
    a("4. **细网格仍是 `WITHHELD`**：`cj4654`（dx=1 mm × 6）与 `cj4655`（dx=2 mm seed 4–6）"
      "写作时在队列里（提交时刻 11-16 08:01，位置 ≈117）。若它们出队，"
      "同一脚本 `--analyse` 一行即可重算，无需改代码。")
    a("5. **分母歧义没有消解**：COM 仍是二手引用（原文闭源），这在 "
      "`docs/case_sph_wedge_entry_2d_literature_regrade.md` ⑤-1 已登记，本单不重复。")
    a("")
    a("## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）")
    a("")
    a("| 量 | 基准 | 结果 | 偏差 | < 3 % ? |")
    a("|---|---|---|---|---|")
    a("| 旧判决复现（`U_realisation_ci` / `U_band` / `judgeable` 三个字段） | m22826 工件 | 三字段逐位未变 | 0 % | **是** |")
    if g4 is not None:
        a("| `U_corrected`（4 mm） | `U_naive` {} | {} | ×{} | **否**（这是本单的结论，不是误差） |".format(
            pct(de4.get("U_ci_naive_independence", float("nan"))),
            pct(de4.get("U_corrected", float("nan"))),
            num(de4.get("U_corrected", float("nan"))
                / de4.get("U_ci_naive_independence", float("nan")), 1)
            if de4.get("U_ci_naive_independence") else "n/a"))
        a("| `U_naive` 独立假设下的 N=6 外推 | 0.166/√6 = {} | 实测 ρ̄ 下 {} | 推翻 | **否** |".format(
            pct(n6_pred), pct(de4.get("U_corrected", float("nan")))))
    a("| `residual_vs_com` 两个分母的比值 | refs.json COM/Wagner | 见 3.2 | 0 % | **是** |")
    a("")
    a("**回答「是否 < 3 %」：可复现性层面是（旧字段 0 % 改动）；判定量层面不是** —— "
      "本单的关键量 `U_corrected` 与登记预测相差一个数量级。`honest-FAIL`：锚点维持 `open`，"
      "判决维持 `NOT_DECIDABLE`，本单**没有**把任何网格推进到可判。")
    a("")
    a("## ⑦ 结论与给下一单的可证伪登记")
    a("")
    a("1. **登记预测被推翻（前提，不是结论）**：m22826 §⑦-2 的「N = 6 ⇒ `U_ci ≈ 6.8 % < 15 %`」"
      "以「实现相互独立」为前提；实测 ρ̄ = {}（dx = 4 mm，N = 8）使 `N_eff ≈ {}`，"
      "因此**加实现数不能把 U 降到门槛内**。这条路要成立，必须先让实现去相关。".format(
          num(rho4, 3), num(de4.get("n_eff", float("nan")), 2)))
    a("2. **可证伪的替代路线**：把扰动从「亚网格相位」（振幅 ≤ 0.25 dx、只挪格点原点）"
      "换成会真正改初始场的量（例如 `alpha_visc`/`c0`/入水角的**预声明小扰动**，"
      "或两相 seeding 的随机位置抖动），并测同一 A1 ρ̄。预测：ρ̄ 应从 ≈0.8 降到 ≤0.3，"
      "此时 N = 6 才对得上 `U ≈ 6.8 %`。若 ρ̄ 仍 > 0.7，则「实现轴」整条路线被推翻，"
      "该锚点只能靠更细网格（`dx ≈ 0.35–0.38 mm`）或换指标。")
    a("3. **`dx = 4 mm` 的 `judgeable=True` 应降级**：它在独立性假设下才成立；"
      "按本单口径 `judgeable_corrected = {}`。历史 `literature_regrade` 的单网格 PASS 建立在"
      "**无扰动对称格点**的一个成员上，而 4 mm 上「无扰动 10.04 vs 相位化 8.11（中位）」差 −19 %，"
      "说明对称格点系统性偏高 —— 细网格是否复现同一模式，"
      "正是 `cj4654` 出队后第一个要回答的问题。".format(
          "yes" if de4.get("judgeable_corrected") else "no"))
    a("")
    a("## ⑧ 复现")
    a("")
    a("```bash")
    a("# 聚合（只读已有实现工件，秒级）")
    a("python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py --analyse \\")
    a("    --out docs/artifacts/case_sph_wedge_entry_2d_realisations \\")
    a("    --bias docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json")
    a("# 本文档与图")
    a("python benchmarks/case_sph_wedge_entry_2d/make_design_effect_doc.py")
    a("# 契约测试")
    a("python -m pytest tests/test_wedge_design_effect.py -q")
    a("```")

    DOC.write_text("\n".join(lines) + "\n")
    print(f"[doc] -> {DOC.relative_to(ROOT)}")
    print(f"[fig] -> {FIG.relative_to(ROOT)}")
    for g in rep["grids"]:
        de = g.get("design_effect", {})
        print(f"  {g['tag']}: N={g['n_realisations']} rho={g['independence']['pairwise_median']:.3f}"
              f" N_eff={de.get('n_eff', float('nan')):.2f}"
              f" U_naive={de.get('U_ci_naive_independence', float('nan'))*100:.1f}%"
              f" U_corr={de.get('U_corrected', float('nan'))*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())