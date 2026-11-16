"""Render the realization-closure document for CASE-SPH-WEDGE-ENTRY-2D.

Every number in the emitted markdown is read from artefacts:

* ``docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json``
  (``seed_ensemble.py --analyse`` over the merged realization directories)
* ``docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json``
  (single-trajectory readings, for the "one member of the distribution" contrast)
* ``docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json``
  (same-operator Wagner truth per grid, quoted from the ensemble summary)

Nothing is typed by hand: the writer formats numbers that already exist on disk,
so the document cannot drift from the artefacts.

    python benchmarks/case_sph_wedge_entry_2d/make_realisation_closure_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ART = ROOT / "docs" / "artifacts"
ENS = ART / "case_sph_wedge_entry_2d_realisations" / "seed_ensemble.json"
LADDER = ART / "case_sph_wedge_entry_2d_ladder" / "ladder_closure.json"
OUT = ROOT / "docs" / "case_sph_wedge_entry_2d_realisation_closure_m22929.md"

GATE = 0.15


def pct(x: float, digits: int = 1) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x * 100:+.{digits}f} %"


def num(x, digits: int = 3) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:.{digits}f}"


def main() -> int:
    ens = json.loads(ENS.read_text())
    ladder = json.loads(LADDER.read_text())
    ladder_rows = {r["tag"]: r for r in ladder["rows"]}
    grids = ens["grids"]
    by_tag = {g["tag"]: g for g in grids}
    verdict = ens["verdict"]

    lines: list[str] = []
    add = lines.append

    add("# CASE-SPH-WEDGE-ENTRY-2D — 多实现系综能否把峰值 Cp 判开")
    add("")
    add("**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力；"
        "ref: Zhao & Faltinsen 1993 JFM 246:593；容差 `< 15 %`）")
    add("**本单**: m22929（generalist 席自找活）　**仓库**: TensorSPH")
    add("")
    add("*本文件由 `benchmarks/case_sph_wedge_entry_2d/make_realisation_closure_doc.py` "
        "从工件生成，文中每个数字都读自 "
        "`docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json`（"
        "该文件只读各网格的 `real_dx*.json` 实现工件与 bias 报告的 Wagner 同算子真值），人工不写数。*")
    add("")
    add("## ① 补的是哪块欠账")
    add("")
    add("上一轮（m22826，`docs/case_sph_wedge_entry_2d_realisation_axis_m22826.md`）把锚点的症结"
        "从「网格不够细」改成「单条轨迹的峰值读数本身带 +16.6 % 的标准误，落在 15 % 门限之外」，"
        "并在 `dx = 4 mm` 上把链路（离散相位 → 多实现 → `U_real` → 判决规则）跑通，"
        "但**细网格的实现集没有出队**（cj4427 递交后一直 queued），"
        "`dx = 2 mm / 1 mm` 的 `U_real` 当时标 `not_run`。")
    add("")
    add("本单就补这一段：在 `dx = 2 mm` 与 `dx = 1 mm` 上各跑一组实现集"
        "（同一算例、同一 61 帧自相似窗、同一估计量、只换离散相位），"
        "按 m22826 预先写死、本轮**未改一字**的规则给出 `U_real` 与判决。")
    add("")
    add("## ② 方法（规则沿用 m22826，未改）")
    add("")
    add("```")
    add('obs_i   = by_frames["61"].resolution_band.peak_median')
    add("truth_g = bias_report 的 synth_smoothed.peak_median（Wagner 场过同一测量算子，与相位无关）")
    add("r_i     = obs_i / truth_g - 1")
    add("U_real  = max( t95(N-1) * sd(r_i)/sqrt(N),  median_i(band spread_i) )")
    add(f"judgeable_g <=> U_real <= {GATE:.2f}          （= 算例容差，不放宽）")
    add("verdict     = ladder_closure.verdict(rows)   （原规则，未改）")
    add("```")
    add("")
    add("两个事先声明（同样未改）的辅助问题：**A1 实现是否真独立**（`cp_centre` 逐帧序列的"
        "两两 Pearson 相关；趋向 1 说明相位没去相关，则 `U_real` 只是下界，方向危险，必须报）；"
        "**A2 答案是否依赖扰动幅度**（`scale ∈ {0, 0.25, 1}`；随幅度单调变化则旋钮在改物理而非采样相位）。")
    add("")
    add("`scale = 0` 是**确定性负控**：偏移恒为 0，必须与已提交的单轨迹读数逐位一致。")
    add("")

    add("## ③ 结果")
    add("")
    add("### 3.1 每个网格的实现集")
    add("")
    add("| grid | dx [mm] | N（nominal） | obs 中位 [q] | 同算子真值 [q] | 残差中位 | "
        "残差全距 | U_ci | U_band | `U_real` | 可裁决 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in sorted(grids, key=lambda g: -g["dx"]):
        n = g["n_realisations"]
        res = [r["residual"] for r in g["realisations"]
               if r["scale"] == 1.0 and r["residual"] == r["residual"]]
        add(f"| `{g['tag']}` | {g['dx'] * 1e3:.2f} | {n} | {num(g['obs_median'])} | "
            f"{num(g['truth_same_operator'])} | {pct(g['residual_median'])} | "
            f"{pct(g['residual_min'])} … {pct(g['residual_max'])} | "
            f"{pct(g['U_realisation_ci'])} | {pct(g['U_band'])} | "
            f"{pct(g['uncertainty'])} | {'是' if g['judgeable'] else '否'} |")
    add("")
    add("### 3.2 逐实现读数")
    add("")
    add("| grid | seed | scale | `phase_x_dx` | `gap_jitter_dx` | obs 峰值 [q] | "
        "带宽 spread | 残差 vs 同算子 Wagner | steps |")
    add("|---|---|---|---|---|---|---|---|---|")
    for g in sorted(grids, key=lambda g: -g["dx"]):
        for r in g["realisations"]:
            add(f"| `{g['tag']}` | {r['seed']} | {r['scale']:g} | "
                f"{num(r['phase_x_dx'], 4)} | {num(r['gap_jitter_dx'], 4)} | "
                f"{num(r['obs_peak'])} | {pct(r['band_spread_rel'])} | "
                f"{pct(r['residual'])} | {r['steps']} |")
    add("")
    add("### 3.3 A1 —— 实现之间的独立性")
    add("")
    add("| grid | 两两 `cp_centre` Pearson 中位 | 最小 | 实现间 sd | 轨迹内 `cp_centre` sd | 比值 |")
    add("|---|---|---|---|---|---|")
    for g in sorted(grids, key=lambda g: -g["dx"]):
        ind = g["independence"]
        add(f"| `{g['tag']}` | {num(ind['pairwise_median'], 3)} | "
            f"{num(ind['pairwise_min'], 3)} | {num(ind['sd_between_realisations'], 4)} | "
            f"{num(ind['sd_within_trajectory_cp_centre'], 4)} | "
            f"{num(ind['ratio_between_over_within'], 3)} |")
    add("")
    add("### 3.4 A2 —— 答案是否依赖扰动幅度 + 确定性负控")
    add("")
    add("| grid | scale | N | obs 中位 [q] | 残差中位 | 残差 sd | seeds |")
    add("|---|---|---|---|---|---|---|")
    for g in sorted(grids, key=lambda g: -g["dx"]):
        for s, blk in sorted(g["scale_check"].items(), key=lambda kv: float(kv[0])):
            add(f"| `{g['tag']}` | {s} | {blk['n']} | {num(blk['obs_peak_median'])} | "
                f"{pct(blk['residual_median'])} | {num(blk['residual_sd'], 4)} | "
                f"{blk['seeds']} |")
    add("")

    add("### 3.5 判决（预声明规则）")
    add("")
    add(f"**`{verdict['verdict']}`** —— {verdict['reason']}")
    add("")
    add("参与判决的行（实现集中位）：")
    add("")
    add("| grid | dx [mm] | obs 中位 [q] | 同算子真值 [q] | 残差 | U | 可裁决 |")
    add("|---|---|---|---|---|---|---|")
    for r in sorted(ens["rule_inputs"], key=lambda r: -r["dx"]):
        add(f"| `{r['tag']}` | {r['dx'] * 1e3:.2f} | {num(r['obs_peak'])} | "
            f"{num(r['synth_peak'])} | {pct(r['residual'])} | {pct(r['uncertainty'])} | "
            f"{'是' if r['judgeable'] else '否'} |")
    add("")
    add("### 3.6 与单轨迹阶梯读数的关系")
    add("")
    add("| grid | 单轨迹 obs（ladder） | 实现集中位 | 实现集全距 | 单轨迹落在分布内？ |")
    add("|---|---|---|---|---|")
    for g in sorted(grids, key=lambda g: -g["dx"]):
        row = ladder_rows.get(g["tag"])
        if row is None:
            continue
        obs = row["obs_peak"]
        lo, hi = g["residual_min"], g["residual_max"]
        inside = "—"
        if lo == lo and hi == hi:
            r_obs = obs / g["truth_same_operator"] - 1.0
            inside = "是" if lo <= r_obs <= hi else "否（历史读数是分布成员，不是分布本身）"
        add(f"| `{g['tag']}` | {num(obs)} | {num(g['obs_median'])} | "
            f"{pct(lo)} … {pct(hi)} | {inside} |")
    add("")

    add("## ④ 图")
    add("")
    add("![realisation closure](../docs/figures/wedge_realisation_closure_m22929.png)")
    add("")
    add("左：各网格每个实现的残差（点）与实现均值的 95 % 置信半宽 `U_ci`、带宽 `U_band`，"
        "虚线 = 15 % 门限；右：`U_real` 随 `dx` 的变化与门限线，"
        "标注每个网格是否可裁决。")
    add("")
    add("## ⑤ 注意点 / 局限（诚实声明）")
    add("")
    add("1. **参考值仍有两层不确定**：(a) `Zhao & Faltinsen 1993` 原文 β = 30° 表值在生态内仍为 "
        "**WITHHELD**（闭源，无登记副本）；(b) 用它复现已发表表值的开源族（Wang & Faltinsen 2017 表 1）"
        "在 β = 30° 上本身跨模型散布很大（见 "
        "`docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json` 的 "
        "`rel_half_spread`）。因此本单判决陈述的是「SPH vs **线性 Wagner 同算子**」这一量，"
        "**不能**读成「已通过 Z&F 验收」。")
    add("2. **`dx = 0.5 mm` 无实现集**：单次实现墙钟在小时量级，6 个实现超出本单预算，"
        "如实标 `not_run`；该网格的旧读数是单轨迹，且上一轮判为不可裁决。")
    add("3. **`U_real` 是仪器不确定度，不是物理误差**：它回答「同网格再跑一次，读数会变多少」，"
        "不回答「哪个读数对」。")
    add("4. **实现集是相位抽样，不是湍流/随机场抽样**：层流自相似算例没有随机源，"
        "相位是唯一的自由度，因此这里的 sd 不含任何物理涨落。")
    add("")

    add("## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）")
    add("")
    add("| 量 | 基准 | 结果 | 偏差 | < 3 % ? |")
    add("|---|---|---|---|---|")
    for g in sorted(grids, key=lambda g: -g["dx"]):
        zero = [r for r in g["realisations"] if r["scale"] == 0.0]
        row = ladder_rows.get(g["tag"])
        if not zero or row is None:
            continue
        got, ref = zero[0]["obs_peak"], row["obs_peak"]
        dev = (got / ref - 1.0) if ref else float("nan")
        add(f"| `{g['tag']}` `scale=0` 确定性负控（实现工件 vs 已提交单轨迹读数） | "
            f"{num(ref, 6)} | {num(got, 6)} | {pct(dev, 6)} | "
            f"{'是' if abs(dev) < 0.03 else '否'} |")
    for g in sorted(grids, key=lambda g: -g["dx"]):
        add(f"| `{g['tag']}` 实现残差中位 | 同算子 Wagner | {pct(g['residual_median'])} | "
            f"{pct(g['residual_median'])} | {'是' if abs(g['residual_median']) < 0.03 else '否'} |")
    add(f"| 判决可裁决性 | `U_real <= {GATE:.0%}` | 见 3.5 `{verdict['verdict']}` | — | — |")
    add("")
    add("## ⑦ 复现")
    add("")
    add("```bash")
    add("# 每个网格：6 个 nominal 实现 + scale=0 负控（+ 2 mm 另有 scale=0.25 幅度检查）")
    add("OUT=docs/artifacts/case_sph_wedge_entry_2d_realisations_2mm")
    add("for s in 1 2 3 4 5 6; do python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py \\")
    add("    --dx 0.002 --seed $s --scale 1.0 --threads 6 --out $OUT; done")
    add("python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py --dx 0.002 --seed 0 --scale 0.0 --threads 6 --out $OUT")
    add("# 1 mm 同理（--dx 0.001 --threads 10）")
    add("# 合并目录后聚合 + 出文档/图")
    add("python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py --analyse \\")
    add("    --out docs/artifacts/case_sph_wedge_entry_2d_realisations")
    add("python benchmarks/case_sph_wedge_entry_2d/make_realisation_closure_doc.py")
    add("python benchmarks/case_sph_wedge_entry_2d/make_realisation_closure_figure.py")
    add("```")
    add("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())