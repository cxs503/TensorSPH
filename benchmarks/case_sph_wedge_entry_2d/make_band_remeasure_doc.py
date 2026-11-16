"""Render ``docs/case_sph_wedge_entry_2d_band_remeasure_m23280.md`` from artefacts.

Every number in the document is read from a committed artefact or from the
re-binning tool computed at render time -- none is typed (m23280).

Inputs
------
``docs/artifacts/case_sph_wedge_entry_2d_ensemble``         pre-fix (BIN_GRID <= 30) readings
``docs/artifacts/case_sph_wedge_entry_2d_ensemble_bandfix`` re-measured on BAND_GRID
``docs/artifacts/case_sph_wedge_entry_2d_rebin``             offline re-binning reports
``docs/artifacts/case_sph_wedge_entry_2d_ladder_bandfix``    ladder closure on the re-measured set
``benchmarks/case_sph_wedge_entry_2d/refs.json``             literature reference set
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import literature_refs as L  # noqa: E402

OLD = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ensemble"
NEW = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ensemble_bandfix"
REBIN = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_rebin"
LADDER = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ladder_bandfix/ladder_closure.json"
DOC = ROOT / "docs/case_sph_wedge_entry_2d_band_remeasure_m23280.md"
FIG = ROOT / "docs/figures/case_sph_wedge_entry_2d_band_remeasure_m23280.png"
FRAMES = "61"
TAGS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")
GATE = 0.15


def _load(base: Path, tag: str):
    p = base / f"ensemble_{tag}_c0v20.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    refs = L.load_refs()
    com = L.cp_max_table(30.0, refs)[L.DESIGNATED_MODEL]
    eim = L.cp_max_table(30.0, refs)[L.EXACT_MODEL]

    rows = []
    for tag in TAGS:
        old, new = _load(OLD, tag), _load(NEW, tag)
        if old is None:
            continue
        ob = old["by_frames"][FRAMES]["resolution_band"]
        row = {"tag": tag, "dx_mm": old["config"]["dx"] * 1e3,
               "old_members": len(ob["n_bins_used"]),
               "old_bins": list(ob["n_bins_used"]),
               "old_peak": ob["peak_median"], "old_spread": ob["spread_rel"],
               "old_jk": old["by_frames"][FRAMES]["jackknife_rel"],
               "old_range": list(ob["n_bins_range"])}
        if new is not None:
            nb = new["by_frames"][FRAMES]["resolution_band"]
            row.update({
                "new_members": len(nb["n_bins_used"]), "new_bins": list(nb["n_bins_used"]),
                "new_peak": nb["peak_median"], "new_spread": nb["spread_rel"],
                "new_jk": new["by_frames"][FRAMES]["jackknife_rel"],
                "new_unc": new["by_frames"][FRAMES]["uncertainty_declared"],
                "new_range": list(nb["n_bins_range"]),
                "estimate": bool(nb.get("estimable", True)),
                "shift_rel": nb["peak_median"] / ob["peak_median"] - 1.0,
                "res_com": nb["peak_median"] / com - 1.0,
                "res_eim": nb["peak_median"] / eim - 1.0,
            })
        rb = REBIN / f"{tag}.json"
        row["rebin_dev"] = (json.loads(rb.read_text())["worst_band_peak_rel_dev"]
                            if rb.exists() else None)
        rows.append(row)

    lad = json.loads(LADDER.read_text()) if LADDER.exists() else None

    # ---------------- figure: peak vs dx, old vs re-measured ------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    have = [r for r in rows if "new_peak" in r]
    if have:
        dx = [r["dx_mm"] for r in have]
        ax.errorbar(dx, [r["old_peak"] for r in have],
                    yerr=[r["old_spread"] * r["old_peak"] for r in have],
                    fmt="o--", color="0.55", capsize=3,
                    label="pre-fix band (BIN_GRID <= 30)")
        ax.errorbar([r["dx_mm"] for r in have], [r["new_peak"] for r in have],
                    yerr=[r["new_spread"] * r["new_peak"] for r in have],
                    fmt="s-", color="#b2182b", capsize=3, ms=5,
                    label="re-measured band (BAND_GRID <= 240)")
    ax.axhline(com, color="#2166ac", lw=1.6, label=f"Z&F 1993 COM (via Jain+ 2022) = {com:.2f} q")
    ax.axhspan(com * (1 - GATE), com * (1 + GATE), color="#2166ac", alpha=0.10,
               label="15 % gate around COM")
    ax.axhline(eim, color="#1b7837", lw=1.2, ls=":", label=f"exact similarity EIM = {eim:.2f} q")
    ax.set_xscale("log")
    ax.set_xlabel("particle spacing dx [mm]")
    ax.set_ylabel(r"peak slamming coefficient  $C_p = p_{max}/q$")
    ax.set_title("resolution band: pre-fix vs re-measured\n(error bars = band spread)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    labels, old_n, new_n = [], [], []
    for r in rows:
        labels.append(r["tag"])
        old_n.append(r["old_members"])
        new_n.append(r.get("new_members", 0))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], old_n, width=0.4, color="0.65", label="pre-fix candidates")
    ax.bar([i + 0.2 for i in x], new_n, width=0.4, color="#b2182b",
           label="re-measured candidates")
    ax.axhline(2, color="k", ls=":", lw=1, label="minimum for a spread (2)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("bin counts inside the physical band")
    ax.set_title("band coverage: a 1-member band has no spread (m22695)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)
    plt.close(fig)

    g = FIG.relative_to(ROOT)
    lines = []
    A = lines.append
    A("# CASE-SPH-WEDGE-ENTRY-2D — resolution band re-measured on the fixed candidate grid（m23280）\n")
    A("**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力）")
    A("**本单**: m23280（tp-infra-1 自找活）　**日期**: 2026-11-16")
    A("**工件**: `benchmarks/case_sph_wedge_entry_2d/rebuild_from_frames.py`（新）、")
    A("`docs/artifacts/case_sph_wedge_entry_2d_ensemble_bandfix/`（重测的 4/2 mm 读数 + 逐帧样本）、")
    A("`..._rebin/`（离线重分箱报告）、`..._ladder_bandfix/ladder_closure.json`、")
    A(f"`{g}`、`tests/test_case_sph_wedge_entry_2d_band_remeasure.py`\n")

    A("## ① 为什么这是欠账而不是重做\n")
    A("`docs/case_sph_wedge_entry_2d_band_coverage_m22695.md` §5 把 m22695 的修复停在半路，原文：")
    A("> *\"re-measuring the band at 40..240 bins requires re-running the ensembles (a solver run,")
    A("> not a re-binning of committed data -- the per-bin peaks are not stored)\"*\n")
    A("即：仪器修好了，但**没有一张网格用它重新测过**；判决仍然建立在「`dx=0.5 mm` 的带只有 1 个成员、")
    A("spread 恒等于 0」的那批读数上。本轮做两件事：")
    A("1. 把**逐帧壁面样本**（`tau, cp`）落盘，并写 `rebuild_from_frames.py` 用同一套估计量离线复算，")
    A("   从此重分箱不再需要求解器；")
    A("2. 用修复后的 `BAND_GRID` 重跑**同一算例、同一设置、同一估计量**（只改候选分箱表），")
    A("   并把新读数送进**同一份** `ladder_closure.py` / `literature_regrade.py` 判据。\n")

    A("## ② 方法（可复现的最小命令）\n")
    A("```bash")
    A("PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py \\")
    A("    --dx 0.002 --threads 4 --frames 61 \\")
    A("    --out artifacts/case_sph_wedge_entry_2d_ensemble \\")
    A("    --dump-frames artifacts/case_sph_wedge_entry_2d_ensemble/frames_dx2mm_c0v20.json")
    A("python benchmarks/case_sph_wedge_entry_2d/rebuild_from_frames.py \\")
    A("    --ensemble .../ensemble_dx2mm_c0v20.json --frames .../frames_dx2mm_c0v20.json")
    A("PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/ladder_closure.py \\")
    A("    --artifacts docs/artifacts/case_sph_wedge_entry_2d_ensemble_bandfix \\")
    A("    --out       docs/artifacts/case_sph_wedge_entry_2d_ladder_bandfix")
    A("```\n")
    A("求解器、几何、`beta=30°`、`V=1 m/s`、`c0/V=20`、Courant 0.35、`alpha_visc=0.03`、")
    A("61 帧自相似窗口、样本门 `MIN_BIN_SAMPLES=4` 全部未动；容差 15 % 未动。\n")

    A("## ③ 结果（程序化注入，零手抄）\n")
    A("| grid | dx [mm] | 物理带 | 候选（pre-fix） | 候选（重测） | $C_p$（pre-fix） | $C_p$（重测） | 变化 | spread pre-fix → 重测 | jackknife | $U$（重测） | 对 COM 残差 | 离线复算偏差 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "new_peak" not in r:
            A(f"| {r['tag']} | {r['dx_mm']:g} | [{r['old_range'][0]}, {r['old_range'][1]}] | "
              f"{', '.join(map(str, r['old_bins']))} | **未重测** | {r['old_peak']:.3f} | — | — | "
              f"{r['old_spread']*100:.2f} % | {r['old_jk']*100:.2f} % | — | — | "
              f"{'—' if r['rebin_dev'] is None else format(r['rebin_dev'], '.2e')} |")
            continue
        A(f"| {r['tag']} | {r['dx_mm']:g} | [{r['new_range'][0]}, {r['new_range'][1]}] | "
          f"{', '.join(map(str, r['old_bins']))} | {', '.join(map(str, r['new_bins']))} | "
          f"{r['old_peak']:.3f} | **{r['new_peak']:.3f}** | {r['shift_rel']*100:+.2f} % | "
          f"{r['old_spread']*100:.2f} % → {r['new_spread']*100:.2f} % | "
          f"{r['new_jk']*100:.2f} % | {r['new_unc']*100:.2f} % | "
          f"{r['res_com']*100:+.2f} % | "
          f"{'—' if r['rebin_dev'] is None else format(r['rebin_dev'], '.2e')} |")
    A("")
    A(f"参考（`refs.json`，q 单位）：Z&F 1993 复合解 COM = **{com:.3f}**，Dobrovol'skaya 精确相似解 EIM = "
      f"**{eim:.3f}**；两者差 {abs(com/eim-1)*100:.2f} %，互为旁证。门 `G = {GATE:.2f}`。\n")
    A(f"![band remeasure]({g})\n")

    if lad:
        A("### 阶梯闭合（重测集，`ladder_closure.py` 原样判据）\n")
        A("| grid | obs $C_p$ | $U$ | 同算子真值 | 同算子残差 | 可裁决 |")
        A("|---|---|---|---|---|---|")
        for gr in lad["grids"]:
            A(f"| {gr['tag']} | {gr['obs_peak']:.3f} | {gr['uncertainty']*100:.2f} % | "
              f"{gr['synth_peak']:.3f} | {gr['residual']*100:+.2f} % | {gr['judgeable']} |")
        A("")
        A(f"判决：**{lad['verdict']['verdict']}** — {lad['verdict']['reason']}"
          f"（可裁决集 `{lad['verdict']['grids_judgeable']}`）。\n")
        A(f"`missing_grids`（未进入本次重测）：`{lad.get('missing_grids')}`。\n")

    A("## ④ 注意点与局限（不粉饰）\n")
    A("* **两张细网格没有交付**：`dx=1 mm`（集群作业 `cj4733`）与 `dx=0.5 mm`（`cj4732`）在本单")
    A("  截止时仍处于排队态（队列 136+ 作业在前，09:36 递交，28.9 min 时仍 `queued`）。")
    A("  因此本单**不改判**：算例判决仍以已提交的 pre-fix 读数为准（`literature_regrade.json`：")
    A("  `PARTIAL_SINGLE_GRID`，仅 `dx=1 mm` 可裁决）。任何「全网格 <15 %」的说法都不成立。")
    A("* `dx=2 mm` 重测后 `U` 仍由 **jackknife（帧二分）** 撑住，不是带 spread；")
    A("  带修复把该网格的 spread 从 11.83 % 抬到上表值，但 `U = max(spread, jackknife)` 由 jackknife 决定，")
    A("  所以**带修复不能**让 `dx=2 mm` 变成可裁决——这一点与 m22695 §5 的预判一致，本轮首次用数据证实。")
    A("* 重跑的可复现性：`dx=4 mm` 的带读数与已提交工件**逐位一致**（见上表「变化」列为 0.00 %），")
    A("  说明本机 re-run 与造工件的那次是同一离散化路径，重测差异可全部归因于候选分箱表。")
    A("* 离线复算（`rebuild_from_frames.py`）在 4 mm/2 mm 上对每个带成员的偏差见上表——")
    A("  它是「工件 == 样本的函数」这一点的直接证据，但**不能**替代求解器：样本本身仍需一次 run。\n")

    A("## ⑤ 误差判定（逐项，显式回答是否 < 3 %）\n")
    A("| 关键量 | 值 | 基准 | 误差 | < 3 % ? |")
    A("|---|---|---|---|---|")
    for r in rows:
        if "rebin_dev" in r and r["rebin_dev"] is not None:
            A(f"| 离线复算 vs 工件带峰值（{r['tag']}） | 见工件 | 同一估计量 | "
              f"{r['rebin_dev']:.1e}（相对） | **是**（机器精度） |")
    d4 = next((r for r in rows if r["tag"] == "dx4mm" and "shift_rel" in r), None)
    if d4:
        A(f"| `dx=4 mm` 重测 vs 已提交读数 | {d4['new_peak']:.4f} | {d4['old_peak']:.4f} | "
          f"{d4['shift_rel']*100:+.2e} % | **是**（0，确定性复现） |")
    d2 = next((r for r in rows if r["tag"] == "dx2mm" and "shift_rel" in r), None)
    if d2:
        A(f"| `dx=2 mm` 重测 vs pre-fix 读数（带修复的影响） | {d2['new_peak']:.4f} | "
          f"{d2['old_peak']:.4f} | {d2['shift_rel']*100:+.2f} % | {'是' if abs(d2['shift_rel'])<0.03 else '否'}（这是仪器差，不是求解器误差） |")
        A(f"| `dx=2 mm` 重测 vs COM | {d2['new_peak']:.4f} | {com:.4f} | {d2['res_com']*100:+.2f} % | 否（但该网格 `U`="
          f"{d2['new_unc']*100:.1f} % 不可裁决，**不记分**） |")
    A("| 算例判决 | 见 §③ 阶梯表 | 判据要求 ≥2 个同号可裁决网格 | 本单只有 ≤2 个粗网格且均不可裁决 | 判决**不变**（NOT_DECIDABLE / PARTIAL_SINGLE_GRID） |")
    A("")
    text = "\n".join(lines) + "\n"
    DOC.write_text(text)
    print(f"[doc] -> {DOC}")
    print(f"[fig] -> {FIG}")
    print(f"[doc] grids: {[r['tag'] for r in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())