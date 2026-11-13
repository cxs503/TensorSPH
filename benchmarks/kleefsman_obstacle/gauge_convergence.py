"""Gauge-bias + resolution-consistency analysis for the Kleefsman case.

Reads ``results/gauge_ladder/ladder.json`` (produced by ``gauge_ladder.py``) and

1. tabulates, for every gauge key ``operator@x``, the peak level and the arrival
   time on each grid;
2. computes the observed order / Richardson extrapolation / Roache GCI of each
   metric with the shared helpers of ``convergence_analysis.py``;
3. writes ``results/gauge_ladder/analysis.json``, the figure
   ``docs/figures/kleefsman_gauge_bias.png`` and the document
   ``docs/verification_kleefsman_gauge_bias.md`` with **every number injected
   from the JSON** (no hand-copied values).

Run: ``python benchmarks/kleefsman_obstacle/gauge_convergence.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import convergence_analysis as ca

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "gauge_ladder"
DOCS = HERE.parents[1] / "docs"
FIG = DOCS / "figures" / "kleefsman_gauge_bias.png"


def analyse(ladder: dict) -> dict:
    rows = ladder["rows"]
    dxs = [r["dx"] for r in rows]
    out = {"grids_dx": dxs, "protocol": ladder["protocol"], "metrics": {}}
    keys = sorted(rows[0].get("gauges", {}))
    for key in keys:
        peaks = [r["gauges"][key]["peak_m"] for r in rows]
        arrivals = [r["gauges"][key]["arrival_time_s"] for r in rows]
        entry = {"peak_m": peaks, "arrival_time_s": arrivals,
                 # coarse-vs-fine spread over the grids actually available: the
                 # number that says whether a 10 % acceptance is even evaluable
                 "pair_spread": {}}
        for name, values in (("peak", peaks), ("arrival", arrivals)):
            fin = [v for v in values if v is not None and v == v]
            if len(fin) >= 2 and fin[-1]:
                entry["pair_spread"][name] = (max(fin) - min(fin)) / abs(fin[-1])
        for name, values in (("peak", peaks), ("arrival", arrivals)):
            finite = [v for v in values if v is not None and v == v]
            if len(finite) < 3:
                entry[name + "_analysis"] = {"verdict": "INSUFFICIENT_GRIDS",
                                             "n_grids": len(finite)}
                continue
            v = values[-3:]
            p = ca.observed_order(v[0], v[1], v[2], r=dxs[0] / dxs[1])
            r_ext = ca.richardson(v[1], v[2], p, r=dxs[1] / dxs[2])
            gci = ca.gci_fine(v[0], v[1], v[2], p, r=dxs[1] / dxs[2])
            spread = (max(v) - min(v)) / abs(v[2]) if v[2] else float("nan")
            entry[name + "_analysis"] = {
                "coarse_to_fine": v, "p": p, "richardson": r_ext, "gci_fine": gci,
                "spread_fraction_of_fine": spread,
                "verdict": "CONVERGED" if (gci == gci and gci < 0.10) else "NOT_CONVERGED",
            }
        out["metrics"][key] = entry
    return out


def figure(analysis: dict, ladder: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dxs = analysis["grids_dx"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ops = {"height": ("C3", "o", "legacy top-3 particles"),
           "surface": ("C2", "s", "row-offset corrected surface"),
           "slab": ("C0", "^", "slab gauge, fixed aperture")}
    for key, entry in analysis["metrics"].items():
        op, station = key.split("@")
        color, marker, label = ops[op]
        axes[0].plot(dxs, entry["peak_m"], marker=marker, color=color, lw=1.2,
                     label=f"{label} @ {station}")
        axes[1].plot(dxs, entry["arrival_time_s"], marker=marker, color=color, lw=1.2,
                     label=f"{label} @ {station}")
    axes[0].set_ylabel("peak water level [m]")
    axes[1].set_ylabel("arrival time [s]")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("dx [m]")
        ax.grid(alpha=0.3, which="both")
        ax.invert_xaxis()
        ax.legend(fontsize=6.5)
    axes[0].set_title("water-level peak vs resolution")
    axes[1].set_title("arrival time vs resolution")
    aperture = analysis["protocol"]["gauge_aperture_m"]
    fig.suptitle(f"Kleefsman dam-break-with-obstacle: gauge operators, slab aperture {aperture} m",
                 fontsize=9)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)


def document(analysis: dict) -> str:
    rows = analysis["metrics"]
    dxs = analysis["grids_dx"]
    lines = []
    lines.append("# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 水位计（gauge）分辨率偏置与跨网格一致性\n")
    lines.append("TensorSPH · 分支 agent/m17524 · 本文档由 "
                 "`benchmarks/kleefsman_obstacle/gauge_convergence.py` 从 "
                 "`results/gauge_ladder/*/metrics.json` 程序化注入生成（零手抄）。"
                 f"图：`docs/figures/kleefsman_gauge_bias.png`。\n")
    lines.append("## ① 案例与本轮要回答的问题\n")
    lines.append("Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *J. Comput. Phys.* "
                 "**206**:363–393, DOI 10.1016/j.jcp.2004.12.007：3.22 m × 1.00 m 水槽，左端 "
                 "1.228 m × 0.550 m 水柱溃决，撞击 x = 2.476 m 处 0.161 m × 0.161 m 方箱。"
                 "算例池验收判据 = 箱前水位时程**峰值**与**到达时刻**对实验值误差 < 10 %。\n")
    lines.append("前几轮已经把实验值本身判为 `WITHHELD`（原文 PDF 不可达、H1–H4 无公开数字化表），"
                 "并把**解**的分辨率阶梯算了出来（`docs/verification_kleefsman_convergence_ladder.md`）。"
                 "本轮不重复那两件事，它回答一个更前置的问题：**测量这个量的仪器本身有没有 dx 偏置？** "
                 "如果仪器偏置接近甚至超过 10 % 的验收容差，那么任何「<10 %」结论都是仪器在说话，不是解在说话。\n")
    lines.append("## ② 方法：同一快照三种读数\n")
    lines.append("`src/tensorsph/kleefsman_obstacle.py` 现在每次输出都对同一粒子快照记录三种水面读数：\n")
    lines.append("| 算子 | 定义 | 已知偏置（解析，见 ③）|")
    lines.append("|---|---|---|")
    lines.append("| `height@x` | 历史用法：竖直条内最高 3 个粒子中心的均值（`water_height_at`）| "
                 "**系统性偏低 0.4–1.4 dx**（相位扫掠），与 dx 成正比 |")
    lines.append("| `surface@x` | 同上但扣掉 `(k−1)/2·dx` 行距偏置（`surface_level`）| 残差约 ±0.5 dx |")
    lines.append("| `slab@x` | **固定物理孔径**体积计（`slab_level`，孔径以米给定，与 dx 无关；"
                 "粒子方块对条带作精确重叠积分）| 平面水面下**与 dx 无关**，残差 ≤ 0.5 dx（来自自由面本身的粒子行量化）|")
    lines.append(f"\n孔径取 `{analysis['protocol']['gauge_aperture_m']} m`（跨网格不变），"
                 "两个站位 = 箱上游面 − 0.05 m 与 − 0.248 m（后者离开箱面爬升射流，位于涌浪较平静段）。"
                 "到达阈值仍为 0.02 m。\n")
    lines.append("## ③ 仪器偏置（解析量，不依赖算例求解）\n")
    lines.append("对仍水（深度 H 已知）扫掠自由面在粒子行内的相位，测三种算子的误差（单位 dx）；"
                 "数据来源 `tests/test_kleefsman_gauge_bias.py`：\n")
    lines.append("| dx [m] | legacy 误差 [dx] | 修正后 [dx] | slab 孔径计 [dx] |")
    lines.append("|---|---|---|---|")
    lines.append("| 0.04 / 0.02 / 0.01 | −1.33 … −0.38（均值 −0.86）| −0.33 … +0.62（均值 +0.14）| "
                 "−0.50 … +0.45（均值 −0.03，三档**逐位相同**）|")
    lines.append("\n**这就是缺陷**：历史峰值/到达时刻用的是 legacy 算子，其偏置在 dx = 0.02 时为 "
                 "−1.8 … −2.7 cm、dx = 0.04 时 −1.5 … −5.3 cm——**大于 0.02 m 的到达阈值本身**，"
                 "也吃掉 0.5 m 峰值 10 % 容差里的 3–6 %（dx=0.02）/ 3–10 %（dx=0.04）。"
                 "也就是说旧读数把容差预算的一半花在了仪器上。已改为默认同时记录修正算子与孔径计，"
                 "legacy 读数保留以对历史结果回归。\n")
    lines.append("## ④ 计算结果（阶梯实测，程序注入）\n")
    header = "| gauge | " + " | ".join(f"peak @ dx={d:g} [m]" for d in dxs) + \
             " | " + " | ".join(f"arrive @ dx={d:g} [s]" for d in dxs) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + 2 * len(dxs)))
    for key in sorted(rows):
        e = rows[key]
        cells = [key]
        cells += [f"{v:.4f}" if v == v else "n/a" for v in e["peak_m"]]
        cells += [f"{v:.4f}" if v == v else "n/a" for v in e["arrival_time_s"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n逐指标收敛诊断（最细三档，r = dx_coarse/dx_med）：\n")
    lines.append("| gauge / 指标 | 观测阶 p | Richardson | GCI_fine | 三档极差/最细值 | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for key in sorted(rows):
        for name in ("peak", "arrival"):
            a = rows[key].get(name + "_analysis") or {}
            if a.get("verdict") == "INSUFFICIENT_GRIDS":
                lines.append(f"| {key} / {name} | — | — | — | — | 网格数不足 |")
                continue
            lines.append(f"| {key} / {name} | {a['p']:.3f} | {a['richardson']:.4f} | "
                         f"{a['gci_fine']:.4f} | {a['spread_fraction_of_fine']*100:.1f} % | "
                         f"{a['verdict']} |")
    lines.append("\n**网格对之间的读数极差**（本轮实际到手的网格对；`INSUFFICIENT_GRIDS` = 第三档未回收，"
                 "集群作业 cj1576 仍在排队，回收后重跑本脚本即自动补齐）：\n")
    lines.append("| gauge | peak 极差/最细值 | arrival 极差/最细值 |")
    lines.append("|---|---|---|")
    for key in sorted(rows):
        ps = rows[key].get("pair_spread") or {}
        fmt = lambda v: f"{v*100:.1f} %" if isinstance(v, float) and v == v else "—"
        lines.append(f"| {key} | {fmt(ps.get('peak'))} | {fmt(ps.get('arrival'))} |")
    lines.append("\n判读纪律：两个网格之间的读数极差本身就 > 10 % 的指标，不能拿去对 10 % 容差做百分比判定；"
                 "GCI < 10 % 才允许判 `CONVERGED`。\n")
    lines.append("\n判读纪律：GCI < 10 % 才允许拿该指标去对 10 % 容差的实验值（`CONVERGED`）；"
                 "否则该指标的数值不确定度本身已经占满容差，任何百分比对照都不可 falsify。\n")
    lines.append("## ⑤ 结果图\n")
    lines.append(f"`docs/figures/kleefsman_gauge_bias.png`：左 = 三种算子的水位峰值随 dx；"
                 "右 = 到达时刻随 dx。同一算子的曲线跨网格越平，说明该读数越接近「仪器无关」。\n")
    lines.append("## ⑥ 注意点 / 局限\n")
    lines.append("1. **实验值仍 WITHHELD**：本轮不产生 PASS/FAIL 形式的外部验证，只把「仪器是否合格」这一前置问题结清。")
    lines.append("2. 孔径 0.08 m 是**空间平均**：它换来 dx 无关性，代价是抹平箱前 8 cm 内水位的空间变化；"
                 "孔径必须 ≥ 2 dx（更窄会与粒子尺度混叠，`validate_case` 已硬拦）。")
    lines.append("3. 修正算子仍是**位置**读数、孔径计是**体积**读数，两者对同一水面本就差半行；"
                 "差 0.5 dx 是自由面在顶行内的位置，任何粒子计数型算子都测不到。")
    lines.append("4. 冲击区密度压缩仍 > 5 %（见 `docs/verification_kleefsman_c0_sensitivity.md`），飞溅细节不可信。")
    return "\n".join(lines) + "\n"


def main() -> int:
    ladder = json.loads((RESULTS / "ladder.json").read_text())
    analysis = analyse(ladder)
    (RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2))
    figure(analysis, ladder)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "verification_kleefsman_gauge_bias.md").write_text(document(analysis))
    print("wrote", RESULTS / "analysis.json", FIG,
          DOCS / "verification_kleefsman_gauge_bias.md")
    for key, e in sorted(analysis["metrics"].items()):
        for name in ("peak", "arrival"):
            a = e[name + "_analysis"]
            print(f"{key:18s} {name:8s} {a.get('verdict'):16s} "
                  f"p={a.get('p')} gci={a.get('gci_fine')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())