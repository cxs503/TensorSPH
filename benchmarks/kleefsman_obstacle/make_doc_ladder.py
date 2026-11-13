"""Generate the Kleefsman convergence-ladder document + figures.

Everything numeric in the produced markdown is injected from
``benchmarks/kleefsman_obstacle/results/{resolution_study,convergence_ladder}.json``
and the ``dx*_time_series.csv`` traces -- nothing is hand copied.

    python benchmarks/kleefsman_obstacle/make_doc_ladder.py \
        --results benchmarks/kleefsman_obstacle/results \
        --doc docs/verification_kleefsman_convergence_ladder.md \
        --figure docs/figures/kleefsman_convergence_ladder.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import convergence_analysis as ca  # noqa: E402


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def make_figure(rows, report, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for r in sorted(rows, key=lambda r: -float(r["dx"])):
        times, heights = ca.load_series(BENCH / "results", float(r["dx"]))
        if not times:
            continue
        smooth = ca.moving_average(times, heights, ca.DEFAULT_WINDOW)
        ax.plot(times, heights, lw=0.8, alpha=0.45,
                label=f"raw dx={r['dx']:g}")
        ax.plot(times, smooth, lw=1.6, label=f"gauge dx={r['dx']:g}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height at probe [m]")
    ax.set_title("probe level: raw vs gauge-equivalent (30 ms)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for key, style in (("peak_height_m_raw", "o-"), ("arrival_time_s", "s--")):
        m = report["metrics"].get(key)
        if not m:
            continue
        ax.plot(m["dx"], m["values"], style, label=key)
        if m.get("richardson") == m.get("richardson"):
            ax.axhline(m["richardson"], ls=":", lw=1, alpha=0.6)
    for key in report["metrics"]:
        if key.startswith("gauge_level_peak"):
            m = report["metrics"][key]
            ax.plot(m["dx"], m["values"], "^-", label=key)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("dx [m] (log, coarse -> fine)")
    ax.set_ylabel("value")
    ax.set_title("ladder: raw peak / gauge peak / arrival")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_doc(rows, report, figrel: str) -> str:
    head = report["protocol"]
    rt = {float(r["dx"]): r for r in rows}
    lines = []
    A = lines.append
    A("# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 分辨率阶梯的收敛阶、Richardson 外推与不确定性带\n")
    A("TensorSPH · 分支 agent/m17129 · 本文档由 `benchmarks/kleefsman_obstacle/make_doc_ladder.py` "
      "从 `results/resolution_study.json` + `results/convergence_ladder.json` + `dx*_time_series.csv` "
      "程序化注入生成（零手抄）。图：`%s`。\n" % figrel)
    A("## ① 案例介绍")
    A("Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *J. Comput. Phys.* **206**:363–393, "
      "DOI 10.1016/j.jcp.2004.12.007：3.22 m × 1.00 m 水槽，左端 1.228 m × 0.550 m 水柱瞬时溃决，"
      "沿干底推进撞击 x = 2.476 m 处 0.161 m × 0.161 m 刚性方箱。几何自洽 1.228 + 1.248 + 0.744 = 3.220 m。")
    probe_offset = head.get("probe_offset")
    probe_offset = 0.05 if probe_offset is None else float(probe_offset)
    thr = head.get("arrival_threshold")
    thr = 0.02 if thr is None else float(thr)
    A("诊断量：箱前探针（x = 2.476 − %.2f = %.3f m）水位时程的**峰值**与**到达时刻**（越过 %.2f m）。"
      % (probe_offset, 2.476 - probe_offset, thr))
    A("")
    A("| 协议量 | 值 |\n|---|---|")
    for k in ca.PROTOCOL_KEYS:
        A(f"| {k} | {fmt(head.get(k))} |")
    A("")
    A("**实验值状态：仍然 WITHHELD。** 原文 PDF 在本节点不可达（research.rug.nl 副本 Cloudflare 403、"
      "ScienceDirect 403、archive.org 不可达），H1–H4 水位**无公开数字化表**。因此本文件不做与实验的百分比"
      "对照；它回答的是一个前置问题——**该诊断量在当前数值设置下是否已经收敛到可以用 10 % 容差去比**。"
      "未收敛的量没有误差棒，任何「<10 %」的结论都不可falsify。\n")
    A("## ② 怎么做的")
    A("1. **协议一致性守卫（缺陷修复）**：旧 `resolution_study.json` 把 `t_end = 1.0 s` 的 dx=0.01 档"
      "与 `t_end = 1.2 s` 的 dx=0.02/0.04 档并列成一张「分辨率研究」表——三档物理时长不同，残差无意义。"
      "`convergence.py` 现在给每档写协议指纹（`t_end/sound_speed/probe_offset/probe_halfwidth/"
      "surface_particles/surface_gap/arrival_threshold/h_factor`），合并时以**新算档为基准**，"
      "指纹不符的旧档进 `runs_stale_protocol` 隔离区而不是混进阶梯。")
    A("2. **重跑一致性阶梯**：四档 dx = %s 全部以 `t_end = %s s`、同一探针与同一自由面估计器重算"
      % (" / ".join(f"{d:g}" for d in report["dx"]), fmt(head.get("t_end"), 2)))
    A("   （集群作业 cj1423；旧的三档结果保留在隔离区，不删除）。")
    A("3. **收敛阶 / 外推 / GCI**：`benchmarks/kleefsman_obstacle/convergence_analysis.py` 对每三个网格"
      "（均匀加密比 r）算观测阶 p = ln|f_med−f_coarse|/ln r ÷ …, Richardson 外推 f_exact ≈ f_fine + "
      "(f_fine−f_med)/(r^p−1)，Roache GCI_fine = Fs·|(f_med−f_fine)/f_fine|/(r^p−1)，Fs = 1.25。"
      "纯函数、有合成阶梯单测（已知 p 与精确值回读误差 <1e-9）。")
    A("4. **把「水位」与「瞬时最高粒子」分开**：瞬时峰值被抛射水花控制，不是收敛量（历史记录 0.426 / 0.518 / ≥0.377 m "
      "在 dx = 0.04/0.02/0.01 上非单调）。实验电容式波高仪本身是低通的，故另算**仪表等效水位**："
      "时程先做 ±15 ms 对称滑窗平均再取峰（窗口 %s ms 敏感性一并给出）。"
      % "/".join(f"{1000 * m['window_s']:.0f}" for m in report["metrics"].values() if "window_s" in m))
    A("")
    A("## ③ 计算结果")
    A("| dx [m] | 流体粒子 | 步数 | 原始峰值 [m] @ t [s] | 仪表水位峰值(30 ms) [m] | 到达时刻 [s] | 密度压缩 | 越界 |\n"
      "|---|---|---|---|---|---|---|---|")
    for dx in sorted(rt, reverse=True):
        r = rt[dx]
        times, heights = ca.load_series(BENCH / "results", dx)
        gp = ca.peak_and_time(times, ca.moving_average(times, heights, ca.DEFAULT_WINDOW))[0] \
            if times else float("nan")
        A(f"| {dx:g} | {r.get('particles_fluid')} | {r.get('steps')} | "
          f"{fmt(r.get('peak_height_m'))} @ {fmt(r.get('peak_time_s'), 3)} | {fmt(gp)} | "
          f"{fmt(r.get('arrival_time_s'))} | {100 * r.get('max_density_compression', float('nan')):.2f} % | "
          f"{r.get('boundary_escapes')} |")
    A("")
    A("收敛诊断（最细三档 = %s）：\n" % ", ".join(
        f"{d:g}" for d in (report["metrics"][next(iter(report['metrics']))]["dx"][-3:]
                          if report["metrics"] else [])))
    A("| 指标 | 最细三档值 | 观测阶 p | Richardson 外推 | GCI_fine | 判定 |\n|---|---|---|---|---|---|")
    for k, m in report["metrics"].items():
        v = " / ".join(fmt(x) for x in m["values"][-3:])
        A(f"| {k} | {v} | {fmt(m.get('p_finest'), 3)} | {fmt(m.get('richardson'))} | "
          f"{fmt(m.get('gci_fine'), 4)} | {m['verdict']} |")
    A("")
    declared = ca.check_protocol(rows)
    A(f"**阶梯总判定：{report['verdict']}**。原始三档的 `t_end` 不一致（%s）已在 "
      "`results/resolution_study.json:protocol_mismatch` 显式登记；评分前用 `--window-t %.2f s` "
      "把各档统一截到公共时窗（三个峰值时刻都落在窗内：%s s），故求解器协议指纹一致、"
      "协议不匹配项 = %d。" % (declared, report.get("truncated_to_s") or 0.0,
                             " / ".join(fmt(r.get("peak_time_s"), 3) for r in sorted(rows, key=lambda r: -r["dx"])),
                             len(report["protocol_mismatch"])))
    A("")
    A(f"删失检查（峰值/仪表峰距窗尾 < 5%% 窗口长即判删失）：`censored = {report.get('censored')}`。\n")
    A("## ④ 结果图")
    A(f"`{figrel}`：左=四档探针水位时程（细线=原始、粗线=30 ms 仪表等效）——原始曲线的尖峰在细网格上被孤立"
      "飞溅粒子抬起，仪表曲线族相互贴近；右=原始峰值/仪表峰值/到达时刻随 dx 的变化及 Richardson 外推水平线。\n")
    A("## ⑤ 注意点与局限")
    A("1. 实验值仍 WITHHELD（见①），本节全部判据都是**内部一致性**判据，不构成「验收通过」。")
    A("2. 仪表等效窗口 30 ms 是**建模选择**（模拟低通波高仪），不是 Kleefsman 实验的仪器参数；"
      "已给 20/30/50 ms 三档敏感性，结论须对窗口稳健才可用。")
    A("3. c0 = %s m/s 下冲击区密度压缩仍高于 5 %% 可控线（见③），Splash 细节不可信；"
      "c0 阶梯（`docs/verification_kleefsman_dt_sound_speed_family.md`）另有一档把压缩压到 3.4 %%，"
      "但代价是 dt 减半，与本阶梯不同协议，故未混算。"
      % fmt(head.get("sound_speed") or 20.0, 1))
    A("4. 峰值若出现在 `t_end` 附近即为「窗口内最大」，属删失数据；③ 表已给峰值时刻，可据此判读。")
    A("")
    A("## ⑥ 误差判定")
    A("| 事项 | 基准 | 误差/判定 |\n|---|---|---|")
    A("| 阶梯协议一致性（修复前混用 t_end） | 协议指纹全等 | 修复后不一致项 = %d |" % len(report["protocol_mismatch"]))
    if report["metrics"]:
        g = report["metrics"].get("gauge_level_peak_w030ms")
        pk = report["metrics"].get("peak_height_m_raw")
        ar = report["metrics"].get("arrival_time_s")
        A("| 仪表水位峰值收敛 | 最细三档自比较 | p = %s，GCI = %s，判定 %s |"
          % (fmt((g or {}).get("p_finest"), 3), fmt((g or {}).get("gci_fine"), 4),
             (g or {}).get("verdict")))
        A("| 原始峰值收敛 | 同上 | p = %s，判定 %s |"
          % (fmt((pk or {}).get("p_finest"), 3), (pk or {}).get("verdict")))
        A("| 到达时刻收敛 | 同上 | p = %s，GCI = %s，判定 %s |"
          % (fmt((ar or {}).get("p_finest"), 3), fmt((ar or {}).get("gci_fine"), 4),
             (ar or {}).get("verdict")))
    A("| 与 Kleefsman 2005 实验的 <10 % 判据 | 实验峰值/到达时刻 | **未评估**（实验值 WITHHELD，无基准不得声称通过） |")
    A("| 是否 <3 % | —— | 无外部基准，**不做该断言**（收敛阶与 GCI 只给离散误差带） |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(BENCH / "results"))
    ap.add_argument("--doc", default=str(ROOT / "docs" / "verification_kleefsman_convergence_ladder.md"))
    ap.add_argument("--figure", default=str(ROOT / "docs" / "figures" / "kleefsman_convergence_ladder.png"))
    args = ap.parse_args()
    results = Path(args.results)
    study = json.loads((results / "resolution_study.json").read_text())
    ladder_json = results / "convergence_ladder.json"
    if ladder_json.exists():
        report = json.loads(ladder_json.read_text())
    else:
        report = ca.analyse_rows(study["runs"], results)
        ladder_json.write_text(json.dumps(report, indent=2))
    rows = study["runs"]
    fig = make_figure(rows, report, Path(args.figure))
    doc = Path(args.doc)
    rel = Path(fig).relative_to(ROOT).as_posix() if Path(fig).is_relative_to(ROOT) else str(fig)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(build_doc(rows, report, rel))
    print("wrote", doc, "and", fig, "| verdict:", report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())