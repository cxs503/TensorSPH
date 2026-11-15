"""Generate the periodic-window FSI2 verdict report from run artefacts.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Every number in the emitted markdown is read from the artefacts (nothing is
hand copied), and every figure is drawn from the same arrays, so the report and
the raw series cannot drift apart.

    python benchmarks/xval_fsi2/make_viv_report.py \
        --artifact artifacts/xval_fsi2/viv_dx005_aitken.json \
        --artifact artifacts/xval_fsi2/viv_dx002_aitken.json \
        --out-md docs/xval_fsi2/VIV_PERIODIC_VERDICT.md \
        --fig-dir docs/xval_fsi2/assets

The verdict itself is *not* re-implemented here: it comes from
``verdict_body_force_transient.build_verdict``, the same gate the previous
round used, so this script can only make the report, never the answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from verdict_body_force_transient import build_verdict  # noqa: E402
from tensorsph.fsi2_reference import BENCHMARK_FORCES, BENCHMARK_TIP  # noqa: E402


def _rel(v):
    return "n/a" if v is None else f"{100.0 * v:+.2f} %"


def _num(v, fmt="{:.5g}"):
    if v is None:
        return "n/a"
    return fmt.format(v)


def _row_table(rows, label):
    out = [f"| {label} | 基准 (Turek–Hron) | 实测 | 相对误差 | <15 % | <3 % |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| `{r['quantity']}` | {_num(r['reference'])} | {_num(r['computed'])} "
            f"| {_rel(r['rel_error'])} | {'✔' if r['pass_15'] else '✘'} "
            f"| {'✔' if r['pass_3'] else '✘'} |")
    return "\n".join(out)


def _fig(hist, verdict, dx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(hist["t"], float)
    tipy = np.asarray(hist["tip_y"], float)
    cd = np.asarray(hist["body_drag_fit"], float)
    cl = np.asarray(hist["body_lift_fit"], float)
    scale = verdict["coefficients"]["scale_0.5_rho_U2_D"]
    t0 = verdict["window_start_t"] or t[0]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].plot(t, 1e3 * tipy, lw=1.0, color="#1f77b4")
    axes[0].axvline(t0, ls="--", lw=1.0, color="0.4",
                    label=f"分析窗口起点 t={t0:.2f} s")
    ref = BENCHMARK_TIP["tip_y_amplitude"]
    axes[0].axhline(1e3 * ref, lw=1.0, ls=":", color="#d62728",
                    label=f"基准幅值 {1e3*ref:.2f} mm")
    axes[0].axhline(-1e3 * ref, lw=1.0, ls=":", color="#d62728")
    axes[0].set_ylabel("梁尖 y 位移 [mm]")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(f"FSI2 周期性瞬态（dx = {dx} m, 强耦合 aitken）")

    axes[1].plot(t, cd / scale, lw=0.9, color="#2ca02c", label="Cd（面元拟合臂）")
    axes[1].plot(t, cl / scale, lw=0.9, color="#9467bd", label="Cl（面元拟合臂）")
    axes[1].axhline(BENCHMARK_FORCES["drag_mean"], lw=1.0, ls=":", color="#d62728",
                    label=f"基准 Cd 均值 {BENCHMARK_FORCES['drag_mean']}")
    axes[1].axvline(t0, ls="--", lw=1.0, color="0.4")
    axes[1].set_xlabel("物理时间 t [s]")
    axes[1].set_ylabel("力系数 [-]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _bar(verdicts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, vals, colors = [], [], []
    for dx, v in verdicts.items():
        for group in ("tip", "fit_arm"):
            for r in v["gates"][group]["rows"]:
                if r["rel_error"] is None:
                    continue
                labels.append(f"{r['quantity']}\n(dx={dx})")
                vals.append(100.0 * r["rel_error"])
                colors.append("#2ca02c" if r["pass_15"] else "#d62728")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels)), 3.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors)
    ax.axhline(15.0, ls="--", color="k", lw=1.0, label="验收门 15 %")
    ax.axhline(-15.0, ls="--", color="k", lw=1.0)
    ax.axhline(3.0, ls=":", color="0.4", lw=1.0, label="报告目标 3 %")
    ax.axhline(-3.0, ls=":", color="0.4", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("相对误差 [%]")
    ax.set_title("FSI2 报告量 vs Turek–Hron 基准（周期性窗口）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.3)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--fig-dir", default=str(ROOT / "docs" / "xval_fsi2" / "assets"))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    verdicts, payloads = {}, {}
    for a in args.artifact:
        p = json.loads(Path(a).read_text())
        p["_source"] = Path(a).name
        dx = p.get("dx")
        payloads[dx] = p
        verdicts[dx] = build_verdict(p, warmup_frac=args.warmup_frac)

    fig_dir = Path(args.fig_dir)
    fig_names = []
    if args.figures:
        for dx, v in verdicts.items():
            name = f"viv_dx{str(dx).replace('.', '')}.png"
            _fig(payloads[dx]["history"], v, dx, fig_dir / name)
            fig_names.append(name)
        _bar(verdicts, fig_dir / "viv_error_bars.png")
        fig_names.append("viv_error_bars.png")

    lines = []
    lines.append("# FSI2 周期性窗口判定：强耦合长时瞬态 vs Turek–Hron 基准")
    lines.append("")
    lines.append("**锚点**：`XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`　"
                 "**仓**：TensorSPH　**域**：SPH / FSI")
    lines.append("")
    lines.append("> 本文件由 `benchmarks/xval_fsi2/make_viv_report.py` 从运行工件程序化生成，"
                 "所有数字与图均直接取自 JSON 序列；判定由 "
                 "`verdict_body_force_transient.build_verdict` 给出（与上一轮同一道门，"
                 "本轮未改阈值、未改判据）。")
    lines.append("")
    lines.append("## ① 案例介绍")
    lines.append("")
    lines.append("Turek–Hron FSI2（LNCSE 53, 2006）：2-D 通道 2.5 × 0.41 m，D = 0.1 m "
                 "刚性圆柱（心 (0.2, 0.2)）+ L = 0.35 m、t = 0.02 m 弹性梁，Re = 100，"
                 "ρ_s/ρ_f = 1。入口抛物型，U_mean = 0.2 m/s。")
    lines.append("")
    lines.append("| 量 | 基准值 | 出处 |")
    lines.append("|---|---|---|")
    for k, v in BENCHMARK_TIP.items():
        lines.append(f"| 梁尖 `{k}` | {v} m | Turek–Hron 2006（二手，见 `fsi2_reference.py`）|")
    for k, v in BENCHMARK_FORCES.items():
        lines.append(f"| `{k}` | {v} | 同上 |")
    lines.append("")
    lines.append("## ② 怎么做的")
    lines.append("")
    lines.append("驱动 `benchmarks/xval_fsi2/run_body_force_transient.py`，界面用 "
                 "强耦合子迭代（ρ_s/ρ_f = 1 时松耦合一步发散，见 `COUPLED_RUN_DEFECTS.md`）。"
                 "本轮唯一的方法学改动：加速器 **`aitken`（每步重启 secant 记账）**，"
                 "依据 `COUPLING_CONVERGENCE_BUDGET.md` —— IQN-ILS 在该界面的归约地板 ~1e-2，"
                 "Aitken 可达 1e-10，且 `max_iter = 3` 后截断对报告量的污染 <3 %。")
    lines.append("")
    lines.append("| 运行 | dx [m] | 步数 | 物理时长 [s] | dt [s] | 界面 | 平均子迭代 | 收敛步占比 | 中止 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for dx, v in verdicts.items():
        p = payloads[dx]
        lines.append(
            f"| `{p['_source']}` | {dx} | {v['n_steps']} | {v['t_end']:.3f} | "
            f"{v['dt']:.3e} | {p.get('interface_solver', 'iqn-ils')} | "
            f"{_num(v['coupling']['mean_iterations'], '{:.2f}')} | "
            f"{_num(v['coupling']['converged_fraction'], '{:.3f}')} | "
            f"{'是' if v['aborted'] else '否'} |")
    lines.append("")
    lines.append("## ③ 计算结果")
    lines.append("")
    for dx, v in verdicts.items():
        w = v["window"]
        lines.append(f"### dx = {dx} m（分析窗口 t = "
                     f"{v['window_start_t']:.3f} … {v['window_end_t']:.3f} s）")
        lines.append("")
        lines.append(f"* 信号自相关测得主导周期 **{_num(w['period_s'], '{:.4f}')} s**，"
                     f"窗口覆盖 **{_num(w['periods_in_window'], '{:.2f}')} 个周期** "
                     f"→ 周期性判据 `regime_ok = {v['regime_ok']}`")
        lines.append(f"* 力系数标度 `0.5 ρ U_mean² D` = {v['coefficients']['scale_0.5_rho_U2_D']:.4g}"
                     f"（U_mean = {v['coefficients']['u_mean']:.3g} m/s）")
        adm = v.get("admissibility", {})
        lines.append(f"* **可判定性筛检** `admissible = {adm.get('admissible')}`"
                     f"（已核 {', '.join(adm.get('checked', [])) or '无'}；"
                     f"未核 {', '.join(adm.get('skipped', [])) or '无'}）")
        for r in adm.get("reasons", []):
            lines.append(f"    * ⛔ {r}")
        if not v["gates"]["tip"].get("comparable", True):
            lines.append("")
            lines.append("> 下表相对误差**不是**精度判定：本算例不满足可判定性条件，"
                         "判定降级为 `WITHHELD`，数值仅作取证保留。")
        lines.append("")
        lines.append(_row_table(v["gates"]["tip"]["rows"], "梁尖位移"))
        lines.append("")
        lines.append(_row_table(v["gates"]["fit_arm"]["rows"], "力系数（面元臂）"))
        lines.append("")
        if v["gates"].get("fit_plus_viscous_cd_mean"):
            r = v["gates"]["fit_plus_viscous_cd_mean"]
            lines.append(f"压力臂 + 黏性臂合并的 `drag_mean`："
                         f"{_num(r['computed'])} vs {_num(r['reference'])}（{_rel(r['rel_error'])}）")
            lines.append("")
        lines.append(f"判定：梁尖 **{v['gates']['tip']['verdict']}**，"
                     f"力系数 **{v['gates']['fit_arm']['verdict']}**；"
                     f"最坏相对误差 {_rel(v['gates']['tip']['worst_rel_error'])} / "
                     f"{_rel(v['gates']['fit_arm']['worst_rel_error'])}")
        if v.get("withheld_reason"):
            lines.append(f"")
            lines.append(f"`withheld_reason = {v['withheld_reason']}`")
        lines.append("")
    if fig_names:
        lines.append("### 结果图")
        lines.append("")
        for n in fig_names:
            lines.append(f"![{n}](assets/{n})")
            lines.append("")
    lines.append("## ④ 误差判定（显式回答「是否 <3 %」）")
    lines.append("")
    lines.append("| 量 | 门槛 | 本次最坏相对误差 | <15 % | <3 % | 可判定 |")
    lines.append("|---|---|---|---|---|---|")
    for dx, v in verdicts.items():
        for group in ("tip", "fit_arm"):
            g = v["gates"][group]
            comparable = g.get("comparable", True)
            # a non-comparable row is not an accuracy answer, in either direction
            p15 = "否" if not comparable else (
                "是" if (g['worst_rel_error'] is not None and abs(g['worst_rel_error']) <= 0.15) else "否")
            p3 = "否" if not comparable else (
                "是" if (g['worst_rel_error'] is not None and abs(g['worst_rel_error']) <= 0.03) else "否")
            lines.append(f"| dx={dx} {group} | 15 % | {_rel(g['worst_rel_error'])} | "
                         f"{p15} | {p3} | {'是' if comparable else '否（非物理解）'} |")
    lines.append("")
    lines.append(Path(args.out_md).parent.joinpath("_viv_notes.md").exists()
                 and Path(args.out_md).parent.joinpath("_viv_notes.md").read_text()
                 or "")
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({sum(1 for v in verdicts)} runs)")
    for dx, v in verdicts.items():
        print(f"dx={dx} tip={v['gates']['tip']['verdict']} "
              f"force={v['gates']['fit_arm']['verdict']} "
              f"period={v['window']['period_s']} periods={v['window']['periods_in_window']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())