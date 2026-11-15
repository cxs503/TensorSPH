"""Generate ``docs/verification_kleefsman_viscosity_ordering.md`` from the
committed viscosity-sweep artifacts (zero hand transcription).

Every number, verdict and figure in the document is derived here from

* ``results/viscosity_sweep/<tag>/summary.json`` (written by
  :mod:`benchmarks.kleefsman_obstacle.viscosity_sweep`),
* ``results/viscosity_sweep/<tag>/dx*_nu*_time_series.csv`` (probe history),
* ``docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.csv`` (the digitised
  Kleefsman 2005 Fig. 19 experiment/COMFLOW curves).

Usage::

    python benchmarks/kleefsman_obstacle/make_doc_viscosity.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "viscosity_sweep"
DOC = ROOT / "docs" / "verification_kleefsman_viscosity_ordering.md"
FIGDIR = ROOT / "docs" / "figures"
EXP_CSV = ROOT / "docs" / "m20884" / "outputs" / "kleefsman_h2_ref" / "h2_fig19.csv"

# Tolerances that decide the verdict.  The case pool asks for < 10 % on both
# scored metrics; < 3 % is the stricter "publication grade" bar this document
# answers separately so a PASS at 10 % cannot be mistaken for a 3 % match.
TOL_POOL = 0.10
TOL_TIGHT = 0.03
OLD_CAL = {"x0": 661.0, "x1": 1179.0, "y_top": 6.5, "y_bot": 413.5}



def blasius_sink(nu: float, u_ref: float, travel: float, depth: float,
                 rho: float = 1000.0) -> dict:
    """Order-of-magnitude bed-friction sink for a laminar boundary layer.

    Not a solution of this case: it answers "how big a front-speed deficit
    should a kinematic viscosity nu produce at all?" so the sweep's measured
    deficit has something to be compared with *before* looking at the answer.

    tau_w = 0.664 rho U^2 Re_L^-1/2 (Blasius, laminar flat plate over the
    travel length L), deceleration a = tau_w / (rho H) for a layer of depth H,
    deficit over the travel time t = L/U:  dU = a t.
    """
    if nu <= 0:
        return {"re_L": float("inf"), "tau_w": 0.0, "decel": 0.0,
                "du_pred": 0.0, "du_rel_pred": 0.0}
    re_l = u_ref * travel / nu
    tau_w = 0.664 * rho * u_ref ** 2 / math.sqrt(re_l)
    decel = tau_w / (rho * depth)
    t = travel / u_ref
    du = decel * t
    return {"re_L": re_l, "tau_w": tau_w, "decel": decel,
            "du_pred": du, "du_rel_pred": du / u_ref}

def load_summaries() -> list[dict]:
    out = []
    for path in sorted(RESULTS.glob("*/summary.json")):
        out.append(json.loads(path.read_text()) | {"_path": str(path)})
    return out


def read_series(path: Path) -> tuple[list[float], list[float], list[float]]:
    ts, hs, fs = [], [], []
    with path.open() as fh:
        next(fh)
        for line in fh:
            t, h, f = (float(v) for v in line.split(",")[:3])
            ts.append(t)
            hs.append(h)
            fs.append(f)
    return ts, hs, fs


def read_experiment() -> tuple[list[float], list[float]]:
    ts, hs = [], []
    with EXP_CSV.open() as fh:
        next(fh)
        for line in fh:
            t, h = (float(v) for v in line.split(",")[:2])
            ts.append(t)
            hs.append(h)
    return ts, hs


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "∞" if math.isinf(v) else "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def fig_ordering(summaries: list[dict]) -> Path:
    """Arrival time and peak height against the viscosity setting, per dx."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    colors = {0.04: "#c62828", 0.02: "#1565c0", 0.01: "#2e7d32"}
    for s in summaries:
        runs = sorted(s["runs"], key=lambda r: -r["viscosity"])
        if not runs:
            continue
        dx = runs[0]["dx"]
        col = colors.get(dx, "#555555")
        nu = [r["viscosity"] for r in runs]
        arr = [r["arrival_time_s"] for r in runs]
        pk = [r["peak_height_m"] for r in runs]
        axes[0].plot(nu, arr, "o-", color=col, label=f"dx = {dx:g} m")
        axes[1].plot(nu, pk, "o-", color=col, label=f"dx = {dx:g} m")
        axes[2].plot([r["reynolds"] for r in runs], arr, "o-", color=col,
                     label=f"dx = {dx:g} m")
    exp = summaries[0]["experiment"] if summaries else {}
    tol = summaries[0]["tolerance_relative"] if summaries else TOL_POOL
    a_ref = exp.get("arrival_time_s")
    p_ref = exp.get("peak_height_m")
    if a_ref:
        for ax in (axes[0], axes[2]):
            ax.axhline(a_ref, color="k", ls=":", lw=1.2, label="experiment")
            ax.axhspan(a_ref * (1 - tol), a_ref * (1 + tol), color="k", alpha=0.08,
                       label="±10 % band")
    if p_ref:
        axes[1].axhline(p_ref, color="k", ls=":", lw=1.2, label="experiment")
        axes[1].axhspan(p_ref * (1 - tol), p_ref * (1 + tol), color="k", alpha=0.08,
                        label="±10 % band")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"kinematic viscosity $\nu$ [m$^2$/s]")
    axes[0].set_ylabel("arrival time [s]")
    axes[0].set_title("arrival time vs viscosity")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"kinematic viscosity $\nu$ [m$^2$/s]")
    axes[1].set_ylabel("peak water height [m]")
    axes[1].set_title("peak height vs viscosity")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Re = U·L/ν  (U from arrival time)")
    axes[2].set_ylabel("arrival time [s]")
    axes[2].set_title("arrival time vs Reynolds number")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Kleefsman (2005) obstacle case — viscosity ordering of the "
                 "scored metrics", fontsize=11)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "kleefsman_viscosity_ordering.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_histories(summaries: list[dict]) -> Path | None:
    """Probe water-height history for the most-viscous and least-viscous run."""
    exp_t, exp_h = read_experiment()
    picks = []
    for s in summaries:
        runs = sorted(s["runs"], key=lambda r: -r["viscosity"])
        if not runs:
            continue
        tagdir = Path(s["_path"]).parent
        picks.append((runs[0], tagdir))
        if runs[-1]["viscosity"] != runs[0]["viscosity"]:
            picks.append((runs[-1], tagdir))
    if not picks:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(exp_t, exp_h, "k--", lw=1.4, label="Kleefsman 2005 Fig. 19 (digitised)")
    for run, tagdir in picks:
        csv = tagdir / f"{run['tag']}_time_series.csv"
        if not csv.exists():
            continue
        ts, hs, _ = read_series(csv)
        ax.plot(ts, hs, lw=1.2,
                label=f"SPH dx={run['dx']:g}, ν={run['viscosity']:g} "
                      f"(arr {run['arrival_time_s']:.3f} s, peak "
                      f"{run['peak_height_m']:.3f} m)")
    ax.axhline(0.02, color="grey", ls="-.", lw=0.9, label="arrival threshold 0.02 m")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height at the probe [m]")
    ax.set_xlim(0.0, max(exp_t) if exp_t else 1.2)
    ax.set_ylim(0.0, 0.45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("probe height history: viscosity setting vs experiment")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "kleefsman_viscosity_histories.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def table_rows(summaries: list[dict]) -> list[str]:
    lines = ["| run | dx [m] | ν [m²/s] | Re | δ/dx | 到达 [s] | 到达误差 | 峰值 [m] | 峰值误差 | v_max [m/s] | 压缩 | 逃逸 | 判定 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for r in sorted(s["runs"], key=lambda r: (r["dx"], -r["viscosity"])):
            e = r.get("rel_error", {})
            p = r.get("pass", {})
            verdict = "PASS" if p and all(p.values()) else "FAIL"
            lines.append(
                f"| `{r['tag']}` | {r['dx']:g} | {r['viscosity']:g} | "
                f"{fmt(r['reynolds'], 0)} | {fmt(r['boundary_layer_over_dx'], 2)} | "
                f"{fmt(r['arrival_time_s'])} | {fmt(e.get('arrival_time'))} | "
                f"{fmt(r['peak_height_m'])} | {fmt(e.get('peak_height'))} | "
                f"{fmt(r['max_speed'], 2)} | {fmt(r['max_density_compression'], 3)} | "
                f"{r['boundary_escapes']} | {verdict} |")
    return lines


def main() -> int:
    summaries = load_summaries()
    if not summaries:
        raise SystemExit(f"no summaries under {RESULTS}")
    fig1 = fig_ordering(summaries)
    fig2 = fig_histories(summaries)

    exp = summaries[0]["experiment"]
    tol = summaries[0]["tolerance_relative"]
    checks = []
    for s in summaries:
        for k, v in s.get("monotonic_checks", {}).items():
            checks.append((k, v))

    # best run under the pool tolerance, and the tightest error seen
    all_runs = [r for s in summaries for r in s["runs"]]
    ok = [r for r in all_runs if r.get("pass") and all(r["pass"].values())]
    best = min(all_runs, key=lambda r: max((r.get("rel_error") or {"x": 9.9}).values()))
    tight = [r for r in all_runs
             if r.get("rel_error") and max(r["rel_error"].values()) < TOL_TIGHT]

    body = []
    body.append("# Kleefsman (2005) 带障碍物溃坝：到达时刻偏差的粘性归因（ν 扫描）\n")
    body.append("TensorSPH · 分支 `agent/m21279` · mission 21279 · 锚点 "
                "`CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`\n")
    body.append("本文档由 `benchmarks/kleefsman_obstacle/make_doc_viscosity.py` 从 "
                "`benchmarks/kleefsman_obstacle/results/viscosity_sweep/*/summary.json` "
                "与探针时序 CSV 程序化注入（零手抄）。\n")

    body.append("## ① 案例与既有结论\n")
    body.append("算例 = Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005, "
                "*J. Comput. Phys.* **206**:363–393, DOI 10.1016/j.jcp.2004.12.007) "
                "带障碍物溃坝冲击：3.22 m 长、1.00 m 高水槽左端 1.228 × 0.550 m 水柱瞬时溃决，"
                "沿干底推进撞击箱上游面 x = 2.476 m 处 0.161 × 0.161 m 刚性方箱。"
                "判据 = 箱前水位时程的**峰值**与**到达时刻**对实验相对误差均 < 10 %。\n")
    body.append(f"实验参考（同一 `reference.json`，Fig. 19 右面板数字化，"
                f"m20884）：峰值 **{fmt(exp.get('peak_height_m'), 3)} m**，"
                f"到达 **{fmt(exp.get('arrival_time_s'), 4)} s**，探针 x = "
                f"{fmt(exp.get('probe_x_m'), 3)} m（箱面 −0.05 m 约定）。\n")
    body.append("**本单要解决的问题**：`docs/benchmarks/kleefsman-validation-m21019.md` "
                "给出的分辨率梯（dx = 0.04 / 0.02 / 0.01）峰值在细网格已达 2.0 %，"
                "但到达时刻**收敛到偏晚 ~24 %**（GCI 1.33 %，即数值已收敛到错值）；"
                "`docs/benchmarks/kleefsman-arrival-attribution-m21173.md` 进一步排除了"
                "可压缩性（c0 扫描 ~2 %）把偏差折算成「等效锋面厚度」。"
                "但**粘性设置**这项从未被检验：算例沿用 `Config.viscosity = 0.01` m²/s，"
                "比水的 1e-6 m²/s 大四个量级，对应 Re ≈ 400、床面边界层 "
                "δ = √(νL/U) ≈ 60 mm（dx = 0.02 时是 3 个粒子间距）。"
                "Ritter 型无粘溃坝解根本不该在 Re ~ 400 下成立，故这是最后一个高杠杆候选。\n")

    body.append("## ② 方法\n")
    body.append("固定几何、初场、边界、核长、声速与时间步（dt = min(2e-4, dt_acoustic)），"
                "**只改运动粘度 ν**，在每条网格上扫 ν ∈ "
                "{" + ", ".join(f"{v:g}" for v in sorted({r['viscosity'] for r in all_runs}, reverse=True)) + "} m²/s，"
                "记录峰值、到达时刻（首次越过 0.02 m，线性插值）以及稳定性诊断"
                "（v_max、最大密度压缩、逃逸粒子数、质量误差）。\n")
    body.append("派生量（程序化计算，非手抄）：\n")
    body.append("- 平均锋面速度 U = (x_probe − x_gate)/t_arrival，传播距离 L = 1.198 m；\n"
                "- Re = U·L/ν；\n"
                "- 床面边界层厚度 δ = √(ν·L/U)，以 δ/dx 表示（> 1 意味着边界层跨多个粒子，"
                "惯性主导的假设失效）。\n")
    body.append("复现：\n\n```\n"
                "python benchmarks/kleefsman_obstacle/viscosity_sweep.py "
                "--dx 0.04 --viscosities 0.01 0.003 0.001 0.0001 --t-end 1.2 --tag dx0.04\n"
                "python benchmarks/kleefsman_obstacle/viscosity_sweep.py "
                "--dx 0.02 --viscosities 0.01 0.001 0.0001 --t-end 1.2 --tag dx0.02\n"
                "python benchmarks/kleefsman_obstacle/make_doc_viscosity.py\n"
                "python -m pytest tests/test_kleefsman_viscosity_sweep.py -q\n```\n")

    body.append("## ③ 结果（程序化注入）\n")
    body.extend(table_rows(summaries))
    body.append("")
    body.append("列含义：到达误差/峰值误差 = |got − ref|/|ref|；判定 = 两个指标同时 "
                f"< {tol:.0%} 才算 PASS。\n")

    if checks:
        body.append("### 3.1 单调性（ν → 0 的极限检验）\n")
        body.append("| 网格 | ν 降序 | 到达时刻 [s] | 峰值 [m] | 到达随 1/ν 单调不增 | 到达跨度 [s] |")
        body.append("|---|---|---|---|---|---|")
        for key, v in checks:
            body.append(f"| {key} | {', '.join(f'{x:g}' for x in v['viscosity_desc'])} | "
                        f"{', '.join(fmt(x, 4) for x in v['arrival_s'])} | "
                        f"{', '.join(fmt(x, 4) for x in v['peak_m'])} | "
                        f"{'是' if v['arrival_monotonically_decreasing_in_1_over_nu'] else '否'} | "
                        f"{fmt(v['arrival_span_s'], 4)} |")
        body.append("")

    body.append("### 3.2 摩阻量级先验 vs 实测锋面亏损\n")
    body.append("扫描前先算一个与该算例无关的量级参照：层流平板摩阻会从溃坝前锋拿走多少速度。"
                "取 Blasius 平板 τ_w = 0.664 ρU²Re_L^(−1/2)、水深 H = 0.550 m、行程 L = 1.198 m、"
                "参照速度 U_ref = L/t_exp（把实验到达时刻当作无粘参照），"
                "则减速度 a = τ_w/(ρH)、行程内亏损 ΔU = a·L/U_ref。它只用来判断"
                "「这个 ν 是否可能有 20 % 量级的影响」，不作为验收基准。\n")
    travel = 1.198  # box face 2.476 m - gate 1.228 m
    u_ref = travel / exp["arrival_time_s"]
    body.append("| run | ν [m²/s] | Re_L | τ_w [Pa] | a [m/s²] | ΔU_pred [m/s] | ΔU_pred/U_ref | U_sim [m/s] | 实测亏损 vs U_ref |")
    body.append("|---|---|---|---|---|---|---|---|---|")
    for s_ in summaries:
        for r in sorted(s_["runs"], key=lambda r: (r["dx"], -r["viscosity"])):
            sk = blasius_sink(r["viscosity"], u_ref, travel, 0.550)
            u_sim = r["mean_front_speed"]
            body.append(
                f"| `{r['tag']}` | {r['viscosity']:g} | {fmt(sk['re_L'], 0)} | "
                f"{fmt(sk['tau_w'], 3)} | {fmt(sk['decel'], 3)} | {fmt(sk['du_pred'], 3)} | "
                f"{fmt(sk['du_rel_pred'], 3)} | {fmt(u_sim, 3)} | "
                f"{fmt(1.0 - u_sim / u_ref, 3)} |")
    body.append("")
    body.append(f"U_ref = L/t_exp = {fmt(u_ref, 3)} m/s。先验亏损随 √ν 变化："
                "若实测亏损与这一列同量级并同步下降，说明偏晚是粘性摩阻造成的"
                "（建模选择），而不是离散误差（那需要靠加密网格解决）。\n")

    audit_path = RESULTS / "axis_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        body.append("### 3.3 参考曲线自身的时间标定审计（排除「基准被我搬歪了」）\n")
        fr = audit.get("frame", {})
        body.append(f"实验/COMFLOW 两条参考曲线来自同一张栅格图（Fig. 19 右面板），"
                    f"两者互相吻合只能证明**相对**数字化可靠，无法发现**共同**的坐标标定误差。"
                    f"故直接从 JPEG 重新量取面板边框："
                    f"frame_cols = {fr.get('frame_cols')}，frame_rows = {fr.get('frame_rows')}，"
                    f"旧标定 = x {OLD_CAL['x0']}..{OLD_CAL['x1']} / y {OLD_CAL['y_top']}..{OLD_CAL['y_bot']}"
                    f"（x 右边界差了 {abs(OLD_CAL['x1'] - audit['measured_calibration']['x1']):g} px）。\n")
        body.append("| 曲线 | 到达(旧标定) [s] | 到达(实测标定) [s] | 到达位移 [s] | 峰值(旧) [m] | 峰值(实测) [m] | 峰值位移 [m] |")
        body.append("|---|---|---|---|---|---|---|")
        for key, v in audit["curves"].items():
            body.append(f"| {key} | {fmt(v['arrival_old_s'])} | {fmt(v['arrival_new_s'])} | "
                        f"{fmt(v['arrival_shift_s'])} | {fmt(v['peak_old_m'])} | "
                        f"{fmt(v['peak_new_m'])} | {fmt(v['peak_shift_m'])} |")
        body.append("")
        res = audit.get("residual")
        if res:
            body.append(f"残差（{res['best_run']} 到达 {fmt(res['best_arrival_s'])} s − 实验 "
                        f"{fmt(res['experiment_arrival_s'])} s）= **{fmt(res['residual_s'])} s**，"
                        f"折合 **{fmt(res['residual_px'], 1)} px**；标定可移动量只有 "
                        f"{fmt(res['calibration_shift_px'], 1)} px（{fmt(res['calibration_shift_s'])} s）。"
                        f"因此残差**{'落在' if res['residual_explained_by_calibration'] else '不落在'}** "
                        f"数字化标定不确定度内 → "
                        f"{'基准误差可以解释残差' if res['residual_explained_by_calibration'] else '残差是真实的求解器偏差，不能归给基准'}。\n")
        body.append(f"口径说明：`reference.json` 记的到达 0.3591 s 是首个越过 0.02 m 的采样点，"
                    f"本审计用相邻采样线性插值得 {fmt(audit['curves']['dashed_experiment']['arrival_old_s'])} s，"
                    f"两者差 {fmt(0.3591 - audit['curves']['dashed_experiment']['arrival_old_s'])} s，"
                    f"小于数字化自身 ±0.01 s 的不确定度；下文判定一律用 `reference.json` 的值（0.3591 s）。\n")

    body.append("### 3.4 判定汇总\n")
    body.append(f"- 达到算例池判据（两指标均 < {tol:.0%}）的 run 数：**{len(ok)}** / {len(all_runs)}"
                + (f"（{', '.join('`' + r['tag'] + '`' for r in ok)}）" if ok else "") + "；")
    if best.get("rel_error"):
        worst = max(best["rel_error"].items(), key=lambda kv: kv[1])
        body.append(f"- 误差最小 run：`{best['tag']}`（ν = {best['viscosity']:g} m²/s），"
                    f"最差指标 = {worst[0]} {worst[1]:.4f}（{worst[1]:.2%}）；")
    body.append(f"- 达到「论文级」判据（两指标均 < {TOL_TIGHT:.0%}）的 run 数：**{len(tight)}**"
                + (f"（{', '.join('`' + r['tag'] + '`' for r in tight)}）" if tight else "") + "。")
    body.append("")

    body.append("## ④ 结果图\n")
    body.append(f"![粘性排序]({fig1.relative_to(ROOT / 'docs').as_posix() if False else 'figures/' + fig1.name})\n")
    body.append("左：到达时刻 vs ν（点线 = 实验值，灰带 = ±10 % 容差）；中：峰值 vs ν；"
                "右：到达时刻 vs Re —— 若曲线在右侧趋于平台，说明 Re ≳ 1e3 后解已与 ν 无关。\n")
    if fig2 is not None:
        body.append(f"![水位时程](figures/{fig2.name})\n")
        body.append("探针水位时程：黑虚线 = Kleefsman 2005 Fig. 19 数字化实验，"
                    "彩线 = 最粘/最不粘两个 ν 设置的 SPH 结果；灰色点划线 = 0.02 m 到达阈值。\n")

    body.append("## ⑤ 注意点 / 局限\n")
    body.append("1. **这不是「调参对实验」**。唯一物理正确的 ν 是水的 1e-6 m²/s（Re ≈ 1.6e6），"
                "在 dx ≥ 0.01 m 时亚粒子（δ/dx ≪ 1），SPH 无法分辨；因此正确做法是取"
                "「数值上能稳定且已不影响判据」的最小 ν，并把该 ν 与导出的 Re、δ/dx 一并报告。"
                "本文档给出的通过判定只在该 ν 下成立，不能外推到任意 ν。\n"
                "2. **峰值对 ν 的敏感度**与到达不同：细网格峰值本身含飞溅/二次撞击的非单调性"
                "（m21019 已记录 NOT_CONVERGED），故峰值仍应看趋势而非单点。\n"
                "3. **探针 x 未获权威值**（箱面 − 0.05 m 约定）：这一点本单无法修复，见 "
                "`reference.json` 与 m21173 的站位灵敏度分析。\n"
                "4. 本单不改变几何、时间步与离散，因此**不覆盖** dx < 0.01 或三维效应；"
                "t_end = 1.2 s，未覆盖 t = 1.70 s 的回流二次峰值。\n")
    body.append("## ⑥ 误差判定（逐项 vs 实验基准）\n")
    body.append("| 指标 | 基准（Kleefsman 2005） | 最小误差 run | 相对误差 | < 10 %？ | < 3 %？ |")
    body.append("|---|---|---|---|---|---|")
    for key, ref_val, unit in (("arrival_time", exp.get("arrival_time_s"), "s"),
                               ("peak_height", exp.get("peak_height_m"), "m")):
        cand = [r for r in all_runs if (r.get("rel_error") or {}).get(key) is not None]
        if not cand:
            continue
        r = min(cand, key=lambda r: r["rel_error"][key])
        err = r["rel_error"][key]
        body.append(f"| {key} | {fmt(ref_val, 4)} {unit} | `{r['tag']}` "
                    f"{fmt(r['arrival_time_s'] if key == 'arrival_time' else r['peak_height_m'], 4)} {unit} | "
                    f"{err:.2%} | {'✅' if err < TOL_POOL else '❌'} | "
                    f"{'✅' if err < TOL_TIGHT else '❌'} |")
    body.append("")
    passed_pool = sum(1 for r in all_runs
                      if (r.get("rel_error") or {}).get("arrival_time", 9) < TOL_POOL)
    passed_tight = sum(1 for r in all_runs
                       if (r.get("rel_error") or {}).get("arrival_time", 9) < TOL_TIGHT)
    body.append(f"**显式回答**：到达时刻在 {passed_pool}/{len(all_runs)} 个 run 中 < 10 %，"
                f"在 {passed_tight}/{len(all_runs)} 个 run 中 < 3 %；"
                "峰值见上表。未达到的 run 如实记为 FAIL，不做四舍五入美化。\n")

    DOC.write_text("\n".join(body) + "\n")
    print(f"wrote {DOC}")
    print(f"wrote {fig1}")
    if fig2 is not None:
        print(f"wrote {fig2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())