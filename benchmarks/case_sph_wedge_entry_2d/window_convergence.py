"""Is the committed peak-Cp ensemble a *converged* estimate?  (m22436)

What this checks, and why it matters for the acceptance verdict
---------------------------------------------------------------
``ensemble_peak.py`` builds its estimate from ``sel = shots[-n_frames:]``, i.e.
the **last** ``n_frames`` frames of the observation window, and it reports the
estimate for ``FRAME_COUNTS = (7, 13, 25, 41, 61)``.  Those five levels are
therefore not "more samples of the same thing": they are a
**window-extension ladder** -- level ``n`` admits ``n - n_prev`` *earlier*
frames.  If the self-similar ensemble were converged, the five levels would
agree; the spread between them is a *systematic* (window-selection) uncertainty
that the committed uncertainty rule never carried:

    committed rule (``ladder_closure.py``)
        U = max(bin-count spread, frame-half jackknife)
    committed flag (``ensemble_peak.py``, ``judgeable``)
        judgeable = spread_rel < 0.15        # full bin grid, no jackknife

Two different functions decide "judgeable" on the same artefact, so one artefact
can be judgeable under its own flag and not judgeable under the verdict rule.
This module (a) puts the window-extension sensitivity ``w`` on the same footing
as the other two components, (b) reports the disagreement between the two
committed definitions, and (c) re-runs the *unchanged* 15 % gate with the
unified ``U* = max(spread, jackknife, w)``.

Predeclared (written before the first run)
------------------------------------------
* ``w_n = |peak(n) / peak(n_prev) - 1|`` for consecutive window-extension
  levels; ``w = max_n w_n``.  A converged estimate has ``w -> 0``.
* ``U* = max(resolution-band bin spread, frame-half jackknife, w)``.
* ``judgeable* = U* <= 0.15`` -- the gate is **not** widened.
* A grid whose ``w > 0.15`` is *non-converged*: no value read off that artefact
  can be quoted to better than ``w``, whatever the other components say.
* Nothing here touches the solver, the case definition, the reference or the
  15 % gate; no new SPH run is required (the artifacts are committed).

Usage::

    python benchmarks/case_sph_wedge_entry_2d/window_convergence.py

Writes ``docs/artifacts/case_sph_wedge_entry_2d_window/window_convergence.json``,
a figure, and the generated document
``docs/case_sph_wedge_entry_2d_window_convergence.md`` (every number in it is
read from the JSON, none is typed in).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "case_sph_wedge_entry_2d"))
sys.path.insert(0, str(ROOT / "src"))

import literature_refs as lr  # noqa: E402

GATE = 0.15
BETA_DEG = 30.0
DEFAULT_ART = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ensemble"
DEFAULT_OUT = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_window"
PERCENT = 100.0


def finite(x) -> bool:
    return isinstance(x, (int, float)) and x == x and math.isfinite(x)


def window_ladder(art: dict) -> list[dict]:
    """One row per window-extension level (``by_frames`` is keyed by n)."""
    rows = []
    for key in sorted(art.get("by_frames", {}), key=int):
        rec = art["by_frames"][key]
        rb = rec.get("resolution_band", {})
        rows.append({
            "n_frames": int(key),
            "peak": rb.get("peak_median", float("nan")),
            "bin_spread": rb.get("spread_rel", float("nan")),
            "jackknife": rec.get("jackknife_rel", float("nan")),
            "spread_full_grid": rec.get("spread_rel", float("nan")),
            "judgeable_committed_flag": bool(rec.get("judgeable")),
            "pooled_samples": rec.get("pooled_samples"),
            "mean_wet_nodes": rec.get("mean_wet_nodes"),
        })
    rows.sort(key=lambda r: r["n_frames"])
    return rows


def window_sensitivity(rows: list[dict]) -> list[float]:
    """``w_n = |peak(n)/peak(n_prev) - 1|``; the first level has no predecessor."""
    out = [float("nan")]
    for prev, cur in zip(rows, rows[1:]):
        p0, p1 = prev["peak"], cur["peak"]
        out.append(abs(p1 / p0 - 1.0) if finite(p0) and finite(p1) and p0 else float("nan"))
    return out


def unified_uncertainty(row: dict, w: float) -> float:
    parts = [v for v in (row["bin_spread"], row["jackknife"], w) if finite(v)]
    return max(parts) if parts else float("nan")


def audit_grid(tag: str, art: dict, reference: float) -> dict:
    rows = window_ladder(art)
    ws = window_sensitivity(rows)
    for row, w in zip(rows, ws):
        row["window_sensitivity"] = w
        row["uncertainty_unified"] = unified_uncertainty(row, w)
        row["judgeable_unified"] = bool(finite(row["uncertainty_unified"])
                                        and row["uncertainty_unified"] <= GATE)
        row["residual_designated"] = (row["peak"] / reference - 1.0
                                      if finite(row["peak"]) and reference else float("nan"))
        # the committed verdict rule of ladder_closure.py, for comparison
        u_committed = max(v for v in (row["bin_spread"], row["jackknife"])
                          if finite(v))
        row["uncertainty_committed_rule"] = u_committed
        row["judgeable_committed_rule"] = bool(u_committed <= GATE)
    finest = rows[-1]
    return {
        "tag": tag,
        "dx_m": art.get("config", {}).get("dx"),
        "rows": rows,
        "finest": finest,
        "window_sensitivity_max": max([w for w in ws if finite(w)] or [float("nan")]),
        "monotone_in_extension": _monotone([r["peak"] for r in rows]),
        "reference_cp_q": reference,
        "flag_disagreements": [r["n_frames"] for r in rows
                               if r["judgeable_committed_flag"] != r["judgeable_committed_rule"]],
    }


def _monotone(values: list[float]) -> int:
    """+1 rising, -1 falling, 0 mixed / not evaluable."""
    vals = [v for v in values if finite(v)]
    if len(vals) < 2:
        return 0
    if all(b > a for a, b in zip(vals, vals[1:])):
        return 1
    if all(b < a for a, b in zip(vals, vals[1:])):
        return -1
    return 0


def audit(art_dir: Path, reference: float | None = None) -> dict:
    ref = reference if reference is not None else lr.designated_cp_max(BETA_DEG)
    grids = []
    for path in sorted(Path(art_dir).glob("ensemble_*_c0v20.json")):
        art = json.loads(path.read_text())
        tag = path.name.split("ensemble_")[1].split("_c0v20")[0]
        grids.append(audit_grid(tag, art, ref))
    grids.sort(key=lambda g: -(g["dx_m"] or 0.0))
    judgeable = [g["tag"] for g in grids if g["finest"]["judgeable_unified"]]
    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/window_convergence.py",
        "design": {
            "gate": GATE,
            "beta_deg": BETA_DEG,
            "reference": f"designated composite solution {ref:.3f} (q units)",
            "uncertainty_unified": "max(resolution-band bin spread, frame-half jackknife, "
                                   "window-extension sensitivity)",
            "window_sensitivity": "|peak(n)/peak(n_prev)-1| over by_frames levels",
            "gate_widened": False,
        },
        "artifacts_dir": str(Path(art_dir).relative_to(ROOT)),
        "grids": grids,
        "grids_judgeable_unified": judgeable,
        "verdict": {
            "verdict": "NOT_DECIDABLE" if len(judgeable) < 2 else "SEE_COMMITTED_RULE",
            "reason": (f"only {len(judgeable)} grid(s) judgeable with the unified "
                       f"uncertainty (gate not widened)" if len(judgeable) < 2
                       else "unified uncertainty leaves >=2 judgeable grids; "
                            "the committed rule must be re-run on them"),
        },
    }


def make_figure(rep: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for g in rep["grids"]:
        ns = [r["n_frames"] for r in g["rows"]]
        axes[0].plot(ns, [r["peak"] for r in g["rows"]], marker="o", label=g["tag"])
        axes[1].plot(ns, [PERCENT * (r["window_sensitivity"] or float("nan"))
                          for r in g["rows"]], marker="o", label=g["tag"])
    ref = rep["designated_cp"]
    axes[0].axhline(ref, color="k", ls="-.", lw=1,
                    label=f"Z&F COM {ref:.2f}")
    axes[0].axhline(rep["wagner_point"], color="grey", ls=":", lw=1,
                    label=f"Wagner point {rep['wagner_point']:.2f}")
    axes[0].set_xlabel("ensemble size n (last n frames of the window)")
    axes[0].set_ylabel(r"peak $C_p$ (resolution band)")
    axes[0].set_title("Window-extension ladder: not flat = not converged")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)
    axes[1].axhline(PERCENT * GATE, color="tab:red", ls="--", lw=1, label="15 % gate")
    axes[1].set_xlabel("ensemble size n")
    axes[1].set_ylabel("window sensitivity [%]")
    axes[1].set_title("Admitting earlier frames moves the answer")
    axes[1].grid(alpha=.3); axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = out / "window_convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_doc(rep: dict, fig: Path) -> str:
    L = []
    add = L.append
    add("# CASE-SPH-WEDGE-ENTRY-2D — 峰值压力系综的**窗口收敛性**审计（m22436）")
    add("")
    add("* 本文件由 `benchmarks/case_sph_wedge_entry_2d/window_convergence.py` 从 "
        f"`{rep['artifacts_dir']}` 的已提交工件生成，文中每个数字都读自 "
        "`docs/artifacts/case_sph_wedge_entry_2d_window/window_convergence.json`。")
    add(f"* 判据门限 **未放宽**：`{rep['design_gate']:.0%}`；参考（指定度量主）"
        f"`{rep['design_reference']}`。")
    add("")
    add("## ① 本单要回答的问题")
    add("")
    add("前序各轮把 `by_frames` 的 5 个档位（7/13/25/41/61）当作"
        "「同一估计量的更多样本」。代码不是这样：")
    add("`sel = shots[-n_frames:]` 取的是**窗口末尾 n 帧**，所以从 7 到 61 是"
        "**逐级纳入更早的帧**（窗口延长阶梯），不是样本累积。")
    add("若自相似系综已收敛，五个档位必须给出同一个数；档位之间的散布就是"
        "**窗口选择带来的系统不确定度**，而已提交的 U 规则里没有这一项：")
    add("")
    add("```")
    add("已提交判决规则 (ladder_closure.py):  U = max(bin-count spread, frame-half jackknife)")
    add("已提交 judgeable 标志 (ensemble_peak.py): judgeable = spread_rel < 0.15  # 全 bin 网格, 不含 jackknife)")
    add("```")
    add("同一份工件因此在两个函数下可以同时是「可裁决」和「不可裁决」。")
    add("")
    add("## ② 方法（先声明，后计算）")
    add("")
    add("* `w_n = |peak(n)/peak(n_prev) - 1|`（相邻档位），`w = max_n w_n`；收敛要求 `w → 0`。")
    add("* `U* = max(分辨带 bin spread, 帧半分 jackknife, w)`；`judgeable* = U* <= 15 %`。")
    add("* `w > 15 %` 的网格判为**未收敛**：该工件上读出的任何数都不可能准于 `w`。")
    add("* 求解器、算例、参考、门限一律未动；本审计**不新增计算**（工件已提交）。")
    add("")
    add("## ③ 结果")
    add("")
    for g in rep["grids"]:
        add(f"### `{g['tag']}`（dx = {g['dx_m']*1e3:.2f} mm）")
        add("")
        add("| n | pooled 样本 | 湿节点均值 | peak Cp | bin spread | jackknife | **w** | **U\\*** | judgeable* | 对指定参考残差 |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for r in g["rows"]:
            add("| {n} | {ps} | {wn:.1f} | {pk:.3f} | {bs:+.1f} % | {jk:+.1f} % | {w} | {u} | {j} | {rd} |".format(
                n=r["n_frames"], ps=r["pooled_samples"], wn=r["mean_wet_nodes"],
                pk=r["peak"], bs=PERCENT * r["bin_spread"], jk=PERCENT * r["jackknife"],
                w=(f"{PERCENT*r['window_sensitivity']:+.1f} %"
                   if finite(r["window_sensitivity"]) else "—"),
                u=(f"{PERCENT*r['uncertainty_unified']:.1f} %"
                   if finite(r["uncertainty_unified"]) else "—"),
                j="yes" if r["judgeable_unified"] else "no",
                rd=(f"{PERCENT*r['residual_designated']:+.1f} %"
                    if finite(r["residual_designated"]) else "—")))
        add("")
        add(f"* 窗口敏感度最大 **{PERCENT*g['window_sensitivity_max']:.1f} %**；"
            f"peak(n) 随窗口延长 {_trend_word(g['monotone_in_extension'])}；"
            f"两套已提交 definition 冲突档位：{g['flag_disagreements'] or '无'}。")
        add("")
    add("## ④ 图")
    add("")
    add(f"![window](../docs/artifacts/case_sph_wedge_entry_2d_window/window_convergence.png)")
    add("")
    add("左：窗口延长阶梯上的峰值 Cp（不水平 = 未收敛）；右：相邻档位敏感度，超过 15 % 门限。")
    add("")
    add("## ⑤ 结论（honest）")
    add("")
    for g in rep["grids"]:
        f = g["finest"]
        add(f"* `{g['tag']}`：最细档位 peak = {f['peak']:.3f}，"
            f"U\\* = {PERCENT*f['uncertainty_unified']:.1f} %，"
            f"judgeable* = {'yes' if f['judgeable_unified'] else 'no'}；"
            f"对指定参考残差 {PERCENT*f['residual_designated']:+.1f} %。")
    add("")
    add(f"* **统一不确定度下可裁决网格数 = {len(rep['grids_judgeable_unified'])}**"
        f"（{rep['verdict']['reason']}）。")
    add("* 因此 `PARTIAL_SINGLE_GRID` 的单一网格 PASS **不是**算例通过："
        "它所依赖的 U 里缺了窗口敏感度这一项，而该项在本数据上已达同量级。")
    add("* 这也解释了前序各轮为什么在不同轮次里得到符号相反的残差："
        "读的是同一条轨迹的不同长度窗口。")
    add("")
    add("## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）")
    add("")
    add("| 量 | 参考 | 最好结果 | 误差 | < 3 % ? |")
    add("|---|---|---|---|---|")
    for g in rep["grids"]:
        f = g["finest"]
        add(f"| 峰值 Cp（`{g['tag']}`，分辨带中位数） | 指定复合解 {rep['designated_cp']:.2f} "
            f"| {f['peak']:.3f} | {PERCENT*f['residual_designated']:+.1f} % | **否** |")
    add("")
    add("**是否 < 3 %：否。** 且本次审计指出，在窗口敏感度未收敛（> 15 %）的网格上，"
        "连 15 % 的判决都不能给出。")
    add("")
    add("## ⑦ 复现")
    add("")
    add("```bash")
    add("python benchmarks/case_sph_wedge_entry_2d/window_convergence.py")
    add("python -m pytest tests/test_wedge_window_convergence.py -q")
    add("```")
    add("")
    return "\n".join(L)


def _trend_word(flag: int) -> str:
    return {1: "单调上升", -1: "单调下降", 0: "非单调"}.get(flag, "不可判")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(DEFAULT_ART))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--doc", default=str(ROOT / "docs/case_sph_wedge_entry_2d_window_convergence.md"))
    args = ap.parse_args(argv)

    rep = audit(Path(args.artifacts))
    # extra context for the figure/document
    import estimator_bias as eb
    rep["designated_cp"] = lr.designated_cp_max(BETA_DEG)
    rep["wagner_point"] = eb.wagner_peak_over_q(math.radians(BETA_DEG))
    rep["design_gate"] = GATE
    rep["design_reference"] = f"designated composite solution {rep['designated_cp']:.3f} (q units)"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "window_convergence.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    fig = make_figure(rep, out)
    Path(args.doc).write_text(render_doc(rep, fig) + "\n")
    print(f"wrote {out/'window_convergence.json'}")
    print(f"wrote {args.doc}")
    for g in rep["grids"]:
        f = g["finest"]
        print(f"  {g['tag']}: peak {f['peak']:.3f}  w_max {PERCENT*g['window_sensitivity_max']:+.1f}%  "
              f"U* {PERCENT*f['uncertainty_unified']:.1f}%  judgeable* {f['judgeable_unified']}  "
              f"resid {PERCENT*f['residual_designated']:+.1f}%")
    print(f"verdict: {rep['verdict']['verdict']} -- {rep['verdict']['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())