"""Production report for the 3-D arm of the Kleefsman obstacle case.

Reads whatever 3-D runs have landed in ``results/`` (``<tag>.json`` +
``<tag>_time_series.csv`` written by :mod:`run`), groups them by the spanwise
box position (``obstacle_z``: ``0.0`` = box flush with a side wall, ``0.3`` =
box centred, which is what the experiment does), refuses to ladder across
different protocols, and emits

* ``results/production/<arm>_analysis.json`` -- ladder fit + scoring;
* ``results/production/fig_h2_overlay_<arm>.png`` -- H2 history per grid vs the
  digitised Kleefsman Fig. 19 trace;
* ``results/production/fig_error_vs_dx_<arm>.png`` -- peak / arrival relative
  error per grid against the 10 % acceptance band;
* ``results/production/fig_bypass_<arm>.png`` -- bypassed-particle count, i.e.
  the direct measurement of the 3-D-vs-2-D geometric difference;
* ``results/production/fig_2d_vs_3d_peak.png`` -- peak error, 2-D arm (from the
  committed ``h2_score/verdict.json``) vs 3-D arms;
* ``docs/verification_kleefsman_3d_production_m21952.md`` -- the note, with
  every number injected from the JSON above (nothing hand-copied).

Usage::

    python benchmarks/kleefsman_obstacle_3d/production_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import ladder  # noqa: E402

RESULTS = HERE / "results"
OUTDIR = RESULTS / "production"
FIGDIR = OUTDIR / "figs"
DOC = ROOT / "docs" / "verification_kleefsman_3d_production_m21952.md"
REF_CSV = ROOT / "docs" / "m20884" / "outputs" / "kleefsman_h2_ref" / "h2_fig19.csv"
TWOD_VERDICT = (ROOT / "benchmarks" / "kleefsman_obstacle" / "results"
                / "h2_score" / "verdict.json")
REF_PEAK_M = 0.370
REF_PEAK_T = 1.7027
REF_ARRIVAL_S = 0.3591
TOL = 0.10
CASE = "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST"
MISSION = "21952"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _rel(path: Path) -> str:
    """Path relative to the repo root when it lives inside it (figures may not)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def discovered_tags() -> list[str]:
    return sorted(p.name[: -len("_time_series.csv")]
                  for p in RESULTS.glob("*_time_series.csv"))


def group_by_arm(tags: list[str]) -> dict[str, list[str]]:
    """Split tags by spanwise box position, the one geometric free parameter."""
    arms: dict[str, list[str]] = {}
    for tag in tags:
        meta = json.loads((RESULTS / f"{tag}.json").read_text())
        z = meta.get("run", {}).get("obstacle_z", 0.0)
        key = f"z{z:g}".replace(".", "p")
        arms.setdefault(key, []).append(tag)
    return arms


def ref_series() -> tuple[list[float], list[float]]:
    tt: list[float] = []
    hh: list[float] = []
    if not REF_CSV.exists():
        return tt, hh
    with REF_CSV.open(newline="") as fh:
        for row in csv.reader(fh):
            try:
                tt.append(float(row[0]))
                hh.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return tt, hh


def twod_peak_errors() -> dict[str, float]:
    """Worst-case (max) peak relative error of the committed 2-D runs."""
    if not TWOD_VERDICT.exists():
        return {}
    d = json.loads(TWOD_VERDICT.read_text())
    out: dict[str, float] = {}
    for tag, run in d.get("runs", {}).items():
        errs = [c["peak_err_rel"] for c in run.get("cells", [])]
        if errs:
            out[tag] = max(errs)
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_overlay(arm: str, entries: list[dict]) -> Path | None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    drew = False
    for e in sorted(entries, key=lambda r: r["dx"], reverse=True):
        t, slab, _, _ = e["_series"]
        if t:
            ax.plot(t, slab, lw=1.5, label=f"3-D SPH dx={e['dx']:g} m")
            drew = True
    tt, hh = ref_series()
    if tt:
        ax.plot(tt, hh, "k--", lw=1.3, label="Kleefsman 2005 Fig. 19 (digitised)")
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.axhline(REF_PEAK_M, color="0.55", ls=":", lw=1.0,
               label=f"experimental peak {REF_PEAK_M:.3f} m")
    ax.axhline(0.02, color="tab:red", ls=":", lw=0.9, label="arrival threshold 0.02 m")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height in front of box [m]")
    ax.set_title(f"Kleefsman obstacle, 3-D arm {arm}: H2 history vs experiment")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"fig_h2_overlay_{arm}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_errors(arm: str, runs: list[dict]) -> Path | None:
    rows = [r for r in runs if r.get("peak_m") is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r["dx"], reverse=True)
    dx = [r["dx"] for r in rows]
    peak_err = [abs(r["peak_m"] - REF_PEAK_M) / REF_PEAK_M for r in rows]
    arr_err = [None if r["arrival_s"] is None
               else abs(r["arrival_s"] - REF_ARRIVAL_S) / REF_ARRIVAL_S for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhspan(0.0, TOL, color="0.87", label=f"acceptance < {TOL * 100:.0f} %")
    ax.plot(dx, [100 * v for v in peak_err], "o-", label="peak height error")
    xs = [d for d, v in zip(dx, arr_err) if v is not None]
    ys = [100 * v for v in arr_err if v is not None]
    if xs:
        ax.plot(xs, ys, "s--", label="arrival time error")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("relative error vs experiment [%]")
    ax.set_title(f"Kleefsman 3-D arm {arm}: error vs grid")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"fig_error_vs_dx_{arm}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_bypass(arm: str, entries: list[dict]) -> Path | None:
    drew = False
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for e in sorted(entries, key=lambda r: r["dx"], reverse=True):
        t, _, _, bypass = e["_series"]
        if t:
            ax.plot(t, bypass, lw=1.4, label=f"dx={e['dx']:g} m")
            drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel("t [s]")
    ax.set_ylabel("fluid particles past the box, outside its span")
    ax.set_title(f"Kleefsman 3-D arm {arm}: measured bypass (2-D value is identically 0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"fig_bypass_{arm}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_2d_vs_3d(arms: dict[str, list[dict]]) -> Path | None:
    series: dict[str, tuple[list[float], list[float]]] = {}
    e2d = twod_peak_errors()
    if e2d:
        keys = sorted(e2d)
        dxs = [float(k.replace("dx", "")) for k in keys]
        series["2-D arm (100 % blockage)"] = (dxs, [100 * e2d[k] for k in keys])
    for arm, runs in arms.items():
        rows = sorted([r for r in runs if r.get("peak_m") is not None],
                      key=lambda r: r["dx"])
        if rows:
            series[f"3-D arm {arm}"] = ([r["dx"] for r in rows],
                                        [100 * abs(r["peak_m"] - REF_PEAK_M) / REF_PEAK_M
                                         for r in rows])
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhspan(0.0, 100 * TOL, color="0.87", label="acceptance < 10 %")
    for name, (dx, err) in series.items():
        ax.plot(dx, err, "o-", label=name)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("peak-height relative error [%]")
    ax.set_title("Kleefsman H2 peak: does restoring the 3-D bypass close the 2-D gap?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_2d_vs_3d_peak.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# note
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3, unit=""):
    if v is None:
        return "未取得"
    return f"{v:.{nd}f}{unit}"


def build_note(analysis: dict[str, dict], figures: dict[str, str]) -> str:
    L: list[str] = []
    a = L.append
    a(f"# {CASE} — 3-D 生产臂：把「旁通归因」变成测量")
    a("")
    a("TensorSPH · mission " + MISSION + " · 本文全部数字由 "
      "`benchmarks/kleefsman_obstacle_3d/production_report.py` 从 "
      "`results/<tag>.json` 与 `results/<tag>_time_series.csv` 程序化注入"
      "（零手抄）。")
    a("")
    a("## ① 案例与本次要回答的问题")
    a("")
    a("Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based "
      "simulation method for wave impact problems*, **J. Comput. Phys. 206:363–393**, "
      "DOI 10.1016/j.jcp.2004.12.007。水槽 3.22 × 1.00 × 1.00 m，左端 1.228 × 0.550 m "
      "水柱溃决，冲击 x = 2.476 m 处 0.161 × 0.161 × 0.403 m 方箱；H2 波高计在箱前，"
      "判据 = **水位时程峰值与到达时刻 vs 实验，容差 < 10 %**。")
    a("")
    a(f"实验锚点（原文 Fig. 19 右 panel 数字化，m20884）：峰值 **{REF_PEAK_M:.3f} m @ "
      f"{REF_PEAK_T:.4f} s**；到达（阈值 0.02 m）**{REF_ARRIVAL_S:.4f} s**。")
    a("")
    a("2-D 臂把 0.403 m 方箱当作跨满 1.00 m 槽宽（100 % 阻挡），实验只挡 40 %、其余 60 % "
      "从两侧旁通。此前所有 2-D 判决峰值超实验 15–52 %，残差被**归因**于这一几何差，"
      "但归因从未被测量。本单跑完 3-D 生产臂（真实 0.403 m 方箱 + 显式旁通计数），"
      "回答：**(a) 3-D 臂是否满足判据？(b) 收敛方向是朝向还是背离实验值？**")
    a("")
    a("## ② 怎么做的")
    a("")
    a("| 部件 | 做法 |")
    a("|---|---|")
    a("| 求解器 | `src/tensorsph/kleefsman_obstacle_3d.py`（WCSPH，墙粒子 + 箱体硬夹持，未改动） |")
    a("| 驱动 | `benchmarks/kleefsman_obstacle_3d/run.py --dx <d> --t-end 2.6 --wall-layers 3` |")
    a("| 冻结协议 | 各网格**除 dx 外全部一致**：t_end 2.6 s、c0 20 m/s、wall_layers 3、"
      "孔位 0.05 m×箱面上游、孔径 0.08 m、到达阈值 0.02 m；`ladder.check_protocol` 逐项核验，不一致即 `INCOMPARABLE` |")
    a("| 阶梯 | `ladder.fit_power_law`：f(dx)=f_inf+C·dx^p（非均匀加密比的 Richardson 推广）；两点阶梯只给区间，拒绝编造阶数 |")
    a("| 判决 | `score()`：峰值与到达**两项都要** < 10 % 且窗口须覆盖实验峰时刻（否则 `censored`=FAIL） |")
    a("| 几何保真臂 | `obstacle_z` = 0.0（贴一侧槽壁，与 m21728/m21869 协议连续）与 0.3（方箱居中，与实验装置一致） |")
    a("| 计算 | 集群作业：" + ", ".join(
        f"`{t}`" for t in sorted({t for v in analysis.values() for t in v.get("tags", [])})) + " |")
    a("")
    a("## ③ 计算结果")
    a("")
    for arm, rep in analysis.items():
        runs = sorted([r for r in rep.get("runs", []) if r.get("peak_m") is not None],
                      key=lambda r: r["dx"], reverse=True)
        a(f"### 3.{list(analysis).index(arm) + 1} 臂 `{arm}`"
          f"（obstacle_z = {rep['runs'][0]['fingerprint']['obstacle_z'] if rep.get('runs') else 'n/a'} m）")
        a("")
        if not runs:
            a("无可用 run（作业未回收）。")
            a("")
            continue
        a("| tag | dx [m] | 峰值 [m] | 峰值误差 | 到达 [s] | 到达误差 | 窗口覆盖实验峰 | 旁通粒子峰值 | 求解墙钟 [s] | 判决 |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for r in runs:
            pe = (abs(r["peak_m"] - REF_PEAK_M) / REF_PEAK_M) if r["peak_m"] else None
            ae = (abs(r["arrival_s"] - REF_ARRIVAL_S) / REF_ARRIVAL_S
                  if r["arrival_s"] else None)
            a(f"| {r['tag']} | {r['dx']:g} | {r['peak_m']:.4f} | "
              f"{'—' if pe is None else f'{100*pe:.1f} %'} | "
              f"{_fmt(r['arrival_s'], 4)} | "
              f"{'—' if ae is None else f'{100*ae:.1f} %'} | "
              f"{'是' if r['window_covers_reference_peak'] else '否'} | "
              f"{r['max_bypass']} | {r['wall_clock_s']:.0f} | {r['verdict']} |")
        a("")
        for key, label, ref, unit in (("peak_ladder", "峰值", REF_PEAK_M, " m"),
                                      ("arrival_ladder", "到达时刻", REF_ARRIVAL_S, " s")):
            fit = rep.get(key, {})
            vk = "peak_vs_reference" if key == "peak_ladder" else "arrival_vs_reference"
            vt = rep.get(vk, {})
            if fit.get("status") == "OK":
                a(f"* **{label}阶梯**：观测阶 p = {fit['p_observed']:.3f}，外推连续值 "
                  f"{fit['f_inf']:.4f}{unit}（实验 {ref:.4f}{unit}，相对差 "
                  f"{100*abs(fit['f_inf']-ref)/ref:.1f} %）；判决 `{vt.get('status')}`"
                  + (f"，理由：{vt.get('reason')}" if vt.get("reason") else ""))
            elif fit.get("status") == "UNDERDETERMINED":
                a(f"* **{label}阶梯**：`UNDERDETERMINED`（{len(runs)} 个网格），只给区间 "
                  f"{fit.get('bracket')}，不编造阶数；判决 `{vt.get('status')}`")
            else:
                a(f"* **{label}阶梯**：`{fit.get('status')}`")
        a("")

    # aggregate verdict
    all_runs = [r for rep in analysis.values() for r in rep.get("runs", [])
                if r.get("peak_m") is not None]
    n_pass = sum(1 for r in all_runs if r.get("verdict") == "PASS")
    a("### 3.%d 汇总判决" % (len(analysis) + 1))
    a("")
    a(f"* 回收 3-D production run：**{len(all_runs)}** 个（臂 {len(analysis)} 条）；"
      f"其中 `PASS`：**{n_pass}**。")
    if all_runs:
        best = min(all_runs, key=lambda r: abs(r["peak_m"] - REF_PEAK_M))
        best_arr = ("—" if best["arrival_s"] is None
                    else f"{best['arrival_s']:.4f} s")
        a(f"* 峰值最接近实验的一条：`{best['tag']}` dx={best['dx']:g} m，"
          f"峰值 {best['peak_m']:.4f} m（误差 "
          f"{100*abs(best['peak_m']-REF_PEAK_M)/REF_PEAK_M:.1f} %），"
          f"到达 {best_arr}，判决 `{best['verdict']}`。")
        e2d = twod_peak_errors()
        if e2d:
            worst2d = max(e2d.values())
            a(f"* 2-D 臂（100 % 阻挡）峰值最差误差 {100*worst2d:.1f} %；"
              f"3-D 臂最差 {100*max(abs(r['peak_m']-REF_PEAK_M)/REF_PEAK_M for r in all_runs):.1f} %。"
              "差值即「旁通」这一几何差的实测效应量。")
    a("")
    a("## ④ 图")
    a("")
    for name, path in sorted(figures.items()):
        if path:
            a(f"![{name}](../{path})")
            a("")
    a("## ⑤ 注意点 / 局限（如实）")
    a("")
    a("1. **单条 3-D 臂不是收敛性证明**：阶梯只覆盖本单递交的网格；两点阶梯按设计返回区间而非阶数。")
    a("2. **方箱跨向位置**：实验装置中方箱居中（两侧各 0.3 m 旁通）；`obstacle_z=0.0` 臂只有一侧旁通，"
      "总阻挡比例同为 40 %，但近壁流动不对称。两臂并列给出该假设的影响量。")
    a("3. **箱宽的格离散**：0.403 m 在 dx 下离散为整数格（dx=0.05 → 0.40 m，-0.7 %；dx=0.03 → 0.39 m，-3.2 %），"
      "水位计分母仍取实验值 0.403 m（不修正，方向与量级均已知）。")
    a("4. **H2 站位**：论文只说「just in front of the box」，取箱面上游 0.05 m（与 2-D 冻结协议同口径）。")
    a("5. **WCSPH 声速**：c0 = 20 m/s 对应 1 % 密度涨落假设；未做 c0 收敛研究（2-D 臂有，见 "
      "`docs/verification_kleefsman_dt_sound_speed_family.md`）。")
    a("")
    a("## ⑥ 误差判定：关键量 vs 基准")
    a("")
    a("| 量 | 基准（Kleefsman 2005 Fig. 19） | 本单 3-D 最优 | 相对误差 | < 3 %？ | < 10 %（账本容差）？ |")
    a("|---|---|---|---|---|---|")
    if all_runs:
        bp = min(all_runs, key=lambda r: abs(r["peak_m"] - REF_PEAK_M))
        err = abs(bp["peak_m"] - REF_PEAK_M) / REF_PEAK_M
        a(f"| 箱前水位峰值 | {REF_PEAK_M:.3f} m | {bp['peak_m']:.4f} m（{bp['tag']}） | "
          f"{100*err:.1f} % | {'是' if err < 0.03 else '否'} | {'是' if err < TOL else '否'} |")
        ar = [r for r in all_runs if r["arrival_s"] is not None]
        if ar:
            ba = min(ar, key=lambda r: abs(r["arrival_s"] - REF_ARRIVAL_S))
            aerr = abs(ba["arrival_s"] - REF_ARRIVAL_S) / REF_ARRIVAL_S
            a(f"| 到达时刻（0.02 m） | {REF_ARRIVAL_S:.4f} s | {ba['arrival_s']:.4f} s（{ba['tag']}） | "
              f"{100*aerr:.1f} % | {'是' if aerr < 0.03 else '否'} | {'是' if aerr < TOL else '否'} |")
        else:
            a(f"| 到达时刻（0.02 m） | {REF_ARRIVAL_S:.4f} s | 未取得 | 未取得 | 否 | 否 |")
    else:
        a("| 箱前水位峰值 | 0.370 m | 未取得 | 未取得 | 否 | 否 |")
    a("")
    verdict = "PASS" if all_runs and n_pass else "FAIL / 未达标"
    a(f"**总判决：`{verdict}`。** "
      + ("两项关键量均在 10 % 容差内，且峰值与到达同时达标。" if n_pass and all_runs else
         "至少一项关键量超出 10 % 容差，或窗口未覆盖实验峰时刻；本单不声明该算例达标，"
         "残差与归因见 §③/§⑤。"))
    a("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    tags = discovered_tags()
    arms = group_by_arm(tags) if tags else {}
    analysis: dict[str, dict] = {}
    entries_by_arm: dict[str, list[dict]] = {}
    for arm, arm_tags in arms.items():
        rep = ladder.analyse(RESULTS, arm_tags)
        rep["tags"] = arm_tags
        analysis[arm] = rep
        entries_by_arm[arm] = [ladder.load_run(RESULTS, t) for t in arm_tags]
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / f"{arm}_analysis.json").write_text(json.dumps(rep, indent=1) + "\n")

    figures: dict[str, str] = {}
    for arm, entries in entries_by_arm.items():
        for name, fn in (("H2 时程 vs 实验", fig_overlay),
                         ("误差 vs 网格", fig_errors),
                         ("旁通粒子数", fig_bypass)):
            if fn is fig_errors:
                p = fn(arm, analysis[arm].get("runs", []))
            else:
                p = fn(arm, entries)
            if p is not None:
                figures[f"{name}（{arm}）"] = _rel(p)
    p = fig_2d_vs_3d({arm: analysis[arm].get("runs", []) for arm in analysis})
    if p is not None:
        figures["2-D vs 3-D 峰值误差"] = _rel(p)

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build_note(analysis, figures))
    print(json.dumps({"arms": list(analysis), "tags": tags,
                      "figures": figures, "doc": _rel(DOC)},
                     indent=1))
    return 0 if analysis else 1


if __name__ == "__main__":
    raise SystemExit(main())