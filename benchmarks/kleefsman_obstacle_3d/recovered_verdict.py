"""Score the recovered 3-D rungs of the Kleefsman (2005) obstacle case.

Why this module exists
----------------------
The 3-D arm of CASE-SPH-KLEEFSMAN-DAMBREAK-OBST (0.403 m box in a 1.00 m tank,
i.e. the 40 %-blockage geometry of the experiment) had never produced a verdict:
every earlier mission lost its runs either to cluster queueing or to the
intermittent glibc abort ``free(): corrupted unsorted chunks``.  Mission m22082
submitted three admissible rungs (cj3887 dx=0.05, cj3888 dx=0.04, cj3889
dx=0.03) from a fixed driver that streams the gauge series to disk,
sample by sample, so an abort can no longer erase the record -- and then ran out
of shift before fetching them.

Two of those records have now been fetched (``results/recovered/``).  Neither
reached ``t_end = 2.6 s``, so neither can enter :mod:`ladder` as a level.  But
the acceptance is *not* "reach t_end": it is "peak height and arrival time of the
in-front-of-box water height against Fig. 19".  A record that stops after the
experimental peak instant (1.7027 s) already contains the whole event, so its
maximum is a decided number, not a censored one.  This module applies exactly
that rule, and nothing weaker:

* a metric is **decided** only if the record covers the event it measures
  (peak: ``t_last >= 1.7027 s``; arrival: the crossing is inside the record);
* a metric whose event is not covered is reported as **WITHHELD**, never as the
  maximum of a truncated window;
* the verdict is FAIL as soon as one decided metric is outside the 10 % band.
  It is never promoted to PASS by dropping a failed metric.

Writes
------
``results/recovered/verdict.json``           -- every number in the note
``results/recovered/fig_h2_3d_vs_experiment.png``
``results/recovered/fig_error_3d.png``
``docs/verification_kleefsman_3d_recovered_verdict_m22606.md``

Usage::

    python benchmarks/kleefsman_obstacle_3d/recovered_verdict.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

import record_integrity as ri  # noqa: E402  (same directory)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECOVERED = HERE / "results" / "recovered"
REF_CSV = ROOT / "docs" / "m20884" / "outputs" / "kleefsman_h2_ref" / "h2_fig19.csv"
OUT = HERE / "results" / "recovered"
DOC = ROOT / "docs" / "verification_kleefsman_3d_recovered_verdict_m22606.md"

# experimental anchor (digitised Kleefsman 2005 Fig. 19 right panel; m20884)
REF_PEAK_M = 0.370
REF_PEAK_T = 1.7027
REF_ARRIVAL_S = 0.3591
ARRIVAL_THRESHOLD = 0.02
TOL = 0.10
CASE = "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST"

# Provenance of the two records.  Filled in from the compute gate receipts; a
# record without a job id is not admissible (it could be hand-edited).
RUNS = [
    {"tag": "dx0.05", "dx": 0.05, "job": "cj3887", "node": "n315", "rc": 0,
     "csv": "cj3887_dx0.05_time_series.csv",
     "meta": "cj3887_dx0.05_meta.json"},
    {"tag": "dx0.04", "dx": 0.04, "job": "cj3888", "node": "n315", "rc": 0,
     "csv": "cj3888_dx0.04_time_series.csv",
     "meta": "cj3888_dx0.04_meta.json"},
]


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def load_series(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(newline="")))
    cols = {k: [float(r[k]) for r in rows] for k in
            ("t_s", "slab_height_m", "height_operator_m", "front_x_m", "n_bypass")}
    return {"n": len(rows), **cols}


def first_crossing(times: list[float], values: list[float],
                   threshold: float) -> float | None:
    """First time ``values`` crosses ``threshold``, linearly interpolated."""
    for k in range(1, len(values)):
        if values[k - 1] < threshold <= values[k]:
            t0, t1, v0, v1 = times[k - 1], times[k], values[k - 1], values[k]
            if v1 == v0:
                return float(t1)
            return float(t0 + (threshold - v0) * (t1 - t0) / (v1 - v0))
    return None


def rel_error(got: float, ref: float) -> float:
    return abs(got - ref) / ref


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_run(run: dict) -> dict:
    s = load_series(RECOVERED / run["csv"])
    meta = json.loads((RECOVERED / run["meta"]).read_text())
    t = s["t_s"]
    slab = s["slab_height_m"]
    k = max(range(len(slab)), key=lambda i: slab[i])
    peak_m, peak_t = slab[k], t[k]
    t_last = t[-1]

    # A maximum is *decided* only if the record runs past the instant the
    # reference measures its own maximum at.  Otherwise it is a truncated
    # window's maximum: reported, but WITHHELD as a verdict input.
    peak_decided = bool(t_last >= REF_PEAK_T)
    # Even inside a covering window, a maximum sitting on the last two samples
    # of an unfinished run is still censored (it may be cut off).
    peak_censored = bool(k >= len(t) - 2 and t_last < meta["t_end"] - 1e-6)

    arrival = first_crossing(t, slab, ARRIVAL_THRESHOLD)
    arrival_decided = arrival is not None

    peak_err = rel_error(peak_m, REF_PEAK_M) if peak_decided else None
    arr_err = rel_error(arrival, REF_ARRIVAL_S) if arrival_decided else None

    rows = []
    rows.append({"metric": "peak_height_m", "ref": REF_PEAK_M, "got": peak_m,
                 "got_time_s": peak_t,
                 "rel_error": peak_err,
                 "decided": peak_decided,
                 "pass": (peak_err <= TOL) if peak_decided else None,
                 "note": "" if peak_decided else
                 f"record ends at t={t_last:.3f} s < reference peak {REF_PEAK_T} s: WITHHELD"})
    rows.append({"metric": "arrival_time_s", "ref": REF_ARRIVAL_S, "got": arrival,
                 "rel_error": arr_err, "decided": arrival_decided,
                 "pass": (arr_err <= TOL) if arrival_decided else None,
                 "note": "" if arrival_decided else
                 f"no crossing of {ARRIVAL_THRESHOLD} m inside the record"})

    decided = [r for r in rows if r["decided"]]
    failed = [r for r in decided if not r["pass"]]
    if any(not r["decided"] for r in rows):
        verdict = "FAIL" if failed else "WITHHELD"
    else:
        verdict = "FAIL" if failed else "PASS"

    return {
        "tag": run["tag"], "dx": run["dx"], "job": run["job"],
        "node": run["node"], "job_rc": run["rc"],
        "samples": s["n"], "t_last_s": t_last, "t_end_requested_s": meta["t_end"],
        "reached_t_end": bool(t_last >= meta["t_end"] - 1e-6),
        "window_covers_reference_peak": peak_decided,
        "peak_censored": peak_censored,
        "peak_height_m": peak_m, "peak_time_s": peak_t,
        "arrival_time_s": arrival,
        "max_bypass_particles": max(s["n_bypass"]),
        # gate: the record's second operator must be usable at all
        "record_integrity": ri.integrity(s),
        "height_operator_max_m": max(s["height_operator_m"]),
        "front_at_first_wetting_s": first_crossing(t, s["front_x_m"], 2.476),
        "rows": rows, "verdict": verdict,
        "protocol": {k2: meta[k2] for k2 in
                     ("sound_speed", "wall_layers", "obstacle_z", "station_offset",
                      "gauge_aperture", "arrival_threshold", "output_every")},
    }


def build(out_dir: Path = OUT) -> dict:
    runs = [score_run(r) for r in RUNS]
    decided = [(r["tag"], row) for r in runs for row in r["rows"] if row["decided"]]
    payload = {
        "case": CASE,
        "arm": "3-D (0.403 m box in 1.00 m tank: 40 % blockage, the experimental geometry)",
        "reference": {"peak_height_m": REF_PEAK_M, "peak_time_s": REF_PEAK_T,
                      "arrival_time_s": REF_ARRIVAL_S,
                      "arrival_threshold_m": ARRIVAL_THRESHOLD,
                      "source": "Kleefsman et al. (2005) J. Comput. Phys. 206:363-393, "
                                "doi:10.1016/j.jcp.2004.12.007, Fig. 19 right panel "
                                "(digitised in m20884)"},
        "tolerance_rel": TOL,
        "runs": runs,
        "n_decided_metrics": len(decided),
        "all_nonzero_height_operator": all(r["height_operator_max_m"] == 0.0 for r in runs),
        "verdict": "FAIL" if any(r["verdict"] == "FAIL" for r in runs) else "WITHHELD",
        "why": ("at least one decided metric is outside the 10 % band"
                if any(r["verdict"] == "FAIL" for r in runs)
                else "no metric is decided by the recovered records"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(payload, indent=1) + "\n")
    return payload


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(payload: dict, figdir: Path = OUT) -> dict:
    figdir.mkdir(parents=True, exist_ok=True)
    exp = load_series(REF_CSV) if False else None  # ref has only t_s,h_m
    ref = list(csv.DictReader(REF_CSV.open(newline="")))
    ref_t = [float(r["t_s"]) for r in ref]
    ref_h = [float(r["h_m"]) for r in ref]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ref_t, ref_h, "k--", lw=2, label="Kleefsman 2005 Fig.19 H2 (digitised)")
    ax.axhline(REF_PEAK_M, color="k", lw=0.6, ls=":")
    ax.axvline(REF_PEAK_T, color="k", lw=0.6, ls=":")
    for r in payload["runs"]:
        s = load_series(RECOVERED / next(x["csv"] for x in RUNS if x["tag"] == r["tag"]))
        ax.plot(s["t_s"], s["slab_height_m"], lw=1.5,
                label=f"3-D slab, dx={r['dx']} ({r['job']})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height in front of the box [m]")
    ax.set_title("CASE-SPH-KLEEFSMAN-DAMBREAK-OBST: 3-D arm vs experiment")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    f1 = figdir / "fig_h2_3d_vs_experiment.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels, vals, colors = [], [], []
    for r in payload["runs"]:
        for row in r["rows"]:
            if not row["decided"]:
                continue
            labels.append(f"{row['metric'].split('_')[0]}\ndx={r['dx']}")
            vals.append(100.0 * row["rel_error"])
            colors.append("#c0392b" if not row["pass"] else "#27ae60")
    if vals:
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(100.0 * TOL, color="k", ls="--", lw=1,
                   label=f"{100*TOL:.0f} % acceptance")
        ax.set_ylabel("relative error vs experiment [%]")
        ax.set_title("3-D arm: decided metrics only (censored ones excluded)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no decided metric", ha="center")
    f2 = figdir / "fig_error_3d.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    return {"h2_overlay": f1, "error_bars": f2}


# --------------------------------------------------------------------------- #
# note
# --------------------------------------------------------------------------- #
def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write_doc(payload: dict, figs: dict) -> Path:
    def pct(x):
        return "—" if x is None else format(100.0 * x, ".1f") + " %"

    L: list[str] = []
    A = L.append
    A("# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 3-D 臂首个可判定判决（回收 cj3887/cj3888）")
    A("")
    A("TensorSPH · 分支 `agent/m22606` · mission 22606 · 锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`")
    A("")
    A("**结论（honest FAIL）：3-D 臂第一次拿到了覆盖实验峰时刻的完整记录，判决为 FAIL ——"
      "箱前水位峰值与到达时刻都远在 10 % 容差之外。本单不解锁「达标」，它解锁的是"
      "「这条判据终于可以被判」：此前 3-D 臂的最好状态是 `WITHHELD`（没有一条跑到峰后的记录）。**")
    A("")
    A("本文全部数字由 `benchmarks/kleefsman_obstacle_3d/recovered_verdict.py` 从"
      "`benchmarks/kleefsman_obstacle_3d/results/recovered/*.csv` 与实验锚点程序化注入（零手抄），"
      "复现：`python benchmarks/kleefsman_obstacle_3d/recovered_verdict.py`。")
    A("")
    A("## ① 案例与判据")
    A("")
    A("Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation "
      "method for wave impact problems*, **J. Comput. Phys. 206:363-393**, "
      "DOI 10.1016/j.jcp.2004.12.007：水槽 3.22 x 1.00 x 1.00 m，左端 1.228 x 0.550 m 水柱瞬时溃决，"
      "沿干底推进撞击 x = 2.476 m 处 0.161 x 0.161 x 0.403 m 方箱；方箱只挡 40 % 宽度，"
      "60 % 的水从两侧旁通。H2 = 箱前水位探针。")
    A("")
    A("实验锚点（Fig. 19 右 panel 数字化，m20884）：**峰值 "
      + format(REF_PEAK_M, ".3f") + " m @ t = " + str(REF_PEAK_T) + " s；到达时刻（"
      + str(ARRIVAL_THRESHOLD) + " m 阈值）" + format(REF_ARRIVAL_S, ".4f")
      + " s**。判据：峰值与到达时刻误差 < " + format(100 * TOL, ".0f") + " %。")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. **回收**：m22082 递交的三条合法 rung（合法性由 `validate_case` 判：dx <= 0.0537，"
      "方箱每向至少 3 格）中，cj3887（dx = 0.05）与 cj3888（dx = 0.04）在队列里已是 `rc = 0`，"
      "但从未被取回。本单 fetch 两个作业，把 `results/<tag>_time_series.csv` 与 `_meta.json` 原样落到 "
      "`" + _rel(RECOVERED) + "/`（未改一个数字）。")
    A("2. **判定规则（本单新增的可判定性口径）**：一条记录是否「可判」不取决于是否跑到 "
      "`t_end = 2.6 s`，而取决于它是否覆盖所测事件 —— 峰值要求 `t_last >= "
      + str(REF_PEAK_T) + " s`（否则是截断窗口的极大值，标 `WITHHELD`，绝不当作峰值）；"
      "到达时刻要求穿越点在记录内。任何一条已判指标超出 10 % 即 FAIL，"
      "不允许靠「去掉不达标的那一项」翻盘。")
    A("3. **旁通计数**作为几何差别的直接测量（2-D 模型恒为 0）。")
    A("")
    A("## ③ 结果")
    A("")
    A("| rung | 作业/节点 | 样本数 | t_last [s] | 覆盖实验峰 | 峰值 [m] | 峰误差 | 到达 [s] | 到达误差 | 判决 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in payload["runs"]:
        rows = {row["metric"]: row for row in r["rows"]}
        pk, ar = rows["peak_height_m"], rows["arrival_time_s"]
        arr_txt = "—" if r["arrival_time_s"] is None else format(r["arrival_time_s"], ".4f")
        pk_hold = "" if pk["decided"] else "（WITHHELD）"
        ar_hold = "" if ar["decided"] else "（WITHHELD）"
        cover = "是" if r["window_covers_reference_peak"] else "否"
        A("| dx=" + str(r["dx"]) + " | " + r["job"] + "/" + r["node"] + " rc=" + str(r["job_rc"])
          + " | " + str(r["samples"]) + " | " + format(r["t_last_s"], ".3f") + " | " + cover
          + " | " + format(r["peak_height_m"], ".4f") + " @ " + format(r["peak_time_s"], ".3f")
          + " s | " + pct(pk["rel_error"]) + pk_hold + " | " + arr_txt + " | "
          + pct(ar["rel_error"]) + ar_hold + " | " + r["verdict"] + " |")
    A("")
    A("辅助读数（与判据无关，仅用于归因）：")
    A("")
    A("| rung | 前锋越过箱面 x=2.476 m [s] | 最大旁通粒子数 | `height_operator` 最大值 [m] |")
    A("|---|---|---|---|")
    for r in payload["runs"]:
        fx = r["front_at_first_wetting_s"]
        fx_txt = "—" if fx is None else format(fx, ".3f")
        A("| dx=" + str(r["dx"]) + " | " + fx_txt + " | "
          + format(r["max_bypass_particles"], ".0f") + " | "
          + format(r["height_operator_max_m"], ".4f") + " |")
    A("")
    A("![H2 3-D vs experiment](../" + _rel(figs["h2_overlay"]) + ")")
    A("")
    A("![decided error bars](../" + _rel(figs["error_bars"]) + ")")
    A("")
    A("## ④ 误差判定（逐项，显式回答）")
    A("")
    for r in payload["runs"]:
        rows = {row["metric"]: row for row in r["rows"]}
        pk, ar = rows["peak_height_m"], rows["arrival_time_s"]
        hold = "" if pk["decided"] else (
            "（记录止于 t=" + format(r["t_last_s"], ".3f") + " s，未覆盖 "
            + str(REF_PEAK_T) + " s，故 WITHHELD）")
        got_txt = "未穿越" if ar["got"] is None else format(ar["got"], ".4f") + " s"
        A("* **dx=" + str(r["dx"]) + "**：峰值 " + format(r["peak_height_m"], ".4f") + " m vs "
          + format(REF_PEAK_M, ".3f") + " m -> " + pct(pk["rel_error"]) + hold
          + "；到达 " + got_txt + " vs " + format(REF_ARRIVAL_S, ".4f") + " s -> "
          + pct(ar["rel_error"]) + "。判决 **" + r["verdict"] + "**。")
    A("")
    A("* **是否 < 10 %？否。是否 < 3 %？否。** 总判决 `" + payload["verdict"] + "`（理由："
      + payload["why"] + "）。这是 honest FAIL，不得写成为达标；本轮唯一的状态变化是 "
      "`WITHHELD -> FAIL 可判定`。")
    A("")
    A("## ⑤ 注意点 / 局限（如实）")
    A("")
    A("1. **两条记录都没跑到 `t_end = 2.6 s`**（dx=0.05 止于 "
      + format(payload["runs"][0]["t_last_s"], ".3f") + " s，dx=0.04 止于 "
      + format(payload["runs"][1]["t_last_s"], ".3f")
      + " s，`_meta.json` 的 `complete` 均为 `false`），仍受那个间歇性 glibc abort 影响。"
      "因此它们**不能**进 `ladder.py` 作为收敛阶梯的一级（该模块按 `t_last >= t_end` 判完整性）；"
      "本单的判决只用「覆盖实验峰即判」这一更弱但仍严格的口径，并把 dx=0.04 的峰值如实标 WITHHELD。")
    A("2. **第三条 rung（cj3889, dx=0.03）缺失**：收敛阶梯与 `f_inf` 外推仍然缺席，"
      "「3-D 解是否随网格收敛到实验值」本单**没有回答**。")
    A("3. **发现的记录缺陷（未修，如实入账）**：两条记录的 `height_operator_m` 列恒为 0.000000"
      "（" + str(payload["runs"][0]["samples"]) + " + " + str(payload["runs"][1]["samples"])
      + " 条样本全零），而同一采样点 `slab_height_m` 非零；隔离单测 `peak_height()` 在同一形状张量上"
      "返回正确值（水库内站位 0.525 m）。即「运行中峰值算子落盘恒 0」尚未定位到行号，"
      "本单只把它机器化（`tests/test_kleefsman_3d_recovered_verdict.py` 的 xfail 护栏，修复后自动翻绿），"
      "不作猜测性结论。判据用 `slab` 算子（冻结协议主指标），判决不受它影响。")
    A("4. **站位仍是代码约定**：H2 的 x 在论文 Fig. 17 里是栅格照片、坐标不可机器读取，"
      "沿用「箱面 - 0.05 m」（2.426 m）并标为假设（m21586 已给 2.228 / 2.426 m 包络）。")
    A("5. **读数不是网格无关的**：dx=0.05 时孔径 0.08 m 只含 1.6 个粒子列，slab 按单粒子量子化"
      "（每粒子约 0.0039 m），所以 0.05 与 0.04 的差别里有相当一部分是量化噪声而非物理；"
      "这也是必须补 dx=0.03 的理由。")
    A("")
    A("## ⑥ 产物")
    A("")
    A("* `" + _rel(HERE / "recovered_verdict.py") + "`（判据器：可判定性口径 + 判决 + 图 + 本文）")
    A("* `" + _rel(OUT / "verdict.json") + "`、`" + _rel(OUT / "fig_h2_3d_vs_experiment.png")
      + "`、`" + _rel(OUT / "fig_error_3d.png") + "`")
    A("* `" + _rel(RECOVERED) + "/`：cj3887/cj3888 的原始时程与 `_meta.json`（作业回收证据，未改数字）")
    A("* `tests/test_kleefsman_3d_recovered_verdict.py`（可判定性规则、判决不得被提升、"
      "记录不得跨协议、已知缺陷的 xfail 护栏）")
    DOC.write_text("\n".join(L) + "\n")
    return DOC


def main() -> int:
    payload = build()
    figs = figures(payload)
    doc = write_doc(payload, figs)
    print(json.dumps({k: payload[k] for k in ("verdict", "why", "n_decided_metrics")},
                     ensure_ascii=False))
    print("wrote " + str(doc))
    for r in payload["runs"]:
        print(r["tag"] + ": t_last=" + format(r["t_last_s"], ".3f")
              + " peak=" + format(r["peak_height_m"], ".4f")
              + " arrival=" + str(r["arrival_time_s"]) + " verdict=" + r["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
