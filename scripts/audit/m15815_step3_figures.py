#!/usr/bin/env python3
"""m15815 board audit step 3 — figures for the CASE-WAVE-FLUME cross-audit.

fig1: committed m15493 gauge records against the target +/-H/2 envelope, plus the
      blow-up screen (max|v| vs sound speed).
fig2: bar chart of per-gauge height error for the two committed records against
      the 10% gate, with the auditor's independent LS estimate overlaid.
fig3: the reference "ruler" itself -- piston transfer H/S and laminar damping
      mu(T,h) recomputed from closed form (no subject code).
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "m15815"
FIG = REPO / "docs" / "board_audit" / "m15815"
FIG.mkdir(parents=True, exist_ok=True)
SUBJ = Path("/root/tp-agent/work/m15493")

graded = json.loads((RUNS / "grade_committed_records.json").read_text())
indep = json.loads((RUNS / "independent_references.json").read_text())
figs = []


def fig1():
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.2), sharex=False)
    for k, (name, label) in enumerate((("wf_local_D20", "committed run A: h=0.20 m, H=0.030 m"),
                                       ("wf_quick", "committed run B: h=0.15 m, H=0.0225 m"))):
        rec = json.loads((SUBJ / "runs" / (name + ".json")).read_text())
        t = rec["times"]
        eta = rec["probe_eta"][0]
        tgt = graded[name]["grade"]["target_height"]
        a = ax[k]
        a.plot(t, eta, lw=0.7, color="#1f4e79", label="gauge x=%.3f m" % rec["probe_x"][0])
        a.axhline(0.5 * tgt, color="#c00000", ls="--", lw=1.0)
        a.axhline(-0.5 * tgt, color="#c00000", ls="--", lw=1.0,
                  label="target envelope $\\pm H_{target}/2$ (%.4f m)" % tgt)
        d = rec["diagnostics"]
        a.set_title("%s   max|v|=%.1f m/s (c0=%.0f m/s), compression=%.3f, escaped=%d"
                    % (label, d["max_speed"], rec["config"]["sound_speed"],
                       d["max_density_compression"], d["escaped"]), fontsize=9)
        a.set_ylabel("$\\eta$ [m]")
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="upper right")
    ax[1].set_xlabel("t [s]")
    fig.suptitle("fig1  m15493 committed flume records are divergent, not regular waves", fontsize=10)
    fig.tight_layout()
    p = FIG / "fig1_committed_series.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {"name": "fig1_committed_series.png",
            "caption": "m15493 已提交的两条波槽记录的首探头自由面时程 vs 目标 ±H/2 包络；"
                       "幅值、相位均不连贯，且 max|v| 分别达 109.3 / 42.6 m/s（声速 30 m/s），压缩率 0.449/0.101，"
                       "属数值发散而非规则波。"}


def fig2():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width = 0.35
    for k, name in enumerate(("wf_local_D20", "wf_quick")):
        g = graded[name]["grade"]
        xs = [gg["x"] for gg in g["gauges"]]
        errs = [100.0 * gg.get("error_vs_target", float("nan")) for gg in g["gauges"]]
        ind = indep["independent_gauge_heights"][name]
        tgt = ind["target"]
        ierrs = [100.0 * (h["height"] - tgt) / tgt for h in ind["gauges"]]
        ax.bar([i + (k - 0.5) * width for i in range(len(xs))], errs, width,
               label="subject chain (%s)" % name, color="#1f4e79" if k == 0 else "#2e75b6")
        ax.plot([i + (k - 0.5) * width for i in range(len(xs))], ierrs, "o--",
                color="#c00000" if k == 0 else "#ff7f0e", ms=5,
                label="auditor independent LS (%s)" % name)
    ax.axhline(10, color="k", ls="--", lw=1.0, label="gate +/-10%")
    ax.axhline(-10, color="k", ls="--", lw=1.0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["probe 1", "probe 2", "probe 3", "probe 4"])
    ax.set_ylabel("wave-height error vs target [%]")
    ax.set_title("fig2  both committed records fail the 10% height gate (worst 30.5% / 33.8%)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = FIG / "fig2_height_error.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {"name": "fig2_height_error.png",
            "caption": "逐探头波高误差：主体判级链（柱）与审计独立最小二乘谐波估计（点线）对照；"
                       "两记录最差 30.5% / 33.8%，均超 10% 门；两估计器最大差 52%（run A 首探头），"
                       "说明该记录不构成单频规则波，波高判据对窗口高度敏感。"}


def fig3():
    refs = indep["references"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for h in (0.15, 0.25, 0.5, 1.0):
        Ts = [0.4 + 0.03 * i for i in range(70)]
        ratios, mus = [], []
        for T in Ts:
            d = refs.get("_cache_T%s_h%s" % (T, h))
            if d is None:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "s2", str(REPO / "scripts/audit/m15815_step2_references.py"))
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                d = m.dispersion(T, h)
                d["mu"] = m.laminar_decay(d)
                refs["_cache_T%s_h%s" % (T, h)] = d
            ratios.append(2.0 * (math.cosh(2 * d["kh"]) - 1) / (math.sinh(2 * d["kh"]) + 2 * d["kh"]))
            mus.append(d["mu"])
        ax[0].plot(Ts, ratios, lw=1.2, label="h=%.2f m" % h)
        ax[1].semilogy(Ts, mus, lw=1.2, label="h=%.2f m" % h)
    ax[0].set_xlabel("period T [s]"); ax[0].set_ylabel("H/S")
    ax[0].set_title("piston transfer H/S = 2(cosh2kh-1)/(sinh2kh+2kh)", fontsize=8)
    ax[1].set_xlabel("period T [s]"); ax[1].set_ylabel("$\\mu$ [1/m]")
    ax[1].set_title("laminar bed damping $\\mu=\\sigma^2\\sqrt{\\nu\\sigma}/(4g c_g \\sinh^2 kh)$", fontsize=8)
    d = refs["graded_case_T1.0_h0.25"]
    ax[0].plot([1.0], [d["target_height"] / d["target_stroke"]], "k*", ms=10,
               label="graded case T=1.0 h=0.25")
    ax[1].plot([1.0], [d["mu_per_m"]], "k*", ms=10)
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig3_reference_ruler.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {"name": "fig3_reference_ruler.png",
            "caption": "审计独立复算的参考尺子：活塞一阶传递函数 H/S 与层流底边界层衰减 μ(T,h)（闭式，"
                       "未调用被审仓库任何函数；与仓库 laminar_boundary_layer_damping 交叉核对差 <5e-16 相对）。"
                       "分级工况 T=1.0 s, h=0.25 m 给 μ=1.1685e-3 1/m，即 2 m 试验段物理衰减仅 0.233%。"}


if __name__ == "__main__":
    figs = [fig1(), fig2(), fig3()]
    (RUNS / "figures.json").write_text(json.dumps(figs, indent=1, ensure_ascii=False))
    print(json.dumps([f["name"] for f in figs]))