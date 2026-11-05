"""Figure for the TensorSPH PR #110 (agent/m3265) conflict-resolution report.

Panels
------
(a) relaxation coverage ``t / t_sigma`` of the three committed convergence runs
    before the cure (base), on the PR side (m3265) and on the merged main side;
(b) spurious-current ratio ``max|v| / sqrt(sigma/(rho R))`` against the
    Adami et al. (2012) criterion 0.1;
(c) curvature band mean against the exact 1/R;
(d) reproduction parity: the committed ``relax_dx0.0020.json`` versus a fresh
    run of ``benchmarks/laplace_law/run.py`` on the merged tree.

All numbers are read from the git objects / json files -- nothing is typed by
hand.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASE = "184fef9"
SIDES = {"base (184fef9)": BASE, "PR m3265": "origin/agent/m3265", "main (merged)": "origin/main"}
CASES = ["relax_dx0.0015", "relax_dx0.0020", "relax_dx0.0025"]
COLORS = {"base (184fef9)": "#888888", "PR m3265": "#1f77b4", "main (merged)": "#d62728"}


def load(ref: str, case: str) -> dict:
    path = f"benchmarks/laplace_law/convergence/{case}.json"
    txt = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=REPO)
    return json.loads(txt)


def main() -> None:
    data = {name: {c: load(ref, c) for c in CASES} for name, ref in SIDES.items()}
    merged = json.loads(
        (REPO / "benchmarks/laplace_law/convergence/relax_dx0.0020.json").read_text()
    )
    repro_path = Path("/tmp/m3950_repro_0002.json")
    repro = json.loads(repro_path.read_text())["relax"] if repro_path.exists() else None

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    x = range(len(CASES))
    width = 0.26

    def bars(ax, field, title, ylabel, extra=None, logy=False):
        for i, (name, series) in enumerate(data.items()):
            vals = [field(series[c]) for c in CASES]
            bars_ = ax.bar([k + (i - 1) * width for k in x], vals, width,
                           label=name, color=COLORS[name], alpha=0.85)
            for rect, v in zip(bars_, vals):
                ax.annotate(f"{v:.3g}", (rect.get_x() + rect.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=7, rotation=90)
        if extra:
            extra(ax)
        ax.set_xticks(list(x))
        ax.set_xticklabels([c.replace("relax_", "") for c in CASES])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    def gate(ax):
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.text(len(CASES) - 0.45, 0.52, "gate t/t_sigma>0.5", fontsize=7, ha="right")

    bars(axes[0, 0], lambda d: d["setup"]["t_over_t_sigma"],
         "(a) relaxation coverage of committed convergence runs",
         r"$t / t_\sigma$", gate)

    def adami(ax):
        ax.axhline(0.1, color="k", ls="--", lw=1)
        ax.text(len(CASES) - 0.45, 0.105, "Adami 2012 criterion 0.1", fontsize=7, ha="right")

    bars(axes[0, 1], lambda d: d["spurious_current"]["ratio"],
         "(b) spurious current vs literature threshold",
         r"$\max|v| / \sqrt{\sigma/(\rho R)}$", adami)

    def curv(ax):
        ax.axhline(50.0, color="k", ls="--", lw=1)
        ax.text(-0.45, 51.0, "exact $1/R = 50\\ \\mathrm{m^{-1}}$", fontsize=7)

    bars(axes[1, 0], lambda d: d["curvature_final"]["band_mean_1_per_m"],
         "(c) curvature band mean (consistent scheme)", r"band mean $[1/\mathrm{m}]$", curv)

    ax = axes[1, 1]
    if repro is not None:
        keys = ["laplace/rel_error_pct", "curvature_final/band_mean_1_per_m"]
        labels = ["Laplace $\\Delta p$\nrel. error [%]", "curvature band\nmean [1/m]"]
        committed_vals = [merged["laplace"]["rel_error_pct"],
                          merged["curvature_final"]["band_mean_1_per_m"]]
        repro_vals = [repro["laplace"]["rel_error_pct"],
                      repro["curvature_final"]["band_mean_1_per_m"]]
        idx = range(len(keys))
        ax.bar([i - 0.17 for i in idx], committed_vals, 0.34,
               label="committed JSON", color="#d62728", alpha=0.85)
        ax.bar([i + 0.17 for i in idx], repro_vals, 0.34,
               label="fresh run on merged tree", color="#2ca02c", alpha=0.85)
        for i, (c, r) in enumerate(zip(committed_vals, repro_vals)):
            ax.annotate(f"{c:.6g}", (i - 0.17, c), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7)
            ax.annotate(f"{r:.6g}", (i + 0.17, r), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7)
        ax.set_xticks(list(idx))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("(d) reproduction parity, relax_dx0.0020 (900 steps)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    axes[0, 0].legend(fontsize=7)
    fig.suptitle("TensorSPH PR #110 (agent/m3265) conflict resolution: "
                 "committed convergence data after the three-way merge", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_name("m3950-pr110-m3265-merge.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
