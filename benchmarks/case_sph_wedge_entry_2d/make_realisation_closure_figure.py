"""Figure for the realization-closure document (CASE-SPH-WEDGE-ENTRY-2D).

Reads the aggregated ensemble artefact and draws, with numbers never typed by hand:

* left  panel: every realization's same-operator residual per grid (dots), the
  realization-mean +/- ``U_ci`` and the ``U_band`` bar, against the 15 % gate;
* right panel: ``U_real`` vs ``dx`` with the gate line, each grid marked
  judgeable / not judgeable.

    python benchmarks/case_sph_wedge_entry_2d/make_realisation_closure_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ENS = (ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_realisations"
       / "seed_ensemble.json")
OUT = ROOT / "docs" / "figures" / "wedge_realisation_closure_m22929.png"
GATE = 0.15


def main() -> int:
    ens = json.loads(ENS.read_text())
    grids = sorted(ens["grids"], key=lambda g: -g["dx"])
    tags = [g["tag"] for g in grids]
    x = list(range(len(grids)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    for xi, g in zip(x, grids):
        nom = [r for r in g["realisations"] if r["scale"] == 1.0]
        res = [r["residual"] for r in nom if r["residual"] == r["residual"]]
        if res:
            ax0.scatter([xi] * len(res), [v * 100 for v in res], s=46,
                        color="#1f77b4", zorder=3,
                        label="realizations" if xi == x[0] else None)
            mean = sum(res) / len(res)
            ax0.plot([xi], [mean * 100], marker="_", markersize=22, mew=3,
                     color="#d62728", zorder=4,
                     label="realization mean" if xi == x[0] else None)
            if g["U_realisation_ci"] == g["U_realisation_ci"]:
                u = g["U_realisation_ci"] * 100
                ax0.errorbar([xi + 0.16], [mean * 100], yerr=[[u], [u]],
                             fmt="none", ecolor="#d62728", capsize=5, lw=1.6,
                             zorder=4)
        if g["U_band"] == g["U_band"]:
            ax0.errorbar([xi - 0.16], [g["residual_median"] * 100],
                         yerr=[[g["U_band"] * 100], [g["U_band"] * 100]],
                         fmt="none", ecolor="#2ca02c", capsize=5, lw=1.6,
                         zorder=4,
                         label="U_band" if xi == x[0] else None)

    ax0.axhline(0.0, color="k", lw=0.8)
    ax0.axhline(GATE * 100, color="k", ls="--", lw=1.0)
    ax0.axhline(-GATE * 100, color="k", ls="--", lw=1.0,
                label=f"gate ±{GATE:.0%}")
    ax0.set_xticks(x)
    ax0.set_xticklabels(tags)
    ax0.set_ylabel("same-operator residual  (%)")
    ax0.set_title("realization residuals vs Wagner (same operator)")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8, loc="upper right")

    u = [g["uncertainty"] * 100 if g["uncertainty"] == g["uncertainty"] else float("nan")
         for g in grids]
    colors = ["#2ca02c" if g["judgeable"] else "#d62728" for g in grids]
    ax1.bar([str(g["dx"] * 1e3) + " mm" for g in grids], u, color=colors, width=0.55)
    for i, (val, g) in enumerate(zip(u, grids)):
        ax1.text(i, val + 0.8, f"{val:.1f} %\n{'judgeable' if g['judgeable'] else 'not judgeable'}",
                 ha="center", va="bottom", fontsize=8)
    ax1.axhline(GATE * 100, color="k", ls="--", lw=1.2,
                label=f"gate {GATE:.0%}")
    ax1.set_xlabel("dx")
    ax1.set_ylabel("U_real  (%)")
    ax1.set_ylim(0, max([v for v in u if v == v] + [GATE * 100]) * 1.45)
    ax1.set_title(f"estimator uncertainty (rule verdict: {ens['verdict']['verdict']})")
    ax1.grid(alpha=0.25, axis="y")
    ax1.legend(fontsize=8)

    fig.suptitle("CASE-SPH-WEDGE-ENTRY-2D — independent realizations as the "
                 "uncertainty axis (m22929)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())