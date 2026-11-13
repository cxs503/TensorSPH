"""Figures for the frame-ensemble peak study (reads committed artefacts only).

Panel (a): the pooled peak estimate against the ensemble size, with the
estimator's spread over the bin-count grid as an error bar, both grids, and the
analytic Wagner value with its +-15 % acceptance band.
Panel (b): the per-frame single-frame peak against normalised time -- the
quantity the old 120-bin metric was maximising -- with the pooled ensemble value
drawn as a level.  Nothing is typed in: every number comes from the JSON
artefacts in ``docs/artifacts/case_sph_wedge_entry_2d_ensemble``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
FIGS = ROOT / "docs" / "figures"


def load():
    docs = {}
    for path in sorted(ART.glob("ensemble_*.json")):
        doc = json.loads(path.read_text())
        docs[f"dx={doc['config']['dx']*1e3:g} mm"] = doc
    return docs


def main() -> int:
    docs = load()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colours = ["#1f77b4", "#d62728", "#2ca02c"]

    ax = axes[0]
    for colour, (label, doc) in zip(colours, docs.items()):
        ref = doc["wagner_cp_max"]
        keys = sorted(doc["by_frames"], key=int)
        xs = [int(k) for k in keys]
        med = [doc["by_frames"][k]["peak_median"] for k in keys]
        lo = [doc["by_frames"][k]["peak_min"] for k in keys]
        hi = [doc["by_frames"][k]["peak_max"] for k in keys]
        ax.errorbar(xs, med,
                    yerr=[[m - l for m, l in zip(med, lo)],
                          [h - m for m, h in zip(med, hi)]],
                    marker="o", capsize=3, color=colour,
                    label=f"SPH {label} (pooled, gated)")
        ax.axhline(ref * 1.15, color="grey", ls=":", lw=1)
        ax.axhline(ref * 0.85, color="grey", ls=":", lw=1)
    ref = next(iter(docs.values()))["wagner_cp_max"]
    ax.axhline(ref, color="black", ls="--", lw=1,
               label=r"Wagner $C^2+1$ = %.4f" % ref)
    ax.axhspan(ref * 0.85, ref * 1.15, color="grey", alpha=0.08)
    ax.set_xlabel("frames in the self-similar ensemble")
    ax.set_ylabel(r"peak wall $C_p = p_{max}/(\frac{1}{2}\rho V^2)$")
    ax.set_title("(a) pooled peak vs ensemble size (bars = bin-count spread)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for colour, (label, doc) in zip(colours, docs.items()):
        scan = [s for s in doc["frame_scan"]
                if s["cp_peak_single_frame"] == s["cp_peak_single_frame"]]
        t = [s["t"] / doc["t_end"] for s in scan]
        cp = [s["cp_peak_single_frame"] for s in scan]
        ax.plot(t, cp, marker=".", lw=0.8, ms=4, color=colour,
                label=f"single-frame peak, {label}")
        pooled = doc["by_frames"][str(max(int(k) for k in doc["by_frames"]))]
        ax.axhline(pooled["peak_median"], color=colour, ls="--", lw=1,
                   label=f"pooled (61 frames) {label}: {pooled['peak_median']:.2f}")
    ax.axhline(ref, color="black", ls="-", lw=1,
               label=r"Wagner %.4f" % ref)
    ax.set_xlabel(r"$t / t_{end}$ (self-similar window)")
    ax.set_ylabel(r"$C_p$ peak, one frame")
    ax.set_title("(b) per-frame peak scan: the spike lives in the first frames")
    ax.legend(fontsize=6.5, loc="upper center")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGS / "wedge_ensemble_peak.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())