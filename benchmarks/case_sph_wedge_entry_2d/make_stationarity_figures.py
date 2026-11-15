"""Figures for the frame-ensemble stationarity audit (reads the committed JSONs only).

Produces ``docs/figures/case_sph_wedge_entry_2d_halfsplit.png`` with three panels:

1. half-split resolved by bin count -- the decisive panel: if the split is flat in
   the bin count it is not a binning artefact;
2. early vs late pooled readings at every admissible bin count (the direction of
   the drift is visible here);
3. the resolution-band reading with its bootstrap percentile band, i.e. the
   sampling uncertainty that does not depend on splitting the window.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
HS_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_halfsplit"


def load(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    grids = []
    for dx, tag in ((0.004, "dx4mm"), (0.002, "dx2mm"), (0.001, "dx1mm"), (0.0005, "dx0.5mm")):
        p = HS_DIR / f"halfsplit_dx{dx}.json"
        d = load(p)
        if d:
            grids.append((dx, tag, d))
    if not grids:
        print("no half-split artefacts", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    for dx, tag, d in grids:
        bins = sorted(int(k) for k in d["common_bin_splits"])
        splits = [100.0 * d["common_bin_splits"][str(b)]["rel_split"] for b in bins]
        ax.plot(bins, splits, "o-", label=f"{tag}")
        ax.axhline(100.0 * d["legacy_half_split"], color=ax.lines[-1].get_color(),
                   ls=":", lw=1)
    ax.axhline(15.0, color="crimson", ls="--", lw=1, label="15 % gate")
    ax.set_xlabel("pooled bin count (operator)")
    ax.set_ylabel("early/late half split (%)")
    ax.set_title("split resolved by bin count\n(dotted = legacy 10/12/16 median)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for dx, tag, d in grids:
        bins = sorted(int(k) for k in d["common_bin_splits"])
        e = [d["common_bin_splits"][str(b)]["early"] for b in bins]
        l = [d["common_bin_splits"][str(b)]["late"] for b in bins]
        ax.plot(bins, e, "o-", label=f"{tag} early")
        ax.plot(bins, l, "s--", label=f"{tag} late")
    ax.set_xlabel("pooled bin count")
    ax.set_ylabel("pooled, sample-gated peak $C_p$")
    ax.set_title("early vs late half reading\n(same operator, both halves)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[2]
    xs, med, lo, hi = [], [], [], []
    for dx, tag, d in grids:
        b = d["bootstrap"]
        xs.append(dx * 1000)
        med.append(b["peak_median"])
        lo.append(b["peak_median"] - b["p05"])
        hi.append(b["p95"] - b["peak_median"])
    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o-", capsize=4, label="bootstrap p05-p95")
    for dx, tag, d in grids:
        ax.plot(dx * 1000, d["resolution_band_peak_median"], "x", color="black")
        ax.annotate(tag, (dx * 1000, d["resolution_band_peak_median"]),
                    textcoords="offset points", xytext=(5, -12), fontsize=7)
    ax.axhline(6.92, color="green", ls="--", lw=1,
               label="Zhao & Faltinsen 1993 (COM, q units)")
    ax.set_xscale("log")
    ax.set_xlabel("$dx$ (mm)")
    ax.set_ylabel("resolution-band peak $C_p$")
    ax.set_title("reading vs grid, frame-bootstrap band\n(x = point reading)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("CASE-SPH-WEDGE-ENTRY-2D — pooled-peak estimator: half-split vs operator, "
                 "drift direction, sampling band", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "docs" / "figures" / "case_sph_wedge_entry_2d_halfsplit.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())