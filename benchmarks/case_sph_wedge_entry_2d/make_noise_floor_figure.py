"""Figure for the frame-level same-operator study (reads the committed JSON only).

Four panels, one file: ``docs/figures/case_sph_wedge_entry_2d_noise_floor_m22798.png``

1. the operator response ``A12(t) / Wagner point peak``: what the estimator would
   read on the exact solution, i.e. the discretisation part of the measurement;
2. the per-frame residual ``obs/A12 - 1`` against the frame time, with its median:
   the quantity whose dispersion is the noise floor;
3. the grid medians on log-log axes with the fitted exponent and the two
   predicted slopes (0.5 = extreme-value noise of a max over bin means,
   1.0 = second-order scheme convergence);
4. the declared uncertainty per grid against the 15 % case gate.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_noise_floor" / "noise_floor.json"
OUT = ROOT / "docs" / "figures" / "case_sph_wedge_entry_2d_noise_floor_m22798.png"

COLOURS = {"dx4mm": "#b04a3a", "dx2mm": "#c98a2b", "dx1mm": "#2f7d4f", "dx0.5mm": "#1f4e79"}


def main() -> int:
    if not ART.exists():
        print(f"missing artefact: {ART}", file=sys.stderr)
        return 1
    data = json.loads(ART.read_text())
    grids = data["grids"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    ax = axes[0][0]
    for g in grids:
        col = COLOURS.get(g["tag"], "k")
        ts = [f["t"] * 1e3 for f in g["frames"] if f["a12"] == f["a12"]]
        rel = [f["a12"] / g["wagner_point_peak"] for f in g["frames"] if f["a12"] == f["a12"]]
        ax.plot(ts, rel, "-", color=col, lw=1.4, label=f"{g['tag']}")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="Wagner point peak")
    ax.set_xlabel("frame time  t  [ms]")
    ax.set_ylabel(r"$A_{12}(t)\;/\;(C^2+1)$")
    ax.set_title("(a) operator response on the exact field\n(the discretisation part of the reading)")
    ax.set_ylim(0.0, 1.08)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for g in grids:
        col = COLOURS.get(g["tag"], "k")
        pts = [(f["t"] * 1e3, 100.0 * f["residual"]) for f in g["frames"]
               if f["n_wet"] >= 20 and f["residual"] == f["residual"]]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=col,
                   alpha=0.55, label=f"{g['tag']} frames")
        ax.axhline(100.0 * g["reading"]["median"], color=col, lw=1.8)
    ax.axhspan(-15, 15, color="0.85", zorder=0, label="±15 % gate")
    ax.set_xlabel("frame time  t  [ms]")
    ax.set_ylabel(r"frame residual  $\mathrm{obs}/A_{12}-1$  [%]")
    ax.set_title("(b) per-frame residual (dots) and its median (line)")
    ax.set_ylim(-60, 300)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    dxs = [g["dx"] for g in grids]
    meds = [g["reading"]["median"] for g in grids]
    ax.loglog(dxs, [100.0 * m for m in meds], "o", ms=8, color="#1f4e79",
              label="grid median residual")
    for g, m in zip(grids, meds):
        ax.annotate(g["tag"], (g["dx"], 100.0 * m), textcoords="offset points",
                    xytext=(6, -12), fontsize=8)
    xs = [min(dxs), max(dxs)]
    p = data["exponent"]["slope"]
    anchor_dx, anchor_m = dxs[-1], meds[-1]
    ax.loglog(xs, [100.0 * anchor_m * (x / anchor_dx) ** p for x in xs], "-",
              color="#1f4e79", lw=1.6, label=f"fit  p = {p:+.2f}")
    ax.loglog(xs, [100.0 * anchor_m * (x / anchor_dx) ** 0.5 for x in xs], "--",
              color="#b04a3a", lw=1.2, label="noise (extreme-value)  p = 0.5")
    ax.loglog(xs, [100.0 * anchor_m * (x / anchor_dx) ** 1.0 for x in xs], ":",
              color="#2f7d4f", lw=1.2, label="scheme convergence  p = 1")
    ax.axhline(15.0, color="0.3", lw=1.0, ls="-.")
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing  dx  [m]")
    ax.set_ylabel("median residual  [%]")
    ax.set_title("(c) residual scaling with the grid")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][1]
    uncs = [100.0 * g["reading"]["uncertainty"] for g in grids]
    boots = [100.0 * g["reading"]["bootstrap"]["half_width"] for g in grids]
    splits = [100.0 * g["reading"]["half_split_term"] for g in grids]
    idx = range(len(grids))
    ax.bar([i - 0.22 for i in idx], boots, width=0.22, color="#1f4e79",
           label="moving-block bootstrap")
    ax.bar([i for i in idx], splits, width=0.22, color="#c98a2b", label="early/late half-split")
    ax.bar([i + 0.22 for i in idx], uncs, width=0.22, color="#2f7d4f",
           label="declared uncertainty (max)")
    ax.axhline(15.0, color="k", ls="--", lw=1.2, label="15 % case gate")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([g["tag"] for g in grids])
    ax.set_ylabel("uncertainty of the median  [%]")
    ax.set_title("(d) declared uncertainty vs the gate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("CASE-SPH-WEDGE-ENTRY-2D — frame-level peak residual against a same-operator "
                 "Wagner reference", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())