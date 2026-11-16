"""Publication figures for the realization-closure round (m22929).

Two PNGs, all numbers read from the committed artefacts:

1. ``wedge_realisation_closure_m22929.png``  - residual scatter + U_real vs gate
   (produced by ``make_realisation_closure_figure.py``, not here);
2. ``wedge_realisation_trajectories_m22929.png`` - per-frame ``cp_centre(t)`` of
   every realization per grid (visualises A1: same shape, phase-shifted), plus
   the frame-level peak distribution that the estimator takes a median of.

    python benchmarks/case_sph_wedge_entry_2d/make_realisation_trajectory_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ENS = (ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_realisations"
       / "seed_ensemble.json")
OUT = ROOT / "docs" / "figures" / "wedge_realisation_trajectories_m22929.png"


def main() -> int:
    ens = json.loads(ENS.read_text())
    grids = sorted(ens["grids"], key=lambda g: -g["dx"])
    src = {}
    for d in (ROOT / "docs" / "artifacts").glob(
            "case_sph_wedge_entry_2d_realisations_*mm"):
        for f in sorted(d.glob("real_dx*.json")):
            src[f.name] = json.loads(f.read_text())

    fig, axes = plt.subplots(2, len(grids), figsize=(4.3 * len(grids), 6.6),
                             squeeze=False)
    for col, g in enumerate(grids):
        ax = axes[0][col]
        peaks = []
        for name, art in src.items():
            if abs(art["config"]["dx"] - g["dx"]) > 1e-12:
                continue
            ser = art["series"]
            t = ser["t"]
            cp = ser["cp_centre"]
            lab = f"seed {art['realisation']['seed']} (scale {art['realisation']['scale']:g})"
            ax.plot(t, cp, lw=1.0, alpha=0.85,
                    ls="--" if art["realisation"]["scale"] != 1.0 else "-",
                    label=lab)
            if art["realisation"]["scale"] == 1.0:
                peaks.append((max(v for v in cp if v == v),
                              art["by_frames"]["61"]["resolution_band"]["peak_median"]))
        ax.axhline(g["truth_same_operator"], color="k", ls=":", lw=1.0,
                   label="same-operator Wagner")
        ax.set_title(f"{g['tag']}: cp_centre(t) per realization", fontsize=9)
        ax.set_xlabel("t  (self-similar window)")
        ax.set_ylabel("Cp at apex")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2)

        ax2 = axes[1][col]
        if peaks:
            xs = [p[0] for p in peaks]
            ys = [p[1] for p in peaks]
            ax2.scatter(xs, ys, s=40, color="#1f77b4", label="per realization")
            ax2.plot([min(xs), max(xs)], [min(xs), max(xs)], color="grey",
                     lw=0.8, ls="--", label="y = x")
        ax2.axhline(g["obs_median"], color="#d62728", lw=1.4,
                    label="ensemble median (graded)")
        ax2.axhline(g["truth_same_operator"], color="k", ls=":", lw=1.0,
                    label="same-operator Wagner")
        ax2.set_title(f"{g['tag']}: frame max vs graded median [q]", fontsize=9)
        ax2.set_xlabel("max cp_centre over frames [q]")
        ax2.set_ylabel("resolution-band peak median [q]")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=6)

    fig.suptitle("CASE-SPH-WEDGE-ENTRY-2D — realization trajectories and the "
                 "graded statistic (m22929)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())