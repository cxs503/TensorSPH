"""Measure the frame dependence the committed bootstrap assumes away (m22416).

Reads the *committed* artefacts (no solver), extracts the per-frame footprint
series that ``frame_scan`` already stores, and reports the lag-1..5
autocorrelation, the AR(1) effective sample size and the predicted inflation of
the bootstrap half-width.  Writes ``docs/artifacts/case_sph_wedge_entry_2d_frame_audit/
autocorr_committed.json`` and a figure.

Caveat, stated up front: ``frame_scan.cp_footprint_max`` is the footprint peak of
a *single* frame, not the frame's contribution to the pooled window mean.  It is
the only per-frame series the committed artefacts kept, and it is used here only
to measure the *dependence structure* of the run -- never as a reading.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import uncertainty_audit as ua  # noqa: E402

ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint"
OUT = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_frame_audit"
TAGS = ("dx4mm", "dx2mm")


def autocorr(series, lag):
    vals = [v for v in series if v == v]
    n = len(vals)
    if n <= lag + 1:
        return float("nan")
    mean = sum(vals) / n
    den = sum((v - mean) ** 2 for v in vals)
    num = sum((vals[i] - mean) * (vals[i + lag] - mean) for i in range(n - lag))
    return num / den if den else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"case": "CASE-SPH-WEDGE-ENTRY-2D",
           "script": "benchmarks/case_sph_wedge_entry_2d/autocorr_committed.py",
           "quantity": "frame_scan.cp_footprint_max (single-frame footprint peak)",
           "why": ("the committed i.i.d. frame bootstrap assumes these frames are "
                   "exchangeable; this measures whether they are"),
           "grids": {}}
    for tag in TAGS:
        art = json.loads((ART / f"ensemble_{tag}_c0v20.json").read_text())
        series = [f["cp_footprint_max"] for f in art["frame_scan"]]
        rho = {str(k): autocorr(series, k) for k in range(1, 6)}
        r1 = rho["1"]
        cells = 8.0e-3 / art["config"]["dx"]
        boot = art["footprint_peak"]["bootstrap"].get(str(cells), {})
        rep["grids"][tag] = {
            "dx_m": art["config"]["dx"], "n_frames": len(series),
            "footprint_cells": cells,
            "rho_lag1_to_5": rho,
            "n_effective": ua.n_effective(len(series), r1),
            "analytic_inflation": (math.sqrt((1.0 - r1) / (1.0 + r1))
                                   if -1.0 < r1 < 1.0 else float("nan")),
            "committed_U_iid": boot.get("rel_half_width"),
            "committed_U_would_be_if_ar1": (
                boot.get("rel_half_width") * math.sqrt((1.0 + r1) / (1.0 - r1))
                if boot.get("rel_half_width") and -1.0 < r1 < 1.0 else float("nan")),
            "series": series,
        }
    (OUT / "autocorr_committed.json").write_text(json.dumps(rep, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for tag, g in rep["grids"].items():
        axes[0].plot(range(len(g["series"])), g["series"], marker=".", ms=3,
                     label=f"{tag} (rho1={g['rho_lag1_to_5']['1']:+.2f})")
    axes[0].set_xlabel("frame index (t = k t_end / 60)")
    axes[0].set_ylabel(r"single-frame $C_{p,\Phi}$")
    axes[0].set_title("Per-frame footprint peak: the signal is a transient cluster")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)

    lags = [1, 2, 3, 4, 5]
    for tag, g in rep["grids"].items():
        axes[1].plot(lags, [g["rho_lag1_to_5"][str(k)] for k in lags],
                     marker="o", label=tag)
    axes[1].axhline(0.0, color="k", lw=.8)
    axes[1].set_xlabel("lag [frames]"); axes[1].set_ylabel("autocorrelation")
    axes[1].set_title("Frames are not exchangeable: ACF of the per-frame series")
    axes[1].grid(alpha=.3); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_autocorr_committed.png", dpi=140)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "series"}
                      for k, v in rep["grids"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())