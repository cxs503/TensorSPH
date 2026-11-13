"""Figures for the same-operator peak-Cp calibration (m18382).

Reads only committed artefacts:
``docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json`` (produced by
``estimator_bias.py``) and the committed per-frame scans under
``docs/artifacts/case_sph_wedge_entry_2d_ensemble/``.  No number is typed in.

Panels
------
(a) peak Cp vs ensemble size: measured (SPH, pooled) and the analytic Wagner
    field pushed through the *same* measurement operator (wall-node sampling +
    Adami kernel extrapolation + binned peak), with the point reference and the
    15 % band;
(b) same-operator residual ``(obs - operator_truth)/operator_truth`` vs dx with
    the same 15 % band: the quantity that actually competes with the acceptance
    gate, instead of a point-value comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias"
FIG = ROOT / "docs" / "figures"


def main() -> int:
    rep = json.loads((ART / "bias_report.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    frames = rep["design"]["frame_counts"]
    ref = rep["wagner_cp_max_point"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    styles = {"dx2mm": ("#1f77b4", "o", "dx = 2 mm"), "dx4mm": ("#d62728", "s", "dx = 4 mm")}
    for grid, (col, mk, label) in styles.items():
        bf = rep["grids"][grid]["by_frames"]
        obs = [bf[str(n)]["obs_peak_median"] for n in frames if str(n) in bf]
        sm = [bf[str(n)]["synth_smoothed"]["peak_median"] for n in frames if str(n) in bf]
        xs = [n for n in frames if str(n) in bf]
        ax.plot(xs, obs, mk + "-", color=col, label=f"SPH, pooled ({label})")
        ax.plot(xs, sm, mk + "--", color=col, alpha=0.65,
                label=f"Wagner through the same operator ({label})")
    ax.axhline(ref, color="k", lw=1.2, label=f"Wagner point $C^2+1$ = {ref:.3f}")
    ax.fill_between([min(frames), max(frames)], ref * 0.85, ref * 1.15,
                    color="grey", alpha=0.18, label="±15 % acceptance band (point value)")
    ax.set_xlabel("ensemble size (last N frames of the window)")
    ax.set_ylabel(r"peak $C_p = p/(\rho V^2/2)$")
    ax.set_title("(a) like-for-like: same estimator on both sides")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    dxs = [rep["grids"][g]["dx"] * 1e3 for g in ("dx2mm", "dx4mm")]
    res = [rep["verdict"][g]["residual"] * 100.0 for g in ("dx2mm", "dx4mm")]
    ax.axhspan(-15, 15, color="green", alpha=0.12, label="±15 % acceptance")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(dxs, res, "o-", color="#2ca02c", label="same-operator residual (61 frames)")
    for x, y in zip(dxs, res):
        ax.annotate(f"{y:+.1f} %", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(dxs)
    ax.set_xticklabels([f"{d:g} mm" for d in dxs])
    ax.set_ylabel("residual vs operator-processed Wagner  [%]")
    ax.set_title("(b) the residual that competes with the gate")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    fig.suptitle("CASE-SPH-WEDGE-ENTRY-2D  β=30°, V=1 m/s — measurement-operator bias "
                 "of the peak slamming pressure", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "wedge_operator_bias.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    osize = out.stat().st_size
    print(f"png_bytes={osize}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())