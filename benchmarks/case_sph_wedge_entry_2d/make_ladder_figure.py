"""Figures for the closed wedge-entry grid ladder (m20161).

Reads only committed/generated artefacts -- the ensemble JSONs and the ladder
closure summary -- so no number is hand-copied into a plot.

Two panels:
  1. measured peak ``Cp`` per grid (with its estimator band) against the
     same-operator Wagner truth and the Wagner point value;
  2. same-operator residual vs ``dx`` on log-log axes with the 15 % gate and the
     observed-order / Richardson annotation, if the order is trustworthy.

Usage::

    python benchmarks/case_sph_wedge_entry_2d/make_ladder_figure.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ART = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ensemble"
LADDER = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json"
FIGS = ROOT / "docs/figures"


def band(dx_tag: str):
    """Resolution-band min/max of the 61-frame ensemble at one grid."""
    path = ART / f"ensemble_{dx_tag}_c0v20.json"
    if not path.exists():
        return None
    rb = json.loads(path.read_text())["by_frames"]["61"]["resolution_band"]
    return rb["peak_min"], rb["peak_max"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", type=Path, default=LADDER)
    ap.add_argument("--out", type=Path, default=FIGS / "wedge_ladder_closure.png")
    args = ap.parse_args(argv)

    rep = json.loads(args.ladder.read_text())
    rows = rep["grids"]
    dx = [r["dx"] * 1e3 for r in rows]
    obs = [r["obs_peak"] for r in rows]
    truth = [r["synth_peak"] for r in rows]
    resid = [r["residual"] * 100.0 for r in rows]
    wag_point = rep["wagner_cp_max_point"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes[0]
    for r in rows:
        b = band(r["tag"])
        if b and math.isfinite(b[0]) and math.isfinite(b[1]):
            ax.errorbar(r["dx"] * 1e3, r["obs_peak"],
                        yerr=[[r["obs_peak"] - b[0]], [b[1] - r["obs_peak"]]],
                        fmt="o", ms=6, capsize=4, color="C3",
                        label="SPH peak Cp (61-frame ensemble, band = estimator spread)"
                        if r is rows[0] else None)
        else:
            ax.plot(r["dx"] * 1e3, r["obs_peak"], "o", ms=6, color="C3")
    ax.plot(dx, truth, "s--", color="C0", ms=5,
            label="Wagner field through the same measurement operator")
    ax.axhline(wag_point, color="C0", lw=1.2, ls=":",
               label=f"Wagner point value $C^2+1$ = {wag_point:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("$dx$ [mm]")
    ax.set_ylabel(r"peak $C_p = p_{max}/(0.5\,\rho V^2)$")
    ax.set_title(r"$\beta=30^\circ$ wedge entry: peak $C_p$ vs grid"
                 "\n(Zhao & Faltinsen 1993 $\\beta$=30 value: WITHHELD)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    ax.axhspan(-15, 15, color="0.85", label="15 % acceptance gate")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(dx, resid, "o-", color="C3")
    for r in rows:
        ax.annotate(f"{r['tag']}\n{r['residual']*100:+.1f} %",
                    (r["dx"] * 1e3, r["residual"] * 100.0),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    gc = rep.get("grid_convergence", {})
    if gc.get("available") and gc.get("trustworthy"):
        note = (f"observed order p = {gc['order_mean']:.2f}\n"
                f"Richardson residual = {gc['residual_asymptotic']*100:+.1f} %\n"
                f"extrapolated peak Cp = {gc['obs_peak_asymptotic']:.2f}")
    elif gc.get("available"):
        note = (f"observed order p = {gc['order_mean']:.2f} -- NOT trustworthy\n"
                "(no Richardson extrapolation)")
    else:
        note = f"order unavailable: {gc.get('reason', 'n/a')}"
    ax.text(0.03, 0.05, note, transform=ax.transAxes, fontsize=8,
            bbox=dict(fc="white", ec="0.5", alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("$dx$ [mm]")
    ax.set_ylabel("same-operator residual [%]")
    ax.set_title(f"same-operator residual: {rep['verdict']['verdict']}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[figure] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())