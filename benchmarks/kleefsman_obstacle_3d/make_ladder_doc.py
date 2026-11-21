"""Build the 3-D ladder note + figures from whatever 3-D results exist.

The figures are *derived*, never hand-drawn:

1. ``fig_h2_overlay.png``   -- H2 water-height history: every recovered 3-D run
   against the digitised Kleefsman Fig. 19 reference.  Skipped (and the reason
   printed) if no run has been recovered.
2. ``fig_peak_ladder.png``  -- peak height vs dx with the 10 % band around the
   experimental workbook peak, plus the fitted continuum extrapolate when the ladder
   has >= 3 levels.
3. ``fig_tool_verification.png`` -- the ladder *estimator* validated against
   synthetic data of known order: recovered p vs true p.  This panel is what
   makes the tool falsifiable, so it is emitted even when no production run has
   landed.

Usage::

    python benchmarks/kleefsman_obstacle_3d/make_ladder_doc.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorsph.marin_probes import H2_ARRIVAL_S, H2_PEAK_M  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import ladder  # noqa: E402

RESULTS = HERE / "results"
FIGDIR = HERE / "results" / "ladder_figs"
REF_CSV = ROOT / "benchmarks" / "marin_box_loads" / "h2_figure19_red.csv"
REF_PEAK_M = H2_PEAK_M
REF_ARRIVAL_S = H2_ARRIVAL_S
TOL = 0.10


def _tags() -> list[str]:
    return sorted(p.name[: -len("_time_series.csv")]
                  for p in RESULTS.glob("*_time_series.csv"))


def fig_overlay(tags: list[str]) -> Path | None:
    if not tags:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    drew = False
    for tag in tags:
        t, slab, _, _, _ = ladder.load_run(RESULTS, tag)["_series"]
        if not t:
            continue
        ax.plot(t, slab, lw=1.4, label=f"3-D SPH {tag}")
        drew = True
    if REF_CSV.exists():
        tt, hh = [], []
        with REF_CSV.open(newline="") as fh:
            for row in csv.reader(fh):
                try:
                    tt.append(float(row[0]))
                    hh.append(float(row[1]))
                except (ValueError, IndexError):
                    continue
        if tt:
            ax.plot(tt, hh, "k--", lw=1.2, label="Kleefsman 2005 Fig.19 (digitised)")
            drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.axhline(REF_PEAK_M, color="0.6", ls=":", lw=1.0,
               label=f"experimental peak {REF_PEAK_M:.3f} m")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height in front of box [m]")
    ax.set_title("Kleefsman obstacle case, 3-D arm (40 % blockage)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_h2_overlay.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_peak_ladder(report: dict) -> Path | None:
    runs = [r for r in report.get("runs", [])
            if r.get("peak_m") is not None and r.get("dx") == r.get("dx")]
    if not runs:
        return None
    runs = sorted(runs, key=lambda r: r["dx"])
    dx = [r["dx"] for r in runs]
    pk = [r["peak_m"] for r in runs]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(REF_PEAK_M * (1 - TOL), REF_PEAK_M * (1 + TOL), color="0.85",
               label="experiment $\\pm$10 %")
    ax.axhline(REF_PEAK_M, color="k", lw=1.0)
    ax.plot(dx, pk, "o-", lw=1.5, label="3-D SPH peak")
    fit = report.get("peak_ladder", {})
    if fit.get("status") == "OK":
        ax.plot([0.0], [fit["f_inf"]], "r*", ms=13,
                label=f"extrapolate {fit['f_inf']:.3f} m, p={fit['p_observed']:.2f}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("peak height [m]")
    ax.set_title("3-D peak vs grid; does the lift close the 2-D +30 % gap?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_peak_ladder.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_tool_verification() -> Path:
    """Recovered order of accuracy vs the true one, on synthetic ladders."""
    truths = [1.0, 1.5, 2.0, 3.0, 4.0]
    dxs = [0.08, 0.04, 0.03]
    got = []
    for p in truths:
        vals = [0.30 + 2.0e-2 * d ** p for d in dxs]
        got.append(ladder.fit_power_law(dxs, vals)["p_observed"])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(truths, truths, "k--", lw=1.0, label="exact")
    ax.plot(truths, got, "o", ms=7, label="recovered p")
    ax.set_xlabel("imposed order of accuracy p")
    ax.set_ylabel("recovered p")
    ax.set_title("ladder estimator, non-uniform ladder dx=0.08/0.04/0.03")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_tool_verification.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    tags = _tags()
    report = ladder.analyse(RESULTS, tags)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ladder.json").write_text(json.dumps(report, indent=1) + "\n")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = {
        "overlay": str(fig_overlay(tags)) if tags else None,
        "peak_ladder": str(fig_peak_ladder(report)),
        "tool_verification": str(fig_tool_verification()),
    }
    print(json.dumps({"tags": tags, "status": report.get("status"),
                      "figures": made}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())