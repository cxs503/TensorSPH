"""Figures for the wedge-entry compressibility-closure study.

Reads only ``summary.json`` written by ``compressibility.py --stage accept``;
nothing is typed in by hand, so the plots cannot drift from the numbers.

    python benchmarks/case_sph_wedge_entry_2d/make_compressibility_figures.py

Outputs (into ``docs/figures/``):
  case_sph_wedge_entry_2d_compressibility_trend.png   Cp vs c0/V against the anchor
  case_sph_wedge_entry_2d_compressibility_profile.png wall-pressure profiles vs Wagner
  case_sph_wedge_entry_2d_compressibility_floor.png   density noise floor vs measured Cp
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.wedge_entry import wagner_p_over_q  # noqa: E402

OUT = ROOT / "artifacts" / "case_sph_wedge_entry_2d" / "compressibility"
FIGS = ROOT / "docs" / "figures"
BETA = 30.0


def load() -> dict:
    path = OUT / "summary.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run compressibility.py --stage accept first")
    return json.loads(path.read_text())


def fig_trend(summary: dict) -> Path:
    rows = sorted([r for r in summary["rows"] if r["role"] == "sound_ladder"],
                  key=lambda r: r["c0_over_V"])
    c0 = [r["c0_over_V"] for r in rows]
    ref = summary["reference"]["wagner"]["cp_max"]
    tol = summary["verdict"]["cp_tol"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhspan(ref * (1 - tol), ref * (1 + tol), color="0.85",
               label=f"anchor $\\pm${tol:.0%} (Cp$_{{max}}$={ref:.3f})")
    ax.axhline(ref, color="k", lw=1, ls="--")
    ax.plot(c0, [r["cp_band_peak"] for r in rows], "o-", lw=1.8,
            label="Cp band peak (ensemble mean, $\\tau\\leq1$)")
    ax.plot(c0, [r["cp_at_peak_tau"] for r in rows], "s--", lw=1.2,
            label="Cp at $\\tau_{peak}$=0.93")
    ax.plot(c0, [r["cp_centre_mean"] for r in rows], "^:", lw=1.2,
            label="Cp at apex")
    ax.plot(c0, [r["cp_density_floor"] for r in rows], "v-.", color="0.45", lw=1.2,
            label="density noise floor $2(c_0/V)^2/\\gamma[\\rho_\\gamma-1]$")
    ax.set_xscale("log")
    ax.set_xlabel("$c_0/V$  (inverse Mach number)")
    ax.set_ylabel("$C_p = p/(\\frac{1}{2}\\rho V^2)$")
    ax.set_title("CASE-SPH-WEDGE-ENTRY-2D: $\\beta$=30°, V=1 m/s, dx=4 mm\n"
                 "compressibility closure — " + summary["verdict"]["verdict"].replace("_", " "))
    ax.set_xticks(c0)
    ax.set_xticklabels([f"{c:g}" for c in c0])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / "case_sph_wedge_entry_2d_compressibility_trend.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_profile(summary: dict) -> Path:
    rows = sorted([r for r in summary["rows"] if r["role"] == "sound_ladder"],
                  key=lambda r: r["c0_over_V"])
    pick = [rows[0], rows[-1]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tau = np.linspace(0.0, 1.15, 400)
    wagner = np.array([wagner_p_over_q(float(t), math.radians(BETA)) for t in tau])
    ax.plot(tau, wagner, "k--", lw=1.6, label="Wagner (1932), $\\beta$=30°")
    styles = ["o-", "s-"]
    for r, st in zip(pick, styles):
        taus = np.array([t for t in r["profile_tau"]], dtype=float)
        cps = np.array([c if c is not None else np.nan for c in r["profile_cp"]],
                       dtype=float)
        keep = taus <= 1.15
        ax.plot(taus[keep], cps[keep], st, ms=3, lw=1.2,
                label=f"SPH $c_0/V$={r['c0_over_V']:g} (band peak {r['cp_band_peak']:.2f})")
    ax.set_xlabel("$\\tau=|x|/(CVt)$  (self-similar abscissa)")
    ax.set_ylabel("$C_p$")
    ax.set_ylim(0, max(12.0, min(40.0, 1.2 * max(
        [c for r in pick for c in r["profile_cp"] if c is not None] or [12.0]))))
    ax.set_title("Wall-pressure profile, ensemble mean of last 5 frames")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / "case_sph_wedge_entry_2d_compressibility_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_floor(summary: dict) -> Path:
    rows = sorted([r for r in summary["rows"] if r["role"] == "sound_ladder"],
                  key=lambda r: r["c0_over_V"])
    c0 = [r["c0_over_V"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax1.plot(c0, [100 * (r["max_rho_ratio"] - 1) for r in rows], "o-")
    ax1.set_xscale("log")
    ax1.set_xticks(c0)
    ax1.set_xticklabels([f"{c:g}" for c in c0])
    ax1.set_xlabel("$c_0/V$")
    ax1.set_ylabel("max $(\\rho/\\rho_0-1)$  [%]")
    ax1.grid(alpha=0.3)
    ax1.set_title("density excursion")
    ax2.plot(c0, [r["cp_density_floor"] for r in rows], "s-", label="noise floor")
    ax2.plot(c0, [r["cp_band_peak"] for r in rows], "o-", label="measured Cp band peak")
    ref = summary["reference"]["wagner"]["cp_max"]
    ax2.axhline(ref, color="k", ls="--", lw=1, label="Wagner $C_{p,max}$")
    ax2.set_xscale("log")
    ax2.set_xticks(c0)
    ax2.set_xticklabels([f"{c:g}" for c in c0])
    ax2.set_xlabel("$c_0/V$")
    ax2.set_ylabel("$C_p$")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("floor vs signal")
    fig.suptitle("Is the excess noise? floor and measured peak move together")
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / "case_sph_wedge_entry_2d_compressibility_floor.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    summary = load()
    for p in (fig_trend(summary), fig_profile(summary), fig_floor(summary)):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())