"""Figures for CASE-SPH-WEDGE-ENTRY-2D.

Every plotted number is read back from an artefact JSON in the artefact
directory (``summary.json``, ``refs_derived.json``, ``<case>.json``); nothing is
recomputed here, so a figure can always be traced to a file and a line.

    python benchmarks/case_sph_wedge_entry_2d/make_figures.py [--art-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

ART = ROOT / "artifacts" / "case_sph_wedge_entry_2d"


def _cjk() -> bool:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                 "Source Han Sans SC", "SimHei"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK = _cjk()


def L(zh: str, en: str) -> str:
    """Bilingual label: Chinese when a CJK font exists, English otherwise."""
    return f"{zh}\n{en}" if CJK else en


def wagner_profile(taus, ref: dict):
    """Wagner eq. (2) evaluated from the reference constants in refs_derived.json."""
    C = ref["wagner"]["C"]
    out = []
    for tau in taus:
        t = min(max(tau, 0.0), 0.999)
        s = 1.0 / math.sqrt(max(1.0 - t * t, 1e-12))
        out.append(2.0 * C * s - s * s + 1.0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", default=str(ART))
    args = ap.parse_args(argv)
    art = Path(args.art_dir)
    summary = json.loads((art / "summary.json").read_text())
    ref = json.loads((art / "refs_derived.json").read_text())
    cases = {}
    for p in sorted(art.glob("*.json")):
        d = json.loads(p.read_text())
        if "case_id" in d and "series" in d:
            cases[d["case_id"]] = d
    ladder = sorted([c for c in summary["rows"] if c["kind"] == "ladder"],
                    key=lambda r: -r["dx"])
    figs = []

    # ---- fig 1: wall-pressure profile in the self-similar abscissa --------- #
    sys.path.insert(0, str(HERE))
    from run import ensemble_profile  # noqa: E402
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for row in ladder:
        cid = row["case_id"]
        taus, cps, n = ensemble_profile(cases[cid])
        xs = [t for t, v in zip(taus, cps) if v == v]
        ys = [v for v in cps if v == v]
        ax.plot(xs, ys, marker=".", ms=3, lw=1.0,
                label=f"SPH dx = {row['dx'] * 1e3:g} mm (peak {row['cp_band_peak']:.2f})")
    if ladder:
        taus, _, _ = ensemble_profile(cases[ladder[-1]["case_id"]])
        ax.plot(taus, wagner_profile(taus, ref), "k--", lw=1.6,
                label=f"Wagner C^2+1 = {ref['wagner']['cp_max']:.3f}")
        ax.axvline(summary["tau_peak"], color="0.6", ls=":", lw=1.0)
    ax.set_xlabel(r"$\tau = |x|/(C V t)$")
    ax.set_ylabel(r"$C_p = p/(0.5\rho V^2)$")
    ax.set_title(L("楔形体壁面压力剖面 vs Wagner 解析解 (β=30°, V=1 m/s)",
                   "Wall pressure profile vs Wagner (beta=30 deg, V=1 m/s)"))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    f1 = art / "fig1_profile_vs_wagner.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    figs.append(f1)

    # ---- fig 2: time series of the three surface readings ------------------ #
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fine = ladder[-1] if ladder else None
    if fine:
        c = cases[fine["case_id"]]
        t = [r["t"] * 1e3 for r in c["series"]]
        ax.plot(t, [r["cp_max_footprint"] for r in c["series"]], ".-",
                label="max over footprint")
        ax.plot(t, [r["cp_max_interior"] for r in c["series"]], ".-",
                label="max over |x| <= 0.9 c")
        ax.plot(t, [r["cp_centre"] for r in c["series"]], ".-",
                label="apex (stagnation line)")
        ax.axhline(ref["wagner"]["cp_max"], color="k", ls="--", lw=1.2,
                   label=f"Wagner peak {ref['wagner']['cp_max']:.2f}")
        ax.axhline(ref["wagner"]["cp_centre"], color="0.4", ls=":", lw=1.2,
                   label=f"Wagner apex C = {ref['wagner']['cp_centre']:.2f}")
        ax.set_xlabel("t [ms]")
        ax.set_ylabel(r"$C_p$")
        ax.set_title(L(f"表面压力系数时程 (dx = {fine['dx'] * 1e3:g} mm)",
                       f"Surface pressure coefficient vs time (dx = {fine['dx'] * 1e3:g} mm)"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    f2 = art / "fig2_cp_time_series.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    figs.append(f2)

    # ---- fig 3: grid convergence of the graded metrics --------------------- #
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
    if ladder:
        dxs = [r["dx"] * 1e3 for r in ladder]
        axs[0].plot(dxs, [r["cp_band_peak"] for r in ladder], "o-",
                    label="SPH band peak")
        axs[0].axhline(ref["wagner"]["cp_max"], color="k", ls="--",
                       label=f"Wagner {ref['wagner']['cp_max']:.2f}")
        axs[0].set_xscale("log")
        axs[0].set_xlabel("dx [mm]")
        axs[0].set_ylabel(r"$C_p$")
        axs[0].set_title(L("峰值压力系数网格收敛", "Peak Cp vs dx"))
        axs[0].set_ylim(0, 14)
        axs[0].grid(alpha=0.3, which="both")
        axs[0].legend(fontsize=7)
        axs[1].plot(dxs, [100 * r["err_band_peak"] for r in ladder], "o-",
                    label="band peak vs Wagner")
        axs[1].plot(dxs, [100 * r["err_centre_mean"] for r in ladder], "s-",
                    label="apex vs Wagner")
        axs[1].axhline(15.0, color="r", ls="--", label="15 % tolerance")
        axs[1].set_xscale("log")
        axs[1].set_xlabel("dx [mm]")
        axs[1].set_ylabel("relative error [%]")
        axs[1].set_title(L("相对误差", "Relative error"))
        axs[1].grid(alpha=0.3, which="both")
        axs[1].legend(fontsize=7)
    fig.tight_layout()
    f3 = art / "fig3_grid_convergence.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    figs.append(f3)

    # ---- fig 4: wetted width against Wagner's C --------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in ladder:
        c = cases[row["case_id"]]
        ax.plot([r["t"] * 1e3 for r in c["series"]],
                [r["c_sph"] / (c["config"]["velocity"] * max(r["t"], 1e-12))
                 for r in c["series"]], ".-", label=f"dx = {row['dx'] * 1e3:g} mm")
    ax.axhline(ref["wagner"]["C"], color="k", ls="--",
               label=f"Wagner C = {ref['wagner']['C']:.3f}")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel(r"$c/(Vt)$")
    ax.set_title(L("湿半宽 vs Wagner", "Wetted half-width vs Wagner"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    f4 = art / "fig4_wetting.png"
    fig.savefig(f4, dpi=150)
    plt.close(fig)
    figs.append(f4)

    print(json.dumps([str(p) for p in figs], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())