"""Figures for the wedge-entry metric-resolution study (CASE-SPH-WEDGE-ENTRY-2D).

Every number plotted is read from the JSON artefacts written by
``metric_resolution.py`` (no hand-copied values) except the analytic Wagner
profile, which is evaluated from ``tensorsph.wedge_entry.wagner_p_over_q`` -- the
in-repo reference, not a typed table.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/make_metric_figures.py \
        --artifacts artifacts/case_sph_wedge_entry_2d_metric \
        --out docs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorsph.wedge_entry import (  # noqa: E402
    wagner_c_coeff,
    wagner_p_over_q,
    wagner_peak_over_q,
)

BETA_DEG = 30.0
TOL = 0.15


def load(art_dir: Path) -> list[dict]:
    out = []
    for path in sorted(art_dir.glob("metric_*.json")):
        out.append(json.loads(path.read_text()))
    if not out:
        raise SystemExit(f"no metric artefacts under {art_dir}")
    return out


def tag(a: dict) -> str:
    c = a["case"]
    return (f"dx={c['dx'] * 1e3:g}mm delta={c['delta']:g} "
            f"{c['form']} ens={c.get('ensemble_frames', '?')}")


def pick(arts, **want):
    for a in arts:
        c = a["case"]
        if all(abs(c[k] - v) < 1e-12 if isinstance(v, float) else c[k] == v
               for k, v in want.items()):
            return a
    return None


def fig_bin_sensitivity(arts, out: Path) -> Path:
    ref = wagner_peak_over_q(np.radians(BETA_DEG))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for a in [x for x in arts if x["case"]["delta"] == 0.0]:
        counts = a["estimators"]["counts"]
        nx = sorted(int(k) for k in counts)
        ny = [counts[str(k)]["cp_band_peak"] for k in nx]
        ax.plot(nx, ny, "o-", label=f"{tag(a)}  (n_wet={a['derived']['wetted_nodes_last']})")
        ax.plot([a["estimators"]["adaptive"]["n_bins"]],
                [a["estimators"]["adaptive"]["cp_band_peak"]], "D",
                color=ax.lines[-1].get_color())
    ax.axhline(ref, color="k", lw=1.2, ls="--", label=f"Wagner $C^2+1$ = {ref:.3f}")
    ax.fill_between([8, 130], ref * (1 - TOL), ref * (1 + TOL), color="k", alpha=0.08,
                    label="±15 % tolerance")
    ax.set_xscale("log")
    ax.set_xlabel("number of wall-profile bins over $\\tau\\in[0,1.2]$")
    ax.set_ylabel("$C_p$ band peak (last-frame ensemble mean)")
    ax.set_title("Same trajectory, different estimator: the peak is a binning artefact\n"
                 "(diamonds = resolution-consistent rule, bin width $=2\\Delta x$)")
    ax.set_ylim(0, 34)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "wedge_metric_fig1_bin_sensitivity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_profile(arts, out: Path) -> Path:
    ref_C = wagner_c_coeff(np.radians(BETA_DEG))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for a in [x for x in arts if x["case"]["delta"] == 0.0]:
        e = a["estimators"]["adaptive"]
        taus = np.array(e["profile_tau"], dtype=float)
        cps = np.array(e["profile_cp"], dtype=float)
        keep = np.isfinite(cps)
        ax.plot(taus[keep], cps[keep], "o-", ms=4,
                label=f"SPH {tag(a)}")
    # wagner_p_over_q takes tau = sqrt(1 - 1/s^2); parametrise the curve in s.
    s = np.linspace(1.0, 6.0, 400)
    tau_s = np.sqrt(1.0 - 1.0 / s ** 2)
    cps_s = [wagner_p_over_q(t, np.radians(BETA_DEG)) for t in tau_s]
    ax.plot(tau_s, cps_s, "k--", lw=1.4,
            label="Wagner (in-repo closed form)")
    ax.axvline(np.sqrt(1 - 1 / ref_C ** 2), color="grey", ls=":", lw=1)
    ax.text(np.sqrt(1 - 1 / ref_C ** 2) + 0.005, 1.0,
            r"$\tau_{peak}=\sqrt{1-1/C^2}$", rotation=90, fontsize=7, color="grey")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-3, 30)
    ax.set_xlabel(r"self-similar abscissa $\tau = |x| / (C V t)$")
    ax.set_ylabel(r"wall pressure coefficient $p / (\frac{1}{2}\rho V^2)$")
    ax.set_title("Wall $C_p$ profile: the SPH maximum sits at Wagner's own peak abscissa,\n"
                 "the Wagner curve turns negative beyond it (jet region unresolved)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "wedge_metric_fig2_profile_vs_wagner.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_delta(arts, out: Path) -> Path:
    sel = [a for a in arts if abs(a["case"]["velocity"] - 1.0) < 1e-12]
    if not sel:
        raise SystemExit("no velocity=1 artefacts for the delta sweep")
    sel.sort(key=lambda a: (a["case"]["form"], a["case"]["delta"]))
    labels = [f"{a['case']['form']}\n$\\delta$={a['case']['delta']:g}\ndx={a['case']['dx'] * 1e3:g}mm"
              for a in sel]
    rho = [a["shot_metrics"]["max_rho_ratio"] - 1.0 for a in sel]
    peak = [a["estimators"]["adaptive"]["cp_band_peak"] for a in sel]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].bar(range(len(sel)), [1e2 * r for r in rho], color="tab:red")
    axes[0].set_ylabel(r"max $(\rho/\rho_0-1)$ over the run  [%]")
    axes[0].set_title("Density excursion: no monotone gain from $\\delta$-SPH here")
    axes[1].bar(range(len(sel)), peak, color="tab:blue")
    axes[1].axhline(wagner_peak_over_q(np.radians(BETA_DEG)), color="k", ls="--",
                    lw=1.2, label="Wagner $C^2+1$")
    axes[1].set_ylabel("$C_p$ band peak (bin $=2\\Delta x$)")
    axes[1].set_title("… and no systematic gain in the peak either")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Diffusive density term as a cure for wall-pressure noise (falsified)", fontsize=9)
    fig.tight_layout()
    path = out / "wedge_metric_fig3_density_diffusion.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_robust_metrics(arts, out: Path) -> Path:
    ref_C = wagner_c_coeff(np.radians(BETA_DEG))
    sel = [a for a in arts if a["case"]["delta"] == 0.0 and a["case"]["velocity"] == 1.0]
    sel.sort(key=lambda a: -a["case"]["dx"])
    dxs = [a["case"]["dx"] * 1e3 for a in sel]
    metrics = {
        "wetted width $c/(CVt)$": (
            [a["shot_metrics"]["c_over_Vt_last"] for a in sel], ref_C),
        "stagnation $C_p(0)$": (
            [a["shot_metrics"]["cp_centre_mean"] for a in sel],
            wagner_p_over_q(0.0, np.radians(BETA_DEG))),
        "band peak $C_p$ (bin $=2\\Delta x$)": (
            [a["estimators"]["adaptive"]["cp_band_peak"] for a in sel],
            wagner_peak_over_q(np.radians(BETA_DEG))),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for k, (vals, ref) in metrics.items():
        err = [100.0 * (v - ref) / ref for v in vals]
        ax.plot(dxs, err, "o-", label=k)
    ax.axhspan(-100 * TOL, 100 * TOL, color="k", alpha=0.08, label="±15 %")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$\\Delta x$ [mm] (finer to the right)")
    ax.set_ylabel("relative error vs Wagner [%]")
    ax.set_title("Which metrics the measurement can actually support at $\\beta=30°$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "wedge_metric_fig4_robust_metrics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"
                                              / "case_sph_wedge_entry_2d_metric"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args(argv)
    art_dir, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arts = load(art_dir)
    for fn in (fig_bin_sensitivity, fig_profile, fig_robust_metrics):
        print(fn(arts, out))
    try:
        print(fig_delta(arts, out))
    except SystemExit as exc:
        print(f"delta figure skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())