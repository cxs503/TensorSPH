"""Figures for the Young-Laplace verification of the CSF surface-tension model.

Reads the JSON produced by ``run.py`` and writes PNGs next to it.  Every number
in the figures comes from that file, so the figures cannot drift from the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figures(report: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report["resolution_study"]
    dx = np.array([r["dx_m"] for r in rows])
    ratio = np.array([r["dp_implied_Pa"] / r["dp_theory_Pa"] for r in rows])
    kappa = np.array([r["kappa_absmedian_band"] for r in rows])
    kappa_over_theory = kappa * report["setup"]["radius_m"]
    ctl = report["positive_control"]
    ctl_dx = np.array([c["dx_m"] for c in ctl])
    ctl_ratio = np.array([c["dp_recovered_Pa"] / c["dp_theory_Pa"] for c in ctl])
    paths = []

    # ---------------- Figure 1: does the model deliver sigma/R? ------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.loglog(dx, ratio, "o-", color="crimson", lw=2, ms=7,
              label="CSF heuristic colour (shipped)")
    ax.loglog(ctl_dx, ctl_ratio, "s--", color="seagreen", lw=1.6, ms=6,
              label="positive control (analytic delta)")
    ax.axhline(1.0, color="k", lw=1.2, ls=":", label=r"target $\Delta p=\sigma/R$")
    ax.set_xlabel(r"$\Delta x$ [m]")
    ax.set_ylabel(r"$\Delta p_{\rm implied} \,/\, (\sigma/R)$  $=\sigma_{\rm eff}/\sigma$")
    ax.set_title("(a) Effective surface tension under refinement")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.loglog(dx, kappa_over_theory, "o-", color="navy", lw=2, ms=7,
              label=r"$\mathrm{median}\,|\kappa|\,R$  (should be 1)")
    ax.axhline(1.0, color="k", lw=1.2, ls=":", label="target $\\kappa = 1/R$")
    ref = kappa_over_theory[0] * (dx[0] / dx) ** 1.0
    ax.loglog(dx, ref, "--", color="gray", lw=1.2, label=r"slope $\propto 1/\Delta x$")
    ax.set_xlabel(r"$\Delta x$ [m]")
    ax.set_ylabel(r"median $|\kappa| \, R$  [-]")
    ax.set_title("(b) Interface curvature: diverges instead of converging")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "fig1_sigma_eff_and_curvature.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # ---------------- Figure 2: shape of the radial force field ------------ #
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(rows)))
    for row, c in zip(rows, colours):
        prof = np.array(row["profile"])
        if prof.size == 0:
            continue
        axes[0].plot(prof[:, 0], prof[:, 1], "-", color=c, lw=1.4,
                     label=f"$\\Delta x$={row['dx_m']*1e3:.2f} mm")
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_yscale("symlog", linthresh=10.0)
    axes[0].set_ylabel(r"$\rho\, f_r$  [Pa/m]")
    axes[0].set_xlabel(r"$r/R$")
    axes[0].set_title("(a) Radial force density: sign changes at the rim")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for row, c in zip(rows, colours):
        prof = np.array(row["diagnostic_color_variant"]["profile"])
        if prof.size == 0:
            continue
        axes[1].plot(prof[:, 0], prof[:, 1], "-", color=c, lw=1.4,
                     label=f"$\\Delta x$={row['dx_m']*1e3:.2f} mm")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_yscale("symlog", linthresh=10.0)
    axes[1].set_ylabel(r"$\rho\, f_r$  [Pa/m]")
    axes[1].set_xlabel(r"$r/R$")
    axes[1].set_title("(b) Diagnostic colour-function variant (not shipped)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p2 = out_dir / "fig2_radial_force_profile.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # ---------------- Figure 3: inward-direction fidelity ------------------ #
    inward = np.array([r["inward_frac_dot_gt_0p5"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(dx, inward * 100, "o-", color="darkorange", lw=2, ms=7)
    ax.axhline(100.0, color="k", ls=":", lw=1.2, label="target 100 %")
    ax.set_xlabel(r"$\Delta x$ [m]")
    ax.set_ylabel("interface particles with inward force [%]")
    ax.set_ylim(0, 105)
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p3 = out_dir / "fig3_inward_fraction.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)
    return paths


if __name__ == "__main__":
    root = Path("artifacts/surface_tension_laplace")
    report = json.loads((root / "laplace_report.json").read_text())
    for p in make_figures(report, root):
        print(f"wrote {p}")