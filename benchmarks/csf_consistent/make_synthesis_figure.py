"""Generate a verification status figure for the CSF surface-tension synthesis report.

Uses the existing benchmark harness (benchmarks/csf_consistent/run.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.csf_consistent.run import instant_diagnostics

def main():
    dxs = [0.003, 0.002, 0.0015]

    data = {"mls": [], "fit_quadratic": [], "legacy": []}
    for dx in dxs:
        # consistent + mls curvature
        row = instant_diagnostics(dx=dx, scheme="consistent")
        data["mls"].append({
            "dx": dx,
            "cut_ratio": row["cut_force"]["ratio_measured_over_theory"],
            "kappa_rel_pct": row["kappa"]["rel_error_pct"],
            "ds_integral": row["delta_s"]["integral_dn"],
        })
        # consistent + fit_quadratic curvature
        row = instant_diagnostics(dx=dx, scheme="consistent")
        # Need to re-run with fit_quadratic - the harness uses cfg.csf_curvature
        from tensorsph.csf_consistent import csf_fields, surface_tension_acceleration
        from tensorsph.droplet_impact import Config
        from tensorsph.neighbor_search import cell_linked_pairs
        from benchmarks.laplace_law.run import hexagonal_droplet
        import torch

        RADIUS = 0.02
        RHO0 = 1000.0
        SIGMA = 7.2

        cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                     gamma=7.0, gravity=0.0, viscosity=0.01,
                     surface_tension_coefficient=SIGMA,
                     csf_scheme="consistent", csf_curvature="fit_quadratic")
        h = cfg.h_factor * dx
        pos, mass = hexagonal_droplet(RADIUS, dx, RHO0)
        pairs = cell_linked_pairs(pos, 2.0 * h)
        state = csf_fields(pos, mass, pairs, h, RHO0, curvature_scheme="fit_quadratic")
        acc, _ = surface_tension_acceleration(
            pos, torch.full_like(mass, RHO0), mass, pairs, h, SIGMA, RHO0)

        right = pos[:, 0] > 0
        f_cut = float((mass[right, None] * acc[right]).sum(dim=0)[0])
        w = state.delta_s
        kappa_w = float((state.kappa.abs() * w).sum() / w.sum())
        import math
        r = torch.linalg.vector_norm(pos, dim=1)
        m = (r > 0.6 * RADIUS) & (r < 1.05 * RADIUS)
        d_v = mass / RHO0
        ds_int = float((state.delta_s[m] * d_v[m] / (2.0 * math.pi * r[m].clamp_min(1e-30))).sum())

        data["fit_quadratic"].append({
            "dx": dx,
            "cut_ratio": f_cut / (-2.0 * SIGMA),
            "kappa_rel_pct": (kappa_w - 1.0 / RADIUS) * RADIUS * 100,
            "ds_integral": ds_int,
        })
        # legacy
        row = instant_diagnostics(dx=dx, scheme="legacy")
        data["legacy"].append({
            "dx": dx,
            "cut_ratio": row["cut_force"]["ratio_measured_over_theory"],
            "kappa_rel_pct": row["kappa"]["rel_error_pct"],
            "ds_integral": row["delta_s"]["integral_dn"],
        })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Cut force identity
    ax = axes[0]
    for key, label, style in [("legacy", "legacy CSF", "s--"),
                               ("mls", "consistent (mls)", "o-"),
                               ("fit_quadratic", "consistent (fit_quad)", "^-")]:
        xs = [d["dx"] for d in data[key]]
        ys = [d["cut_ratio"] for d in data[key]]
        ax.plot(xs, ys, style, label=label, markersize=6)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.axhspan(0.85, 1.15, color="g", alpha=0.1, label="15% band")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$F_{cut}/(2\sigma)$")
    ax.set_title("(a) Cut-force identity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Curvature error
    ax = axes[1]
    for key, label, style in [("legacy", "legacy CSF", "s--"),
                               ("mls", "consistent (mls)", "o-"),
                               ("fit_quadratic", "consistent (fit_quad)", "^-")]:
        xs = [d["dx"] for d in data[key]]
        ys = [abs(d["kappa_rel_pct"]) for d in data[key]]
        ax.plot(xs, ys, style, label=label, markersize=6)
    ax.axhline(15.0, color="r", ls="--", alpha=0.7, label="15% gate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("|curvature error| [%]")
    ax.set_title("(b) Curvature error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (c) Delta-s integral
    ax = axes[2]
    for key, label, style in [("legacy", "legacy CSF", "s--"),
                               ("mls", "consistent (mls)", "o-"),
                               ("fit_quadratic", "consistent (fit_quad)", "^-")]:
        xs = [d["dx"] for d in data[key]]
        ys = [d["ds_integral"] for d in data[key]]
        ax.plot(xs, ys, style, label=label, markersize=6)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.axhspan(0.85, 1.15, color="g", alpha=0.1, label="15% band")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\int \delta_s dn$")
    ax.set_title(r"(c) Delta-s normalization")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = HERE / "csf_verification_status.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
