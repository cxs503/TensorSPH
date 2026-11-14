"""Figures for the laterally periodic lake-at-rest variant.

Reads the two committed artefacts

* ``artifacts/lake_at_rest/rest_dx020_wall.json``     (honest FAIL, side walls)
* ``artifacts/lake_at_rest/rest_dx020_periodic.json`` (anchor tolerance met)

and re-derives the two field maps it needs by re-running the case module -- the
per-particle acceleration is not stored in the artefact, and the maps are the
whole point (the wall band is where the residual lives).

    PYTHONPATH=src python3 benchmarks/lake_at_rest/make_figures.py

Writes ``artifacts/lake_at_rest/fieldmaps_dx020.npz`` and three PNGs under
``docs/``.  Every number in the figures comes from the two artfefacts or from
the recomputed fields; nothing is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    reconstruct, residual_report)

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "lake_at_rest"
DOCS = ROOT / "docs"


def field_map(periodic: bool) -> dict:
    cfg = LakeConfig(periodic_x=periodic)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)["mass"]
        solution = reconstruct(state, cfg, prior=prior)
        report = residual_report(state, cfg, solution["mass"])
        return {
            "x": state["pos"][:, 0].numpy(),
            "y": state["pos"][:, 1].numpy(),
            "ax_over_g": (report["acceleration"][:, 0] / cfg.gravity).numpy(),
            "ay_over_g": (report["acceleration"][:, 1] / cfg.gravity).numpy(),
            "mass_over_unit": (solution["mass"] / state["m_unit"]).numpy(),
            "rho_over_rho0": (solution["density"] / cfg.rho0).numpy(),
            "dx": cfg.dx,
            "depth": cfg.depth,
            "width": cfg.width,
        }
    finally:
        torch.set_default_dtype(previous)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ART.mkdir(parents=True, exist_ok=True)
    walls = field_map(periodic=False)
    periodic = field_map(periodic=True)
    np.savez_compressed(ART / "fieldmaps_dx020.npz",
                        wall={k: v for k, v in walls.items()},
                        periodic={k: v for k, v in periodic.items()})

    wall_json = json.loads((ART / "rest_dx020_wall.json").read_text())
    per_json = json.loads((ART / "rest_dx020_periodic.json").read_text())

    # ---- figure 1: where the horizontal residual lives ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    panels = ((walls, "side walls: max|a$_x$|/g = "
                      f"{float(np.abs(walls['ax_over_g']).max()):.2e}"),
              (periodic, "x-periodic: max|a$_x$|/g = "
                         f"{float(np.abs(periodic['ax_over_g']).max()):.2e}"))
    for ax, (field, title) in zip(axes, panels):
        scale = np.abs(field["ax_over_g"]).max()
        sc = ax.scatter(field["x"], field["y"], c=field["ax_over_g"],
                        cmap="RdBu_r", vmin=-max(scale, 1e-16),
                        vmax=max(scale, 1e-16), s=42, marker="s",
                        edgecolors="k", linewidths=0.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046,
                     label="a$_x$/g of the reconstructed state")
    axes[0].set_ylabel("y [m]")
    fig.suptitle("CASE-SPH-LAKE-AT-REST, dx = 0.02 m: horizontal residual of the "
                 "discrete hydrostatic equilibrium", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "lake_at_rest_periodic_fieldmap_ax.png", dpi=160)
    plt.close(fig)

    # ---- figure 2: density profile + Newton history ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    # lattice spacing available in walls["dx"] if a panel ever needs it
    depth = walls["depth"]
    rho0, c0, g, gamma = 1000.0, 20.0, 9.81, 7.0
    y_axis = np.linspace(0.0, depth, 200)
    bulk = rho0 * c0**2 / gamma
    p_eos = bulk * ((1.0 + rho0 * g * (depth - y_axis) / bulk) ** (1.0 / gamma) - 1.0)
    rho_eos = rho0 * (1.0 + p_eos / bulk) ** (1.0 / gamma)
    rho_incomp = rho0 * (1.0 + g * (depth - y_axis) / c0**2)
    ax.plot(rho_eos, y_axis, "k-", lw=2, label="Tait equilibrium (exact EOS)")
    ax.plot(rho_incomp, y_axis, "k--", lw=1, label="incompressible $p=\\rho_0 g(H-y)$")
    for field, style, label in ((walls, "o", "side walls"),
                                (periodic, "s", "x-periodic")):
        rows = {}
        for xi, yi, rho in zip(field["x"], field["y"], field["rho_over_rho0"]):
            rows.setdefault(round(float(yi), 9), []).append(float(rho))
        ys = np.array(sorted(rows))
        rs = np.array([np.mean(rows[y]) for y in ys])
        ax.plot(rs, ys, style, ms=4, fillstyle="none", label=f"{label} (reconstruction)")
    ax.set_xlabel("$\\rho/\\rho_0$")
    ax.set_ylabel("y [m]")
    ax.set_title("reconstructed density against the analytic profiles", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(wall_json["equilibrium"]["residual_history"], "o-",
                label="side walls (mirror-pair unknowns)")
    ax.semilogy(per_json["equilibrium"]["residual_history"], "s-",
                label="x-periodic (row unknowns)")
    ax.axhline(1e-9, color="k", ls=":", lw=1, label="anchor tolerance 1e-9")
    ax.set_xlabel("Gauss-Newton iteration")
    ax.set_ylabel("max|a$_y$|/g")
    ax.set_title("reconstruction of the discrete equilibrium", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(DOCS / "lake_at_rest_periodic_profiles.png", dpi=160)
    plt.close(fig)

    # ---- figure 3: does the lake stay at rest? ----------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for payload, style, label in ((wall_json, "o-", "side walls"),
                                  (per_json, "s-", "x-periodic")):
        samples = payload["rest"]["samples"]
        t = [s["t"] for s in samples]
        v = [max(s["v_over_c0"], 1e-18) for s in samples]
        ax.semilogy(t, v, style, ms=4, label=f"{label} (N={payload['geometry']['particles']})")
    ax.axhline(1e-9, color="k", ls=":", lw=1, label="anchor tolerance 1e-9")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("max|v|/c$_0$")
    ax.set_title("CASE-SPH-LAKE-AT-REST, dx = 0.02 m: does the lake stay still?",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(DOCS / "lake_at_rest_periodic_rest.png", dpi=160)
    plt.close(fig)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
