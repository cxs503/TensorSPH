"""Figures for the Morris low-Re Poiseuille case, from runs/poiseuille_morris/results.json.

    PYTHONPATH=src:. python benchmarks/poiseuille_morris/make_figs.py \
        [--results runs/poiseuille_morris/results.json] [--out runs/poiseuille_morris]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _select(cases, route, re_c, h_over_dx, columns=12):
    return [c for c in cases
            if c["reynolds_c"] == re_c and c["h_over_dx"] == h_over_dx
            and c["columns"] == columns]


def fig_profiles(payload, out: Path):
    isph = payload["cases"]["isph_projection"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, re_c in zip(axes, (1.0, 12.26)):
        sel = sorted(_select(isph, "isph", re_c, 1.90), key=lambda c: c["resolution_H_over_dx"])
        if not sel:
            continue
        rows0 = sel[0]["metrics"]["rows"]
        y = np.array([r["y_over_H"] for r in rows0])
        ax.plot(np.array([r["u_over_umax_reference"] for r in rows0]), y,
                "k-", lw=2.2, label="exact (Navier-Stokes steady)")
        for c in sel:
            rows = c["metrics"]["rows"]
            ax.plot([r["u_over_umax_measured"] for r in rows],
                    [r["y_over_H"] for r in rows], "o-", ms=4, lw=1.2,
                    label=f"H/dx={c['resolution_H_over_dx']}"
                          f" (e_L2={c['metrics']['interior']['l2_over_umax_pct']:.2f}%)")
        ax.set_title(f"ISPH projection, $Re_c$={re_c:g}")
        ax.set_xlabel("$u/u_{max}$,  $u_{max}=gH^2/(8\\nu)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("y/H")
    fig.suptitle("Plane Poiseuille steady profile vs exact solution (Morris case geometry)")
    fig.tight_layout()
    p = out / "fig1_profiles.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_convergence(payload, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    styles = {"isph_projection": ("o-", "ISPH projection"),
              "wcsp_production": ("s--", "WCSPH production (droplet_impact.rhs)")}
    for route, (style, label) in styles.items():
        cases = payload["cases"][route]
        hfd = 1.90 if route == "isph_projection" else 1.3
        for re_c, colour in ((1.0, "C0"), (12.26, "C3")):
            sel = sorted(_select(cases, route, re_c, hfd), key=lambda c: c["resolution_H_over_dx"])
            if not sel:
                continue
            dx = np.array([c["dx"] for c in sel])
            e = np.array([c["metrics"]["interior"]["l2_over_umax_pct"] for c in sel])
            ax.loglog(dx, e, style, color=colour, ms=5,
                      label=f"{label}, $Re_c$={re_c:g}")
    ax.axhline(5.0, color="k", ls=":", lw=1.2)
    ax.text(ax.get_xlim()[0] * 1.05, 5.3, "5% case tolerance", fontsize=8)
    ax.set_xlabel("dx = H/(H/dx)  [m]")
    ax.set_ylabel("interior $L_2$ error / $u_{max}$  [%]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Grid convergence of the steady profile")
    fig.tight_layout()
    p = out / "fig2_convergence.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_reynolds(payload, out: Path):
    isph = payload["cases"]["isph_projection"]
    res = 16
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for c in sorted(_select(isph, "isph", None, 1.90) if False else
                    [x for x in isph
                     if x["resolution_H_over_dx"] == res and x["h_over_dx"] == 1.90
                     and x["columns"] == 12], key=lambda x: x["reynolds_c"]):
        rows = c["metrics"]["rows"]
        axes[0].plot([r["u_over_umax_measured"] for r in rows],
                     [r["y_over_H"] for r in rows], "o-", ms=4,
                     label=f"$Re_c$={c['reynolds_c']:g}")
        axes[1].plot([100.0 * r["err_over_umax"] for r in rows],
                     [r["y_over_H"] for r in rows], "o-", ms=4,
                     label=f"$Re_c$={c['reynolds_c']:g}")
    rows0 = [x for x in isph if x["resolution_H_over_dx"] == res
             and x["h_over_dx"] == 1.90 and x["columns"] == 12][0]["metrics"]["rows"]
    axes[0].plot([r["u_over_umax_reference"] for r in rows0],
                 [r["y_over_H"] for r in rows0], "k-", lw=2.0, label="exact")
    axes[0].set_xlabel("$u/u_{max}$")
    axes[1].set_xlabel("$(u-u_{ref})/u_{max}$  [%]")
    axes[1].axvline(0.0, color="k", lw=0.8)
    axes[0].set_ylabel("y/H")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title(f"Profiles collapse: $Re_c$ ladder at H/dx={res}")
    axes[1].set_title("Row-wise residual (same runs)")
    fig.tight_layout()
    p = out / "fig3_reynolds.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_field(payload, out: Path):
    isph = payload["cases"]["isph_projection"]
    cand = [c for c in isph if c["h_over_dx"] == 1.90 and c["columns"] == 12
            and c["reynolds_c"] == 1.0]
    if not cand:
        cand = [c for c in isph if c["h_over_dx"] == 1.90 and c["columns"] == 12]
    finest = max(cand, key=lambda c: c["resolution_H_over_dx"])
    rows = finest["metrics"]["rows"]
    y = np.array([r["y_over_H"] for r in rows])
    u = np.array([r["u_over_umax_measured"] for r in rows])
    u_ref = np.array([r["u_over_umax_reference"] for r in rows])
    # the flow is x-invariant by construction: tile the row-averaged field over
    # one period so the contour is the measured field, not a reconstruction
    x = np.linspace(0.0, finest["length_over_height"], 24)
    field = np.tile(u.reshape(-1, 1), (1, x.size))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    im = axes[0].pcolormesh(x, y, field, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="$u/u_{max}$")
    axes[0].set_xlabel("x/H (one period)")
    axes[0].set_ylabel("y/H")
    axes[0].set_title(f"$u_x/u_{{max}}$ field, ISPH H/dx={finest['resolution_H_over_dx']}, "
                      f"$Re_c$={finest['reynolds_c']:g}")
    axes[0].plot([], [], " ", label="")  # keep layout stable
    axes[1].plot(100.0 * (u - u_ref), y, "o-", ms=4)
    axes[1].axvline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("$(u-u_{ref})/u_{max}$ [%]")
    axes[1].set_ylabel("y/H")
    axes[1].set_title("Residual profile")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig4_field.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_sensitivity(payload, out: Path):
    isph = payload["cases"]["isph_projection"]
    base = [c for c in isph if c["reynolds_c"] == 1.0 and c["resolution_H_over_dx"] == 16
            and c["columns"] == 12 and c["h_over_dx"] == 1.90]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    hs = sorted([c for c in isph if c["reynolds_c"] == 1.0
                 and c["resolution_H_over_dx"] == 16 and c["columns"] == 12],
                key=lambda c: c["h_over_dx"])
    if hs:
        axes[0].plot([c["h_over_dx"] for c in hs],
                     [c["metrics"]["interior"]["l2_over_umax_pct"] for c in hs], "o-")
        axes[0].set_xlabel("h/dx (support radius)")
        axes[0].set_ylabel("interior $L_2$ error [%]")
        axes[0].set_title("Support-radius sensitivity (H/dx=16, $Re_c$=1)")
        axes[0].grid(alpha=0.3)
    cols = sorted([c for c in isph if c["reynolds_c"] == 1.0
                   and c["resolution_H_over_dx"] == 16 and c["h_over_dx"] == 1.90],
                  key=lambda c: c["columns"])
    if cols:
        axes[1].plot([c["length_over_height"] for c in cols],
                     [c["metrics"]["interior"]["l2_over_umax_pct"] for c in cols], "s-")
        axes[1].set_xlabel("L/H (periodic domain length)")
        axes[1].set_ylabel("interior $L_2$ error [%]")
        axes[1].set_title("Periodic-length sensitivity (H/dx=16, $Re_c$=1)")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig5_sensitivity.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("runs/poiseuille_morris/results.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/poiseuille_morris"))
    args = parser.parse_args(argv)
    payload = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_profiles, fig_convergence, fig_reynolds, fig_field, fig_sensitivity):
        made.append(fn(payload, args.out))
    manifest = {"figures": [{"name": p.name, "bytes": p.stat().st_size} for p in made]}
    (args.out / "figures.json").write_text(json.dumps(manifest, indent=2))
    for p in made:
        print("[figs]", p, p.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())