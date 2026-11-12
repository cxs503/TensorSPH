"""Figures for CASE-HOLLOW-SPHERE-3D (read ``report.json``; no hand-typed numbers).

Panels
------
1. ``dp_ratio.png``      ``Delta p / Delta p_theory`` against ``R_out/dx`` for the
                         drop and the hollow sphere, one curve per curvature
                         route, with the 5 % platform band drawn.
2. ``p_profile.png``     the radial pressure field of the finest hollow sphere
                         against the exact two-level staircase
                         (``2 sigma/R_out`` in the shell, ``2 sigma(1/R_in+1/R_out)``
                         in the cavity) -- checks the whole field, not two probes.
3. ``curvature.png``     measured ``kappa / (2/R)`` on the two faces against
                         ``R/dx``, i.e. the curvature operator on its own.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
SIGMA = 0.072


def main():
    rep = json.loads((HERE / "report.json").read_text())
    rows = rep["levels"]
    tol = rep["platform_tolerance"]
    FIG.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhspan(1 - tol, 1 + tol, color="0.85", label=f"platform gate {tol:.0%}")
    for geom, marker in (("drop", "o"), ("shell", "s")):
        for cur, ls in (("analytic", "-"), ("mls", "--")):
            sel = sorted([r for r in rows if r["geometry"] == geom and r["curvature"] == cur],
                         key=lambda r: r["R_out_over_dx"])
            if not sel:
                continue
            x = [r["R_out_over_dx"] for r in sel]
            y = [r["dp_over_theory"] for r in sel]
            ax.plot(x, y, ls, marker=marker, label=f"{geom}, {cur} curvature")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel(r"$R_{out}/\Delta x$")
    ax.set_ylabel(r"$\Delta p / [2\sigma(1/R_{in}+1/R_{out})]$")
    ax.set_title("Hollow sphere: measured Laplace jump vs resolution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "dp_ratio.png", dpi=150)
    plt.close(fig)

    # 2. pressure profile of the finest shell level
    shell = [r for r in rows if r["geometry"] == "shell" and r["curvature"] == "analytic"]
    finest = max(shell, key=lambda r: r["R_out_over_dx"])
    r_in, r_out = finest["radius_in_m"], finest["radius_out_m"]
    r = finest["p_profile_r"]
    p = finest["p_profile_p"]
    scale = 2.0 * SIGMA * (1.0 / r_in + 1.0 / r_out)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(r, [v / scale for v in p], "o", ms=3, label="SPH projection pressure")
    exact = []
    for ri in r:
        if ri < r_in:
            exact.append(1.0)
        elif ri < r_out:
            exact.append((2.0 * SIGMA / r_out) / scale)
        else:
            exact.append(0.0)
    ax.plot(r, exact, "k--", lw=1.2, label="exact: $2\\sigma(1/R_{in}+1/R_{out})$ / $2\\sigma/R_{out}$ / 0")
    ax.axvline(r_in, color="0.6", lw=0.8, ls=":")
    ax.axvline(r_out, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("r [m]")
    ax.set_ylabel(r"$p(r)\,/\,2\sigma(1/R_{in}+1/R_{out})$")
    ax.set_title(f"Radial pressure, shell $R_{{in}}$={r_in:.0f} $R_{{out}}$={r_out:.0f} dx")
    ax.set_xlim(0, r_out + 4.0)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p_profile.png", dpi=150)
    plt.close(fig)

    # 3. curvature on each face
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(-tol, tol, color="0.85")
    for label, key in (("inner face $2/R_{in}$", "kappa_inner_error_pct"),
                       ("outer face $2/R_{out}$", "kappa_outer_error_pct")):
        sel = [r for r in shell if r.get(key) is not None]
        sel = sorted(sel, key=lambda r: r["R_out_over_dx"])
        ax.plot([r["R_out_over_dx"] for r in sel], [r[key] for r in sel], "o-",
                label="analytic route: " + label)
    ax.set_xlabel(r"$R_{out}/\Delta x$")
    ax.set_ylabel("curvature error [%]")
    ax.set_title("Two curvature radii recovered independently")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "curvature.png", dpi=150)
    plt.close(fig)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()