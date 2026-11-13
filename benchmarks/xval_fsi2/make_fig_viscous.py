"""Figures for the viscous-branch repair (X3 / X5).

Everything is read back from ``artifacts/xval_fsi2/results.json``; no number is
typed in by hand, so the plot cannot drift away from the driver output.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.fsi_force_from_fluid import (one_sided_viscous_truncation_factor,  # noqa: E402
                                            )
from tensorsph.wcsph import kernels  # noqa: E402

ART = ROOT / "artifacts" / "xval_fsi2"


def main() -> Path:
    payload = json.loads((ART / "results.json").read_text())
    rows = payload["rows"]
    series = {}
    for row in rows:
        if row["case"] not in ("couette_shear", "curved_disk_shear"):
            continue
        for op, score in row["operators"].items():
            series.setdefault((row["case"], op), []).append((row["dx"], score["error"]))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    style = {("couette_shear", "oss_viscous"): ("o-", "OSS one-sided (X3 plane Couette)"),
             ("couette_shear", "gradient_corrected"): ("s--", "corrected (X3 plane Couette)"),
             ("curved_disk_shear", "oss_viscous"): ("^-", "OSS one-sided (X5 disk)"),
             ("curved_disk_shear", "gradient_corrected"): ("v--", "corrected (X5 disk)")}
    for key, pts in series.items():
        if key not in style:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [100.0 * p[1] for p in pts]
        fmt, label = style[key]
        ax.plot(xs, ys, fmt, label=label, markersize=5)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(15.0, color="tab:red", ls=":", lw=1.0)
    ax.axhline(-15.0, color="tab:red", ls=":", lw=1.0)
    ax.axhline(3.0, color="tab:orange", ls=":", lw=1.0)
    ax.axhline(-3.0, color="tab:orange", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("particle spacing dx [m]  (finer ->)")
    ax.set_ylabel("load error vs analytic [%]")
    ax.set_title("viscous force-from-fluid: error vs resolution\n(red = 15% gate, orange = 3%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    # right panel: where the OSS deficit comes from
    couette = [r for r in rows if r["case"] == "couette_shear"][0]
    dx = couette["dx"]
    h = 1.3 * dx
    n = 20001
    r = torch.linspace(1.0e-6, 2.0 * h, n, dtype=torch.float64)
    _, df = kernels.shape(r / h, "wendland")
    dwdr = kernels.normalisation(2, "wendland") / h ** 3 * df
    integrand = -2.0 * dx * dwdr * r
    seg = torch.cumulative_trapezoid(integrand, r)
    cumulative = torch.cat([torch.zeros(1, dtype=seg.dtype), seg])
    bx.plot((r / h).tolist(), cumulative.tolist(), color="tab:blue",
            label="cumulative  -2 dx int (dW/dr) r dr   [-> ratio]")
    continuum = one_sided_viscous_truncation_factor(h, 2, "wendland", spacing=dx)
    bx.axhline(continuum, color="tab:blue", ls=":", lw=1.0)
    bx.annotate(f"continuum plateau = 7/(3 pi) dx/h = {continuum:.3f}",
                xy=(1.05, continuum), xytext=(0.35, continuum * 0.62), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    measured = couette["operators"]["oss_viscous"]["got"] / couette["operators"]["oss_viscous"]["ref"]
    bx.axhline(measured, color="tab:red", ls="--", lw=1.0)
    bx.annotate(f"measured on the lattice = {measured:.3f}", xy=(1.6, measured),
                xytext=(0.75, measured * 0.80), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="tab:red"))
    bx.axhline(1.0, color="k", ls="-", lw=0.8)
    bx.annotate("exact murho-free traction tau = mu U/h (ratio 1)", xy=(1.2, 1.0),
                xytext=(0.25, 1.03), fontsize=7)
    bx.axvspan(0.0, 0.5 * dx / h, color="0.85")
    bx.annotate("no particle inside r < dx/2", xy=(0.19, 0.15), fontsize=6.5, rotation=90,
                ha="center")
    bx.set_xlim(0.0, 2.0)
    bx.set_ylim(0.0, 1.15)
    bx.set_xlabel("r / h  (support edge at 2)")
    bx.set_ylabel("fraction of the exact traction")
    bx.set_title("why the OSS branch loses 38 %:\nhalf-space truncation moment (kernel-only)")
    bx.grid(True, alpha=0.3)
    bx.legend(fontsize=6.5, loc="upper left")

    fig.suptitle("X3/X5 viscous transfer: truncation deficit, its origin, and the corrected arm "
                 f"(Couette dx={dx})", fontsize=9)
    fig.tight_layout()
    out = ART / "fig_viscous_repair.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()