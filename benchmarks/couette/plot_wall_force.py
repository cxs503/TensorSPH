#!/usr/bin/env python3
"""Figure for the force-based Couette wall-shear reading (see
``docs/xval_couette_wall_force_reading_m23304.md``).

Left panel: the two estimators on the same runs, both walls, vs the owner 15 %
standard.  Right panel: the near-wall velocity profile of the finest arm with the
analytic ``u = U y/H`` so the reader can see the lag the secant amplifies.

Run: PYTHONPATH=src python benchmarks/couette/plot_wall_force.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "benchmarks" / "couette" / "out_force" / "wall_force_reading.json"
OUT = REPO / "docs" / "figures" / "m23304_wall_force_reading.png"


def main() -> int:
    art = json.loads(ART.read_text())
    main_arms = sorted((a for a in art["arms"] if a["layout"] == "periodic3"),
                       key=lambda a: a["resolution_H_over_dx"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    labels, force_err, secant_err = [], [], []
    for a in main_arms:
        r = a["resolution_H_over_dx"]
        labels += [f"H/dx={r}\ntop", f"H/dx={r}\nbottom"]
        force_err += [a["err_top_force_pct"], a["err_bottom_force_pct"]]
        secant_err += [a["err_top_secant_pct"], a["err_bottom_secant_pct"]]
    x = range(len(labels))
    ax = axes[0]
    ax.bar([i - 0.2 for i in x], force_err, width=0.4, label="force (Newton)", color="#2b6cb0")
    ax.bar([i + 0.2 for i in x], secant_err, width=0.4, label="1-dx secant (banked)",
           color="#c05621")
    ax.axhline(-15.0, ls="--", lw=1.2, color="k", label="owner 15 % standard")
    ax.axhline(15.0, ls="--", lw=1.2, color="k")
    ax.axhline(0.0, lw=0.8, color="grey")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("|1 - tau_SPH/tau_exact|  [%]")
    ax.set_title("same run, two estimators (t_end = 3.6 H$^2$/nu)")
    ax.legend(fontsize=7)

    ax = axes[1]
    for a in main_arms:
        ys = [p["y_over_H"] for p in a["profile"]]
        us = [p["u_over_U"] for p in a["profile"]]
        ax.plot(us, ys, "o-", ms=3, lw=1.0,
                label=f"H/dx={a['resolution_H_over_dx']}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="analytic u = U y/H")
    ax.set_xlabel("u/U")
    ax.set_ylabel("y/H")
    ax.set_title("profile (time-averaged, last 20 %)")
    ax.legend(fontsize=7)
    fig.suptitle("Plane Couette: wall stress from the wall reaction force", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
