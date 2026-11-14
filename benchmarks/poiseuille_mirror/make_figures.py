#!/usr/bin/env python3
"""Generate publication-quality figures for the Poiseuille SPH verification case.

Reads the committed results.json and produces:
  1. velocity_profile.png  -- SPH vs analytic velocity profiles at all resolutions
  2. convergence.png       -- error metrics vs H/dx (log-log)
  3. field_snapshot.png    -- velocity field snapshot (H/dx=16)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmarks" / "poiseuille_mirror" / "out" / "results.json"
FIGDIR = REPO / "benchmarks" / "poiseuille_mirror" / "figures"

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
RHO0 = 1000.0

U_MAX_REF = BODY_FORCE * HEIGHT**2 / (8.0 * VISCOSITY)  # 1.22625


def analytic_u(y: float) -> float:
    return BODY_FORCE * (HEIGHT - y) * y / (2.0 * VISCOSITY)


def load_data() -> dict:
    return json.loads(RESULTS.read_text())


# --------------------------------------------------------------------------- #
# Figure 1: Velocity profiles vs analytic
# --------------------------------------------------------------------------- #

def make_velocity_profile(data: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))

    # Analytic curve (smooth)
    y_fine = np.linspace(0, HEIGHT, 200)
    u_fine = [analytic_u(y) for y in y_fine]
    ax.plot(u_fine, y_fine, "k-", linewidth=2, label="Analytic $u(y)=g(H-y)y/(2\\nu)$")

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    markers = ["o", "s", "^"]

    for run, color, marker in zip(
        sorted(data["results"], key=lambda r: r["resolution_H_over_dx"]),
        colors, markers,
    ):
        res = run["resolution_H_over_dx"]
        ys = [r["y"] for r in run["rows"]]
        us = [r["u_mean"] for r in run["rows"]]
        ax.plot(us, ys, color=color, marker=marker, markersize=5,
                linestyle="", label=f"SPH $H/dx={res}$")

    ax.set_xlabel("Velocity $u_x$ (m/s)", fontsize=12)
    ax.set_ylabel("Wall-normal coordinate $y$ (m)", fontsize=12)
    ax.set_title("Plane Poiseuille Flow: SPH vs Analytic Profile", fontsize=13)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.05)

    # Add error annotation
    for run, color in zip(
        sorted(data["results"], key=lambda r: r["resolution_H_over_dx"]),
        colors,
    ):
        res = run["resolution_H_over_dx"]
        err = run["u_max_error_pct"]
        ax.text(0.02, 0.95 - 0.06 * (res // 8),
                f"$H/dx={res}$: $u_{{max}}$ err = {err:+.3f}%",
                transform=ax.transAxes, fontsize=9, color=color,
                verticalalignment="top")

    fig.tight_layout()
    out = FIGDIR / "velocity_profile.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Figure 2: Convergence plot
# --------------------------------------------------------------------------- #

def make_convergence(data: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    resolutions = [r["resolution_H_over_dx"] for r in data["results"]]
    x = np.array(resolutions, dtype=float)

    # Panel A: centreline error
    errs = [abs(r["u_max_error_pct"]) for r in data["results"]]
    axes[0].plot(x, errs, "o-", color="#e41a1c", markersize=8, linewidth=2)
    axes[0].axhline(5.0, color="gray", linestyle="--", linewidth=1, label="5 % threshold")
    axes[0].set_xlabel("Resolution $H/dx$")
    axes[0].set_ylabel("$|u_{max} - u_{ref}|/u_{ref}$ (%)")
    axes[0].set_title("A: Centreline Velocity Error")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    for xi, yi in zip(x, errs):
        axes[0].annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)

    # Panel B: profile RMS
    rms_int = [r["profile_rms_interior_pct"] for r in data["results"]]
    rms_all = [r["profile_rms_all_rows_pct"] for r in data["results"]]
    axes[1].plot(x, rms_int, "s-", color="#377eb8", markersize=8, linewidth=2,
                 label="Interior rows (0.1–0.9)")
    axes[1].plot(x, rms_all, "^-", color="#4daf4a", markersize=8, linewidth=2,
                 label="All rows")
    axes[1].axhline(5.0, color="gray", linestyle="--", linewidth=1, label="5 % threshold")
    axes[1].set_xlabel("Resolution $H/dx$")
    axes[1].set_ylabel("RMS relative error (%)")
    axes[1].set_title("B: Profile Shape Error")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel C: wall shear error
    ws_err = [abs(r["wall_shear_bottom_error_pct"]) for r in data["results"]]
    axes[2].plot(x, ws_err, "D-", color="#984ea3", markersize=8, linewidth=2)
    axes[2].axhline(5.0, color="gray", linestyle="--", linewidth=1, label="5 % threshold")
    axes[2].set_xlabel("Resolution $H/dx$")
    axes[2].set_ylabel("$|\\tau_w - \\tau_{ref}|/\\tau_{ref}$ (%)")
    axes[2].set_title("C: Wall Shear Stress Error")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    for xi, yi in zip(x, ws_err):
        axes[2].annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)

    fig.suptitle("CASE-POISEUILLE-SPH: Convergence with Resolution", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIGDIR / "convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Figure 3: Field snapshot (H/dx=16)
# --------------------------------------------------------------------------- #

def make_field_snapshot(data: dict) -> str:
    run = [r for r in data["results"] if r["resolution_H_over_dx"] == 16][0]
    field = run["field"]

    x = np.array(field["x"])
    y = np.array(field["y"])
    u = np.array(field["u"])

    fig, ax = plt.subplots(figsize=(8, 2.5))

    # Scatter plot of velocity magnitude
    sc = ax.scatter(x, y, c=u, cmap="viridis", s=8, edgecolors="none",
                    vmin=0, vmax=U_MAX_REF * 1.1)
    cbar = fig.colorbar(sc, ax=ax, label="$u_x$ (m/s)")

    # Draw wall lines
    ax.axhline(0, color="red", linewidth=1.5, linestyle="-", alpha=0.7)
    ax.axhline(HEIGHT, color="red", linewidth=1.5, linestyle="-", alpha=0.7)

    ax.set_xlabel("Streamwise coordinate $x$ (m)")
    ax.set_ylabel("Wall-normal coordinate $y$ (m)")
    ax.set_title(f"Velocity Field Snapshot ($H/dx=16$, $N_{{fluid}}={run['n_fluid']}$)", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(0, run["dx"] * 12)
    ax.set_ylim(-0.005, HEIGHT + 0.005)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = FIGDIR / "field_snapshot.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Figure 4: Error distribution per row
# --------------------------------------------------------------------------- #

def make_error_distribution(data: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    markers = ["o", "s", "^"]

    for run, color, marker in zip(
        sorted(data["results"], key=lambda r: r["resolution_H_over_dx"]),
        colors, markers,
    ):
        res = run["resolution_H_over_dx"]
        ys = [r["y_over_H"] for r in run["rows"]]
        errs = [r["rel_error"] * 100 for r in run["rows"]]  # convert to %
        ax.plot(ys, errs, color=color, marker=marker, markersize=6,
                linestyle="-", linewidth=1.5, label=f"$H/dx={res}$")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.axhline(-5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Normalized wall-normal coordinate $y/H$")
    ax.set_ylabel("Relative error $(u_{SPH} - u_{ref})/u_{ref}$ (%)")
    ax.set_title("Per-Row Relative Error vs Analytic Solution")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGDIR / "error_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    paths = []
    paths.append(make_velocity_profile(data))
    paths.append(make_convergence(data))
    paths.append(make_field_snapshot(data))
    paths.append(make_error_distribution(data))
    for p in paths:
        print(f"  wrote {p}")
    print("All figures generated.")
