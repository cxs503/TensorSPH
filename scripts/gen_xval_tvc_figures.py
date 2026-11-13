#!/usr/bin/env python3
"""Generate comparison figures for XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION.

Reads the existing benchmark JSON data and produces:
1. Z/a vs tau comparison curves (all variants vs experiment)
2. Error vs alpha sensitivity plot
3. Convergence plot (MAE vs dx for best variant)
"""
from __future__ import annotations

import json
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "xval_sphinx_tvc" / "data"
OUT_DIR = ROOT / "docs" / "xval_sphinx_tvc"

VARIANT_LABELS = {
    "baseline": "Baseline (no correction)",
    "tvc_a005_all": "TVC α=0.05 (all)",
    "tvc_a010_all": "TVC α=0.10 (all)",
    "tvc_a020_all": "TVC α=0.20 (all, OSS default)",
    "tvc_a005_bulk": "TVC α=0.05 (bulk scope)",
    "tvc_a005_all_c5": "TVC α=0.05 (every 5th step)",
    "pst_lind": "PST (Lind et al. 2012)",
}

VARIANT_COLORS = {
    "baseline": "#1f77b4",
    "tvc_a005_all": "#ff7f0e",
    "tvc_a010_all": "#2ca02c",
    "tvc_a020_all": "#d62728",
    "tvc_a005_bulk": "#9467bd",
    "tvc_a005_all_c5": "#8c564b",
    "pst_lind": "#e377c2",
}


def load_exp():
    exp_path = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"
    return json.loads(exp_path.read_text())


def load_data(dx_str):
    path = DATA_DIR / f"grid_dx{dx_str}.json"
    return json.loads(path.read_text())


def fig_z_over_a_comparison(dx_str="004"):
    """Z/a vs tau: all variants vs experiment (dx=0.04)."""
    d = load_data(dx_str)
    exp = load_exp()
    exp_tau = [p["tau"] for p in exp["data"]]
    exp_za = [p["Z_over_a"] for p in exp["data"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_tau, exp_za, "ko-", linewidth=2, markersize=6,
            label="Martin & Moyce (1952)", zorder=10)

    for r in d["records"]:
        v = r["variant"]
        label = VARIANT_LABELS.get(v, v)
        color = VARIANT_COLORS.get(v, None)
        ax.plot(r["tau"], r["Z_over_a"], "--", color=color, linewidth=1.5,
                label=f"{label} (MAE={r['score']['mae']:.3f})")

    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=13)
    ax.set_ylabel(r"$Z/a$", fontsize=13)
    ax.set_title(f"Dam-break front position: TensorSPH vs experiment (dx={d['records'][0]['dx']:.2f} m)",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 4.5)
    ax.set_ylim(-0.2, 5.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig_z_over_a_dx003():
    """Z/a vs tau: dx=0.03 resolution."""
    d = load_data("003")
    exp = load_exp()
    exp_tau = [p["tau"] for p in exp["data"]]
    exp_za = [p["Z_over_a"] for p in exp["data"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_tau, exp_za, "ko-", linewidth=2, markersize=6,
            label="Martin & Moyce (1952)", zorder=10)

    for r in d["records"]:
        v = r["variant"]
        label = VARIANT_LABELS.get(v, v)
        color = VARIANT_COLORS.get(v, None)
        ax.plot(r["tau"], r["Z_over_a"], "--", color=color, linewidth=1.5,
                label=f"{label} (MAE={r['score']['mae']:.3f})")

    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=13)
    ax.set_ylabel(r"$Z/a$", fontsize=13)
    ax.set_title(f"Dam-break front position: TensorSPH vs experiment (dx={d['records'][0]['dx']:.2f} m)",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 4.5)
    ax.set_ylim(-0.2, 5.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig_alpha_sensitivity():
    """MAE and mean relative error vs alpha (dx=0.04)."""
    d = load_data("004")
    alphas = []
    maes = []
    rels = []
    for r in d["records"]:
        if r["variant"].startswith("tvc_a") and "_all" in r["variant"] and "c5" not in r["variant"]:
            a = float(r["variant"].split("a")[1].split("_")[0]) / 100.0
            alphas.append(a)
            maes.append(r["score"]["mae"])
            rels.append(r["score"]["mean_rel_error"])

    # Add baseline
    for r in d["records"]:
        if r["variant"] == "baseline":
            alphas.insert(0, 0.0)
            maes.insert(0, r["score"]["mae"])
            rels.insert(0, r["score"]["mean_rel_error"])
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(alphas, maes, "o-", color="#2ca02c", linewidth=2, markersize=8)
    ax1.axhline(0.15, color="red", linestyle="--", linewidth=1, label="Owner gate (0.15)")
    ax1.set_xlabel(r"Correction coefficient $\alpha$", fontsize=12)
    ax1.set_ylabel("MAE in $Z/a$", fontsize=12)
    ax1.set_title("Absolute error vs α (dx=0.04 m)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0, 0.05, 0.10, 0.20])

    ax2.plot(alphas, rels, "o-", color="#ff7f0e", linewidth=2, markersize=8)
    ax2.axhline(0.15, color="red", linestyle="--", linewidth=1, label="Owner gate (15%)")
    ax2.set_xlabel(r"Correction coefficient $\alpha$", fontsize=12)
    ax2.set_ylabel("Mean relative error", fontsize=12)
    ax2.set_title("Relative error vs α (dx=0.04 m)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0, 0.05, 0.10, 0.20])
    ax2.set_ylim(0, 0.6)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig_convergence():
    """Convergence: MAE vs dx for best TVC variant and baseline."""
    d4 = load_data("004")
    d3 = load_data("003")

    dx_vals = []
    baseline_mae = []
    tvc_mae = []

    for d in [d4, d3]:
        dx = d["records"][0]["dx"]
        dx_vals.append(dx)
        for r in d["records"]:
            if r["variant"] == "baseline":
                baseline_mae.append(r["score"]["mae"])
            elif r["variant"] == "tvc_a010_all":
                tvc_mae.append(r["score"]["mae"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dx_vals, baseline_mae, "o-", color="#1f77b4", linewidth=2,
            markersize=10, label="Baseline (no correction)")
    ax.plot(dx_vals, tvc_mae, "s-", color="#2ca02c", linewidth=2,
            markersize=10, label="TVC α=0.10 (all)")
    ax.axhline(0.15, color="red", linestyle="--", linewidth=1,
               label="Owner gate (MAE < 0.15)")
    ax.set_xlabel("Resolution dx (m)", fontsize=12)
    ax.set_ylabel("MAE in $Z/a$", fontsize=12)
    ax.set_title("Convergence: front position error vs resolution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dx_vals)
    ax.set_xticklabels([f"{x:.2f}" for x in dx_vals])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig_clumping():
    """Particle distribution quality: close-pair fraction and median nn distance."""
    d = load_data("004")

    variants = []
    close_frac = []
    median_nn = []
    grad1_mean = []

    for r in d["records"]:
        v = r["variant"]
        label = VARIANT_LABELS.get(v, v)
        variants.append(label)
        c = r.get("clumping", {})
        close_frac.append(c.get("close_pair_fraction", 0))
        median_nn.append(c.get("median_nn_over_dx", 1.0))
        grad1_mean.append(c.get("grad1_residual_mean_h", 0))

    x = np.arange(len(variants))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(x - width/2, close_frac, width, color="#ff7f0e", alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Close-pair fraction (r < 0.7 dx)", fontsize=11)
    ax1.set_title("Particle clumping metric", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([v[:15] for v in variants], rotation=45, ha="right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    bars2 = ax2.bar(x - width/2, median_nn, width, color="#2ca02c", alpha=0.8)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Perfect lattice (1.0)")
    ax2.set_ylabel("Median r_nn / dx", fontsize=11)
    ax2.set_title("Neighbour distance regularity", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([v[:15] for v in variants], rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


if __name__ == "__main__":
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    b64_za_004 = fig_z_over_a_comparison("004")
    (out / "figure_za_dx004.png").write_bytes(base64.b64decode(b64_za_004))
    print("Wrote figure_za_dx004.png")

    b64_za_003 = fig_z_over_a_dx003()
    (out / "figure_za_dx003.png").write_bytes(base64.b64decode(b64_za_003))
    print("Wrote figure_za_dx003.png")

    b64_alpha = fig_alpha_sensitivity()
    (out / "figure_alpha_sensitivity.png").write_bytes(base64.b64decode(b64_alpha))
    print("Wrote figure_alpha_sensitivity.png")

    b64_conv = fig_convergence()
    (out / "figure_convergence.png").write_bytes(base64.b64decode(b64_conv))
    print("Wrote figure_convergence.png")

    b64_clump = fig_clumping()
    (out / "figure_clumping.png").write_bytes(base64.b64decode(b64_clump))
    print("Wrote figure_clumping.png")

    print("All figures generated.")
