"""Generate the CASE-SPH-OSC-DROP-FREQ verification figure from round7.json data."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "out" / "round7.json"
FIG_DIR = HERE.parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with open(DATA) as f:
        return json.load(f)


def plot_convergence_ladder(data):
    """Figure 1: Q(R/dx) convergence ladder for all channels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    rows = data["analysis"]["rows"]
    ladder = data["analysis"]["ladder"]
    gate = data["gate_pct"]

    colors = {"A": "#1f77b4", "A_differential": "#2ca02c", "B": "#ff7f0e"}
    markers = {(2, 0.02): "o", (2, 0.05): "s", (3, 0.02): "^", (3, 0.05): "D"}

    for row in rows:
        ch = row["channel"]
        mode = row["mode"]
        eps = row["eps"]
        key = (mode, eps)
        hist = row["history"]
        rdx_vals = sorted(hist.keys())
        q_vals = [hist[str(r)] for r in rdx_vals]

        label = f"{ch} (m={mode}, ε={eps})"
        ax.plot(rdx_vals, q_vals, marker=markers.get(key, "o"),
                color=colors.get(ch, "gray"), label=label, linewidth=2,
                markersize=8)

    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Exact (Q = 1)")
    ax.axhline(y=1 + gate / 100, color="r", linestyle="--", linewidth=1.0,
               alpha=0.5, label=f"±{gate}% gate")
    ax.axhline(y=1 - gate / 100, color="r", linestyle="--", linewidth=1.0,
               alpha=0.5)

    ax.set_xlabel("Resolution R/dx", fontsize=13)
    ax.set_ylabel("Frequency ratio Q = ω_SPH / ω_Rayleigh", fontsize=13)
    ax.set_title("CASE-SPH-OSC-DROP-FREQ: Convergence to Rayleigh (1879) Theory\n"
                 "2-D Oscillating Drop, Mode m=2,3 | σ=0.072 N/m, ρ=1000 kg/m³, R=0.05 m",
                 fontsize=12)
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ladder)

    fig.tight_layout()
    path = FIG_DIR / "case_sph_osc_drop_freq_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_bars(data):
    """Figure 2: Error at R/dx=24 with V&V 20 uncertainty bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    rows = data["analysis"]["rows"]
    labels = []
    errors = []
    u_v_vals = []
    v_pass = []

    for i, row in enumerate(rows):
        ch = row["channel"]
        mode = row["mode"]
        eps = row["eps"]
        label = f"{ch}\nm={mode}, ε={eps}"
        labels.append(label)
        err = row.get("omega_error_at_24_pct", 0)
        errors.append(err)
        v = row.get("validation", {})
        u_v = v.get("u_v", 0) * 100  # convert to %
        u_v_vals.append(u_v)
        v_pass.append(v.get("pass", False))

    x = np.arange(len(labels))
    bar_colors = ["#2ca02c" if p else "#d62728" for p in v_pass]

    bars = ax.bar(x, errors, color=bar_colors, alpha=0.7, edgecolor="k",
                  linewidth=0.8, label="ω error at R/dx=24")

    # Add uncertainty bars
    ax.errorbar(x, errors, yerr=u_v_vals, fmt="none", color="k",
                capsize=5, linewidth=1.5, alpha=0.6, label="U_V (V&V 20)")

    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.8)
    ax.axhline(y=6, color="r", linestyle="--", linewidth=1.2, alpha=0.7,
               label="±6% gate")
    ax.axhline(y=-6, color="r", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Frequency error vs Rayleigh (%)", fontsize=12)
    ax.set_title("CASE-SPH-OSC-DROP-FREQ: Error at R/dx=24 with V&V 20 Uncertainty\n"
                 "Green = V&V 20 pass | Red = V&V 20 fail",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = FIG_DIR / "case_sph_osc_drop_freq_error_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_order_distribution(data):
    """Figure 3: Observed order of convergence distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    rows = data["analysis"]["rows"]
    p_vals = [r["p_observed"] for r in rows if r["status"] == "ok"]
    monotone = [r.get("monotone", False) for r in rows if r["status"] == "ok"]
    labels = [f"{r['channel']} m={r['mode']}" for r in rows if r["status"] == "ok"]

    colors = ["#2ca02c" if m else "#d62728" for m in monotone]

    y_pos = range(len(p_vals))
    bars = ax.barh(y_pos, p_vals, color=colors, alpha=0.7, edgecolor="k")

    ax.axvline(x=1.0, color="b", linestyle="--", linewidth=1.2, alpha=0.7,
               label="p = 1 (R4 assumption)")
    ax.axvline(x=2.0, color="g", linestyle="--", linewidth=1.2, alpha=0.7,
               label="p = 2 (second-order)")
    median_p = np.median(p_vals)
    ax.axvline(x=median_p, color="r", linestyle="-", linewidth=1.5, alpha=0.7,
               label=f"Median p = {median_p:.2f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Observed order of convergence p", fontsize=12)
    ax.set_title("Observed Convergence Order by Channel/Mode\n"
                 "Green = monotone | Red = non-monotone",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    path = FIG_DIR / "case_sph_osc_drop_freq_order.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    data = load_data()
    print(f"Loaded round7.json: {len(data['analysis']['rows'])} analysis rows")

    p1 = plot_convergence_ladder(data)
    print(f"  Figure 1: {p1.name}")

    p2 = plot_error_bars(data)
    print(f"  Figure 2: {p2.name}")

    p3 = plot_order_distribution(data)
    print(f"  Figure 3: {p3.name}")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
