"""Generate Round 9 figure: force decomposition trajectory for dynamic-arm root cause.

Plots:
  1. |a_press|_max and |a_csf|_max vs step (showing acoustic runaway)
  2. Radial mean of a_press and a_csf (showing pressure dominates)
  3. Velocity and amplitude growth (showing divergence)

All data from out/round9_rootcause.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "out" / "round9_rootcause.json"
FIG_DIR = HERE.parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open(DATA) as f:
        d = json.load(f)

    traj = d["trajectory"]
    steps = [r["step"] for r in traj]
    t_vals = [r["t"] for r in traj]
    a_press_max = [r["a_press_max"] for r in traj]
    a_csf_max = [r["a_csf_max"] for r in traj]
    a_press_rad = [r["a_press_radial_mean"] for r in traj]
    a_csf_rad = [r["a_csf_radial_mean"] for r in traj]
    vmax = [r["vmax"] for r in traj]
    amp = [r["amp"] for r in traj]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fig 1: Max acceleration comparison
    ax1 = axes[0, 0]
    ax1.plot(steps, a_press_max, "o-", color="#d62728", label="|a_press|_max",
             linewidth=2, markersize=6)
    ax1.plot(steps, a_csf_max, "s-", color="#2ca02c", label="|a_csf|_max",
             linewidth=2, markersize=6)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Max acceleration")
    ax1.set_title("Max acceleration: pressure vs CSF")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95,
             f"Pressure grows {a_press_max[0]:.1f} → {a_press_max[-1]:.1f}\n"
             f"CSF stays ~{a_csf_max[0]:.1f}–{max(a_csf_max):.1f}",
             transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Fig 2: Radial mean comparison
    ax2 = axes[0, 1]
    ax2.plot(steps, a_press_rad, "o-", color="#d62728", label="⟨a_press⟩_radial",
             linewidth=2, markersize=6)
    ax2.plot(steps, a_csf_rad, "s-", color="#2ca02c", label="⟨a_csf⟩_radial",
             linewidth=2, markersize=6)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Radial mean acceleration")
    ax2.set_title("Radial mean: pressure vs CSF")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Fig 3: Velocity growth
    ax3 = axes[1, 0]
    ax3.plot(steps, vmax, "D-", color="#1f77b4", label="v_max",
             linewidth=2, markersize=6)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("v_max")
    ax3.set_title("Velocity growth (divergence)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Fig 4: Amplitude growth
    ax4 = axes[1, 1]
    ax4.plot(steps, amp, "^-", color="#ff7f0e", label="Mode amplitude",
             linewidth=2, markersize=6)
    ax4.axhline(y=d["config"]["eps"], color="k", linestyle="--",
                linewidth=1.5, alpha=0.7, label=f"Initial ε={d['config']['eps']}")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Mode amplitude")
    ax4.set_title("Amplitude growth (divergence)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        f"Round 9: Dynamic-arm root cause — acoustic startup instability\n"
        f"Config: R=1, σ=1, ρ₀=1, dx=1/9, c₀=5, γ=7, δ_sph=0.1, α_av=0.02",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig_path = FIG_DIR / "case_sph_osc_drop_freq_round9_rootcause.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Print key numbers for report
    print(f"\n=== Key Numbers ===")
    print(f"Laplace pressure: {d['laplace_pressure']}")
    print(f"Tait compression: {d['tait_compression_rel']:.4%}")
    print(f"Initial |a_press|_max: {traj[0]['a_press_max']:.1f}")
    print(f"Final |a_press|_max: {traj[-1]['a_press_max']:.1f}")
    print(f"Growth factor: {traj[-1]['a_press_max']/traj[0]['a_press_max']:.1f}x")
    print(f"Initial |a_csf|_max: {traj[0]['a_csf_max']:.1f}")
    print(f"Final |a_csf|_max: {traj[-1]['a_csf_max']:.1f}")


if __name__ == "__main__":
    main()
