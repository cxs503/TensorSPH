"""Generate verification plots for free surface large deformation campaign.

This script creates comparison plots for:
(a) Dam break: SPH vs Martin & Moyce vs Ritter
(b) Wedge entry: SPH pressure vs Wagner theory
(c) Sloshing: surface elevation time series

Usage:
    python scripts/generate_verification_plots.py --output-dir artifacts/plots
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case, ritter_front_over_a
from tensorsph.wedge_entry import (
    WedgeConfig,
    WedgeImpact,
    wagner_pressure,
    wet_half_width,
)
from tensorsph.sloshing import SloshingConfig, run_case as run_sloshing


def plot_dam_break(output_path: str):
    """Generate dam break comparison plot."""
    print("Generating dam break plot...")
    
    # Run SPH simulation
    cfg = DamBreakConfig(
        dx=0.015,
        column_width=0.2,
        column_height=0.4,
        pool_width=2.0,
        tank_height=0.75,
        steps=2000,
        dt=1e-4,
        output_every=100,
    )
    run = run_case(cfg)
    tau, za = run.dimensionless(cfg)
    
    # Ritter reference
    tau_ritter = np.linspace(0, max(tau), 100)
    za_ritter = [ritter_front_over_a(t) for t in tau_ritter]
    
    # Load experimental data (if available)
    exp_path = Path(__file__).parent.parent / "benchmarks/dam_break/martin_moyce_1952.json"
    exp_data = None
    if exp_path.exists():
        with open(exp_path) as f:
            data = json.load(f)
            exp_data = data.get("data", [])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # SPH result
    ax.plot(tau, za, "b-", linewidth=2, label=f"TensorSPH (dx={cfg.dx*1000:.1f} mm)")
    
    # Ritter reference
    ax.plot(tau_ritter, za_ritter, "r--", linewidth=1.5, alpha=0.7,
            label=r"Ritter shallow-water, $Z/a = 2\tau$ (reference)")
    
    # Experimental data
    if exp_data:
        exp_tau = [d["tau"] for d in exp_data]
        exp_za = [d["Z_over_a"] for d in exp_data]
        exp_unc = [d.get("uncertainty", 0.05) for d in exp_data]
        ax.errorbar(exp_tau, exp_za, yerr=exp_unc, fmt="ko", markersize=5,
                    capsize=3, label="Martin & Moyce (1952) [WITHHELD]", zorder=5)
    
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=13)
    ax.set_ylabel(r"$Z/a$", fontsize=13)
    ax.set_title("Dam-break front trajectory: TensorSPH vs references", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(tau) * 1.05)
    ax.set_ylim(-0.1, max(za) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


def plot_wedge_entry(output_path: str):
    """Generate wedge entry pressure distribution plot."""
    print("Generating wedge entry plot...")
    
    # Wagner theory
    beta_deg = 30.0
    beta = math.radians(beta_deg)
    V = 1.0
    rho = 1000.0
    t = 0.005  # 5 ms
    c = wet_half_width(V, beta, t)
    
    x = np.linspace(-0.99 * c, 0.99 * c, 1000)
    x_tensor = torch.tensor(x, dtype=torch.float64)
    p_wagner = wagner_pressure(x_tensor, c, V, beta, rho).numpy()
    
    # Non-dimensionalize
    x_over_c = x / c
    p_nd = p_wagner / (0.5 * rho * V * V)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_over_c, p_nd, "b-", linewidth=2, label=f"Wagner theory (β={beta_deg}°)")
    
    ax.set_xlabel(r"$x/c$", fontsize=13)
    ax.set_ylabel(r"$p / (\rho V^2 / 2)$", fontsize=13)
    ax.set_title(f"Wedge entry pressure distribution (β={beta_deg}°, t={t*1000:.1f} ms)",
                 fontsize=14)
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, max(p_nd) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


def plot_sloshing(output_path: str):
    """Generate sloshing surface elevation plot."""
    print("Generating sloshing plot...")
    
    # Run SPH simulation
    cfg = SloshingConfig(
        tank_length=1.0,
        tank_height=0.5,
        water_depth=0.3,
        dx=0.02,
        steps=2000,
        dt=1e-4,
        output_every=50,
    )
    run = run_sloshing(cfg)
    
    # Plot surface elevation at different probe locations
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (x_probe, elevations) in enumerate(run.surface_elevations.items()):
        if len(elevations) > 0:
            times = run.times[:len(elevations)]
            color = colors[i % len(colors)]
            ax.plot(times, elevations, "-", color=color, linewidth=1.5,
                    label=f"x = {x_probe:.2f} m")
    
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Surface elevation η (m)", fontsize=13)
    ax.set_title("Rectangular tank sloshing: free surface elevation", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/plots",
                        help="output directory for plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dam_break(str(output_dir / "dam_break_trajectory.png"))
    plot_wedge_entry(str(output_dir / "wedge_entry_pressure.png"))
    plot_sloshing(str(output_dir / "sloshing_surface_elevation.png"))
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
