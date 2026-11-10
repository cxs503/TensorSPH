#!/usr/bin/env python
"""Slamming campaign benchmark and report generator.

Runs the key slamming cases and generates figures for the report.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorsph.slamming import (
    SlammingConfig,
    SlammingImpact,
    added_mass_force,
    added_mass_per_length,
    slamming_force_coefficient,
    wagner_shape,
    pressure_shape_l2,
    pressure_noise,
    force_order,
)
from tensorsph.wedge_entry import wet_half_width, wagner_pmax, wagner_p_centre

torch.set_num_threads(4)

ART = Path(__file__).resolve().parents[1] / "artifacts" / "slamming_campaign"
ART.mkdir(parents=True, exist_ok=True)


def run_case(beta_deg, velocity, dx, delta_sph=0.1, mls_density=True, frames=11, c_end=0.04):
    """Run a single slamming case."""
    cfg = SlammingConfig(
        beta_deg=beta_deg, velocity=velocity, dx=dx,
        half_width=0.10, depth=0.05, headroom=0.04,
        c_end=c_end, frames=frames,
        delta_sph=delta_sph, mls_density=mls_density,
        initial_gap_dx=0.0,
    )
    cfg.validate()
    sim = SlammingImpact(cfg)
    series = sim.run()
    return cfg, series


def generate_figures():
    """Generate figures for the report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return []

    figures = []

    # Figure 1: Wagner pressure shape
    fig, ax = plt.subplots(figsize=(8, 5))
    c = 0.05
    for beta_deg in [10, 20, 30]:
        beta = math.radians(beta_deg)
        x = torch.linspace(0.001, 0.99 * c, 300, dtype=torch.float64)
        p = wagner_shape(x, c, beta)
        ax.plot(x / c, p.numpy(), label=f"beta={beta_deg}deg")
    ax.set_xlabel("x/c")
    ax.set_ylabel("p/p_max")
    ax.set_title("Wagner self-similar pressure distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = ART / "fig1_wagner_shape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path))

    # Figure 2: Force coefficient vs deadrise angle
    fig, ax = plt.subplots(figsize=(8, 5))
    betas = range(5, 85, 5)
    cs_wagner = [math.pi ** 2 / math.tan(math.radians(b)) for b in betas]
    ax.semilogy(betas, cs_wagner, "k-", label="Wagner C_s = pi^2/tan(beta)")
    ax.set_xlabel("Deadrise angle beta [deg]")
    ax.set_ylabel("Force coefficient C_s")
    ax.set_title("Wagner force coefficient vs deadrise angle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = ART / "fig2_force_coefficient.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path))

    # Figure 3: Added mass scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    ts = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    beta = math.radians(20.0)
    rho, V = 1000.0, 2.0
    ma = [added_mass_per_length(rho, V, beta, t) for t in ts]
    ax.loglog(ts, ma, "o-", label="m_a(t)")
    # Reference line for t^2 scaling
    t_ref = [ts[0], ts[-1]]
    ma_ref = [ma[0], ma[0] * (ts[-1] / ts[0]) ** 2]
    ax.loglog(t_ref, ma_ref, "k--", label="t^2 scaling")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Added mass m_a [kg/m]")
    ax.set_title("Wagner added mass scaling (straight wedge)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = ART / "fig3_added_mass_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path))

    return figures


def main():
    print("Running slamming campaign benchmark...")

    # Run a representative case
    print("Running beta=20, V=2, dx=4mm case...")
    cfg, series = run_case(20.0, 2.0, 4.0e-3, frames=11)

    # Collect metrics
    metrics = {
        "config": {
            "beta_deg": cfg.beta_deg,
            "velocity": cfg.velocity,
            "dx": cfg.dx,
            "c_end": cfg.c_end,
            "delta_sph": cfg.delta_sph,
            "mls_density": cfg.mls_density,
        },
        "n_fluid": int(cfg.half_width * 2 * cfg.depth / (cfg.dx ** 2)),
        "n_frames": len(series),
        "series": [],
    }

    cs_wagner = math.pi ** 2 / math.tan(math.radians(cfg.beta_deg))
    for s in series:
        metrics["series"].append({
            "t": s["t"],
            "c": s["c"],
            "force": s["force"],
            "c_s": s["c_s"],
            "c_s_wagner": cs_wagner,
            "c_s_rel_err": abs(s["c_s"] - cs_wagner) / cs_wagner if s["c_s"] > 0 else None,
            "p_max": s["p_max"],
            "p_max_wagner": s["p_max_wagner"],
            "noise": s["noise"],
            "shape_l2": s["shape_l2"],
            "n_wet": s["n_wet"],
            "energy_drift": s["energy_drift"],
        })

    # Generate figures
    figures = generate_figures()

    # Save results
    out_path = ART / "summary.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Results saved to {out_path}")
    print(f"Figures: {figures}")

    # Print summary
    ser = [s for s in series if s["t"] > 0 and s["n_wet"] >= 2]
    if ser:
        cs_vals = [s["c_s"] for s in ser if s["c_s"] > 0]
        if cs_vals:
            print(f"C_s range: [{min(cs_vals):.4f}, {max(cs_vals):.4f}]")
            print(f"C_s Wagner: {cs_wagner:.4f}")
            print(f"Max rel err: {max(abs(c - cs_wagner) / cs_wagner for c in cs_vals):.4f}")


if __name__ == "__main__":
    main()
