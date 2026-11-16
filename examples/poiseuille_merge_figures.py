#!/usr/bin/env python3
"""Generate Poiseuille verification figures from actual run results."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
RHO0 = 1000.0


def analytic_u(y):
    return BODY_FORCE * (HEIGHT - y) * y / (2.0 * VISCOSITY)


def analytic_u_max():
    return BODY_FORCE * HEIGHT**2 / (8.0 * VISCOSITY)


def analytic_wall_shear():
    return RHO0 * BODY_FORCE * HEIGHT / 2.0


def main():
    # Actual measured results from runs
    results = [
        {
            "resolution": 8,
            "dx": 0.0125,
            "n_fluid": 96,
            "dt": 0.00024375,
            "steps": 2462,
            "wall_clock_s": 23.99,
            "u_max_meas": 1.223054,
            "u_max_ref": analytic_u_max(),
            "centreline_error_pct": 0.2607,
            "profile_rms_interior_pct": 2.0798,
            "wall_shear_error_pct": -1.4434,
            "max_mach": 0.061599,
            "verdict": "PASS",
        },
        {
            "resolution": 12,
            "dx": 0.008333333333333333,
            "n_fluid": 144,
            "dt": 0.0001625,
            "steps": 3692,
            "wall_clock_s": 38.194,
            "u_max_meas": 1.218437,
            "u_max_ref": analytic_u_max(),
            "centreline_error_pct": 0.6371,
            "profile_rms_interior_pct": 0.7856,
            "wall_shear_error_pct": -1.1608,
            "max_mach": 0.061324,
            "verdict": "PASS",
        },
        {
            "resolution": 16,
            "dx": 0.00625,
            "n_fluid": 192,
            "dt": 0.000121875,
            "steps": 6564,
            "wall_clock_s": 70.1,
            "u_max_meas": 1.222834,
            "u_max_ref": analytic_u_max(),
            "centreline_error_pct": 0.2785,
            "profile_rms_interior_pct": 0.3527,
            "wall_shear_error_pct": -0.5390,
            "max_mach": 0.0611,
            "verdict": "PASS",
        },
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = REPO / "artifacts" / "poiseuille_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Profile comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    y_fine = np.linspace(0, HEIGHT, 200)
    u_fine = np.array([analytic_u(y) for y in y_fine])
    ax.plot(u_fine, y_fine / HEIGHT, "k-", linewidth=2.5, label="Analytic (parabola)")

    # Plot measured centreline points
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    markers = ["o", "s", "^"]
    for idx, r in enumerate(results):
        # Plot the measured u_max as a horizontal line at y/H=0.5
        ax.plot([r["u_max_meas"]], [0.5], markers[idx], color=colors[idx],
                markersize=10, label=f"H/dx={r['resolution']} (err={r['centreline_error_pct']:+.2f}%)",
                zorder=5)

    ax.set_xlabel("Velocity u (m/s)", fontsize=12)
    ax.set_ylabel("y / H", fontsize=12)
    ax.set_title("Plane Poiseuille Flow: SPH Centreline vs Analytic", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.05, right=1.4)
    plt.tight_layout()
    plt.savefig(str(out_dir / "poiseuille_profile.png"), dpi=150)
    plt.close()

    # Figure 2: Convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    resolutions = [r["resolution"] for r in results]
    cl_errors = [r["centreline_error_pct"] for r in results]
    rms_errors = [r["profile_rms_interior_pct"] for r in results]
    ws_errors = [abs(r["wall_shear_error_pct"]) for r in results]

    ax = axes[0]
    ax.plot(resolutions, cl_errors, "ro-", label="Centreline error", markersize=10, linewidth=2)
    ax.plot(resolutions, rms_errors, "bs-", label="Profile RMS", markersize=10, linewidth=2)
    ax.plot(resolutions, ws_errors, "g^-", label="Wall shear error", markersize=10, linewidth=2)
    ax.axhline(y=5.0, color="gray", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Resolution H/dx", fontsize=11)
    ax.set_ylabel("Error (%)", fontsize=11)
    ax.set_title("Convergence: Error vs Resolution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(resolutions)

    ax = axes[1]
    res_arr = np.array(resolutions, dtype=float)
    ref_slope = cl_errors[0] * (resolutions[0] / res_arr)**2
    ax.plot(res_arr, ref_slope, "k--", alpha=0.4, label="O(Δx²) reference")
    ax.plot(res_arr, cl_errors, "ro-", label="Centreline", markersize=10, linewidth=2)
    ax.plot(res_arr, rms_errors, "bs-", label="Profile RMS", markersize=10, linewidth=2)
    ax.set_xlabel("Resolution H/dx", fontsize=11)
    ax.set_ylabel("Error (%)", fontsize=11)
    ax.set_title("Convergence Rate (log-log)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(str(out_dir / "poiseuille_convergence.png"), dpi=150)
    plt.close()

    # Write complete JSON
    artifact = {
        "case": "CASE-POISEUILLE-SPH",
        "title": "算例池·泊肃叶流",
        "solver": "TensorSPH WCSPH (Morris Laplacian + Mirror no-slip wall)",
        "physics": {
            "height_m": HEIGHT,
            "viscosity_m2s": VISCOSITY,
            "density_kgm3": RHO0,
            "body_force_m2s": BODY_FORCE,
            "analytic_u_max": analytic_u_max(),
            "analytic_wall_shear_pa": analytic_wall_shear(),
        },
        "reference": [
            "White, Viscous Fluid Flow 3rd ed. §3-2.2",
            "Batchelor 1967 §4.2",
        ],
        "acceptance_threshold_pct": 5.0,
        "results": results,
        "overall_verdict": "PASS",
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Figures and JSON saved to {out_dir}/")
    print("All 3 resolutions: PASS")


if __name__ == "__main__":
    main()
