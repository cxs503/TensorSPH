"""Generate hydrostatic pressure profile figure from results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def generate_figure(results_path: str | Path, out_path: str | Path):
    """Generate the hydrostatic profile comparison figure."""
    with open(results_path) as f:
        data = json.load(f)

    measured = data["measured"]
    config = data["config"]
    gate = data["gate"]
    bands = measured["bands"]

    # Analytic profile
    H = config["free_surface_height"]
    rho0 = 1000.0
    g = config["gravity_reference"]
    y_analytic = np.linspace(0, H, 100)
    p_analytic = rho0 * g * (H - y_analytic)

    # Simulation data
    y_interior = []
    p_interior = []
    y_surface = []
    p_surface = []
    for band in bands:
        if band["free_surface"]:
            y_surface.append(band["y"])
            p_surface.append(band["pressure"])
        else:
            y_interior.append(band["y"])
            p_interior.append(band["pressure"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pressure profile
    ax1.plot(p_analytic, y_analytic, "k-", linewidth=2, label="Analytic: $p = \\rho_0 g (H - y)$")
    if y_interior:
        ax1.plot(p_interior, y_interior, "bo", markersize=8, label="SPH (interior)")
    if y_surface:
        ax1.plot(p_surface, y_surface, "rs", markersize=8, label="SPH (free surface)")
    ax1.set_xlabel("Pressure (Pa)")
    ax1.set_ylabel("Height y (m)")
    ax1.set_title("Hydrostatic Pressure Profile")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: relative error by band
    interior_bands = [b for b in bands if not b["free_surface"]]
    if interior_bands:
        y_err = [b["y"] for b in interior_bands]
        err = [b["rel_error_pct"] for b in interior_bands]
        colors = ["steelblue" if abs(e) <= gate["limit_pct"] else "red" for e in err]
        ax2.barh(y_err, err, color=colors, height=0.015)
        ax2.axvline(x=0, color="k", linewidth=0.5)
        ax2.axvline(x=gate["limit_pct"], color="red", linestyle="--", linewidth=1, label=f"Gate limit ({gate['limit_pct']}%)")
        ax2.axvline(x=-gate["limit_pct"], color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Relative Error (%)")
        ax2.set_ylabel("Height y (m)")
        ax2.set_title("Interior Band Pressure Error")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        f"Hydrostatic Pressure Profile: dx={config['dx']}, "
        f"N={measured['n_particles']} particles, "
        f"max error={measured['max_band_error_pct']:.2f}% "
        f"({'PASS' if gate['pass'] else 'FAIL'})"
    )
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main():
    results_path = REPO / "benchmarks" / "verified" / "sph_hydrostatic" / "results.json"
    out_path = REPO / "artifacts" / "hydrostatic" / "figures" / "hydrostatic_profile.png"

    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return 1

    generate_figure(results_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
