"""Generate hydrostatic pressure profile figure from benchmark report.

Reads artifacts/hydrostatic/report.json and produces:
  artifacts/hydrostatic/figures/hydrostatic_profile.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def generate_figure(report_path: str | Path, out_path: str | Path):
    """Generate the hydrostatic profile comparison figure."""
    with open(report_path) as f:
        report = json.load(f)

    result = report["result"]
    ref = report["reference"]
    bands = result["measured"]["bands"]

    # Analytic profile
    H = ref["H"]
    rho0 = ref["rho0"]
    g = ref["g"]
    y_analytic = np.linspace(0, H, 100)
    p_analytic = rho0 * g * (H - y_analytic)

    # Simulation data
    y_sim = []
    p_sim = []
    p_analytic_at_bands = []
    errors = []
    for band in bands:
        y_sim.append(band["y"])
        p_sim.append(band["pressure"])
        p_analytic_at_bands.append(band["pressure_analytic"])
        errors.append(band["rel_error_pct"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pressure profile
    ax1.plot(p_analytic, y_analytic, "k-", linewidth=2, label="Analytic: $p = \\rho_0 g (H - y)$")
    colors = ["red" if b["free_surface"] else "blue" for b in bands]
    labels_used = False
    for i, (y, p, c) in enumerate(zip(y_sim, p_sim, colors)):
        label = "SPH (interior)" if c == "blue" and not labels_used else None
        label2 = "SPH (free surface)" if c == "red" and not labels_used else None
        if c == "blue":
            ax1.plot(p, y, "bo", markersize=8, label="SPH (interior)" if not labels_used else None)
            labels_used = True
        else:
            ax1.plot(p, y, "rs", markersize=8, label="SPH (free surface)" if "free" not in str(ax1.get_lines()) else None)
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
        ax2.barh(y_err, err, color="steelblue", height=0.015)
        ax2.axvline(x=0, color="k", linewidth=0.5)
        ax2.axvline(x=2.0, color="red", linestyle="--", linewidth=1, label="Gate limit (2%)")
        ax2.axvline(x=-2.0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Relative Error (%)")
        ax2.set_ylabel("Height y (m)")
        ax2.set_title("Interior Band Pressure Error")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        f"Hydrostatic Pressure Profile: dx={report['case']['dx']}, "
        f"N={result['measured']['n_particles']} particles, "
        f"max error={result['measured']['max_band_error_pct']:.1f}%"
    )
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main():
    report_path = REPO / "artifacts" / "hydrostatic" / "report.json"
    out_path = REPO / "artifacts" / "hydrostatic" / "figures" / "hydrostatic_profile.png"

    if not report_path.exists():
        print(f"Report not found: {report_path}")
        print("Run: python scripts/run_hydrostatic_benchmark.py --tier smoke")
        return 1

    generate_figure(report_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
