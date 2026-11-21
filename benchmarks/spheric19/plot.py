"""Overlay experimental samples and marker histories from one or more runs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    path = Path(__file__).with_name("markers_h030_fig3.csv")
    reference = np.genfromtxt(path, delimiter=",", names=True)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "spheric19"})
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.6, 6.3), layout="constrained")
    for root in args.runs:
        report = json.loads((root / "metrics.json").read_text())
        curve = np.genfromtxt(root / "markers.csv", delimiter=",", names=True)
        method = "δ⁺-SPH" if report.get("method") == "delta-plus" else "WCSPH"
        label = (f'{method} · {report["dx_m"]*1000:g} mm, '
                 f'CFL {report["courant_limit"]:g}, c₀ {report["sound_speed_m_per_s"]:g} m/s')
        if report.get("tank_contact"):
            label += " + tank contact"
        # Fixed display grid keeps the SVG compact; scores use every solver step.
        shown = np.linspace(curve["t_s"][0], curve["t_s"][-1], 1501)
        for i, ax in enumerate(axes, start=1):
            dx = np.interp(shown, curve["t_s"], curve[f"marker{i}_dx_m"])
            ax.plot(shown, 1000*dx, lw=1.35, label=label)
    for i, ax in enumerate(axes, start=1):
        pts = reference[reference["marker"] == i]
        ax.scatter(pts["t_s"], 1000*pts["dx_m"], s=15, facecolors="white",
                   edgecolors="black", linewidths=.9, zorder=4, label="Liao et al. (digitized)")
        ax.set_title(f"Marker {i} · initial height {[87.5, 65, 40][i-1]:g} mm",
                     fontsize=10, loc="left")
        ax.set_ylabel("Δx (mm)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Time after gate starts lifting (s)")
    axes[-1].set_xlim(.20, .46)
    fig.suptitle("SPHERIC Test 19 · dam break against a flexible plate · H = 0.30 m")
    fig.savefig(args.output, metadata={"Date": None} if args.output.suffix == ".svg" else {})


if __name__ == "__main__":
    main()
