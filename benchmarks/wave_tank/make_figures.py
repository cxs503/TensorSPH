"""Figures for the wave-tank report.

Reads the case JSON files and the field dumps from ``viz_case.py`` and writes the
panels of ``docs/benchmarks/wave-tank-2d.template.md``:

    docs/figures/wave_tank_dispersion.png   phase speed and wave number vs kh
    docs/figures/wave_tank_ladder.png       dispersion error against resolution
    docs/figures/wave_tank_decay.png        wave height along the tank
    docs/figures/wave_tank_field.png        particle cloud + elevation profiles
    docs/figures/wave_tank_solitary.png     solitary wave profiles and celerity

    python benchmarks/wave_tank/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "runs", "wave_tank")
FIGURES = os.path.join(HERE, "..", "..", "docs", "figures")

KHS = ("dispersion_kh0.5", "dispersion_kh1", "dispersion_kh2", "dispersion_kh3")
LADDER = ("ladder_dp0.020", "ladder_dp0.015", "ladder_dp0.010", "ladder_dp0.0075")


def load(name: str):
    path = os.path.join(RUNS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fig_dispersion() -> None:
    rows = [(name, load(name)) for name in KHS]
    rows = [(name, payload) for name, payload in rows if payload is not None]
    if not rows:
        print("skip dispersion: no cases")
        return
    kh = [payload["config"]["kh"] for _, payload in rows]
    measured = [payload["analysis"]["dispersion"]["celerity_measured"] for _, payload in rows]
    theory = [payload["analysis"]["dispersion"]["celerity_theory"] for _, payload in rows]
    k_measured = [payload["analysis"]["dispersion"]["k_measured"] for _, payload in rows]
    k_theory = [payload["analysis"]["k_theory"] for _, payload in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].axhspan(0.97, 1.03, color="0.85", label="±3 % gate")
    axes[0].plot(kh, [m / t for m, t in zip(measured, theory)], "o-", color="C0")
    axes[0].axhline(1.0, color="0.4", lw=0.8, ls="--")
    axes[0].set_xlabel("kh")
    axes[0].set_ylabel("$c_{SPH} / c_{Airy}(k_{SPH})$")
    axes[0].set_title("phase speed vs linear dispersion")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_theory, k_measured, "s-", color="C3")
    limit = [min(k_theory) * 0.9, max(k_theory) * 1.1]
    axes[1].plot(limit, limit, color="0.4", lw=0.8, ls="--")
    axes[1].set_xlabel("$k$ from $\\omega^2 = gk\\tanh(kh)$ (1/m)")
    axes[1].set_ylabel("$k$ from the probe phase lag (1/m)")
    axes[1].set_title("wave number, two-probe measurement")
    axes[1].grid(alpha=0.3)
    for index, (name, payload) in enumerate(rows):
        error = 100.0 * payload["analysis"]["dispersion"]["celerity_error"]
        axes[0].annotate(f"{error:.1f}%", (kh[index], measured[index] / theory[index]),
                         textcoords="offset points", xytext=(4, 6), fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "wave_tank_dispersion.png"), dpi=140)
    plt.close(figure)
    print("wrote wave_tank_dispersion.png")


def fig_ladder() -> None:
    points = []
    for name in LADDER:
        payload = load(name)
        if payload is None:
            continue
        points.append((payload["config"]["dx_over_depth"],
                       payload["analysis"]["dispersion"]["celerity_error"],
                       payload["run"]["wall_seconds"]))
    if len(points) < 2:
        print("skip ladder: not enough cases")
        return
    dp = np.array([point[0] for point in points])
    error = np.array([point[1] for point in points])
    cost = np.array([point[2] for point in points])

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].loglog(dp, error, "o-", color="C0", label="measured")
    slope = np.polyfit(np.log(dp), np.log(error), 1)[0]
    fit = error[0] * (dp / dp[0]) ** slope
    axes[0].loglog(dp, fit, ":", color="0.4", label=f"fit: error ~ dp^{slope:.2f}")
    axes[0].axhline(0.03, color="C3", ls="--", lw=0.8, label="3 % gate")
    axes[0].set_xlabel("dp / h")
    axes[0].set_ylabel("dispersion error")
    axes[0].set_title("resolution ladder at kh = 3")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].loglog(cost, error, "s-", color="C2")
    for dp_value, error_value, cost_value in points:
        axes[1].annotate(f"{dp_value:.4f}h", (cost_value, error_value),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[1].set_xlabel("wall-clock cost (s)")
    axes[1].set_ylabel("dispersion error")
    axes[1].set_title("error against cost")
    axes[1].grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "wave_tank_ladder.png"), dpi=140)
    plt.close(figure)
    print("wrote wave_tank_ladder.png")


def fig_decay() -> None:
    payload = load("decay10")
    if payload is None:
        print("skip decay: no case")
        return
    analysis = payload["analysis"]
    wavelength = payload["config"]["wavelength"]
    x = np.array(analysis["probe_x"]) / wavelength
    heights = np.array(analysis["probe_heights"])
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.plot(x, heights / heights[0], "o-", color="C0", label="H(x)/H(0.5 λ)")
    axes.axhline(0.95, color="C3", ls="--", lw=0.9, label="0.95 gate")
    axes.set_xlabel("distance from the paddle (wavelengths)")
    axes.set_ylabel("wave height / first-probe height")
    axes.set_title("wave-height decay over ten wavelengths (kh = 3, dp = 0.02 h)")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    ratio = analysis["decay"]["ratio"]
    axes.annotate(f"H/H$_0$ = {ratio:.3f} after "
                  f"{analysis['decay']['wavelengths_between']:.1f} λ",
                  (0.55, 0.15), xycoords="axes fraction", fontsize=9)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "wave_tank_decay.png"), dpi=140)
    plt.close(figure)
    print("wrote wave_tank_decay.png")


def fig_field() -> None:
    path = os.path.join(RUNS, "viz_regular.npz")
    if not os.path.exists(path):
        print("skip field: no viz dump")
        return
    data = np.load(path)
    count = len([key for key in data.files if key.startswith("x_")])
    figure = plt.figure(figsize=(10.0, 6.0))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    for index in range(min(3, count)):
        axis = figure.add_subplot(grid[0, index])
        speed = data[f"speed_{index}"]
        scatter = axis.scatter(data[f"x_{index}"], data[f"y_{index}"], c=speed,
                               s=1.2, cmap="viridis", vmin=0.0,
                               vmax=float(np.percentile(data[f"speed_{index}"], 99.5)))
        axis.set_xlim(0.0, float(data["length"]))
        axis.set_ylim(0.0, float(data["depth"]) * 1.5)
        axis.set_title(f"t = {float(data[f't_{index}']):.2f} s", fontsize=9)
        axis.set_xlabel("x (m)", fontsize=8)
        axis.tick_params(labelsize=7)
        figure.colorbar(scatter, ax=axis, label="|v| (m/s)", shrink=0.85)

    axis = figure.add_subplot(grid[1, :])
    reference = None
    for index in range(min(3, count)):
        if reference is None:
            reference = data[f"eta_{index}"]
        axis.plot(data[f"centres_{index}"], 1000.0 * data[f"eta_{index}"],
                  lw=1.2, label=f"t = {float(data[f't_{index}']):.2f} s")
    axis.plot(data["probe_x"], [0.0] * len(data["probe_x"]), "k|", ms=8,
              label="probes")
    axis.axvline(float(data["sponge_start"]), color="C3", ls="--", lw=0.9,
                 label="sponge start")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("elevation η (mm)")
    axis.set_title("free-surface elevation profiles (kh = 3, dp = 0.02 h)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=4)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "wave_tank_field.png"), dpi=140)
    plt.close(figure)
    print("wrote wave_tank_field.png")


def fig_solitary() -> None:
    payload = load("solitary")
    path = os.path.join(RUNS, "viz_solitary.npz")
    if payload is None or not os.path.exists(path):
        print("skip solitary: missing case or viz dump")
        return
    data = np.load(path)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(data["analytic_x"], 1000.0 * data["analytic_x0"], "--", color="0.4",
                 label="Boussinesq H sech²(β(x-x₀))")
    count = len([key for key in data.files if key.startswith("eta_")])
    for index in range(count):
        axes[0].plot(data[f"centres_{index}"], 1000.0 * data[f"eta_{index}"], lw=1.4,
                     label=f"t = {float(data[f't_{index}']):.2f} s")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("η (mm)")
    axes[0].set_title("solitary wave: initial and propagated profile")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    analysis = payload["analysis"]
    times = analysis["crest_times"]
    positions = payload["config"]["probe_x"]
    measured = analysis["celerity_measured"]
    theory = analysis["celerity_theory"]
    axes[1].plot(times, positions, "o", color="C0", label="crest arrival")
    span = [min(times) - 0.1, max(times) + 0.1]
    axes[1].plot(span, [positions[0] + measured * (t - times[0]) for t in span],
                 "-", color="C0", lw=1.0, label=f"fit c = {measured:.3f} m/s")
    axes[1].plot(span, [positions[0] + theory * (t - times[0]) for t in span],
                 "--", color="C3", lw=1.0, label=f"√(g(h+H)) = {theory:.3f} m/s")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("crest position (m)")
    axes[1].set_title(f"celerity error {100 * analysis['celerity_error']:.1f} %, "
                      f"waveform L2 {100 * analysis['waveform']['l2_error']:.1f} %")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "wave_tank_solitary.png"), dpi=140)
    plt.close(figure)
    print("wrote wave_tank_solitary.png")


def main() -> int:
    os.makedirs(FIGURES, exist_ok=True)
    fig_dispersion()
    fig_ladder()
    fig_decay()
    fig_field()
    fig_solitary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
