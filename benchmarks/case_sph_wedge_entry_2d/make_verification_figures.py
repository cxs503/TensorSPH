"""Generate figures for CASE-SPH-WEDGE-ENTRY-2D verification report.

Uses existing artifact data and summary data (no SPH runs needed).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "case_sph_wedge_entry_2d"
SUMMARY = ROOT / "benchmarks" / "wedge_entry" / "summary_v2.json"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Wagner reference functions
def wagner_c_coeff(beta):
    return math.pi / (2.0 * math.tan(beta))

def wagner_p_over_q(tau, beta):
    """Wagner pressure coefficient at tau = |x|/c."""
    C = wagner_c_coeff(beta)
    s_star = C + math.sqrt(C * C + 1.0)
    tau_max = math.sqrt(max(0.0, 1.0 - 1.0 / (s_star * s_star)))
    if tau > tau_max or tau < 0:
        return 0.0
    s = 1.0 / math.sqrt(max(1.0 - tau * tau, 1e-12))
    return 2.0 * C * s - s * s + 1.0

BETA_DEG = 30.0
BETA = math.radians(BETA_DEG)
C = wagner_c_coeff(BETA)


def load_artifact(dx_mm):
    """Load artifact by dx in mm."""
    dx_m = dx_mm / 1000.0
    dx_label = int(round(dx_m * 1e4))
    path = ART / f"beta30_V1_dx{dx_label}e-4.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def fig1_pressure_profile():
    """Figure 1: Pressure profile comparison (SPH vs Wagner) from dx=4mm artifact."""
    art = load_artifact(4.0)  # coarsest grid (only one available)
    if art is None:
        print("WARNING: No artifact for dx=4mm, skipping fig1")
        return

    # Get ensemble profile from last 5 frames
    series = art["series"]
    valid_frames = [r for r in series if any(
        v is not None and math.isfinite(v) for v in r.get("profile_cp", []))]
    if len(valid_frames) < 2:
        print("WARNING: Not enough valid frames for ensemble")
        return

    sel = valid_frames[-5:]
    taus = sel[-1]["profile_tau"]
    n_bins = len(taus)

    # Ensemble mean
    means = []
    for k in range(n_bins):
        vals = [r["profile_cp"][k] for r in sel
                if k < len(r["profile_cp"]) and r["profile_cp"][k] is not None
                and math.isfinite(r["profile_cp"][k])]
        means.append(sum(vals) / len(vals) if vals else float("nan"))

    # Wagner reference
    wagner_cp = [wagner_p_over_q(t, BETA) for t in taus]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, wagner_cp, "k--", linewidth=2, label="Wagner (1932)")
    ax.plot(taus, means, "ro-", linewidth=1, markersize=3, label=f"SPH (dx=4mm, ensemble)")

    # Mark peak locations
    valid_means = [(t, m) for t, m in zip(taus, means) if math.isfinite(m)]
    if valid_means:
        sph_peak_tau, sph_peak = max(valid_means, key=lambda x: x[1])
        wagner_peak = max(wagner_cp)
        wagner_peak_tau = taus[wagner_cp.index(wagner_peak)]

        ax.axvline(x=wagner_peak_tau, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel(r"$\tau = |x|/c(t)$", fontsize=12)
        ax.set_ylabel(r"$C_p = p / (\frac{1}{2}\rho V^2)$", fontsize=12)
        ax.set_title(f"Wedge Entry Pressure Profile, $\\beta={BETA_DEG}°$, $V=1$ m/s", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, max(wagner_peak, sph_peak) * 1.15)

        # Annotate peak values
        ax.annotate(f"Wagner peak: {wagner_peak:.2f}",
                    xy=(wagner_peak_tau, wagner_peak),
                    xytext=(wagner_peak_tau + 0.15, wagner_peak * 0.85),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="k"))
        ax.annotate(f"SPH peak: {sph_peak:.2f}",
                    xy=(sph_peak_tau, sph_peak),
                    xytext=(sph_peak_tau + 0.15, sph_peak * 0.75),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="r"))

    plt.tight_layout()
    path = FIG_DIR / "wedge_entry_pressure_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig2_wetted_width():
    """Figure 2: Wetted half-width time history from dx=4mm artifact."""
    art = load_artifact(4.0)
    if art is None:
        print("WARNING: No artifact for dx=4mm, skipping fig2")
        return

    series = art["series"]
    t_vals = [r["t"] for r in series if r["t"] > 0]
    c_sph = [r["c_sph"] for r in series if r["t"] > 0]
    c_wagner = [r["c_wagner"] for r in series if r["t"] > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_vals, c_wagner, "k--", linewidth=2, label="Wagner: $c(t) = CVt$")
    ax.plot(t_vals, c_sph, "ro-", linewidth=1, markersize=4, label="SPH (dx=4mm)")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Wetted half-width $c(t)$ [m]", fontsize=12)
    ax.set_title(f"Wetted Width vs Time, $\\beta={BETA_DEG}°$, $V=1$ m/s", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate final error
    if len(t_vals) > 0:
        final_err = abs(c_sph[-1] - c_wagner[-1]) / c_wagner[-1] * 100
        ax.text(0.05, 0.95, f"Final error: {final_err:.1f}%",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = FIG_DIR / "wedge_entry_wetted_width.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig3_grid_convergence():
    """Figure 3: Grid convergence of Cp_max and c/(Vt)."""
    # Data from existing verification doc and summary_v2.json
    dx_mm = [4.0, 2.0, 1.0, 0.5]
    # Cp_max errors from verification_wedge_entry.md Table 3.2 and summary
    cp_max_err = [9.2, 26.5, 12.4, 1.1]  # percent
    # c/(Vt) values from verification_wedge_entry.md Table 3.2
    c_coeff = [2.679, 2.783, 2.657, 2.686]
    c_ref = C  # 2.7207
    c_coeff_err = [abs(c - c_ref) / c_ref * 100 for c in c_coeff]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Cp_max error
    ax1.semilogx(dx_mm, cp_max_err, "ro-", linewidth=2, markersize=8)
    ax1.axhline(y=15, color="r", linestyle="--", alpha=0.5, label="15% tolerance")
    ax1.axhline(y=3, color="g", linestyle="--", alpha=0.5, label="3% (high precision)")
    ax1.set_xlabel("Particle spacing $dx$ [mm]", fontsize=11)
    ax1.set_ylabel("$C_{p,\\max}$ relative error [%]", fontsize=11)
    ax1.set_title("Peak Pressure Coefficient Error", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    for x, y in zip(dx_mm, cp_max_err):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # Right: c/(Vt) error
    ax2.semilogx(dx_mm, c_coeff_err, "bs-", linewidth=2, markersize=8)
    ax2.axhline(y=5, color="r", linestyle="--", alpha=0.5, label="5% GCI threshold")
    ax2.axhline(y=3, color="g", linestyle="--", alpha=0.5, label="3% (high precision)")
    ax2.set_xlabel("Particle spacing $dx$ [mm]", fontsize=11)
    ax2.set_ylabel("$c/(Vt)$ relative error [%]", fontsize=11)
    ax2.set_title("Wetted Width Coefficient Error", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    for x, y in zip(dx_mm, c_coeff_err):
        ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "wedge_entry_grid_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig4_cp_centre_convergence():
    """Figure 4: Stagnation point pressure convergence."""
    # Cp_centre values from verification_wedge_entry.md
    dx_mm = [4.0, 2.0, 1.0, 0.5]
    cp_centre = [4.82, 5.12, 5.28, 5.38]
    cp_centre_ref = 2 * C  # 5.4414
    cp_centre_err = [abs(c - cp_centre_ref) / cp_centre_ref * 100 for c in cp_centre]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(dx_mm, cp_centre_err, "gs-", linewidth=2, markersize=8)
    ax.axhline(y=15, color="r", linestyle="--", alpha=0.5, label="15% tolerance")
    ax.axhline(y=3, color="g", linestyle="--", alpha=0.5, label="3% (high precision)")
    ax.set_xlabel("Particle spacing $dx$ [mm]", fontsize=11)
    ax.set_ylabel("$C_{p,\\text{centre}}$ relative error [%]", fontsize=11)
    ax.set_title("Stagnation Point Pressure Convergence", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for x, y in zip(dx_mm, cp_centre_err):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "wedge_entry_cp_centre_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    fig1_pressure_profile()
    fig2_wetted_width()
    fig3_grid_convergence()
    fig4_cp_centre_convergence()
    print("All figures generated.")
