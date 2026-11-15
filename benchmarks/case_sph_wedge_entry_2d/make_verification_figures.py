"""Generate figures for CASE-SPH-WEDGE-ENTRY-2D verification report.

Uses only committed artifact data (no SPH runs needed).
Artifacts live in docs/artifacts/case_sph_wedge_entry_2d_*/.
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
ENS_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
BIAS_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias"
LADDER_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ladder"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Wagner reference functions
# --------------------------------------------------------------------------- #
def wagner_c_coeff(beta):
    """Wagner wetted-width coefficient C = pi / (2 tan beta)."""
    return math.pi / (2.0 * math.tan(beta))


def wagner_p_over_q(tau, beta):
    """Wagner pressure coefficient at tau = |x|/c.

    Returns Cp = p / (0.5 * rho * V^2) for 0 <= tau <= tau_max.
    """
    C = wagner_c_coeff(beta)
    s_star = C + math.sqrt(C * C + 1.0)
    tau_max = math.sqrt(max(0.0, 1.0 - 1.0 / (s_star * s_star)))
    if tau > tau_max or tau < 0:
        return 0.0
    s = 1.0 / math.sqrt(max(1.0 - tau * tau, 1e-12))
    return 2.0 * C * s - s * s + 1.0


BETA_DEG = 30.0
BETA = math.radians(BETA_DEG)
C_WAGNER = wagner_c_coeff(BETA)
CP_MAX_REF = C_WAGNER ** 2 + 1.0          # 8.4022
CP_CENTRE_REF = 2.0 * C_WAGNER             # 5.4414

# Published c/(Vt) coefficients from verification_wedge_entry.md Table 3.2
PUBLISHED_C_COEFF = {4.0: 2.679, 2.0: 2.783, 1.0: 2.657, 0.5: 2.686}
# Published Cp_max values from verification_wedge_entry.md Table 3.2
PUBLISHED_CP_MAX = {4.0: 9.179, 2.0: 6.180, 1.0: 7.357, 0.5: 8.31}
# Published Cp_centre values from verification_wedge_entry.md Table 3.2
PUBLISHED_CP_CENTRE = {4.0: 4.82, 2.0: 5.12, 1.0: 5.28, 0.5: 5.38}


def _load_ensemble(dx_mm):
    """Load ensemble artifact by dx in mm."""
    candidates = list(ENS_DIR.glob(f"ensemble_dx{dx_mm}mm*.json"))
    if not candidates:
        dx_m = dx_mm / 1e3
        candidates = list(ENS_DIR.glob(f"ensemble_dx{dx_m:g}*.json"))
    if candidates:
        return json.loads(candidates[0].read_text())
    return None


def _load_ladder():
    path = LADDER_DIR / "ladder_closure.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# --------------------------------------------------------------------------- #
# Figure 1: Pressure profile comparison (SPH peak vs Wagner curve)
# --------------------------------------------------------------------------- #
def fig1_pressure_profile():
    """Figure 1: Wagner pressure profile with SPH peak markers from ensemble."""
    tau = np.linspace(0, 1.15, 500)
    cp_wagner = np.array([wagner_p_over_q(t, BETA) for t in tau])

    # Collect SPH peak data from all ensemble grids
    dx_list = [4.0, 2.0, 1.0, 0.5]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau, cp_wagner, "k--", linewidth=2, label="Wagner (1932)")

    for dx, color in zip(dx_list, colors):
        art = _load_ensemble(dx)
        if art is None:
            continue
        f61 = art["by_frames"].get("61", {})
        rb = f61.get("resolution_band", {})
        peak = rb.get("peak_median")
        err = rb.get("err_vs_wagner_median")
        if peak is None:
            continue
        # Place peak marker at Wagner's tau_peak for visual comparison
        tau_peak = art.get("wagner_tau_peak", 0.85)
        ax.scatter([tau_peak], [peak], color=color, s=80, zorder=5,
                   label=f"SPH dx={dx}mm  Cp={peak:.2f}  err={err:.1f}%")

    ax.set_xlabel(r"$\tau = |x|/c(t)$", fontsize=12)
    ax.set_ylabel(r"$C_p = p / (\frac{1}{2}\rho V^2)$", fontsize=12)
    ax.set_title(f"Wedge Entry Pressure Profile, $\\beta={BETA_DEG}°$, $V=1$ m/s", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, CP_MAX_REF * 1.25)

    # Annotate Wagner peak
    tau_vals = np.linspace(0, 1.0, 200)
    cp_vals = [wagner_p_over_q(t, BETA) for t in tau_vals]
    idx_max = int(np.argmax(cp_vals))
    ax.annotate(f"Wagner peak: {CP_MAX_REF:.2f}",
                xy=(tau_vals[idx_max], cp_vals[idx_max]),
                xytext=(0.55, CP_MAX_REF * 0.75),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="k"))

    plt.tight_layout()
    path = FIG_DIR / "wedge_entry_pressure_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# Figure 2: Wetted width time history
# --------------------------------------------------------------------------- #
def fig2_wetted_width():
    """Figure 2: Wagner wetted-width c(t)=CVt with SPH c/(Vt) coefficients."""
    # Time axis (observation window ends when c_end = 0.05 m)
    c_end = 0.05
    t_end = 2.0 * c_end * math.tan(BETA) / (math.pi * 1.0)  # ~0.0184 s
    t = np.linspace(0, t_end, 200)
    c_wagner = C_WAGNER * 1.0 * t  # V = 1 m/s

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t * 1000, c_wagner * 1000, "k--", linewidth=2,
            label=f"Wagner: $c(t) = {C_WAGNER:.4f}\\,Vt$")

    # SPH c/(Vt) coefficients from published values
    dx_list = [4.0, 2.0, 1.0, 0.5]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for dx, color in zip(dx_list, colors):
        c_coeff = PUBLISHED_C_COEFF.get(dx)
        if c_coeff is not None:
            c_sph = c_coeff * 1.0 * t
            ax.plot(t * 1000, c_sph * 1000, "o-",
                    color=color, linewidth=1, markersize=4,
                    label=f"SPH dx={dx}mm  c/(Vt)={c_coeff:.3f}")

    ax.set_xlabel("Time [ms]", fontsize=12)
    ax.set_ylabel("Wetted half-width $c(t)$ [mm]", fontsize=12)
    ax.set_title(f"Wetted Width vs Time, $\\beta={BETA_DEG}°$, $V=1$ m/s", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate final error for finest grid
    c_finest = PUBLISHED_C_COEFF.get(0.5, C_WAGNER)
    final_err = abs(c_finest - C_WAGNER) / C_WAGNER * 100
    ax.text(0.05, 0.95, f"Finest grid (dx=0.5mm) error: {final_err:.1f}%",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = FIG_DIR / "wedge_entry_wetted_width.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# Figure 3: Grid convergence of Cp_max and c/(Vt)
# --------------------------------------------------------------------------- #
def fig3_grid_convergence():
    """Figure 3: Grid convergence of Cp_max and c/(Vt) from committed data."""
    dx_mm = [4.0, 2.0, 1.0, 0.5]

    # Cp_max errors: compute from published values vs Wagner reference
    cp_max_err = []
    for dx in dx_mm:
        cp_sph = PUBLISHED_CP_MAX.get(dx)
        if cp_sph is not None:
            cp_max_err.append(abs(cp_sph - CP_MAX_REF) / CP_MAX_REF * 100)

    # c/(Vt) errors
    c_coeff_err = []
    for dx in dx_mm:
        c_coeff = PUBLISHED_C_COEFF.get(dx)
        if c_coeff is not None:
            c_coeff_err.append(abs(c_coeff - C_WAGNER) / C_WAGNER * 100)

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


# --------------------------------------------------------------------------- #
# Figure 4: Stagnation point pressure convergence
# --------------------------------------------------------------------------- #
def fig4_cp_centre_convergence():
    """Figure 4: Stagnation point pressure convergence from published data."""
    dx_mm = [4.0, 2.0, 1.0, 0.5]
    cp_centre_err = []
    for dx in dx_mm:
        cp_c = PUBLISHED_CP_CENTRE.get(dx)
        if cp_c is not None:
            cp_centre_err.append(abs(cp_c - CP_CENTRE_REF) / CP_CENTRE_REF * 100)

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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    fig1_pressure_profile()
    fig2_wetted_width()
    fig3_grid_convergence()
    fig4_cp_centre_convergence()
    print("All figures generated.")
