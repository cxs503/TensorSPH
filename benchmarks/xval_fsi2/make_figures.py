"""Generate publication-quality figures for the FSI2 force-from-fluid xval report.

Produces:
  fig_error_by_case.png   -- bar chart of relative error by case and operator
  fig_convergence.png     -- error vs dx for each operator on each case
  fig_viscous_deficit.png -- OSS viscous deficit vs continuum prediction
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "artifacts" / "xval_fsi2"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

RESULTS = ART / "results.json"


def load_results():
    with open(RESULTS) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Figure 1: Error by case and operator (grouped bar chart)
# --------------------------------------------------------------------------- #
def fig_error_by_case(data):
    """Grouped bar chart: relative error (%) for each operator on each case."""
    # Collect unique cases and operators
    case_ops = {}
    for row in data["rows"]:
        key = (row["case"], row["dx"])
        case_ops[key] = row["operators"]

    # Group by case name
    cases = {}
    for (case, dx), ops in case_ops.items():
        cases.setdefault(case, []).append((dx, ops))

    # Operators to show (pressure branch)
    op_labels = {
        "oss": "OSS (NoRiemann)",
        "oss_riemann": "OSS+Riemann",
        "panel": "Panel (WLS)",
        "pairwise_legacy": "Pairwise (legacy)",
    }
    op_labels_viscous = {
        "oss_viscous": "OSS viscous",
        "gradient_corrected": "Gradient-corrected",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    case_names = ["vertical_wall", "horizontal_gate", "fsi2_insert_linear", "couette_shear"]
    case_titles = [
        "X1: Vertical wall (hydrostatic)",
        "X2: Floor/gate (hydrostatic)",
        "X4: FSI2 insert body (linear field)",
        "X3: Plane Couette (viscous)",
    ]

    for idx, (case, title) in enumerate(zip(case_names, case_titles)):
        ax = axes[idx]
        if case == "couette_shear":
            ops_to_show = op_labels_viscous
        else:
            ops_to_show = op_labels

        dx_vals = sorted(set(dx for (c, dx) in case_ops if c == case))
        x = np.arange(len(dx_vals))
        width = 0.2
        n_ops = len(ops_to_show)

        for j, (op_key, op_label) in enumerate(ops_to_show.items()):
            errors = []
            for dx in dx_vals:
                ops = case_ops.get((case, dx), {})
                if op_key in ops:
                    errors.append(ops[op_key]["error"] * 100)
                else:
                    errors.append(np.nan)
            ax.bar(x + j * width, errors, width, label=op_label, alpha=0.85)

        ax.axhline(15, color="red", linestyle="--", alpha=0.5, label="±15% gate")
        ax.axhline(-15, color="red", linestyle="--", alpha=0.5)
        ax.axhline(3, color="green", linestyle="--", alpha=0.3, label="±3% target")
        ax.axhline(-3, color="green", linestyle="--", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x + width * (n_ops - 1) / 2)
        ax.set_xticklabels([f"dx={d}" for d in dx_vals], fontsize=8)
        ax.set_ylabel("Relative error (%)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("FSI2 Force-from-Fluid: Operator Error by Case and Resolution",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = FIGS / "xval_fsi2_error_by_case.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Figure 2: Convergence plot (error vs dx)
# --------------------------------------------------------------------------- #
def fig_convergence(data):
    """Log-log plot of |error| vs dx for each operator."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    case_names = ["vertical_wall", "horizontal_gate", "fsi2_insert_linear", "couette_shear"]
    case_titles = [
        "X1: Vertical wall",
        "X2: Floor/gate",
        "X4: FSI2 insert body",
        "X3: Plane Couette (viscous)",
    ]

    op_colors = {
        "oss": "#e74c3c",
        "oss_riemann": "#e67e22",
        "panel": "#2ecc71",
        "pairwise_legacy": "#9b59b6",
        "oss_viscous": "#e74c3c",
        "gradient_corrected": "#2ecc71",
    }
    op_markers = {
        "oss": "o",
        "oss_riemann": "s",
        "panel": "^",
        "pairwise_legacy": "d",
        "oss_viscous": "o",
        "gradient_corrected": "^",
    }

    for idx, (case, title) in enumerate(zip(case_names, case_titles)):
        ax = axes[idx]
        if case == "couette_shear":
            ops = ["oss_viscous", "gradient_corrected"]
        else:
            ops = ["oss", "panel", "pairwise_legacy"]

        for op in ops:
            dxs, errs = [], []
            for row in data["rows"]:
                if row["case"] == case and op in row["operators"]:
                    dxs.append(row["dx"])
                    errs.append(abs(row["operators"][op]["error"]) * 100)
            if dxs:
                ax.loglog(dxs, errs, marker=op_markers[op], color=op_colors[op],
                         label=op, linewidth=2, markersize=8)

        ax.axhline(15, color="red", linestyle="--", alpha=0.5, label="15% gate")
        ax.axhline(3, color="green", linestyle="--", alpha=0.3, label="3% target")
        ax.set_xlabel("dx [m]", fontsize=9)
        ax.set_ylabel("|Relative error| (%)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("FSI2 Force-from-Fluid: Convergence of Operator Error",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = FIGS / "xval_fsi2_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Figure 3: Viscous deficit vs continuum prediction
# --------------------------------------------------------------------------- #
def fig_viscous_deficit(data):
    """OSS viscous deficit compared to continuum truncation factor."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Couette shear results
    dxs_oss, errs_oss = [], []
    dxs_gc, errs_gc = [], []
    continuum_factors = []

    for row in data["rows"]:
        if row["case"] == "couette_shear":
            if "oss_viscous" in row["operators"]:
                dxs_oss.append(row["dx"])
                errs_oss.append(abs(row["operators"]["oss_viscous"]["error"]) * 100)
            if "gradient_corrected" in row["operators"]:
                dxs_gc.append(row["dx"])
                errs_gc.append(abs(row["operators"]["gradient_corrected"]["error"]) * 100)
            if "oss_viscous_continuum_factor" in row:
                # The continuum factor is the fraction recovered, so deficit = 1 - factor
                factor = row["oss_viscous_continuum_factor"]
                continuum_factors.append((1.0 - factor) * 100)

    # Continuum prediction (7/(3pi) ≈ 0.7427, so deficit ≈ 25.7%)
    continuum_deficit = (1.0 - 7.0 / (3.0 * math.pi)) * 100

    ax.semilogx(dxs_oss, errs_oss, "ro-", label="OSS viscous (measured)",
               linewidth=2, markersize=8)
    ax.semilogx(dxs_gc, errs_gc, "gs-", label="Gradient-corrected (measured)",
               linewidth=2, markersize=8)
    ax.axhline(continuum_deficit, color="blue", linestyle="--", linewidth=2,
              label=f"Continuum prediction ({continuum_deficit:.1f}%)")
    ax.axhline(38, color="orange", linestyle=":", alpha=0.5,
              label="Measured OSS deficit (~38%)")
    ax.axhline(15, color="red", linestyle="--", alpha=0.3, label="15% gate")
    ax.set_xlabel("dx [m]", fontsize=10)
    ax.set_ylabel("Relative error (%)", fontsize=10)
    ax.set_title("X3: Plane Couette — Viscous Branch Deficit",
                fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIGS / "xval_fsi2_viscous_deficit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Figure 4: Sensitivity to kernel and h/dx
# --------------------------------------------------------------------------- #
def fig_sensitivity(data):
    """Sensitivity of OSS error to kernel type and h/dx ratio."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sens = data.get("sensitivity", [])
    if not sens:
        print("No sensitivity data found")
        return None

    # Group by case
    cases_sens = {}
    for s in sens:
        cases_sens.setdefault(s["case"], []).append(s)

    for case, items in cases_sens.items():
        labels = [f"{s['kernel']}, h/dx={s['h_factor']}" for s in items]
        errors = [s["error"] * 100 for s in items]
        x = np.arange(len(labels))
        ax.bar(x, errors, alpha=0.7, label=case)
        for i, (l, e) in enumerate(zip(labels, errors)):
            ax.text(i, e + 1, f"{e:+.1f}%", ha="center", fontsize=8)

    ax.axhline(15, color="red", linestyle="--", alpha=0.5, label="15% gate")
    ax.axhline(-15, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks([])
    ax.set_ylabel("OSS relative error (%)", fontsize=10)
    ax.set_title("Sensitivity: Kernel and h/dx ratio", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGS / "xval_fsi2_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main():
    data = load_results()
    fig_error_by_case(data)
    fig_convergence(data)
    fig_viscous_deficit(data)
    fig_sensitivity(data)
    print("All figures generated.")


if __name__ == "__main__":
    main()
