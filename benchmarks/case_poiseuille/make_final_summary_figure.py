#!/usr/bin/env python3
"""Generate the comprehensive CASE-POISEUILLE-SPH summary figure.

Produces a multi-panel figure consolidating all verification results:
  Panel A: Velocity profiles (SPH vs analytic) for all resolutions
  Panel B: Centreline error convergence
  Panel C: Wall shear gauge battery
  Panel D: Womersley amplitude & phase vs Wo
  Panel E: Startup transient convergence
  Panel F: Verification gate summary (all-pass dashboard)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Load results
with open(REPO / "benchmarks/case_poiseuille/out/result.json") as f:
    result = json.load(f)
with open(REPO / "benchmarks/poiseuille_mirror/out/results.json") as f:
    mirror = json.load(f)

FIGDIR = REPO / "docs/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Analytic helpers
# --------------------------------------------------------------------------- #

H = 0.1
NU = 0.01
G0 = 9.81
RHO0 = 1000.0

def analytic_u(y):
    return G0 * (H - y) * y / (2.0 * NU)

u_max_ref = G0 * H**2 / (8.0 * NU)
tau_ref = RHO0 * G0 * H / 2.0

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

# ---- Panel A: Velocity profiles ----
ax = fig.add_subplot(gs[0, 0])
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for i, res in enumerate(mirror["results"]):
    hd = res["resolution_H_over_dx"]
    rows = res["rows"]
    y_vals = [r["y"] for r in rows]
    u_vals = [r["u_mean"] for r in rows]
    u_ana = [analytic_u(y) for y in y_vals]
    ax.plot(u_vals, y_vals, "o-", color=colors[i], label=f"H/dx={hd}", markersize=5)
    ax.plot(u_ana, y_vals, "--", color=colors[i], alpha=0.5)

y_fine = np.linspace(0, H, 100)
ax.plot(analytic_u(y_fine), y_fine, "k-", linewidth=1.5, label="Analytic", alpha=0.7)
ax.set_xlabel("u (m/s)")
ax.set_ylabel("y (m)")
ax.set_title("A: Velocity Profiles (SPH vs Analytic)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Panel B: Centreline error convergence ----
ax = fig.add_subplot(gs[0, 1])
resolutions = [r["resolution_H_over_dx"] for r in mirror["results"]]
errors = [abs(r["u_max_error_pct"]) for r in mirror["results"]]
ax.semilogy(resolutions, errors, "o-", color="#e74c3c", markersize=8, linewidth=2)
ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="5% threshold")
for r, e in zip(resolutions, errors):
    ax.annotate(f"{e:.3f}%", (r, e), textcoords="offset points",
                xytext=(5, 8), fontsize=8, color="#e74c3c")
ax.set_xlabel("Resolution H/dx")
ax.set_ylabel("|u_max error| (%)")
ax.set_title("B: Centreline Error Convergence")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xticks(resolutions)

# ---- Panel C: Wall shear gauge battery ----
ax = fig.add_subplot(gs[0, 2])
gauges = ["G1_secant_wall_plane", "G2_secant_row_pair", "G3_three_point_quad",
          "G4_global_fit_slope", "G5_budget_drag"]
gauge_labels = ["G1\nsecant", "G2\nrow pair", "G3\n3-pt", "G4\nfit", "G5\nbudget"]
x = np.arange(len(gauges))
width = 0.25

for i, hd in enumerate(["8", "12", "16"]):
    errs = []
    for g in gauges:
        gdata = mirror["results"][i]["fit"]
        # Use the wall shear data from the verdict
        pass

# Simplified: use the verdict data
verdict_path = REPO / "benchmarks/poiseuille_mirror/case_verdict.json"
with open(verdict_path) as f:
    verdict = json.load(f)

gate_c = [g for g in verdict["gates"] if g["id"] == "C_wall_shear_battery"][0]
per_ratio = gate_c["per_ratio"]

for i, hd in enumerate(["8", "12", "16"]):
    errs = []
    for g in gauges:
        if g in per_ratio[hd]:
            errs.append(per_ratio[hd][g]["error_pct"])
        else:
            errs.append(0)
    offset = (i - 1) * width
    ax.bar(x + offset, errs, width, label=f"H/dx={hd}", alpha=0.8)

ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5)
ax.axhline(y=-5.0, color="red", linestyle="--", alpha=0.5)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(gauge_labels, fontsize=7)
ax.set_ylabel("Wall shear error (%)")
ax.set_title("C: Wall Shear Gauge Battery")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# ---- Panel D: Womersley response ----
ax = fig.add_subplot(gs[1, 0])
harmonic_runs = result.get("harmonic_runs", [])
if harmonic_runs:
    wo_vals = [hr.get("womersley_number", 0) for hr in harmonic_runs]
    amp_errs = [hr.get("amp_centre_error_pct", 0) for hr in harmonic_runs]
    phase_errs = [hr.get("phase_centre_pct_of_quarter_turn", 0) for hr in harmonic_runs]

    ax2 = ax.twinx()
    ax.plot(wo_vals, amp_errs, "o-", color="#3498db", label="Amplitude error %", markersize=6)
    ax2.plot(wo_vals, phase_errs, "s-", color="#e67e22", label="Phase error (% of π/2)", markersize=6)
    ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(y=5.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Womersley number Wo")
    ax.set_ylabel("Amplitude error (%)")
    ax2.set_ylabel("Phase error (% of π/2)")
    ax.set_title("D: Womersley Frequency Response")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

# ---- Panel E: Startup transient ----
ax = fig.add_subplot(gs[1, 1])
constant_ladder = result.get("constant_ladder", {})
rungs = constant_ladder.get("rungs", [])
if rungs:
    for rung in rungs:
        hd = rung.get("resolution_H_over_dx", 0)
        snapshots = rung.get("snapshots", [])
        if snapshots:
            times = [s.get("t_requested", 0) for s in snapshots]
            rms = [s.get("profile_rms_pct", 0) for s in snapshots]
            ax.plot(times, rms, "o-", label=f"H/dx={hd}", markersize=4)

ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="5% threshold")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Profile RMS error (%)")
ax.set_title("E: Startup Transient Convergence")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ---- Panel F: Gate dashboard ----
ax = fig.add_subplot(gs[1, 2])
gates = result.get("gates", {})
gate_names = list(gates.keys())
gate_vals = list(gates.values())
y_pos = np.arange(len(gate_names))

colors_gate = ["#2ecc71" if v else "#e74c3c" for v in gate_vals]
bars = ax.barh(y_pos, [1.0 if v else 0.0 for v in gate_vals], color=colors_gate, alpha=0.8)

# Add gate labels
for i, (name, val) in enumerate(zip(gate_names, gate_vals)):
    short = name.replace("_5pct", "").replace("_1pct", "").replace("_of_quarter_turn", "")
    ax.text(0.05, i, f"{'✓' if val else '✗'} {short}", va="center", fontsize=7,
            color="white" if val else "white", fontweight="bold")

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.5, len(gate_names) - 0.5)
ax.set_yticks([])
ax.set_xlabel("Pass (green) / Fail (red)")
ax.set_title(f"F: Verification Gates ({sum(gate_vals)}/{len(gate_vals)} pass)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

# ---- Panel G: Profile error distribution ----
ax = fig.add_subplot(gs[2, 0])
for i, res in enumerate(mirror["results"]):
    hd = res["resolution_H_over_dx"]
    rows = res["rows"]
    y_over_H = [r["y_over_H"] for r in rows]
    rel_err = [abs(r["rel_error"]) * 100 for r in rows]
    ax.plot(y_over_H, rel_err, "o-", color=colors[i], label=f"H/dx={hd}", markersize=5)

ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("y/H")
ax.set_ylabel("|Relative error| (%)")
ax.set_title("G: Profile Error Distribution")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ---- Panel H: Effective viscosity ----
ax = fig.add_subplot(gs[2, 1])
nu_eff_ratios = []
for res in mirror["results"]:
    hd = res["resolution_H_over_dx"]
    fit = res.get("fit", {})
    nu_eff_ratios.append(fit.get("nu_eff_over_nu", 1.0))

ax.plot(resolutions, nu_eff_ratios, "o-", color="#9b59b6", markersize=8, linewidth=2)
ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="ν_eff = ν (ideal)")
for r, v in zip(resolutions, nu_eff_ratios):
    ax.annotate(f"{v:.4f}", (r, v), textcoords="offset points",
                xytext=(5, 8), fontsize=8, color="#9b59b6")
ax.set_xlabel("Resolution H/dx")
ax.set_ylabel("ν_eff / ν")
ax.set_title("H: Effective Viscosity Ratio")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xticks(resolutions)

# ---- Panel I: Summary metrics table ----
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")

table_data = [
    ["Metric", "H/dx=8", "H/dx=12", "H/dx=16", "Threshold"],
    ["|u_max error|", f"{abs(mirror['results'][0]['u_max_error_pct']):.3f}%",
     f"{abs(mirror['results'][1]['u_max_error_pct']):.3f}%",
     f"{abs(mirror['results'][2]['u_max_error_pct']):.3f}%", "<5%"],
    ["Profile RMS (all)", f"{mirror['results'][0]['profile_rms_all_rows_pct']:.3f}%",
     f"{mirror['results'][1]['profile_rms_all_rows_pct']:.3f}%",
     f"{mirror['results'][2]['profile_rms_all_rows_pct']:.3f}%", "<5%"],
    ["Wall shear (G4)", f"{abs(mirror['results'][0]['wall_shear_bottom_error_pct']):.3f}%",
     f"{abs(mirror['results'][1]['wall_shear_bottom_error_pct']):.3f}%",
     f"{abs(mirror['results'][2]['wall_shear_bottom_error_pct']):.3f}%", "<5%"],
    ["Wall slip/u_max", f"{mirror['results'][0]['wall_slip_intercept_over_umax']:.4f}",
     f"{mirror['results'][1]['wall_slip_intercept_over_umax']:.4f}",
     f"{mirror['results'][2]['wall_slip_intercept_over_umax']:.4f}", "<0.05"],
    ["Gates passed", f"{sum(gate_vals)}/{len(gate_vals)}", "", "", "All"],
]

table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                 colWidths=[0.28, 0.18, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.5)

# Color header
for j in range(5):
    table[(0, j)].set_facecolor("#34495e")
    table[(0, j)].set_text_props(color="white", fontweight="bold")

# Color pass/fail
for i in range(1, len(table_data)):
    for j in range(1, 5):
        cell = table[(i, j)]
        if j < 4:
            val = table_data[i][j].replace("%", "").replace("<", "")
            try:
                if float(val) <= 5.0 or (i == 4 and float(val) <= 0.05):
                    cell.set_facecolor("#d5f5e3")
                else:
                    cell.set_facecolor("#fadbd8")
            except:
                cell.set_facecolor("#d5f5e3")

ax.set_title("I: Summary Metrics", fontsize=10, fontweight="bold")

# ---- Save ----
outpath = FIGDIR / "case_poiseuille_final_summary.png"
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close(fig)
