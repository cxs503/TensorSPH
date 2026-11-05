"""Evidence figure for mission m3790 (merge PR #86 into main).

Replots -- it does not recompute -- the numbers recorded by the PR in
``benchmarks/delta_s_normalization/validation_report.json`` (that file arrived
in this merge), so that the report shows whether the feature the merge carries
actually hits its analytic targets:

  left   : int delta_s dn   (analytic target 1.0, Shepard/CSF consistency)
  right  : F_cut / (2 sigma) (analytic target 1.0, cut-force identity)

Both against the three lattice spacings of the benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks/delta_s_normalization/validation_report.json"
OUT = ROOT / "docs/reports/m3790-delta-s-merge-evidence.png"

report = json.loads(DATA.read_text())
rows = report["instant"]

labels = {
    ("two_sided", "local"): ("#c44", "o", "two_sided / local  (PR baseline)"),
    ("two_sided", "reference"): ("#e8a33d", "s", "two_sided / reference"),
    ("kernel_sum", "local"): ("#3a7", "^", "kernel_sum / local  (PR new)"),
    ("kernel_sum", "reference"): ("#2b6cbf", "D", "kernel_sum / reference"),
}

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=140)
for (scheme, dens), (colour, marker, label) in labels.items():
    sel = sorted((r for r in rows if r["delta_s_scheme"] == scheme
                  and r["force_density"] == dens), key=lambda r: -r["dx_m"])
    dx = [r["dx_m"] * 1e3 for r in sel]
    integ = [r["delta_s"]["integral_dn"] for r in sel]
    cut = [abs(r["cut_force"]["ratio_measured_over_theory"]) for r in sel]
    axes[0].plot(dx, integ, marker=marker, color=colour, label=label, lw=1.6, ms=6)
    axes[1].plot(dx, cut, marker=marker, color=colour, label=label, lw=1.6, ms=6)

for ax, target, title, ylab in (
    (axes[0], 1.0, r"surface delta function normalisation", r"$\int \delta_s\,dn$  (target 1)"),
    (axes[1], 1.0, "cut-force identity", r"$|F_{cut}| / (2\sigma)$  (target 1)"),
):
    ax.axhline(target, color="k", ls="--", lw=1.0, label="analytic target")
    ax.axhspan(target * 0.85, target * 1.15, color="0.85", alpha=0.5, zorder=0)
    ax.set_xlabel(r"lattice spacing $\Delta x$ [mm]")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xlim(3.2, 1.3)
    ax.grid(alpha=0.3)

axes[0].set_ylim(0.55, 1.2)
axes[1].set_ylim(0.55, 1.2)
axes[1].legend(fontsize=7.5, loc="lower left")
fig.suptitle("m3790: delta_s schemes after merging PR #86 into main "
             "(numbers re-plotted from benchmarks/delta_s_normalization/validation_report.json)",
             fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT)
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")

# console table for the report (programmatic, never hand-copied)
print(f"{'scheme':11s} {'density':10s} {'dx[mm]':>7s} {'int ds':>8s} {'Fcut/2s':>8s} {'kappa err %':>11s}")
for r in sorted(rows, key=lambda r: (r["delta_s_scheme"], r["force_density"], -r["dx_m"])):
    print(f"{r['delta_s_scheme']:11s} {r['force_density']:10s} {r['dx_m']*1e3:7.2f} "
          f"{r['delta_s']['integral_dn']:8.3f} {r['cut_force']['ratio_measured_over_theory']:8.3f} "
          f"{r['kappa']['rel_error_pct']:11.2f}")