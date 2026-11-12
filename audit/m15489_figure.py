"""Figure for the m15489 board audit of CASE-CAPILLARY-WAVE (m15371).

Panel (a): per-lattice-phase frequency error vs particles-per-wavelength, with the
case's ensemble mean and its 10 % tolerance band.  Shows that the design-point
pass (mean +0.24 %) is the average of phases at +-10 %.

Panel (b): surface-tension ladder -- the measured stiffness ratio is bit-identical
for sigma x 0.5/1/2, because sigma multiplies both numerator and denominator, so
the shipped 'sigma linearity' gate cannot fail (contrast with (a)).

Run::  PYTHONPATH=src OMP_NUM_THREADS=4 python audit/m15489_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
probe = json.loads((HERE / "capillary_probe_m15489.json").read_text())

ppws = (16, 24, 32, 48)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

ax = axes[0]
for i, ppw in enumerate(ppws):
    block = probe["per_phase"][f"ppw{ppw}_amp0.02"]
    errs = [r["omega_error_pct"] for r in block["rows"]]
    ax.plot([ppw] * len(errs), errs, "o", ms=6, mfc="none",
            color="#1f4e79", label="single lattice phase" if i == 0 else None)
    ax.plot(ppw, block["mean_omega_error_pct"], "s", ms=9, color="#c00000",
            label="4-phase ensemble mean (the reported number)" if i == 0 else None)
ax.axhspan(-10, 10, color="#c00000", alpha=0.08, label="case tolerance +-10 %")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel("particles per wavelength")
ax.set_ylabel(r"$\omega_{got}/\omega_{ref}-1$  [%]")
ax.set_title("(a) A/lambda = 0.02: per-phase vs reported ensemble mean")
ax.set_xticks(ppws)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower left")

ax = axes[1]
sig = [s["sigma_over_ref"] for s in probe["sigma_ladder"]]
ratio = [s["stiffness_ratio"] for s in probe["sigma_ladder"]]
ax.plot(sig, ratio, "o-", color="#c00000", label="measured stiffness ratio")
ax.axhline(1.0, color="k", ls="--", lw=1.0, label="exact (ratio = 1)")
ax.set_xlabel(r"$\sigma/\sigma_{ref}$")
ax.set_ylabel(r"$\omega^2_{got}/\omega^2_{ref}$")
ax.set_ylim(0.9, 1.1)
ax.set_title("(b) sigma ladder is exactly invariant by construction")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.text(0.5, 1.072, "identical to machine precision: gate D cannot fail",
        ha="center", fontsize=8, color="#c00000")

fig.suptitle("m15489 audit of CASE-CAPILLARY-WAVE (m15371, TensorSPH): what the "
             "ensemble mean hides", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / "capillary_audit_m15489.png"
fig.savefig(out, dpi=160)
print("wrote", out, out.stat().st_size, "bytes")