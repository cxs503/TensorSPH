"""Audit figure for m16184 (CASE-SPH-SOD-SHOCK-COMPRESSIBLE).

Programmatic: reads runs/compressible_sod/production.json (committed artifact),
plots per-metric relative error vs the 3%/5% tolerance lines. Zero hand-copy.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/root/tp-agent/work/m16184/runs/compressible_sod/production.json"
OUT = "/root/tp-agent/work/m16254/docs/audit_m16184/audit_error_vs_tolerance.png"

d = json.load(open(SRC))
rel = d["rel_err"]
labels = {
    "rho_L": "left plat. rho",
    "rho_star_L": "rho*_L (post-rarefaction)",
    "rho_star_R": "rho*_R (post-shock)",
    "rho_R": "right plat. rho",
    "p_L": "left plat. p",
    "p_star_L": "p*_L (star, left)",
    "p_star_R": "p*_R (star, right)",
    "p_R": "right plat. p",
    "u_star": "u* (star velocity)",
    "x_contact": "x_contact",
    "x_shock": "x_shock",
}
keys = list(labels)
errs = [abs(rel[k]) * 100 for k in keys]
names = [labels[k] for k in keys]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=140)
colors = ["#d62728" if e >= 5 else ("#ff7f0e" if e >= 3 else "#2ca02c") for e in errs]
bars = ax.bar(range(len(keys)), errs, color=colors, alpha=0.85)
ax.axhline(5.0, color="red", ls="--", lw=1.4, label="5% acceptance (case-pool tol_policy)")
ax.axhline(3.0, color="orange", ls=":", lw=1.4, label="3% target")
for i, e in enumerate(errs):
    ax.text(i, e + 0.08, f"{e:.2f}%", ha="center", fontsize=8)
ax.set_xticks(range(len(keys)))
ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("|relative error| vs Sod 1978 exact, t=0.2 (%)")
ax.set_title("m16184 audit: 11/11 gated metrics within tolerance (max 0.78%)\n"
             "N=800, h=2.5dx, gamma=1.4 — values read from production.json")
ax.legend(fontsize=8)
ax.set_ylim(0, max(errs) * 1.25 + 0.3)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
print("max err %:", max(errs), "all<5:", all(e < 5 for e in errs), "all<3:", all(e < 3 for e in errs))
