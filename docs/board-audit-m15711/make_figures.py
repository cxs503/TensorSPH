#!/usr/bin/env python3
"""m15711 审计图：交付剖面 vs 解析解、四项门与 15% 精度标准的对照。

数据全部读 runs/couette_repro/results.json（本次独立重跑，零手抄）。
产出 docs/board-audit-m15711/fig_couette_audit.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = pathlib.Path("/root/tp-agent/work/m15711")
R = json.load(open(W / "runs/couette_repro/results.json"))
OUT = W / "docs/board-audit-m15711/fig_couette_audit.png"

runs = {r["label"]: r for r in R["ladder"]}
U = R["wall_speed"]
H = R["height"]
gates = R["gates"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

# (a) 剖面：got vs analytic
for label, style in (("ladder-r10", "o:"), ("ladder-r16", "s-"), ("ladder-r24", "^-")):
    rows = runs[label]["metrics"]["rows"]
    ax1.plot([p["u_mean"] / U for p in rows], [p["y_over_H"] for p in rows],
             style, ms=3.5, lw=1.2,
             label="%s (H/dx=%d, L2=%.2f%%)" % (label, runs[label]["resolution_H_over_dx"],
                                                100 * runs[label]["metrics"]["interior_l2"]))
ys = [p["y_over_H"] for p in runs["ladder-r16"]["metrics"]["rows"]]
ax1.plot(ys, ys, "k--", lw=1.6, label=r"analytic $u/U=y/H$ (Batchelor 1967)")
neg = R["friction_sweep"][0]
ax1.plot([p["u_mean"] / U for p in neg["metrics"]["rows"]],
         [p["y_over_H"] for p in neg["metrics"]["rows"]],
         "x--", color="0.55", ms=3, lw=1.0,
         label="negative control f=0 (flux ratio %.3f)" % neg["metrics"]["flux_ratio"])
ax1.set_xlabel(r"$u/U$"); ax1.set_ylabel(r"$y/H$")
ax1.set_title("(a) SPH plane Couette profile vs closed form")
ax1.grid(alpha=0.25); ax1.legend(fontsize=7, loc="upper left")

# (b) 每项指标偏差 vs 自设 5% 门 + 生态 15% 精度标准
metrics = [
    ("interior L2\nprofile", [100 * runs[k]["metrics"]["interior_l2"] for k in ("ladder-r10", "ladder-r16", "ladder-r24")]),
    ("|flux ratio-1|", [100 * abs(runs[k]["metrics"]["flux_ratio"] - 1) for k in ("ladder-r10", "ladder-r16", "ladder-r24")]),
    ("|wall shear-1|", [100 * abs(runs[k]["metrics"]["wall_shear_ratio"] - 1) for k in ("ladder-r10", "ladder-r16", "ladder-r24")]),
    ("steady drift", [100 * runs[k]["steady_drift"] for k in ("ladder-r10", "ladder-r16", "ladder-r24")]),
]
w = 0.26
for i, (name, vals) in enumerate(metrics):
    offs = [-w, 0.0, w]
    for j, v in enumerate(vals):
        ax2.bar(i + offs[j], v, width=w, label=["r10", "r16", "r24"][j] if i == 0 else None)
ax2.axhline(100 * max(gates["interior_l2"], gates["steady_drift"]) / 1.0, color="C3", ls="--", lw=1.2)
ax2.axhline(15.0, color="C0", ls=":", lw=1.4)
ax2.annotate("predeclared gates (1%~5%)", xy=(2.55, 5.2), fontsize=7, color="C3")
ax2.annotate("ecosystem accuracy standard 15%", xy=(2.55, 15.8), fontsize=7, color="C0")
ax2.set_yscale("log")
ax2.set_xticks(range(len(metrics))); ax2.set_xticklabels([m[0] for m in metrics], fontsize=8)
ax2.set_ylabel("deviation from analytic [%]")
ax2.set_title("(b) delivered deviations vs gates (log scale)")
ax2.grid(alpha=0.25, axis="y"); ax2.legend(fontsize=8)

fig.suptitle("CASE-COUETTE-SPH / m15412 board audit (m15711 independent reproduction, 0-diff)",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)