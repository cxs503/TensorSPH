#!/usr/bin/env python3
"""Audit figure for m17633: got-vs-ref error ladder, with the ship-default path overlaid.

Inputs (all committed in docs/audit_m17633/evidence/): results_committed_cc48564.json and
probe_default_kwargs.log. No number is typed here.
"""
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
EV = HERE / "evidence"
J = json.loads((EV / "results_committed_cc48564.json").read_text())

def series(prefix, key="dp_error_pct"):
    rows = [r for r in J["runs"] if r["arm"].startswith(prefix)]
    rows.sort(key=lambda r: r["R_over_dx"])
    return [r["R_over_dx"] for r in rows], [r[key] for r in rows]

rx_o, e_o = series("ours[mls")
rx_s, e_s = series("oss[paper")

# default-kwargs path (fit_quadratic), parsed from the auditor probe transcript
rx_d, e_d = [], []
for ln in (EV / "probe_default_kwargs.log").read_text().splitlines():
    m = re.search(r"R/dx=([\d.]+).*err=([+-][\d.]+)%", ln)
    if m:
        rx_d.append(float(m.group(1)))
        e_d.append(float(m.group(2)))

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
ax.axhspan(-3, 3, color="tab:green", alpha=0.10, label="|err| < 3 % (owner strict)")
ax.axhspan(-15, 15, color="tab:orange", alpha=0.07, label="|err| < 15 % (owner gate)")
ax.axhline(0, color="k", lw=0.8)
ax.semilogx(rx_o, e_o, "o-", color="tab:blue", label="TensorSPH laplace_isph (mls) — audited arm")
ax.semilogx(rx_s, e_s, "s--", color="tab:purple", label="PySPH-source port (paper psi form)")
ax.semilogx(rx_d, e_d, "^:", color="tab:red",
            label="laplace_isph_case() with ship DEFAULTS (fit_quadratic)")
for x, y in zip(rx_o, e_o):
    ax.annotate(f"{y:+.2f}%", (x, y), textcoords="offset points", xytext=(2, 8), fontsize=7,
                color="tab:blue")
for x, y in zip(rx_d, e_d):
    ax.annotate(f"{y:+.2f}%", (x, y), textcoords="offset points", xytext=(2, -12), fontsize=7,
                color="tab:red")
ax.set_xlabel(r"resolution  $R/\Delta x$")
ax.set_ylabel(r"error of  $\Delta p$  vs analytic  $\sigma/R$   [%]")
ax.set_title("m17633 audit of m17594 — projection-SPH Young–Laplace, got vs analytic referee")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
out = HERE / "AUDIT_m17594_error_ladder.png"
fig.savefig(out)
print("wrote", out, "series:",
      {"ours": list(zip(rx_o, e_o)), "oss": list(zip(rx_s, e_s)), "default": list(zip(rx_d, e_d))})