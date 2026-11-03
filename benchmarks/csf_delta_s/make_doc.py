"""Figure + Markdown tables for the delta-function normalisation verification.

All numbers are read from ``validation_report.json`` written by
``benchmarks/csf_delta_s/run.py``; nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorsph.droplet_impact import cubic_spline_kernel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_delta_function_normalisation.md"


def figure(payload, out=HERE / "csf_delta_s.png"):
    rows = payload["instant"]
    dxs = sorted({r["dx_m"] for r in rows})
    schemes = ["factor2", "influid"]
    style = {"factor2": dict(marker="o", color="#c0392b", ls="--"),
             "influid": dict(marker="s", color="#1f6feb", ls="-")}

    def series(scheme, path):
        out = []
        for dx in dxs:
            row = next(r for r in rows
                       if r["dx_m"] == dx and r["delta_s_scheme"] == scheme)
            for key in path.split("."):
                row = row[key]
            out.append(row)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    ax = axes[0, 0]
    for s in schemes:
        ax.plot(dxs, series(s, "delta_s.integral_dn"), label=f"delta_s scheme = {s}",
                **style[s])
    ax.axhline(1.0, color="k", lw=1, label="analytic int delta_s dn = 1")
    ax.axhspan(0.97, 1.03, color="tab:green", alpha=0.12, label="+/-3 %")
    ax.axhspan(0.85, 1.15, color="tab:orange", alpha=0.08, label="+/-15 % (gate)")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("int delta_s dn [-]")
    ax.set_title("(a) delta-function normalisation")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for s in schemes:
        ax.plot(dxs, series(s, "cut_force_density.ratio_measured_over_theory"),
                label=f"F_cut on f = -sigma kappa delta_s n ({s})", **style[s])
    ax.plot(dxs, [1.0 / v for v in
                  series("factor2", "density.shell_rho_over_rho0_weighted")],
            marker="^", color="tab:purple", ls=":",
            label="inflation 1/(rho_shell/rho0), delta_s-weighted")
    ax.axhline(1.0, color="k", lw=1)
    ax.axhspan(0.97, 1.03, color="tab:green", alpha=0.12)
    ax.axhspan(0.85, 1.15, color="tab:orange", alpha=0.08)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("F_cut / (2 sigma) [-]")
    ax.set_title("(b) cutting-force identity, force density vs density deficit")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for s in schemes:
        ax.plot(dxs, series(s, "cut_force.ratio_measured_over_theory"),
                label=f"F_cut on a = f/rho ({s})", **style[s])
    ax.axhline(1.0, color="k", lw=1)
    ax.axhspan(0.85, 1.15, color="tab:orange", alpha=0.08)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("F_cut / (2 sigma) [-]")
    ax.set_title("(c) the same identity measured through the acceleration")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    relax = payload.get("relax", [])
    if relax:
        for row in relax:
            h = row["history"]
            ax.plot([z["t_over_t_sigma"] for z in h], [z["delta_p_Pa"] for z in h],
                    label=f"Delta p, {row['delta_s_scheme']}",
                    **style[row["delta_s_scheme"]])
        ax.axhline(relax[0]["dp_theory_Pa"], color="k", lw=1,
                   label="analytic sigma/R")
        ax.set_xlabel("t / t_sigma [-]")
        ax.set_ylabel("Delta p = p_core - p_shell [Pa]")
        ax.set_title("(d) end-to-end Young-Laplace jump (relaxation)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Surface delta-function normalisation: 2|grad c| vs |grad c|/(1-c_s)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def tables(payload):
    rows = payload["instant"]
    dxs = sorted({r["dx_m"] for r in rows})
    lines = []
    for scheme in ("factor2", "influid"):
        lines.append(f"\n#### delta_s scheme = `{scheme}`\n")
        lines.append("| dx [m] | N | ∫δs dn [-] | err | <3 %? | <15 %? | "
                     "F_cut on f [-] | err | F_cut on a [-] | err | "
                     "ρ_shell/ρ0 (δs-weighted) | kappa [1/m] | κ err | cos |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for dx in dxs:
            r = next(x for x in rows
                     if x["dx_m"] == dx and x["delta_s_scheme"] == scheme)
            d, c, cd, den, k, dr = (r["delta_s"], r["cut_force"],
                                    r["cut_force_density"], r["density"],
                                    r["kappa"], r["direction"])
            lines.append(
                f"| {dx:.4f} | {r['n_particles']} | {d['integral_dn']:.4f} | "
                f"{d['rel_error_pct']:+.2f} % | {'yes' if d['within_3pct'] else 'no'} | "
                f"{'yes' if d['within_15pct'] else 'no'} | "
                f"{cd['ratio_measured_over_theory']:.4f} | {cd['rel_error_pct']:.2f} % | "
                f"{c['ratio_measured_over_theory']:.4f} | {c['rel_error_pct']:.2f} % | "
                f"{den['shell_rho_over_rho0_weighted']:.4f} | "
                f"{k['delta_s_weighted_mean_1_per_m']:.2f} | "
                f"{k['rel_error_pct']:+.1f} % | {dr['cos_band_median']:+.4f} |")
    if payload.get("relax"):
        lines.append("\n#### 时间推进（relax，dx = 0.002 m）\n")
        lines.append("| scheme | Δp [Pa] | err vs σ/R | spurious max\\|v\\|/√(σ/ρR) | <0.1? |")
        lines.append("|---|---|---|---|---|")
        for r in payload["relax"]:
            lines.append(
                f"| `{r['delta_s_scheme']}` | {r['delta_p_Pa']:.2f} | "
                f"{r['delta_p_rel_error_pct']:+.1f} % | "
                f"{r['spurious_vmax_over_vref']:.4f} | "
                f"{'yes' if r['spurious_within_0p1'] else 'no'} |")
    return "\n".join(lines)


def main():
    payload = json.loads((HERE / "validation_report.json").read_text())
    fig = figure(payload)
    body = tables(payload)
    (HERE / "tables.md").write_text(body)
    doc = DOC.read_text()
    marker = "<!-- TABLES -->"
    assert marker in doc, "doc lost its injection marker"
    doc = doc.split(marker)[0] + marker + "\n" + body + "\n"
    DOC.write_text(doc)
    print(f"figure: {fig}\ntables injected into {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()