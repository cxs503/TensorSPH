"""Figures for the free-surface pressure closure verification.

Reads ``report_force_balance.json`` (static force balance, produced by
``analysis.py``) and ``raw/relax_ladder.json`` (the relaxation ladder, produced
by ``relax_ladder.py``) and writes the two comparison figures used in
``docs/verification_laplace_pressure_closure.md``. Nothing is hand-entered.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FORCE = HERE / "report_force_balance.json"
LADDER = HERE / "raw" / "relax_ladder.json"
OUT = HERE / "laplace_pressure_closure.png"

CASE_LABEL = {
    "closure_off": "closure off (Tait clamp)",
    "closure_on_all_donors": "closure on, all donors",
    "closure_on_interior_layers2": "closure on, interior, 2 layers",
    "closure_on_interior_layers4": "closure on, interior, 4 layers",
}


def figure(force: dict, ladder: dict | None) -> Path:
    ref = force["reference"]["analytic_dp_Pa"]
    cases = force["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) where the Laplace pressure is generated: cumulative flux integral
    ax = axes[0]
    for c in cases:
        prof = c["force_side"]["profile"]
        r = [p["r_over_R"] for p in prof]
        cum, run = [], 0.0
        for p in prof:
            run += p["delta_p_contribution_Pa"]
            cum.append(run)
        ax.plot(r, [-v for v in cum], lw=1.6,
                label=rf"$\Delta p_{{\rm flux}}$, $\Delta x$={c['dx_m']*1e3:.1f} mm")
    ax.axhline(ref, color="k", ls="--", lw=1.2, label=r"$\sigma/R$")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$-\int_0^r \rho a_r$ [Pa]")
    ax.set_title("(a) discrete CSF force can hold $\\sigma/R$\n(flux estimator, bin-insensitive)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) closure variants on the relaxation: absolute core pressure vs sigma/R
    ax = axes[1]
    if ladder and ladder.get("cases"):
        labels, tails, finals, cols = [], [], [], []
        for c in ladder["cases"]:
            labels.append(CASE_LABEL.get(c["case"], c["case"]))
            tails.append(c["tail_rel_error_pct"])
            finals.append(c["final_abs_rel_error_pct"])
            cols.append("C0" if c["case"] == "closure_off" else "C1")
        idx = range(len(labels))
        ax.bar([i - 0.2 for i in idx], tails, width=0.4, color=cols,
               label="time-averaged tail")
        ax.bar([i + 0.2 for i in idx], finals, width=0.4, color=cols, alpha=0.45,
               label="final snapshot")
        ax.axhspan(-15, 15, color="g", alpha=0.10, label=r"$\pm15\%$ gate")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(list(idx))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
        ax.set_ylabel(r"error vs $\sigma/R$ [%]")
        ax.set_title(rf"(b) relax: $\Delta x$={ladder['dx_m']*1e3:.1f} mm, "
                     rf"$c_0$={ladder['sound_speed']:g} m/s")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "relaxation ladder not run", ha="center")

    # (c) relaxed pressure profile per case against sigma/R
    ax = axes[2]
    if ladder and ladder.get("cases"):
        for c in ladder["cases"]:
            prof = c.get("pressure_profile") or []
            if not prof:
                continue
            ax.plot([p["r_over_R_mid"] for p in prof], [p["mean"] for p in prof],
                    "o-" if c["case"] == "closure_off" else "s-", ms=3, lw=1.2,
                    label=CASE_LABEL.get(c["case"], c["case"]))
        ax.axhline(ref, color="k", ls="--", lw=1.2, label=r"$\sigma/R$")
        ax.set_xlabel(r"$r/R$")
        ax.set_ylabel(r"$p$ [Pa]")
        ax.set_title("(c) relaxed pressure profile")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    plt.close(fig)
    return OUT


def main():
    force = json.loads(FORCE.read_text())
    ladder = json.loads(LADDER.read_text()) if LADDER.exists() else None
    path = figure(force, ladder)
    print("wrote", path)


if __name__ == "__main__":
    main()
