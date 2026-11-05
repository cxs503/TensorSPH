"""Generate the closure figures and the doc tables from closure_ladder.json.

Numbers in ``docs/verification_free_surface_pressure_closure.md`` are pasted by
this script (nothing hand-copied): it writes the figure PNG next to the ladder
and prints the markdown fragments.

Usage
-----
    python benchmarks/laplace_pressure_floor/make_closure_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LADDER = HERE / "closure_ladder.json"
FIG = HERE / "closure_ladder.png"
DOC = HERE.parents[1] / "docs" / "verification_free_surface_pressure_closure.md"

COLORS = {(0.002, 20.0): "#1f77b4", (0.002, 5.0): "#ff7f0e",
          (0.0015, 20.0): "#2ca02c", (0.0015, 5.0): "#d62728"}


def label(key):
    dx, c0 = key
    return f"dx={dx * 1000:.1f} mm, c0={c0:.0f} m/s"


def main():
    data = json.loads(LADDER.read_text())
    rows = data["ladder"]
    analytic = data["reference"]["delta_p_Pa"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # (a) error vs sweeps ---------------------------------------------------
    ax = axes[0]
    points = sorted({(r["dx_m"], r["sound_speed"]) for r in rows})
    for key in points:
        sel = sorted((r for r in rows if (r["dx_m"], r["sound_speed"]) == key),
                     key=lambda r: r["passes"])
        xs = [max(r["passes"], 0.5) for r in sel]
        ys = [r["rel_error_pct"] for r in sel]
        ax.plot(xs, ys, "o-", color=COLORS[key], label=label(key))
        bad = [(x, r["rel_error_pct"]) for x, r in zip(xs, sel)
               if not r["spurious_current_ok"]]
        if bad:
            ax.scatter([b[0] for b in bad], [b[1] for b in bad],
                       marker="x", s=70, color=COLORS[key], zorder=5)
    ax.axhspan(-15.0, 15.0, color="0.85", zorder=0,
               label="P0 acceptance band (+-15 %)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(-38.0, color="0.45", ls="--", lw=0.8,
               label="open-loop residual (-38 %)")
    ax.set_xlabel("closure sweeps per step  (0.5 = off)")
    ax.set_ylabel("error in $\\Delta p$ vs $\\sigma/R$  [%]")
    ax.set_xticks([0.5, 1, 2, 3, 6])
    ax.set_xticklabels(["off", "1", "2", "3", "6"])
    ax.set_title("(a) time-averaged Laplace jump vs closure sweeps\n"
                 "crosses = Adami spurious-current criterion violated")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # (b) radial pressure profile, off vs three sweeps ----------------------
    ax = axes[1]
    for key in points:
        for passes, style in ((0, ":"), (3, "-")):
            sel = [r for r in rows if (r["dx_m"], r["sound_speed"]) == key
                   and r["passes"] == passes]
            if not sel:
                continue
            prof = sel[0]["pressure_profile"]
            r = [p["r_over_R_mid"] for p in prof]
            p = [p["mean"] for p in prof]
            ax.plot(r, p, style, marker="o" if passes == 3 else None,
                    ms=3, color=COLORS[key],
                    label=f"{label(key)}  {'off' if passes == 0 else '3 sweeps'}")
    ax.axhline(analytic, color="k", ls="--", lw=1.0,
               label=f"$\\sigma/R$ = {analytic:.0f} Pa")
    ax.set_xlabel("$r/R$")
    ax.set_ylabel("mean pressure in shell  [Pa]")
    ax.set_title("(b) pressure profile: clipping the shell at $p=0$\n"
                 "removes the only place the jump can be held")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle("Free-surface pressure closure (Adami et al. 2012 §3.3) on the "
                 "Young–Laplace drop", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG.name}")

    # --- doc tables --------------------------------------------------------
    for passes in sorted({r["passes"] for r in rows}):
        print(f"\n<!-- passes={passes} -->")
        print("| dx [mm] | c0 [m/s] | N | steps | dp [Pa] | err vs sigma/R | "
              "err (final) | compression | vmax/vsigma | verdict |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted((q for q in rows if q["passes"] == passes),
                        key=lambda q: (q["dx_m"], q["sound_speed"])):
            verdict = "pass" if r["within_15pct"] and r["spurious_current_ok"] else (
                "unstable" if not r["spurious_current_ok"] else "outside 15 %")
            print(f"| {r['dx_m'] * 1000:.1f} | {r['sound_speed']:.0f} | "
                  f"{r['n_particles']} | {r['steps']} | {r['delta_p_Pa']:.1f} | "
                  f"{r['rel_error_pct']:+.1f} % | "
                  f"{r['rel_error_final_snapshot_pct']:+.1f} % | "
                  f"{r['compression_ratio']:.2f} | {r['vmax_over_vref']:.3f} | "
                  f"{verdict} |")


if __name__ == "__main__":
    main()
