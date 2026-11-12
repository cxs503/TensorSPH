"""Figures for the settling study, generated from ``settling_matrix.json`` only.

No number in a figure comes from anywhere else: the traces are the ones the
runner recorded, the band and the Adami limit are the matrix's own rules.  The
figure exists so that a reader can see the discriminator (`v_max(t)` decay)
rather than trust a table.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/figures_settling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "settling_matrix.json"
PNG = HERE / "laplace_pressure_settling.png"


def main() -> None:
    m = json.loads(MATRIX.read_text())
    rules = m["rules"]
    rows = m["readouts"]
    dp = m["reference"]["analytic_dp_Pa"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # (a) the discriminator: does the spurious current decay over the window?
    ax = axes[0]
    for r in rows:
        t = [p[0] for p in r["vmax_trace"]]
        v = [p[1] for p in r["vmax_trace"]]
        c = r["config"]
        ax.plot(t, v, label=(f"dx={c['dx_m'] * 1e3:.1f} mm, c0={c['sound_speed']:g} m/s, "
                             f"{'closure' if c['closure'] else 'no closure'}"))
    ax.axhline(rules["spurious_current_limit"], color="k", ls="--", lw=1.0,
               label="Adami et al. (2012) limit")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t / t_\sigma$")
    ax.set_ylabel(r"$v_{max}/v_{ref}$")
    ax.set_title("(a) spurious current: transient or plateau?")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")

    # (b) the read-out against the band
    ax = axes[1]
    labels = [f"{r['config']['dx_m'] * 1e3:.1f}mm\nc0={r['config']['sound_speed']:g}\n"
              f"{'on' if r['config']['closure'] else 'off'}" for r in rows]
    for i, r in enumerate(rows):
        ok = r["within_15pct"] and r["admissible"]
        ax.errorbar(i, r["rel_error_pct"], fmt="o", ms=6,
                    color=("tab:green" if ok else ("tab:orange" if r["within_15pct"]
                                                   else "tab:red")))
        ax.annotate(f"{r['rel_error_pct']:+.1f}%\nv={r['vmax_over_vref_late']:.2f}",
                    (i, r["rel_error_pct"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=6.5)
    ax.axhspan(-rules["band_pct"], rules["band_pct"], color="tab:green", alpha=0.10)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(r"$\langle p_{core}\rangle$ error [%]")
    ax.set_title("(b) settled read-out vs the $\\pm$15 % band (green = admissible)")
    ax.grid(alpha=0.3, axis="y")

    # (c) the settled pressure trace against the analytic jump
    ax = axes[2]
    for r in rows:
        if not r["config"]["closure"]:
            continue
        t = [p[0] for p in r["p_core_trace"]]
        p = [p[1] for p in r["p_core_trace"]]
        ax.plot(t, p, label=f"dx={r['config']['dx_m'] * 1e3:.1f} mm, "
                            f"c0={r['config']['sound_speed']:g} m/s")
    ax.axhline(dp, color="k", ls="--", lw=1.0, label=r"$\sigma/R$")
    ax.set_xlabel(r"$t / t_\sigma$")
    ax.set_ylabel(r"$p_{core}$ [Pa]")
    ax.set_title("(c) core pressure of the closed-surface drop")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    fig.suptitle("Young-Laplace read-out with the free-surface pressure closure: "
                 "long-window settling study", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PNG, dpi=150)
    print("wrote", PNG)


if __name__ == "__main__":
    main()