#!/usr/bin/env python3
"""m16318 board-audit figure: RT-MP growth-rate evidence (zero hand-typed numbers).

Reads benchmarks/rayleigh_taylor/runs/*.json (the m16244 artifacts) and draws
  (a) sigma_num(dx) vs sqrt(A g k) with the <10 % tolerance band,
  (b) raw interface amplitude history with the fitted growth window.
Writes docs/benchmarks/rayleigh-taylor-mp-audit/fig1_growth_audit.png
and prints the numbers that the audit report quotes.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "benchmarks" / "rayleigh_taylor" / "runs"
OUTDIR = REPO / "docs" / "benchmarks" / "rayleigh-taylor-mp-audit"


def load(name):
    return json.loads((RUNS / name).read_text())


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ref = load("reference.json")
    table = load("table.json")
    sig_ref = ref["sigma_ideal_Agk"]

    rows = [r for r in table
            if isinstance(r.get("sigma"), float) and math.isfinite(r["sigma"])]
    rows.sort(key=lambda r: r["dx"])
    dx = [r["dx"] for r in rows]
    sig = [r["sigma"] for r in rows]
    vel = [r["vel_ratio"] * sig_ref for r in rows]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax.axhspan(0.9 * sig_ref, 1.1 * sig_ref, color="#c8e6c9", alpha=0.7,
               label=r"$\pm10\%$ tolerance band")
    ax.axhline(sig_ref, color="k", lw=1.2,
               label=rf"Taylor 1950 $\sqrt{{Agk}}$={sig_ref:.4f} s$^{{-1}}$")
    ax.plot(dx, sig, "o-", color="#c62828", label="centroid gauge (reported)")
    ax.plot(dx, vel, "s--", color="#1565c0", label="velocity gauge (reported)")
    for x, y in zip(dx, sig):
        ax.annotate(f"{100 * (y / sig_ref - 1):+.1f}%", (x, y),
                    textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing dx [m] (finer →)")
    ax.set_ylabel(r"growth rate $\sigma$ [s$^{-1}$]")
    ax.set_title("(a) RT linear growth rate vs Taylor anchor (m16244 artifacts)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    for name, color, lab in (
            ("dx0.0125_nu0.001_a0.01.json", "#c62828", "dx=0.0125 m"),
            ("dx0.02_nu0.001_a0.01.json", "#1565c0", "dx=0.02 m")):
        d = load(name)
        t = d["series"]["t"]
        a = d["series"]["amplitude"]
        fit = d["fit"]
        xs = [ti for ti, ai in zip(t, a) if ai > 0]
        ys = [math.log(ai) for ti, ai in zip(t, a) if ai > 0]
        bx.plot(xs, ys, lw=1.0, color=color, label=f"{lab}, fit σ={fit['sigma']:.3f}")
        bx.axvspan(fit["t_start"], fit["t_end"], color=color, alpha=0.10)
        t0, t1 = fit["t_start"], fit["t_end"]
        bx.plot([t0, t1],
                [math.log(fit["a_start"]) + 0.0,
                 math.log(fit["a_end"]) + 0.0], "k--", lw=1.0) if False else None
        bx.plot([t0, t1],
                [math.log(fit["a_start"]),
                 math.log(fit["a_start"]) + fit["sigma"] * (t1 - t0)],
                "--", color=color, lw=1.2)
    bx.axhline(math.log(0.1), color="grey", ls=":", lw=1,
               label="upper window 0.1λ")
    bx.axhline(math.log(0.02), color="grey", ls=":", lw=1,
               label="lower window 0.02λ")
    bx.set_xlabel("t [s]")
    bx.set_ylabel(r"$\ln \eta$ [m]")
    bx.set_title("(b) interface amplitude history + fitted window")
    bx.grid(alpha=0.3)
    bx.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = OUTDIR / "fig1_growth_audit.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    for r in rows:
        print(f"dx={r['dx']:<7} sigma={r['sigma']:.4f} "
              f"err={100 * (r['sigma'] / sig_ref - 1):+.2f}% vel_err="
              f"{100 * (r['vel_ratio'] - 1):+.2f}%")


if __name__ == "__main__":
    main()