#!/usr/bin/env python3
"""Figures for the m16239 cross-check audit (CASE-SPH-STOKES-2ND-PROBLEM).

fig1  amplitude decay of the delivered SPH run vs the two analytic references
      (half-space exp(-y/delta) and the finite-depth channel solution), y/delta
      on a log amplitude axis.
fig2  error-vs-resolution: the L2 error of the shipped rung (delta/dx=8) and of
      the rung that the assembler dropped (delta/dx=12, still on disk in /tmp),
      against the 5% criterion of case-pool SPH-STOKES-2ND-PROBLEM.

Only matplotlib + stdlib; the numbers come from the delivered artifact, never
from a hand-copied table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
SRC = Path("/root/tp-agent/work/m16239/benchmarks/stokes_second/out")
FIG = HERE / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    ladder = json.loads((SRC / "results_ladder.json").read_text())
    run = ladder["runs"][0]
    delta, u0 = run["delta"], run["u0"]
    rows = run["rows"]
    y = [r["y"] / delta for r in rows]
    amp = [r["amplitude"] / u0 for r in rows]
    ref = [math.exp(-v) for v in y]
    fdep = [r["amplitude_finite_depth"] / u0 for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].semilogy(y, amp, "o", ms=3.2, label=f"SPH, $\\delta/\\Delta x$={run['resolution_delta_over_dx']} (m16239)")
    ax[0].semilogy(y, ref, "k-", lw=1.6, label=r"half-space $e^{-y/\delta}$ (Landau sec.24)")
    ax[0].semilogy(y, fdep, "k--", lw=1.1, label="finite depth, $H=6\\delta$ (exact)")
    ax[0].axvline(1.0, color="0.6", ls=":", lw=1.0)
    ax[0].text(1.03, 0.55, r"$y=\delta$", color="0.4", fontsize=8)
    ax[0].set_xlabel(r"$y/\delta$")
    ax[0].set_ylabel(r"$A(y)/U_0$")
    ax[0].set_title("velocity-amplitude decay through the Stokes layer")
    ax[0].set_xlim(0, 3.0)
    ax[0].set_ylim(1e-3, 1.4)
    ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].grid(alpha=0.3)

    # error vs resolution: shipped rung + the probe the assembler dropped
    pts = [(8, run["metrics"]["amplitude_l2_half_space_pct"], "shipped artifact")]
    p12 = Path("/tmp/p12/results_probe.json")
    if p12.exists():
        r12 = json.loads(p12.read_text())["runs"][0]
        pts.append((12, r12["metrics"]["amplitude_l2_half_space_pct"],
                    "probe on disk, dropped by assembler"))
    res = [p[0] for p in pts]
    err = [p[1] for p in pts]
    ax[1].loglog(res, err, "o-", color="C0", label="SPH error (amplitude L2, Stokes layer)")
    ax[1].axhline(5.0, color="C3", ls="--", lw=1.6, label="case-pool criterion 5%")
    for r, e, lbl in pts:
        ax[1].annotate(f"{e:.4f}%\n{r}: {lbl}", (r, e), textcoords="offset points",
                       xytext=(6, 8), fontsize=7)
    ax[1].set_xlabel(r"resolution $\delta/\Delta x$")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("error vs resolution (2 rungs that exist on disk)")
    ax[1].set_xticks(res)
    ax[1].set_xticklabels([str(r) for r in res])
    ax[1].set_xlim(6.5, 15)
    ax[1].set_ylim(1e-2, 10)
    ax[1].legend(fontsize=7.5)
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out1 = FIG / "m16324_stokes2_amplitude_decay.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print("wrote", out1)

    # numbers used in the report (printed, not hand-copied)
    layer = [r for r in rows if r["y"] / delta <= 1.0 + 1e-12]
    print("rows", len(rows), "in layer", len(layer))
    print("first row A/U0 %.6f vs exp(-y/d) %.6f"
          % (rows[0]["amplitude_over_U0"], math.exp(-rows[0]["y"] / delta)))
    print("L2 shipped %.4f%%  " % pts[0][1], ("L2 dx12 %.4f%%" % pts[1][1]) if len(pts) > 1 else "")
    sens = json.loads((SRC / "results_sensitivity.json").read_text())["sensitivity"]
    for e in sens:
        m = e["metrics"]
        print("control %-28s amp_max/U0=%.5f l2=%.4f%% h/dx=%s"
              % (e["label"], m["amplitude_max_over_U0"], m["amplitude_l2_half_space_pct"],
                 round(e["h"] / e["dx"], 2)))
    return out1


if __name__ == "__main__":
    raise SystemExit(main())