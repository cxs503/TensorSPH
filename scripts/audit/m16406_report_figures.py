#!/usr/bin/env python3
"""m16406 — report figures (programmatic; every number read from the evidence JSONs).

  fig1_audit.png          tip-displacement history + rod-vs-elastica shape   (audit script)
  fig2_load_transfer.png  V1 hydrostatic load transfer, panel vs legacy pairwise, 3 lattices
  fig3_gate_force.png     fluid->gate force history vs the hydrostatic reference

Run after scripts/audit/m16406_case_fsi_board_audit.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "docs" / "board_audit" / "m16406"
IND = json.loads((AUD / "independent_checks.json").read_text())
RUNS = json.loads((AUD / "artifacts" / "runs.json").read_text())


def fig_load_transfer() -> Path:
    dxs = [0.0125, 0.01, 0.0075]
    panel = [IND["V1"][f"panel_dx{d}"]["rel_error"] * 100 for d in dxs]
    pair = [IND["V1"][f"pairwise_dx{d}"]["rel_error"] * 100 for d in dxs]
    ref = IND["V1_reference_N_per_m"]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(dxs))
    a = ax[0]
    a.axhspan(-3, 3, color="#d9ead3", alpha=0.7, label="+/-3 % band")
    a.plot(x, panel, "o-", color="#1f77b4", lw=1.6, label="panel transfer (delivered fix)")
    a.plot(x, pair, "s--", color="#b3001b", lw=1.4, label="legacy Adami pairwise")
    for i, (p, q) in enumerate(zip(panel, pair)):
        a.annotate(f"{p:+.3f} %", (i, p), textcoords="offset points", xytext=(0, 8),
                   ha="center", fontsize=7, color="#1f77b4")
        a.annotate(f"{q:+.1f} %", (i, q), textcoords="offset points", xytext=(0, -13),
                   ha="center", fontsize=7, color="#b3001b")
    a.set_xticks(x, [f"dx={d*1000:g} mm" for d in dxs])
    a.set_ylabel("total horizontal load error [%]")
    a.set_title(f"V1 load transfer vs 0.5 rho g h$_s^2$ = {ref:.2f} N/m")
    a.grid(alpha=0.3)
    a.legend(fontsize=7, loc="lower right")
    a.set_ylim(-36, 8)

    b = ax[1]
    got_panel = [IND["V1"][f"panel_dx{d}"]["load_sph"] for d in dxs]
    got_pair = [IND["V1"][f"pairwise_dx{d}"]["load_sph"] for d in dxs]
    b.axhline(ref, color="k", lw=1.2, label="reference")
    b.plot(x, got_panel, "o-", color="#1f77b4", label="panel")
    b.plot(x, got_pair, "s--", color="#b3001b", label="legacy pairwise")
    b.set_xticks(x, [f"dx={d*1000:g} mm" for d in dxs])
    b.set_ylabel(r"gate load [N/m]")
    b.set_title("delivered vs reference load")
    b.grid(alpha=0.3)
    b.legend(fontsize=7)
    fig.tight_layout()
    p = AUD / "fig2_load_transfer.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    return p


def fig_gate_force() -> Path:
    rk = list(RUNS)[0]
    h = RUNS[rk]["history"]
    t = np.asarray(h["t"])
    f = np.asarray(h["gate_force_x"])
    tip = np.asarray(h["tip_x"]) * 1e3
    ref = IND["run_artifacts"]["hydrostatic_reference_N_per_m"]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    a = ax[0]
    a.axhline(ref, color="k", lw=1.1, label=f"hydrostatic reference {ref:.1f} N/m")
    a.plot(t, f, lw=0.9, color="#6a3d9a")
    a.set_yscale("symlog", linthresh=1e3)
    a.set_xlabel("t [s]")
    a.set_ylabel("fluid -> gate horizontal force [N/m]")
    a.set_title(f"t=0 force = {f[0]:.3f} N/m;  peak = {f.max():.0f} N/m = {f.max()/ref:.0f}x reference")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)

    b = ax[1]
    b.plot(t, tip, lw=0.9, color="#b3001b", label="tip x (delivered run)")
    b.axhspan(-360, 360, color="#d9ead3", alpha=0.6, label="|tip| <= gate height (360 mm)")
    b.set_xlabel("t [s]")
    b.set_ylabel("tip displacement [mm]")
    b.set_title(f"first violation t={IND['run_artifacts']['first_abs_tip_over_H_s']:.4f} s, "
                f"{IND['run_artifacts']['n_samples_beyond_H']}/"
                f"{len(t)} samples beyond")
    b.grid(alpha=0.3)
    b.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    p = AUD / "fig3_gate_force.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for p in (fig_load_transfer(), fig_gate_force()):
        print(p, p.stat().st_size)