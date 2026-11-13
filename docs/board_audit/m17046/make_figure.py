"""Figures for board audit m17046 (REPRO-SPH-MONAGHAN1994, mission m17020).

Reads only committed artefacts (trajectories, reference tables, summary.json and —
if present — m17020's error_budget.json) and writes a 3-panel PNG.  No simulation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "benchmarks" / "repro_monaghan_1994"
RES = CASE / "results"
DAM = ROOT / "benchmarks" / "dam_break"
OUT = ROOT / "docs" / "board_audit" / "m17046"
WINDOW = (0.5, 3.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cross = json.loads((DAM / "martin_moyce_1952_crosscheck.json").read_text())
    bundled = json.loads((DAM / "martin_moyce_1952.json").read_text())
    c_tau = [r["tau"] for r in cross["data"]]
    c_za = [r["X_over_a"] - 1.0 for r in cross["data"]]
    rows = bundled.get("data") or bundled.get("points") or bundled
    b_tau = [r["tau"] for r in rows]
    b_za = [r.get("Z_over_a", r.get("X_over_a")) for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) trajectories vs the two transcriptions
    for case, colour in (("dx0.04_c025", "0.65"), ("dx0.02_c025", "0.4"), ("dx0.01_c025", "C0")):
        traj = json.loads((RES / f"trajectory_{case}.json").read_text())
        ax[0].plot(traj["tau"], traj["za"], colour, lw=1.6,
                   label=f"SPH dx={traj['dx']} m ({traj['particles']} p)")
    ax[0].plot(c_tau, c_za, "ks", ms=4, label="Martin & Moyce 1952 (crosscheck digitisation)")
    ax[0].plot(b_tau, b_za, "r^", ms=4, mfc="none",
               label="bundled digitisation (graded WITHHELD)")
    ax[0].axvspan(*WINDOW, color="C0", alpha=0.06, label="pre-registered window")
    ax[0].set_xlabel(r"$\tau=t\sqrt{2g/a}$")
    ax[0].set_ylabel("Z/a")
    ax[0].set_title("(a) dam-break front, 3 resolutions vs 2 transcriptions")
    ax[0].legend(fontsize=6.5, loc="upper left")
    ax[0].grid(alpha=0.3)

    # (b) residual vs tau, per resolution (bias crosses zero -> non-monotone |MAE|)
    for case in ("dx0.04_c025", "dx0.02_c025", "dx0.01_c025"):
        traj = json.loads((RES / f"trajectory_{case}.json").read_text())
        sel = [(t, z) for t, z in zip(traj["tau"], traj["za"]) if WINDOW[0] <= t <= WINDOW[1]]
        taus = [t for t, _ in sel]
        # linear interpolation of the crosscheck table at the trajectory samples
        def ref_at(t):
            if t <= c_tau[0]:
                return c_za[0]
            if t >= c_tau[-1]:
                return c_za[-1]
            for i in range(1, len(c_tau)):
                if c_tau[i] >= t:
                    w = (t - c_tau[i - 1]) / (c_tau[i] - c_tau[i - 1])
                    return c_za[i - 1] + w * (c_za[i] - c_za[i - 1])
            return c_za[-1]
        deltas = [z - ref_at(t) for t, z in sel]
        ax[1].plot(taus, deltas, lw=1.5, label=f"dx={traj['dx']} m")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel(r"$\tau$")
    ax[1].set_ylabel(r"$Z/a$ (SPH $-$ crosscheck)")
    ax[1].set_title("(b) signed residual: bias crosses zero $\\Rightarrow$ |MAE| non-monotone")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    # (c) m17020's "error budget": components vs the total they claim to split
    summary = json.loads((RES / "summary.json").read_text())
    total = summary["comparisons"]["dx0.01_c025"]["crosscheck"]["mae_over_amplitude"]
    labels = ["total MAE/amplitude\n(graded)"]
    values = [total]
    colours = ["C0"]
    ebf = RES / "error_budget.json"
    if ebf.exists():
        comp = json.loads(ebf.read_text())["error_budget"]["components"]
        labels += ["reference pair\nspread", "deviation from\nRitter (2 tau)", "max compression\n(rho/rho0-1)"]
        values += [comp["reference_uncertainty"]["spread_over_amplitude"],
                   comp["ritter_deviation"]["mae_over_amplitude"],
                   comp["compressibility"]["max_rho_over_rho0_minus_1"]]
        colours += ["C3", "C1", "C2"]
    bars = ax[2].bar(range(len(values)), values, color=colours)
    for rect, v in zip(bars, values):
        ax[2].text(rect.get_x() + rect.get_width() / 2, v * 1.15, f"{100*v:.2f}%",
                   ha="center", fontsize=8)
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(len(labels)))
    ax[2].set_xticklabels(labels, fontsize=7)
    ax[2].set_ylabel("fraction of reference amplitude (log)")
    if len(values) > 1:
        summed = sum(values[1:])
        ax[2].set_title(f"(c) m17020 'budget': pieces sum to {summed/total:.1f}x the total\n"
                        f"= {100*summed:.1f}% vs {100*total:.2f}% $\\Rightarrow$ not a decomposition")
    else:
        ax[2].set_title("(c) no error_budget.json on this tree")

    fig.suptitle("REPRO-SPH-MONAGHAN1994 board audit m17046 — recomputed from committed artefacts",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "fig1_audit_m17046.png"
    fig.savefig(path, dpi=150)
    print("written", path)


if __name__ == "__main__":
    main()