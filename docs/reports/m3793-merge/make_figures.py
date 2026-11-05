#!/usr/bin/env python3
"""Figure for the m3793 merge report (all numbers injected, nothing hand-copied).

Panel (a): the capillary window ``t_end / t_sigma`` of every rung of the
laplace-law ladder, for the run kept by the merge (agent/m3163, fixed window)
and for the run dropped from the artefacts (main / m3312, fixed 900 steps).

Panel (b): the historical ``bandA`` read-out ``p(r<0.5R) - p(0.80-0.95R)``
against dx for both runs, plus the absolute read-out that agent/m3163 added.

Panel (c): spurious-current ratio against dx for both runs.

Usage::

    python docs/reports/m3793-merge/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2] / "benchmarks" / "laplace_law" / "convergence"
CASES = ("0015", "0020", "0025")


def load(directory: Path, case: str, suffix: str = "") -> dict:
    return json.loads((directory / f"relax_dx0.{case}{suffix}.json").read_text())


def series(directory: Path, suffix: str = "") -> dict[str, list[float]]:
    rows = [load(directory, case, suffix) for case in CASES]
    rows.sort(key=lambda r: r["setup"]["dx_m"])
    return {
        "dx_mm": [r["setup"]["dx_m"] * 1e3 for r in rows],
        "n": [r["setup"]["n_particles"] for r in rows],
        "steps": [r["setup"]["steps"] for r in rows],
        "t": [r["setup"]["t_over_t_sigma"] for r in rows],
        "banda": [r["laplace"]["rel_error_pct"] for r in rows],
        "absolute": [r["laplace"].get("rel_error_absolute_pct") for r in rows],
        "spurious": [r["spurious_current"]["ratio"] for r in rows],
    }


def main() -> int:
    kept = series(BENCH)
    dropped = series(HERE, ".main-m3312-900step")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))

    ax = axes[0]
    ax.plot(kept["dx_mm"], kept["t"], "o-", color="#3b7dd8",
            label=f"kept: agent/m3163 ({'/'.join(str(s) for s in kept['steps'])} steps)")
    ax.plot(dropped["dx_mm"], dropped["t"], "s--", color="#999999",
            label="dropped: main / m3312 (900 steps)")
    ax.axhline(0.5, color="#c0392b", ls="-.", lw=1.0, label="live pin: $t > 0.5\\,t_\\sigma$")
    ax.axhline(1.0, color="#3b7dd8", ls=":", lw=0.9)
    for x, t in zip(kept["dx_mm"], kept["t"], strict=True):
        ax.annotate(f"{t:.3f}", (x, t), textcoords="offset points", xytext=(2, 6), fontsize=8)
    for x, t in zip(dropped["dx_mm"], dropped["t"], strict=True):
        ax.annotate(f"{t:.3f}", (x, t), textcoords="offset points", xytext=(2, -12),
                    fontsize=8, color="#666666")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("$t_{end} / t_\\sigma$")
    ax.set_title("(a) relaxation window per rung", fontsize=10)
    ax.set_ylim(0.0, 1.55)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(kept["dx_mm"], kept["banda"], "o-", color="#3b7dd8",
            label="bandA, kept run (m3163)")
    ax.plot(dropped["dx_mm"], dropped["banda"], "s--", color="#999999",
            label="bandA, dropped run (main)")
    ax.plot(kept["dx_mm"], kept["absolute"], "^-", color="#e67e22",
            label="absolute, kept run")
    ax.axhline(-15.0, color="#c0392b", ls="-.", lw=1.0, label="15 % bar")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("error vs $\\sigma/R$ [%]")
    ax.set_title("(b) pressure-jump read-outs", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(kept["dx_mm"], [r * 100.0 for r in kept["spurious"]], "o-", color="#3b7dd8",
            label="kept run (m3163)")
    ax.plot(dropped["dx_mm"], [r * 100.0 for r in dropped["spurious"]], "s--",
            color="#999999", label="dropped run (main)")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("$v_{max}/v_{ref}$ [%]")
    ax.set_title("(c) spurious current", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    for axis in axes:
        axis.set_xticks(kept["dx_mm"])
        axis.set_xticklabels([f"{x:.1f}\nN={n}" for x, n in zip(kept["dx_mm"], kept["n"],
                                                              strict=True)], fontsize=8)

    fig.suptitle("m3793 merge: laplace-law ladder before/after the fixed-window resolution",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "merge_ladder.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
