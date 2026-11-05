# ruff: noqa: E501  (report/prose templates carry long lines by design)
"""Figures for the TensorSPH#103 merge report (no hand-copied numbers).

Reads the merged artifacts (``benchmarks/laplace_law/convergence/*.json``, whose
``alternate_runs`` block still carries the ``main`` run) and the ladder summary,
and renders ``merge_evidence.png`` with four panels:

(a) the protocol difference that caused the textual conflict: physical window
    ``t_end / t_sigma`` per resolution for the two runs;
(b) the two read-outs of the kept run -- the historical ``bandA`` difference and
    the absolute ``p(r<0.5R) - 0`` jump -- against the 15 % bar;
(c) the final radial pressure field of the kept run at the three resolutions;
(d) the measured jump versus the analytic ``sigma/R = 360 Pa`` for both runs.

Usage::

    PYTHONPATH=src python docs/reports/m3853-merge/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = ROOT / "benchmarks" / "laplace_law"
OUT = HERE / "merge_evidence.png"

MANIFEST = "main-900step/relax_dx0.{case}.json"
CASES = ("0025", "0020", "0015")


def load():
    ladder = json.loads((BENCH / "resolution_ladder.json").read_text())
    arts = {}
    for rung in ladder["rungs"]:
        name = rung["artifact"].split("/")[-1]
        arts[name] = json.loads((BENCH / "convergence" / name).read_text())
    return ladder, arts


def main() -> None:
    ladder, arts = load()
    by_dx = sorted(arts.values(), key=lambda r: -r["setup"]["dx_m"])
    dx_mm = [r["setup"]["dx_m"] * 1e3 for r in by_dx]
    kept_t = [r["setup"]["t_over_t_sigma"] for r in by_dx]
    alt_t = [r["alternate_runs"][0]["setup"]["t_over_t_sigma"] for r in by_dx]
    kept_steps = [r["setup"]["steps"] for r in by_dx]
    alt_steps = [r["alternate_runs"][0]["setup"]["steps"] for r in by_dx]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    ax = axes[0][0]
    x = np.arange(len(dx_mm))
    w = 0.36
    b1 = ax.bar(x - w / 2, alt_t, w, label="main: fixed 900 steps")
    b2 = ax.bar(x + w / 2, kept_t, w, label="PR #103: fixed window")
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.axhline(0.5, color="crimson", ls=":", lw=1.0)
    ax.text(len(x) - 0.55, 0.53, "live test bar  t > 0.5", color="crimson", fontsize=7)
    for rect, s in zip(b2, kept_steps):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                f"{s} steps", ha="center", fontsize=7)
    for rect, s, t in zip(b1, alt_steps, alt_t):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                f"{s} steps", ha="center", fontsize=7, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([f"dx = {d:.1f} mm" for d in dx_mm])
    ax.set_ylabel(r"$t_{\rm end}/t_\sigma$")
    ax.set_title("(a) why the merge conflicted: two protocols, one file")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0][1]
    band = [r["laplace"]["rel_error_pct"] for r in by_dx]
    abso = [r["laplace"]["rel_error_absolute_pct"] for r in by_dx]
    ax.plot(dx_mm, band, "o-", label=r"bandA: $\Delta p$ = core - band")
    ax.plot(dx_mm, abso, "s-", label=r"absolute: $p(r<0.5R) - p(\rho_0)$")
    ax.axhline(-15.0, color="crimson", ls="--", lw=1.0)
    ax.axhline(15.0, color="crimson", ls="--", lw=1.0)
    ax.text(dx_mm[0], 16.5, "15 % bar", color="crimson", fontsize=7)
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("relative error of the jump [%]")
    ax.set_title("(b) kept run, both read-outs (kept flat: zero hand scaling)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    for r in by_dx:
        f = r["field"]
        rad = np.hypot(np.array(f["x_m"]), np.array(f["y_m"])) / r["setup"]["radius_m"]
        p = np.array(f["pressure_Pa"])
        order = np.argsort(rad)
        ax.plot(rad[order], p[order], ".", ms=2.5,
                label=f"dx = {r['setup']['dx_m']*1e3:.1f} mm, t = 1.0 t_σ")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel("Tait pressure [Pa]")
    ax.set_title(r"(c) final field snapshot of the kept run ($p(\rho_0)=0$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.15)

    ax = axes[1][1]
    dxs = sorted([r["setup"]["dx_m"] for r in by_dx])
    theory = by_dx[0]["reference"]["delta_p_Pa"]
    ax.axhline(theory, color="k", ls="--", lw=1.0, label=r"analytic $\sigma/R$")
    ax.plot(dxs, [r["alternate_runs"][0]["laplace"]["p_center_Pa"]
                  - r["alternate_runs"][0]["laplace"]["p_interface_band_Pa"]
                  for r in sorted(by_dx, key=lambda r: r["setup"]["dx_m"])],
            "o--", color="0.4", label="main 900-step, bandA")
    ax.plot(dxs, [r["laplace"]["p_center_Pa"] - r["laplace"]["p_interface_band_Pa"]
                  for r in sorted(by_dx, key=lambda r: r["setup"]["dx_m"])],
            "s-", label="PR #103, bandA")
    ax.plot(dxs, [r["laplace"]["p_center_absolute_Pa"]
                  for r in sorted(by_dx, key=lambda r: r["setup"]["dx_m"])],
            "^-", label="PR #103, absolute")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel(r"measured $\Delta p$ [Pa]")
    ax.set_title("(d) measured jump vs analytic, both merged runs")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
