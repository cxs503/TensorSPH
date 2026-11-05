"""Figure source for docs/ci_fix_pr237.md (PR #237 CI triage).

Two panels, both built from committed numbers, no hand-typed values:

(a) the measurement the flaky test re-runs: log-linear amplitude decay of the
    projected shear-wave mode for the three cases of
    ``benchmarks/viscosity_shear_wave/report.json``, against the analytic rate;
(b) the reproduction gate: relative drift between the re-run and the committed
    ``nu_effective`` (log scale) for the deviation actually observed in CI, the
    deployed tolerance, the 15 % acceptance band, and the two injected "rot"
    scenarios used to prove the gate still has teeth.

Usage: python artifacts/pr237/make_figure.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPORT = REPO / "benchmarks" / "viscosity_shear_wave" / "report.json"

TOL = 1e-3          # REPRODUCTION_TOL in tests/test_viscosity_shear_wave.py
OBSERVED = 3.36e-5  # |reno - committed|/committed seen on the failing CI run
ROT = {"rot 1 %\n(injected)": 9.93e-3, "rot 10 %\n(injected)": 9.09e-2}


def main() -> int:
    report = json.loads(REPORT.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) amplitude decay ---------------------------------------------------
    ax = axes[0]
    colors = {"nominal": "C0", "coarse": "C1", "zero_viscosity": "C2"}
    for name, series in report["series"].items():
        case = report["cases"][name]
        t = series["t"]
        a = series["amplitude"]
        ax.semilogy(t, a, "o", ms=3, color=colors[name], label=f"{name} (solver)")
        if name != "zero_viscosity":
            a0 = case["amplitude_initial"]
            rate = case["decay_rate_analytic"]
            ax.semilogy(t, [a0 * math.exp(-rate * ti) for ti in t], "-", lw=1.0,
                        color=colors[name],
                        label=rf"{name}: $\nu_{{\rm eff}}/\nu$ = "
                              f"{case['nu_ratio']:.3f}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("mode amplitude A(t) = <u_x, sin(ky)> / <sin, sin>  [m/s]")
    ax.set_title("(a) shear-wave decay, committed report.json")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (b) reproduction gate -------------------------------------------------
    ax = axes[1]
    labels = ["observed CI\nre-run", "REPRODUCTION_TOL", "acceptance band\n(15 %)"]
    values = [OBSERVED, TOL, 0.15]
    bar_colors = ["C0", "C3", "0.7"]
    bars = ax.bar(labels, values, color=bar_colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.1e}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=8)
    x0 = len(labels) - 0.5
    for i, (name, value) in enumerate(ROT.items()):
        ax.bar(x0 + i, value, color="C1", width=0.55)
        ax.annotate(f"{value:.1e}\nFAILS", (x0 + i, value), ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(list(range(len(labels))) + [x0 + i for i in range(len(ROT))])
    ax.set_xticklabels(labels + list(ROT), fontsize=7)
    ax.axhline(TOL, color="C3", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylabel(r"$|nu_{\rm eff}^{\rm re-run}/\nu_{\rm eff}^{\rm committed} - 1|$")
    ax.set_title("(b) reproduction gate: what passes, what still fails")
    ax.grid(alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    out = HERE / "fig_reproduction_gate.png"
    fig.savefig(out, dpi=100)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())