"""Render ``docs/figures/pr309_round3_controls.png`` from the probe JSON.

Every number is read from ``docs/reports/pr309_round3_probe.json``; nothing in
the figure is typed by hand, so the picture cannot disagree with the tests that
read the same file.

    PYTHONPATH=$PWD/src python docs/reports/pr309_round3_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "docs" / "reports" / "pr309_round3_probe.json"
OUT = ROOT / "docs" / "figures" / "pr309_round3_controls.png"
GATE_COLOUR = "#c0392b"
OK_COLOUR = "#1f6f3f"


def main() -> int:
    p = json.loads(PROBE.read_text())
    a, b, c = (p["control_a_kernel_sum"], p["control_b_wall_pressure"],
               p["control_c_initial_density"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # ---- A: the gate is a lattice property, not a reduction property --------
    ax = axes[0]
    names = list(a["bias_by_variant"])
    vals = [abs(a["bias_by_variant"][k]) for k in names]
    ax.bar(range(len(names)), vals, color="#2b6cb0")
    ax.axhline(a["gate"], color=GATE_COLOUR, ls="--",
               label=f"gate {a['gate']:.0e}")
    ax.axhline(abs(a["old_pin"]), color="#7f8c8d", ls=":",
               label=f"PR #309 pin {abs(a['old_pin']):.2e}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("|kernel-sum bias|  C-1")
    ax.set_title(f"A. lattice bias vs reduction\nspread {a['spread_max_minus_min']:.2e}"
                 f" = {a['spread_over_gate']:.2e} x gate", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ---- B: old gate red only where it should be ---------------------------
    ax = axes[1]
    rows = b["rows"]
    labels = [f"mass x{r['mass_scale']}\n"
              f"({r['acc_rhs']:.3f} vs {r['acc_closed_form']:.3f})" for r in rows]
    errs = [r["relerr"] for r in rows]
    ax.bar(range(len(rows)), errs, color=["#2b6cb0", "#c0392b"])
    ax.axhline(b["old_gate"]["rel_tol"], color="#7f8c8d", ls="--",
               label=f"pre-#309 rtol {b['old_gate']['rel_tol']:.0e}")
    ax.axhline(b["shipped_gate"]["rel_tol"], color=OK_COLOUR, ls="-",
               label=f"shipped rtol {b['shipped_gate']['rel_tol']:.0e}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("relative error of a_y vs closed form")
    ax.set_title("B. wall pressure: two-term state reddens\n"
                 "the pre-#309 tolerance (margin "
                 f"{b['shipped_gate_margin']:.1f}x now)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ---- C: where the calibration default is load bearing ------------------
    ax = axes[2]
    bars = [
        ("square raw\n(clamp -> 0 Pa)", c["square"]["p_ratio_raw"]),
        ("hex raw\n(not clamped)", c["hex_same_h"]["p_ratio_raw"]),
        ("hex calibrated", max(c["square"]["p_ratio_calibrated"], 1e-12)),
    ]
    ax.bar(range(3), [max(v, 1e-12) for _, v in bars],
           color=["#2b6cb0", "#c0392b", OK_COLOUR])
    ax.axhline(c["gate"], color=GATE_COLOUR, ls="--", label=f"gate {c['gate']:.0e}")
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _ in bars], fontsize=8)
    ax.set_ylabel(r"$p_{false}/(\rho_0 c_0^2)$")
    ax.set_title("C. calibration default is load bearing\n"
                 f"on a positive-bias lattice ({c['hex_same_h']['abs_p_false_Pa']:.0f} Pa)",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.suptitle("PR #309 relaxations, round 3: the failing side of each gate "
                 f"(main {p['main'][:9]}, {p['arch']}, torch {p['torch']})",
                 fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
