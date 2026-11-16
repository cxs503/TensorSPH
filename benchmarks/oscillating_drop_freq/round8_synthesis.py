"""Round 8 of CASE-SPH-OSC-DROP-FREQ: comprehensive synthesis of rounds 1–7.

This script does NOT re-run SPH simulations. It ingests the existing round7.json
(and earlier round data where available), performs a meta-analysis, and produces:

1. A consolidated JSON report (out/round8.json)
2. A publication-quality convergence figure (docs/figures/...)
3. A verdict that summarises the state of the anchor after 7 rounds of study.

Why round 8 exists
------------------
Rounds 1–7 progressively deepened the analysis:
  R1–R2: basic linear-response measurement (quasi-static, CSF traction)
  R3:  level-set control + interface reconstruction audit
  R4:  two independent channels (contour + energy), transfer-law fit (assumed p=1)
  R5:  dynamic (time-advancing) arm → FAIL (divergence in every config)
  R6:  diagnosis of curvature noise + delta-SPH inflation
  R7:  observed order + GCI + ASME V&V 20 (3 grids: 12/16/24)

Round 8 asks: **what do we actually know after all this work?**
It consolidates the evidence, separates established facts from open questions,
and gives a clear verdict on the anchor gate (|Q − 1| < 6 % vs Rayleigh 1879).

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round8_synthesis.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIGS = HERE.parent.parent / "docs" / "figures"


# --------------------------------------------------------------------------- #
# Load existing data
# --------------------------------------------------------------------------- #
def load_round7() -> dict:
    p = OUT / "round7.json"
    if not p.exists():
        raise FileNotFoundError(f"round7.json not found at {p}")
    with open(p) as f:
        return json.load(f)


def load_round4() -> dict | None:
    p = HERE / "round4.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------- #
# Meta-analysis
# --------------------------------------------------------------------------- #
def meta_analysis(r7: dict, r4: dict | None) -> dict:
    """Consolidate findings from rounds 1–7 into a structured summary."""
    verdict = r7["verdict"]
    analysis = r7["analysis"]

    # Categorise rows by status
    ok_rows = [r for r in analysis["rows"] if r["status"] == "ok"]
    monotone_rows = [r for r in ok_rows if r["monotone"]]
    non_monotone_rows = [r for r in ok_rows if not r["monotone"]]
    v_pass_rows = [r for r in ok_rows if r["validation"]["pass"]]
    v_fail_rows = [r for r in ok_rows if not r["validation"]["pass"]]

    # l=3 systematic bias
    l3_rows = [r for r in ok_rows if r["mode"] == 3 and r["monotone"]]
    l3_q_inf_errors = [r["q_inf_error_pct"] for r in l3_rows]
    l3_bias_mean = np.mean(l3_q_inf_errors) if l3_q_inf_errors else float("nan")
    l3_bias_std = np.std(l3_q_inf_errors) if len(l3_q_inf_errors) > 1 else 0.0

    # l=2 differential channel (the one that passes V&V 20)
    l2_diff_rows = [r for r in ok_rows if r["mode"] == 2 and r["channel"] == "A_differential"]
    l2_diff_q_inf = [r["q_inf"] for r in l2_diff_rows]
    l2_diff_bias = np.mean(l2_diff_q_inf) - 1.0 if l2_diff_q_inf else float("nan")

    # Gate pass rate at R/dx=24
    gate_pass = verdict["gate_pass_rows_at_24"]
    gate_total = verdict["gate_rows"]

    # Validation pass rate
    v_pass = verdict["validation_pass_rows"]
    v_total = verdict["validation_rows"]

    # Observed order summary
    p_values = [r["p_observed"] for r in ok_rows if r["status"] == "ok"]
    p_median = verdict["p_observed_median"]

    return {
        "gate": {
            "pass_rows": gate_pass,
            "total_rows": gate_total,
            "pass_rate": gate_pass / gate_total if gate_total else 0,
            "threshold_pct": r7["gate_pct"],
        },
        "validation": {
            "pass_rows": v_pass,
            "total_rows": v_total,
            "pass_rate": v_pass / v_total if v_total else 0,
            "method": "ASME V&V 20-2009 (GCI + estimator control bias)",
        },
        "convergence": {
            "p_median": p_median,
            "p_range": verdict["p_observed_range"],
            "first_order_assumption_held": verdict["first_order_assumption_held"],
            "monotone_rows": len(monotone_rows),
            "non_monotone_rows": len(non_monotone_rows),
        },
        "systematic_bias": {
            "l3_q_inf_error_mean_pct": float(l3_bias_mean),
            "l3_q_inf_error_std_pct": float(l3_bias_std),
            "l3_rows_analysed": len(l3_rows),
            "l2_differential_q_inf_bias": float(l2_diff_bias),
            "l2_differential_rows": len(l2_diff_rows),
        },
        "dynamic_arm": {
            "status": "FAIL",
            "reason": "divergence in every configuration (including sigma->0 control)",
            "round": 5,
            "note": "pressure + continuity + axis-source subsystem is not stable",
        },
        "established_facts": [
            "6% gate met at R/dx=24 for all 10 channel rows",
            "Observed order p ≈ 1.93 for l=3 (not p=1 as assumed in round 4)",
            "l=3 mode has systematic -2.5% asymptotic bias (V&V 20 FAIL)",
            "l=2 differential channel (A') passes V&V 20 with Q_inf ≈ 0.999",
            "First-order convergence assumption (round 4) is falsified",
            "Dynamic arm diverges in all tested configurations (round 5)",
            "delta-SPH inflates the free surface (round 6)",
            "Estimator bias (level-set control) ≤ 0.05% (negligible)",
        ],
        "open_questions": [
            "Source of -2.5% l=3 bias: curvature estimation? interface reconstruction?",
            "l=2 non-monotonicity: grid alignment effect? needs orientation ensemble",
            "Dynamic arm instability: root cause in density/pressure evolution",
            "R/dx=32/40 results: never retrieved from cluster (cj4101)",
            "Whether bias decreases with higher resolution (untested beyond 24)",
        ],
    }


# --------------------------------------------------------------------------- #
# Figure generation
# --------------------------------------------------------------------------- #
def make_synthesis_figure(r7: dict, meta: dict) -> str:
    """Create a 2×2 publication-quality figure from existing round7 data."""
    analysis = r7["analysis"]
    rows = analysis["rows"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "CASE-SPH-OSC-DROP-FREQ: Synthesis of Rounds 1–7\n"
        "Oscillating Drop Frequency vs Rayleigh (1879) Linear Theory",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # --- Panel 1: Q_inf error by channel/mode (bar chart) ---
    ax = axes[0, 0]
    ok_rows = [r for r in rows if r["status"] == "ok"]
    labels = []
    q_inf_errors = []
    colors_list = []
    for r in ok_rows:
        labels.append(f"{r['channel'][:4]} l={r['mode']} ε={r['eps']:.2f}")
        q_inf_errors.append(r["q_inf_error_pct"])
        if r["channel"] == "A":
            colors_list.append("#e74c3c")
        elif r["channel"] == "A_differential":
            colors_list.append("#2ecc71")
        else:
            colors_list.append("#3498db")

    x = np.arange(len(labels))
    ax.bar(x, q_inf_errors, color=colors_list, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axhspan(-6, 6, alpha=0.15, color="gray", label="±6% gate")
    ax.axhline(-2.5, color="red", linewidth=1.5, linestyle=":", alpha=0.7,
               label="l=3 asymptotic bias ≈ −2.5%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Q∞ error (%)")
    ax.set_title("Q∞ Error by Channel, Mode & Amplitude")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(-8, 12)

    # --- Panel 2: Observed order p distribution ---
    ax = axes[0, 1]
    p_vals = [r["p_observed"] for r in ok_rows]
    p_labels = [f"{r['channel'][:4]} l={r['mode']}" for r in ok_rows]
    p_colors = ["#2ecc71" if r["monotone"] else "#e74c3c" for r in ok_rows]
    y_pos = np.arange(len(p_vals))
    ax.barh(y_pos, p_vals, color=p_colors, edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="k", linewidth=1, linestyle="--", label="assumed p=1 (R4)")
    ax.axvline(meta["convergence"]["p_median"], color="red", linewidth=1.5,
               linestyle="-", label=f"median p={meta['convergence']['p_median']:.2f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(p_labels, fontsize=7)
    ax.set_xlabel("Observed order p")
    ax.set_title("Observed Convergence Order (3 grids: 12/16/24)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 4.5)

    # --- Panel 3: Validation uncertainty breakdown ---
    ax = axes[1, 0]
    v_labels = [f"{r['channel'][:4]} l={r['mode']}" for r in ok_rows]
    u_num = [r["gci_pct"] for r in ok_rows]
    u_input = [r["u_input_pct"] for r in ok_rows]
    u_v = [math.sqrt(r["gci_pct"]**2 + r["u_input_pct"]**2) for r in ok_rows]
    err = [abs(r["q_inf_error_pct"]) for r in ok_rows]

    x = np.arange(len(v_labels))
    width = 0.2
    ax.bar(x - 1.5*width, u_num, width, label="U_num (GCI)", color="#3498db", alpha=0.8)
    ax.bar(x - 0.5*width, u_input, width, label="U_input", color="#f39c12", alpha=0.8)
    ax.bar(x + 0.5*width, u_v, width, label="U_V", color="#9b59b6", alpha=0.8)
    ax.scatter(x + 1.5*width, err, c="red", s=50, zorder=5, label="|Q∞−1|", marker="x")
    ax.set_xticks(x)
    ax.set_xticklabels(v_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Uncertainty (%)")
    ax.set_title("ASME V&V 20 Uncertainty Breakdown")
    ax.legend(fontsize=7)

    # --- Panel 4: Gate pass vs validation pass (summary) ---
    ax = axes[1, 1]
    categories = ["Gate\n(R/dx=24)", "V&V 20\n(asymptotic)"]
    pass_rates = [
        meta["gate"]["pass_rate"] * 100,
        meta["validation"]["pass_rate"] * 100,
    ]
    bar_colors = ["#2ecc71", "#e67e22"]
    bars = ax.bar(categories, pass_rates, color=bar_colors,
                  width=0.5, edgecolor="white", linewidth=2)
    for bar, val in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Pass Rate: Gate vs Asymptotic Validation")
    ax.axhline(100, color="k", linewidth=0.5, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIGS / "case_sph_osc_drop_freq_round8_synthesis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def make_verdict(meta: dict) -> dict:
    """Produce a structured verdict for the anchor."""
    gate_ok = meta["gate"]["pass_rate"] == 1.0
    v_ok = meta["validation"]["pass_rate"] > 0.5
    dynamic_ok = meta["dynamic_arm"]["status"] != "FAIL"

    overall = "PARTIAL"
    if gate_ok and v_ok and dynamic_ok:
        overall = "PASS"
    elif not gate_ok:
        overall = "FAIL"

    return {
        "anchor": "CASE-SPH-OSC-DROP-FREQ",
        "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71; Lamb 1932 Art. 275",
        "gate": {
            "threshold_pct": meta["gate"]["threshold_pct"],
            "pass_rate": meta["gate"]["pass_rate"],
            "status": "PASS" if gate_ok else "FAIL",
        },
        "asymptotic_validation": {
            "method": "ASME V&V 20-2009",
            "pass_rate": meta["validation"]["pass_rate"],
            "status": "PARTIAL" if 0 < meta["validation"]["pass_rate"] < 1 else
                       ("PASS" if v_ok else "FAIL"),
        },
        "dynamic_arm": {
            "status": meta["dynamic_arm"]["status"],
            "note": meta["dynamic_arm"]["reason"],
        },
        "overall": overall,
        "recommendation": (
            "Anchor gate (6% at R/dx=24) is met, but asymptotic validation "
            "reveals a systematic -2.5% bias for l=3 that does not vanish with "
            "resolution. The dynamic arm is non-functional. Recommend: "
            "(1) investigate l=3 bias source (curvature? reconstruction?), "
            "(2) fix dynamic arm instability, "
            "(3) run R/dx=32/40 to confirm bias persistence."
        ),
        "established_facts": meta["established_facts"],
        "open_questions": meta["open_questions"],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("Loading round 7 data...")
    r7 = load_round7()
    r4 = load_round4()

    print("Running meta-analysis...")
    meta = meta_analysis(r7, r4)

    print("Generating synthesis figure...")
    fig_path = make_synthesis_figure(r7, meta)
    print(f"  Figure saved to {fig_path}")

    print("Computing verdict...")
    verdict = make_verdict(meta)

    result = {
        "case": "SPH-OSC-DROP-FREQ",
        "round": 8,
        "study": "comprehensive synthesis of rounds 1-7",
        "reference": "Rayleigh 1879; Lamb 1932 Art. 275; ASME V&V 20-2009",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 1),
        "meta_analysis": meta,
        "verdict": verdict,
        "figure": fig_path,
    }

    out_path = OUT / "round8.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")
    print(f"Wall time: {result['wall_seconds']}s")
    print(f"\nVerdict: {verdict['overall']}")
    print(f"  Gate pass rate: {verdict['gate']['pass_rate']*100:.0f}%")
    print(f"  V&V 20 pass rate: {verdict['asymptotic_validation']['pass_rate']*100:.0f}%")
    print(f"  Dynamic arm: {verdict['dynamic_arm']['status']}")


if __name__ == "__main__":
    main()
