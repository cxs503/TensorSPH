#!/usr/bin/env python
"""Independent re-derivation of the channel wall-shear gate (anchor: KPI-QC-RECHECK).

Audits the merged deliverable of mission m16581
(``docs/verification_poiseuille_wall_shear_library_gate.md``, PR #573) by

1. recomputing the momentum-budget wall shear **from the committed artefacts
   with this file's own least squares** (no import of
   :mod:`tensorsph.wall_shear` for the primary numbers -- an audit that reuses
   the audited code cannot detect its errors),
2. re-deriving the same reading over a ladder of drift windows, which turns a
   single quoted number into the estimator's own uncertainty band,
3. pinning the **window-convention collision** between the new module
   (``fraction`` = window *length*) and the legacy script
   ``benchmarks/poiseuille/wall_shear_estimators.py`` (``fraction`` = window
   *start*), including the legacy script's comment/code disagreement.

Sources (read only): ``benchmarks/poiseuille/out/results.json`` (WCSPH ladder,
merged before m16581) and ``runs/isph_channel/m16654/results.json`` (ISPH
ladder re-run by this mission).  Nothing here advances an SPH state.

Usage::

    PYTHONPATH=src python benchmarks/wall_shear_gate_audit/run.py
    PYTHONPATH=src python benchmarks/wall_shear_gate_audit/make_doc.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

WCSPH_ARTEFACT = REPO / "benchmarks" / "poiseuille" / "out" / "results.json"
ISPH_ARTEFACT = REPO / "runs" / "isph_channel" / "m16654" / "results.json"

#: Drift windows as **window length / run length** (``t_lo = (1-f) t_end``).
WINDOW_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

#: The windows the m16581 gate actually relies on (post-transient only).
GATE_FRACTIONS = (0.2, 0.3, 0.4, 0.5)

#: The single window the merged module declares as its default.
DECLARED_FRACTION = 0.4

GATE_LIMIT_PCT = 3.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def drift_slope(trace, t_end: float, fraction: float, key: str = "u_mean"):
    """Least squares slope over ``[(1-f) t_end, t_end]`` -- audit's own code."""
    t = np.array([float(p["t"]) for p in trace], dtype=float)
    u = np.array([float(p[key]) for p in trace], dtype=float)
    keep = (t >= (1.0 - fraction) * t_end - 1e-12) & (t <= t_end + 1e-12)
    tw, uw = t[keep], u[keep]
    n = int(tw.size)
    if n < 2:
        return {"available": False, "n_samples": n, "slope": float("nan")}
    sxx = float(np.sum((tw - tw.mean()) ** 2))
    slope = float(np.sum((tw - tw.mean()) * (uw - uw.mean())) / sxx)
    resid = uw - (uw.mean() + slope * (tw - tw.mean()))
    dof = n - 2
    stderr = (math.sqrt(float(np.sum(resid ** 2)) / dof / sxx) if dof > 0 else float("nan"))
    return {
        "available": True,
        "n_samples": n,
        "slope": slope,
        "stderr": stderr,
        "window": [float(tw[0]), float(tw[-1])],
        "window_s": float(tw[-1] - tw[0]),
    }


def budget(trace, t_end, height, body_force, rho0, fraction):
    """``tau_w = (1/2) rho H (g - d<u>/dt)`` and its relative error (audit code)."""
    d = drift_slope(trace, t_end, fraction)
    required = 0.5 * rho0 * body_force * height
    if not d["available"]:
        return {"available": False, "n_samples": d["n_samples"],
                "tau_wall": float("nan"), "rel_error_pct": float("nan")}
    tau = 0.5 * rho0 * height * (body_force - d["slope"])
    return {
        "available": True,
        "n_samples": d["n_samples"],
        "drift": d["slope"],
        "drift_stderr": d["stderr"],
        "tau_wall": tau,
        "tau_wall_required": required,
        "rel_error_pct": 100.0 * (tau - required) / required,
    }


def window_ladder(trace, t_end, height, body_force, rho0,
                  fractions=WINDOW_FRACTIONS):
    per = {}
    errors = []
    for f in fractions:
        b = budget(trace, t_end, height, body_force, rho0, f)
        per[f"{f:g}"] = b
        if b["available"] and math.isfinite(b["rel_error_pct"]):
            errors.append(b["rel_error_pct"])
    gate = [per[f"{f:g}"]["rel_error_pct"] for f in GATE_FRACTIONS
            if per[f"{f:g}"]["available"]]
    band = [min(gate), max(gate)] if gate else [float("nan")] * 2
    worst = max((abs(v) for v in gate), default=float("nan"))
    return {
        "per_fraction": per,
        "post_transient_fractions": list(GATE_FRACTIONS),
        "rel_error_pct_band": band,
        "worst_abs_rel_error_pct": worst,
        "spread_pct_points": (band[1] - band[0]) if gate else float("nan"),
        # consumer-facing honesty: the KPI says >15 % is unacceptable, the gate
        # is 3 %; report whether *every* post-transient window clears both.
        "within_gate_all_windows": bool(worst == worst and worst <= GATE_LIMIT_PCT),
        "full_run_rel_error_pct": per["1"]["rel_error_pct"] if per["1"]["available"] else float("nan"),
    }


def audited_kpi(cases):
    """Cross-check the m16581 §6 table against the published post-transient band."""
    rows = []
    for tag, ladder, published in cases:
        band = ladder["rel_error_pct_band"]
        pub = published
        worst = ladder["worst_abs_rel_error_pct"]
        rows.append({
            "ladder": tag,
            "declared_window_pct": ladder["per_fraction"][f"{DECLARED_FRACTION:g}"]["rel_error_pct"],
            "band_pct": band,
            "spread_pct_points": ladder["spread_pct_points"],
            "published_pct": pub,
            "published_matches_declared_window": (
                pub is not None and math.isfinite(pub)
                and abs(pub - ladder["per_fraction"][f"{DECLARED_FRACTION:g}"]["rel_error_pct"]) < 0.02),
            "worst_abs_pct": worst,
            "gate_limit_pct": GATE_LIMIT_PCT,
            "verdict": "PASS" if worst == worst and worst <= GATE_LIMIT_PCT else "FAIL",
            "verdict_is_band_based": True,
        })
    return rows


def convention_check(trace, t_end):
    """Legacy start-fraction vs module length-fraction, on one trace."""
    legacy = drift_slope(trace, t_end, fraction=0.6)          # legacy 0.4 == last 60 %
    module = drift_slope(trace, t_end, fraction=0.4)          # module 0.4  == last 40 %
    same_window = drift_slope(trace, t_end, fraction=0.6)
    return {
        "legacy_start_fraction_0.4_window": legacy["window"],
        "module_length_fraction_0.4_window": module["window"],
        "legacy_rate": legacy["slope"],
        "module_rate": module["slope"],
        "legacy_0.4_equals_module_0.6": bool(
            legacy["available"] and same_window["available"]
            and abs(legacy["slope"] - same_window["slope"]) < 1e-9),
        "same_number_different_window": bool(
            abs(legacy["slope"] - module["slope"]) > 1e-6),
        "rate_difference_pct_of_g": 100.0 * (legacy["slope"] - module["slope"]) / 9.81,
    }


def wcsph(data):
    H = float(data["height"])
    g = float(data["body_force"])
    rho0 = 1000.0
    out = []
    for case in data["results"]:
        ladder = window_ladder(case["trace"], float(data["t_end"]), H, g, rho0)
        ladder["resolution_H_over_dx"] = int(case["resolution_H_over_dx"])
        ladder["published_wall_shear_ratio"] = case["wall_shear_ratio"]
        ladder["published_tau_pa"] = case["wall_shear_measured"]
        ladder["n_trace_samples"] = len(case["trace"])
        out.append(ladder)
    control = data["free_slip_control"]
    ctrl = window_ladder(control["trace"], float(data["t_end"]), H, g, rho0)
    return {"cases": out, "free_slip_control": ctrl,
            "convention": convention_check(data["results"][0]["trace"], float(data["t_end"]))}


def isph(data):
    cfg = data["config"]
    out = []
    for case in data["results"]:
        ladder = window_ladder(case["trace"], float(cfg["t_end"]),
                              float(cfg["height"]), float(cfg["body_force"]),
                              float(cfg["rho0"]))
        ladder["resolution_H_over_dx"] = int(case["resolution_H_over_dx"])
        recorded = case["gates"]["wall_shear_budget"]
        ladder["recorded_gate_rel_error_pct"] = recorded["rel_error_pct"]
        ladder["recorded_gate_band"] = recorded.get("rel_error_pct_band")
        ladder["recorded_gate_pass"] = recorded["pass"]
        ladder["near_wall_diagnostic_pct"] = case["gates"]["wall_shear_near_wall"]["rel_error_pct"]
        ladder["centre_velocity_pct"] = case["gates"]["centre_velocity"]["rel_error_pct"]
        ladder["n_trace_samples"] = len(case["trace"])
        out.append(ladder)
    return {"cases": out}


def figure(report, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fractions = [f for f in WINDOW_FRACTIONS]

    ax = axes[0]
    for case in report["isph"]["cases"]:
        ys = [case["per_fraction"][f"{f:g}"]["rel_error_pct"] for f in fractions]
        ax.plot(fractions, ys, "o-", ms=4, label=f"ISPH $H/\\Delta x$={case['resolution_H_over_dx']}")
    for case in report["wcsph"]["cases"]:
        ys = [case["per_fraction"][f"{f:g}"]["rel_error_pct"] for f in fractions]
        ax.plot(fractions, ys, "s--", ms=4, alpha=0.7,
                label=f"WCSPH $H/\\Delta x$={case['resolution_H_over_dx']}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(GATE_LIMIT_PCT, color="r", ls=":", lw=1.0)
    ax.axhline(-GATE_LIMIT_PCT, color="r", ls=":", lw=1.0)
    ax.axvspan(min(GATE_FRACTIONS), max(GATE_FRACTIONS), color="0.85", zorder=0)
    ax.set_xlabel("drift window length / run length  (window $=[(1-f)t_{end},t_{end}]$)")
    ax.set_ylabel(r"$\tau_w$ error vs $\rho g H/2$  [%]")
    ax.set_title("same trace, different window: the reading is window-dominated")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, worst, spread = [], [], []
    for case in report["isph"]["cases"]:
        labels.append(f"ISPH\n{case['resolution_H_over_dx']}")
        worst.append(case["worst_abs_rel_error_pct"])
        spread.append(case["spread_pct_points"])
    for case in report["wcsph"]["cases"]:
        labels.append(f"WCSPH\n{case['resolution_H_over_dx']}")
        worst.append(case["worst_abs_rel_error_pct"])
        spread.append(case["spread_pct_points"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, worst, width=0.4, label="worst |error| in post-transient band")
    ax.bar(x + 0.2, spread, width=0.4, label="band width (estimator uncertainty)")
    ax.axhline(GATE_LIMIT_PCT, color="r", ls=":", lw=1.0, label="3 % gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("[%]")
    ax.set_title("quoted precision vs estimator spread (m16581 claim: 0.05 %)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "wall_shear_gate_audit"
                                        / "sensitivity.json"))
    ap.add_argument("--figure", default=str(REPO / "docs" / "figures"
                                            / "wall_shear_window_sensitivity.png"))
    args = ap.parse_args(argv)

    missing = [str(p) for p in (WCSPH_ARTEFACT, ISPH_ARTEFACT) if not p.exists()]
    if missing:
        print("missing artefacts: " + ", ".join(missing), file=sys.stderr)
        return 2

    wdata = json.loads(WCSPH_ARTEFACT.read_text())
    idata = json.loads(ISPH_ARTEFACT.read_text())

    report = {
        "case": "channel_wall_shear_gate_audit",
        "audited_deliverable": "docs/verification_poiseuille_wall_shear_library_gate.md (m16581, PR #573)",
        "audit_mission": "m16654",
        "anchor": "KPI-QC-RECHECK",
        "sources": {
            str(WCSPH_ARTEFACT.relative_to(REPO)): _sha256(WCSPH_ARTEFACT),
            str(ISPH_ARTEFACT.relative_to(REPO)): _sha256(ISPH_ARTEFACT),
        },
        "definitions": {
            "budget": "tau_w = (1/2) rho0 H (g - d<u>/dt), least squares over the window",
            "fraction": "window length / run length; window = [(1-f) t_end, t_end]",
            "gate_limit_pct": GATE_LIMIT_PCT,
            "declared_fraction": DECLARED_FRACTION,
            "post_transient_fractions": list(GATE_FRACTIONS),
        },
        "wcsph": wcsph(wdata),
        "isph": isph(idata),
    }

    published = {
        "wcsph": [-0.39, 1.94, 0.85],     # m16581 §3 "动量收支（未校准）"
        "isph": [-0.05, -0.05, -0.05],    # m16581 §3 ISPH ladder
    }
    report["kpi_recheck"] = {
        "wcsph": audited_kpi([(f"WCSPH H/dx={c['resolution_H_over_dx']}", c, published["wcsph"][i])
                              for i, c in enumerate(report["wcsph"]["cases"])]),
        "isph": audited_kpi([(f"ISPH H/dx={c['resolution_H_over_dx']}", c, published["isph"][i])
                             for i, c in enumerate(report["isph"]["cases"])]),
    }
    report["kpi_recheck"]["all_bands_within_gate"] = all(
        row["verdict"] == "PASS"
        for group in ("wcsph", "isph") for row in report["kpi_recheck"][group])
    report["kpi_recheck"]["all_published_match_declared_window"] = all(
        row["published_matches_declared_window"]
        for group in ("wcsph", "isph") for row in report["kpi_recheck"][group])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    figure(report, Path(args.figure))
    print(f"wrote {out} and {args.figure}")
    for group in ("wcsph", "isph"):
        for row, case in zip(report["kpi_recheck"][group], report[group]["cases"]):
            print(f"  {row['ladder']:<16} declared={row['declared_window_pct']:+.3f}% "
                  f"band=[{row['band_pct'][0]:+.3f},{row['band_pct'][1]:+.3f}] "
                  f"spread={row['spread_pct_points']:.3f}pt "
                  f"full-run={case['full_run_rel_error_pct']:+.3f}% "
                  f"published={row['published_pct']:+.2f}% -> {row['verdict']}"
                  f"{'' if row['published_matches_declared_window'] else ' (published !~ declared window)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())