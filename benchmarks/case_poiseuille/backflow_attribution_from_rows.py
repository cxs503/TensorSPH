#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH -- C6 attribution from the committed rows (no solver).

The full resolution/column sweep is submitted to the cluster
(``backflow_attribution.py``); this script answers the two questions that need
**no new runs** by re-deriving them from the rows already committed in
``out/combined.json``:

* is the declared ``min_k u_k`` estimator what fails, or the profile?  -> apply
  the interior-band parabola fit (the C4 wall-shear estimator) to the same rows;
* what does the gate's "5 %" mean in velocity?  -> the peak is only
  ``gauge/|u_peak|`` of the gauge every other gate uses, so the same 5 % is that
  many times tighter, and a C1-compliant profile can fail C6 by construction.

Everything is recomputed from ``runs[*].rows``; nothing is read from the stored
percentages.  The record mirrors the schema of ``backflow_attribution.json`` so
the same tests and the same document generator consume it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import combined as cmb  # noqa: E402

OUT = REPO / "benchmarks" / "case_poiseuille" / "out"
H_FACTOR = 1.3
SUPPORT_DX = 2.0 * H_FACTOR
GATE = cmb.GATES["C6_backflow_peak_pct"]


def analytic_quadratic(g, uw, h=cmb.HEIGHT, nu=cmb.VISCOSITY):
    return {"a": -g / (2.0 * nu), "b": g * h / (2.0 * nu) + uw / h, "c": 0.0}


def fit_peak(y, u):
    sel = (y / cmb.HEIGHT >= 0.15) & (y / cmb.HEIGHT <= 0.85)
    a, b, c = np.polyfit(y[sel], u[sel], 2)
    if a <= 0.0:
        return {"y": float("nan"), "u": float("nan"), "a": float(a),
                "b": float(b), "c": float(c), "n_rows": int(sel.sum())}
    ye = -b / (2.0 * a)
    return {"y": float(ye), "u": float(a * ye * ye + b * ye + c), "a": float(a),
            "b": float(b), "c": float(c), "n_rows": int(sel.sum())}


def main() -> int:
    d = json.loads((OUT / "combined.json").read_text())
    p = d["parameters"]
    g, uw, h, nu = p["g"], p["wall_speed"], p["height"], p["viscosity"]
    ext = cmb.analytic_extrema()
    gauge = cmb.gauge_scale(g=g, wall_speed=uw)
    peak_abs = abs(ext["u"])
    ana_q = analytic_quadratic(g, uw)
    sweep = []
    for run in d["runs"]:
        rows = run["rows"]
        y = np.array([r["y"] for r in rows], dtype=float)
        u = np.array([r["u"] for r in rows], dtype=float)
        ua = np.array([r["u_analytic"] for r in rows], dtype=float)
        dx = run["dx"]
        k = int(np.argmin(u))
        y_min, u_min = float(y[k]), float(u[k])
        u_ref = float(ua[k])
        fit = fit_peak(y, u)
        sweep.append({
            "resolution_H_over_dx": run["resolution_H_over_dx"],
            "n_cols": 8, "n_fluid": run["n_fluid"], "dx": dx,
            "dt": run["dt"], "steps": run["steps"],
            "wall_clock_s": run["wall_clock_s"],
            "declared": {"estimator": "min_k u_k, graded against the closed form at that row",
                         "y_at_min": y_min, "u_min": u_min,
                         "u_min_analytic_at_same_row": u_ref,
                         "peak_pct": 100.0 * (u_min - u_ref) / abs(u_ref),
                         "pass_5pct": bool(abs(100.0 * (u_min - u_ref) / abs(u_ref)) <= GATE)},
            "extremum_row_over_dx": y_min / dx,
            "support_over_dx": SUPPORT_DX,
            "inside_kernel_support": bool(y_min / dx < SUPPORT_DX),
            "row_residual_at_min_pct": 100.0 * (u_min - u_ref) / abs(u_ref),
            "fit": {"estimator": "extremum of polyfit(y, u, 2) on 0.15<=y/H<=0.85",
                    "y": fit["y"], "u": fit["u"], "u_analytic": ext["u"],
                    "pct": 100.0 * (fit["u"] - ext["u"]) / peak_abs,
                    "pass_5pct": bool(abs(100.0 * (fit["u"] - ext["u"]) / peak_abs) <= GATE),
                    "y_pct": 100.0 * (fit["y"] - ext["y"]) / ext["y"],
                    "n_rows": fit["n_rows"], "curvature_a": fit["a"]},
            "scale": {"gauge": float(gauge), "peak_abs": float(peak_abs),
                      "gauge_over_peak": float(gauge / peak_abs),
                      "C6_tolerance_m_s": float(GATE / 100.0 * peak_abs),
                      "C6_tolerance_in_gauge_pct": float(100.0 * GATE / 100.0 * peak_abs / gauge),
                      "max_abs_row_error": float(np.max(np.abs(u - ua))),
                      "max_row_error_pct_of_gauge": float(100.0 * np.max(np.abs(u - ua)) / gauge),
                      "rms_error_pct_of_gauge": float(100.0 * np.sqrt(np.mean((u - ua) ** 2)) / gauge),
                      "max_row_error_pct_of_peak": float(100.0 * np.max(np.abs(u - ua)) / peak_abs),
                      "rms_error_pct_of_peak": float(100.0 * np.sqrt(np.mean((u - ua) ** 2)) / peak_abs),
                      "note": "peak is gauge/(peak) smaller, so 5 % of it is that much tighter"},
            "fit_coefficients": {"a": fit["a"], "b": fit["b"], "c": fit["c"],
                                 "a_pct": 100.0 * (fit["a"] - ana_q["a"]) / ana_q["a"],
                                 "b_pct": 100.0 * (fit["b"] - ana_q["b"]) / ana_q["b"],
                                 "c_m_s": fit["c"], "c_pct_of_peak": 100.0 * fit["c"] / peak_abs},
            "profile_rms_pct": run["profile_rms_pct"],
            "tau_bottom_pct": run["tau_bottom_pct"],
            "tau_top_pct": run["tau_top_pct"],
            "rows": [{"y": float(yy), "y_over_H": float(yy / h), "y_over_dx": float(yy / dx),
                      "u": float(uu), "u_analytic": float(aa),
                      "residual_pct_of_peak": float(100.0 * (uu - aa) / peak_abs)}
                     for yy, uu, aa in zip(y, u, ua)],
        })
    sweep.sort(key=lambda r: r["resolution_H_over_dx"])
    hits, misses = [], []
    for r in sweep:
        rec = {"resolution_H_over_dx": r["resolution_H_over_dx"], "n_cols": 8,
               "extremum_row_over_dx": r["extremum_row_over_dx"],
               "predicted_fail": r["inside_kernel_support"],
               "observed_fail": not r["declared"]["pass_5pct"],
               "peak_pct": r["declared"]["peak_pct"]}
        (hits if rec["predicted_fail"] == rec["observed_fail"] else misses).append(rec)
    worst_fit = max(sweep, key=lambda r: abs(r["fit"]["pct"]))
    sc0 = sweep[0]["scale"]
    rec = {
        "case": cmb.CASE_KEY, "anchor": "CASE-POISEUILLE-SPH",
        "title": "attribution of the honest C6 backflow-peak FAIL (from committed rows)",
        "generated_by": "benchmarks/case_poiseuille/backflow_attribution_from_rows.py",
        "plan": {"mode": "postprocess_published_rows", "runs_started": 0,
                 "resolutions": [r["resolution_H_over_dx"] for r in sweep],
                 "column_sweep": {}, "boundary_probe": [],
                 "note": "no solver was run here; rows come from out/combined.json"},
        "parameters": {"height": h, "viscosity": nu, "rho0": p["rho0"], "g": g,
                       "wall_speed": uw, "sound_speed": p["sound_speed"],
                       "h_factor": H_FACTOR, "support_over_dx": SUPPORT_DX,
                       "t_end": p["t_end"], "steady_window": p["steady_window"],
                       "gate_C6_pct": GATE},
        "crossover": {"resolution": float(SUPPORT_DX / (ext["y"] / h)),
                      "criterion": "y_ext/dx >= support_over_dx = 2*h_factor",
                      "tested": False},
        "analytic": {"extremum_y": ext["y"], "extremum_u": ext["u"],
                     "extremum_row_over_dx": float(ext["y"] / h)},
        "hypotheses": {
            "H1": "C6 FAIL iff the extremum row is inside one kernel support (y_ext/dx < 2.6)",
            "H2": "the declared raw-min estimator is the cause; the interior fit rescues it",
            "H0": "the coarse error is transverse quadrature noise (8 columns per row)"},
        "declared_gate": {"estimator": "raw row minimum, tolerance 5 %",
                          "verdict_preserved": "FAIL at the coarse rungs"},
        "resolution_sweep": sweep, "boundary_probe": [], "column_sweep": {},
        "verdict_table": {"n": len(sweep), "hits": hits, "misses": misses,
                          "hypothesis_H1_supported": not misses},
        "column_effect": {},
        "unconcluded": {
            "H1_crossover_14_vs_12": "not tested here (needs H/dx = 14; queued: cj4530 full, cj4546 quick)",
            "H0_column_sweep": "not tested here (needs n_cols = 16/32/64; queued: cj4530)",
            "H2_status": "falsified: with the fitted constant c kept in the peak value, the interior fit fails wherever the raw minimum fails"},
        "summary": {"worst_declared_peak_pct": min(r["declared"]["peak_pct"] for r in sweep),
                    "worst_declared_at_Hdx": sweep[0]["resolution_H_over_dx"],
                    "worst_fit_pct": worst_fit["fit"]["pct"],
                    "worst_fit_at_Hdx": worst_fit["resolution_H_over_dx"],
                    "declared_verdict_at_Hdx8_independent_of_n_cols": False,
                    "wall_clock_s": float(d.get("wall_clock_s") or 0.0)},
        "device": d.get("device"), "torch": d.get("torch"), "python": d.get("python"),
    }
    path = OUT / "backflow_attribution.json"
    path.write_text(json.dumps(rec, indent=1))
    for r in sweep:
        print(f"H/dx={r['resolution_H_over_dx']:>2} y_ext/dx={r['extremum_row_over_dx']:.2f} "
              f"{'inside ' if r['inside_kernel_support'] else 'outside'} "
              f"declared={r['declared']['peak_pct']:+7.3f}% fit={r['fit']['pct']:+7.3f}% "
              f"c={r['fit_coefficients']['c_m_s']:+.5f} gauge%={r['scale']['max_row_error_pct_of_gauge']:.3f}")
    print(f"H1 {len(hits)}/{len(sweep)} supported; worst fit {worst_fit['fit']['pct']:+.3f}% "
          f"@Hdx{worst_fit['resolution_H_over_dx']}")
    print(f"gate 5% of peak = {sc0['C6_tolerance_in_gauge_pct']:.3f} % of gauge "
          f"(gauge/peak = {sc0['gauge_over_peak']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
