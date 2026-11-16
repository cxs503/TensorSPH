#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH -- attribution of the honest C6 backflow-peak FAIL.

Why this study exists
---------------------
The case-pool record for ``CASE-POISEUILLE-SPH`` carries eight of nine
combined-case gates as PASS and one as an honest FAIL: ``C6`` (the backflow
peak of the combined Couette-Poiseuille run) is **-12.370 %** at ``H/dx = 8``
and **-8.232 %** at ``H/dx = 12``, against a pre-declared tolerance of 5 %,
while it passes at ``H/dx = 16`` (**-1.321 %**).  A published FAIL with no
mechanism is an unfinished verification: nobody can tell whether it is a solver
defect, an estimator defect, or a resolution effect, so nobody can decide
whether the tolerance or the measurement should change.

Pre-declared hypotheses (written before any run of this script)
--------------------------------------------------------------
The closed form of the combined case is one quadratic,

    u(y) = g y (H - y) / (2 nu) + U_w y / H ,

whose backflow peak sits at a *fixed physical height*, ``y_ext = 0.020031 m``
(``y_ext/H = 0.2003``).  Measured in cell spacings that height is
``y_ext/dx = 0.2003 * (H/dx)``, while the mirror/ghost no-slip planes act on the
fluid through one kernel support, ``2h = 2 * h_factor * dx = 2.6 dx``.  Hence

    H1 (geometry/kernel truncation, not statistics)
        The C6 error is governed by whether the extremum row sits *inside* one
        kernel support of the wall.  It is large while ``y_ext < 2h`` and small
        once ``y_ext > 2h``; the crossover is at ``H/dx = 2.6 / 0.2003 = 12.98``.
        Predictions, rung by rung (0.2003 * H/dx compared with 2.6):
            8 -> 1.60  inside   -> |error| > 5 %   (already observed: 12.37 %)
           10 -> 2.00  inside   -> |error| > 5 %
           12 -> 2.40  inside   -> |error| > 5 %   (already observed:  8.23 %)
           14 -> 2.80  outside  -> |error| < 5 %
           16 -> 3.21  outside  -> |error| < 5 %   (already observed:  1.32 %)
           20 -> 4.01  outside  -> |error| < 5 %
           24 -> 4.81  outside  -> |error| < 5 %
        The decisive neighbours are 12 vs 14: their ratio of ``y_ext`` to the
        support differs by 17 %, the sign of the verdict must flip.

    H2 (estimator, not physics)
        The declared C6 estimator is the raw minimum of the row-mean profile,
        ``min_k u_k``, graded against the closed form *at that same row*.  The
        interior solution is demonstrably right (the C4 wall shears, which come
        from a least-squares quadratic fit over ``0.15 <= y/H <= 0.85``, pass at
        every rung with |error| < 2 %).  The same fit contains the whole
        parabola and therefore the peak; if its extremum is inside 5 % at the
        coarse rungs too, then C6 as declared measures the near-wall rows, not
        the solution.

    H0 (the alternative we must be able to falsify)
        The coarse-rung error is transverse quadrature noise: with only 8
        columns the row mean is a mean over 8 particles.  Falsification test:
        sweep ``n_cols = 8, 16, 32, 64`` at fixed ``H/dx``.  Under H1 the error
        must barely move; if it collapses, H1 is wrong and the FAIL is a
        sampling artefact of the channel width.

Discipline
----------
Nothing here re-grades the anchor's verdict.  The declared gate is reported
exactly as declared (raw row minimum, 5 %), and the study adds: a mechanism,
a falsification test, and a second estimator reported *separately*.  If the
mechanism lands, the honest FAIL stays a FAIL -- it becomes an explained one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import combined as cmb  # noqa: E402
from benchmarks.case_poiseuille import run as caserun  # noqa: E402

TAG = "CASE-POISEUILLE-SPH/C6-attribution"

HEIGHT = caserun.HEIGHT
VISCOSITY = caserun.VISCOSITY
SOUND_SPEED = cmb.SOUND_SPEED
G = cmb.G_ADVERSE
WALL_SPEED = cmb.WALL_SPEED

# lattice kernel support in units of dx: 2h = 2 * h_factor * dx
H_FACTOR = 1.3
SUPPORT_DX = 2.0 * H_FACTOR                       # 2.6
EXTREMUM_Y = cmb.analytic_extrema()["y"]          # 0.020030581039755357
EXTREMUM_U = cmb.analytic_extrema()["u"]          # -0.09840022935779819
CROSSOVER_RESOLUTION = SUPPORT_DX / (EXTREMUM_Y / HEIGHT)  # H/dx = 12.98

RESOLUTION_SWEEP = (8, 10, 12, 14, 16, 20, 24)
BOUNDARY_PROBE = (13,)                            # y_ext/dx = 2.604 ~ 2.6
COLUMN_SWEEP = {8: (8, 16, 32, 64), 12: (8, 16, 32)}
GATE_C6_PCT = cmb.GATES["C6_backflow_peak_pct"]
T_END = cmb.T_END
STEADY_TIMES = cmb.STEADY_TIMES


# --------------------------------------------------------------------------- #
# One rung
# --------------------------------------------------------------------------- #

def run_rung(resolution: int, n_cols: int, device) -> dict:
    """One steady combined run at ``H/dx = resolution`` over ``n_cols`` columns."""
    dx = HEIGHT / resolution
    length = n_cols * dx
    base = caserun.make_config(dx, VISCOSITY, 1.0, device)
    cfg = dataclasses.replace(base, sound_speed=SOUND_SPEED,
                              pool_width=length, pool_depth=HEIGHT)
    planes = [(0.0, +1), (HEIGHT, -1, (WALL_SPEED, 0.0))]
    case = caserun.drive(cfg, HEIGHT, length, lambda t: G, T_END, STEADY_TIMES,
                         device, planes=planes)
    y = np.asarray(case["y_rows"], dtype=float)
    u = np.mean(case["u_samples"], axis=0)
    case.update({"resolution_H_over_dx": resolution, "y": y, "u_profile": u,
                 "g": G, "wall_speed": WALL_SPEED, "dx": dx,
                 "n_cols": n_cols, "length": length})
    return case


def analytic_quadratic() -> dict:
    """Exact coefficients of ``u = a y^2 + b y + c`` for the combined case.

    ``u = g y (H-y)/(2 nu) + U_w y / H`` -> ``a = -g/(2 nu)``,
    ``b = g H/(2 nu) + U_w/H``, ``c = 0`` (the no-slip condition *is* ``c = 0``,
    which is why the fitted constant is the cleanest single measure of a
    near-wall offset).
    """
    return {"a": -G / (2.0 * VISCOSITY),
            "b": G * HEIGHT / (2.0 * VISCOSITY) + WALL_SPEED / HEIGHT,
            "c": 0.0}


def fit_extremum(y, u) -> dict:
    """Interior-band least-squares quadratic -> its extremum (interior minimum).

    The steady combined profile *is* a parabola, so a fit over the interior band
    (``0.15 <= y/H <= 0.85``, the band the C4 wall-shear gate already uses)
    determines the peak without ever differencing a near-wall row.  Returned
    with the fit's own coefficients so a caller can check the curvature sign.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    sel = (y / HEIGHT >= 0.15) & (y / HEIGHT <= 0.85)
    a, b, c = np.polyfit(y[sel], u[sel], 2)
    # u_yy = -g/nu > 0 for the adverse gradient, so the backflow peak is the
    # interior *minimum* of an upward parabola: a > 0 is the physical sign
    if a <= 0.0:                      # no interior minimum: fit is not physical
        return {"y": float("nan"), "u": float("nan"), "a": float(a),
                "b": float(b), "c": float(c), "n_rows": int(sel.sum())}
    y_ext = -b / (2.0 * a)
    return {"y": float(y_ext), "u": float(a * y_ext**2 + b * y_ext + c),
            "a": float(a), "b": float(b), "c": float(c),
            "n_rows": int(sel.sum())}


def attribute(case: dict) -> dict:
    """Grade one run with the declared estimator and the fit estimator."""
    graded = cmb.grade_run(case, f"combined_Hdx{case['resolution_H_over_dx']}"
                                 f"_nc{case['n_cols']}")
    y, u = case["y"], case["u_profile"]
    dx = case["dx"]
    bf = graded["backflow"]
    fit = fit_extremum(y, u)
    ana_q = analytic_quadratic()
    gauge = cmb.gauge_scale(g=G, wall_speed=WALL_SPEED)
    peak_abs = abs(EXTREMUM_U)
    u_a = cmb.analytic_profile(y, g=G, wall_speed=WALL_SPEED)
    row_sel = np.argmin(u)
    return {
        "resolution_H_over_dx": case["resolution_H_over_dx"],
        "n_cols": case["n_cols"],
        "n_fluid": case["n_fluid"],
        "dx": dx,
        "dt": case["dt"],
        "steps": case["steps"],
        "wall_clock_s": case["wall_clock_s"],
        # --- the declared C6 estimator (raw row minimum), unchanged ----------
        "declared": {
            "estimator": "min_k u_k, graded against the closed form at that row",
            "y_at_min": bf["y_at_min"],
            "u_min": bf["u_min"],
            "u_min_analytic_at_same_row": bf["u_min_analytic_at_same_row"],
            "peak_pct": bf["peak_pct"],
            "pass_5pct": bool(abs(bf["peak_pct"]) <= GATE_C6_PCT),
        },
        # --- shape diagnostics: is the extremum row inside one support? -----
        "extremum_row_over_dx": float(bf["y_at_min"] / dx),
        "support_over_dx": SUPPORT_DX,
        "inside_kernel_support": bool(bf["y_at_min"] / dx < SUPPORT_DX),
        "row_residual_at_min_pct": float(
            100.0 * (bf["u_min"] - bf["u_min_analytic_at_same_row"])
            / abs(bf["u_min_analytic_at_same_row"])),
        # --- the fit estimator (reported separately, never substituted ------
        "fit": {
            "estimator": "extremum of polyfit(y, u, 2) on 0.15<=y/H<=0.85",
            "y": fit["y"],
            "u": fit["u"],
            "u_analytic": EXTREMUM_U,
            "pct": float(100.0 * (fit["u"] - EXTREMUM_U) / abs(EXTREMUM_U)),
            "pass_5pct": bool(abs(100.0 * (fit["u"] - EXTREMUM_U)
                                  / abs(EXTREMUM_U)) <= GATE_C6_PCT),
            "y_pct": float(100.0 * (fit["y"] - EXTREMUM_Y) / EXTREMUM_Y),
            "n_rows": fit["n_rows"],
            "curvature_a": fit["a"],
        },
        # --- scale coherence of the gate: what does "5 %" mean here? -------
        "scale": {
            "gauge": float(gauge),
            "peak_abs": float(peak_abs),
            "gauge_over_peak": float(gauge / peak_abs),
            "C6_tolerance_m_s": float(GATE_C6_PCT / 100.0 * peak_abs),
            "C6_tolerance_in_gauge_pct": float(100.0 * GATE_C6_PCT / 100.0
                                               * peak_abs / gauge),
            "max_abs_row_error": float(np.max(np.abs(u - u_a))),
            "max_row_error_pct_of_gauge": float(
                100.0 * np.max(np.abs(u - u_a)) / gauge),
            "rms_error_pct_of_gauge": float(
                100.0 * np.sqrt(np.mean(np.square(u - u_a))) / gauge),
            "max_row_error_pct_of_peak": float(
                100.0 * np.max(np.abs(u - u_a)) / peak_abs),
            "rms_error_pct_of_peak": float(
                100.0 * np.sqrt(np.mean(np.square(u - u_a))) / peak_abs),
            "note": ("the peak is only gauge/(peak) times smaller than the "
                     "gauge, so the same 5 % relative tolerance is that many "
                     "times tighter in absolute velocity"),
        },
        "fit_coefficients": {
            "a": fit["a"], "b": fit["b"], "c": fit["c"],
            "a_pct": float(100.0 * (fit["a"] - ana_q["a"]) / ana_q["a"]),
            "b_pct": float(100.0 * (fit["b"] - ana_q["b"]) / ana_q["b"]),
            "c_m_s": fit["c"],
            "c_pct_of_peak": float(100.0 * fit["c"] / peak_abs),
        },
        "profile_rms_pct": graded["profile_rms_pct"],
        "tau_bottom_pct": graded["tau_bottom_pct"],
        "tau_top_pct": graded["tau_top_pct"],
        # --- the wall-normal error profile, for the figure ------------------
        "rows": [{"y": float(yy), "y_over_H": float(yy / HEIGHT),
                  "y_over_dx": float(yy / dx),
                  "u": float(uu), "u_analytic": float(aa),
                  "residual_pct_of_peak": float(100.0 * (uu - aa) / abs(EXTREMUM_U))}
                 for yy, uu, aa in zip(y, u, u_a)],
    }


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #

def verdict_table(rows) -> dict:
    """Do the pre-declared labels reproduce the observed verdicts?"""
    hits, misses = [], []
    for r in rows:
        predicted_large = r["inside_kernel_support"]
        observed_large = not r["declared"]["pass_5pct"]
        rec = {"resolution_H_over_dx": r["resolution_H_over_dx"],
               "n_cols": r["n_cols"],
               "extremum_row_over_dx": r["extremum_row_over_dx"],
               "predicted_fail": bool(predicted_large),
               "observed_fail": bool(observed_large),
               "peak_pct": r["declared"]["peak_pct"]}
        (hits if predicted_large == observed_large else misses).append(rec)
    return {"n": len(rows), "hits": hits, "misses": misses,
            "hypothesis_H1_supported": not misses}


def column_effect(runs_by_res) -> dict:
    """H0 test: does the coarse-rung error depend on the number of columns?"""
    out = {}
    for res, rows in runs_by_res.items():
        peaks = {r["n_cols"]: r["declared"]["peak_pct"] for r in rows}
        vals = np.array(list(peaks.values()), dtype=float)
        out[str(res)] = {
            "peak_pct_by_n_cols": {str(k): float(v) for k, v in peaks.items()},
            "spread_pp": float(vals.max() - vals.min()),
            "all_fail": bool(np.all(np.abs(vals) > GATE_C6_PCT)),
            "all_pass": bool(np.all(np.abs(vals) <= GATE_C6_PCT)),
        }
    out["H0_transverse_quadrature_noise"] = bool(
        all(v["all_fail"] or v["all_pass"] for v in out.values()
            if isinstance(v, dict)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("out")))
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="reduced plan (rungs 8/12/14/16 + one H0 control) so a "
                         "loaded host can still decide the crossover")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    resolutions = (8, 12, 14, 16) if args.quick else RESOLUTION_SWEEP
    boundary_plan = () if args.quick else BOUNDARY_PROBE
    column_plan = {8: (8, 32)} if args.quick else COLUMN_SWEEP
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall = time.time()

    sweep, boundary, by_res = [], [], {}
    for res in resolutions:
        print(f"[{TAG}] H/dx={res} cols=8 ...", flush=True)
        r = attribute(run_rung(res, 8, device))
        sweep.append(r)
        print(f"[{TAG}]   declared {r['declared']['peak_pct']:+.3f} % "
              f"({'FAIL' if not r['declared']['pass_5pct'] else 'pass'})  "
              f"y_ext/dx={r['extremum_row_over_dx']:.2f} "
              f"({'inside' if r['inside_kernel_support'] else 'outside'} 2h)  "
              f"fit {r['fit']['pct']:+.3f} %", flush=True)
    for res in boundary_plan:
        print(f"[{TAG}] H/dx={res} cols=8 (boundary probe) ...", flush=True)
        r = attribute(run_rung(res, 8, device))
        boundary.append(r)
        print(f"[{TAG}]   declared {r['declared']['peak_pct']:+.3f} %  "
              f"y_ext/dx={r['extremum_row_over_dx']:.3f}", flush=True)
    for res, cols in column_plan.items():
        by_res[res] = []
        for nc in cols:
            print(f"[{TAG}] H/dx={res} cols={nc} (H0 test) ...", flush=True)
            r = attribute(run_rung(res, nc, device))
            by_res[res].append(r)
            print(f"[{TAG}]   declared {r['declared']['peak_pct']:+.3f} %  "
                  f"fit {r['fit']['pct']:+.3f} %", flush=True)

    all_rows = sweep + boundary + [r for rows in by_res.values() for r in rows]
    table = verdict_table(all_rows)
    cols = column_effect(by_res)

    worst_raw = min(sweep, key=lambda r: r["declared"]["peak_pct"])
    worst_fit = max(sweep, key=lambda r: abs(r["fit"]["pct"]))
    record = {
        "case": cmb.CASE_KEY,
        "anchor": "CASE-POISEUILLE-SPH",
        "title": "attribution of the honest C6 backflow-peak FAIL",
        "generated_by": "benchmarks/case_poiseuille/backflow_attribution.py",
        "parameters": {"height": HEIGHT, "viscosity": VISCOSITY,
                       "g": G, "wall_speed": WALL_SPEED,
                       "sound_speed": SOUND_SPEED, "h_factor": H_FACTOR,
                       "support_over_dx": SUPPORT_DX, "t_end": T_END,
                       "steady_window": [STEADY_TIMES[0], STEADY_TIMES[-1]],
                       "gate_C6_pct": GATE_C6_PCT},
        "crossover": {"resolution": float(CROSSOVER_RESOLUTION),
                      "criterion": "y_ext/dx >= support_over_dx = 2*h_factor"},
        "analytic": {"extremum_y": EXTREMUM_Y, "extremum_u": EXTREMUM_U,
                     "extremum_row_over_dx": float(EXTREMUM_Y / HEIGHT)},
        "hypotheses": {
            "H1": "C6 FAIL iff the extremum row lies inside one kernel support "
                  "(y_ext/dx < 2h/dx = 2.6); crossover at H/dx = 12.98",
            "H2": "the declared raw-min estimator measures near-wall rows; the "
                  "interior-band quadratic fit recovers the peak",
            "H0": "alternative: the coarse error is transverse quadrature noise "
                  "with 8 columns (falsified by the column sweep)",
        },
        "plan": {"resolutions": list(resolutions),
                 "boundary_probe": list(boundary_plan),
                 "column_sweep": {str(k): list(v) for k, v in column_plan.items()},
                 "quick": bool(args.quick)},
        "declared_gate": {"estimator": "raw row minimum, tolerance 5 %",
                          "verdict_preserved": "FAIL at the coarse rungs"},
        "resolution_sweep": sweep,
        "boundary_probe": boundary,
        "column_sweep": {str(k): v for k, v in by_res.items()},
        "verdict_table": table,
        "column_effect": cols,
        "summary": {
            "worst_declared_peak_pct": worst_raw["declared"]["peak_pct"],
            "worst_declared_at_Hdx": worst_raw["resolution_H_over_dx"],
            "worst_fit_pct": worst_fit["fit"]["pct"],
            "worst_fit_at_Hdx": worst_fit["resolution_H_over_dx"],
            "declared_verdict_at_Hdx8_independent_of_n_cols": cols["8"]["all_fail"],
            "wall_clock_s": time.time() - wall,
        },
        "device": str(device),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    path = outdir / "backflow_attribution.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"[{TAG}] wrote {path} ({record['summary']['wall_clock_s']:.1f}s)")
    print(f"[{TAG}] H1 supported: {table['hypothesis_H1_supported']} "
          f"({len(table['hits'])}/{table['n']})")
    for k, v in cols.items():
        if isinstance(v, dict):
            print(f"[{TAG}] H0 @Hdx={k}: spread {v['spread_pp']:.3f} pp, "
                  f"all_fail={v['all_fail']}")
    ok = table["hypothesis_H1_supported"]
    print(f"[{TAG}] H1 prediction reproduced: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())