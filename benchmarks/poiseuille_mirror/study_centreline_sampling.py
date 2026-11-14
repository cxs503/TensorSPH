#!/usr/bin/env python3
"""Gate-A sampling-bias study for the mirror-wall Poiseuille case.

Motivation
----------
The canonical run (:mod:`benchmarks.poiseuille_mirror.run`) grades the
centre-line gate A as

    err_A = max_k u_row(k) / u_max_analytic - 1,     u_max = g H^2 / (8 nu)

i.e. it compares the *largest binned row mean* against the *continuum*
centre-line value.  The lattice rows sit at ``y_k = dx/2 + k dx``, so unless
``H/dx`` is odd no row falls on ``y = H/2``.  For a parabola the nearest row
under-reads the centre-line value by exactly

    u(y_nearest) / u(H/2) = 1 - (dx/H)^2 ,

which is a pure quadrature/sampling artefact of the metric, not a solver
error.  At ``H/dx = 8`` it is already -1.56 %, larger than the 5 % gate by a
comfortable margin and larger than the reported error itself.  The published
headline ("centre-line velocity accurate to 0.02 %", ``docs/
case_poiseuille_sph_report.md``) therefore mixes a metric artefact with the
discretisation error, and the resolution trend of ``err_A`` is dominated by
it.

What this script measures
-------------------------
For each ``H/dx`` in the requested ladder it runs the *unmodified* canonical
case and evaluates four independent centre-line estimators:

``A_legacy``      row-max vs analytic centre-line (the current gate A)
``A_same_row``    row-max vs analytic evaluated at that row's ``y``
``A_fit_full``    quadratic LSQ fit over all rows, evaluated at ``y = H/2``
``A_fit_mid``     quadratic LSQ fit over ``0.25 <= y/H <= 0.75``, at ``H/2``

and the closed-form prediction of the sampling artefact,

``bias_pred = 1 - (dx/H)^2``,  ``A_pred = A_legacy_biasfree * bias_pred``.

Odd resolutions (15, 17, 25) put a row exactly on ``y = H/2``; even ones
(16, 24) do not.  If the artefact hypothesis is right, the legacy metric must
jump between neighbouring odd/even rungs while the three corrected estimators
stay smooth across the pair.  If it is wrong, the odd/even pair agrees and the
hypothesis is falsified -- both outcomes are recorded.

Usage
-----
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_centreline_sampling.py \
        --resolutions 15 16 17 24 25 --threads 4 \
        --out benchmarks/poiseuille_mirror/out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from benchmarks.poiseuille.run import analytic, fit_profile, make_config
from benchmarks.poiseuille_mirror.run import evaluate, run_case_mirror
from tensorsph.droplet_impact import resolve_device

REPO = Path(__file__).resolve().parents[2]


def centreline_estimators(res: dict, height: float, body_force: float,
                          viscosity: float) -> dict:
    """Four centre-line estimators from one converged profile."""
    rows = res["rows"]
    u_max_ref = body_force * height ** 2 / (8.0 * viscosity)
    dx = height / res["resolution_H_over_dx"]

    peak = max(rows, key=lambda r: r["u_mean"])
    legacy = res["u_max_measured"] / u_max_ref - 1.0

    # (b) same-row comparison: analytic evaluated at the row's own y
    u_ref_row = analytic(peak["y"], height, body_force, viscosity)
    same_row = peak["u_mean"] / u_ref_row - 1.0

    # (c)/(d) parabola fits, evaluated analytically at y = H/2
    def fit_at_centre(lo: float, hi: float) -> float | None:
        profile = [{"y": r["y"], "y_over_H": r["y_over_H"],
                    "u_mean": r["u_mean"]} for r in rows]
        try:
            fit = fit_profile(profile, height, body_force, viscosity,
                              lo=lo, hi=hi)
        except Exception:
            return None
        c0, c1, c2 = fit["c0"], fit["c1"], fit["c2"]
        u_c = c0 + c1 * (height / 2.0) + c2 * (height / 2.0) ** 2
        return u_c / u_max_ref - 1.0

    fit_full = fit_at_centre(0.0, 1.0)
    fit_mid = fit_at_centre(0.25, 0.75)

    # closed-form sampling artefact of the nearest row
    delta = dx / (2.0 * height)
    bias_pred = 1.0 - (2.0 * delta) ** 2          # = 1 - (dx/H)^2
    row_on_centre = abs(peak["y"] - height / 2.0) < 1e-12

    return {
        "resolution_H_over_dx": res["resolution_H_over_dx"],
        "dx_over_H": dx / height,
        "nearest_row_y_over_H": peak["y_over_H"],
        "row_sits_on_centreline": bool(row_on_centre),
        "u_max_ref": u_max_ref,
        "u_max_measured": res["u_max_measured"],
        "A_legacy": legacy,
        "A_same_row": same_row,
        "A_fit_full": fit_full,
        "A_fit_mid": fit_mid,
        "bias_pred_factor": bias_pred,
        # legacy metric predicted from the sampling-free fit value
        "A_legacy_from_fit_full": (1.0 + fit_full) * bias_pred - 1.0
        if fit_full is not None else None,
        "profile_rms_interior_pct": res["profile_rms_interior_pct"],
        "profile_rms_all_pct": res["profile_rms_all_rows_pct"],
        "wall_shear_bottom_error_pct": res["wall_shear_bottom_error_pct"],
        "steady_drift": res["steady_drift"],
        "max_mach": res["max_mach"],
        "n_fluid": res["n_fluid"],
        "steps": res["steps"],
        "dt": res["dt"],
        "wall_clock_s": res["wall_clock_s"],
        "gates": res.get("gates", {}),
    }


def monotonicity(series: list[dict], key: str) -> dict:
    """Is ``|key|`` monotone decreasing with resolution, and its order fit."""
    pts = [(r["resolution_H_over_dx"], abs(r[key])) for r in series
           if r.get(key) is not None]
    if len(pts) < 2:
        return {"n_points": len(pts)}
    decreasing = all(b[1] <= a[1] for a, b in zip(pts, pts[1:]))
    orders = []
    for (n1, e1), (n2, e2) in zip(pts, pts[1:]):
        if e1 > 0 and e2 > 0 and n2 > n1:
            orders.append(math.log(e1 / e2) / math.log(n2 / n1))
    return {
        "n_points": len(pts),
        "monotone_decreasing": bool(decreasing),
        "points": [[n, e] for n, e in pts],
        "pairwise_orders": orders,
    }


def odd_even_pairs(series: list[dict]) -> list[dict]:
    """Does the legacy metric jump across a lattice-parity change?

    For an odd ``H/dx`` a row sits exactly on ``y = H/2`` (no sampling
    penalty); for the neighbouring even values it does not.  The corrected
    estimators should be blind to the parity; the legacy one should not.
    """
    pairs = []
    by_n = {e["resolution_H_over_dx"]: e for e in series}
    for n in sorted(by_n):
        if n % 2 == 1 and (n + 1) in by_n and (n - 1) in by_n:
            odd, up, dn = by_n[n], by_n[n + 1], by_n[n - 1]
            pairs.append({
                "odd": n,
                "jump_legacy_pct": 100 * (up["A_legacy"] - odd["A_legacy"]),
                "jump_same_row_pct": 100 * (up["A_same_row"]
                                            - odd["A_same_row"]),
                "jump_fit_full_pct": 100 * ((up["A_fit_full"] or 0)
                                            - (odd["A_fit_full"] or 0)),
                "neighbour_spread_legacy_pct":
                    100 * (up["A_legacy"] - dn["A_legacy"]),
                "neighbour_spread_fit_full_pct":
                    100 * ((up["A_fit_full"] or 0) - (dn["A_fit_full"] or 0)),
            })
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=int, nargs="+",
                    default=[15, 16, 17, 24, 25])
    ap.add_argument("--height", type=float, default=0.1)
    ap.add_argument("--viscosity", type=float, default=0.01)
    ap.add_argument("--body-force", type=float, default=9.81)
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--t-average", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks/poiseuille_mirror/out")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    base_payload = {
        "case": "plane_poiseuille_channel_mirror_wall_centreline_sampling",
        "anchor": "CASE-POISEUILLE-SPH",
        "question": ("does gate A compare a row mean with the continuum "
                     "centre-line value, and how large is that sampling "
                     "artefact?"),
        "hypothesis": ("A_legacy carries a 1-(dx/H)^2 sampling factor; "
                       "A_same_row / A_fit_* are free of it"),
        "setup": {
            "height": args.height, "viscosity": args.viscosity,
            "body_force": args.body_force, "columns": args.columns,
            "t_end": args.t_end, "t_average": args.t_average,
            "threads": args.threads, "torch": torch.__version__,
            "device": str(device),
        },
    }
    # Incremental persistence: a rung may abort (an allocator abort was seen
    # once on the loong64 nodes), and the rungs already computed must survive.
    artefact = args.out / "centreline_sampling.json"
    series = []
    failures = []
    if artefact.exists():
        try:
            series = json.loads(artefact.read_text()).get("series", [])
        except Exception:
            series = []
    series = [e for e in series
              if e["resolution_H_over_dx"] not in args.resolutions]

    def dump(extra: dict | None = None) -> None:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = dict(base_payload)
        payload["series"] = sorted(series,
                                   key=lambda e: e["resolution_H_over_dx"])
        payload["failures"] = failures
        if extra:
            payload.update(extra)
        artefact.write_text(json.dumps(payload, indent=2))

    for r in args.resolutions:
        print(f"[centreline-sampling] H/dx={r} ...", flush=True)
        dx = args.height / r
        length = args.columns * dx
        try:
            cfg = make_config(dx, args.viscosity, 1.0, device)
            case = run_case_mirror(cfg, args.height, length, args.body_force,
                                   args.t_end, args.t_average, device)
            res = evaluate(case, args.height, args.body_force, args.viscosity,
                           1000.0, r)
            res["resolution_H_over_dx"] = r
            entry = centreline_estimators(res, args.height, args.body_force,
                                          args.viscosity)
        except Exception as exc:  # keep going: one bad rung is not a study
            failures.append({"resolution_H_over_dx": r,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"[centreline-sampling] H/dx={r} FAILED: {exc}", flush=True)
            dump()
            continue
        series.append(entry)
        dump()
        print("[centreline-sampling] ENTRY " + json.dumps(entry), flush=True)
        print("[centreline-sampling]   A_legacy %+.3f%%  A_same_row %+.3f%%  "
              "A_fit_full %+.3f%%  bias_pred %.4f  (%.0fs)"
              % (100 * entry["A_legacy"], 100 * entry["A_same_row"],
                 100 * (entry["A_fit_full"] or float("nan")),
                 entry["bias_pred_factor"], entry["wall_clock_s"]), flush=True)

    series = sorted(series, key=lambda e: e["resolution_H_over_dx"])
    checks = {}
    for key in ("A_legacy", "A_same_row", "A_fit_full", "A_fit_mid"):
        checks[key] = monotonicity(series, key)
    pairs = odd_even_pairs(series)
    dump({"checks": checks, "odd_even_pairs": pairs,
          "wall_clock_s": sum(e["wall_clock_s"] for e in series)})
    print("[centreline-sampling] SUMMARY " + json.dumps(
        {"series": series, "checks": checks, "odd_even_pairs": pairs}))
    print(f"[centreline-sampling] wrote {args.out / 'centreline_sampling.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())