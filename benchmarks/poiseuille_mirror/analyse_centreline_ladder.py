#!/usr/bin/env python3
"""Re-grade the published Poiseuille ladder with a sampling-free gate A.

The ladder in ``out/adversarial_recheck.json`` (H/dx = 8, 12, 16, 24, 32) was
graded with the legacy centre-line metric, which compares the largest binned
row mean against the *continuum* centre-line value.  On an even lattice no row
sits on ``y = H/2``, so that metric carries the structural penalty
``1 - (dx/H)^2``.  This script recomputes, from the stored per-row profiles,
the sampling-free estimators (parabola fit at ``y = H/2``) for every rung and
reports how much of the published resolution trend was metric artefact.

Outputs ``out/centreline_ladder_analysis.json``, consumed by
``docs/case_poiseuille_centreline_metric.md``.

Usage:
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/analyse_centreline_ladder.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/poiseuille_mirror/out"

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
U_MAX = BODY_FORCE * HEIGHT ** 2 / (8.0 * VISCOSITY)


def fit_centre(rows: list[dict], lo: float, hi: float) -> float:
    """Least-squares parabola over ``lo <= y/H <= hi``, evaluated at H/2."""
    sel = [r for r in rows if lo <= r["y_over_H"] <= hi]
    if len(sel) < 4:
        return float("nan")
    ys = [r["y"] for r in sel]
    us = [r["u"] for r in sel]
    k = len(ys)
    sy = sum(ys)
    sy2 = sum(y * y for y in ys)
    sy3 = sum(y ** 3 for y in ys)
    sy4 = sum(y ** 4 for y in ys)
    su = sum(us)
    suy = sum(u * y for u, y in zip(us, ys))
    suy2 = sum(u * y * y for u, y in zip(us, ys))
    # 3x3 normal equations
    a = [[k, sy, sy2], [sy, sy2, sy3], [sy2, sy3, sy4]]
    b = [su, suy, suy2]
    # gaussian elimination
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, 3):
            f = a[r][i] / a[i][i]
            for c in range(i, 3):
                a[r][c] -= f * a[i][c]
            b[r] -= f * b[i]
    x = [0.0, 0.0, 0.0]
    for i in reversed(range(3)):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, 3))) / a[i][i]
    c0, c1, c2 = x
    return c0 + c1 * HEIGHT / 2 + c2 * (HEIGHT / 2) ** 2


def orders(points: list[tuple[float, float]]) -> dict:
    pts = [(n, abs(e)) for n, e in points if e == e and abs(e) > 1e-12]
    pairs = [[a[0], b[0], math.log(a[1] / b[1]) / math.log(b[0] / a[0])]
             for a, b in zip(pts, pts[1:])]
    if not pairs:
        return {"pairs": []}
    ps = [p[2] for p in pairs]
    return {"pairs": pairs, "p_mean": sum(ps) / len(ps),
            "p_last": pairs[-1][2],
            "monotone": all(abs(b[1]) <= abs(a[1]) for a, b in zip(pts, pts[1:]))}


def main() -> int:
    recheck = json.loads((OUT / "adversarial_recheck.json").read_text())
    ladder = []
    for entry in sorted(recheck["ladder"], key=lambda e: e["resolution_H_over_dx"]):
        n = entry["resolution_H_over_dx"]
        rows = entry["rows"]
        peak = max(rows, key=lambda r: r["u"])
        fit_all = fit_centre(rows, 0.0, 1.0) / U_MAX - 1.0
        fit_mid = fit_centre(rows, 0.25, 0.75) / U_MAX - 1.0
        bias = 1.0 - (1.0 / n) ** 2
        legacy = entry["u_max_error_pct"] / 100.0
        ladder.append({
            "resolution_H_over_dx": n,
            "row_on_centreline": bool(abs(peak["y_over_H"] - 0.5) < 1e-9),
            "nearest_row_y_over_H": peak["y_over_H"],
            "A_legacy": legacy,
            "A_fit_full": fit_all,
            "A_fit_mid": fit_mid,
            "bias_pred_factor": bias,
            "A_legacy_from_fit": (1.0 + fit_all) * bias - 1.0,
            "residual_after_correction": legacy - ((1.0 + fit_all) * bias - 1.0),
            "profile_rms_interior_pct": entry["profile_rms_interior_pct"],
            "profile_rms_all_pct": entry["profile_rms_all_pct"],
            "tau_bottom_error_pct": entry["tau_bottom_error_pct"],
        })

    # odd-parity companion rungs from the sampling study, if present
    comp = []
    sampling_path = OUT / "centreline_sampling.json"
    if sampling_path.exists():
        comp = json.loads(sampling_path.read_text())["series"]

    payload = {
        "case": "plane_poiseuille_channel_mirror_wall_ladder_regrade",
        "anchor": "CASE-POISEUILLE-SPH",
        "source": "out/adversarial_recheck.json",
        "u_max_reference": U_MAX,
        "ladder": ladder,
        "order_fits": {
            "A_legacy": orders([(e["resolution_H_over_dx"], e["A_legacy"])
                                for e in ladder]),
            "A_fit_full": orders([(e["resolution_H_over_dx"], e["A_fit_full"])
                                  for e in ladder]),
            "A_fit_mid": orders([(e["resolution_H_over_dx"], e["A_fit_mid"])
                                 for e in ladder]),
            "profile_rms_interior_pct": orders(
                [(e["resolution_H_over_dx"], e["profile_rms_interior_pct"])
                 for e in ladder]),
        },
        "sampling_study_resolutions": [e["resolution_H_over_dx"] for e in comp],
        "worst_sampling_penalty_pct": 100 * max(
            (e["bias_pred_factor"] - 1.0 for e in ladder), key=abs),
    }
    (OUT / "centreline_ladder_analysis.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"ladder": [
        {k: (round(v, 5) if isinstance(v, float) else v) for k, v in e.items()}
        for e in ladder], "order_fits": payload["order_fits"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())