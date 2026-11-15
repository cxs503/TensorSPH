"""Matched-bandwidth re-reading of the wedge-entry peak-pressure ladder (m21677).

What was wrong with the committed reading
-----------------------------------------
The case grades ``Cp_max = p_max / (rho V^2 / 2)`` of a wedge (beta = 30 deg,
constant velocity) against a 15 % gate.  The estimator that produced every
committed number pools the self-similar frames and takes the maximum of a
*binned* profile ``p/q`` vs ``tau = |x| / c``.  The bin count is drawn from the
fixed list ``BIN_GRID = (6, 8, ..., 30)`` in ``ensemble_peak.py``.

A bin is a measurement cell, so the bin width has to be tied to the resolution:
one bin per surface particle spacing means
``n_bins = TAU_MAX * c_end / dx``.  With ``c_end = 50 mm`` that is

    dx = 4.00 mm ->  15 bins      dx = 1.00 mm ->  60 bins
    dx = 2.00 mm ->  30 bins      dx = 0.50 mm -> 120 bins

``BIN_GRID`` stops at 30, so the two finest grids could never be read at a
matched bandwidth: they were silently smoothed over 1.7 and 3.3 cells instead of
one, and the resulting operator bias was then absorbed by the acceptance gate.
The bias is not small -- the analytic chain below measures it.

The consistency of the estimator is checkable *without the SPH data*: push the
Wagner field (the analytic solution of the same idealisation) through the same
sampling operator (committed in ``estimator_bias.py``) and look at the operator
bias ``b(dx) = peak_operator(dx) / Cp_max(Wagner point) - 1`` as a function of
the bin width in cells.  At matched bandwidth ``b`` must converge to 0 as the
grid refines; at the truncated bandwidth of ``BIN_GRID`` it cannot, because the
smoothing stays at 3.3 cells.

Predeclared reading of this study (fixed before the runs)
---------------------------------------------------------
* **Estimator**: one bin per surface cell, ``n_bins = round(TAU_MAX c_end/dx)``;
  sample gate and ``tau <= 1`` region unchanged from ``ensemble_peak.py``.
* **Uncertainty**: ``U_g = max(frame bootstrap half-width, bandwidth
  sensitivity)``, the second being ``|peak(1 cell) - peak(1/2 cell)| /
  peak(1 cell)`` -- i.e. the reading is only trusted as far as it is insensitive
  to halving the cell.
* **Judgeable**: ``U_g <= 15 %``.
* **Residual**: ``r_g = obs_g / synth_g - 1`` (the same-operator residual of
  ``ladder_closure.py``), where ``synth_g`` is Wagner through the *matched*
  operator, so estimator bias is removed from the physics residual.
* **Verdict**: exactly the rule of ``ladder_closure.py`` -- the two finest
  judgeable grids must share the sign of ``r`` and satisfy ``|r| <= 15 %`` for a
  PASS; a sign flip or fewer than two judgeable grids is NOT_DECIDABLE.

The mission's literal metric owner (Zhao & Faltinsen 1993 composite solution,
``COM`` = 6.92 in q units, read from ``refs.json``) is reported next to this,
flagged as *reference side*: it is never used to choose the estimator, only to
state the acceptance residual.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/bandwidth_matched.py \
        --artifacts docs/artifacts/case_sph_wedge_entry_2d_bandwidth \
        --out docs/artifacts/case_sph_wedge_entry_2d_bandwidth
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import estimator_bias as eb  # noqa: E402
import literature_refs as lr  # noqa: E402

GATE = 0.15
GRID_TAGS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")
FRAMES = "61"
LEGACY_BIN_MAX = 30          # the cap that caused the truncation


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and v == v and math.isinf(v) is False


def operator_bias_curve(dx: float, t_end: float, n_cells: float,
                        point: float, n_points: int = 9) -> list[dict]:
    """Analytic field through the estimator, as a function of the bin width.

    Purely reference side: no SPH data is read here, so this curve is a property
    of the measurement operator, not of the run.
    """
    chain = eb.SynthChain(dx, eb.FRAMES_TOTAL, t_end, True)
    taus: list[float] = []
    cps: list[float] = []
    for tau, cp in chain.frames:
        taus.append(tau)
        cps.append(cp)
    import numpy as np
    T = np.concatenate(taus)
    P = np.concatenate(cps)
    out = []
    for f in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0][:n_points]:
        n_bins = max(4, int(round(n_cells / f)))
        peak, tau_at = eb.estimator_peak(T, P, n_bins)
        out.append({"cells_per_bin": f, "n_bins": n_bins,
                    "cp_peak_operator": peak, "tau_at_peak": tau_at,
                    "bias_vs_point": peak / point - 1.0 if _finite(peak) else float("nan")})
    return out


def grid_record(tag: str, art_dir: Path, legacy_dir: Path, point: float,
                beta_deg: float) -> dict:
    path = Path(art_dir) / f"ensemble_{tag}_c0v20.json"
    art = json.loads(path.read_text())
    dx = art["config"]["dx"]
    t_end = art["t_end"]
    matched = art.get("matched_bandwidth")
    if not matched:
        raise SystemExit(f"{path} has no matched_bandwidth block -- rerun "
                         "ensemble_peak.py (m21677 version) for this grid")
    n_cells = matched["n_cells"]
    n_m = int(matched["n_bins_matched"])
    sweep = {int(r["n_bins"]): r for r in matched["bandwidth_sweep"]}
    peak = matched["peak"]
    half = sweep.get(max(4, int(round(n_cells / 2.0))))
    bw_rel = (abs(peak - half["cp_peak_gated"]) / abs(peak)
              if half and _finite(half["cp_peak_gated"]) else float("nan"))
    boot = matched["bootstrap"]
    u_boot = boot.get("rel_half_width", float("nan"))
    u = max(v for v in (u_boot, bw_rel) if _finite(v)) if any(
        _finite(v) for v in (u_boot, bw_rel)) else float("nan")

    # same-operator truth at the *matched* bandwidth (analytic field, no SPH)
    chain = eb.SynthChain(dx, eb.FRAMES_TOTAL, t_end, True)
    import numpy as np
    T = np.concatenate([f[0] for f in chain.frames])
    P = np.concatenate([f[1] for f in chain.frames])
    synth, synth_tau = eb.estimator_peak(T, P, n_m)
    residual = (peak - synth) / synth if _finite(peak) and _finite(synth) and synth else float("nan")

    # legacy (committed) reading, for the record -- same grid, BIN_GRID band
    legacy_path = Path(legacy_dir) / f"ensemble_{tag}_c0v20.json"
    legacy = {}
    if legacy_path.exists():
        lart = json.loads(legacy_path.read_text())
        lrow = lart["by_frames"][FRAMES]
        legacy = {"resolution_band_peak": lrow["resolution_band"]["peak_median"],
                  "resolution_band_spread_rel": lrow["resolution_band"]["spread_rel"],
                  "resolution_band_n_bins_used": lrow["resolution_band"]["n_bins_used"],
                  "jackknife_rel": lrow.get("jackknife_rel")}

    designated = lr.designated_cp_max(beta_deg)
    exact = lr.cp_max_model(lr.EXACT_MODEL, beta_deg)
    return {
        "tag": tag, "dx_m": dx, "n_cells": n_cells, "n_bins_matched": n_m,
        "cells_per_bin_committed": n_cells / LEGACY_BIN_MAX,
        "truncated_by_bin_grid": bool(n_cells > LEGACY_BIN_MAX),
        "obs_peak_matched": peak,
        "obs_bootstrap": boot,
        "obs_bandwidth_rel": bw_rel,
        "uncertainty": u,
        "synth_peak_matched": synth,
        "synth_tau_at_peak": synth_tau,
        "operator_bias_vs_point": (synth / point - 1.0) if _finite(synth) else float("nan"),
        "residual": residual,
        "judgeable": bool(_finite(u) and u <= GATE and _finite(residual)),
        "err_vs_designated_rel": (peak / designated - 1.0) if _finite(peak) else float("nan"),
        "err_vs_exact_rel": (peak / exact - 1.0) if _finite(peak) else float("nan"),
        "legacy": legacy,
        "bandwidth_sweep": matched["bandwidth_sweep"],
    }


def verdict(rows: list[dict], gate: float = GATE) -> dict:
    judgeable = [r for r in rows if r["judgeable"]]
    if len(judgeable) < 2:
        return {"verdict": "NOT_DECIDABLE",
                "reason": f"only {len(judgeable)} judgeable grid(s) "
                          f"(U > {gate:.0%})",
                "grids_judgeable": [r["tag"] for r in judgeable]}
    fine, coarse = judgeable[-1], judgeable[-2]
    signs = {fine["residual"] < 0.0, coarse["residual"] < 0.0}
    if len(signs) > 1:
        return {"verdict": "NOT_DECIDABLE",
                "reason": f"sign flip between {coarse['tag']} "
                          f"({coarse['residual']:+.1%}) and {fine['tag']} "
                          f"({fine['residual']:+.1%})",
                "grids_judgeable": [r["tag"] for r in judgeable]}
    worst = max(judgeable, key=lambda r: abs(r["residual"]))
    ok = all(abs(r["residual"]) <= gate for r in judgeable)
    return {"verdict": "PASS" if ok else
            ("FAIL_above" if worst["residual"] > 0 else "FAIL_below"),
            "reason": (f"two finest judgeable grids ({coarse['tag']}, "
                       f"{fine['tag']}) share sign; worst |r| = "
                       f"{abs(worst['residual']):.1%} at {worst['tag']}"),
            "grids_judgeable": [r["tag"] for r in judgeable],
            "paired": [coarse["tag"], fine["tag"]]}


def bias_order(rows: list[dict]) -> dict:
    """Observed order of the operator bias on the two finest grids."""
    bias = [(r["dx_m"], abs(r["operator_bias_vs_point"])) for r in rows
            if _finite(r["operator_bias_vs_point"])]
    bias.sort(key=lambda p: -p[0])
    if len(bias) < 2:
        return {"available": False}
    d1, b1 = bias[-2]
    d2, b2 = bias[-1]
    order = (math.log(b1 / b2) / math.log(d1 / d2)
             if b1 > 0 and b2 > 0 and d1 != d2 else float("nan"))
    return {"available": True, "finest": [d2, b2], "next": [d1, b1],
            "observed_order": order,
            "all": [{"dx_m": d, "abs_bias": b} for d, b in bias]}


def build(art_dir: Path, legacy_dir: Path, out_dir: Path) -> dict:
    refs = lr.load_refs()
    beta_deg = 30.0
    point = eb.wagner_peak_over_q(math.radians(beta_deg))
    rows = [grid_record(tag, art_dir, legacy_dir, point, beta_deg)
            for tag in GRID_TAGS]
    rows.sort(key=lambda r: -r["dx_m"])
    curves = {}
    for row in rows:
        tag = row["tag"]
        art = json.loads((Path(art_dir) / f"ensemble_{tag}_c0v20.json").read_text())
        curves[tag] = operator_bias_curve(row["dx_m"], art["t_end"],
                                         row["n_cells"], point)
    designated = lr.designated_cp_max(beta_deg)
    report = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/bandwidth_matched.py",
        "design": {"gate": GATE, "frames": FRAMES,
                   "estimator": "one bin per surface cell, "
                                "n_bins = round(TAU_MAX c_end / dx)",
                   "uncertainty": "max(frame bootstrap, bandwidth sensitivity)",
                   "residual": "obs_matched / synth_matched - 1",
                   "legacy_bin_grid_max": LEGACY_BIN_MAX,
                   "verdict_rule": "ladder_closure.py rule, unchanged"},
        "wagner_cp_max_point": point,
        "designated_model": {"model": lr.DESIGNATED_MODEL, "cp_max_q": designated},
        "exact_model": {"model": lr.EXACT_MODEL,
                        "cp_max_q": lr.cp_max_model(lr.EXACT_MODEL, beta_deg)},
        "reference_half_spread_rel": lr.band(beta_deg)["rel_half_spread"],
        "grids": rows,
        "verdict": verdict(rows),
        "operator_bias_order": bias_order(rows),
        "operator_bias_vs_bandwidth": curves,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bandwidth_matched.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(ROOT / "docs/artifacts/"
                                               "case_sph_wedge_entry_2d_bandwidth"))
    ap.add_argument("--legacy", default=str(ROOT / "docs/artifacts/"
                                            "case_sph_wedge_entry_2d_ensemble"))
    ap.add_argument("--out", default=str(ROOT / "docs/artifacts/"
                                         "case_sph_wedge_entry_2d_bandwidth"))
    args = ap.parse_args(argv)
    rep = build(Path(args.artifacts), Path(args.legacy), Path(args.out))
    print("[bandwidth] matched-bandwidth ladder")
    for r in rep["grids"]:
        print(f"  {r['tag']:8s} cells={r['n_cells']:6.1f} cells/bin(committed)="
              f"{r['cells_per_bin_committed']:.2f}  obs={r['obs_peak_matched']:.3f}"
              f"  synth={r['synth_peak_matched']:.3f}"
              f"  r={r['residual']:+.3f}  U={r['uncertainty']:.3f}"
              f"  judgeable={r['judgeable']}")
    print(f"  verdict: {rep['verdict']['verdict']} -- {rep['verdict']['reason']}")
    ob = rep["operator_bias_order"]
    if ob.get("available"):
        print(f"  operator bias order (two finest): {ob['observed_order']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())