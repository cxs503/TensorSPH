"""Verdict for the rigid-precursor experiment: near-field vs far-field drag.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The coupled ``fsi2`` run of this repository reads ``Cd_mean = 0.052`` against the
Turek-Hron ``drag_mean = 3.2`` (``docs/xval_fsi2/VIV_PERIODIC_VERDICT.md``), a
-98 % deficit that the audits could not attribute: the surface load channel has
its own bias (``+31..36 %`` on a curved body at the coupled spacing,
``CYLINDER_LOAD_AT_BENCHMARK_SCALE.md``), and the fluid solve has never been
measured on a developed field at that spacing.  Two quantities therefore have
to be read on the *same* field:

``cd_total``     the wall-station (near-field) load the driver actually reports,
``cd_farfield``  the control-volume momentum budget between two planes
                 (``benchmarks/xval_fsi2/rigid_cfd_precursor.py``), an estimator
                 that never touches the body.

This module holds the pre-declared acceptance logic.  The thresholds below are
fixed *before* the run and are not tuned afterwards:

``NEAR_FAR_TOL = 0.15``
    The two estimators must agree within 15 % on the time average.  This is the
    only *comparative* question the experiment can answer: both numbers are
    measured on the same particle field, so a disagreement is a defect of one of
    the two reductions, whereas an agreement bounds the reduction error whatever
    the fluid does.

``ORDER_BAND = (1/3, 3)``
    Attribution gate: the far-field drag must be within one decade of the
    Turek-Hron ``drag_mean = 3.2``.  This is deliberately loose -- a frozen
    structure is not the FSI2 configuration -- and it is *not* an accuracy
    claim.  Passing it means "the -98 % deficit is not explained by the fluid
    half being unable to carry O(1) drag at this spacing"; failing it means the
    fluid solve is the defect.  No <15 % accuracy statement is made against
    Turek-Hron here; that needs a flexible beam and stays ``not_run``.

The unsteady control-volume term is not available (see
:mod:`tensorsph.farfield_force`), so only statistics over a stationary window
are interpretable, and single-sample far-field numbers are not.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = [
    "NEAR_FAR_TOL",
    "ORDER_BAND",
    "DRAG_REF",
    "STROUHAL_REF",
    "stationary_window",
    "block_stats",
    "near_far_gate",
    "order_gate",
    "strouhal_from_lift",
    "verdict",
]

#: Near-field vs far-field agreement tolerance on the time average.
NEAR_FAR_TOL = 0.15
#: Order-of-magnitude attribution band around the reference drag (see module doc).
ORDER_BAND = (1.0 / 3.0, 3.0)
#: Turek-Hron FSI2 drag mean (``tensorsph.fsi2_reference.BENCHMARK_FORCES``).
DRAG_REF = 3.2
#: Schaefer-Turek 2D-2 Strouhal number, only used when a clean cylinder wake is
#: measurable; with the beam attached the wake is blunt-body, so the value is
#: reported as a diagnostic and never gated.
STROUHAL_REF = 0.30


def block_stats(values: Sequence[float], n_blocks: int = 8) -> Dict[str, float]:
    """Block average: mean, standard error of the mean, and spread of blocks.

    A shedding signal is correlated in time, so the naive ``std/sqrt(n)`` of the
    samples understates the uncertainty.  Blocking over ~1 shedding cycle per
    block keeps the blocks close to independent.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {"mean": float("nan"), "sem": float("nan"),
                "spread": float("nan"), "n": int(v.size)}
    n_blocks = max(2, min(int(n_blocks), v.size))
    chunks = np.array_split(v, n_blocks)
    means = np.array([c.mean() for c in chunks])
    sem = means.std(ddof=1) / math.sqrt(means.size) if means.size > 1 else float("nan")
    return {"mean": float(v.mean()), "sem": float(sem),
            "spread": float(means.max() - means.min()), "n": int(v.size)}


def stationary_window(t: Sequence[float], values: Sequence[float],
                      frac: float = 0.5, n_blocks: int = 6,
                      tol: float = 0.30) -> Dict[str, Any]:
    """Trailing window that passes a block-mean stationarity test.

    The window is the trailing ``frac`` of the record.  It is accepted only if
    the block means agree within ``tol`` (relative to the window mean); the
    result always reports ``accepted`` and the numbers behind it, so a
    non-stationary record cannot be silently averaged.  The window is *not*
    shortened until it passes -- that would be a tuning loop over the data.
    """
    tt = np.asarray(list(t), dtype=float)
    vv = np.asarray(list(values), dtype=float)
    if tt.size != vv.size or tt.size < 4:
        raise ValueError("t and values must be equal length, >= 4 samples")
    start = int(round((1.0 - frac) * tt.size))
    tw, vw = tt[start:], vv[start:]
    stats = block_stats(vw, n_blocks=n_blocks)
    denom = abs(stats["mean"]) if abs(stats["mean"]) > 0.0 else float("nan")
    rel_spread = stats["spread"] / denom if denom == denom else float("nan")
    return {
        "t0": float(tw.min()), "t1": float(tw.max()), "n": int(tw.size),
        "mean": stats["mean"], "sem": stats["sem"],
        "block_spread_rel": float(rel_spread),
        "accepted": bool(rel_spread == rel_spread and rel_spread <= tol),
        "tol": float(tol),
    }


def near_far_gate(cd_near: float, cd_far: float,
                  tol: float = NEAR_FAR_TOL) -> Dict[str, Any]:
    """15 % agreement gate between the surface load and the momentum budget."""
    both = float(cd_near) + float(cd_far)
    denom = 0.5 * (abs(float(cd_near)) + abs(float(cd_far)))
    if not math.isfinite(both) or denom == 0.0:
        return {"rel_diff": float("nan"), "passed": False, "status": "not_run",
                "tol": tol}
    rel = (float(cd_far) - float(cd_near)) / denom
    ok = abs(rel) <= tol and float(cd_near) * float(cd_far) > 0.0
    return {"rel_diff": float(rel), "passed": bool(ok),
            "status": "pass" if ok else "fail", "tol": tol,
            "same_sign": bool(float(cd_near) * float(cd_far) > 0.0)}


def order_gate(cd_far: float, ref: float = DRAG_REF,
               band: tuple = ORDER_BAND) -> Dict[str, Any]:
    """Attribution gate: is the fluid half within one decade of ``ref``?"""
    if not math.isfinite(float(cd_far)) or cd_far <= 0.0:
        return {"ratio": float("nan"), "passed": False, "status": "not_run",
                "band": list(band)}
    ratio = float(cd_far) / float(ref)
    ok = band[0] <= ratio <= band[1]
    return {"ratio": ratio, "passed": bool(ok),
            "status": "fluid_capable" if ok else "fluid_defective",
            "band": list(band), "ref": float(ref)}


def strouhal_from_lift(t: Sequence[float], cl: Sequence[float],
                       u_mean: float, d: float,
                       min_cycles: int = 3) -> Dict[str, Any]:
    """Shedding frequency from upward zero crossings of the lift history.

    Reported as a diagnostic (``status='diagnostic'``) unless at least
    ``min_cycles`` crossings fall inside the window, in which case the period is
    the mean spacing and ``St = f d / u_mean``.
    """
    tt = np.asarray(list(t), dtype=float)
    cc = np.asarray(list(cl), dtype=float)
    cc = cc - cc.mean()
    if tt.size < 4:
        return {"st": float("nan"), "status": "not_run", "n_crossings": 0}
    cross = np.where((cc[:-1] <= 0.0) & (cc[1:] > 0.0))[0]
    if cross.size < min_cycles:
        return {"st": float("nan"), "status": "not_run",
                "n_crossings": int(cross.size)}
    # linear interpolation of the crossing instant
    times = []
    for i in cross:
        y0, y1 = cc[i], cc[i + 1]
        w = -y0 / (y1 - y0) if y1 != y0 else 0.5
        times.append(tt[i] + w * (tt[i + 1] - tt[i]))
    period = float(np.mean(np.diff(times)))
    st = (d / (period * u_mean)) if period > 0 else float("nan")
    return {"st": float(st), "period": period, "status": "diagnostic",
            "n_crossings": int(cross.size),
            "ref": STROUHAL_REF,
            "rel_diff_vs_ref": (float(st) / STROUHAL_REF - 1.0
                                if math.isfinite(st) else float("nan"))}


def verdict(artifact: Dict[str, Any], frac: float = 0.5) -> Dict[str, Any]:
    """Full verdict from a ``rigid_cfd_precursor.py`` artifact."""
    hist = artifact["history"]
    t = np.asarray(hist["t"], dtype=float)
    near = stationary_window(t, hist["cd_total"], frac=frac)
    far = stationary_window(t, hist["cd_farfield"], frac=frac)
    gate = near_far_gate(near["mean"], far["mean"])
    order = order_gate(far["mean"])
    sel = t >= near["t0"]
    st = strouhal_from_lift(t[sel], np.asarray(hist["cl_total"])[sel],
                            float(artifact["case"]["u_mean"]),
                            float(artifact["case"].get("cylinder_d", 0.1)))
    accepted = near["accepted"] and far["accepted"]
    return {
        "case": artifact["case"],
        "window": {"t0": near["t0"], "t1": near["t1"], "n": near["n"]},
        "window_stationary": accepted,
        "near_field": near, "far_field": far,
        "near_far_gate": gate, "order_gate": order, "strouhal": st,
        "status": ("judged" if accepted else "withheld"),
        "reason": ("" if accepted else
                   "stationary window rejected: block means still drifting"),
        "not_run": ["Cd against Turek-Hron <15 % (needs the flexible beam)",
                    "single-sample far-field drag (unsteady CV term absent)"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    art = json.loads(Path(args.artifact).read_text())
    v = verdict(art, frac=args.frac)
    text = json.dumps(v, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
