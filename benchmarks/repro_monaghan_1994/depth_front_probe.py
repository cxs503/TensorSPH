#!/usr/bin/env python3
"""Does the graded front have a resolution-independent definition?

Why this probe exists
---------------------
``results/summary.json`` grades the M&M front trajectory through
``front_position`` = ``max`` of every particle's ``x``.  The merged audit
(``docs/audit_repro_monaghan1994_m16936.md``, F4) tested whether that maximum is
decided by a stray particle and falsified it - but only at dx = 4 cm and 2 cm,
where the cloud has 50 and 200 particles, so ``p99`` *is* essentially ``max``
(0.5 % of 200 particles is one particle).  The 1 cm level was left unprobed.

Re-running the ladder at all three resolutions (this probe) shows the reading
choice is not cosmetic at the finest level:

    dx = 1 cm, 800 particles:  max -> 4.38 %   p99 -> 1.19 %   (of amplitude)

i.e. a factor 3.7 between two defensible readings of the same simulation.  A
quantile is, however, still an arbitrary, resolution-dependent rule (``p99`` of
800 particles is 8 particles deep, of 50 particles it is half a particle).  So
this probe also evaluates a rule that is *stated in physical units* rather than
in particle counts:

    the front is the furthest column of width ``dx`` whose water depth - the
    number of particles in that column times ``dx`` (each particle carries
    ``dx**2`` of area in 2-D) - reaches a threshold ``h`` given in *centimetres*

which is a property of the free surface, not of the discretisation.  The ladder
of every candidate reading is then reported against the audited transcription,
so the case's headline number can be quoted with the estimator it belongs to.

Nothing in ``src/`` is touched: the module-level ``front_position`` the solver
calls is rebound for the duration of each run.

Usage::

    python benchmarks/repro_monaghan_1994/depth_front_probe.py --dx 0.04,0.02,0.01
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for candidate in (str(ROOT), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import tensorsph.dam_break as db  # noqa: E402

RESULTS = HERE / "results"
WINDOW = (0.5, 3.0)
N_SAMPLES = 161
#: depth thresholds as *fractions of the initial column height* H = 2a = 0.4 m.
#: The first version of this probe used absolute thresholds (0.5/1/2 cm) and they
#: were wrong by construction: once the column collapses, the layer over the tank
#: is ~A/W = (0.2*0.4)/2 = 4 cm deep, so every absolute threshold <= 2 cm is met
#: almost everywhere and the "front" is the far wall (measured bias +3.9 Z/a).
#: Stating the threshold relative to H is the resolution-independent scale of the
#: problem, and it keeps the rule a statement about the free surface.
DEPTH_FRACTIONS = (0.05, 0.10, 0.20)


def _interp(xs, ys, xq):
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0:
        return ys[lo]
    w = (xq - xs[lo]) / span
    return ys[lo] * (1.0 - w) + ys[hi] * w


def compare(tau, za, ref) -> dict:
    lo = max(WINDOW[0], min(tau), min(ref["tau"]))
    hi = min(WINDOW[1], max(tau), max(ref["tau"]))
    if not hi > lo:
        return {"n_samples": 0}
    grid = [lo + (hi - lo) * i / (N_SAMPLES - 1) for i in range(N_SAMPLES)]
    d, r = [], []
    for g in grid:
        s, e = _interp(tau, za, g), _interp(ref["tau"], ref["za"], g)
        d.append(s - e)
        r.append(e)
    amp = max(r)
    d = np.asarray(d)
    return {
        "n_samples": len(d),
        "amplitude": amp,
        "mae_over_amplitude": float(np.abs(d).mean() / amp),
        "bias": float(d.mean()),
        "max_abs_over_amplitude": float(np.abs(d).max() / amp),
    }


def column_depth(x: np.ndarray, dx: float):
    """Depth of every column of width ``dx``: ``count * dx`` (each particle = dx^2 area)."""
    if x.size == 0:
        return 0.0, np.zeros(1, dtype=np.int64)
    lo = int(np.floor(x.min() / dx))
    idx = np.floor(x / dx).astype(np.int64) - lo
    return lo, np.bincount(idx)


def depth_front_from(counts: np.ndarray, lo: int, dx: float, h: float) -> float | None:
    """Furthest column whose depth reaches ``h``; ``None`` if the rule cannot fire.

    ``None`` is the honest answer when no column reaches the threshold: returning
    a number there would silently shorten the trajectory and look like a front.
    """
    ok = np.nonzero(counts * dx >= h)[0]
    if ok.size == 0:
        return None
    return float((lo + ok[-1] + 0.5) * dx)


def run_case(dx: float, seconds: float, c0: float) -> dict:
    cfg = db.DamBreakConfig(
        dx=dx, pool_width=2.0, column_width=0.2, sound_speed=c0,
        max_courant=0.3, device="cpu", wall_friction=0.0,
        density_diffusion=0.0, adaptive_dt=False)
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))

    x0 = cfg.column_width  # initial column edge: the case's Z/a origin
    a = cfg.column_width
    H = cfg.column_height  # depth scale the threshold fractions refer to
    readings: list[dict] = []
    original = db.front_position

    def wrapped(pos, cfg_):
        value = original(pos, cfg_)
        x = pos[:, 0].detach().cpu().numpy()
        lo, counts = column_depth(x, dx)
        rec = {"x_max": value,
               "p995": float(np.quantile(x, 0.995)),
               "p99": float(np.quantile(x, 0.99)),
               "p95": float(np.quantile(x, 0.95)),
               "_column_origin_lo": lo,
               "_column_counts": counts.tolist()}
        for frac in DEPTH_FRACTIONS:
            rec[f"depth_{frac*100:g}%H"] = depth_front_from(counts, lo, dx, frac * H)
        readings.append(rec)
        return value

    db.front_position = wrapped
    try:
        clock = time.perf_counter()
        run = db.run_case(cfg)
        wall = time.perf_counter() - clock
    finally:
        db.front_position = original

    scale = math.sqrt(2.0 * cfg.gravity / a)
    tau = [t * scale for t in run.times]
    n = min(len(tau), len(readings))
    tau, readings = tau[:n], readings[:n]
    scale_x = (1.0 / a)
    names = [k for k in readings[0] if not k.startswith("_")]
    # a threshold that never fires leaves a hole in the trajectory; keep the hole
    # explicit so the metric cannot silently compare against a shortened series
    for n in names:
        for r in readings:
            if r[n] is None:
                r[n] = float("nan")
    return {
        "case": f"dx{dx:g}_c025",
        "dx": dx,
        "particles": int(run.final_positions.shape[0]),
        "wall_clock_s": wall,
        "max_density_compression": run.max_density_compression,
        "tau": tau,
        # every reading converted to the case's Z/a convention (front - a)/a
        "readings": {k: [(r[k] * scale_x) - 1.0 for r in readings] for k in names},
        "depth_profiles": [{"column_origin": r["_column_origin_lo"] * dx,
                            "counts": r["_column_counts"]} for r in readings],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", default="0.04,0.02,0.01")
    ap.add_argument("--seconds", type=float, default=0.34)
    ap.add_argument("--c0", type=float, default=25.0)
    ap.add_argument("--check-committed", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "depth_front_probe.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import run as case_run  # noqa: E402

    tables = case_run.references()
    ref = tables["crosscheck"]

    cases = []
    for dx in [float(x) for x in args.dx.split(",") if x.strip()]:
        rec = run_case(dx, args.seconds, args.c0)
        rec["errors"] = {name: compare(rec["tau"], za, ref)
                         for name, za in rec["readings"].items()}
        if args.check_committed:
            path = RESULTS / f"trajectory_{rec['case']}.json"
            if path.exists():
                committed = json.loads(path.read_text())
                za = np.array(rec["readings"]["x_max"])
                cza = np.array(committed["za"])
                m = min(za.size, cza.size)
                rec["committed_diff"] = {"n_compared": int(m),
                                         "max_abs_za_diff": float(np.max(np.abs(za[:m] - cza[:m])))}
        cases.append(rec)
        print(f"{rec['case']}: {rec['wall_clock_s']:.1f} s, "
              f"max|rho/rho0-1|={rec['max_density_compression']:.4f}", flush=True)

    ladder = {}
    for name in cases[0]["readings"]:
        ladder[name] = [{"dx": c["dx"], "mae_over_amplitude": c["errors"][name]["mae_over_amplitude"],
                         "bias": c["errors"][name]["bias"]} for c in cases]
        maes = [e["mae_over_amplitude"] for e in ladder[name]]
        ladder[name] = {**{"entries": ladder[name],
                           "monotone_nonincreasing": all(b <= a * 1.10 for a, b in zip(maes, maes[1:])),
                           "finest_mae_over_amplitude": maes[-1],
                           "finest_bias": ladder[name][-1]["bias"],
                           "spread_over_ladder": max(maes) - min(maes)}}

    payload = {
        "probe": "front-reading sensitivity for REPRO-SPH-MONAGHAN1994",
        "anchor": "REPRO-SPH-MONAGHAN1994",
        "protocol": {"window": list(WINDOW), "n_samples": N_SAMPLES, "seconds": args.seconds,
                     "c0": args.c0,
                     "depth_threshold_fractions_of_column_height": list(DEPTH_FRACTIONS),
                     "depth_thresholds_m": [f * 0.4 for f in DEPTH_FRACTIONS],
                     "reference": "benchmarks/dam_break/martin_moyce_1952_crosscheck.json",
                     "front_convention": "Z/a = (x_front - a)/a, a = column width"},
        "references": {"crosscheck": {"path": ref.get("path")}},
        "cases": cases,
        "ladder": ladder,
        "reading_spread_at_finest": {
            "min_mae_over_amplitude": min(v["finest_mae_over_amplitude"] for v in ladder.values()),
            "max_mae_over_amplitude": max(v["finest_mae_over_amplitude"] for v in ladder.values()),
            "readings": sorted(ladder, key=lambda k: ladder[k]["finest_mae_over_amplitude"]),
        },
        "gate_contact": False,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    for name, v in sorted(ladder.items(), key=lambda kv: kv[1]["finest_mae_over_amplitude"]):
        print(f"  {name:12s} finest MAE/amp = {v['finest_mae_over_amplitude']*100:5.2f} %"
              f"  bias = {v['finest_bias']:+.3f}  monotone = {v['monotone_nonincreasing']}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())