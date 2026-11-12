#!/usr/bin/env python3
"""Is the M&M front trajectory a robust observable, or a single-particle maximum?

Why this probe exists
---------------------
The committed Monaghan-1994 reproduction reads the leading edge as
``front_position`` = ``max`` of every particle's ``x`` (:func:`tensorsph.dam_break.
front_position`).  A maximum over a particle cloud has no convergence guarantee:
one ejecta particle sets the number, so a finer, more energetic run can *raise*
the measured front without the wave front itself having moved.

The committed ladder (``results/summary.json``) is non-monotone against the
independent transcription - 4.67 % (dx = 4 cm) -> 1.65 % (2 cm) -> 4.38 %
(1 cm) - and the document reports that as a failed gate without saying which
of the two mechanisms it is:

  H1 (*estimator*): the fine runs contain a stray particle, so ``max`` is not
      the water front; a high quantile of ``x`` gives a monotone ladder.
  H2 (*physics*):  ``max`` and every high quantile agree, so the front really
      does run ahead of the digitised table by a resolution-independent amount
      (i.e. the residual is not discretisation error).

Method: run the shipped case unchanged (same ``DamBreakConfig``, same output
cadence) with the module-level ``front_position`` wrapped so that, at every
committed output time, the whole distribution of particle ``x`` is recorded
(``max``, 99.5/99/95 % quantiles, and the count of particles more than ``2 dx``
ahead of the 99.5 % quantile).  Nothing in ``src/`` is touched; the wrapper is
installed at run time by rebinding the module global the solver calls.

Every number in the emitted JSON is measured here; the document that quotes it
(``docs/audit_repro_monaghan1994_front_estimator.md``) reads it back.

Usage
-----
    python benchmarks/repro_monaghan_1994/front_estimator_probe.py \
        --dx 0.04,0.02 --seconds 0.34
    python benchmarks/repro_monaghan_1994/front_estimator_probe.py \
        --dx 0.01 --seconds 0.34 --check-committed
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorsph.dam_break as db  # noqa: E402

OUT = HERE / "results" / "front_estimator_probe.json"
COMMITTED = HERE / "results"

#: Window and metric of the case's pre-registered protocol (``run.py``), reused
#: verbatim so the probe's errors are comparable with ``results/summary.json``.
WINDOW = (0.5, 3.0)
N_SAMPLES = 161


def _interp(xs, ys, xq):
    if not xs:
        return None
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


def compare(tau, za, ref):
    """Independent re-implementation of the case's T1 metric (no import)."""
    lo = max(WINDOW[0], min(tau), min(ref["tau"]))
    hi = min(WINDOW[1], max(tau), max(ref["tau"]))
    if not hi > lo:
        return {"n_samples": 0}
    grid = [lo + (hi - lo) * i / (N_SAMPLES - 1) for i in range(N_SAMPLES)]
    deltas, refs = [], []
    for g in grid:
        s = _interp(tau, za, g)
        e = _interp(ref["tau"], ref["za"], g)
        if s is None or e is None:
            continue
        deltas.append(s - e)
        refs.append(e)
    amp = max(refs)
    mae = float(np.mean(np.abs(deltas)))
    return {
        "n_samples": len(deltas),
        "window": [grid[0], grid[-1]],
        "amplitude": amp,
        "mae": mae,
        "bias": float(np.mean(deltas)),
        "max_abs": float(np.max(np.abs(deltas))),
        "mae_over_amplitude": mae / amp,
    }


def run_case(dx: float, seconds: float, c0: float) -> dict:
    cfg = db.DamBreakConfig(
        dx=dx, pool_width=2.0, column_width=0.2, sound_speed=c0,
        max_courant=0.3, device="cpu", wall_friction=0.0,
        density_diffusion=0.0, adaptive_dt=False)
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))

    probe: list[dict] = []
    original = db.front_position

    def wrapped(pos, cfg_):
        value = original(pos, cfg_)
        x = pos[:, 0].detach().cpu().numpy()
        p995 = float(np.quantile(x, 0.995))
        probe.append({
            "x_max": value,
            "p995": p995,
            "p99": float(np.quantile(x, 0.99)),
            "p95": float(np.quantile(x, 0.95)),
            "n_particles": int(x.size),
            "n_ahead_of_p995_by_2dx": int((x > p995 + 2.0 * dx).sum()),
        })
        return value

    db.front_position = wrapped
    try:
        clock = time.perf_counter()
        run = db.run_case(cfg)
        wall = time.perf_counter() - clock
    finally:
        db.front_position = original

    a = cfg.column_width
    scale = math.sqrt(2.0 * cfg.gravity / a)
    tau = [t * scale for t in run.times]
    n = min(len(tau), len(probe))
    tau, probe = tau[:n], probe[:n]
    estimators = {
        name: [(p[name] - a) / a for p in probe]
        for name in ("x_max", "p995", "p99", "p95")
    }
    return {
        "case": f"dx{dx:g}_c0{c0:g}",
        "dx": dx,
        "sound_speed": c0,
        "particles": int(run.final_positions.shape[0]),
        "dt": cfg.dt,
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall,
        "max_density_compression": run.max_density_compression,
        "n_output_times": n,
        "tau": tau,
        "estimators": estimators,
        "probe": probe,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", default="0.04,0.02")
    ap.add_argument("--seconds", type=float, default=0.34)
    ap.add_argument("--c0", type=float, default=25.0)
    ap.add_argument("--check-committed", action="store_true",
                    help="diff the re-run x_max trajectory against results/trajectory_*.json")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    refs = json.loads((ROOT / "benchmarks" / "dam_break" / "reference_audit.json").read_text())
    del refs  # audit only flags usability; the table itself is read below
    sys.path.insert(0, str(HERE))
    import run as case_run  # noqa: E402  (same loader the case uses)

    tables = case_run.references()

    cases = []
    for dx in [float(x) for x in args.dx.split(",") if x.strip()]:
        record = run_case(dx, args.seconds, args.c0)
        record["errors"] = {
            name: {ref: compare(record["tau"], za, tables[ref]) for ref in tables}
            for name, za in record["estimators"].items()
        }
        if args.check_committed:
            path = COMMITTED / f"trajectory_{record['case']}.json"
            if path.exists():
                committed = json.loads(path.read_text())
                za = np.array(record["estimators"]["x_max"])
                cza = np.array(committed["za"])
                m = min(za.size, cza.size)
                record["committed_diff"] = {
                    "path": str(path.relative_to(ROOT)),
                    "n_compared": int(m),
                    "max_abs_za_diff": float(np.max(np.abs(za[:m] - cza[:m]))),
                    "committed_mae_over_amplitude": compare(
                        committed["tau"], committed["za"], tables["crosscheck"]
                    ).get("mae_over_amplitude"),
                }
        cases.append(record)
        print(f"{record['case']}: {record['wall_clock_s']:.1f} s, "
              f"max|rho/rho0-1|={record['max_density_compression']:.4f}", flush=True)

    payload = {
        "probe": "front estimator sensitivity for REPRO-SPH-MONAGHAN1994",
        "protocol": {"window": list(WINDOW), "n_samples": N_SAMPLES,
                     "seconds": args.seconds, "c0": args.c0,
                     "estimators": ["x_max", "p995", "p99", "p95"]},
        "references": {k: {"path": v.get("path"), "usable": v.get("usable")} for k, v in tables.items()},
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())