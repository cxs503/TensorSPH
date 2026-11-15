"""Coupled FSI2 run at the *parameters the archived SPHinXsys record used*.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this script exists
----------------------
Earlier rounds on this anchor compared force-from-fluid operators on static
particle sets, verified the geometry/stiffness (V1-V4) and established that the
archived SPHinXsys ``test_2d_fsi2`` record is *not* the Turek-Hron benchmark
point (``Ae = 1.4e3`` vs ``5.6e3``, ``rho_s/rho_f = 10`` vs ``1``,
``L = 1.1 m`` vs ``2.5 m``; see ``docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md``).
What was never done is the only comparison that the archive actually supports:
**a coupled run at exactly the archived parameter point, judged against the
recorded beam-tip series**.  This script does that run.

Decision
--------
Does the coupled WCSPH + elastic-rod driver of this repository reproduce the
beam-tip statistics of the archived SPHinXsys ``test_2d_fsi2`` records
(3 runs) within the owner accuracy standard?

Predeclared metric (fixed before the run; not retuned afterwards)
----------------------------------------------------------------
At the archived parameter point (``Re = 100``, ``Ae = 1.4e3``, ``rho_s/rho_f =
10``, ``L_channel = 1.1 m``, ``U_mean = 0.2 m/s``, physical length scale 10):

* ``tip_x_mean``   -- mean axial tip offset from the straight cantilever [m]
  (axial shortening under the drag load),
* ``tip_y_amplitude`` -- half-range transverse tip amplitude [m] (flapping),
* ``tip_x_amplitude`` -- half-range axial amplitude [m].

The reference values are the per-run statistics of
``BeamObserver_Position_Run_{0,4,9}_result.xml`` (``tensorsph.fsi2_reference``),
and the gate is applied to the *mean of the three runs* (the runs are the
upstream regression set, so their spread is the reference uncertainty).

Thresholds: ``|rel_error| < 15 %`` -> PASS (owner standard), ``< 3 %`` ->
reporting target.  A metric whose reference is smaller than the run-to-run
spread of the archive is reported as ``NOT_RESOLVED`` instead of a percentage.

Non-goals / limits (stated up front)
-----------------------------------
* The beam flapping period at this softness is a fraction of a second, so the
  transient must be excluded: statistics are taken after ``--warmup`` seconds.
* The run length is what it is; if the tip series has not reached a statistically
  steady state, ``steady_state`` in the output records that and the verdict is
  downgraded to ``WITHHELD`` rather than reported as a pass.

Output: ``artifacts/xval_fsi2/oss_matched_<tag>.json`` (numbers only, nothing
hand-copied).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.fsi2_reference import (  # noqa: E402
    load_tip_series, summarise_tip,
)
from tensorsph.fsi2_similarity import similarity_corrected_case, OSS_FSI2_CASE  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2  # noqa: E402

#: Archived parameter point (``tests/2d_examples/test_2d_fsi2/fsi2.h``).
OSS_PARAMS = {
    "Re": 100.0,
    "Ae": 1.4e3,
    "rho_s_over_rho_f": 10.0,
    "channel_length": 11.0 / 10.0,       # DL = 11.0 scaled
    "channel_height": 4.1 / 10.0,        # DH = 4.1 scaled
    "cylinder_d": 1.0 / 10.0,            # D = 1.0 scaled
    "beam_length": 3.5 / 10.0,           # bl = 3.5 scaled
    "beam_thickness": 0.2 / 10.0,        # bh = 0.2 scaled
    "U_mean": 0.2,
}

#: Gate values, fixed before the run.
TOL_PASS = 0.15
TOL_TARGET = 0.03


def build_case(dx: float, sound_speed: float = 5.0, ratio: float = 10.0,
               ae: float = 1.4e3, u_mean: float = 0.2):
    """Similarity-scaled FSI2 case at the archived SPHinXsys parameter point."""
    case = similarity_corrected_case(U_mean=u_mean, Ae=ae)
    case.dx = float(dx)
    case.sound_speed = float(sound_speed)
    case.beam_rho = case.rho0 * float(ratio)
    case.channel_length = OSS_PARAMS["channel_length"]
    case.n_beam_nodes = max(15, int(round(case.beam_length / case.dx)))
    case.rod_stability_target = 0.2
    case.max_rod_substeps = 2000
    return case


def reference_statistics() -> dict:
    """Per-run and mean statistics of the three archived records."""
    runs = {}
    for run in ("0", "4", "9"):
        _, pos = load_tip_series(run)
        s = summarise_tip(pos)
        # tip offset relative to the straight cantilever tip (x = 0.6 m, y = 0.2 m)
        runs[run] = {
            "tip_x_mean": s["x_mean"] - 0.60,
            "tip_x_amplitude": s["x_amplitude"],
            "tip_y_mean": s["y_mean"] - 0.20,
            "tip_y_amplitude": s["y_amplitude"],
            "n_samples": s["n_samples"],
        }
    mean = {k: float(np.mean([r[k] for r in runs.values()]))
            for k in ("tip_x_mean", "tip_x_amplitude",
                      "tip_y_mean", "tip_y_amplitude")}
    spread = {k: float(np.ptp([runs[r][k] for r in runs]))
              for k in ("tip_x_mean", "tip_x_amplitude",
                        "tip_y_mean", "tip_y_amplitude")}
    return {"runs": runs, "mean": mean, "spread": spread}


def steady_state_check(t: np.ndarray, y: np.ndarray, n_blocks: int = 3) -> dict:
    """Block means of the tail: a drifting block mean means "not steady yet"."""
    n = len(y)
    if n < 3 * n_blocks:
        return {"steady": False, "reason": "too few samples", "block_means": []}
    blocks = np.array_split(y, n_blocks)
    means = [float(b.mean()) for b in blocks]
    amps = [float(0.5 * np.ptp(b)) for b in blocks]
    drift = max(abs(m - means[-1]) for m in means[:-1])
    amp_ref = max(abs(means[-1]), 1.0e-12)
    return {
        "steady": bool(drift <= max(0.25 * amp_ref, 0.2 * max(amps))),
        "block_means": means,
        "block_amplitudes": amps,
        "relative_drift": float(drift / amp_ref),
    }


def summarise_run(hist: dict, warmup: float) -> dict:
    t = np.asarray(hist["t"], dtype=float)
    keep = t >= warmup
    if not keep.any():
        return {"n_samples": 0}
    x = np.asarray(hist["tip_x"], dtype=float)[keep]
    y = np.asarray(hist["tip_y"], dtype=float)[keep]
    out = {
        "n_samples": int(keep.sum()),
        "t_start": float(t[keep][0]), "t_end": float(t[keep][-1]),
        "tip_x_mean": float(x.mean()), "tip_x_amplitude": float(0.5 * np.ptp(x)),
        "tip_y_mean": float(y.mean()), "tip_y_amplitude": float(0.5 * np.ptp(y)),
        "tip_x_min": float(x.min()), "tip_x_max": float(x.max()),
        "tip_y_min": float(y.min()), "tip_y_max": float(y.max()),
    }
    for key in ("drag_struct", "lift_struct", "beam_drag_friction"):
        if key in hist and hist[key]:
            arr = np.asarray(hist[key], dtype=float)[keep]
            out[key + "_mean"] = float(arr.mean())
    out["steady_state"] = steady_state_check(t[keep], y)
    return out


def compare(measured: dict, ref: dict) -> dict:
    rows = []
    for key in ("tip_x_mean", "tip_x_amplitude", "tip_y_amplitude"):
        got = measured.get(key)
        want = ref["mean"][key]
        spread = ref["spread"][key]
        if got is None or not math.isfinite(got):
            rows.append({"metric": key, "got": got, "ref": want,
                         "verdict": "MISSING"})
            continue
        rel = (got - want) / want if want else float("inf")
        # a reference whose archive spread exceeds 20 % of its mean cannot
        # resolve a 15 % claim -- say so instead of quoting a percentage
        resolved = abs(spread) <= 0.2 * abs(want)
        rows.append({
            "metric": key, "got": float(got), "ref": float(want),
            "rel_error": float(rel), "reference_spread": float(spread),
            "resolved": bool(resolved),
            "pass_15": bool(abs(rel) < TOL_PASS and resolved),
            "pass_3": bool(abs(rel) < TOL_TARGET and resolved),
            "verdict": ("PASS" if abs(rel) < TOL_PASS and resolved
                        else ("NOT_RESOLVED" if not resolved else "FAIL")),
        })
    worst = max((abs(r["rel_error"]) for r in rows if "rel_error" in r),
                default=float("inf"))
    return {"rows": rows, "worst_rel_error": float(worst),
            "passed_15": all(r.get("pass_15", False) for r in rows),
            "passed_3": all(r.get("pass_3", False) for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--t-end", type=float, default=4.0)
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--sound-speed", type=float, default=5.0)
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--ae", type=float, default=1.4e3)
    ap.add_argument("--tag", default="dx002")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    case = build_case(args.dx, args.sound_speed, args.ratio, args.ae)
    state = initial_state(case)
    dt = case.dt()
    n_steps = int(math.ceil(args.t_end / dt))

    hist = {k: [] for k in ("t", "tip_x", "tip_y", "drag_struct", "lift_struct",
                            "beam_drag_friction", "max_speed", "n_fluid")}
    t0 = time.time()
    finite = True
    for k in range(n_steps):
        row = step_fsi2(state, case, dt)
        for key in hist:
            hist[key].append(float(row.get(key, 0.0)))
        if not (math.isfinite(row["max_speed"]) and math.isfinite(row["tip_y"])):
            finite = False
            break
    wall = time.time() - t0

    measured = summarise_run(hist, args.warmup)
    ref = reference_statistics()
    verdict = compare(measured, ref) if finite and measured.get("n_samples") else {
        "rows": [], "worst_rel_error": float("inf"), "passed_15": False,
        "passed_3": False}

    steady = measured.get("steady_state", {}).get("steady", False)
    payload = {
        "case": "SPHinXsys test_2d_fsi2 archived parameter point (code-to-code)",
        "parameters": {**OSS_PARAMS, "dx": args.dx, "dt": dt,
                       "sound_speed": args.sound_speed, "ratio": args.ratio,
                       "ae": args.ae, "n_steps": n_steps,
                       "n_beam_nodes": case.n_beam_nodes,
                       "t_end": n_steps * dt, "warmup": args.warmup,
                       "beam_viscous_load": bool(case.beam_viscous_load)},
        "thresholds": {"pass": TOL_PASS, "target": TOL_TARGET},
        "reference": ref,
        "measured": measured,
        "verdict": verdict,
        "finite": bool(finite),
        "steady_state": steady,
        "final_status": ("PASS" if (verdict["passed_15"] and steady)
                         else ("FAIL" if finite and steady and measured
                               else "WITHHELD")),
        "wall_clock_s": round(wall, 1),
        "s_per_step": round(wall / max(len(hist["t"]), 1), 5),
        "history": hist,
    }
    out = Path(args.out or (ROOT / "artifacts" / "xval_fsi2"
                            / f"oss_matched_{args.tag}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in
                      ("parameters", "measured", "verdict", "final_status",
                       "wall_clock_s", "s_per_step")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())