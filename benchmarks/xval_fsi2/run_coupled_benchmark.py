"""Coupled FSI2 benchmark: TensorSPH vs SPHinXsys archived reference.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Decision
--------
Can TensorSPH's strong-coupling FSI2 driver reproduce the beam-tip displacement
amplitudes of the archived SPHinXsys ``test_2d_fsi2`` records within the owner
accuracy standard (<15 %)?

Alternatives
------------
* ``strong-iqn-ils`` -- interface quasi-Newton (Degroote 2009), the standard
  cure for added-mass instability at ``rho_s/rho_f = 1``.
* ``strong-aitken``  -- dynamic secant relaxation (Irons-Tuck), cheaper but
  less robust for high-gain interfaces.
* ``loose``          -- shipped driver, known to diverge at ratio 1.

Predeclared metric
------------------
Beam-tip displacement amplitudes (half-range) in physical units [m]:
``tip_x_amplitude``, ``tip_y_amplitude``.  Compared against the SPHinXsys
archived record (``benchmarks/xval_fsi2/reference/oss/``) which has been
converted to physical units by ``tensorsph.fsi2_reference``.

Thresholds (fixed before the run):
* ``|rel_error| < 15 %``  → PASS (owner acceptance standard)
* ``|rel_error| < 3 %``   → PASS (reporting target)

Stopping rule: no threshold is retuned after the run; every variant is reported
whether it passes or fails.

Environment
-----------
torch CPU float64, dx = 0.02 m (coarse probe), similarity-corrected case at
``U_mean = 0.2 m/s`` (Re = 100, Ae = 5600, ``rho_s/rho_f = 1``).
The long production run (dx = 0.005 m, t_end = 10 s) is submitted to the
cluster queue; this script is the probe.

Output: ``artifacts/xval_fsi2/coupled_benchmark.json`` (nothing hand-typed).
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
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.fsi2_reference import (  # noqa: E402
    LENGTH_SCALE, summarise_tip, compare_to_benchmark, BENCHMARK_TIP,
)
from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_strong_coupling import run_fsi2_strong  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2  # noqa: E402


def _case(dx: float, ratio: float = 1.0, c0: float = 5.0,
          u_mean: float = 0.2) -> "FSI2Case":
    """Similarity-corrected FSI2 case at the benchmark density ratio."""
    case = similarity_corrected_case(U_mean=u_mean)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = 1000.0 * ratio
    case.n_beam_nodes = max(15, int(case.beam_length / dx))
    case.max_rod_substeps = 2000
    case.rod_stability_target = 0.2
    return case


def _tip_amplitudes(tip_x_series: list[float],
                    tip_y_series: list[float],
                    warmup_steps: int = 50) -> dict[str, float]:
    """Half-range amplitudes after warmup, in physical units."""
    if len(tip_x_series) <= warmup_steps:
        return {"tip_x_amplitude": 0.0, "tip_y_amplitude": 0.0,
                "n_samples": max(0, len(tip_x_series) - warmup_steps)}
    x = np.array(tip_x_series[warmup_steps:])
    y = np.array(tip_y_series[warmup_steps:])
    return {
        "tip_x_amplitude": float(0.5 * (x.max() - x.min())),
        "tip_y_amplitude": float(0.5 * (y.max() - y.min())),
        "x_mean": float(x.mean()),
        "y_mean": float(y.mean()),
        "n_samples": len(x),
    }


def run_variant(dx: float, n_steps: int, variant: str,
                max_iter: int = 15, tol: float = 1e-3) -> dict:
    """Run one coupling variant and return the tip-amplitude summary."""
    case = _case(dx)
    state = initial_state(case)
    dt = case.dt()

    tip_x, tip_y = [], []
    finite = True
    aborted = False
    t0 = time.time()

    if variant == "loose":
        for k in range(n_steps):
            row = step_fsi2(state, case, dt)
            v = float(state.beam_vel.norm(dim=1).max())
            if not (math.isfinite(v) and math.isfinite(row["max_speed"])):
                finite = False
                break
            tip_x.append(row["tip_x"])
            tip_y.append(row["tip_y"])
    else:
        solver = variant.split("-", 1)[1]
        state, result = run_fsi2_strong(
            case, n_steps, interface_solver=solver,
            max_iter=max_iter, tol=tol, stop_on_abort=True)
        aborted = result["aborted"]
        tip_x = result["history"]["tip_x"]
        tip_y = result["history"]["tip_y"]
        finite = all(math.isfinite(v) for v in tip_x + tip_y)

    wall_clock = time.time() - t0
    amps = _tip_amplitudes(tip_x, tip_y)
    gate = compare_to_benchmark(amps) if finite and amps["n_samples"] > 0 else {
        "passed_15": False, "passed_3": False, "worst_rel_error": float("inf"),
        "rows": [],
    }

    return {
        "variant": variant, "dx": dx, "n_steps": n_steps, "dt": dt,
        "finite": finite, "aborted": aborted,
        "wall_clock_s": round(wall_clock, 2),
        "tip_amplitudes": amps,
        "gate": gate,
        "tip_x_last": tip_x[-1] if tip_x else None,
        "tip_y_last": tip_y[-1] if tip_y else None,
        "n_samples": len(tip_x),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "xval_fsi2"
                                         / "coupled_benchmark.json"))
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--variants", default="loose,strong-iqn-ils,strong-aitken")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    rows = []
    t0 = time.time()

    for variant in variants:
        print(f"Running {variant} (dx={args.dx}, n={args.n_steps})...", flush=True)
        r = run_variant(args.dx, args.n_steps, variant)
        rows.append(r)
        gate = r["gate"]
        print(f"  {variant:18s} finite={r['finite']} aborted={r['aborted']} "
              f"tip_y_amp={r['tip_amplitudes']['tip_y_amplitude']:.6f} m "
              f"pass_15={gate['passed_15']} pass_3={gate['passed_3']} "
              f"worst_err={gate['worst_rel_error']*100:.1f}% "
              f"wall={r['wall_clock_s']:.1f}s", flush=True)

    # Reference comparison
    oss_runs = {}
    for run in ("0", "4", "9"):
        try:
            from tensorsph.fsi2_reference import load_tip_series
            _, pos = load_tip_series(run)
            oss_runs[run] = summarise_tip(pos)
        except Exception:
            pass

    payload = {
        "decision": ("Can TensorSPH reproduce the SPHinXsys FSI2 beam-tip "
                     "displacement amplitudes within 15 %?"),
        "metric": "tip_x_amplitude, tip_y_amplitude [m] (half-range after warmup)",
        "thresholds": {"pass_15": "< 15 %", "pass_3": "< 3 %"},
        "reference_benchmark": BENCHMARK_TIP,
        "reference_oss_runs": oss_runs,
        "parameters": {
            "dx": args.dx, "n_steps": args.n_steps,
            "u_mean": 0.2, "Re": 100, "Ae": 5600, "rho_s_over_rho_f": 1.0,
        },
        "rows": rows,
        "wall_clock_total_s": round(time.time() - t0, 2),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
