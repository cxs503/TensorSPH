#!/usr/bin/env python3
"""Before/after measurement of particle shifting on two WCSPH cases.

The point of the run is *evidence*, not a demo: for each case the same solver,
the same lattice, the same dt and the same number of steps are integrated with
shifting off and on, and the quantities that the repository already grades are
compared.

Case R (resting column, the shape of defect F1)
    Water at rest in a tank.  Nothing drives it, so every metre per second the
    solver reports is spurious.  Reported: the late-window RMS speed
    ``v_rms = sqrt(2 KE / M)`` of the column, i.e. exactly the readout used by
    ``benchmarks/startup_quiescence`` for F1 (ceiling 0.06 m/s), the peak
    summation-density error, and mass conservation.

Case D (B2 dam break, the published benchmark)
    Shifting must not perturb the published answer.  Reported: front
    trajectory against the unshifted run (max |dx_front|), the front at the last
    frame, peak density compression, escaped particles and wall-clock.

Usage::

    PYTHONPATH=src python benchmarks/particle_shifting/run.py            # full
    PYTHONPATH=src python benchmarks/particle_shifting/run.py --quick    # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402

OUT = Path(__file__).resolve().parent


def resting_config(dx: float, seconds: float) -> DamBreakConfig:
    """A column at rest in a tank: no gate, no gravity-driven surge."""
    # The column fills the tank: there is no gate and no free surface to
    # collapse, so every metre per second the solver reports is spurious.  (A
    # *narrow* column in a wide tank is a dam break, not a resting pool -- the
    # first version of this script measured 0.70 m/s for exactly that reason.)
    cfg = DamBreakConfig(dx=dx, column_width=1.0, column_height=0.3,
                         pool_width=1.0, tank_height=0.4, sound_speed=20.0,
                         viscosity=0.01, density_diffusion=0.0)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = 10
    return cfg


def dam_break_config(dx: float, seconds: float) -> DamBreakConfig:
    cfg = DamBreakConfig(dx=dx)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = 50
    return cfg


def late_rms_speed(run, mass: float, tail: float = 0.4) -> float:
    """RMS speed over the last ``tail`` fraction of the record."""
    ke = run.kinetic_energy
    if not ke:
        return float("nan")
    start = int(len(ke) * (1.0 - tail))
    window = ke[start:]
    return math.sqrt(2.0 * sum(window) / len(window) / mass)


def run_one(cfg: DamBreakConfig, shift_every: int) -> dict:
    cfg.shift_every = shift_every
    t0 = time.perf_counter()
    run = run_case(cfg)
    wall = time.perf_counter() - t0
    return {
        "shift_every": shift_every,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall,
        "late_rms_speed": late_rms_speed(run, run.mass),
        "max_density_compression": run.max_density_compression,
        "max_acoustic_courant": run.max_acoustic_courant,
        "escaped": run.escaped,
        "mass_drift": abs(run.final_mass - run.initial_mass) / run.initial_mass,
        "front_final": run.front[-1],
        "front_series": run.front,
        "times": run.times,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "results.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    dx_r, sec_r = (0.04, 0.2) if args.quick else (0.02, 0.4)
    dx_d, sec_d = (0.04, 0.4) if args.quick else (0.04, 0.8)
    shift_every = 5

    report: dict = {"benchmark": "particle_shifting",
                    "scheme": "Lind et al. (2012) Fickian shift, tangential at the free surface",
                    "shift_every": shift_every,
                    "threads": args.threads, "cases": {}}

    print(f"[R] resting column dx={dx_r} t={sec_r}s")
    base_r = run_one(resting_config(dx_r, sec_r), 0)
    shift_r = run_one(resting_config(dx_r, sec_r), shift_every)
    report["cases"]["resting"] = {
        "dx": dx_r, "seconds": sec_r, "off": base_r, "on": shift_r,
        "late_rms_ratio": (shift_r["late_rms_speed"] / base_r["late_rms_speed"]
                           if base_r["late_rms_speed"] else None),
    }

    print(f"[D] dam break dx={dx_d} t={sec_d}s")
    base_d = run_one(dam_break_config(dx_d, sec_d), 0)
    shift_d = run_one(dam_break_config(dx_d, sec_d), shift_every)
    n = min(len(base_d["front_series"]), len(shift_d["front_series"]))
    front_delta = max(abs(a - b) for a, b in
                      zip(base_d["front_series"][:n], shift_d["front_series"][:n]))
    report["cases"]["dam_break"] = {
        "dx": dx_d, "seconds": sec_d, "off": base_d, "on": shift_d,
        "max_front_delta_m": front_delta,
        "front_delta_over_column": front_delta / 0.2,
    }

    for name, case in report["cases"].items():
        for key in ("off", "on"):
            case[key].pop("front_series", None)
            case[key].pop("times", None)

    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("late_rms_ratio", "max_front_delta_m",
                                    "front_delta_over_column")}
                      for k, v in report["cases"].items()}, indent=1))
    print("wrote", args.out)
    print("resting v_rms off/on: "
          f"{report['cases']['resting']['off']['late_rms_speed']:.4f} / "
          f"{report['cases']['resting']['on']['late_rms_speed']:.4f} m/s")
    print("dam-break front final off/on: "
          f"{report['cases']['dam_break']['off']['front_final']:.4f} / "
          f"{report['cases']['dam_break']['on']['front_final']:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())