#!/usr/bin/env python3
"""Run one (or more) released-first-mode sloshing cases of CASE-SLOSHING.

Each run integrates the pure first mode released from rest and measures the
first-mode frequency from the modal amplitude record
(:mod:`tensorsph.freesurface.sloshing_mode`), then compares it with the
linear-theory resonance frequency
(:mod:`tensorsph.freesurface.sloshing_anchor`). One JSON per case, so the cases
can be sharded over a cluster and the aggregate step cannot be affected by the
order they finished in.

Usage
-----
    python benchmarks/sloshing_resonance/run.py --list
    python benchmarks/sloshing_resonance/run.py --case base_cpd12
    python benchmarks/sloshing_resonance/run.py --all --outdir runs/sloshing_resonance
    python benchmarks/sloshing_resonance/run.py --shard 2 4     # cases 2, 6, 10 ...

The JSON contains the case as declared (geometry + the dimensionless groups),
the measured frequency and its error against theory, the damping per period,
mass drift, the peak density compression and the full module/timeseries record
used for the figures.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorsph.freesurface import sloshing_anchor, sloshing_mode  # noqa: E402

import cases as case_table  # noqa: E402


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case: case_table.Case, outdir: Path, threads: int) -> dict:
    torch.set_num_threads(max(1, min(threads, torch.get_num_threads())))
    omega_theory = sloshing_anchor.omega_n(case.tank_length, case.water_depth, 1,
                                           case_table.GRAVITY)
    # one field frame at a quarter of the first-mode period, baseline run only
    snapshot_time = 0.25 * (2.0 * math.pi / omega_theory) \
        if case.label == case_table.BASELINE.label else -1.0
    model = sloshing_mode.SloshingModeCase(
        tank_length=case.tank_length,
        water_depth=case.water_depth,
        tank_height=case.tank_height,
        amplitude=case.amplitude,
        cells_per_depth=case.cells_per_depth,
        sound_speed_ratio=case.sound_speed_ratio,
        viscosity=case.viscosity,
        periods=case.periods,
        output_per_period=case.output_per_period,
        snapshot_time=snapshot_time,
        label=case.label,
        gravity=case_table.GRAVITY,
    )
    started = time.perf_counter()
    record: dict
    try:
        result = sloshing_mode.run_sloshing_mode(model)
        summary = result.summary()
        # the surface records are only dumped here (not in the shared summary())
        # because they are what the report figures are drawn from
        summary["profiles"] = result.profiles_store
        status = "OK"
    except Exception as exc:                     # honest FAIL, not a crash
        result = None
        summary = {}
        status = f"EXCEPTION: {type(exc).__name__}: {exc}"
        traceback.print_exc()
    wall = float(time.perf_counter() - started)

    # primary measurement: horizontal centre of mass (an integral over the whole
    # cloud, so it carries no particle-spacing quantisation). The topmost-particle
    # modal projection that the free-surface study gates on is kept alongside.
    omega_com = float(summary.get("omega_from_com") or float("nan"))
    error_com = (abs(omega_com - omega_theory) / omega_theory
                 if math.isfinite(omega_com) else float("nan"))
    omega = float(summary.get("omega_measured") or float("nan"))
    error = abs(omega - omega_theory) / omega_theory if math.isfinite(omega) else float("nan")
    payload = {
        "benchmark": "CASE-SLOSHING",
        "label": case.label,
        "family": case.family,
        "note": case.note,
        "status": status,
        "case": {
            "tank_length": case.tank_length,
            "water_depth": case.water_depth,
            "tank_height": case.tank_height,
            "amplitude": case.amplitude,
            "cells_per_depth": case.cells_per_depth,
            "sound_speed_ratio": case.sound_speed_ratio,
            "periods": case.periods,
            "output_per_period": case.output_per_period,
            "viscosity": case.viscosity,
            "gravity": case_table.GRAVITY,
            "dx": case.dx,
            # dimensionless groups the result is reported against
            "depth_ratio_H_over_L": case.depth_ratio,
            "amplitude_ratio_A_over_H": case.amplitude_ratio,
            "wavenumber_depth_k1H": math.pi * case.water_depth / case.tank_length,
            "tanh_k1H": math.tanh(math.pi * case.water_depth / case.tank_length),
            "compressibility_gH_over_cs2": (
                case_table.GRAVITY * case.water_depth
                / (case.sound_speed_ratio ** 2 * case_table.GRAVITY * case.water_depth)),
            "nonlinearity_parameter_k1A": math.pi * case.amplitude / case.tank_length,
        },
        "theory": {
            "omega_linear": omega_theory,
            "period_linear": 2.0 * math.pi / omega_theory,
            "frequency_linear_hz": omega_theory / (2.0 * math.pi),
            "omega_linear_fd": sloshing_anchor.omega_n_fd(
                case.tank_length, case.water_depth, 1, case_table.GRAVITY),
            "source": ("linear potential flow: omega_1 = sqrt(g k1 tanh(k1 H)), "
                       "k1 = pi/L (Abramson NASA SP-106 ch.2; Faltinsen & Timokha "
                       "2009, rectangular-tank natural frequencies)"),
        },
        "measured": summary,
        "omega_measured": omega,
        "frequency_error": error,
        "omega_from_com": omega_com,
        "frequency_error_com": error_com,
        "tolerance": case_table.TOLERANCE,
        "pass": bool(math.isfinite(error_com) and error_com <= case_table.TOLERANCE),
        "pass_topmost_gauge": bool(math.isfinite(error) and error <= case_table.TOLERANCE),
        "wall_clock_s": wall,
        "torch_threads": int(torch.get_num_threads()),
        "torch_version": torch.__version__,
    }
    if result is not None and result.snapshot:
        payload["snapshot"] = result.snapshot
        payload["snapshot"]["omega_linear"] = omega_theory
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{case.label}.json"
    path.write_text(json.dumps(payload, indent=1, allow_nan=True))
    payload["_path"] = str(path)
    payload["_sha256"] = sha256_of(path)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[],
                        help="case label (repeatable); default = baseline")
    parser.add_argument("--all", action="store_true", help="every case in the table")
    parser.add_argument("--list", action="store_true", help="print the case table")
    parser.add_argument("--outdir", default=str(REPO / "runs" / "sloshing_resonance"))
    parser.add_argument("--threads", type=int, default=4,
                        help="torch threads (4 measured fastest for these lattices)")
    parser.add_argument("--pilot", action="store_true",
                        help="smoke only: cpd=6, one period (never a reported number)")
    parser.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"),
                        help="run cases I, I+N, I+2N, ... (0-based), for the cluster")
    args = parser.parse_args(argv)

    if args.list:
        print(f"{'label':12s} {'H/L':>6s} {'A/H':>6s} {'cpd':>5s} {'cs/rt':>6s} "
              f"{'omega_th':>9s} {'f_th [Hz]':>9s}  family")
        for case in case_table.CASES:
            print(f"{case.label:12s} {case.depth_ratio:6.2f} {case.amplitude_ratio:6.2f} "
                  f"{case.cells_per_depth:5.1f} {case.sound_speed_ratio:6.1f} "
                  f"{case.omega_theory():9.4f} "
                  f"{case.omega_theory() / (2.0 * math.pi):9.4f}  {case.family}")
        return 0

    if args.shard:
        index, count = args.shard
        if count < 1 or not 0 <= index < count:
            parser.error("--shard I N needs 0 <= I < N")
        chosen = [c for i, c in enumerate(case_table.CASES) if i % count == index]
    elif args.all:
        chosen = list(case_table.CASES)
    elif args.case:
        chosen = [case_table.by_label(label) for label in args.case]
    else:
        chosen = [case_table.baseline()]

    if args.pilot:
        # plumbing check only: the coarse lattice cannot resolve the frequency,
        # so these numbers are never reported
        chosen = [dataclasses.replace(c, cells_per_depth=6.0, periods=1.0,
                                      output_per_period=20) for c in chosen]

    outdir = Path(args.outdir)
    worst = 0.0
    failures = 0
    for case in chosen:
        payload = run_case(case, outdir, args.threads)
        err = payload["frequency_error_com"]
        if math.isfinite(err):
            worst = max(worst, err)
        else:
            failures += 1
        print(f"{case.label:12s} omega_th={payload['theory']['omega_linear']:.4f} "
              f"omega_sim(com)={payload['omega_from_com']:.4f} "
              f"omega_sim(top)={payload['omega_measured']:.4f} "
              f"err={100.0 * err:.2f}%  pass={payload['pass']}  "
              f"wall={payload['wall_clock_s']:.0f}s  -> {payload['_path']}",
              flush=True)
    print(f"[{os.path.basename(__file__)}] {len(chosen)} case(s), "
          f"worst error {100.0 * worst:.2f}%, unmeasured {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())