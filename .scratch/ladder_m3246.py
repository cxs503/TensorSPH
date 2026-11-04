"""Run one (dx, c0, passes) point of the free-surface-closure ladder.

Writes .scratch/ladder_<dx>_<c0>_p<passes>.json so the driver can merge them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "lpf", ROOT / "benchmarks" / "laplace_pressure_floor" / "run.py")
lpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpf)

dx = float(sys.argv[1])
c0 = float(sys.argv[2])
passes = int(sys.argv[3])
flag = passes > 0
t0 = time.time()
run = lpf.relax(dx=dx, c0=c0, free_surface_pressure=flag,
                free_surface_pressure_passes=max(1, passes))
run["wall_clock_s"] = time.time() - t0
out = HERE / f"ladder_{dx}_{c0}_p{passes}.json"
out.write_text(json.dumps(run, indent=1))
print(passes, dx, c0, run["absolute_time_average_tail"]["rel_error_pct"],
      run["finite"], f"{run['wall_clock_s']:.0f}s", flush=True)