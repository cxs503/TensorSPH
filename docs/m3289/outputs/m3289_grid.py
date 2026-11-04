"""Acceptance grid for the shell-pressure closure (mission m3289).

Runs the P0 acceptance grid -- c0 x dx cross, closure off and on, plus a sweep
sensitivity ladder at the coarse point -- and dumps everything to JSON. The
benchmark's ``run.py`` is imported, not copied, so the numbers here are the ones
the gates read.

    setsid env PYTHONPATH=src python3 outputs/m3289_grid.py outputs/m3289_grid.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BENCH = ROOT / "benchmarks" / "laplace_pressure_floor"


def load():
    spec = importlib.util.spec_from_file_location("lpf_run", BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    torch.set_num_threads(2)
    lpf = load()
    out_path = Path(sys.argv[1] if len(sys.argv) > 1
                    else ROOT / "outputs" / "m3289_grid.json")
    rows = []
    for dx in (0.002, 0.0015):
        for c0 in (20.0, 10.0, 5.0):
            for closure in (False, True):
                t0 = time.time()
                res = lpf.relax(dx, c0, closure=closure, sweeps=2)
                wall = time.time() - t0
                rows.append({"dx": dx, "c0": c0, "closure": closure,
                             "sweeps": 2, "wall_s": wall, "result": res})
                tail = res["absolute_time_average_tail"]["rel_error_pct"]
                print(f"dx={dx} c0={c0:5.1f} closure={closure!s:5s} "
                      f"tail={tail:8.2f}% steps={res['steps']} "
                      f"wall={wall:6.1f}s", flush=True)
                out_path.write_text(json.dumps(rows, indent=2))
    # Sweep ladder: how many Gauss-Seidel sweeps does the closure need?
    for sweeps in (1, 3, 4):
        t0 = time.time()
        res = lpf.relax(0.002, 20.0, closure=True, sweeps=sweeps)
        wall = time.time() - t0
        rows.append({"dx": 0.002, "c0": 20.0, "closure": True,
                     "sweeps": sweeps, "wall_s": wall, "result": res})
        print(f"sweeps={sweeps} tail="
              f"{res['absolute_time_average_tail']['rel_error_pct']:8.2f}% "
              f"wall={wall:6.1f}s", flush=True)
        out_path.write_text(json.dumps(rows, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()