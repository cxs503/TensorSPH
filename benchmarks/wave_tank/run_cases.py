"""Run an explicit list of wave-tank cases (cluster fill-in driver).

The campaign is dispatched in two waves: the grouped wave (``run_group.py``) and
this fill-in driver, which takes the cases that are still missing after the first
wave and runs them side by side with a small thread budget each. Cases are
independent, so a crash costs one case and the driver moves on to the next one.

    python benchmarks/wave_tank/run_cases.py --cases dispersion_kh1,csweep_c5 --threads 4 --attempts 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def worker(case: str, threads: int, attempts: int, runs: str, log) -> None:
    out = os.path.join(runs, f"{case}.json")
    for attempt in range(attempts):
        started = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "run.py"), "--case", case,
             "--threads", str(max(1, threads // (attempt + 1))), "--json-out", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        print(f"{case}: attempt {attempt + 1} rc={proc.returncode} "
              f"{time.time() - started:.0f} s json={os.path.exists(out)}", flush=True)
        if os.path.exists(out):
            return
        print(proc.stderr[-600:], flush=True)
    print(f"{case}: GAVE UP", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--runs", default=os.path.join(HERE, "..", "..", "runs", "wave_tank"))
    args = parser.parse_args(argv)
    runs = os.path.abspath(args.runs)
    os.makedirs(runs, exist_ok=True)
    threads = [threading.Thread(target=worker, daemon=False,
                                args=(case.strip(), args.threads, args.attempts, runs, None))
               for case in args.cases.split(",") if case.strip()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    have = sorted(name[:-5] for name in os.listdir(runs) if name.endswith(".json"))
    print("present:", " ".join(have), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())