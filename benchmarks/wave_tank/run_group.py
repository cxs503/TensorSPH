"""Run one group of wave-tank campaign cases (cluster shard entry point).

The campaign is a set of independent cases; grouping them lets the two 128-core
cluster nodes work through the matrix without any two processes competing for the
same cores. Each group is a list of cases plus a thread budget; every case is run
by ``run.py`` in its own process (torch thread pools do not scale past ~8 threads
on this workload, so several small processes beat one wide one).

    python benchmarks/wave_tank/run_group.py --group 3

Groups are numbered 1..8 and are dispatched round-robin over the two nodes by the
cluster ``shard`` action (odd -> n315, even -> n330). The heavy cases (decay over
ten wavelengths, the finest point of the resolution ladder) are alone in a group
so that they define the wall time of their node and nothing else queues behind
them.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

GROUPS = {
    1: dict(threads=24, cases=["decay10"]),
    2: dict(threads=24, cases=["ladder_dp0.0075"]),
    3: dict(threads=24, cases=["ladder_dp0.010", "solitary"]),
    4: dict(threads=12, cases=["ladder_dp0.015", "ladder_dp0.020", "csweep_c20"]),
    5: dict(threads=12, cases=["dispersion_kh0.5", "dispersion_kh1"]),
    6: dict(threads=12, cases=["dispersion_kh2", "dispersion_kh3"]),
    7: dict(threads=12, cases=["reflection", "reflection_control"]),
    8: dict(threads=12, cases=["piston_transfer", "csweep_c5", "hydrostatic"],
            viz=["regular", "solitary"]),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, required=True, choices=sorted(GROUPS))
    parser.add_argument("--threads", type=int, default=0,
                        help="override the per-group thread budget (cluster slots share a node)")
    parser.add_argument("--runs", default=os.path.join(HERE, "..", "..", "runs", "wave_tank"))
    args = parser.parse_args(argv)
    spec = dict(GROUPS[args.group])
    if args.threads:
        spec["threads"] = args.threads
    runs = os.path.abspath(args.runs)
    os.makedirs(runs, exist_ok=True)
    status = 0
    for case in spec["cases"]:
        out = os.path.join(runs, f"{case}.json")
        started = time.time()
        print(f"[group {args.group}] {case}: start (threads={spec['threads']})", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "run.py"), "--case", case,
             "--threads", str(spec["threads"]), "--json-out", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 and not os.path.exists(out):
            # One retry: the clusters run other tenants' jobs beside ours and a
            # run has been seen to die with a glibc heap report (rc=-6) that is
            # not reproducible for the same case on a quieter node.
            print(f"[group {args.group}] {case}: rc={proc.returncode}, retrying "
                  f"with {max(1, spec['threads'] // 2)} threads", flush=True)
            started = time.time()
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "run.py"), "--case", case,
                 "--threads", str(max(1, spec["threads"] // 2)), "--json-out", out],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        elapsed = time.time() - started
        ok = os.path.exists(out)
        print(f"[group {args.group}] {case}: done in {elapsed:.0f} s "
              f"(rc={proc.returncode}, json={ok})", flush=True)
        if proc.returncode != 0 or not ok:
            status = 1
            print(proc.stderr[-2000:], flush=True)
    for viz in spec.get("viz", ()):
        started = time.time()
        print(f"[group {args.group}] viz_{viz}: start", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "viz_case.py"), "--case", viz,
             "--threads", str(spec["threads"])],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"[group {args.group}] viz_{viz}: done in {time.time() - started:.0f} s "
              f"(rc={proc.returncode}) {proc.stdout.strip()}", flush=True)
        if proc.returncode != 0:
            status = 1
            print(proc.stderr[-2000:], flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
