"""Launch one node's share of the wave-tank campaign (cluster entry point).

Each *slot* is a sequence of case groups from ``run_group.py`` run one after the
other in a single process slot; the slots of a node run beside each other. Two
nodes with three slots each keep 44 torch threads busy per node, which is where
this workload peaks: the neighbour search and the pair loops are memory bound, so
more threads per process make each step *slower* (measured: 35 ms/step at 8
threads vs 49 ms/step at 32 on a 12.5k-particle tank).

    python benchmarks/wave_tank/run_node.py --slots 1,3,5+7      # node A
    python benchmarks/wave_tank/run_node.py --slots 2,4+6,8      # node B
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", required=True,
                        help="comma separated slots; '+' joins groups in one slot")
    parser.add_argument("--threads", type=int, default=0,
                        help="per-process thread budget (slots of a node share its cores)")
    args = parser.parse_args(argv)
    slots = [part.split("+") for part in args.slots.split(",")]
    running = []
    for index, groups in enumerate(slots):
        log = os.path.join("/tmp", f"slot{index}.log")
        handle = open(log, "w", encoding="utf-8")
        procs = [subprocess.Popen(
            [sys.executable, os.path.join(HERE, "run_group.py"), "--group", group]
            + (["--threads", str(args.threads)] if args.threads else []),
            stdout=handle, stderr=subprocess.STDOUT) for group in groups]
        running.append((index, groups, procs, handle))
        print(f"slot {index}: groups {groups} -> {log}", flush=True)
    status = 0
    for index, groups, procs, handle in running:
        started = time.time()
        codes = [proc.wait() for proc in procs]
        handle.close()
        print(f"slot {index}: groups {groups} finished in {time.time() - started:.0f} s "
              f"codes={codes}", flush=True)
        if any(codes):
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())