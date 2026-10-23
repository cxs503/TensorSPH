"""Correctness/performance comparison for particle pair search.

Compares three independent implementations on the same point clouds:

* ``brute_force_pairs``      — O(N^2) oracle, no assumptions;
* ``cell_linked_pairs_host`` — previous host-side binning (Python loops after a
  copy to CPU), kept as an independent algorithm to cross-check the vectorized
  search and to make the before/after comparison reproducible;
* ``cell_linked_pairs``      — vectorized, device-resident linked-cell search.

All three must return exactly the same pair set or the script fails.
"""
from __future__ import annotations
import argparse
import json
import time
import torch
from tensorsph.neighbor_search import (
    brute_force_pairs,
    cell_linked_pairs,
    cell_linked_pairs_host,
)


def timed(fn, repeats):
    elapsed, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        if result.device.type == "cuda":
            torch.cuda.synchronize(result.device)
        elapsed.append(time.perf_counter() - start)
    return min(elapsed), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--cutoff", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-host", action="store_true",
                        help="do not run the independent host implementation")
    args = parser.parse_args()
    if args.particles < 0 or args.repeats < 1:
        parser.error("--particles must be non-negative and --repeats positive")
    if args.cutoff <= 0:
        parser.error("--cutoff must be positive")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    positions = torch.rand((args.particles, args.dimension),
                           generator=generator).to(args.device)

    brute_time, brute = timed(lambda: brute_force_pairs(positions, args.cutoff),
                              args.repeats)
    device_time, device_pairs = timed(
        lambda: cell_linked_pairs(positions, args.cutoff), args.repeats)
    host_time = None
    host_equal = None
    if not args.skip_host:
        host_time, host_pairs = timed(
            lambda: cell_linked_pairs_host(positions, args.cutoff), args.repeats)
        host_equal = torch.equal(host_pairs, brute)

    report = {
        "particles": args.particles, "dimension": args.dimension,
        "cutoff": args.cutoff, "device": str(positions.device), "seed": args.seed,
        "pair_count": int(device_pairs.shape[0]),
        "device_equals_brute_force": torch.equal(device_pairs, brute),
        "host_equals_brute_force": host_equal,
        "brute_force_seconds_min": brute_time,
        "device_seconds_min": device_time,
        "host_seconds_min": host_time,
        "speedup_brute_over_device": brute_time / device_time if device_time else None,
        "speedup_host_over_device": (host_time / device_time
                                      if host_time and device_time else None),
        "note": ("Microbenchmark only. The device search keeps all work on the "
                 "input device; the host search is retained as an independent "
                 "algorithm for cross-checking. Timings depend on hardware and "
                 "particle distribution."),
    }
    print(json.dumps(report, indent=2))
    if not report["device_equals_brute_force"]:
        raise SystemExit("ERROR: vectorized cell-linked pairs differ from brute force")
    if not args.skip_host and not host_equal:
        raise SystemExit("ERROR: host cell-linked pairs differ from brute force")


if __name__ == "__main__":
    main()
