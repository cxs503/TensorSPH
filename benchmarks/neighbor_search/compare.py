"""Correctness/performance comparison for particle pair search."""
from __future__ import annotations
import argparse
import json
import time
import torch
from tensorsph.neighbor_search import brute_force_pairs, cell_linked_pairs


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
    args = parser.parse_args()
    if args.particles < 0 or args.repeats < 1:
        parser.error("--particles must be non-negative and --repeats positive")
    if args.cutoff <= 0:
        parser.error("--cutoff must be positive")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    positions = torch.rand((args.particles, args.dimension), generator=generator).to(args.device)
    brute_time, brute = timed(lambda: brute_force_pairs(positions, args.cutoff), args.repeats)
    cell_time, cell = timed(lambda: cell_linked_pairs(positions, args.cutoff), args.repeats)
    equivalent = torch.equal(brute, cell)
    print(json.dumps({
        "particles": args.particles, "dimension": args.dimension,
        "cutoff": args.cutoff, "device": str(positions.device), "seed": args.seed,
        "pair_count": int(cell.shape[0]), "pairs_equal": equivalent,
        "brute_force_seconds_min": brute_time, "cell_linked_seconds_min": cell_time,
        "speedup_brute_over_cell": brute_time / cell_time if cell_time else None,
        "note": "Microbenchmark only; cell-linked coordinates are copied to CPU."
    }, indent=2))
    if not equivalent:
        raise SystemExit("ERROR: cell-linked pairs differ from brute-force reference")


if __name__ == "__main__":
    main()
