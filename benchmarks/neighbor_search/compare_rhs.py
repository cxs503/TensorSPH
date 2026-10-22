"""Compare dense and pair-based WCSPH RHS across particle counts."""
from __future__ import annotations

import argparse
import json
import time

import torch

from tensorsph.droplet_impact import Config, rhs, rhs_all_pairs


def timed(fn, repeats, device):
    samples = []
    result = None
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return min(samples), result


def benchmark_size(n, dimension, repeats, device, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.rand((n, dimension), generator=gen, dtype=torch.float32).to(device)
    vel = (torch.rand((n, dimension), generator=gen, dtype=torch.float32) - 0.5).to(device)
    mass = torch.full((n,), 1.0 / n, device=device)
    cfg = Config(dx=0.025, h_factor=1.3, device=str(device))

    dense_time, dense = timed(lambda: rhs_all_pairs(pos, vel, mass, cfg), repeats, device)
    pair_time, paired = timed(lambda: rhs(pos, vel, mass, cfg), repeats, device)
    names = ("density", "pressure", "acceleration")
    errors = {}
    for name, actual, expected in zip(names, paired, dense):
        difference = (actual - expected).abs()
        errors[name] = {
            "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
            "max_rel": float((difference / expected.abs().clamp_min(1e-12)).max().item())
            if difference.numel() else 0.0,
        }
    close = all(torch.allclose(a, b, rtol=2e-4, atol=2e-5)
                for a, b in zip(paired, dense))
    return {
        "particles": n,
        "dense_rhs_seconds_min": dense_time,
        "pair_rhs_seconds_min": pair_time,
        "dense_over_pair_speed_ratio": dense_time / pair_time if pair_time else None,
        "numerically_close": close,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1000,
                        help="single particle count (used unless --sizes is set)")
    parser.add_argument("--sizes", default=None,
                        help="comma-separated particle counts, e.g. 256,512,1024")
    parser.add_argument("--dimension", type=int, choices=(2,), default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        sizes = ([int(item.strip()) for item in args.sizes.split(",")]
                 if args.sizes else [args.particles])
    except ValueError:
        parser.error("--sizes must be comma-separated integers")
    if not sizes or any(n < 2 for n in sizes) or args.repeats < 1:
        parser.error("all particle counts must be >= 2 and repeats must be positive")

    device = torch.device(args.device)
    results = []
    failed = False
    for n in sizes:
        result = benchmark_size(n, args.dimension, args.repeats, device, args.seed)
        failed |= not result["numerically_close"]
        results.append(result)
    print(json.dumps({
        "dimension": args.dimension,
        "device": str(device),
        "seed": args.seed,
        "repeats": args.repeats,
        "results": results,
        "note": "Microbenchmark only. Cell-linked construction currently copies coordinates to CPU; timings depend on hardware and distribution and are not a production-performance claim."
    }, indent=2))
    if failed:
        raise SystemExit("ERROR: pair-based RHS differs from dense reference beyond tolerance")


if __name__ == "__main__":
    main()
