"""Compare dense and pair-based WCSPH RHS runtime and numerical agreement."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--dimension", type=int, choices=(2,), default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.particles < 2 or args.repeats < 1:
        parser.error("--particles must be >= 2 and --repeats must be positive")

    device = torch.device(args.device)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    pos = torch.rand((args.particles, args.dimension), generator=gen, dtype=torch.float32).to(device)
    vel = (torch.rand((args.particles, args.dimension), generator=gen, dtype=torch.float32) - 0.5).to(device)
    mass = torch.full((args.particles,), 1.0 / args.particles, device=device)
    cfg = Config(dx=0.025, h_factor=1.3, device=str(device))

    dense_time, dense = timed(lambda: rhs_all_pairs(pos, vel, mass, cfg), args.repeats, device)
    pair_time, paired = timed(lambda: rhs(pos, vel, mass, cfg), args.repeats, device)
    names = ("density", "pressure", "acceleration")
    errors = {}
    for name, a, b in zip(names, paired, dense):
        errors[name] = {
            "max_abs": float((a - b).abs().max().item()) if a.numel() else 0.0,
            "max_rel": float(((a - b).abs() / b.abs().clamp_min(1e-12)).max().item()) if a.numel() else 0.0,
        }
    equivalent = all(torch.allclose(a, b, rtol=2e-4, atol=2e-5) for a, b in zip(paired, dense))
    print(json.dumps({
        "particles": args.particles, "dimension": args.dimension,
        "device": str(device), "seed": args.seed, "repeats": args.repeats,
        "dense_rhs_seconds_min": dense_time, "pair_rhs_seconds_min": pair_time,
        "dense_over_pair_speed_ratio": dense_time / pair_time if pair_time else None,
        "numerically_close": equivalent, "errors": errors,
        "note": "Benchmark only; linked-cell construction currently transfers coordinates to CPU. Do not infer production performance from one run."
    }, indent=2))
    if not equivalent:
        raise SystemExit("ERROR: pair-based RHS differs from dense reference beyond tolerance")


if __name__ == "__main__":
    main()
