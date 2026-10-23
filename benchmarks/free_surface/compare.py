"""Side-by-side comparison of free-surface treatments.

Two treatments are available:

* **pressure clamp** (default, ``density_diffusion = 0``) -- the Tait law is
  clamped at zero, which removes tensile instability at the cost of discarding
  all rarefaction;
* **diffusive density term** (``density_diffusion = delta > 0``) -- the term of
  Antuono, Colagrossi & Marrone (2012) applied to the summation density.

This script measures what the difference actually is, on three observables that
are already part of the verification ladder:

1. **B1-like hydrostatics** -- time-averaged interior pressure error and the
   temporal pressure fluctuation that a free-surface treatment is supposed to
   calm. A time average alone hides the noise, so the second moment is
   reported as well.
2. **B2-like dam break** -- the leading-edge trajectory against the clamp run,
   plus bulk and peak density compression.
3. **Droplet impact** -- nearest-neighbour statistics: the smallest pair
   separation and the fraction of pairs closer than ``dx/2``. This is the
   "clumping" evidence the roadmap asks for.

The script reports numbers; it does not decide the default. The measured result
and the decision that followed are recorded in
``docs/benchmarks/free-surface-treatment.md``.

Usage:
    python benchmarks/free_surface/compare.py
    python benchmarks/free_surface/compare.py --deltas 0,0.05,0.1 --json-out out.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import time

import torch

from tensorsph.dam_break import DamBreakConfig, run_case
from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    make_particles,
    resolve_device,
    rhs,
    wall_lattice,
)
from tensorsph.neighbor_search import cell_linked_pairs



def clumping(pos: torch.Tensor, dx: float) -> dict:
    """Nearest-neighbour statistics over fluid particles only."""
    if pos.shape[0] < 2:
        return {"pairs": 0}
    pairs = cell_linked_pairs(pos, 2.6 * dx)
    i, j = pairs.unbind(dim=1)
    distance = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    if distance.numel() == 0:
        return {"pairs": 0}
    return {
        "pairs": int(distance.numel()),
        "min_separation_over_dx": float(distance.min()) / dx,
        "median_separation_over_dx": float(distance.median()) / dx,
        "fraction_closer_than_half_dx": float((distance < 0.5 * dx).float().mean()),
        "fraction_closer_than_dx": float((distance < dx).float().mean()),
    }


def hydrostatics(delta: float, dx: float, seconds: float, device: torch.device,
                 bands: int = 7) -> dict:
    cfg = Config(dx=dx, device=str(device), density_diffusion=delta,
                 max_courant=0.25)
    cfg.dt = cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed
    pos, mass = make_particles(cfg, device)
    keep = pos[:, 1] <= cfg.pool_depth
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    wall = wall_lattice(cfg, device)
    vel = torch.zeros_like(pos)
    steps = max(int(round(seconds / cfg.dt)), 2)
    average_from = steps // 2
    total = torch.zeros(pos.shape[0], dtype=pos.dtype, device=device)
    square = torch.zeros_like(total)
    samples = 0
    started = time.perf_counter()
    for step in range(steps):
        _, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall)
        if step >= average_from:
            total += pressure
            square += pressure.square()
            samples += 1
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        apply_boundary(pos, vel, cfg)
    mean = total / samples
    std = (square / samples - mean.square()).clamp_min(0.0).sqrt()

    # Same band definition as benchmarks/hydrostatic/run.py, so the numbers are
    # comparable with the B1 table (7 bands, top one is the free surface).
    edges = [cfg.pool_depth * k / bands for k in range(bands + 1)]
    errors, fluctuations = [], []
    for k in range(bands - 1):  # interior bands only
        low, high = edges[k], edges[k + 1]
        mask = (pos[:, 1] >= low) & (pos[:, 1] < high)
        if not bool(mask.any()):
            continue
        y = float(pos[mask, 1].mean())
        analytic = cfg.rho0 * cfg.gravity * (cfg.pool_depth - y)
        if analytic <= 0:
            continue
        errors.append(abs(float(mean[mask].mean()) - analytic) / analytic)
        fluctuations.append(float(std[mask].mean()) / analytic)
    return {
        "delta": delta,
        "elapsed_seconds": time.perf_counter() - started,
        "interior_pressure_error": sum(errors) / len(errors),
        "relative_pressure_fluctuation": sum(fluctuations) / len(fluctuations),
        "residual_max_speed": float(torch.linalg.vector_norm(vel, dim=-1).max()),
    }


def dam_break(delta: float, dx: float, seconds: float) -> dict:
    cfg = DamBreakConfig(dx=dx, pool_width=2.0, column_width=0.2,
                         sound_speed=25.0, density_diffusion=delta)
    cfg.column_height = 0.4
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))
    started = time.perf_counter()
    run = run_case(cfg)
    return {
        "delta": delta,
        "elapsed_seconds": time.perf_counter() - started,
        "front_final_over_a": run.front[-1] / cfg.column_width,
        "bulk_compression": run.compression_before_wall_impact,
        "peak_compression": run.max_density_compression,
        "wall_impact_time": run.wall_impact_time,
        "times": run.times,
        "front": run.front,
    }


def impact(delta: float, dx: float, seconds: float, device: torch.device) -> dict:
    cfg = Config(dx=dx, device=str(device), density_diffusion=delta)
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = cfg.steps
    pos, mass = make_particles(cfg, device)
    wall = wall_lattice(cfg, device)
    vel = torch.zeros_like(pos)
    peaks = {"compression": 0.0, "speed": 0.0}
    started = time.perf_counter()
    for step in range(cfg.steps + 1):
        rho, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        peaks["compression"] = max(peaks["compression"],
                                   float((rho / cfg.rho0).max() - 1.0))
        peaks["speed"] = max(peaks["speed"],
                             float(torch.linalg.vector_norm(vel, dim=-1).max()))
        if step == cfg.steps:
            break
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        apply_boundary(pos, vel, cfg)
    elapsed = time.perf_counter() - started
    stats = clumping(pos, cfg.dx)
    return {
        "delta": delta,
        "elapsed_seconds": elapsed,
        "kinetic_energy": float(0.5 * (mass[:, None] * vel.square()).sum()),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "peak_compression": peaks["compression"],
        "peak_speed": peaks["speed"],
        **stats,
    }


def interpolate(times, values, query):
    if query <= times[0]:
        return values[0]
    if query >= times[-1]:
        return values[-1]
    k = bisect.bisect_left(times, query)
    weight = (query - times[k - 1]) / (times[k] - times[k - 1])
    return values[k - 1] + weight * (values[k] - values[k - 1])


def front_difference(reference: dict, other: dict) -> dict:
    """Relative L2 of front(t)/a between two runs of the same geometry."""
    limit = min(reference["times"][-1], other["times"][-1])
    count = 40
    numerator = denominator = 0.0
    for k in range(count + 1):
        t = limit * k / count
        x = interpolate(reference["times"], reference["front"], t)
        y = interpolate(other["times"], other["front"], t)
        numerator += (x - y) ** 2
        denominator += y * y
    return {"window_seconds": limit,
            "relative_l2": math.sqrt(numerator / denominator) if denominator else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", default="0,0.1",
                        help="comma-separated density-diffusion coefficients")
    parser.add_argument("--dx", type=float, default=0.025)
    parser.add_argument("--dam-dx", type=float, default=0.02)
    parser.add_argument("--bands", type=int, default=7)
    parser.add_argument("--hydro-seconds", type=float, default=4.0,
                        help="matches the B1 protocol: 4 s, average over the last half")
    parser.add_argument("--dam-seconds", type=float, default=0.8)
    parser.add_argument("--impact-seconds", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    deltas = [float(item) for item in args.deltas.split(",") if item.strip()]
    if not deltas or any(d < 0 for d in deltas):
        parser.error("--deltas must be non-negative")
    device = resolve_device(args.device)

    report = {"scheme": "Antuono-Colagrossi-Marrone diffusive term on summation density",
              "note": ("Applied every step over cfg.dt. It is not delta-SPH proper: "
                       "this solver obtains density by summation rather than by "
                       "integrating the continuity equation."),
              "deltas": deltas, "hydrostatics": [], "dam_break": [], "impact": []}
    for delta in deltas:
        print(f"  delta={delta:g}: hydrostatics ...", flush=True)
        report["hydrostatics"].append(
            hydrostatics(delta, args.dx, args.hydro_seconds, device, args.bands))
        print(f"  delta={delta:g}: dam break ...", flush=True)
        report["dam_break"].append(
            dam_break(delta, args.dam_dx, args.dam_seconds))
        print(f"  delta={delta:g}: impact ...", flush=True)
        report["impact"].append(
            impact(delta, args.dx, args.impact_seconds, device))

    reference = report["dam_break"][0]
    for item in report["dam_break"][1:]:
        item["front_difference_vs_first"] = front_difference(reference, item)

    base = report["hydrostatics"][0]
    for item in report["hydrostatics"][1:]:
        item["interior_error_change"] = (
            item["interior_pressure_error"] / base["interior_pressure_error"] - 1.0)
    base_impact = report["impact"][0]
    for item in report["impact"][1:]:
        item["clumping_change"] = (
            item["min_separation_over_dx"] / base_impact["min_separation_over_dx"] - 1.0)
        item["cost_ratio"] = item["elapsed_seconds"] / base_impact["elapsed_seconds"]

    text = json.dumps(report, indent=2)
    if args.json_out:
        directory = os.path.dirname(args.json_out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_out, "w") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
