"""B1 hydrostatic benchmark: static pool against the analytical pressure profile.

The reference for a pool of depth H at rest is ``p(y) = rho0 * g * (H - y)``.

Two effects must be separated when reading the result:

1. **Acoustic oscillation.** A lattice initialised at uniform density is not in
   discrete equilibrium, so the pool sloshes for a while. The instantaneous
   pressure profile therefore oscillates by tens of percent even after the
   velocity has decayed. The measured metric is the *time average* over the
   second half of the run, which removes this oscillation.
2. **Free-surface band.** In the topmost band the density is truncated by the
   free surface, so SPH pressure there is not comparable to a hydrostatic
   reference. That band is reported separately and excluded from the interior
   gate.

Usage:
    python benchmarks/hydrostatic/run.py --seconds 4 --boundary wall
    python benchmarks/hydrostatic/run.py --boundary clamp --seconds 4

Output is JSON so it can be recorded as run evidence. Exit status is non-zero
when a gate fails.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    make_particles,
    rhs,
    wall_lattice,
)


def bands_of(pos, cfg, count):
    edges = [cfg.pool_depth * k / count for k in range(count + 1)]
    return [(low, high, low == 0.0, k == count - 1)
            for k, (low, high) in enumerate(zip(edges[:-1], edges[1:]))]


def band_mean(values, pos, low, high):
    mask = (pos[:, 1] >= low) & (pos[:, 1] < high)
    if not bool(mask.any()):
        return None, 0, float("nan")
    return float(values[mask].mean()), int(mask.sum()), float(pos[mask, 1].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="simulated seconds (default 4)")
    parser.add_argument("--dt", type=float, default=None,
                        help="default: the largest step allowed by --max-courant")
    parser.add_argument("--max-courant", type=float, default=0.25)
    parser.add_argument("--boundary", default="wall", choices=("wall", "clamp"))
    parser.add_argument("--dx", type=float, default=0.025)
    parser.add_argument("--bands", type=int, default=7)
    parser.add_argument("--average-fraction", type=float, default=0.5,
                        help="fraction of the run to time-average over")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-interior-error", type=float, default=0.05,
                        help="fail if the mean |error| of interior bands exceeds this")
    parser.add_argument("--max-surface-drop", type=float, default=0.05,
                        help="fail if the free surface sinks more than this fraction of H")
    parser.add_argument("--hydrostatic-init", action="store_true",
                        help="assign masses from the analytical hydrostatic profile "
                             "(start from a discrete equilibrium instead of uniform density)")
    args = parser.parse_args()

    cfg = Config(dx=args.dx, boundary=args.boundary, device=args.device,
                 max_courant=args.max_courant,
                 hydrostatic_init=args.hydrostatic_init)
    cfg.dt = (args.dt if args.dt is not None
              else cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed)
    steps = max(int(round(args.seconds / cfg.dt)), 2)

    device = torch.device(args.device)
    pos, mass = make_particles(cfg, device)
    keep = pos[:, 1] <= cfg.pool_depth  # pool only: this is a static-water test
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    initial_mass = float(mass.sum())
    initial_surface = float(pos[:, 1].max())
    wall = wall_lattice(cfg, device) if cfg.boundary == "wall" else None

    vel = torch.zeros_like(pos)
    average_from = int((1.0 - args.average_fraction) * steps)
    pressure_sum = torch.zeros(pos.shape[0])
    samples = 0
    escaped = 0
    started = time.perf_counter()
    for step in range(steps):
        _, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall)
        if not torch.isfinite(acc).all():
            raise SystemExit("ERROR: non-finite acceleration in the hydrostatic case")
        if step >= average_from:
            pressure_sum = pressure_sum + pressure
            samples += 1
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        escaped += apply_boundary(pos, vel, cfg)
    elapsed = time.perf_counter() - started
    if samples == 0:
        raise SystemExit("ERROR: no samples collected; increase --seconds")

    pressure_mean = pressure_sum / samples
    rho, pressure_now, _ = rhs(pos, vel, mass, cfg, wall=wall)

    profile, interior_errors = [], []
    for low, high, is_bottom, is_top in bands_of(pos, cfg, args.bands):
        measured, count, y = band_mean(pressure_mean, pos, low, high)
        if measured is None:
            continue
        analytic = cfg.rho0 * cfg.gravity * (cfg.pool_depth - y)
        error = (measured - analytic) / analytic if analytic > 0 else float("nan")
        record = {
            "y": y, "band": [low, high], "particles": count,
            "pressure_time_averaged": measured,
            "pressure_analytic": analytic,
            "relative_error": error,
            "kind": "free_surface" if is_top else ("interior" if not is_bottom else "interior"),
        }
        profile.append(record)
        if not is_top:
            interior_errors.append(abs(error))

    interior_error = (sum(interior_errors) / len(interior_errors)
                      if interior_errors else float("nan"))
    surface_drop = (initial_surface - float(pos[:, 1].max())) / cfg.pool_depth
    interior_ok = interior_error <= args.max_interior_error
    surface_ok = surface_drop <= args.max_surface_drop

    report = {
        "benchmark": "B1_static_pool",
        "boundary": cfg.boundary,
        "hydrostatic_init": cfg.hydrostatic_init,
        "initial_mass_kg": initial_mass,
        "particles": int(pos.shape[0]),
        "wall_particles": 0 if wall is None else int(wall.shape[0]),
        "dt": cfg.dt,
        "acoustic_courant": cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx),
        "steps": steps,
        "simulated_time": steps * cfg.dt,
        "averaged_over_seconds": args.average_fraction * steps * cfg.dt,
        "elapsed_seconds": elapsed,
        "initial_surface_y": initial_surface,
        "final_surface_y": float(pos[:, 1].max()),
        "surface_drop_fraction_of_depth": surface_drop,
        "residual_max_speed": float(torch.linalg.vector_norm(vel, dim=-1).max()),
        "density_min": float(rho.min()),
        "density_max": float(rho.max()),
        "hydrostatic_bottom_pressure": cfg.rho0 * cfg.gravity * cfg.pool_depth,
        "mean_abs_relative_error_interior_bands": interior_error,
        "profile": profile,
        "gates": {
            "interior_pressure_profile": {
                "metric": "mean |p_sim - rho0 g (H-y)| / (rho0 g (H-y)), interior bands",
                "measured": interior_error, "limit": args.max_interior_error,
                "result": "pass" if interior_ok else "fail"},
            "surface_drop_fraction": {
                "metric": "(initial - final) free-surface height / pool depth",
                "measured": surface_drop, "limit": args.max_surface_drop,
                "result": "pass" if surface_ok else "fail"},
        },
        "clamped_particles_beyond_box": escaped,
        "notes": [
            "B1 verifies the pressure field of a resting pool; it is not physical "
            "validation of a free-surface flow.",
            "The instantaneous profile oscillates by tens of percent because a "
            "uniform-density lattice is not in discrete equilibrium. The reported "
            "profile is a time average over the second half of the run.",
            "The topmost band is labelled free_surface and excluded from the "
            "interior gate: density summation is truncated there by construction.",
            "Mass is conserved exactly: particle masses never change.",
            "Residual motion does not decay to zero; this prototype has free-slip "
            "walls and no state relaxation.",
        ],
    }
    print(json.dumps(report, indent=2))
    if not (interior_ok and surface_ok):
        raise SystemExit("ERROR: hydrostatic gates failed")


if __name__ == "__main__":
    main()
