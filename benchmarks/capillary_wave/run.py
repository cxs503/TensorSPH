"""Capillary wave benchmark: surface-tension validation against analytical frequency.

A sinusoidal interface perturbation in a 2-D periodic domain oscillates at the
capillary frequency:

    omega^2 = sigma * k^3 / rho

where sigma is the surface tension coefficient, k = 2*pi/lambda is the
wavenumber, and rho is the density. This is the classical result for an
inviscid, deep-water capillary wave (Lamb, 1932, Art. 267).

The benchmark measures the oscillation period of the interface amplitude and
compares it against the analytical prediction.

Usage:
    python benchmarks/capillary_wave/run.py --sigma 0.072 --wavelength 0.2 --amplitude 0.005
    python benchmarks/capillary_wave/run.py --quick  # short run for CI

Exit status is non-zero when the frequency error exceeds the threshold.
"""
# NOTE (verification audit, see docs/verification_droplet_oscillation.md §6):
#     THIS GATE CANNOT PRODUCE A MEASUREMENT.  ``track_amplitude`` returns
#     ``max|y - y0|``, which is non-negative, while ``find_period`` only detects
#     sign changes -- the period is therefore always ``None`` and ``pass`` always
#     ``False``.  The default window (0.2-0.5 s) is also 12 % of the 4.2 s
#     analytic period, and with ``gravity=0``, no periodic boundary and a
#     one-wavelength domain the flat pool contracts under surface tension instead
#     of supporting a plane wave.  The working dynamic gate is
#     ``benchmarks/droplet_oscillation``; this file is kept only as the record of
#     the defect.  Do not cite it as evidence of anything.

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    hydrostatic_density,
    next_dt,
    resolve_device,
    rhs,
    wall_lattice,
)
from tensorsph.neighbor_search import PairListCache


def analytical_frequency(sigma: float, k: float, rho: float) -> float:
    """Analytical capillary wave frequency: omega = sqrt(sigma * k^3 / rho)."""
    return math.sqrt(sigma * k**3 / rho)


def make_capillary_wave(cfg: Config, wavelength: float, amplitude: float,
                        device: torch.device):
    """Create a sinusoidal interface perturbation.

    The domain is [0, pool_width] x [0, pool_depth]. The interface is at
    y = pool_depth/2 + amplitude * sin(2*pi*x/wavelength).
    """
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, cfg.pool_width, dx, device=device)
    ys = torch.arange(dx * 0.5, cfg.pool_depth, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)

    # Interface at mid-depth with sinusoidal perturbation
    interface_y = cfg.pool_depth / 2.0 + amplitude * torch.sin(
        2.0 * math.pi * positions[:, 0] / wavelength)
    # Particles below the interface are fluid
    fluid_mask = positions[:, 1] < interface_y
    positions = positions[fluid_mask]

    # Hydrostatic initialization
    mass = hydrostatic_density(positions[:, 1], cfg,
                               surface=cfg.pool_depth) * dx * dx
    return positions, mass


def track_amplitude(positions: torch.Tensor, cfg: Config, wavelength: float,
                    interface_y0: float) -> float:
    """Estimate the interface amplitude from the particle positions.

    Uses the maximum deviation of the topmost fluid particles from the mean
    interface level.
    """
    # Find particles near the interface (within 3*dx of the mean level)
    margin = 3.0 * cfg.dx
    near_interface = torch.abs(positions[:, 1] - interface_y0) < margin
    if not near_interface.any():
        return 0.0

    interface_particles = positions[near_interface]
    # Group by x-position and find the max y in each group
    # Simple approach: use the max y deviation from the mean
    deviations = interface_particles[:, 1] - interface_y0
    return float(deviations.abs().max())


def run_capillary_wave(cfg: Config, wavelength: float, amplitude: float,
                       steps: int, output_every: int = 10):
    """Integrate the capillary wave and track the amplitude."""
    device = resolve_device(cfg.device)
    positions, mass = make_capillary_wave(cfg, wavelength, amplitude, device)
    velocity = torch.zeros_like(positions)
    interface_y0 = cfg.pool_depth / 2.0

    h = cfg.h_factor * cfg.dx
    fluid_count = positions.shape[0]
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h, fluid_count=fluid_count)

    amplitudes: list[float] = []
    times: list[float] = []
    dt = cfg.dt
    elapsed = 0.0

    for step in range(steps + 1):
        all_positions = positions.clone()
        pairs, _ = cache.get(all_positions)

        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs,
                              wall=None, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")

        if step % output_every == 0 or step == steps:
            amp = track_amplitude(positions, cfg, wavelength, interface_y0)
            amplitudes.append(amp)
            times.append(elapsed)

        velocity = velocity + dt * acc
        positions = positions + dt * velocity
        elapsed += dt

        if cfg.adaptive_dt:
            max_speed = float(torch.linalg.vector_norm(velocity, dim=-1).max())
            dt = next_dt(cfg, dt, max_speed)

    return times, amplitudes


def find_period(times: list[float], amplitudes: list[float]) -> float | None:
    """Estimate the oscillation period from zero-crossings of the amplitude.

    The amplitude oscillates around zero (the mean interface level). We find
    the times when the amplitude crosses zero and compute the period from
    consecutive crossings.
    """
    if len(amplitudes) < 4:
        return None

    # Find zero crossings
    crossings = []
    for i in range(1, len(amplitudes)):
        if amplitudes[i - 1] * amplitudes[i] < 0:
            # Linear interpolation to find the crossing time
            t0, t1 = times[i - 1], times[i]
            a0, a1 = amplitudes[i - 1], amplitudes[i]
            t_cross = t0 - a0 * (t1 - t0) / (a1 - a0)
            crossings.append(t_cross)

    if len(crossings) < 2:
        return None

    # Period is twice the average time between consecutive crossings
    # (since each crossing is half a period apart)
    intervals = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    half_period = sum(intervals) / len(intervals)
    return 2.0 * half_period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=0.072,
                        help="surface tension coefficient (N/m)")
    parser.add_argument("--wavelength", type=float, default=0.2,
                        help="perturbation wavelength (m)")
    parser.add_argument("--amplitude", type=float, default=0.005,
                        help="initial amplitude (m)")
    parser.add_argument("--dx", type=float, default=0.005,
                        help="particle spacing (m)")
    parser.add_argument("--seconds", type=float, default=0.5,
                        help="simulation time (s)")
    parser.add_argument("--max-freq-error", type=float, default=0.15,
                        help="fail if frequency error exceeds this fraction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true",
                        help="short run for CI (0.2 s)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.quick:
        args.seconds = 0.2

    cfg = Config(
        dx=args.dx,
        pool_width=args.wavelength,  # domain width = wavelength for periodic-like
        pool_depth=0.3,
        sound_speed=20.0,
        gamma=7.0,
        gravity=0.0,  # zero gravity for pure capillary wave
        viscosity=0.0,
        density_diffusion=0.0,
        surface_tension_coefficient=args.sigma,
        device=args.device,
        boundary="clamp",  # no walls needed for this case
        adaptive_dt=True,
        max_courant=0.25,
    )
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(args.seconds / cfg.dt))
    cfg.output_every = max(1, cfg.steps // 50)

    k = 2.0 * math.pi / args.wavelength
    omega_analytical = analytical_frequency(args.sigma, k, cfg.rho0)
    period_analytical = 2.0 * math.pi / omega_analytical

    print(f"Analytical: omega = {omega_analytical:.4f} rad/s, "
          f"T = {period_analytical:.4f} s")
    print(f"Simulation: {cfg.steps} steps, dt = {cfg.dt:.2e} s, "
          f"t_end = {args.seconds:.3f} s")

    started = time.perf_counter()
    times, amplitudes = run_capillary_wave(cfg, args.wavelength, args.amplitude,
                                           cfg.steps, cfg.output_every)
    elapsed = time.perf_counter() - started

    period_measured = find_period(times, amplitudes)

    result = {
        "benchmark": "capillary_wave",
        "setup": {
            "sigma": args.sigma,
            "wavelength": args.wavelength,
            "amplitude": args.amplitude,
            "dx": args.dx,
            "rho0": cfg.rho0,
            "k": k,
        },
        "analytical": {
            "omega_rad_s": omega_analytical,
            "period_s": period_analytical,
        },
        "measured": {
            "period_s": period_measured,
            "elapsed_seconds": elapsed,
            "steps": cfg.steps,
            "n_amplitude_samples": len(amplitudes),
        },
    }

    if period_measured is not None:
        freq_error = abs(period_measured - period_analytical) / period_analytical
        result["measured"]["frequency_error"] = freq_error
        result["pass"] = freq_error < args.max_freq_error
        print(f"Measured period: {period_measured:.4f} s "
              f"(error: {freq_error*100:.1f}%)")
    else:
        result["pass"] = False
        result["measured"]["frequency_error"] = None
        print("Could not determine period from amplitude time series")

    print(f"Wall clock: {elapsed:.1f} s")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)

    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
