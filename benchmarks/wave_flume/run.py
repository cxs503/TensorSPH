"""Run one wave-flume case and write a JSON record.

Usage (from the repository root)::

    python benchmarks/wave_flume/run.py --help

The record holds everything the analysis step needs (gauge series, paddle
history, free-surface profiles at the snapshot times, diagnostics) and nothing
that is specific to a machine, so ``analyse.py`` can grade it later and on another
host.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.wave_flume import (  # noqa: E402
    WaveFlumeConfig,
    flume_wave_spec,
    run_case,
)


def _step(args) -> float:
    """Acoustic Courant step of the case (the driver clamps to the same value)."""
    h = 1.3 * args.depth / args.depth_cells
    return min(args.dt, args.max_courant * h / args.sound_speed)


def build_config(args) -> WaveFlumeConfig:
    """Case geometry and resolution from CLI arguments.

    Resolution is given as a number of particles through the still-water depth,
    because the vertical lattice has to divide the depth exactly
    (``validate_flume`` rejects anything else).
    """
    depth = args.depth
    dx = depth / args.depth_cells
    height = args.height_ratio * depth
    if args.wave_period is not None:
        # Keep one wavelength inside the propagation zone: pick the flume length
        # from the wavelength so every case probes the same number of periods.
        from tensorsph.wave_flume import wavelength
        lam = wavelength(args.wave_period, depth, args.gravity)
        flume = max(args.flume_length, 1.35 * lam + 0.9 * lam)
    else:
        flume = args.flume_length
    sponge = 0.9 if args.wave_period is None else max(0.9, 0.75 * lam)
    if args.sponge_length is not None:
        sponge = args.sponge_length
        flume = max(flume, 2.2 * sponge)
    if sponge >= 0.5 * flume:
        flume = 3.0 * sponge
    # Both the depth and the flume length must be whole numbers of particle
    # spacings, otherwise the column next to a wall is under-supported and kicked
    # away from it at start-up (see validate_flume).
    flume = max(round(flume / dx), int(round(2.5 * depth / dx))) * dx
    sponge = min(sponge, 0.4 * flume)
    cfg = WaveFlumeConfig(
        flume_length=flume,
        depth=depth,
        wave_type=args.case,
        wave_height=height,
        wave_period=args.wave_period or 1.0,
        dx=dx,
        # The acoustic Courant limit fixes the step; take it here so that
        # ``--simulate-time`` is the time actually simulated (the driver also
        # clamps dt, and a step shorter than requested would silently cut the run
        # short because the step count is fixed).
        dt=_step(args),
        steps=int(math.ceil(args.simulate_time / _step(args))),
        output_every=args.output_every,
        gravity=args.gravity,
        sponge_length=sponge,
        sponge_strength=args.sponge_strength,
        active_absorption=args.active_absorption,
        settle_time=args.settle,
        xsph=args.xsph,
        xsph_settle=args.xsph_settle,
        viscosity_settle=args.viscosity_settle,
        sound_speed=args.sound_speed,
        ramp_periods=args.ramp_periods,
        viscosity=args.viscosity,
        density_diffusion=args.density_diffusion,
        device=args.device,
    )
    return cfg


def snapshot_times_for(cfg: WaveFlumeConfig, args) -> list:
    """Evenly spaced snapshots over the second half of the run.

    The solitary-wave grading needs the free-surface profile while the wave is in
    the test section, well before it reaches the sponge; four snapshots cover that
    window for every resolution.
    """
    end = args.simulate_time
    count = max(1, args.snapshots)
    return [round(end * (0.45 + 0.4 * i / max(1, count - 1)), 6) for i in range(count)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("regular", "solitary"), default="regular")
    parser.add_argument("--depth", type=float, default=0.25)
    parser.add_argument("--depth-cells", type=int, default=20,
                        help="particles through the still-water depth (sets dx)")
    parser.add_argument("--height-ratio", type=float, default=0.16,
                        help="wave height / still-water depth")
    parser.add_argument("--wave-period", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--flume-length", type=float, default=3.2)
    parser.add_argument("--sound-speed", type=float, default=30.0)
    parser.add_argument("--viscosity", type=float, default=1.0e-6)
    parser.add_argument("--density-diffusion", type=float, default=0.1)
    parser.add_argument("--sponge-strength", type=float, default=4.0)
    parser.add_argument("--active-absorption", type=float, default=0.0)
    parser.add_argument("--ramp-periods", type=float, default=2.0)
    parser.add_argument("--sponge-length", type=float, default=None,
                        help="override the derived sponge length")
    parser.add_argument("--xsph", type=float, default=0.0,
                        help="XSPH velocity-smoothing coefficient")
    parser.add_argument("--xsph-settle", type=float, default=0.0)
    parser.add_argument("--viscosity-settle", type=float, default=0.0)
    parser.add_argument("--settle", type=float, default=0.0,
                        help="seconds the paddle is held at rest before the wave")
    parser.add_argument("--dt", type=float, default=4.0e-4)
    parser.add_argument("--max-courant", type=float, default=0.25)
    parser.add_argument("--simulate-time", type=float, default=5.0)
    parser.add_argument("--output-every", type=int, default=40)
    parser.add_argument("--snapshots", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    cfg = build_config(args)
    spec = flume_wave_spec(cfg)
    stamps = snapshot_times_for(cfg, args)
    try:
        run = run_case(cfg, snapshot_times=stamps)
    except (FloatingPointError, ValueError) as exc:
        # A case that goes numerically unstable is a *result*, not a crash: the
        # record says so and the analysis reports it instead of losing the run.
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "case": args.case, "failed": True, "error": f"{type(exc).__name__}: {exc}",
            "config": {"dx": cfg.dx, "steps": cfg.steps, "simulate_time": args.simulate_time},
        }))
        print(f"FAILED {out}: {exc}")
        return 2

    record = {
        "case": args.case,
        "config": run.config,
        "spec": {
            "kind": spec.kind,
            "height": spec.height,
            "depth": spec.depth,
            "gravity": spec.gravity,
            "stroke": spec.stroke,
            "celerity": spec.celerity,
            "k": spec.k,
            "omega": spec.omega,
            "period": spec.period,
            "wavelength": spec.wavelength,
            "length_scale": spec.length_scale,
        },
        "diagnostics": {
            "steps_taken": run.steps_taken,
            "simulated_time": run.simulated_time,
            "dt": run.dt,
            "mass": run.mass,
            "escaped": run.escaped,
            "max_speed": run.max_speed,
            "max_density_compression": run.max_density_compression,
            "min_density_ratio": run.min_density_ratio,
            "rebuilds": run.rebuilds,
        },
        "probe_x": list(run.probe_x),
        "times": run.times,
        "probe_eta": run.probe_eta,
        "paddle_offset": run.paddle_offset,
        "speed_history": run.speed_history,
        "compression_history": run.compression_history,
        "profiles": [
            {"time": t, "x": x.tolist(), "eta": eta.tolist()}
            for t, x, eta in run.profiles
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB): "
          f"{run.steps_taken} steps, {run.simulated_time:.3f} s, "
          f"{run.config['particle_count']} particles, "
          f"max|v| {run.max_speed:.3f} m/s, compression "
          f"{run.max_density_compression:.2e}")
    if not math.isfinite(run.max_speed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())