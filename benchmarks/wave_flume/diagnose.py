"""Stability ladder for the regular-wave flume case (CASE-WAVE-FLUME).

The shipped D16 record was graded at +65 % wave height and a local D20 record
carried ``max|v| = 109 m/s`` against ``c0 = 30 m/s`` and 45 % density
compression, i.e. the flume was not merely under-resolved -- it was blowing up.
This driver isolates *which* knob controls that: it runs a short regular-wave
case with one configuration change at a time and writes the stability metrics
(acoustic Mach number, density compression, escaped particles, ceiling hits,
free-surface statistics) plus the graded wave height of the same record.

Usage (repository root)::

    PYTHONPATH=src python benchmarks/wave_flume/diagnose.py --out /tmp/diag.json \
        --variant base --variant xsph --simulate-time 2.0 --threads 16

Every variant is a pure kwargs override on top of the same base case, so the
only thing that differs between variants is the knob named in ``variant``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.wave_flume import (  # noqa: E402
    WaveFlumeConfig,
    flume_wave_spec,
    run_case,
)
from tensorsph import wave_flume_grade as grade  # noqa: E402


VARIANTS = {
    # The configuration used by the local D20 record that blew up.
    "base": {},
    # Velocity smoothing during the wave phase: the documented short-wavelength
    # dissipator (rate ~ 2 nu k^2), intended for the particle-scale churn.
    "xsph": {"xsph": 0.05},
    "xsph10": {"xsph": 0.10},
    # Half the acoustic step: tests whether the blow-up is an integration error.
    "half_dt": {"_dt_scale": 0.5},
    # Longer settle window with stronger start-up damping.
    "settle2": {"settle_time": 2.0, "viscosity_settle": 0.05, "xsph_settle": 0.2},
    # More compressible-tolerant acoustic speed (larger c0 -> smaller dt, but
    # lower Mach number at the same velocities).
    "c0_50": {"sound_speed": 50.0},
    # Shorter sponge (the passive layer damps velocity but removes no mass).
    "sponge_short": {"sponge_length": 0.6},
    # Wall layers / boundary handling: thicker wall support.
    "wall3": {"wall_layers": 3},
    # Proportional paddle absorption (impedance relation u = eta sqrt(g/h)):
    # the paddle is passive by default, so it reflects whatever the sponge lets
    # through and a partially standing wave builds up in the test section.
    "active": {"active_absorption": 1.0},
    # Sponge of two wavelengths in a longer flume (the passive layer is 0.75
    # lambda by default, too short to absorb a 1 s wave).
    "long_sponge": {"flume_length": 5.2, "sponge_length": 2.6},
    "active_long": {"flume_length": 5.2, "sponge_length": 2.6,
                    "active_absorption": 1.0},
    # Both absorbing measures plus the wave-phase velocity smoothing.
    "active_xsph": {"active_absorption": 1.0, "xsph": 0.05},
}


def stability_verdict(metrics: dict, sound_speed: float,
                      compression_limit: float = 0.01,
                      mach_limit: float = 0.5) -> dict:
    """Judge whether a flume record is a *wave* or a numerical event.

    Two independent red flags are reported, because they fail differently:

    * ``mach`` = ``max|v| / c0``.  Weakly compressible SPH assumes ``M << 1``;
      a record whose peak particle speed exceeds half the artificial sound speed
      is not resolving an acoustic-free gravity wave any more.
    * ``compression`` = ``max(rho/rho0) - 1``.  Tait's equation of state is used
      at O(1 %) density variation; 45 % is a shock in a gravity-wave case.

    Both are reported with the measured value *and* the flag, so a caller can
    print a number instead of a boolean.
    """
    speed = float(metrics.get("max_speed", float("nan")))
    comp = float(metrics.get("max_compression", float("nan")))
    mach = speed / sound_speed if sound_speed > 0 else float("nan")
    return {
        "mach": mach,
        "compression": comp,
        "mach_flag": bool(mach > mach_limit),
        "compression_flag": bool(comp > compression_limit),
        "stable": not (mach > mach_limit or comp > compression_limit),
    }


def build_config(args, overrides: dict) -> tuple[WaveFlumeConfig, float]:
    depth = args.depth
    dx = depth / args.depth_cells
    height = args.height_ratio * depth
    # validate_flume rejects a partial cell: anchor the box on the lattice.
    flume = round(args.flume_length / dx) * dx
    cfg = WaveFlumeConfig(
        flume_length=flume,
        depth=depth,
        wave_type="regular",
        wave_height=height,
        wave_period=args.wave_period,
        dx=dx,
        dt=args.dt,
        gravity=args.gravity,
        sponge_length=args.sponge_length if args.sponge_length else 0.75 * 1.3038,
        sponge_strength=args.sponge_strength,
        density_diffusion=args.density_diffusion,
        sound_speed=args.sound_speed,
        viscosity=args.viscosity,
        settle_time=args.settle,
        viscosity_settle=args.viscosity_settle,
        xsph_settle=args.xsph_settle,
        ramp_periods=args.ramp_periods,
        output_every=args.output_every,
        steps=1,
    )
    scale = 1.0
    for key, value in overrides.items():
        if key == "_dt_scale":
            scale = float(value)
            continue
        setattr(cfg, key, value)
    return cfg, scale


def run_variant(name: str, args) -> dict:
    cfg, scale = build_config(args, VARIANTS[name])
    spec = flume_wave_spec(cfg)
    h = cfg.h_factor * cfg.dx
    step = min(args.dt, args.max_courant * h / cfg.sound_speed) * scale
    cfg.steps = int(math.ceil(args.simulate_time / step))
    started = time.time()
    result = {"variant": name, "overrides": VARIANTS[name],
              "steps": cfg.steps, "dt": step, "dx": cfg.dx,
              "height_cells": cfg.wave_height / cfg.dx,
              "target_height": cfg.wave_height,
              "wavelength": spec.wavelength}
    try:
        run = run_case(cfg)
    except (FloatingPointError, ValueError) as exc:
        result.update({"failed": True, "error": f"{type(exc).__name__}: {exc}",
                       "wall_seconds": time.time() - started})
        return result
    elapsed = time.time() - started
    times = list(run.times)
    ceiling = cfg.free_surface_top() - cfg.depth
    series = [list(map(float, s)) for s in run.probe_eta]
    finite = [sum(1 for v in s if math.isfinite(v)) for s in series]
    top = [max((v for v in s if math.isfinite(v)), default=float("nan")) for s in series]
    bottom = [min((v for v in s if math.isfinite(v)), default=float("nan")) for s in series]
    mean = []
    for s in series:
        vals = [v for v in s if math.isfinite(v)]
        mean.append(sum(vals) / len(vals) if vals else float("nan"))
    # Graded heights over the late half of the run (steady-state window).
    window = (0.5 * times[-1], times[-1])
    heights = []
    for s in series:
        try:
            block = grade.wave_height(times, s, start=window[0],
                                      end=window[1], omega=spec.omega)
            heights.append({"height": block["height"],
                            "error": block["height"] / cfg.wave_height - 1.0,
                            "height_crossings": block.get("height_crossings")})
        except ValueError as exc:
            heights.append({"error_text": str(exc)})
    speed = list(map(float, run.speed_history))
    comp = list(map(float, run.compression_history))
    result.update({
        "failed": False,
        "wall_seconds": elapsed,
        "particles": run.config["particle_count"],
        "wall_particles": run.config["wall_count"],
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "dt_final": run.dt,
        "max_speed": run.max_speed,
        "max_mach": run.max_speed / cfg.sound_speed,
        "max_compression": run.max_density_compression,
        "min_density_ratio": run.min_density_ratio,
        "escaped": run.escaped,
        "ceiling_elevation": ceiling,
        "ceiling_hits": sum(1 for v in top if v > 0.95 * ceiling),
        "probe_finite_fraction": [f / len(s) for f, s in zip(finite, series)],
        "probe_top": top,
        "probe_bottom": bottom,
        "probe_mean": mean,
        "max_speed_history": speed[:: max(1, len(speed) // 20)],
        "compression_history": comp[:: max(1, len(comp) // 20)],
        "graded": heights,
        "worst_error": max((abs(g["error"]) for g in heights if "error" in g),
                           default=float("nan")),
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS),
                        help="repeatable; default = all")
    parser.add_argument("--variant-index", type=int, default=None,
                        help="1-based index into the sorted variant names "
                             "(for sharded cluster runs)")
    parser.add_argument("--depth", type=float, default=0.25)
    parser.add_argument("--depth-cells", type=int, default=20)
    parser.add_argument("--height-ratio", type=float, default=0.16)
    parser.add_argument("--wave-period", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--flume-length", type=float, default=3.2)
    parser.add_argument("--sponge-length", type=float, default=None)
    parser.add_argument("--sponge-strength", type=float, default=4.0)
    parser.add_argument("--sound-speed", type=float, default=30.0)
    parser.add_argument("--viscosity", type=float, default=1.0e-6)
    parser.add_argument("--density-diffusion", type=float, default=0.1)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--viscosity-settle", type=float, default=0.02)
    parser.add_argument("--xsph-settle", type=float, default=0.1)
    parser.add_argument("--ramp-periods", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=4.0e-4)
    parser.add_argument("--max-courant", type=float, default=0.25)
    parser.add_argument("--simulate-time", type=float, default=2.0)
    parser.add_argument("--output-every", type=int, default=25)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    if args.variant_index is not None:
        names = [sorted(VARIANTS)[args.variant_index - 1]]
    else:
        names = args.variant or sorted(VARIANTS)
    results = []
    for name in names:
        block = run_variant(name, args)
        results.append(block)
        if block.get("failed"):
            print(f"{name:14s} FAILED  {block['error']}  ({block['wall_seconds']:.0f}s)")
        else:
            print(f"{name:14s} M={block['max_mach']:.2f} comp={block['max_compression']:.3f} "
                  f"esc={block['escaped']:3d} ceil={block['ceiling_hits']} "
                  f"worst_err={block['worst_error']:+.1%} ({block['wall_seconds']:.0f}s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case": "wave_flume_stability_ladder",
                               "settings": {k: v for k, v in vars(args).items()},
                               "results": results}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())