"""Rayleigh-Taylor linear growth rate: two-phase WCSPH vs Taylor (1950).

Case
----
A two-layer column, one wavelength wide and periodic in x, heavier fluid on top
(``rho_h = 1500``, ``rho_l = 500`` -> Atwood ``A = 0.5``), single-mode interface
perturbation of amplitude ``eta0 = 0.01 lambda``.  Early-time exponential growth
of the interface is compared with ``sigma = sqrt(A g k)`` (Taylor 1950,
Proc. R. Soc. A 201:192, doi:10.1098/rspa.1950.0052, eq. for the linear stage).

Run::

    PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.025 0.02
    PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.02 --viscosity 1e-2

Writes ``benchmarks/rayleigh_taylor/runs/<tag>.json`` (one per configuration),
``summary.json`` for the whole set, and the figures next to them.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from tensorsph import rayleigh_taylor as rt

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def run_one(cfg: rt.RTConfig, tag: str, tag_suffix: str = "") -> dict:
    t0 = time.time()
    print(f"[{tag}] dx={cfg.dx} nu={cfg.viscosity} eta0={cfg.amplitude} "
          f"c0={cfg.sound_speed} t_end={cfg.end_time}", flush=True)
    result = rt.run_case(
        cfg, snapshot=True,
        progress=lambda t, a, v: print(f"  [{tag}] t={t:.3f} a={a:.5f} "
                                       f"vmax={v:.2f} {time.time() - t0:.0f}s",
                                       flush=True))
    result["wall_seconds"] = time.time() - t0
    result["tag"] = tag + tag_suffix
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{result['tag']}.json"
    out.write_text(json.dumps(result))
    print(f"  [{tag}] sigma={result['fit']['sigma']:.3f} 1/s "
          f"({100 * (result['sigma_over_ideal'] - 1):+.1f}% vs sqrt(Agk)) "
          f"r2={result['fit']['r2']:.4f} in {result['wall_seconds']:.0f}s",
          flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, nargs="+", default=[0.02],
                    help="particle spacings (m) for the convergence study")
    ap.add_argument("--viscosity", type=float, nargs="+", default=[1.0e-3])
    ap.add_argument("--amplitude", type=float, nargs="+", default=[0.01])
    ap.add_argument("--end-time", type=float, default=0.45)
    ap.add_argument("--sound-speed", type=float, default=40.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--time-integrator", type=str, default="euler",
                    choices=["euler", "rk2"], help="Time integrator")
    ap.add_argument("--density-diffusion", type=float, default=0.0,
                    help="δ-SPH density diffusion coefficient")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    summary = []
    for dx in args.grid:
        for nu in args.viscosity:
            for amp in args.amplitude:
                tag = f"dx{dx:g}_nu{nu:g}_a{amp:g}"
                if args.time_integrator != "euler":
                    tag += f"_{args.time_integrator}"
                if args.density_diffusion > 0:
                    tag += f"_dd{args.density_diffusion:g}"
                cfg = rt.RTConfig(dx=dx, viscosity=nu, amplitude=amp,
                                  end_time=args.end_time,
                                  sound_speed=args.sound_speed,
                                  time_integrator=args.time_integrator,
                                  density_diffusion=args.density_diffusion)
                if amp != 0.01:
                    tag += "_amp"
                summary.append(run_one(cfg, tag))
    RUNS.mkdir(parents=True, exist_ok=True)
    slim = [{k: r[k] for k in ("tag", "fit", "fit_velocity_gauge", "reference",
                               "sigma_over_ideal", "sigma_over_finite_depth",
                               "sigma_over_viscous", "sigma_velocity_over_ideal",
                               "max_amplitude", "derived", "wall_seconds")}
            for r in summary]
    (RUNS / "summary.json").write_text(json.dumps(slim, indent=1))
    print("wrote", RUNS / "summary.json")


if __name__ == "__main__":
    main()