"""Reduced-window slamming probe: one admissible Wagner-anchored reading.

Why this file exists
--------------------
`benchmarks/slamming/run.py` is the campaign driver (p129w1r2-n08).  Its
``cases`` stage is nine runs of ``dx = 2 mm`` x 41 frames, i.e. ``t_end =
2 c_end tan(beta) / (pi V)`` with ``c_end = 50 mm``: 445 time steps of
~0.25 s each on 12 000 fluid particles (measured, 16 threads, float64 CPU),
so ~110 s *per case*.  No single case therefore fits a 120 s local shell
window, and the campaign's cluster submissions (cj888-cj890) sat in a queue
with 186 jobs ahead for 283 min without starting, which is why the slamming
evidence was still missing at HEAD.

This probe is the *honest reduced-window* version of the same measurement: the
Wagner self-similar stage means ``C_s = F / (0.5 rho V^2 c)`` is time
independent, so shrinking the observation window from ``c_end = 50 mm`` to
``c_end = 25 mm`` halves the step count without changing the quantity being
measured.  What it *cannot* do is the resolution ladder (``dx = 0.375 mm``
needs ~3.4e5 particles) or the nine-point beta x V matrix.

Declared deviations from the campaign design (all of them are limits, not
tuning):
  * ``c_end = 25 mm`` instead of 50 mm  -> 12.5 particles across the wetted
    half width at ``dx = 2 mm``; the campaign's finest ladder point is 200.
  * one beta x V point (20 deg, 2 m/s) instead of nine.
  * the ``ladder`` stage (observed order of the force peak) is not run.

Usage
-----
    PYTHONPATH=src python benchmarks/slamming/reduced_window_probe.py \
        --out docs/slamming_reduced_window --threads 16
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.slamming import SlammingConfig, SlammingImpact
from tensorsph.wedge_entry import wagner_c_coeff, wet_half_width

ROOT = Path(__file__).resolve().parents[2]
C_WINDOW = 0.5          # same self-similar window rule as run.py


def metrics(series: list[dict], cfg: SlammingConfig) -> dict:
    """Same definitions as ``benchmarks/slamming/run.py::metrics``."""
    post = [s for s in series if s["t"] > 0.0 and s["c"] > 0.0]
    win = [s for s in post if s["c"] >= C_WINDOW * cfg.c_end] or post
    if not win:
        return {}
    cs = [s["c_s"] for s in win]
    forces = [s["force"] for s in win]
    c_err = [
        abs(s["c"] - wet_half_width(cfg.velocity, cfg.beta, s["t"]))
        / wet_half_width(cfg.velocity, cfg.beta, s["t"])
        for s in win
    ]
    noise = [s["noise"] for s in win if s["noise"] == s["noise"]]
    shape = [s["shape_l2"] for s in win if s["shape_l2"] == s["shape_l2"]]
    mean = sum(cs) / len(cs)
    return {
        "n_samples": len(win),
        "c_window_min": C_WINDOW * cfg.c_end,
        "c_s_mean": mean,
        "c_s_max": max(cs),
        "c_s_std": (sum((x - mean) ** 2 for x in cs) / len(cs)) ** 0.5,
        "c_s_wagner": math.pi ** 2 / math.tan(cfg.beta),
        "c_s_rel_err_mean": (mean - math.pi ** 2 / math.tan(cfg.beta))
        / (math.pi ** 2 / math.tan(cfg.beta)),
        "c_s_rel_err_max": (max(cs) - math.pi ** 2 / math.tan(cfg.beta))
        / (math.pi ** 2 / math.tan(cfg.beta)),
        "force_max": max(forces),
        "c_rel_err_mean": sum(c_err) / len(c_err),
        "c_rel_err_max": max(c_err),
        "noise_rms": (sum(n * n for n in noise) / len(noise)) ** 0.5
        if noise else float("nan"),
        "shape_l2_mean": sum(shape) / len(shape) if shape else float("nan"),
        "mass_drift_max": max(s["mass_drift"] for s in win),
        "p_max_peak": max(s["p_max"] for s in win),
        "p_max_wagner": win[0]["p_max_wagner"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/slamming_reduced_window")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--velocity", type=float, default=2.0)
    ap.add_argument("--dx", type=float, default=2.0e-3)
    ap.add_argument("--c-end", type=float, default=0.025)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="truncate the run (declared in the artefact)")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    cfg = SlammingConfig(beta_deg=args.beta, velocity=args.velocity, dx=args.dx,
                         c_end=args.c_end, frames=args.frames,
                         delta_sph=0.1, mls_density=True)
    cfg.validate()
    sim = SlammingImpact(cfg)
    t_end, dt = cfg.resolved_t_end(), sim.dt()
    n_steps = int(round(t_end / dt))
    if args.max_steps:
        n_steps, t_end = min(n_steps, args.max_steps), min(t_end, args.max_steps * dt)
    # Manual loop (instead of ``sim.run``) so the probe reports its own cost:
    # the campaign driver's silent 0.25 s/step is what made the evidence
    # un-landable in the first place.
    import time
    times = torch.linspace(0.0, t_end, args.frames).tolist()
    series: list[dict] = []
    t0 = time.time()
    for k in range(n_steps + 1):
        while len(series) < len(times) and sim.time >= times[len(series)] - 1e-18:
            series.append(sim.slamming_sample())
        if k == n_steps:
            break
        sim.advance(dt)
        if (k + 1) % 20 == 0:
            print(f"[probe] step {k + 1}/{n_steps} t={sim.time:.5f} "
                  f"{time.time() - t0:.1f} s", flush=True)
    while len(series) < len(times):
        series.append(sim.slamming_sample())
    wall = time.time() - t0
    print(f"[probe] {n_steps} steps in {wall:.1f} s "
          f"({wall / max(n_steps, 1):.3f} s/step)", flush=True)
    m = metrics(series, cfg)

    payload = {
        "probe": "slamming_reduced_window",
        "config": {
            "beta_deg": cfg.beta_deg, "velocity": cfg.velocity, "dx": cfg.dx,
            "c_end": cfg.c_end, "frames": cfg.frames, "delta_sph": cfg.delta_sph,
            "mls_density": cfg.mls_density, "rho0": cfg.rho0,
            "sound_speed_factor": cfg.sound_speed_factor,
            "alpha_visc": cfg.alpha_visc, "h_factor": cfg.h_factor,
        },
        "derived": {
            "t_end": t_end, "dt": dt, "steps": int(round(t_end / dt)),
            "n_fluid": int(sim.n_fluid), "n_wall": int(sim.wall_local.shape[0]),
            "steps_run": n_steps, "wall_seconds": wall,
            "seconds_per_step": wall / max(n_steps, 1),
            "window_truncated": bool(args.max_steps),
            "c_coeff_wagner": wagner_c_coeff(cfg.beta),
            "particles_across_wetted_half_width": cfg.c_end / cfg.dx,
        },
        "metrics": m,
        "series": [{k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in s.items()} for s in series],
        "limits": [
            "one beta x V point (20 deg, 2 m/s), not the nine-case matrix",
            "c_end = 25 mm instead of 50 mm: 12.5 particles across the wetted "
            "half width at dx = 2 mm",
            "no resolution ladder, so no observed order for the force peak",
        ],
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({"derived": payload["derived"], "metrics": m}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())