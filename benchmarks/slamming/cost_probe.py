#!/usr/bin/env python
"""Per-step cost probe for the slamming campaign (does this machine fit the run?)

Why this file exists
--------------------
`benchmarks/slamming/run.py` is the campaign driver for p129w1r2-n08.  Its
`cases` stage is nine runs of `dx = 2 mm` x 41 frames with `c_end = 50 mm`;
`m14940` measured ~0.25 s/step at `dx = 2 mm` (445 steps -> ~110 s per case, i.e.
more than one local shell window) and reported the per-step cost growing
20-28x as the wedge approaches the free surface, so the whole campaign does not
finish in a single local slot.  Every later attempt to produce the evidence
therefore starts with the same question -- *does this machine fit N steps of
this resolution right now?* -- and, before m16822, answered it by launching the
campaign and watching it time out.

This probe answers that question directly and cheaply: it runs the same solver
(`SlammingImpact`) for a small, *explicit* number of steps at a requested
resolution and writes the measured wall clock, the particle count, `dt`, the
step count and the per-step minima/maxima to one JSON.  It **makes no physical
claim** -- `C_s` is not recorded and nothing here may be graded against Wagner;
it only decides whether a resolution is affordable on the current machine.

A resolution is reported `affordable` when the projected wall clock of the
*campaign* case (`t_end = 2 c_end tan(beta) / (pi V)` at that `dx`, as in
`run.py`) fits `--budget` seconds.  `projection` is the measured mean step cost
scaled by the campaign step count; it is an extrapolation and is labelled as
such in the JSON.

Usage
-----
    python benchmarks/slamming/cost_probe.py --dx 4e-3 --steps 20 \
        --out runs/m16822_slamming_cost/probe_dx4.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.slamming import SlammingConfig, SlammingImpact

ROOT = Path(__file__).resolve().parents[2]


def _dt(impact) -> float:
    """`dt` is a method on `WedgeImpact`; tolerate an attribute too."""
    d = impact.dt
    return float(d() if callable(d) else d)


def campaign_steps(dx: float, beta_deg: float, velocity: float, c_end: float,
                   frames: int) -> tuple[int, float]:
    """Step count and `dt` of the campaign case at this `dx` (see `run.py`).

    `SlammingImpact` only sets `steps` while it runs, so the count is derived
    from the campaign's own `t_end` and the solver's `dt` instead of read back.
    """
    cfg = SlammingConfig(beta_deg=beta_deg, velocity=velocity, dx=dx,
                         frames=frames, c_end=c_end)
    cfg.validate()
    dt = _dt(SlammingImpact(cfg))
    t_end = 2.0 * c_end * math.tan(math.radians(beta_deg)) / (math.pi * velocity)
    return int(round(t_end / dt)), dt


def probe(dx: float, steps: int, beta_deg: float, velocity: float,
          c_end: float, frames: int, delta_sph: float, mls_density: bool,
          threads: int) -> dict:
    """Run the solver for `steps` steps and report the measured cost."""
    # The observation window is shrunk until the solver's own step budget is the
    # requested `steps`: `t_end = 2 c_end tan(beta) / (pi V)`.
    t_end = steps * _dt_of(dx, beta_deg, velocity)
    c_end_probe = t_end * math.pi * velocity / (2.0 * math.tan(math.radians(beta_deg)))
    cfg = SlammingConfig(beta_deg=beta_deg, velocity=velocity, dx=dx,
                         frames=max(2, frames), c_end=c_end_probe,
                         delta_sph=delta_sph, mls_density=mls_density)
    cfg.validate()
    torch.set_num_threads(max(1, threads))
    sim = SlammingImpact(cfg)
    t0 = time.time()
    series = sim.run()
    wall = time.time() - t0
    n_steps = int(getattr(sim, 'steps', 0) or steps)
    per_step = wall / max(n_steps, 1)
    full_steps, dt_full = campaign_steps(dx, beta_deg, velocity, c_end, frames)
    return {
        "dx_m": dx,
        "beta_deg": beta_deg,
        "velocity_ms": velocity,
        "delta_sph": delta_sph,
        "mls_density": mls_density,
        "threads": threads,
        "n_fluid": int(sim.n_fluid),
        "n_wall": int(sim.wall_local.shape[0]),
        "dt_s": _dt(sim),
        "probe_steps": n_steps,
        "probe_frames": max(2, frames),
        "probe_c_end_m": c_end_probe,
        "wall_s": wall,
        "per_step_s": per_step,
        "per_step_s_first": (wall / max(len(series), 1)),
        "n_series": len(series),
        "campaign_case": {
            "c_end_m": c_end,
            "steps": full_steps,
            "dt_s": dt_full,
            "projected_wall_s": per_step * full_steps,
            "extrapolation": "mean probe step cost x campaign step count; "
                             "m14940 measured the per-step cost rising 20-28x "
                             "before impact, so treat this as a lower bound",
        },
    }


def _dt_of(dx: float, beta_deg: float, velocity: float) -> float:
    cfg = SlammingConfig(beta_deg=beta_deg, velocity=velocity, dx=dx, frames=8,
                         c_end=10 * dx)
    cfg.validate()
    return _dt(SlammingImpact(cfg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--velocity", type=float, default=2.0)
    ap.add_argument("--c-end", type=float, default=0.050,
                    help="campaign observation window used for the projection")
    ap.add_argument("--delta-sph", type=float, default=0.1)
    ap.add_argument("--no-mls", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=float, default=120.0,
                    help="wall clock one local slot may spend on one case")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rec = probe(args.dx, args.steps, args.beta, args.velocity, args.c_end,
                args.frames, args.delta_sph, not args.no_mls, args.threads)
    rec["budget_s"] = args.budget
    rec["affordable"] = bool(rec["campaign_case"]["projected_wall_s"] <= args.budget)
    rec["verdict"] = ("affordable: one campaign case projects inside the budget"
                      if rec["affordable"] else
                      "NOT affordable: one campaign case projects beyond the "
                      "budget, so the campaign needs a compute slot longer than "
                      "one local shell window (or a coarser dx)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("dx_m", "n_fluid", "probe_steps",
                                          "per_step_s", "wall_s")}, indent=1))
    print(rec["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())