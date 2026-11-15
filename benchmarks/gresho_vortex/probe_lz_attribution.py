"""Diagnostic: where does the Gresho anchor's angular-momentum drift come from?

The anchor (CASE-SPH-GRESHO-VORTEX) gates ``angular_momentum_drift_rel`` at
round-off, on the argument that a pairwise central force conserves ``L_z``
exactly.  The committed ladder reports 9.7 / 4.8 / 4.1 % drift and calls the
scheme unable to hold the rotational equilibrium.  Before that attribution is
believed, three cheaper explanations have to be excluded:

1. **measurement**: the ``L_z`` post-processing (unwrap bookkeeping, lever arm)
   loses momentum on its own.  Control: integrate the *same* state with the
   force switched off -- the classical solution of ``a = 0`` is a straight line,
   so ``L_z`` must be conserved to round-off.  Any drift here is the harness.
2. **ledger closure**: if the drift is caused by force asymmetry, then the
   running integral of the discrete torque ``sum_n dt * tau^n / |L_z^0|`` must
   reproduce the measured drift.  If the two numbers disagree the drift is not
   torque-driven.
3. **time step / instability coupling**: the WCSPH state develops particle
   clumping and the acceleration maximum grows far past the centripetal scale
   ``U^2/r_o``.  Control: refine ``dt`` at fixed ``dx``.  A drift that collapses
   under ``dt`` refinement is a stability artefact of the configuration, not a
   static inability of the scheme.

Each control is a one-line change to the same run loop, so the comparison is
apples-to-apples.  Output is plain JSON so the numbers can be pasted into a
report without transcription.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from tensorsph.gresho import (GreshoConfig, analytic_reference, analytic_velocity,
                             angular_momentum, make_state, radius, velocity_error_pct)
from tensorsph.vv.tgv import periodic_pairs, tgv_rhs


def run(dx: float = 0.04, courant: float = 0.3, t_end: float = 0.6,
        force: bool = True, label: str = "") -> dict:
    cfg = GreshoConfig(dx=dx, max_courant=courant, t_end=t_end)
    table = analytic_reference(cfg)
    state = make_state(cfg, table=table)
    pos, vel, mass = state["pos"], state["vel"], state["mass"]
    pos_abs = pos.clone()
    dt = cfg.dt()
    steps = int(math.ceil(t_end / dt))
    dt = t_end / steps
    h = cfg.h()
    cutoff = 2.0 * h + cfg.skin_factor * h
    pairs = periodic_pairs(pos, cutoff, cfg.L)
    centre = torch.tensor(cfg.centre, dtype=pos.dtype)
    lz0 = angular_momentum(pos_abs, vel, mass, cfg)
    budget = 0.0
    acc_max = 0.0
    drift_max = 0.0
    series = []
    for step in range(steps + 1):
        if step % cfg.rebuild_every == 0:
            pairs = periodic_pairs(pos, cutoff, cfg.L)
        _rho, _p, acc = tgv_rhs(pos, vel, mass, cfg, pairs=pairs)
        if not force:
            acc = torch.zeros_like(acc)
        d = pos_abs - centre
        tau = float((mass * (d[:, 0] * acc[:, 1] - d[:, 1] * acc[:, 0])).sum())
        budget += float(dt * tau / abs(lz0) * 100.0)
        acc_max = max(acc_max, float(acc.abs().max()))
        if step % 100 == 0 or step == steps:
            lz = angular_momentum(pos_abs, vel, mass, cfg)
            drift = abs(lz - lz0) / abs(lz0) * 100.0
            drift_max = max(drift_max, drift)
            series.append({"step": step, "t": step * dt, "drift_pct": drift,
                           "acc_max": float(acc.abs().max())})
        if step == steps:
            break
        vel = vel + dt * acc
        pos_abs = pos_abs + dt * vel
        pos = torch.remainder(pos + dt * vel, cfg.L)
    verr = velocity_error_pct(vel, mass, analytic_velocity(pos, cfg, table=table))
    out = {
        "label": label, "dx": dx, "courant": courant, "steps": steps,
        "dt": dt, "force": force, "t_end": t_end,
        "a_reference": cfg.a_reference(),
        "drift_pct_max": drift_max,
        "torque_budget_pct": budget,
        "acc_max": acc_max,
        "acc_max_over_reference": acc_max / cfg.a_reference(),
        "velocity_error_pct_end": verr,
        "r_max": float(radius(pos, cfg).max()),
        "series": series,
    }
    print(f"{label:32s} dx={dx:<6} Co={courant:<5} steps={steps:5d} "
          f"drift%={drift_max:8.4f} budget%={budget:8.4f} "
          f"accmax/a_ref={acc_max / cfg.a_reference():6.2f} verr_end%={verr:6.2f}",
          flush=True)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    dxs = [float(x) for x in (argv[0].split(",") if argv else ["0.04"])]
    cases = []
    for dx in dxs:
        cases.append(run(dx=dx, force=False, label="ballistic (zero force)"))
        cases.append(run(dx=dx, courant=0.30, label="baseline Co=0.30"))
        cases.append(run(dx=dx, courant=0.15, label="dt/2 Co=0.15"))
        cases.append(run(dx=dx, courant=0.075, label="dt/4 Co=0.075"))
    dest = Path(__file__).resolve().parent / "results" / "lz_attribution.json"
    dest.write_text(json.dumps({"cases": cases}, indent=1))
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())