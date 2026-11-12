"""Numerical damping of the interface mode: the A = 0 control.

Same geometry, initial kick and integrator as the Rayleigh-Taylor case, but
``rho_heavy = rho_light`` so the Rayleigh-Taylor drive vanishes.  Whatever
decay is left is the scheme's own dissipation of the interface mode, and it is
what the measured growth-rate deficit is compared against in
``docs/benchmarks/rayleigh-taylor.md``.

Run::

    PYTHONPATH=src python benchmarks/rayleigh_taylor/damping.py --grid 0.05 0.025
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensorsph import rayleigh_taylor as rt

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, nargs="+", default=[0.05, 0.025])
    ap.add_argument("--end-time", type=float, default=0.12)
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = []
    for dx in args.grid:
        cfg = rt.RTConfig(dx=dx, rho_heavy=500.0, rho_light=500.0)
        state = rt.build_column(cfg)
        pos = state["pos"].clone()
        wall = state["wall_pos"]
        vol, is_heavy = state["volume0"], state["is_heavy"]
        pairs = rt.periodic_pairs(torch.cat((pos, wall)), pos.shape[0],
                                  2.0 * cfg.h, cfg.wavelength)
        ref = rt.reference_state(pos, wall, state["wall_rho0"], cfg, pairs,
                                 y_ref=state["y_ref"])
        vel = torch.zeros_like(pos)
        dt = rt.stable_dt(cfg, 0.0)
        t, times, amps = 0.0, [], []
        while t < args.end_time:
            _, _, acc, _ = rt.two_phase_rhs(pos, vel, state["mass"], state["rho0"],
                                            cfg, wall, state["wall_rho0"], pairs, ref)
            vel = vel + dt * acc
            pos = pos + dt * vel
            t += dt
            if not times or t - times[-1] >= 0.008:
                amps.append(rt.modal_amplitude(pos, vol, is_heavy, cfg)[0])
                times.append(t)
            dt = rt.stable_dt(cfg, float(vel.norm(dim=1).max()))
        fit = rt.fit_window(times, amps, 1e-9, float("inf"))
        out.append({"dx": dx, "sigma": fit["sigma"], "r2": fit["r2"],
                    "a0": amps[0], "a_end": amps[-1], "t_end": t,
                    "particles": int(pos.shape[0])})
        print(out[-1])
    (HERE / "runs" / "damping.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
