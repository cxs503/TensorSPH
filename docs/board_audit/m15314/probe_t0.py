#!/usr/bin/env python3
"""Auditor-side t=0 probe for CASE-LAKE-AT-REST-SPH (m15278).

Independent of the delivered run.py: recomputes the initial-state pressure
deviation on the production RHS for the two modes, so the report's t=0 numbers
(-24 % bottom row, 0.012 % interior, max|a| 19.04 / 4.65 m/s^2) can be checked
against an auditor computation instead of being trusted.

Run:  cd <TensorSPH checkout with the m15278 patch> && PYTHONPATH=src:benchmarks/lake_at_rest \
      python3 <this file> --out probe_t0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from tensorsph import droplet_impact as di
from run import build_state, interior_mask  # delivered benchmark helpers (snapshot copy)


def probe(dx: float = 0.02) -> dict:
    torch.set_num_threads(1)
    out = {}
    for mode in ("shipped", "wb"):
        cfg = di.Config(dx=dx, h_factor=1.3, hydrostatic_init=True,
                        density_calibration=True)
        if mode == "wb":
            cfg.equilibrium_level = cfg.pool_depth
        pos, mass, wall = build_state(cfg, torch.device("cpu"))
        h = cfg.h_factor * cfg.dx
        interior = interior_mask(pos, cfg, h)
        _rho, p, acc = di.rhs(pos, torch.zeros_like(pos), mass, cfg,
                              wall=wall, dt=cfg.dt)
        pex = cfg.rho0 * cfg.gravity * (cfg.pool_depth - pos[:, 1])
        norm = cfg.rho0 * cfg.gravity * cfg.pool_depth
        dev = (p - pex).abs() / norm
        bottom = pos[:, 1] <= 0.5 * cfg.dx + 1e-12
        out[mode] = {
            "dx_m": cfg.dx,
            "n_particles": int(pos.shape[0]),
            "n_wall": int(wall.shape[0]),
            "n_interior": int(interior.sum()),
            "interior_dev_max_pct": 100.0 * float(dev[interior].max()),
            "interior_dev_mean_pct": 100.0 * float(dev[interior].mean()),
            "bottom_row_dev_max_pct": 100.0 * float(dev[bottom].max()),
            "acc_max_m_s2": float(acc.abs().max()),
            "h_over_dx": cfg.h_factor,
            "three_h_m": 3.0 * h,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe_t0.json")
    a = ap.parse_args()
    data = {"case_key": "CASE-LAKE-AT-REST-SPH", "probe": "t=0 only, production rhs",
            "result": probe()}
    Path(a.out).write_text(json.dumps(data, indent=1))
    print(json.dumps(data["result"], indent=1))


if __name__ == "__main__":
    sys.exit(main())