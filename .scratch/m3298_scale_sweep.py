"""m3298: does the free-surface delta closure move the Laplace jump into 15%?"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (Config, density_from_neighbor_pairs, rhs,
                                     stable_dt, tait_pressure)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "lpf", HERE.parent / "benchmarks" / "laplace_pressure_floor" / "run.py")
lpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpf)

DP, R, T_SIGMA = lpf.DP_ANALYTIC, lpf.RADIUS, lpf.T_SIGMA


def trial(dx, c0, scale, t_end_over_t_sigma=1.5, max_steps=4000):
    cfg = Config(dx=dx, h_factor=1.3, rho0=1000.0, sound_speed=c0, gamma=7.0,
                 gravity=0.0, viscosity=0.01, surface_tension_coefficient=7.2,
                 csf_scheme="consistent", csf_curvature="fit_quadratic",
                 csf_delta_scale=scale)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_end = t_end_over_t_sigma * T_SIGMA
    steps = min(max_steps, int(math.ceil(t_end / dt)))
    dt = t_end / steps
    pos, mass, h, _, bias, _ = lpf.calibrated_droplet(dx)
    vel = torch.zeros_like(pos)
    v_ref = math.sqrt(7.2 / (1000.0 * R))
    acc_hist = []
    for s in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if s % 50 == 0 or s == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1) / R
            acc_hist.append(float(pressure[r < 0.5].mean()))
    r = torch.linalg.vector_norm(pos, dim=1) / R
    n_tail = max(1, int(0.4 * len(acc_hist)))
    tail = sum(acc_hist[-n_tail:]) / n_tail
    vmax = float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref
    finite = bool(torch.isfinite(pos).all())
    return dict(dx=dx, c0=c0, scale=scale, p_core_final=float(pressure[r < 0.5].mean()),
                tail=tail, err=(tail - DP) / DP * 100.0,
                rho_core=float(rho[r < 0.5].mean() / 1000.0),
                r_mean_over_R=float(r.mean()), vmax_over_vref=vmax,
                finite=finite, steps=steps)


for dx in (0.002, 0.0015):
    for c0 in (20.0, 5.0):
        for scale in (1.0, 1.6, 1.8, 2.0):
            r = trial(dx, c0, scale)
            print(f"dx={r['dx']:g} c0={r['c0']:5g} scale={r['scale']:.1f} "
                  f"tail={r['tail']:7.2f} Pa err={r['err']:+7.1f}% "
                  f"rho_core={r['rho_core']:.5f} r/R={r['r_mean_over_R']:.3f} "
                  f"vmax/vref={r['vmax_over_vref']:.3f} finite={r['finite']}",
                  flush=True)