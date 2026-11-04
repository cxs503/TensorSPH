"""m3298: where does the missing Young-Laplace jump live?

Static momentum budget on the laplace_pressure_floor droplet:

  equilibrium requires   p(0) = int_0^R rho * a_r dr
  (radial balance dp/dr = rho a_r with p(R) = 0).

Split the RHS into the two forces that matter here (gravity = 0, viscosity
negligible at rest):

  * surface tension  ->  int (rho a_st)_r dr   == sigma/R for a correct CSF
  * pressure         ->  int (rho a_p )_r dr   == measured p_core

Measured at t=0 (lattice) and after the benchmark's own 1.5 t_sigma relax.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config, density_from_neighbor_pairs, rhs, stable_dt, tait_pressure)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmarks" / "laplace_pressure_floor"
spec = importlib.util.spec_from_file_location("lpf", BENCH / "run.py")
lpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpf)

DX, C0 = 0.002, 20.0
cfg = Config(dx=DX, h_factor=1.3, rho0=1000.0, sound_speed=C0, gamma=7.0,
             gravity=0.0, viscosity=0.01, surface_tension_coefficient=7.2,
             csf_scheme="consistent", csf_curvature="fit_quadratic")
h = cfg.h_factor * cfg.dx
DP = lpf.DP_ANALYTIC
R = lpf.RADIUS


def shell_integral(pos, force_density, n_bands=40):
    """int_0^R F_r dr by radial binning (band thickness dx)."""
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None].clamp_min(1e-30)
    f_r = (force_density * rhat).sum(dim=1)          # N/m^3, inward < 0
    edges = torch.linspace(0.0, R, n_bands + 1)
    total = 0.0
    per_band = []
    for k in range(n_bands):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if not bool(m.any()):
            continue
        mean = float(f_r[m].mean())
        total += mean * (float(edges[k + 1] - edges[k]))
        per_band.append((float(0.5 * (edges[k] + edges[k + 1]) / R), mean))
    return total, per_band


def budget(pos, mass, rho, pressure, tag):
    r = torch.linalg.vector_norm(pos, dim=1) / R
    pairs_st = cell_linked_pairs(pos, 2.0 * h)
    fluid_pairs = pairs_st[(pairs_st[:, 0] >= 0)]        # all fluid here
    a_st = compute_surface_tension_force(pos, rho, mass, fluid_pairs, cfg)
    F_st = a_st * rho[:, None]                            # N/m^3
    pairs = cell_linked_pairs(pos, 2.0 * h)
    _, _, acc = rhs(pos, torch.zeros_like(pos), mass, cfg, pairs=pairs, dt=1e-9)
    return None


pos, mass, h, _pairs0, bias, bias_after = lpf.calibrated_droplet(DX)
rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0 * h))
pressure = tait_pressure(rho, cfg)


def report_state(pos, vel, mass, rho, pressure, tag):
    r = torch.linalg.vector_norm(pos, dim=1) / R
    pairs2h = cell_linked_pairs(pos, 2.0 * h)
    a_st = compute_surface_tension_force(pos, rho, mass, pairs2h, cfg)
    F_st = a_st * rho[:, None]
    tot_st, bands_st = shell_integral(pos, F_st)
    # pressure force density from the same operator rhs uses (viscosity/grav off)
    cfg0 = Config(**{**cfg.__dict__, "viscosity": 0.0,
                     "surface_tension_coefficient": 0.0})
    _, p_chk, acc_p = rhs(pos, torch.zeros_like(vel), mass, cfg0,
                          pairs=pairs2h, dt=1e-9)
    F_p = acc_p * rho[:, None]
    tot_p, bands_p = shell_integral(pos, F_p)
    print(f"\n[{tag}] p_core = {float(pressure[r < 0.5].mean()):8.2f} Pa "
          f"({float(pressure[r < 0.5].mean()) / DP * 100:6.1f}% of sigma/R)")
    print(f"  int (rho a_st)_r dr = {tot_st:8.2f}  ({tot_st / DP * 100:6.1f}% of sigma/R)"
          f"   [expect sigma/R = {DP:.0f} for a correct CSF]")
    print(f"  int (rho a_p )_r dr = -{tot_p:8.2f}  (pressure gradient part)")
    print("  band-resolved F_st (r/R, N/m^3):",
          " ".join(f"{a:.2f}:{b:+.0f}" for a, b in bands_st[-12:]))
    return bands_st, bands_p


report_state(pos, torch.zeros_like(pos), mass, rho, pressure, "t=0 lattice")

# relax exactly like the benchmark
dt = float(stable_dt(cfg))
t_end = 1.5 * lpf.T_SIGMA
steps = min(4000, int(math.ceil(t_end / dt)))
dt = t_end / steps
vel = torch.zeros_like(pos)
for s in range(steps):
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, dt=dt)
    vel = vel + dt * acc
    pos = pos + dt * vel
print(f"\nrelaxed {steps} steps, dt={dt:.3e}")
report_state(pos, vel, mass, rho, pressure, "relaxed 1.5 t_sigma")

# geometric check: is the CSF force concentrated where the kernel is deficient?
r = torch.linalg.vector_norm(pos, dim=1) / R
a_st = compute_surface_tension_force(pos, rho, mass,
                                    cell_linked_pairs(pos, 2.0 * h), cfg)
F = (a_st * rho[:, None])
f_r = (F * (pos / torch.linalg.vector_norm(pos, dim=1)[:, None])).sum(dim=1)
for lo, hi in ((0.0, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.05), (1.05, 1.3)):
    m = (r >= lo) & (r < hi)
    if bool(m.any()):
        print(f"  band {lo:.2f}-{hi:.2f}: n={int(m.sum()):4d} "
              f"mean F_r={float(f_r[m].mean()):+10.1f} N/m^3 "
              f"sum*dx={float(f_r[m].sum()) * DX:8.2f} Pa")