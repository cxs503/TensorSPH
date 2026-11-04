"""m3298: which CSF factor is short -- curvature kappa or surface delta d_s?"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config, density_from_neighbor_pairs, rhs, stable_dt, tait_pressure)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import csf_fields

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "lpf", HERE.parent / "benchmarks" / "laplace_pressure_floor" / "run.py")
lpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpf)

DX, C0, R = 0.002, 20.0, lpf.RADIUS
cfg = Config(dx=DX, h_factor=1.3, rho0=1000.0, sound_speed=C0, gamma=7.0,
             gravity=0.0, viscosity=0.01, surface_tension_coefficient=7.2,
             csf_scheme="consistent", csf_curvature="fit_quadratic")
h = cfg.h_factor * cfg.dx
KAPPA_TRUE = 1.0 / R


def inspect(pos, rho, tag):
    # exactly what surface_tension_acceleration does: widen to the fit stencil
    from tensorsph import csf_consistent
    stt = csf_consistent.csf_fields(pos, mass_ref,
                                    cell_linked_pairs(pos, 2.0 * h), h,
                                    cfg.rho0, wall=None, contact_angle=None,
                                    curvature_scheme=cfg.csf_curvature,
                                    curvature_stencil=4.0)
    from tensorsph.surface_tension import compute_surface_tension_force
    a_st = compute_surface_tension_force(pos, rho, mass_ref,
                                         cell_linked_pairs(pos, 2.0 * h), cfg)
    r = torch.linalg.vector_norm(pos, dim=1) / R
    rhat = pos / torch.linalg.vector_norm(pos, dim=1)[:, None].clamp_min(1e-30)
    ds = stt.delta_s
    kap = stt.kappa
    iface = ds > 1e-3 * float(ds.max())
    # delta function integral across the interface, per unit area: sum V * d_s
    V = mass_ref / rho
    int_ds = float((V * ds).sum())
    # 2-D band integral: d_s has units 1/m; integral across the interface is 1.
    band = iface & (r > 0.70) & (r < 1.10)
    print(f"\n[{tag}] interface particles: {int(iface.sum())}  "
          f"(band 0.70-1.10R: {int(band.sum())})")
    print(f"  sum(V*d_s)            = {int_ds:8.3f}   (2-D interface: 1/length "
          f"scale-free; a correct delta gives int d_s dn = 1)")
    print(f"  mean kappa (iface)    = {float(kap[iface].mean()):+8.3f} 1/m "
          f"(true {KAPPA_TRUE:+.1f}; ratio {float(kap[iface].mean()) / KAPPA_TRUE:+.3f})")
    print(f"  mean d_s   (iface)    = {float(ds[iface].mean()):+10.2f} 1/m")
    # force = -sigma kappa d_s n  -> radial force density
    F = a_st * rho[:, None]
    f_r = (F * rhat).sum(dim=1)
    # exact 2-D virial: net force on the half disk x>0 must be 2*sigma (N/m)
    half = (pos[:, 0] > 0.0) & (r < 1.15)
    Fx = float((V[half] * F[half, 0]).sum())
    print(f"  virial: sum V F_x over x>0 = {Fx:+8.3f} N/m "
          f"(exact 2 sigma = {2 * 7.2:+.2f}; ratio {Fx / 14.4:+.3f})")
    # delta-function normalisation measured as int F_r dr / (sigma*kappa)
    tot, _ = x_integral(pos, F, rhat)
    print(f"  int F_r dr            = {tot:+8.2f} Pa  "
          f"ratio to sigma*kappa_bar = {tot / (7.2 * float(kap[iface].mean())):+.3f}")
    return kap, ds, iface


def x_integral(pos, F, rhat, n=60):
    r = torch.linalg.vector_norm(pos, dim=1)
    f_r = (F * rhat).sum(dim=1)
    edges = torch.linspace(0.0, R, n + 1)
    tot = 0.0
    out = []
    for k in range(n):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if bool(m.any()):
            mean = float(f_r[m].mean())
            tot += mean * float(edges[k + 1] - edges[k])
            out.append((float(0.5 * (edges[k] + edges[k + 1]) / R), mean))
    return tot, out


pos, mass_ref, h, _, bias, _ = lpf.calibrated_droplet(DX)
rho = density_from_neighbor_pairs(pos, mass_ref, h, cell_linked_pairs(pos, 2.0 * h))

# patch mass_ref into the helpers above via closure on module global
inspect(pos, rho, "t=0 lattice")

dt = float(stable_dt(cfg))
t_end = 1.5 * lpf.T_SIGMA
steps = min(4000, int(math.ceil(t_end / dt)))
dt = t_end / steps
vel = torch.zeros_like(pos)
for s in range(steps):
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho, pressure, acc = rhs(pos, vel, mass_ref, cfg, pairs=pairs, dt=dt)
    vel = vel + dt * acc
    pos = pos + dt * vel
inspect(pos, rho, "relaxed 1.5 t_sigma")