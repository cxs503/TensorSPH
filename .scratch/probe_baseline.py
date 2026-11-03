"""Probe: why does the Young-Laplace pressure jump fail at dx=2mm but pass at
dx=4mm?  Time convergence, diffusion sweep, resolution sweep, force control."""
import json
import math
import sys

import torch

sys.path.insert(0, "src")
from tensorsph.droplet_impact import (Config, density_from_neighbor_pairs,
                                      rhs, stable_dt, tait_pressure)


def hexagonal_droplet(radius, dx, rho0, center=(0.0, 0.0)):
    dy = dx * math.sqrt(3.0) / 2.0
    ms = rho0 * dx * dy
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((center[0] + x, center[1] + y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(ms), dtype=torch.float64)
from tensorsph.neighbor_search import cell_linked_pairs

SIGMA, RHO0, C0 = 7.2, 1000.0, 20.0
R = 0.02


def run(dx, steps, delta, radius=R, sigma=SIGMA, every=100):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=7.0,
                 gravity=0.0, viscosity=0.01,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 density_diffusion=delta)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    pos, mass = hexagonal_droplet(radius, dx, RHO0)
    vel = torch.zeros_like(pos)
    n = pos.shape[0]
    v_ref = math.sqrt(sigma / (RHO0 * radius))
    t_sig = math.sqrt(RHO0 * radius ** 3 / sigma)
    hist = []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1) / radius
            hist.append({
                "step": step + 1, "t/t_sigma": (step + 1) * dt / t_sig,
                "p_core": float(pressure[r < 0.5].mean()),
                "p_band": float(pressure[(r > 0.8) & (r < 0.95)].mean()),
                "vmax/vref": float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref,
                "rho_core": float((rho[r < 0.5] / RHO0).mean()),
            })
    return {"dx": dx, "delta": delta, "n": n, "dt": dt, "hist": hist,
            "dp_theory": sigma / radius, "t_sig": t_sig}


def force_control(dx, radius=R, sigma=SIGMA, delta=0.0):
    """Radial integral of the CSF-only acceleration: int_0^R rho a_r dr must be
    sigma/R for a circular interface (2-D).  Independent of the pressure field
    and of any time integration."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=7.0,
                 gravity=0.0, viscosity=0.01,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 density_diffusion=delta)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    _, _, acc = rhs(pos, torch.zeros_like(pos), mass, cfg, pairs=pairs,
                    wall=None, dt=float(stable_dt(cfg)))
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r.clamp_min(1e-30).unsqueeze(1)
    a_r = (acc * rhat).sum(dim=1) * rho          # force density (N/m^3)
    # integrate along the x axis: sum over y-bins of a_r * dx, cumulative in r
    order = torch.argsort(r)
    rr, aa = r[order], a_r[order]
    return {"dx": dx, "n": int(pos.shape[0]),
            "mean_a_r": float(a_r.mean()), "max_a_r": float(a_r.max()),
            "median_a_r": float(a_r.median()),
            "r_of_max": float(r[torch.argmax(a_r)] / radius),
            "sum_rho_a_r_dx": float(aa.sum() * dx),   # crude, for scale only
            "dp_theory": sigma / radius}


if __name__ == "__main__":
    out = {}
    print("=== force control (CSF-only, no pressure coupling) ===")
    for dx in (0.004, 0.002, 0.0015):
        fc = force_control(dx)
        print(json.dumps(fc))
    for dx, steps in ((0.002, 1500), (0.004, 800)):
        for delta in (0.0, 0.1, 0.3):
            rep = run(dx, steps, delta)
            key = f"dx{dx}_d{delta}"
            out[key] = rep
            h = rep["hist"]
            tail = h[-1]
            print(f"dx={dx} delta={delta} N={rep['n']} steps={steps} "
                  f"t/t_sig={tail['t/t_sigma']:.2f} "
                  f"p_core={tail['p_core']:.1f} (theory {rep['dp_theory']:.0f}) "
                  f"p_band={tail['p_band']:.1f} "
                  f"err={100*(tail['p_core']-tail['p_band']-rep['dp_theory'])/rep['dp_theory']:.1f}% "
                  f"v/vref={tail['vmax/vref']:.3f} rho_core={tail['rho_core']:.6f}")
            print("   hist p_core:", [round(x["p_core"], 1) for x in h])
    json.dump(out, open(".scratch/probe_baseline.json", "w"), indent=1)