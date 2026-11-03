"""Probe 2: equilibrium-based measurement of the Young-Laplace jump.

Hypothesis: the -75.6 % / "pass at dx=4 mm" contradiction is a measurement
artifact: (a) the pressure is sampled on the droplet breathing transient, not at
static equilibrium; (b) p_band is taken at 0.8-0.95 R, i.e. *inside* the
interface shell where p has already decayed toward the vacuum value, which
biases the jump low; (c) at coarse dx the "core" band overlaps the truncated
kernel support, where the summation density carries a *deficit*.
"""
import json
import math
import sys

import torch

sys.path.insert(0, "src")
from tensorsph.droplet_impact import (Config, density_from_neighbor_pairs, rhs,
                                      stable_dt)
from tensorsph.neighbor_search import cell_linked_pairs

SIGMA, RHO0, C0 = 7.2, 1000.0, 20.0
R = 0.02


def hexagonal_droplet(radius, dx, rho0):
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
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(ms), dtype=torch.float64)


def run(dx, steps, viscosity=0.05, delta=0.0, every=100, radius=R):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=7.0,
                 gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 density_diffusion=delta)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_sig = math.sqrt(RHO0 * radius ** 3 / SIGMA)
    v_ref = math.sqrt(SIGMA / (RHO0 * radius))
    pos, mass = hexagonal_droplet(radius, dx, RHO0)
    vel = torch.zeros_like(pos)
    hist = []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            complete = r < radius - 3.0 * h          # full kernel support
            hist.append({
                "t/t_sigma": (step + 1) * dt / t_sig,
                "p_complete": float(pressure[complete].mean()) if complete.any() else float("nan"),
                "p_core50": float(pressure[r < 0.5 * radius].mean()),
                "p_band": float(pressure[(r > 0.8 * radius) & (r < 0.95 * radius)].mean()) if ((r > 0.8 * radius) & (r < 0.95 * radius)).any() else float("nan"),
                "v/vref": float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref,
                "rho_excess": float((rho[complete] / RHO0 - 1.0).mean()) if complete.any() else float("nan"),
            })
    tail = hist[len(hist) // 2:]
    eq = {k: sum(x[k] for x in tail) / len(tail) for k in
          ("p_complete", "p_core50", "p_band", "v/vref", "rho_excess")}
    n_complete = int((torch.linalg.vector_norm(pos, dim=1) < radius - 3.0 * h).sum())
    dp_th = SIGMA / radius
    return {"dx": dx, "n": int(pos.shape[0]), "n_complete": n_complete,
            "steps": steps, "dt": dt, "viscosity": viscosity, "delta": delta,
            "t_sigma_s": t_sig, "dp_theory": dp_th, "equilibrium": eq,
            "err_complete_pct": 100 * (eq["p_complete"] - dp_th) / dp_th,
            "err_band_pct": 100 * (eq["p_core50"] - eq["p_band"] - dp_th) / dp_th,
            "hist": hist}


if __name__ == "__main__":
    import time
    out = []
    for dx, steps in ((0.002, 12000),):
        t0 = time.time()
        rep = run(dx, steps=steps, viscosity=0.05, every=200)
        out.append(rep)
        print(f"dx={dx} N={rep['n']} ncomp={rep['n_complete']} steps={rep['steps']} "
              f"p_complete={rep['equilibrium']['p_complete']:.1f} "
              f"p_core50={rep['equilibrium']['p_core50']:.1f} "
              f"p_band={rep['equilibrium']['p_band']:.1f} "
              f"err_complete={rep['err_complete_pct']:.1f}% "
              f"err_band={rep['err_band_pct']:.1f}% "
              f"v/vref={rep['equilibrium']['v/vref']:.4f}")
        print("   trace:", [round(x["p_complete"], 0) for x in rep["hist"]])
        print("   wall:", round(time.time() - t0, 1), "s")
    json.dump(out, open(".scratch/probe2.json", "w"), indent=1)