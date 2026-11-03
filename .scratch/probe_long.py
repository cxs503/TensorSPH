"""Decisive probe: is there a static equilibrium at all?

Run one coarse droplet for 16 capillary times and watch p_core, the drop radius
and the density excess.  If p never plateaus, then no single sample can be
called "the equilibrium value" and the honest verdict stays WITHHELD with a
quantified drift rate.
"""
import json
import math
import sys

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks/laplace_equilibrium")
from run import complete_support_mask, hexagonal_droplet  # noqa: E402
from tensorsph.droplet_impact import Config, rhs, stable_dt  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

SIGMA, RHO0, C0, RADIUS = 7.2, 1000.0, 20.0, 0.02


def go(dx=0.003, cusp=16.0, nu=0.02, every=100):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=7.0,
                 gravity=0.0, viscosity=nu, surface_tension_coefficient=SIGMA,
                 csf_scheme="consistent")
    h = cfg.h_factor * dx
    dt = float(stable_dt(cfg))
    t_sig = math.sqrt(RHO0 * RADIUS ** 3 / SIGMA)
    v_ref = math.sqrt(SIGMA / (RHO0 * RADIUS))
    steps = int(round(cusp * t_sig / dt))
    pos, mass = hexagonal_droplet(RADIUS, dx, RHO0)
    vel = torch.zeros_like(pos)
    m_tot = float(mass.sum())
    out = []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            full = complete_support_mask(pos, RADIUS, h)
            # area-equivalent radius from the particle count: A = N*dx*dy
            r_eff = math.sqrt(float(pos.shape[0]) * dx * dx * math.sqrt(3) / 2
                              / math.pi)
            r_rms = float((r ** 2).mean().sqrt())
            out.append({
                "t/t_sigma": (step + 1) * dt / t_sig,
                "p/theory": float(pressure[full].mean()) / (SIGMA / RADIUS),
                "p_core50/theory": float(pressure[r < 0.5 * RADIUS].mean())
                / (SIGMA / RADIUS),
                "v/vref": float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref,
                "r_rms": r_rms, "r_eff": r_eff,
                "rho_excess": float((rho[full] / RHO0 - 1.0).mean()),
                "mass_total": float(mass.sum()) / m_tot,
            })
    json.dump({"dx": dx, "nu": nu, "steps": steps, "dt": dt,
               "trace": out}, open(".scratch/probe_long.json", "w"), indent=1)
    for row in out:
        print(f"t/t={row['t/t_sigma']:6.2f} p/th={row['p/theory']:6.3f} "
              f"p50/th={row['p_core50/theory']:6.3f} v/vref={row['v/vref']:.4f} "
              f"r_rms={row['r_rms']*1e3:7.3f}mm rho_exc={row['rho_excess']:+.5f}")


if __name__ == "__main__":
    go()