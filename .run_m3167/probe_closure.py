"""Probe: does the free-surface closure close the Laplace deficit? (m3167, temp)"""
import importlib.util, sys, json
from pathlib import Path
import torch
torch.set_num_threads(2)

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor" / "run.py"
spec = importlib.util.spec_from_file_location("lpf", BENCH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# monkeypatch Config construction inside relax via a wrapper
import tensorsph.droplet_impact as di
orig = di.Config

def patched_relax(dx, c0, closure=True, scheme="fit_quadratic", passes=2, form="symmetric"):
    import math
    cfg = di.Config(dx=dx, h_factor=m.H_FACTOR, rho0=m.RHO0, sound_speed=c0,
                    gamma=m.GAMMA, gravity=0.0, viscosity=m.VISCOSITY,
                    surface_tension_coefficient=m.SIGMA, csf_scheme="consistent",
                    csf_curvature=scheme, free_surface_pressure=closure,
                    free_surface_passes=passes, pressure_gradient_form=form)
    h = cfg.h_factor * cfg.dx
    dt = float(di.stable_dt(cfg))
    t_end = 1.5 * m.T_SIGMA
    steps = min(4000, int(math.ceil(t_end / dt)))
    dt = t_end / steps
    pos, mass, h, pairs, bias, rho_cal = m.calibrated_droplet(dx)
    vel = torch.zeros_like(pos)
    trace = {"vmax_over_vref": []}
    v_ref = math.sqrt(m.SIGMA / (m.RHO0 * m.RADIUS))
    px, prho = None, None
    from tensorsph.droplet_impact import rhs, density_from_neighbor_pairs
    from tensorsph.neighbor_search import cell_linked_pairs
    for step in range(steps):
        rho, pressure, acc = rhs(pos, vel, mass, cfg, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if (step + 1) % 50 == 0 or step == steps - 1:
            trace["vmax_over_vref"].append(float(vel.norm(dim=1).max()) / v_ref)
        px, prho = pressure, rho
        if not torch.isfinite(pos).all():
            break
    # time average over tail
    r = torch.linalg.vector_norm(pos, dim=1) / m.RADIUS
    core = px[r < 0.5].mean()
    return dict(abs_pa=float(core), rel_pct=float((core / m.DP_ANALYTIC - 1) * 100),
                vmax_over=float(trace["vmax_over_vref"][-1]), steps=steps,
                rho_core=float(prho[r < 0.5].mean() / m.RHO0))

import sys
dx = float(sys.argv[1]); c0 = float(sys.argv[2]); closure = sys.argv[3] == "1"
passes = int(sys.argv[4]) if len(sys.argv) > 4 else 2
form = sys.argv[5] if len(sys.argv) > 5 else "symmetric"
r = patched_relax(dx, c0, closure=closure, passes=passes, form=form)
print(f"dx={dx} c0={c0} closure={closure} passes={passes} form={form}: "
      f"abs {r['rel_pct']:+7.1f} %  vmax/vref {r['vmax_over']:.3f} "
      f"rho_core-1 {r['rho_core']-1:+.2e} steps {r['steps']}", flush=True)
