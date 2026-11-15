import math, sys, json, importlib.util, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
torch.set_num_threads(4)
HERE = ROOT / 'benchmarks' / 'csf_consistent'
spec = importlib.util.spec_from_file_location("cc_run", HERE / 'run.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from tensorsph.droplet_impact import (Config, density_from_neighbor_pairs, rhs,
                                      stable_dt, tait_pressure)
from tensorsph.neighbor_search import cell_linked_pairs

sigma, R, rho0, c0 = 7.2, 0.02, 1000.0, 20.0
dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=c0, gamma=7.0, gravity=0.0,
             viscosity=0.01, surface_tension_coefficient=sigma, csf_scheme="consistent")
h = cfg.h_factor * cfg.dx
dt = float(stable_dt(cfg)); t_sig = math.sqrt(rho0 * R**3 / sigma)
pos, mass = m.hexagonal_droplet(R, dx, rho0)
vel = torch.zeros_like(pos)
rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0 * h))
pressure = tait_pressure(rho, cfg)
vref = math.sqrt(sigma / (rho0 * R))
print(f"dx={dx} N={pos.shape[0]} dt={dt:.3e} t_sig={t_sig:.4e} end={steps*dt/t_sig:.2f} t_sig")
t0 = time.time()
hist = []
for step in range(steps):
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
    vel = vel + dt * acc
    pos = pos + dt * vel
    if step % 100 == 0:
        r = torch.linalg.vector_norm(pos, dim=1)
        core = r < 0.7 * R; shell = (r > 0.80 * R) & (r < 0.95 * R)
        hist.append(dict(t=step * dt, t_sig=(step * dt) / t_sig,
                         p_core=float(pressure[core].mean()),
                         p_shell=float(pressure[shell].mean()),
                         p_core_std=float(pressure[core].std()),
                         vmax=float(torch.linalg.vector_norm(vel, dim=1).max()) / vref,
                         rho_core=float(rho[core].mean())))
last = hist[-1]
print("final:", json.dumps({k: round(v, 5) for k, v in last.items()}))
for s in hist[::10]:
    print(f"t={s['t_sig']:5.2f}ts  p_core={s['p_core']:8.2f} +-{s['p_core_std']:6.2f} "
          f"p_shell={s['p_shell']:8.2f} vmax/vref={s['vmax']:.4f} rho_core={s['rho_core']:.5f}")
r = torch.linalg.vector_norm(pos, dim=1)
edges = torch.linspace(0, 1.3, 27)
b = torch.bucketize(r / R, edges)
print("radial profile (r/R, p, rho, n):")
for i in range(1, len(edges)):
    msk = b == i
    if msk.sum() > 0:
        print(f"  {0.5*(edges[i-1]+edges[i]):.2f} {float(pressure[msk].mean()):8.2f} "
              f"{float(rho[msk].mean()):8.3f} {int(msk.sum()):4d}")
print("wall", time.time() - t0)