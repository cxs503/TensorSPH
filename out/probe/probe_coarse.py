"""Diagnostic (coarse dx): does run.height stay 0 on the production geometry?"""
import torch, sys, time
sys.path.insert(0, 'src')
from tensorsph import kleefsman_obstacle_3d as k3

k3.validate_case = lambda cfg: None        # diagnostic: dx=0.1 violates resolution gate
cfg = k3.Kleefsman3DConfig(dx=0.1, t_end=0.70, sound_speed=20.0, wall_layers=3,
                           output_every=10)
cfg.dt = 0.9 * cfg.dt_from_courant()
print("dt", cfg.dt, "station_x", cfg.station_x(), "band", cfg.gauge_aperture)
band = (cfg.station_x() - 0.5*cfg.gauge_aperture, cfg.station_x() + 0.5*cfg.gauge_aperture)
state = {}
def on_sample(i, r):
    p = state["pos"]
    m = k3._station_mask(p, cfg)
    n = int(m.sum().item())
    print(f"S t={r['t_s']:.4f} slab={r['slab_height_m']:.5f} h={r['height_operator_m']:.5f} "
          f"front={r['front_x_m']:.3f} n_band={n} "
          f"ymax_band={float(p[m,1].max()) if n else float('nan'):.4f} "
          f"ymax_all={float(p[:,1].max()):.4f} "
          f"col1_max={float(p[:,1].max()):.3f} col2_max={float(p[:,2].max()):.3f} "
          f"col2_min={float(p[:,2].min()):.3f}", flush=True)

orig = k3.rhs_3d
def traced(pos, *a, **kw):
    state["pos"] = pos.detach().clone()
    return orig(pos, *a, **kw)
k3.rhs_3d = traced
t0 = time.time()
run = k3.run_case(cfg, device="cpu", on_sample=on_sample)
print("wall", round(time.time()-t0,1), "steps", run.steps, "max_height", max(run.height),
      "max_slab", max(run.slab))
