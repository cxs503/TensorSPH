import torch, sys, time
sys.path.insert(0, 'src')
from tensorsph import kleefsman_obstacle_3d as k3
cfg = k3.Kleefsman3DConfig(dx=0.05, t_end=0.62, sound_speed=20.0, wall_layers=3,
                           output_every=5)
cfg.dt = 0.9 * cfg.dt_from_courant()
rows = []
t0 = time.time()
def on_sample(i, r):
    rows.append(dict(r))
    print(f"S {r['t_s']:.6f} slab={r['slab_height_m']:.6f} h={r['height_operator_m']:.6f} "
          f"front={r['front_x_m']:.4f}", flush=True)
run = k3.run_case(cfg, device="cpu", on_sample=on_sample)
print("wall", round(time.time()-t0, 1), "steps", run.steps, flush=True)
nz = [r for r in rows if r["slab_height_m"] > 0]
print("n_samples", len(rows), "n_nonzero_slab", len(nz))
print("max_height", max(run.height), "max_slab", max(run.slab))
for r in nz[:6]:
    print("NONZERO", r)
