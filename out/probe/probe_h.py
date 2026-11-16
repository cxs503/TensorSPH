import torch, sys, time
sys.path.insert(0, 'src')
from tensorsph import kleefsman_obstacle_3d as k3

T_END = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
cfg = k3.Kleefsman3DConfig(dx=0.05, t_end=T_END, sound_speed=20.0, wall_layers=3,
                           output_every=200)
cfg.dt = 0.9 * cfg.dt_from_courant()
print("dt", cfg.dt)
rows = []
t0 = time.time()
run = k3.run_case(cfg, device="cpu", on_sample=lambda i, r: rows.append(dict(r)))
print("wall", round(time.time() - t0, 1), "steps", run.steps)
nonzero = [r for r in rows if r["slab_height_m"] > 0]
print("n samples", len(rows), "nonzero slab", len(nonzero))
for r in (nonzero[:4] + nonzero[-2:]) if nonzero else rows[-3:]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("maxheight", max(run.height), "maxslab", max(run.slab))
