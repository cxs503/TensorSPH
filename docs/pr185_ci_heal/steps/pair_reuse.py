"""CI step: reusing the neighbour list must not change the forces.

Mirrors ``.github/workflows/tests.yml`` step "Reusing the neighbour list must
not change the forces".
"""
import torch

from tensorsph.droplet_impact import Config, make_particles, rhs
from tensorsph.neighbor_search import PairListCache

cfg = Config(dx=0.05, pool_width=0.4, pool_depth=0.15,
             drop_radius=0.05, drop_center_y=0.25, device="cpu")
pos, mass = make_particles(cfg, torch.device("cpu"))
vel = torch.zeros_like(pos)
cache = PairListCache(2.0 * cfg.h_factor * cfg.dx, 0.25 * cfg.h_factor * cfg.dx)
pairs, rebuilt = cache.get(pos)
assert rebuilt
cached = rhs(pos, vel, mass, cfg, pairs=pairs)
fresh = rhs(pos, vel, mass, cfg)
for a, b in zip(cached, fresh):
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
print("pair reuse is physics-neutral")
