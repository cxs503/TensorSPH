"""Does the upper clamp of the colour function fabricate interface particles?

Variants scored on the same static droplet (instant only):
  V1 shipped : c = clamp(kernel_sum, 0, 1)
  V2 low-only: c = clamp(kernel_sum, 0, inf)   (no artificial plateau)
  V3 no-clamp: c = kernel_sum
plus the effect on the delta_s band composition and the delta_s-weighted kappa.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from tensorsph.csf_consistent import (
    mls_divergence, mls_gradient, shepard_smooth)
from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("law", ROOT / "benchmarks/laplace_law/run.py")
law = importlib.util.module_from_spec(spec)
spec.loader.exec_module(law)
R, RHO0 = 0.02, 1000.0


def fields(pos, mass, pairs, h, clamp_hi):
    vol = mass / RHO0
    i, j = pairs.unbind(dim=1)
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)
    c = torch.zeros(pos.shape[0], dtype=pos.dtype)
    c = c.index_add(0, i, vol[j] * w).index_add(0, j, vol[i] * w)
    c = c.clamp_min(0.0)
    if clamp_hi:
        c = c.clamp_max(1.0)
    gc = mls_gradient(pos, c, vol, pairs, h)
    mag = torch.linalg.vector_norm(gc, dim=1, keepdim=True)
    tiny = 1e-12
    nrm = torch.where(mag > tiny, -gc / mag.clamp_min(tiny), torch.zeros_like(gc))
    sm = shepard_smooth(nrm, pos, vol, pairs, h, passes=2)
    sm = sm / torch.linalg.vector_norm(sm, dim=1, keepdim=True).clamp_min(tiny)
    nrm = torch.where(mag > tiny, sm, nrm)
    kap = mls_divergence(pos, nrm, vol, pairs, h)
    ds = 2.0 * mag[:, 0]
    return c, kap, ds


for dx in (0.003, 0.002, 0.0015, 0.001):
    h = 1.3 * dx
    pos, mass = law.hexagonal_droplet(R, dx, RHO0)
    pos, mass = pos.double(), mass.double()
    pairs = cell_linked_pairs(pos, 2.0 * h)
    print(f"\n=== dx={dx} N={pos.shape[0]} exact kappa={1/R}  h={h:.4f}")
    for hi in (True, False):
        c, kap, ds = fields(pos, mass, pairs, h, hi)
        band = ds > 0.05 * float(ds.max())
        w = ds[band]
        km = float((kap[band] * w).sum() / w.sum())
        bulk = band & (c > 0.99)
        # share of the (spurious) momentum that lands on saturated particles
        a_mag = (kap * ds).abs()
        share = float(a_mag[bulk].sum() / a_mag[band].sum())
        print(f"  clamp_hi={hi!s:5s} kappa_wtd {km:7.2f} ({100*(km/(1/R)-1):+6.1f}%)  "
              f"band={int(band.sum()):4d} bulk-in-band={int(bulk.sum()):3d} "
              f"|a|share(bulk)={share:5.1%}  ds_max={float(ds.max()):7.1f} "
              f"int_ds={float((ds*mass/RHO0/(2*3.141592653589793*torch.linalg.vector_norm(pos,dim=1).clamp_min(1e-30))).sum()):.2f}")


# synthetic check: on a perfect lattice with no free surface, is |grad c| ~ 0?
print("\n--- bulk control (radial band 0..0.5R, no free surface) ---")
for dx in (0.003, 0.002, 0.0015, 0.001):
    h = 1.3 * dx
    pos, mass = law.hexagonal_droplet(R, dx, RHO0)
    pos, mass = pos.double(), mass.double()
    pairs = cell_linked_pairs(pos, 2.0 * h)
    r = torch.linalg.vector_norm(pos, dim=1)
    inner = r < 0.5 * R
    for hi in (True, False):
        c, kap, ds = fields(pos, mass, pairs, h, hi)
        print(f"  dx={dx} clamp_hi={hi!s:5s} "
              f"max|grad c| in 0.5R = {float(ds[inner].max())/2:9.4f}  "
              f"(interface scale 1/2h = {1/(2*h):.1f})  "
              f"-> rel {100*float(ds[inner].max())/2*2*h:.2f}%")
