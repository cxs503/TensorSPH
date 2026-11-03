"""Prototype 2: threshold scan for the ghost mask + cut-force identity."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import kernel_sum_color, mls_gradient, shepard_smooth
from tensorsph.csf_curvature import fit_curvature
from tensorsph.droplet_impact import cubic_spline_gradient
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location(
    "law_run", ROOT / "benchmarks/laplace_law/run.py")
law = importlib.util.module_from_spec(spec)
spec.loader.exec_module(law)


def delta_s_integral(pos, mass, rho0, delta_s, radius, lo=0.6, hi=1.05):
    r = torch.linalg.vector_norm(pos, dim=1)
    m = (r > lo * radius) & (r < hi * radius)
    d_v = mass / rho0
    return float((delta_s[m] * d_v[m] / (2.0 * math.pi * r[m].clamp_min(1e-30))).sum())


def ghost_grad(pos, mass, pairs, h, rho0, color, normal, mask):
    vol = mass / rho0
    i, j = pairs.unbind(dim=1)
    d = pos[j] - pos[i]
    n = normal[i]
    d_ref = d - 2.0 * (d * n).sum(dim=1, keepdim=True) * n
    gw = cubic_spline_gradient(-d_ref, h)
    contrib = (vol[j] * (-color[i]))[:, None] * gw
    contrib = torch.where(mask[i][:, None], contrib, torch.zeros_like(contrib))
    out = torch.zeros((pos.shape[0], 2), dtype=pos.dtype)
    return out.index_add(0, i, contrib)


def run(dx, sigma=7.2, radius=0.02, rho0=1000.0, hf=1.3, curvature="fit_quadratic"):
    h = hf * dx
    pos, mass = law.hexagonal_droplet(radius, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0 * h)
    vol = mass / rho0
    c = kernel_sum_color(pos, mass, wide, h, rho0)
    gc = mls_gradient(pos, c, vol, wide, h)
    mag = torch.linalg.vector_norm(gc, dim=1, keepdim=True)
    normal = -gc / mag.clamp_min(1e-12)
    sm = shepard_smooth(normal, pos, vol, wide, h, passes=2)
    normal = sm / torch.linalg.vector_norm(sm, dim=1, keepdim=True).clamp_min(1e-12)
    kappa_mls = torch.zeros(pos.shape[0])
    from tensorsph.csf_consistent import mls_divergence
    kappa_mls = mls_divergence(pos, normal, vol, wide, h)
    # density not needed for the cut-force identity (mass-weighted force sum)
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]

    rows = {}
    for thr in (0.60, 0.70, 0.75, 0.80, 0.85, 1.01):
        mask = c < thr
        extra = ghost_grad(pos, mass, wide, h, rho0, c, normal, mask)
        g2 = gc + extra
        d2 = 2.0 * torch.linalg.vector_norm(g2, dim=1)
        kappa = fit_curvature(pos, normal, d2, wide, h, fallback=kappa_mls,
                              method=curvature, stencil=4.0)
        # force per volume (rho set to rho0: the identity is mass weighted)
        f = -sigma * kappa * d2 / rho0
        acc = f[:, None] * normal
        cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
        act = d2 > 0.05 * d2.max()
        cos = float(((acc * (-rhat)).sum(dim=1)[act] / torch.linalg.vector_norm(acc, dim=1)[act]
                     .clamp_min(1e-30)).median())
        rows[thr] = dict(mask=int(mask.sum()),
                         int_ds=delta_s_integral(pos, mass, rho0, d2, radius),
                         fcut=cut / (-2.0 * sigma), cos=cos)
    # baseline
    d0 = 2.0 * mag.squeeze(1)
    kappa0 = fit_curvature(pos, normal, d0, wide, h, fallback=kappa_mls,
                           method=curvature, stencil=4.0)
    f0 = -sigma * kappa0 * d0 / rho0
    acc0 = f0[:, None] * normal
    cut0 = float((mass[pos[:, 0] > 0, None] * acc0[pos[:, 0] > 0]).sum(dim=0)[0])
    print(f"dx={dx} N={int(pos.shape[0])} c_surface_max={float(c.max()):.2f} "
          f"baseline int_ds={delta_s_integral(pos,mass,rho0,d0,radius):.3f} "
          f"fcut={cut0/(-2*sigma):.3f}")
    for thr, v in rows.items():
        print(f"   thr={thr:4.2f} n={v['mask']:4d} int_ds={v['int_ds']:.4f} "
              f"({(v['int_ds']-1)*100:+6.2f}%)  fcut/2sigma={v['fcut']:+.4f} "
              f"({abs(v['fcut']-1)*100:5.2f}%)  cos={v['cos']:+.3f}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    for dx in (0.003, 0.002, 0.0015):
        run(dx)