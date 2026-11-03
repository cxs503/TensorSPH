"""Prototype: two-sided (ghost) colour gradient for delta_s normalisation.

Measures int delta_s dn on a hexagonal lattice droplet for
  V0: current  delta_s = 2 |grad c|            (one-sided MLS)
  V1: ghost reflection, outermost layer only
  V2: ghost reflection, all interface particles
  V3: support-fraction rescaling |grad c| / A_i
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import kernel_sum_color, mls_gradient, shepard_smooth
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
    """Add the void-side (mirror) contribution to grad c for masked particles."""
    vol = mass / rho0
    i, j = pairs.unbind(dim=1)
    grad = torch.zeros((pos.shape[0], 2), dtype=pos.dtype)
    d = pos[j] - pos[i]
    n = normal[i]
    dn = (d * n).sum(dim=1, keepdim=True)
    d_ref = d - 2.0 * dn * n            # mirror the neighbour across the tangent plane
    gw = cubic_spline_gradient(-d_ref, h)   # grad_i W(x_i - x_ghost)
    contrib = (vol[j] * (-color[i]))[:, None] * gw
    contrib = torch.where(mask[i][:, None], contrib, torch.zeros_like(contrib))
    return grad.index_add(0, i, contrib)


def run(dx):
    radius, sigma, rho0, hf = 0.02, 7.2, 1000.0, 1.3
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

    out = {"dx": dx, "N": int(pos.shape[0])}
    out["V0"] = delta_s_integral(pos, mass, rho0, 2.0 * mag.squeeze(1), radius)

    # support fraction A_i = sum_j V_j W_ij + self term
    from tensorsph.droplet_impact import cubic_spline_kernel
    A = vol * cubic_spline_kernel(torch.zeros(()), h)
    i, j = wide.unbind(dim=1)
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)
    A = A.index_add(0, i, vol[j] * w)
    A = A.index_add(0, j, vol[i] * w)
    out["V3_Amin"] = float(A.min())

    for tag, lo, hi in (("V1", 0.0, 0.75), ("V2", 0.0, 0.99)):
        mask = (c > lo) & (c < hi)
        extra = ghost_grad(pos, mass, wide, h, rho0, c, normal, mask)
        g2 = gc + extra
        delta = 2.0 * torch.linalg.vector_norm(g2, dim=1)
        out[tag] = delta_s_integral(pos, mass, rho0, delta, radius)
        out[tag + "_n_mask"] = int(mask.sum())

    # V4: one-sided gradient rescaled by the missing support fraction
    delta_v4 = 2.0 * mag.squeeze(1) / A.clamp_min(0.1)
    out["V4"] = delta_s_integral(pos, mass, rho0, delta_v4, radius)
    print(out)
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    for dx in (0.003, 0.002, 0.0015):
        run(dx)