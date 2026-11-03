"""Hypothesis: the 25 % delta_s deficit is the *void* half of the indicator that a
single-phase particle cloud cannot sample.  Test the standard remedy -- extend
the colour field across the interface (mirror ghosts, antisymmetric about c=1/2)
-- and score int delta_s dn, F_cut / 2 sigma and kappa.
"""
import math
import importlib.util
import pathlib

import torch

from tensorsph.csf_consistent import kernel_sum_color, mls_gradient, csf_fields
from tensorsph.csf_curvature import fit_curvature
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ll = load("ll", "benchmarks/laplace_law/run.py")
cs = load("cs", "benchmarks/csf_consistent/run.py")
SIGMA, R, RHO0 = 7.2, 0.02, 1000.0


def grad_c_with_ghosts(pos, c, vol, pairs, h, ghost_layers=1):
    """MLS gradient of c with one mirrored ghost layer of the interface shell.

    For a mollified indicator the profile is antisymmetric about c = 1/2 at the
    interface, so the void-side value at the mirror point x_i + 2 s_i n_i is
    c_g = 1 - c_i, with s_i = (c_i - 1/2) / |grad c_i| the distance from the
    particle to the interface along the outward normal.
    """
    g = mls_gradient(pos, c, vol, pairs, h)
    mag = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-30)
    n = -g / mag                                   # outward
    s = (c - 0.5) / mag.squeeze(-1)                # >0 in the fluid
    shell = (c < 0.995) & (s > 0.0) & (s < 2.0 * h)
    src = torch.nonzero(shell).squeeze(-1)
    if src.numel() == 0:
        return g, 0
    keep = torch.zeros(pos.shape[0], dtype=torch.bool)
    keep[src] = True
    for _ in range(ghost_layers):
        pass
    gpos = pos[src] + 2.0 * s[src, None] * n[src]
    gvol = vol[src]
    gc = (1.0 - c[src]).clamp_min(0.0)

    # pairs between real particles (from the original list) and ghost pairs
    from tensorsph.neighbor_search import cell_linked_pairs as clp
    allpos = torch.cat([pos, gpos])
    allc = torch.cat([c, gc])
    allvol = torch.cat([vol, gvol])
    allpairs = clp(allpos, 2.0 * h)
    # keep only pairs touching at least one ghost (real-real already counted once,
    # rebuilding is the simplest way to stay consistent)
    p = allpairs
    touches = (p[:, 0] >= pos.shape[0]) | (p[:, 1] >= pos.shape[0])
    p = p[touches]
    gg = mls_gradient(allpos, allc, allvol, allpairs, h)
    return gg[: pos.shape[0]], int(src.numel())


def score(dx, variant):
    h = 1.3 * dx
    pos, mass = ll.hexagonal_droplet(R, dx, RHO0)
    vol = mass / RHO0
    near = cell_linked_pairs(pos, 2.0 * h)
    wide = cell_linked_pairs(pos, 4.0 * h)
    st = csf_fields(pos, mass, wide, h, RHO0, curvature_scheme="fit_quadratic")
    r = torch.linalg.vector_norm(pos, dim=1)

    if variant == "base":
        g, ds, kappa = st.grad_c, st.delta_s, st.kappa
    else:
        c = st.color
        g, n_ghost = grad_c_with_ghosts(pos, c, vol, near, h)
        ds = torch.linalg.vector_norm(g, dim=1)          # full profile: no 2x
        mag = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-30)
        kappa = fit_curvature(pos, -g / mag, ds, wide, h, fallback=st.kappa,
                              method="fit_quadratic")

    dsi = cs.delta_s_integral(pos, mass, RHO0, ds, R)
    mag = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-30)
    nrm = -g / mag
    acc = (-SIGMA * kappa * ds)[:, None] * nrm / RHO0
    cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
    kw = float((kappa.abs() * ds).sum() / ds.sum())
    return {
        "n": int(pos.shape[0]),
        "dsi": dsi,
        "cut": cut / (-2 * SIGMA),
        "kappa_err": (kw / (1.0 / R) - 1.0) * 100,
        "c_outer": float(st.color[r > R - dx].mean()),
    }


for dx in (0.003, 0.002, 0.0015):
    b = score(dx, "base")
    g = score(dx, "ghost")
    print(f"\ndx={dx}  N={b['n']}   R/h={R/(1.3*dx):.1f}")
    for tag, v in (("base  (2|grad c|)", b), ("ghost (|grad c| ext)", g)):
        print(f"  {tag:22s} int ds dn={v['dsi']:.3f} ({(v['dsi']-1)*100:+5.1f}%)  "
              f"Fcut/2s={v['cut']:.3f}  kappa err={v['kappa_err']:+5.1f}%  "
              f"c_outer={v['c_outer']:.3f}")