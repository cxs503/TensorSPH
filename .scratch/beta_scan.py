"""Fix candidate: keep the (accurate) kernel-sum colour field, compute grad c with
a *narrower* MLS stencil h_g = beta*h.  The transition is ~1 lattice layer thick,
so the 2h-wide MLS stencil averages the gradient spike away (measured ratio 0.76,
resolution independent); a narrower stencil should recover it.
"""
import math
import importlib.util
import pathlib

import torch

from tensorsph.csf_consistent import kernel_sum_color, mls_gradient, csf_fields, _mls_operators
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
R, RHO0, SIGMA = 0.02, 1000.0, 7.2


def exact_gradient_magnitude(d, h, n=401, span=2.05):
    y = torch.linspace(-span * h, span * h, n, dtype=torch.float64)
    dy = y[1] - y[0]
    from tensorsph.droplet_impact import cubic_spline_kernel
    dd = torch.atleast_1d(torch.as_tensor(d, dtype=torch.float64))
    r = torch.sqrt(dd[:, None] ** 2 + y[None, :] ** 2)
    return (cubic_spline_kernel(r, h) * dy).sum(dim=-1)


print(f"{'dx':>7s} {'beta':>5s} | {'int ds dn':>9s} {'err%':>7s} {'Fcut/2s':>8s} "
      f"{'kappa err%':>10s} {'fallback':>8s}")
for dx in (0.003, 0.002, 0.0015):
    h = 1.3 * dx
    pos, mass = ll.hexagonal_droplet(R, dx, RHO0)
    vol = mass / RHO0
    near = cell_linked_pairs(pos, 2.0 * h)
    wide = cell_linked_pairs(pos, 4.0 * h)
    r = torch.linalg.vector_norm(pos, dim=1)
    c = kernel_sum_color(pos, mass, near, h, RHO0)
    ref = csf_fields(pos, mass, wide, h, RHO0, curvature_scheme="fit_quadratic")
    for beta in (1.0, 0.85, 0.7, 0.6, 0.5):
        hg = beta * h
        pairs_g = cell_linked_pairs(pos, 2.0 * hg)
        _, ok = _mls_operators(pos, vol, pairs_g, hg)
        fallback = float((~ok).float().mean())
        g = mls_gradient(pos, c, vol, pairs_g, hg)
        ds = 2.0 * torch.linalg.vector_norm(g, dim=1)
        mag = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-30)
        kappa = fit_curvature(pos, -g / mag, ds, wide, h, fallback=ref.kappa,
                             method="fit_quadratic")
        i = cs.delta_s_integral(pos, mass, RHO0, ds, R)
        acc = (-SIGMA * kappa * ds)[:, None] * (-g / mag) / RHO0
        cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
        kw = float((kappa.abs() * ds).sum() / ds.sum())
        print(f"{dx:7.4f} {beta:5.2f} | {i:9.3f} {(i-1)*100:7.1f} "
              f"{cut/(-2*SIGMA):8.3f} {(kw/(1/R)-1)*100:10.1f} {fallback:8.3f}")