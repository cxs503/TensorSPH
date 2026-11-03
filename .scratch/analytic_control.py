"""Analytic control for the delta_s deficit.

Builds the *exact* mollified indicator of a circular interface,

    c(x) = fraction of the cubic-spline kernel mass that lies inside the drop,

by high-resolution polar quadrature.  On a circle this field is analytic and
symmetric, so the exact in-fluid integral of |grad c| is 1/2 (the indicator goes
from 1 in the bulk to 1/2 on the interface and to 0 in the void).

Sampling that exact field on the hex lattice and running the very same
MLS-gradient + shell quadrature used to score delta_s separates two hypotheses:

  H1 (field): the kernel-sum colour function is the culprit;
  H2 (estimator/operator): the grading pipeline cannot resolve any field whose
      transition is ~1 lattice layer thick, so `int delta_s dn ~ 0.74` is a
      measurement bias, not a 26 % force deficit.
"""
import math
import importlib.util
import pathlib

import torch

from tensorsph.csf_consistent import kernel_sum_color, mls_gradient
from tensorsph.droplet_impact import cubic_spline_kernel
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


def exact_colour(pos, radius, h, n_r=160, n_t=240):
    """c_i = (kernel mass inside the circle) via polar quadrature around x_i."""
    n = pos.shape[0]
    rr = torch.linspace(1e-9, 2.0 * h, n_r, dtype=pos.dtype)
    th = torch.linspace(0.0, 2.0 * math.pi, n_t + 1, dtype=pos.dtype)[:-1]
    w_r = torch.full_like(rr, rr[1] - rr[0])
    w_t = torch.full_like(th, th[1] - th[0])
    jac = (rr * w_r)[:, None] * w_t[None, :]          # (n_r, n_t)
    out = torch.zeros(n, dtype=pos.dtype)
    for s in range(0, n, 64):
        blk = pos[s:s + 64]                            # (m, 2)
        dx = rr[:, None] * torch.cos(th)[None, :]
        dy = rr[:, None] * torch.sin(th)[None, :]
        px = blk[:, 0, None, None] + dx[None]          # (m, n_r, n_t)
        py = blk[:, 1, None, None] + dy[None]
        inside = (px * px + py * py) <= radius * radius
        w = cubic_spline_kernel(rr, h)[:, None] * jac  # (n_r, n_t)
        out[s:s + 64] = (w[None] * inside.to(pos.dtype)).sum(dim=(1, 2))
    return out.clamp(0.0, 1.0)


def shell_integral(pos, mass, delta_s, radius, lo=0.6, hi=1.05):
    return cs.delta_s_integral(pos, mass, RHO0, delta_s, radius)


print(f"{'dx':>7s} {'N':>4s} {'R/h':>5s} | {'c_rms(kernel-sum vs exact)':>26s} "
      f"| {'int|grad c|dn':>13s} {'int|grad c_exact|dn':>19s}")
for dx in (0.003, 0.002, 0.0015):
    h = 1.3 * dx
    pos, mass = ll.hexagonal_droplet(R, dx, RHO0)
    vol = mass / RHO0
    near = cell_linked_pairs(pos, 2.0 * h)
    c_ks = kernel_sum_color(pos, mass, near, h, RHO0)
    c_ex = exact_colour(pos, R, h)
    rms = float(torch.sqrt(((c_ks - c_ex) ** 2).mean()))
    g_ks = mls_gradient(pos, c_ks, vol, near, h)
    g_ex = mls_gradient(pos, c_ex, vol, near, h)
    ds_ks = 2.0 * torch.linalg.vector_norm(g_ks, dim=1)
    ds_ex = 2.0 * torch.linalg.vector_norm(g_ex, dim=1)
    i_ks = shell_integral(pos, mass, ds_ks, R)
    i_ex = shell_integral(pos, mass, ds_ex, R)
    r = torch.linalg.vector_norm(pos, dim=1)
    print(f"{dx:7.4f} {pos.shape[0]:4d} {R/h:5.1f} | {rms:26.4f} "
          f"| {i_ks:13.3f} {i_ex:19.3f}")
    print(f"        c_exact: bulk={float(c_ex[r < 0.6*R].mean()):.4f} "
          f"outer={float(c_ex[r > R - dx].mean()):.4f} min={float(c_ex.min()):.4f}"
          f" | c_ks outer={float(c_ks[r > R - dx].mean()):.4f}")