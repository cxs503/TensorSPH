"""Split the delta_s bias into gradient-magnitude vs shell-quadrature parts.

Exact reference (planar-interface limit, exact for R >> h): for a cubic-spline
mollified indicator, the gradient magnitude at distance d from the interface is

    |grad c|(d) = int_{-inf}^{inf} W(sqrt(d^2 + y^2)) dy ,

so int_0^inf |grad c| dd = 1/2 exactly (indicator jump 1 -> 1/2 over the fluid
half).  Comparing

  (a) sum_i 2|grad c|(d_i) V_i / (2 pi r_i)        (exact magnitude, same quadrature)
  (b) sum_i 2|grad_MLS c_i| V_i / (2 pi r_i)       (what the repo scores)

isolates the operator contribution from the quadrature contribution.
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
R, RHO0 = 0.02, 1000.0


def exact_gradient_magnitude(d, h, n=401, span=2.05):
    """|grad c|(d) = int W(sqrt(d^2+y^2)) dy for the given kernel/h."""
    y = torch.linspace(-span * h, span * h, n, dtype=torch.float64)
    dy = y[1] - y[0]
    dd = torch.atleast_1d(torch.as_tensor(d, dtype=torch.float64))
    r = torch.sqrt(dd[:, None] ** 2 + y[None, :] ** 2)
    val = (cubic_spline_kernel(r, h) * dy).sum(dim=-1)
    return val


# sanity: the analytic integral must be 1/2
h0 = 1.3 * 0.002
d = torch.linspace(0.0, 3.0 * h0, 4001, dtype=torch.float64)
g = exact_gradient_magnitude(d, h0)
integ = float(torch.trapezoid(g, d))
print(f"analytic check: int_0^inf |grad c| dd = {integ:.4f}  (must be 0.5000); "
      f"|grad c|(0) = {float(exact_gradient_magnitude(torch.tensor([0.0], dtype=torch.float64), h0)[0]):.1f} 1/m")

print(f"\n{'dx':>7s} {'N':>4s} {'R/h':>5s} | {'(a) exact |grad c|':>18s} "
      f"{'(b) MLS |grad c|':>17s} {'ratio b/a':>9s}")
for dx in (0.003, 0.002, 0.0015):
    h = 1.3 * dx
    pos, mass = ll.hexagonal_droplet(R, dx, RHO0)
    vol = mass / RHO0
    near = cell_linked_pairs(pos, 2.0 * h)
    c = kernel_sum_color(pos, mass, near, h, RHO0)
    gm = mls_gradient(pos, c, vol, near, h)
    mag = torch.linalg.vector_norm(gm, dim=1)
    r = torch.linalg.vector_norm(pos, dim=1)
    dd = (R - r).clamp_min(0.0)                      # distance into the fluid
    g_exact = exact_gradient_magnitude(dd, h)
    ds_a = 2.0 * g_exact
    ds_b = 2.0 * mag
    ia = cs.delta_s_integral(pos, mass, RHO0, ds_a, R)
    ib = cs.delta_s_integral(pos, mass, RHO0, ds_b, R)
    print(f"{dx:7.4f} {pos.shape[0]:4d} {R/h:5.1f} | {ia:18.3f} {ib:17.3f} "
          f"{ib/ia:9.3f}")
    # peak magnitudes
    shell = dd < 2.0 * h
    print(f"        |grad c| at shell: exact mean={float(g_exact[shell].mean()):7.1f} "
          f"max={float(g_exact.max()):7.1f} | MLS mean={float(mag[shell].mean()):7.1f} "
          f"max={float(mag.max()):7.1f}")