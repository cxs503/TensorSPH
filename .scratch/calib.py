"""Calibrate the tabulated indicator profile against the measured c(d)."""
import importlib.util, pathlib, math
import torch
from tensorsph.csf_delta_s import kernel_profile
from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.csf_consistent import kernel_sum_color
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b", ROOT / "benchmarks/csf_delta_s/run.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

R, dx, RHO0 = 0.02, 0.002, 1000.0
h = 1.3 * dx
d, c, dcd = kernel_profile(h)
print(f"tabulated: c(0)={float(c[0]):.4f} c(0.5h)={float(c[len(c)//4]):.4f} "
      f"c(2h)={float(c[-1]):.4f}  peak dc_dd={float(dcd[0]):.1f}")
print(f"  kernel plane integral check: {float(2*c[-1]-1):.4f} (should be ~1.0)")

# kernel normalisation
yy = torch.linspace(-2.02*h, 2.02*h, 801, dtype=torch.float64)
xx = yy.clone()
X, Y = torch.meshgrid(xx, yy, indexing="ij")
W = cubic_spline_kernel(torch.sqrt(X**2 + Y**2), h)
print(f"  int W dA over support = {float((W.sum()*(yy[1]-yy[0])**2)):.4f}")

pos, mass = b.hexagonal_droplet(R, dx, RHO0)
near = cell_linked_pairs(pos, 2.0*h)
c_ks = kernel_sum_color(pos, mass, near, h, RHO0)
c_ex = b.exact_indicator(pos, R, h)
r = torch.linalg.vector_norm(pos, dim=1)
dd = (R - r)

print(f"\n{'d/h':>6s} {'n':>4s} {'<c_exact>':>10s} {'<c_kernelsum>':>14s} {'table c(d)':>11s}")
edges = torch.linspace(-0.2, 2.2, 13)*h
for k in range(len(edges)-1):
    m = (dd >= edges[k]) & (dd < edges[k+1])
    if not bool(m.any()):
        continue
    mid = 0.5*(edges[k]+edges[k+1])
    idx = int(torch.searchsorted(d, mid).clamp(0, d.numel()-1))
    print(f"{float(mid/h):6.2f} {int(m.sum()):4d} {float(c_ex[m].mean()):10.4f} "
          f"{float(c_ks[m].mean()):14.4f} {float(c[idx]):11.4f}")