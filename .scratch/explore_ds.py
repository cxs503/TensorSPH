"""Where does the delta_s normalisation lose 26%?  Radial profile of c, |grad c|."""
import math
import torch
import importlib.util, pathlib

from tensorsph.csf_consistent import kernel_sum_color, mls_gradient
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ll", ROOT / "benchmarks/laplace_law/run.py")
ll = importlib.util.module_from_spec(spec); spec.loader.exec_module(ll)

R, dx, rho0 = 0.02, 0.002, 1000.0
h = 1.3 * dx
pos, mass = ll.hexagonal_droplet(R, dx, rho0)
vol = mass / rho0
pairs = cell_linked_pairs(pos, 2.0 * h)
r = torch.linalg.vector_norm(pos, dim=1)

c = kernel_sum_color(pos, mass, pairs, h, rho0)
grad_c = mls_gradient(pos, c, vol, pairs, h)
mag = torch.linalg.vector_norm(grad_c, dim=1)
print(f"N={pos.shape[0]}  c range {float(c.min()):.4f}..{float(c.max()):.4f}  "
      f"|grad c| max {float(mag.max()):.4f}")

nb = 14
edges = torch.linspace(0.80, 1.05, nb + 1, dtype=pos.dtype)
print("\n bin r/R      n    <c>     <|grad c|>   shell_contrib")
tot = 0.0
for k in range(nb):
    m = (r >= edges[k] * R) & (r < edges[k + 1] * R)
    if not bool(m.any()):
        continue
    rr = r[m]
    contrib = float((mag[m] * vol[m] / (2 * math.pi * rr)).sum())
    tot += contrib
    print(f" {float(edges[k]):.3f}-{float(edges[k+1]):.3f} {int(m.sum()):4d} "
          f" {float(c[m].mean()):.4f}  {float(mag[m].mean()):.4f}    {contrib:.4f}")
print(f"\n  int |grad c| dn (fluid side) = {tot:.4f}   (ideal 0.500 -> "
      f"delta_s=2x gives {2*tot:.3f} vs 1)")

outer = r > (R - 1.0 * dx)
inner = (r < (R - 1.0 * dx)) & (r > (R - 2.5 * dx))
print(f"  c(outer layer, r>R-dx) = {float(c[outer].mean()):.4f} ; "
      f"c(next in) = {float(c[inner].mean()):.4f} ; n_outer={int(outer.sum())}")
print(f"  drop/(c(outer)) implied: (1-{float(c[outer].mean()):.3f}) = "
      f"{1-float(c[outer].mean()):.3f}")