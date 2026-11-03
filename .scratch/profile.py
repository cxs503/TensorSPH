import math, torch
torch.set_num_threads(2)
import sys
sys.path.insert(0, "src")
from tensorsph.csf_consistent import csf_fields
from tensorsph.droplet_impact import density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs
import importlib.util
spec = importlib.util.spec_from_file_location("law", "benchmarks/laplace_law/run.py")
law = importlib.util.module_from_spec(spec); spec.loader.exec_module(law)

R, rho0 = 0.02, 1000.0
for dx in (0.003, 0.002, 0.0015):
    h = 1.3 * dx
    pos, mass = law.hexagonal_droplet(R, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0 * h)
    st = csf_fields(pos, mass, wide, h, rho0, curvature_scheme="fit_quadratic")
    r = torch.linalg.vector_norm(pos, dim=1)
    m = st.delta_s > 0.05 * st.delta_s.max()
    print(f"--- dx={dx} h={h:.5f} N={pos.shape[0]} active={int(m.sum())}")
    # radial profile of c on the shell
    order = torch.argsort(r[m])
    rr, cc, gg = r[m][order], st.color[m][order], st.delta_s[m][order]
    print("  r/R      c      delta_s   |gradc|*2h")
    for k in range(0, min(len(rr), 200), max(1, len(rr)//25)):
        print(f"  {float(rr[k])/R:.4f}  {float(cc[k]):.4f}  {float(gg[k]):.3f}   {float(gg[k])*2*h:.3f}")
    print("  c_bulk(interior r<0.5R):", float(st.color[r < 0.5*R].mean()),
          " c_min active:", float(st.color[m].min()),
          " c at outer 5%:", float(torch.quantile(st.color[m & (r > 0.9*R)], 0.1)) if (m & (r>0.9*R)).sum()>3 else None)
    dv = mass / rho0
    integ = float(((st.delta_s * dv) * (m & (r > 0.6*R) & (r < 1.05*R))
                   / (2*math.pi*r.clamp_min(1e-30))).sum())
    print(f"  int delta_s dn = {integ:.4f}")