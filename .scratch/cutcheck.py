import math, sys, torch
torch.set_num_threads(2)
sys.path.insert(0, "src")
import importlib.util
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
law = load("law", "benchmarks/laplace_law/run.py")
from tensorsph.csf_consistent import csf_fields, surface_tension_acceleration
from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

R, rho0, sigma = 0.02, 1000.0, 7.2
for dx in (0.003, 0.002, 0.0015):
    h = 1.3*dx
    pos, mass = law.hexagonal_droplet(R, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0*h)
    rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0*h))
    for norm in ("fixed", "jump"):
        st = csf_fields(pos, mass, wide, h, rho0, curvature_scheme="fit_quadratic", delta_s_normalization=norm)
        acc, _ = surface_tension_acceleration(pos, rho, mass, wide, h, sigma, rho0=rho0,
                                             curvature_scheme="fit_quadratic", delta_s_normalization=norm)
        r = torch.linalg.vector_norm(pos, dim=1)
        x = pos[:, 0] > 0
        cut_acc = float((mass[x, None]*acc[x]).sum(dim=0)[0])            # as benchmark measures
        cut_vol = float((mass[x]*rho[x]/rho0*acc[x, 0]).sum())           # continuum: V0*f = (m/rho0)*rho*a
        w = st.delta_s
        rw = float((w*r).sum()/w.sum())/R
        rho_w = float((w*rho).sum()/w.sum())/rho0
        print(f"dx={dx} {norm:5s} Fcut_acc={cut_acc/-14.4:.3f} Fcut_vol={cut_vol/-14.4:.3f} "
              f"rho_w/rho0={rho_w:.3f} r_w/R={rw:.4f} int_ds={float(((st.delta_s*mass/rho0*((r>0.6*R)&(r<1.05*R)))/(2*math.pi*r)).sum()):.3f}")