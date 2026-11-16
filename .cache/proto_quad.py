import sys, math
sys.path.insert(0, 'benchmarks/xval_fsi2'); sys.path.insert(0, 'src')
import torch
import farfield_force_audit as A
from tensorsph.fsi2_turek_hron import (FSI2Case, cylinder_surface_geometry,
                                       boundary_force_from_pressure,
                                       boundary_pressure_from_fluid,
                                       boundary_pressure_from_fluid_linear_exact)
from tensorsph.droplet_impact import cubic_spline_kernel as _k
class _DI: pass
di=_DI()
di.cubic_spline_kernel=_k
from tensorsph import fsi2_turek_hron as f2
from tensorsph.neighbor_search import cell_linked_pairs as clp
from tensorsph.farfield_force import potential_flow_cylinder
DT = torch.float64


def poly_pressure(fluid_pos, pressure, wall_pos, normals, pairs, h, wall_stop, deg):
    n = fluid_pos.shape[0]
    n_wall = wall_stop
    i, j = pairs.unbind(dim=1)
    crossing = (i < n) != (j < n)
    fi = torch.where(i < n, i, j)[crossing]
    wi = torch.where(i < n, j, i)[crossing] - n
    keep = wi < n_wall
    fi, wi = fi[keep], wi[keep]
    delta = fluid_pos[fi] - wall_pos[wi]
    r = torch.linalg.vector_norm(delta, dim=1)
    act = (r > 1e-12) & (r < 2.0 * h)
    fi, wi, delta, r = fi[act], wi[act], delta[act], r[act]
    w = di.cubic_spline_kernel(r, h)
    ps = pressure[fi]
    u, v = delta[:, 0] / h, delta[:, 1] / h
    if deg == 1:
        basis = torch.stack((torch.ones_like(u), u, v), dim=1)
    elif deg == 2:
        basis = torch.stack((torch.ones_like(u), u, v, u * u, u * v, v * v), dim=1)
    else:
        basis = torch.stack((torch.ones_like(u), u, v, u * u, u * v, v * v,
                             u ** 3, u * u * v, u * v * v, v ** 3), dim=1)
    m = basis.shape[1]
    lhs = torch.zeros((n_wall, m, m), dtype=DT)
    lhs = lhs.index_add(0, wi, w[:, None, None] * (basis[:, :, None] * basis[:, None, :]))
    rhs = torch.zeros((n_wall, m), dtype=DT)
    rhs = rhs.index_add(0, wi, w[:, None] * ps[:, None] * basis)
    wsum = torch.zeros(n_wall, dtype=DT).index_add(0, wi, w)
    scale = wsum.clamp_min(1e-30)[:, None, None]
    eye = torch.eye(m, dtype=DT).expand(n_wall, m, m)
    reg = 1e-10 * scale * eye
    sol = torch.linalg.solve(lhs + reg, rhs.unsqueeze(-1)).squeeze(-1)
    return sol[:, 0], wsum


for deg in (1, 2, 3):
    for thick in (3.0, 5.0):
        for dx in (0.02, 0.01):
            case = FSI2Case(); case.dx = dx
            a = 0.5 * case.cylinder_d
            u_inf = case.inlet_u_max * 2.0 / 3.0
            rho = case.rho0
            circ = 2 * math.pi * a * u_inf * 0.8
            exact = -rho * u_inf * circ
            centre = torch.tensor([case.cylinder_x, case.cylinder_y], dtype=DT)
            pos_w, nrm, trib = cylinder_surface_geometry(case)
            n_wall = pos_w.shape[0]
            h = case.h()
            fp = A.annulus_lattice(centre, a + 0.5 * dx, a + thick * h, dx)
            _, pres = potential_flow_cylinder(fp, a, u_inf, circ, rho, centre=centre)
            pairs = clp(torch.cat((fp, pos_w), dim=0), 2.0 * h)
            pdeg, ws = poly_pressure(fp, pres, pos_w, nrm, pairs, h, n_wall, deg)
            F = boundary_force_from_pressure(pdeg, nrm, float(trib)).sum(dim=0)
            print(f"deg={deg} thick={thick} dx={dx} nf={fp.shape[0]} "
                  f"lift_err={100*(float(F[1])-exact)/abs(exact):+.2f}% drag={float(F[0])/abs(exact):+.3f}")