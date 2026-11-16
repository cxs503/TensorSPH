import sys, math
sys.path.insert(0, 'benchmarks/xval_fsi2'); sys.path.insert(0, 'src')
import torch
import farfield_force_audit as A
from tensorsph.fsi2_turek_hron import (FSI2Case, cylinder_surface_geometry,
                                       boundary_force_from_pressure, boundary_pressure_from_fluid)
from tensorsph.droplet_impact import cubic_spline_kernel as kfun
from tensorsph.neighbor_search import cell_linked_pairs as clp
from tensorsph.farfield_force import potential_flow_cylinder
DT = torch.float64


def poly_arm(fluid_pos, pressure, wall_pos, normals, pairs, h, n_wall, deg,
             cone_deg=None):
    n = fluid_pos.shape[0]
    i, j = pairs.unbind(dim=1)
    cross = (i < n) != (j < n)
    fi = torch.where(i < n, i, j)[cross]
    wi = torch.where(i < n, j, i)[cross] - n
    keep = wi < n_wall
    fi, wi = fi[keep], wi[keep]
    delta = fluid_pos[fi] - wall_pos[wi]
    r = torch.linalg.vector_norm(delta, dim=1)
    act = (r > 1e-12) & (r < 2.0 * h)
    if cone_deg is not None:
        # keep only fluid particles inside a cone of half-angle cone_deg about -n
        c = -(delta * normals[wi]).sum(1) / r.clamp_min(1e-30)
        act = act & (c > math.cos(math.radians(cone_deg)))
    fi, wi, delta, r = fi[act], wi[act], delta[act], r[act]
    w = kfun(r, h)
    ps = pressure[fi]
    u, v = delta[:, 0] / h, delta[:, 1] / h
    one = torch.ones_like(u)
    if deg == 1:
        basis = torch.stack((one, u, v), dim=1)
    else:
        basis = torch.stack((one, u, v, u * u, u * v, v * v), dim=1)
    m = basis.shape[1]
    lhs = torch.zeros((n_wall, m, m), dtype=DT)
    lhs = lhs.index_add(0, wi, w[:, None, None] * (basis[:, :, None] * basis[:, None, :]))
    rhs = torch.zeros((n_wall, m), dtype=DT)
    rhs = rhs.index_add(0, wi, w[:, None] * ps[:, None] * basis)
    wsum = torch.zeros(n_wall, dtype=DT).index_add(0, wi, w)
    reg = 1e-9 * wsum.clamp_min(1e-30)[:, None, None] * torch.eye(m, dtype=DT)
    sol = torch.linalg.solve(lhs + reg, rhs.unsqueeze(-1)).squeeze(-1)
    nmin = torch.zeros(n_wall, dtype=DT).index_add(0, wi, torch.ones_like(r))
    return sol[:, 0], wsum, nmin


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
    for thick in (3.0, 5.0):
        fp = A.annulus_lattice(centre, a + 0.5 * dx, a + thick * h, dx)
        _, pres = potential_flow_cylinder(fp, a, u_inf, circ, rho, centre=centre)
        pairs = clp(torch.cat((fp, pos_w), dim=0), 2.0 * h)
        row = [f"dx={dx} shell={thick}h nf={fp.shape[0]}"]
        for cone in (None, 75.0, 60.0, 45.0, 30.0):
            for deg in (1, 2):
                p, ws, nn = poly_arm(fp, pres, pos_w, nrm, pairs, h, n_wall, deg, cone)
                F = boundary_force_from_pressure(p, nrm, float(trib)).sum(0)
                e = (float(F[1]) - exact) / abs(exact)
                row.append(f"c{cone}/d{deg}:{100*e:+.1f}%(n{int(nn.min())})")
        print(" ".join(row), flush=True)