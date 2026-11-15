import math, sys
import numpy as np
import torch

torch.set_num_threads(8)
sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks/xval_fsi2")

import tensorsph.fsi2_turek_hron as F
from tensorsph import droplet_impact as di
from tensorsph.neighbor_search import cell_linked_pairs
from run_rigid_cylinder import rig_case

imp = (len(sys.argv) > 1 and sys.argv[1] == "imp")
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 700
c = rig_case(0.02)
c.wall_impedance = imp
st = F.initial_state(c)
dt = c.dt()
for k in range(nsteps):
    row = F.step_fsi2(st, c, dt)

cfg = c.fluid_config(); cfg.dt = dt
wall = torch.cat((st.wall_static, st.cylinder_particles), 0)
all_pos = torch.cat((st.fluid_pos, wall), 0)
pairs = cell_linked_pairs(all_pos, 2.0 * c.h())
wall_vel = torch.zeros_like(wall)
rho, pressure, acc = di.rhs(st.fluid_pos, st.fluid_vel, st.fluid_mass, cfg,
                            pairs=pairs, wall=wall, dt=dt, wall_vel=wall_vel)
print("fluid p: min %.3f max %.3f  (q-based stagnation 0.5 rho Umax^2 = %.2f)"
      % (float(pressure.min()), float(pressure.max()),
         0.5 * c.rho0 * c.inlet_u_max ** 2))

cs = st.wall_static.shape[0]
nc = st.cylinder_particles.shape[0]
nrm = st.cylinder_normals
trib = st.cylinder_tributary
p_avg, w = F.boundary_pressure_from_fluid(
    st.fluid_pos, pressure, wall, cs, pairs, c.h(),
    normals=nrm if imp else None, vel=st.fluid_vel if imp else None,
    wall_vel=wall_vel if imp else None, rho0=c.rho0, sound_speed=c.sound_speed)
p_fit, wf = F.boundary_pressure_from_fluid_linear_exact(
    st.fluid_pos, pressure, wall, cs, cs + nc, pairs, c.h())

print("station  theta  x      y      n_x   n_y   p_avg   p_fit   p_near  nNeigh")
for i in range(nc):
    x, y = float(st.cylinder_particles[i, 0]), float(st.cylinder_particles[i, 1])
    d = torch.linalg.vector_norm(st.fluid_pos - st.cylinder_particles[i], dim=1)
    j = int(torch.argmin(d))
    print("%3d  %6.1f  %.3f %.3f %+5.2f %+5.2f  %7.2f %7.2f %7.2f  %d"
          % (i, math.degrees(math.atan2(y - 0.2, x - 0.2)), x, y,
             float(nrm[i, 0]), float(nrm[i, 1]), float(p_avg[i]), float(p_fit[i]),
             float(pressure[j]),
             int(((all_pos[pairs[:, 0]] - st.cylinder_particles[i]).norm(dim=1) < 2 * c.h()).sum())
             if False else -1))
q = 0.5 * c.rho0 * c.inlet_u_mean() ** 2 * c.cylinder_d
for name, p in (("avg", p_avg), ("fit", p_fit)):
    fx = float((-p * nrm[:, 0] * trib).sum())
    fy = float((-p * nrm[:, 1] * trib).sum())
    print("%s: Fx=%+.3f N/m Cd=%+.2f   Fy=%+.3f Cl=%+.2f" % (name, fx, fx / q, fy, fy / q))
print("t=%.3f s, nfluid=%d" % (st.time, st.fluid_pos.shape[0]))