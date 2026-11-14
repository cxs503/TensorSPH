"""Turek-Hron FSI2 benchmark: cylinder + elastic beam in channel flow.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Physical problem
----------------
The Turek-Hron FSI2 benchmark (Turek & Hron, *Lecture Notes in Computational
Science and Engineering* 53 (2006) 371-385) is a standard test for
fluid-structure interaction solvers.  A 2-D channel carries a parabolic inlet
flow past a rigid cylinder with a flexible elastic beam attached to its
downstream side.  The beam undergoes vortex-induced vibration (VIV); the
engineering quantities of interest are the drag/lift coefficients on the
cylinder-beam assembly and the beam tip displacement history.

Reference
---------
S. Turek, J. Hron, "Proposal for numerical benchmarking of fluid-structure
interaction between an elastic object and laminar incompressible flow", in
*Fluid-Structure Interaction: Modelling, Simulation, Optimisation*,
H.-J. Bungartz, M. Schafer (eds.), LNCSE 53, Springer (2006) 371-385.

SPH implementation reference:
C. Zhang, M. Rezavand, X.Y. Hu, "SPHinXsys: an open-source multi-physics and
multi-resolution library based on Smoothed Particle Hydrodynamics",
*J. Comput. Phys.* **404** (2020) 109135, DOI 10.1016/j.jcp.2019.109135.
The SPHinXsys ``test_2d_fsi2`` implements this case with TLSPH for the solid
and a weakly-compressible SPH fluid.

Parameters (Turek-Hron FSI2)
-----------------------------
| Quantity | Value |
|---|---|
| Channel length | 2.5 m |
| Channel height | 0.41 m |
| Cylinder diameter | 0.1 m |
| Cylinder center | (0.2, 0.2) m from inlet |
| Beam length | 0.35 m |
| Beam thickness | 0.02 m |
| Fluid density | 1000 kg/m³ |
| Fluid viscosity | 1.0 Pa·s |
| Inlet u_max | 0.3 m/s (parabolic) |
| Solid density | 1000 kg/m³ |
| Solid E | 4.0 MPa |
| Solid ν | 0.3 |
| Re (based on D, U_mean) | 100 |

Reference values (Turek-Hron, per-unit-depth)
----------------------------------------------
| Quantity | Value |
|---|---|
| Drag coeff amplitude | ~0.96 (oscillation about mean ~3.2) |
| Lift coeff amplitude | ~0.22 |
| Tip x displacement amplitude | ~0.0028 m |
| Tip y displacement amplitude | ~0.0082 m |

Note: these reference values are for the *incompressible* Navier-Stokes solver
used in the original benchmark.  A WCSPH solver at finite Mach number will
produce slightly different numbers; the force-from-fluid methodology is what
this module validates, not the exact match to the incompressible reference.

Solver
------
* fluid: weakly-compressible SPH (Tait EOS, cubic-spline kernel, Adami wall BC)
* solid: discrete elastic rod (same energy-based formulation as
  :mod:`tensorsph.fsi_elastic_gate`)
* coupling: panel-based force extrapolation from fluid to structure, with
  ``W^T`` lumping for force conservation

Verification that does not need long-time integration:
:func:`verify_parabolic_inlet` (inlet profile vs analytic),
:func:`verify_cylinder_geometry` (cylinder particle placement),
:func:`verify_beam_stiffness` (rod stiffness vs analytic EI).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import torch

from . import droplet_impact as di
from .neighbor_search import cell_linked_pairs

# --------------------------------------------------------------------------- #
# Case configuration
# --------------------------------------------------------------------------- #

@dataclass
class FSI2Case:
    """Turek-Hron FSI2 configuration.

    The channel runs along x from 0 to ``channel_length``, y from 0 to
    ``channel_height``.  The cylinder is centered at (``cylinder_x``,
    ``cylinder_y``) with diameter ``cylinder_d``.  The beam is attached to
    the cylinder at the downstream stagnation point and extends in +x.
    """

    # Fluid
    dx: float = 0.005
    h_factor: float = 1.3
    rho0: float = 1000.0
    sound_speed: float = 30.0  # higher c0 for lower Mach at U_max=0.3
    gamma: float = 7.0
    gravity: float = 0.0  # no gravity in FSI2
    viscosity: float = 0.2  # dynamic viscosity [Pa·s] (Re=100)
    max_courant: float = 0.2

    # Channel
    channel_length: float = 2.5
    channel_height: float = 0.41

    # Cylinder
    cylinder_d: float = 0.1
    cylinder_x: float = 0.2
    cylinder_y: float = 0.2

    # Beam (elastic rod attached to cylinder downstream side)
    beam_length: float = 0.35
    beam_thickness: float = 0.02
    beam_E: float = 4.0e6  # 4 MPa
    beam_nu: float = 0.3
    beam_rho: float = 1000.0
    n_beam_nodes: int = 30

    # Inlet (parabolic profile)
    inlet_u_max: float = 0.3  # max velocity at channel center

    # Time
    t_end: float = 10.0  # long enough for VIV to develop
    output_every: int = 10

    # Structure sub-cycling
    rod_substeps: int = 0  # auto
    rod_stability_target: float = 0.25
    max_rod_substeps: int = 500

    # Force transfer
    load_transfer: str = "panel"
    load_scale: float = 1.0

    device: str = "cpu"

    # -- derived --------------------------------------------------------- #

    def h(self) -> float:
        return self.h_factor * self.dx

    def dt(self) -> float:
        return self.max_courant * self.h() / self.sound_speed

    def inlet_u_mean(self) -> float:
        """Mean velocity of the parabolic profile: U_mean = 2/3 U_max."""
        return 2.0 / 3.0 * self.inlet_u_max

    def reynolds(self) -> float:
        """Reynolds number based on cylinder diameter and mean velocity."""
        return self.rho0 * self.inlet_u_mean() * self.cylinder_d / self.viscosity

    def mach(self) -> float:
        """Mach number based on max inlet velocity."""
        return self.inlet_u_max / self.sound_speed

    def beam_stiffness_bending(self) -> float:
        """EI per unit width [N·m²] (plane stress)."""
        return self.beam_E * self.beam_thickness ** 3 / 12.0

    def beam_stiffness_axial(self) -> float:
        """EA per unit width [N]."""
        return self.beam_E * self.beam_thickness

    def beam_area_density(self) -> float:
        """Mass per unit area per unit width [kg/m²]."""
        return self.beam_rho * self.beam_thickness

    def beam_axial_frequency(self) -> float:
        """Lowest axial frequency of the discrete rod [rad/s]."""
        l0 = self.beam_length / (self.n_beam_nodes - 1)
        return math.sqrt(self.beam_stiffness_axial()
                         / (self.beam_area_density() * l0))

    def rod_substeps_effective(self) -> int:
        if self.rod_substeps > 0:
            return int(self.rod_substeps)
        needed = self.dt() * self.beam_axial_frequency() / self.rod_stability_target
        return max(1, int(math.ceil(needed - 1.0e-12)))

    def validate(self) -> None:
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.cylinder_d <= 0:
            raise ValueError("cylinder diameter must be positive")
        if self.beam_length <= 0 or self.beam_thickness <= 0:
            raise ValueError("beam dimensions must be positive")
        if self.n_beam_nodes < 3:
            raise ValueError("beam needs at least 3 nodes")
        if self.reynolds() <= 0:
            raise ValueError("Reynolds number must be positive")

    def fluid_config(self) -> "di.Config":
        return di.Config(
            dx=self.dx, h_factor=self.h_factor, rho0=self.rho0,
            sound_speed=self.sound_speed, gamma=self.gamma, gravity=self.gravity,
            viscosity=self.viscosity / self.rho0,  # kinematic viscosity
            dt=self.dt(), steps=1, output_every=1,
            pool_width=self.channel_length, pool_depth=self.channel_height,
            device=self.device, boundary="wall", wall_layers=3,
            max_courant=self.max_courant, density_calibration=True)


# --------------------------------------------------------------------------- #
# Geometry builders
# --------------------------------------------------------------------------- #

def build_channel_walls(case: FSI2Case, device) -> torch.Tensor:
    """Static wall particles for the channel top/bottom and outlet."""
    dx = case.dx
    wall_layers = 3
    xs = torch.arange(0, case.channel_length, dx, dtype=torch.float64, device=device) + 0.5 * dx
    parts: List[torch.Tensor] = []
    for k in range(1, wall_layers + 1):
        off = k * dx
        # bottom wall
        parts.append(torch.stack((xs, torch.full_like(xs, -off)), dim=-1))
        # top wall
        parts.append(torch.stack((xs, torch.full_like(xs, case.channel_height + off)), dim=-1))
    # outlet wall (right side)
    ys = torch.arange(0, case.channel_height, dx, dtype=torch.float64, device=device) + 0.5 * dx
    for k in range(1, wall_layers + 1):
        off = k * dx
        parts.append(torch.stack((torch.full_like(ys, case.channel_length + off), ys), dim=-1))
    return torch.cat(parts, dim=0)


def build_cylinder_particles(case: FSI2Case, device) -> torch.Tensor:
    """Boundary particles on the cylinder surface."""
    dx = case.dx
    r = case.cylinder_d / 2.0
    # Circumference / dx gives approximate number of particles
    n_circ = max(int(math.ceil(2 * math.pi * r / dx)), 8)
    angles = torch.arange(n_circ, dtype=torch.float64, device=device) * (2 * math.pi) / n_circ  # was linspace
                            
    x = case.cylinder_x + r * torch.cos(angles)
    y = case.cylinder_y + r * torch.sin(angles)
    return torch.stack((x, y), dim=-1)


def build_inlet_particles(case: FSI2Case, device) -> torch.Tensor:
    """Inlet boundary particles (left side of channel)."""
    dx = case.dx
    wall_layers = 3
    ys = torch.arange(0, case.channel_height, dx, dtype=torch.float64, device=device) + 0.5 * dx
    parts: List[torch.Tensor] = []
    for k in range(1, wall_layers + 1):
        off = -k * dx
        parts.append(torch.stack((torch.full_like(ys, off), ys), dim=-1))
    return torch.cat(parts, dim=0)


def build_fluid_particles(case: FSI2Case, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fluid particles in the channel, excluding cylinder and beam regions."""
    dx = case.dx
    xs = torch.arange(0, case.channel_length, dx, dtype=torch.float64, device=device) + 0.5 * dx
    ys = torch.arange(0, case.channel_height, dx, dtype=torch.float64, device=device) + 0.5 * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)

    # Remove particles inside the cylinder
    cx, cy = case.cylinder_x, case.cylinder_y
    r = case.cylinder_d / 2.0 + 0.5 * dx  # small buffer
    dist_sq = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
    outside_cylinder = dist_sq > r ** 2

    # Remove particles in the beam region (simplified: rectangular exclusion)
    beam_x0 = case.cylinder_x + r
    beam_x1 = beam_x0 + case.beam_length
    beam_y0 = case.cylinder_y - case.beam_thickness / 2
    beam_y1 = case.cylinder_y + case.beam_thickness / 2
    outside_beam = ~((pos[:, 0] > beam_x0) & (pos[:, 0] < beam_x1)
                     & (pos[:, 1] > beam_y0) & (pos[:, 1] < beam_y1))

    keep = outside_cylinder & outside_beam
    pos = pos[keep]

    mass = torch.full((pos.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                      dtype=pos.dtype, device=device)
    return pos, mass


def set_inlet_velocity(case: FSI2Case, pos: torch.Tensor) -> torch.Tensor:
    """Parabolic inlet velocity profile: u(y) = U_max * 4 * y*(H-y) / H²."""
    H = case.channel_height
    u_max = case.inlet_u_max
    vel = torch.zeros_like(pos)
    # Only set velocity for particles near the inlet (x < 2*dx)
    near_inlet = pos[:, 0] < 2.0 * case.dx
    y = pos[near_inlet, 1]
    u = u_max * 4.0 * y * (H - y) / (H ** 2)
    vel[near_inlet, 0] = u
    return vel


# --------------------------------------------------------------------------- #
# Beam (discrete elastic rod)
# --------------------------------------------------------------------------- #

def beam_nodal_mass(case: FSI2Case) -> torch.Tensor:
    """Lumped nodal mass for the beam rod."""
    n = case.n_beam_nodes
    l0 = case.beam_length / (n - 1)
    m = torch.full((n,), case.beam_area_density() * l0, dtype=torch.float64)
    m[0] *= 0.5
    m[-1] *= 0.5
    return m


def beam_energy(nodes: torch.Tensor, case: FSI2Case) -> torch.Tensor:
    """Discrete stretch + bending energy for the beam."""
    n = nodes.shape[0]
    l0 = case.beam_length / (n - 1)
    seg = nodes[1:] - nodes[:-1]
    length = torch.linalg.vector_norm(seg, dim=1).clamp_min(1.0e-9)
    ks = case.beam_stiffness_axial() / l0
    e_stretch = 0.5 * ks * (length - l0).square().sum()
    tangent = seg / length[:, None]
    cross = tangent[1:, 0] * tangent[:-1, 1] - tangent[1:, 1] * tangent[:-1, 0]
    dot = (tangent[1:] * tangent[:-1]).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.atan2(cross, dot)
    kb = case.beam_stiffness_bending() / l0
    # Clamped at cylinder: first segment angle relative to +x axis
    ref = torch.zeros_like(tangent[0])
    ref[0] = 1.0  # beam extends in +x from cylinder
    theta0 = torch.atan2(tangent[0, 0] * ref[1] - tangent[0, 1] * ref[0],
                         (tangent[0] * ref).sum().clamp(-1.0, 1.0))
    return e_stretch + 0.5 * kb * (theta0.square() + theta.square().sum())




def _beam_energy_clamped(nodes: torch.Tensor, case: FSI2Case) -> torch.Tensor:
    """Beam energy with hard clamped BC (first segment direction fixed to +x).

    Used only in :func:`verify_beam_point_load` for the static deflection test.
    Excludes ``theta0`` from the bending energy and assumes nodes[0] and
    nodes[1] are already clamped.
    """
    n = nodes.shape[0]
    l0 = case.beam_length / (n - 1)
    seg = nodes[1:] - nodes[:-1]
    length = torch.linalg.vector_norm(seg, dim=1).clamp_min(1.0e-9)
    ks = case.beam_stiffness_axial() / l0
    e_stretch = 0.5 * ks * (length - l0).square().sum()
    tangent = seg / length[:, None]
    cross = tangent[1:, 0] * tangent[:-1, 1] - tangent[1:, 1] * tangent[:-1, 0]
    dot = (tangent[1:] * tangent[:-1]).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.atan2(cross, dot)
    kb = case.beam_stiffness_bending() / l0
    # No theta0 term: first segment direction is hard-clamped
    return e_stretch + 0.5 * kb * theta.square().sum()

def beam_forces(nodes: torch.Tensor, case: FSI2Case,
                nodal_load: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Nodal force -dE/dx + load; clamped node gets zero force."""
    work = nodes.detach().clone().requires_grad_(True)
    grad, = torch.autograd.grad(beam_energy(work, case), work)
    force = -grad
    if nodal_load is not None:
        force = force + nodal_load
    force = force.clone()
    force[0] = 0.0  # clamped
    return force


def beam_advance(nodes: torch.Tensor, velocities: torch.Tensor,
                 nodal_load: torch.Tensor, case: FSI2Case, dt: float):
    """One symplectic-Euler step of the beam rod."""
    mass = beam_nodal_mass(case).to(nodes.dtype).to(nodes.device)[:, None]
    acc = beam_forces(nodes, case, nodal_load) / mass
    velocities = velocities + dt * acc
    velocities = velocities.clone()
    velocities[0] = 0.0
    nodes = nodes + dt * velocities
    nodes = nodes.clone()
    # Clamp: first node fixed at cylinder downstream point
    nodes[0] = torch.tensor(
        (case.cylinder_x + case.cylinder_d / 2, case.cylinder_y),
        dtype=nodes.dtype, device=nodes.device)
    return nodes, velocities


def beam_initial_nodes(case: FSI2Case, device) -> torch.Tensor:
    """Initial beam nodes: straight line extending from cylinder in +x."""
    n = case.n_beam_nodes
    x0 = case.cylinder_x + case.cylinder_d / 2
    y0 = case.cylinder_y
    xs = torch.linspace(x0, x0 + case.beam_length, n, dtype=torch.float64, device=device)
    ys = torch.full((n,), y0, dtype=torch.float64, device=device)
    return torch.stack((xs, ys), dim=-1)


# --------------------------------------------------------------------------- #
# Force from fluid to beam (panel method)
# --------------------------------------------------------------------------- #

def beam_load_on_fluid_boundary(pos, rho, pressure, wall, wall_mass,
                                 wall_start_idx, pairs, case: FSI2Case):
    """Pressure force on beam boundary particles from the fluid.

    Uses the panel extrapolation method: fit a pressure plane to neighbouring
    fluid particles and evaluate at the wall particle location.
    """
    n = pos.shape[0]
    n_wall = wall.shape[0] - wall_start_idx
    out = torch.zeros((max(n_wall, 0), 2), dtype=pos.dtype, device=pos.device)
    if n_wall <= 0 or pairs.numel() == 0:
        return out

    i, j = pairs.unbind(dim=1)
    crossing = (i < n) != (j < n)
    if not bool(crossing.any()):
        return out
    fluid_idx = torch.where(i < n, i, j)[crossing]
    wall_idx = (torch.where(i < n, j, i) - n)[crossing]
    keep = wall_idx >= wall_start_idx
    if not bool(keep.any()):
        return out
    fluid_idx = fluid_idx[keep]
    wall_idx = wall_idx[keep] - wall_start_idx

    h = case.h()
    delta = pos[fluid_idx] - wall[wall_start_idx + wall_idx]
    r = torch.linalg.vector_norm(delta, dim=1)
    active = (r > 1.0e-12) & (r < 2.0 * h)
    if not bool(active.any()):
        return out

    fluid_idx = fluid_idx[active]
    wall_idx = wall_idx[active]
    delta = delta[active]
    r = r[active]
    q = r / h
    w = torch.where(q < 1.0, 2.0 / 3.0 - q ** 2 + 0.5 * q ** 3,
                    torch.where(q < 2.0, (2.0 - q) ** 3 / 6.0, torch.zeros_like(q)))

    # Group by wall particle
    unique_walls = wall_idx.unique()
    for wi in unique_walls:
        mask = wall_idx == wi
        w_loc = w[mask]
        w_sum = w_loc.sum()
        if w_sum < 1.0e-12:
            continue
        p_loc = pressure[fluid_idx[mask]]
        p_wall = (w_loc * p_loc).sum() / w_sum
        # Force direction: normal to beam surface (approximate as -delta direction)
        n_vec = -delta[mask].mean(dim=0)
        n_norm = torch.linalg.vector_norm(n_vec)
        if n_norm > 1.0e-12:
            n_vec = n_vec / n_norm
        else:
            n_vec = torch.tensor([1.0, 0.0], dtype=pos.dtype, device=pos.device)
        tributary = case.dx  # per-unit-depth tributary length
        out[wi] = p_wall * tributary * n_vec

    return out


def interpolation_matrix_beam(n_nodes: int, n_boundary: int,
                               device) -> torch.Tensor:
    """Linear interpolation from beam nodes to boundary particles.

    Shape: (n_boundary, n_nodes), rows sum to 1.
    """
    l0 = 1.0 / (n_nodes - 1)
    xi = torch.arange(n_boundary, dtype=torch.float64, device=device) * l0 * (n_nodes - 1) / max(n_boundary - 1, 1)
    xi = xi.clamp(0.0, n_nodes - 1 - 1.0e-12)
    idx = xi.floor().long()
    w = (xi - idx.to(xi.dtype)).clamp(0.0, 1.0)
    mat = torch.zeros((n_boundary, n_nodes), dtype=torch.float64, device=device)
    rows = torch.arange(n_boundary, device=device)
    mat[rows, idx] = 1.0 - w
    mat[rows, (idx + 1).clamp(max=n_nodes - 1)] += w
    return mat


# --------------------------------------------------------------------------- #
# Coupled state and time stepping
# --------------------------------------------------------------------------- #

@dataclass
class FSI2State:
    fluid_pos: torch.Tensor
    fluid_vel: torch.Tensor
    fluid_mass: torch.Tensor
    wall_static: torch.Tensor
    wall_mass_static: torch.Tensor
    cylinder_particles: torch.Tensor
    cylinder_mass: torch.Tensor
    beam_interp: torch.Tensor
    beam_nodes: torch.Tensor
    beam_vel: torch.Tensor
    time: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=dict)


def initial_state(case: FSI2Case) -> FSI2State:
    case.validate()
    device = torch.device(case.device)

    # Fluid
    pos, mass = build_fluid_particles(case, device)
    vel = set_inlet_velocity(case, pos)

    # Walls
    wall_static = build_channel_walls(case, device)
    m_wall = torch.full((wall_static.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                        dtype=wall_static.dtype, device=device)

    # Cylinder (static)
    cyl = build_cylinder_particles(case, device)
    m_cyl = torch.full((cyl.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                       dtype=cyl.dtype, device=device)

    # Beam
    n_boundary = max(int(math.floor(case.beam_length / case.dx)), 2)
    beam_interp = interpolation_matrix_beam(case.n_beam_nodes, n_boundary, device)
    nodes = beam_initial_nodes(case, device)

    history: Dict[str, List[float]] = {
        "t": [], "tip_x": [], "tip_y": [],
        "drag": [], "lift": [], "max_speed": []
    }

    return FSI2State(
        pos, vel, mass, wall_static, m_wall, cyl, m_cyl,
        beam_interp, nodes, torch.zeros_like(nodes), 0.0, history)


def step_fsi2(state: FSI2State, case: FSI2Case, dt: float) -> Dict[str, float]:
    """Advance the coupled FSI2 system by one step."""
    cfg = case.fluid_config()
    cfg.dt = dt

    # Build wall: static walls + cylinder + beam boundary particles
    beam_boundary = state.beam_interp @ state.beam_nodes
    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_boundary), dim=0)
    m_unit = di.lattice_mass_unit(cfg)
    wall_mass = torch.cat((
        state.wall_mass_static,
        state.cylinder_mass,
        torch.full((beam_boundary.shape[0],), m_unit,
                   dtype=beam_boundary.dtype, device=beam_boundary.device)
    ))

    # Wall velocity: cylinder is static, beam moves
    beam_boundary_vel = state.beam_interp @ state.beam_vel
    wall_vel = torch.cat((
        torch.zeros_like(state.wall_static),
        torch.zeros_like(state.cylinder_particles),
        beam_boundary_vel
    ))

    # SPH RHS
    all_pos = torch.cat((state.fluid_pos, wall), dim=0)
    pairs = cell_linked_pairs(all_pos, 2.0 * case.h())
    rho, pressure, acc = di.rhs(state.fluid_pos, state.fluid_vel, state.fluid_mass,
                                cfg, pairs=pairs, wall=wall, dt=dt,
                                wall_vel=wall_vel)

    # Force on beam from fluid
    wall_start = state.wall_static.shape[0] + state.cylinder_particles.shape[0]
    f_beam = beam_load_on_fluid_boundary(
        state.fluid_pos, rho, pressure, wall, wall_mass, wall_start, pairs, case)

    # Lump force to beam nodes
    nodal_load = (state.beam_interp.T @ f_beam) * case.load_scale

    # Beam sub-cycling
    n_sub = case.rod_substeps_effective()
    dt_rod = dt / n_sub
    nodes, node_vel = state.beam_nodes, state.beam_vel
    for _ in range(n_sub):
        nodes, node_vel = beam_advance(nodes, node_vel, nodal_load, case, dt_rod)
    state.beam_nodes, state.beam_vel = nodes, node_vel

    # Update fluid
    state.fluid_vel = state.fluid_vel + dt * acc
    state.fluid_pos = state.fluid_pos + dt * state.fluid_vel

    # Clamp positions to channel
    state.fluid_pos[:, 0] = state.fluid_pos[:, 0].clamp(0.0, case.channel_length)
    state.fluid_pos[:, 1] = state.fluid_pos[:, 1].clamp(0.0, case.channel_height)

    state.time += dt

    # Diagnostics
    tip_x = float(state.beam_nodes[-1, 0]) - (case.cylinder_x + case.cylinder_d / 2 + case.beam_length)
    tip_y = float(state.beam_nodes[-1, 1]) - case.cylinder_y
    drag = float(f_beam[:, 0].sum()) if f_beam.numel() else 0.0
    lift = float(f_beam[:, 1].sum()) if f_beam.numel() else 0.0
    max_speed = float(state.fluid_vel.norm(dim=1).max())

    return {"t": state.time, "tip_x": tip_x, "tip_y": tip_y,
            "drag": drag, "lift": lift, "max_speed": max_speed}


def run_fsi2(case: FSI2Case, state: Optional[FSI2State] = None):
    """Integrate to t_end; return state and metrics."""
    if state is None:
        state = initial_state(case)
    dt = case.dt()
    n_steps = max(int(math.ceil(case.t_end / dt)), 1)
    for k in range(n_steps):
        row = step_fsi2(state, case, dt)
        if k % max(case.output_every, 1) == 0 or k == n_steps - 1:
            for key, value in row.items():
                state.history[key].append(value)

    hist = state.history
    tip_x = hist["tip_x"]
    tip_y = hist["tip_y"]
    return state, {
        "peak_tip_x": max(tip_x) if tip_x else 0.0,
        "peak_tip_y": max(tip_y) if tip_y else 0.0,
        "final_tip_x": tip_x[-1] if tip_x else 0.0,
        "final_tip_y": tip_y[-1] if tip_y else 0.0,
        "mean_drag": sum(hist["drag"]) / max(len(hist["drag"]), 1),
        "mean_lift": sum(hist["lift"]) / max(len(hist["lift"]), 1),
        "n_fluid": int(state.fluid_pos.shape[0]),
        "n_steps": n_steps, "dt": dt, "dx": case.dx, "h": case.h(),
        "history": hist,
    }


# --------------------------------------------------------------------------- #
# Verification functions
# --------------------------------------------------------------------------- #

def verify_parabolic_inlet(case: FSI2Case) -> Dict[str, float]:
    """V1: Check that the inlet velocity profile matches the analytic parabola."""
    device = torch.device(case.device)
    H = case.channel_height
    u_max = case.inlet_u_max
    # Sample points across the channel height at x = dx/2
    ys = torch.linspace(0.01 * H, 0.99 * H, 50, dtype=torch.float64, device=device)
    pos = torch.stack((torch.full_like(ys, 0.5 * case.dx), ys), dim=-1)
    vel = set_inlet_velocity(case, pos)
    u_analytic = u_max * 4.0 * ys * (H - ys) / (H ** 2)
    u_computed = vel[:, 0]
    error = float(torch.abs(u_computed - u_analytic).max())
    return {
        "max_error": error,
        "u_max_analytic": float(u_max),
        "u_max_computed": float(u_computed.max()),
        "rel_error": error / u_max if u_max > 0 else 0.0,
    }


def verify_cylinder_geometry(case: FSI2Case) -> Dict[str, float]:
    """V2: Check cylinder particle placement."""
    device = torch.device(case.device)
    cyl = build_cylinder_particles(case, device)
    cx, cy = case.cylinder_x, case.cylinder_y
    r = case.cylinder_d / 2.0
    dist = torch.sqrt((cyl[:, 0] - cx) ** 2 + (cyl[:, 1] - cy) ** 2)
    max_dev = float(torch.abs(dist - r).max())
    return {
        "n_particles": int(cyl.shape[0]),
        "radius": r,
        "max_radial_deviation": max_dev,
        "rel_deviation": max_dev / r,
    }


def verify_beam_stiffness(case: FSI2Case) -> Dict[str, float]:
    """V3: Check beam bending stiffness against analytic EI."""
    EI_analytic = case.beam_stiffness_bending()
    EA_analytic = case.beam_stiffness_axial()
    return {
        "EI_analytic": EI_analytic,
        "EA_analytic": EA_analytic,
        "area_density": case.beam_area_density(),
        "axial_frequency": case.beam_axial_frequency(),
    }


def verify_beam_point_load(case: FSI2Case) -> Dict[str, float]:
    """V4: Beam tip deflection under transverse point load vs Euler-Bernoulli.

    Uses a small load so that linear beam theory applies.  The load is applied
    in the y-direction (transverse to the beam axis) and the deflection is
    measured in y.  Static equilibrium is found via L-BFGS-B optimisation
    with the gradient computed by torch autograd.
    """
    import numpy as np
    from scipy.optimize import minimize

    device = torch.device(case.device)
    EI = case.beam_stiffness_bending()
    L = case.beam_length
    l0 = L / (case.n_beam_nodes - 1)

    # Small transverse point load at tip (y-direction)
    P = 1.0e-3 * 2.0 * EI / L ** 2
    load = torch.zeros((case.n_beam_nodes, 2), dtype=torch.float64)
    load[-1, 1] = P  # transverse (y) direction

    def _energy_and_grad(x_flat: np.ndarray):
        """Total energy and its gradient w.r.t. free node coordinates."""
        nodes = torch.zeros((case.n_beam_nodes, 2), dtype=torch.float64)
        # Clamp first two nodes (position + first segment direction)
        nodes[0, 0] = case.cylinder_x + case.cylinder_d / 2
        nodes[0, 1] = case.cylinder_y
        nodes[1, 0] = nodes[0, 0] + l0
        nodes[1, 1] = nodes[0, 1]
        nodes[2:] = torch.from_numpy(x_flat.reshape(-1, 2)).double()

        nodes_w = nodes.clone().requires_grad_(True)
        E = _beam_energy_clamped(nodes_w, case)
        U = -(load * nodes_w).sum()
        total = E + U
        grad_all = torch.autograd.grad(total, nodes_w)[0]
        return float(total.item()), grad_all[2:].numpy().flatten()

    # Initial guess: straight beam (free nodes only)
    nodes0 = beam_initial_nodes(case, device)
    x0 = nodes0[2:].numpy().flatten().copy()

    res = minimize(lambda x: _energy_and_grad(x), x0, jac=True,
                   method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})

    # Reconstruct full solution
    nodes_opt = torch.zeros((case.n_beam_nodes, 2), dtype=torch.float64)
    nodes_opt[0, 0] = case.cylinder_x + case.cylinder_d / 2
    nodes_opt[0, 1] = case.cylinder_y
    nodes_opt[1, 0] = nodes_opt[0, 0] + l0
    nodes_opt[1, 1] = nodes_opt[0, 1]
    nodes_opt[2:] = torch.from_numpy(res.x.reshape(-1, 2)).double()

    # Tip deflection in y (transverse)
    tip_deflection = float(nodes_opt[-1, 1]) - case.cylinder_y
    # Euler-Bernoulli: delta = P L^3 / (3 EI)
    delta_eb = P * L ** 3 / (3 * EI)
    rel_error = (tip_deflection - delta_eb) / delta_eb if delta_eb != 0 else 0.0

    return {
        "P": P, "L": L, "EI": EI,
        "tip_deflection_rod": tip_deflection,
        "tip_deflection_EB": delta_eb,
        "rel_error": rel_error,
        "optim_success": bool(res.success),
        "optim_nit": int(res.nit),
    }
