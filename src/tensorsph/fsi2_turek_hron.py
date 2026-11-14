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

Similarity warning: the benchmark is posed in SI units, so running it at a lower
inlet velocity requires the beam stiffness to be re-scaled as well.  With the
shipped default (``beam_E = 4 MPa`` at ``U_mean = 0.2 m/s``) the stiffness group
``Ae = E / (rho_f U_mean^2)`` is 1.0e5 against the benchmark's 5.6e3, i.e. the
beam is ~18x too stiff and the expected tip amplitude is ~94 % low; check with
:func:`tensorsph.fsi2_similarity.similarity_report` and use
:func:`tensorsph.fsi2_similarity.similarity_corrected_case` for a comparison
against the table above (see ``docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md``).

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
    load_transfer: str = "two_face"
    load_scale: float = 1.0

    # Loosely-coupled FSI: under-relaxation of the fluid load on the beam.
    # 1.0 = no relaxation (one fluid solve per step).  Needed when the added
    # mass is comparable to the structural mass (``rho_s/rho_f`` near 1).
    coupling_relaxation: float = 1.0

    # Acoustic-impedance (Riemann star-state) wall pressure in the load
    # transfer: p* = p_f + rho0 c0 (v_wall - v_f).n.  Without this damping the
    # loosely coupled scheme diverges at rho_s/rho_f = 1.
    wall_impedance: bool = True

    # Open boundaries (the benchmark channel is a through-flow, not a tank)
    open_boundaries: bool = True
    inlet_ramp_time: float = 1.0  # s, cosine ramp of the inlet profile
    inlet_zone: float = 2.0       # inlet velocity imposed for x < inlet_zone*dx

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

    # Remove particles in the beam region.  The clearance is the half thickness
    # plus half a particle spacing: the beam's boundary particles sit exactly on
    # the physical surface (+-t/2) and a fluid row that landed on that surface
    # would be a zero-distance neighbour (density -> 3494 kg/m3, pressure ->
    # 8e8 Pa, i.e. the step-0 blow-up recorded in
    # docs/xval_fsi2/COUPLED_RUN_DEFECTS.md).  Single precision on the lattice
    # makes the coincidence generic, not a corner case: t/2 = 0.01 m is a
    # multiple of dx for dyadic dx.
    beam_x0 = case.cylinder_x + case.cylinder_d / 2
    beam_x1 = beam_x0 + case.beam_length
    clearance = 0.5 * case.beam_thickness + 0.5 * dx + 1.0e-9
    inside_beam = ((pos[:, 0] >= beam_x0 - 0.5 * dx) & (pos[:, 0] <= beam_x1 + 0.5 * dx)
                   & ((pos[:, 1] - case.cylinder_y).abs() <= clearance))
    outside_beam = ~inside_beam

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

def interpolation_matrix_beam(n_nodes: int, n_boundary: int,
                               device, dtype=torch.float64) -> torch.Tensor:
    """Linear interpolation from beam nodes to boundary stations.

    Shape: ``(n_boundary, n_nodes)``, rows sum to 1.  Station ``k`` sits at
    ``xi_k = k (n_nodes - 1) / (n_boundary - 1)`` in *node-index* space, so the
    stations span the whole beam, node 0 to node ``n_nodes - 1``.

    .. warning::
       Until this round the parameter was normalised to ``[0, 1]`` instead of
       ``[0, n_nodes-1]``, which squeezed every boundary particle between node
       0 and node 1.  For the shipped FSI2 set-up that is 17 particles in
       12 mm instead of 350 mm -- 0.04 dx apart.  The overlapping wall
       particles then inflated the local density to 3494 kg/m3 and the
       pressure to 8.2e8 Pa on the *first* coupled step, launching the fluid at
       3777 m/s.  See ``docs/xval_fsi2/COUPLED_RUN_DEFECTS.md``.
    """
    if n_nodes < 2 or n_boundary < 2:
        raise ValueError("beam interpolation needs at least 2 nodes and 2 stations")
    idx_dtype = torch.float64
    xi = (torch.arange(n_boundary, dtype=idx_dtype, device=device)
          * (n_nodes - 1) / (n_boundary - 1))
    xi = xi.clamp(0.0, n_nodes - 1 - 1.0e-12)
    lo = xi.floor().long()
    w = (xi - lo.to(idx_dtype)).clamp(0.0, 1.0)
    mat = torch.zeros((n_boundary, n_nodes), dtype=dtype, device=device)
    rows = torch.arange(n_boundary, device=device)
    mat[rows, lo] += 1.0 - w
    mat[rows, (lo + 1).clamp(max=n_nodes - 1)] += w
    return mat


def beam_boundary_geometry(nodes: torch.Tensor, case: FSI2Case):
    """Two-sided boundary layer wrapping the elastic beam.

    The benchmark beam is a plate of thickness ``t`` (0.02 m = 4 dx at
    dx = 5 mm), not a zero-thickness line.  A single row of boundary particles
    on the beam centreline cannot represent it: both faces see that one row, so
    the two pressure loads are averaged into a single normal and the net
    transverse load collapses towards zero.  This function returns the
    boundary particles of *both* faces, offset ``t/2`` from the centreline
    along the local normal, together with the outward normal of each particle
    (pointing from the solid into the fluid).

    Returns ``(pos, normals, interp, tributary)`` with

    * ``pos``      ``(2 M, 2)``: upper row first, then the lower row,
    * ``normals``  ``(2 M, 2)``: unit outward normals,
    * ``interp``   ``(2 M, n_nodes)``: node -> boundary-station interpolation,
    * ``tributary`` scalar: arc length per boundary particle.
    """
    n = nodes.shape[0]
    n_b = max(int(math.floor(case.beam_length / case.dx)), 2)
    interp1 = interpolation_matrix_beam(n, n_b, nodes.device, dtype=nodes.dtype)

    seg = nodes[1:] - nodes[:-1]
    seg = seg / torch.linalg.vector_norm(seg, dim=1, keepdim=True).clamp_min(1.0e-12)
    tan = torch.empty_like(nodes)
    tan[0] = seg[0]
    tan[-1] = seg[-1]
    if n > 2:
        tan[1:-1] = seg[:-1] + seg[1:]
    tan = tan / torch.linalg.vector_norm(tan, dim=1, keepdim=True).clamp_min(1.0e-12)
    nrm = torch.stack((-tan[:, 1], tan[:, 0]), dim=1)

    stations = interp1 @ nodes
    st_normals = interp1 @ nrm
    st_normals = st_normals / torch.linalg.vector_norm(
        st_normals, dim=1, keepdim=True).clamp_min(1.0e-12)

    half = 0.5 * case.beam_thickness
    pos = torch.cat((stations + half * st_normals,
                     stations - half * st_normals), dim=0)
    normals = torch.cat((st_normals, -st_normals), dim=0)
    interp = torch.cat((interp1, interp1), dim=0)
    tributary = case.beam_length / (n_b - 1)
    return pos, normals, interp, tributary


def boundary_pressure_from_fluid(pos, pressure, wall, wall_start_idx, pairs, h,
                                 normals=None, vel=None, wall_vel=None,
                                 rho0=None, sound_speed=None):
    """Kernel-weighted fluid pressure extrapolated onto moving wall particles.

    ``wall`` holds the boundary particles from index ``wall_start_idx`` on.
    Returns ``(p_wall, w_sum)`` of shape ``(W_moving,)``; entries with no fluid
    neighbour inside ``2h`` get ``p_wall = 0`` and ``w_sum = 0`` so a caller can
    tell "no load" from "no data".  The kernel is the solver's own cubic-spline
    kernel (:func:`tensorsph.droplet_impact.cubic_spline_kernel`), not the
    Wendland C2 kernel used by the earlier one-sided port, which the audit in
    ``docs/xval_fsi2/PRESSURE_MOMENT_NORMALISATION.md`` showed carries a
    +24..28 % kernel-mismatch overload.
    """
    n = pos.shape[0]
    n_moving = max(wall.shape[0] - wall_start_idx, 0)
    p_out = torch.zeros((n_moving,), dtype=pos.dtype, device=pos.device)
    w_out = torch.zeros((n_moving,), dtype=pos.dtype, device=pos.device)
    if n_moving == 0 or pairs.numel() == 0:
        return p_out, w_out

    i, j = pairs.unbind(dim=1)
    crossing = (i < n) != (j < n)
    if not bool(crossing.any()):
        return p_out, w_out
    fluid_idx = torch.where(i < n, i, j)[crossing]
    wall_idx = (torch.where(i < n, j, i) - n)[crossing]
    keep = wall_idx >= wall_start_idx
    if not bool(keep.any()):
        return p_out, w_out
    fluid_idx = fluid_idx[keep]
    wall_idx = wall_idx[keep] - wall_start_idx

    delta = pos[fluid_idx] - wall[wall_start_idx + wall_idx]
    r = torch.linalg.vector_norm(delta, dim=1)
    active = (r > 1.0e-12) & (r < 2.0 * h)
    if not bool(active.any()):
        return p_out, w_out
    fluid_idx = fluid_idx[active]
    wall_idx = wall_idx[active]
    w = di.cubic_spline_kernel(r[active], h)

    p_src = pressure[fluid_idx]
    if normals is not None and vel is not None and wall_vel is not None:
        # Acoustic-impedance (Riemann star-state) wall pressure:
        #   p* = p_f - rho0 c0 (v_wall - v_f).n_solid
        # with n_solid pointing from the solid into the fluid.  A face closing
        # on the fluid (v_wall.n < v_f.n) compresses it and therefore sees a
        # *higher* pressure -- the standard linearised Riemann star state.
        # This is what the compressible fluid does at the wall; without it the
        # loosely coupled scheme has no damping at all in the added-mass limit
        # (rho_s/rho_f -> 1) and diverges, see docs/xval_fsi2/COUPLED_RUN_DEFECTS.md.
        n_w = normals[wall_idx]
        dv = wall_vel[wall_start_idx + wall_idx] - vel[fluid_idx]
        u_n = (dv * n_w).sum(dim=1)
        p_src = p_src - rho0 * sound_speed * u_n

    w_sum = torch.zeros_like(p_out).index_add(0, wall_idx, w)
    wp = torch.zeros_like(p_out).index_add(0, wall_idx, w * p_src)
    covered = w_sum > 1.0e-30
    p_out = torch.where(covered, wp / w_sum.clamp_min(1.0e-30), p_out)
    return p_out, w_sum


def boundary_force_from_pressure(p_wall, normals, tributary):
    """Traction on a boundary particle: ``F = -p n trib`` (n points into fluid)."""
    if p_wall.numel() == 0:
        return torch.zeros((0, 2), dtype=p_wall.dtype, device=p_wall.device)
    return -(p_wall[:, None] * tributary * normals)


def beam_load_on_fluid_boundary(pos, rho, pressure, wall, wall_mass,
                                 wall_start_idx, pairs, case: FSI2Case):
    """Pressure force on a single row of beam boundary particles (legacy arm).

    Kept for the audit scripts that reproduce the one-sided OSS numbers.  The
    force direction is the mean wall->fluid direction, i.e. the outward normal
    of a solid whose surface is the wall row, so the load is
    ``-p * trib * n_fluid``.  The coupled driver does **not** use this arm any
    more -- see :func:`beam_boundary_geometry`, which gives each face its own
    normal.
    """
    n = pos.shape[0]
    n_wall = max(wall.shape[0] - wall_start_idx, 0)
    out = torch.zeros((n_wall, 2), dtype=pos.dtype, device=pos.device)
    if n_wall == 0 or pairs.numel() == 0:
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

    delta = pos[fluid_idx] - wall[wall_start_idx + wall_idx]
    r = torch.linalg.vector_norm(delta, dim=1)
    active = (r > 1.0e-12) & (r < 2.0 * case.h())
    if not bool(active.any()):
        return out
    fluid_idx = fluid_idx[active]
    wall_idx = wall_idx[active]
    delta = delta[active]
    r = r[active]
    w = di.cubic_spline_kernel(r, case.h())

    unique_walls = wall_idx.unique()
    for wi in unique_walls:
        mask = wall_idx == wi
        w_loc = w[mask]
        w_sum = w_loc.sum()
        if float(w_sum) < 1.0e-30:
            continue
        p_loc = pressure[fluid_idx[mask]]
        p_wall = (w_loc * p_loc).sum() / w_sum
        n_vec = delta[mask].mean(dim=0)
        n_norm = torch.linalg.vector_norm(n_vec)
        if float(n_norm) > 1.0e-12:
            n_vec = n_vec / n_norm
        else:
            n_vec = torch.tensor([1.0, 0.0], dtype=pos.dtype, device=pos.device)
        out[wi] = -p_wall * case.dx * n_vec
    return out


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
    load: Optional[torch.Tensor] = None


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
        "drag": [], "lift": [], "max_speed": [],
        "n_fluid": [], "injected": [], "removed": [], "force_coverage": [],
    }

    return FSI2State(
        pos, vel, mass, wall_static, m_wall, cyl, m_cyl,
        beam_interp, nodes, torch.zeros_like(nodes), 0.0, history)


def _inlet_ramp(case: FSI2Case, t: float) -> float:
    """Cosine start-up ramp so the inlet does not shock the fluid at t = 0."""
    if case.inlet_ramp_time <= 0.0:
        return 1.0
    x = min(max(t / case.inlet_ramp_time, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * x))


def _apply_open_boundaries(state: "FSI2State", case: FSI2Case) -> Dict[str, int]:
    """Inflow column injection + outflow removal for the through-flow channel.

    Without this the channel is a closed tank: fluid piles up against the
    downstream end and the benchmark's through-flow (and hence its drag level)
    cannot exist.  The scheme is the standard "buffer-zone" treatment for
    particle methods:

    * particles that reach the downstream end (``x > L - dx/4``) are deleted;
    * when the inlet column has drifted one lattice spacing downstream
      (``min x > 1.5 dx``) a fresh column of fluid particles is laid at
      ``x = dx/2`` with the parabolic inlet velocity.

    Particle count therefore stays within one lattice column of its initial
    value; the exact count is logged in ``state.history``.
    """
    dx = case.dx
    injected = removed = 0

    keep = state.fluid_pos[:, 0] < case.channel_length - 0.25 * dx
    if not bool(keep.all()):
        removed = int((~keep).sum())
        state.fluid_pos = state.fluid_pos[keep]
        state.fluid_vel = state.fluid_vel[keep]
        state.fluid_mass = state.fluid_mass[keep]

    n_now = state.fluid_pos.shape[0]
    if n_now == 0:
        return {"injected": 0, "removed": removed}

    if float(state.fluid_pos[:, 0].min()) > 1.5 * dx:
        ys = torch.arange(0.5 * dx, case.channel_height, dx,
                          dtype=state.fluid_pos.dtype, device=state.fluid_pos.device)
        xs = torch.full_like(ys, 0.5 * dx)
        col = torch.stack((xs, ys), dim=-1)
        # keep the inlet column clear of the cylinder / beam footprint
        cx, cy = case.cylinder_x, case.cylinder_y
        r = 0.5 * case.cylinder_d + 0.5 * dx
        col = col[((col[:, 0] - cx) ** 2 + (col[:, 1] - cy) ** 2) > r ** 2]
        if col.shape[0]:
            vel = set_inlet_velocity(case, col) * _inlet_ramp(case, state.time)
            mass = torch.full((col.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                              dtype=state.fluid_pos.dtype, device=state.fluid_pos.device)
            state.fluid_pos = torch.cat((state.fluid_pos, col), dim=0)
            state.fluid_vel = torch.cat((state.fluid_vel, vel), dim=0)
            state.fluid_mass = torch.cat((state.fluid_mass, mass), dim=0)
            injected = int(col.shape[0])

    # imposed inlet velocity in the inlet buffer zone
    zone = case.inlet_zone * dx
    on_inlet = state.fluid_pos[:, 0] < zone
    if bool(on_inlet.any()):
        u_in = set_inlet_velocity(case, state.fluid_pos[on_inlet]) * _inlet_ramp(case, state.time)
        state.fluid_vel[on_inlet] = u_in

    return {"injected": injected, "removed": removed}


def step_fsi2(state: FSI2State, case: FSI2Case, dt: float) -> Dict[str, float]:
    """Advance the coupled FSI2 system by one step."""
    cfg = case.fluid_config()
    cfg.dt = dt

    counters = {"injected": 0, "removed": 0}
    if case.open_boundaries:
        counters = _apply_open_boundaries(state, case)

    # Beam boundary layer: one row per face, outward normals per particle
    beam_boundary, beam_normals, beam_interp, tributary = beam_boundary_geometry(
        state.beam_nodes, case)
    wall_start = state.wall_static.shape[0] + state.cylinder_particles.shape[0]

    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_boundary), dim=0)
    m_unit = di.lattice_mass_unit(cfg)
    wall_mass = torch.cat((
        state.wall_mass_static,
        state.cylinder_mass,
        torch.full((beam_boundary.shape[0],), m_unit,
                   dtype=beam_boundary.dtype, device=beam_boundary.device)
    ))

    beam_boundary_vel = beam_interp @ state.beam_vel
    wall_vel = torch.cat((
        torch.zeros_like(state.wall_static),
        torch.zeros_like(state.cylinder_particles),
        beam_boundary_vel
    ))

    all_pos = torch.cat((state.fluid_pos, wall), dim=0)
    pairs = cell_linked_pairs(all_pos, 2.0 * case.h())
    rho, pressure, acc = di.rhs(state.fluid_pos, state.fluid_vel, state.fluid_mass,
                                cfg, pairs=pairs, wall=wall, dt=dt,
                                wall_vel=wall_vel)

    # Force on the beam from the fluid: each face carries its own normal
    p_wall, w_sum = boundary_pressure_from_fluid(
        state.fluid_pos, pressure, wall, wall_start, pairs, case.h(),
        normals=beam_normals if case.wall_impedance else None,
        vel=state.fluid_vel if case.wall_impedance else None,
        wall_vel=wall_vel if case.wall_impedance else None,
        rho0=case.rho0, sound_speed=case.sound_speed)
    f_beam = boundary_force_from_pressure(p_wall, beam_normals, tributary)
    coverage = float((w_sum > 1.0e-30).to(torch.float64).mean()) if p_wall.numel() else 0.0

    raw_load = (beam_interp.T @ f_beam) * case.load_scale
    alpha = float(case.coupling_relaxation)
    if alpha >= 1.0 or state.load is None:
        nodal_load = raw_load
    else:
        nodal_load = alpha * raw_load + (1.0 - alpha) * state.load
    state.load = nodal_load

    # Beam sub-cycling
    n_sub = case.rod_substeps_effective()
    dt_rod = dt / n_sub
    nodes, node_vel = state.beam_nodes, state.beam_vel
    for _ in range(n_sub):
        nodes, node_vel = beam_advance(nodes, node_vel, nodal_load, case, dt_rod)
    state.beam_nodes, state.beam_vel = nodes, node_vel

    # Fluid update
    state.fluid_vel = state.fluid_vel + dt * acc
    state.fluid_pos = state.fluid_pos + dt * state.fluid_vel
    state.fluid_pos[:, 1] = state.fluid_pos[:, 1].clamp(0.0, case.channel_height)
    if not case.open_boundaries:
        state.fluid_pos[:, 0] = state.fluid_pos[:, 0].clamp(0.0, case.channel_length)

    state.time += dt

    tip_x = float(state.beam_nodes[-1, 0]) - (case.cylinder_x + case.cylinder_d / 2 + case.beam_length)
    tip_y = float(state.beam_nodes[-1, 1]) - case.cylinder_y
    drag = float(f_beam[:, 0].sum()) if f_beam.numel() else 0.0
    lift = float(f_beam[:, 1].sum()) if f_beam.numel() else 0.0

    return {"t": state.time, "tip_x": tip_x, "tip_y": tip_y,
            "drag": drag, "lift": lift,
            "max_speed": float(state.fluid_vel.norm(dim=1).max()),
            "n_fluid": int(state.fluid_pos.shape[0]),
            "injected": counters["injected"], "removed": counters["removed"],
            "force_coverage": coverage}


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
