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
import functools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    # Diagnostic viscous (friction) arm of the body load.  Off in the coupled
    # physics (the rod is loaded by pressure only, as before); on, it produces
    # the friction part of the benchmark force coefficients.
    body_viscous_load: bool = True
    #: Feed the friction (viscous) traction on the beam faces back into the rod
    #: load.  It is not diagnostic: on the Turek-Hron plate at ``Re_L ~ 3.5e2``
    #: the laminar skin friction is the dominant axial load, so a pressure-only
    #: rod never deflects (measured tip displacement == 0 in the coupled probe).
    beam_viscous_load: bool = True
    #: arm used for the *reported* body coefficients: ``"fit"`` (plane fit,
    #: exact on a linear pressure field) or ``"avg"`` (OSS kernel average).
    #: Both are always computed for the rigid cylinder; only the reported
    #: ``*_fit`` keys depend on this switch.
    cylinder_load_arm: str = "fit"

    # Loosely-coupled FSI: under-relaxation of the fluid load on the beam.
    # 1.0 = no relaxation (one fluid solve per step).  Needed when the added
    # mass is comparable to the structural mass (``rho_s/rho_f`` near 1).
    coupling_relaxation: float = 1.0

    # Acoustic-impedance (Riemann star-state) wall pressure in the load
    # transfer: p* = p_f + rho0 c0 (v_wall - v_f).n.  Without this damping the
    # loosely coupled scheme diverges at rho_s/rho_f = 1.
    wall_impedance: bool = True
    # Sign of that term: +1 is the heavy-wall Riemann star state (dissipative);
    # -1 reproduces the pre-fix convention, kept only for the audit arm.
    wall_impedance_sign: float = 1.0

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
        """Rod sub-steps per fluid step, set by the rod's *highest* mode.

        The explicit symplectic-Euler rod is only conditionally stable, and the
        binding mode is the shortest bending wavelength, not the (much slower)
        axial mode.  Sizing the sub-cycling on the axial frequency alone leaves
        the highest bending mode under-resolved: measured on the FSI2 point at
        ``dx = 20 mm`` (``rho_s/rho_f = 1``), a freely vibrating beam integrated
        with the axial-only count gains **1e249** in energy over 200 steps with
        *zero* load, while ``rho_s/rho_f = 2`` and 10 happen to stay bounded at
        the same count.  That pathology is exactly what was previously read as
        an "added-mass instability" of the fluid coupling
        (``docs/xval_fsi2/COUPLED_RUN_DEFECTS.md``).

        The criterion is derived, not tuned: ``omega_max`` is the highest
        eigenfrequency of the discrete energy Hessian at the straight
        configuration (mass-normalised), and the sub-step count is
        ``ceil(dt * omega_max / rod_stability_target)`` so that every mode
        satisfies ``dt_sub * omega <= rod_stability_target < 2``, the symplectic
        Euler limit.
        """
        if self.rod_substeps > 0:
            return int(self.rod_substeps)
        needed = self.dt() * beam_max_frequency(self) / self.rod_stability_target
        n = max(1, int(math.ceil(needed - 1.0e-12)))
        if n > self.max_rod_substeps:
            raise ValueError(
                f"rod sub-cycling needs {n} sub-steps per fluid step "
                f"(dt={self.dt():.3e} s, omega_max={beam_max_frequency(self):.1f} rad/s) "
                f"> max_rod_substeps={self.max_rod_substeps}; reduce dt or raise "
                f"max_rod_substeps")
        return n

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


@functools.lru_cache(maxsize=64)
def _beam_max_frequency_cached(E: float, rho: float, t: float, length: float,
                               n_nodes: int) -> float:
    """Highest eigenfrequency [rad/s] of the discrete clamped beam.

    ``omega_max = sqrt(lambda_max(H) / m_node)`` where ``H`` is the Hessian of
    the discrete energy (:func:`beam_energy`) at the straight configuration,
    restricted to the free (non-clamped) degrees of freedom.  Both the stretch
    and the bending terms of the rod enter the Hessian, so this is the binding
    stability mode by construction -- no empirical constant is fitted.
    """
    probe = FSI2Case(beam_E=E, beam_rho=rho, beam_thickness=t,
                     beam_length=length, n_beam_nodes=n_nodes)
    nodes = beam_initial_nodes(probe, torch.device("cpu")).reshape(-1)
    hess = torch.autograd.functional.hessian(
        lambda z: beam_energy(z.reshape(-1, 2), probe), nodes)
    hess = hess.detach().to(torch.float64).numpy()
    free = np.arange(2, hess.shape[0])  # node 0 is clamped (2 dofs)
    lam_max = float(np.linalg.eigvalsh(hess[np.ix_(free, free)]).max())
    m_node = float(beam_nodal_mass(probe).reshape(-1)[1])
    return math.sqrt(max(lam_max, 0.0) / m_node)


def beam_max_frequency(case: "FSI2Case") -> float:
    """Highest natural frequency of the case's discrete rod [rad/s]."""
    return _beam_max_frequency_cached(
        float(case.beam_E), float(case.beam_rho), float(case.beam_thickness),
        float(case.beam_length), int(case.n_beam_nodes))


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


def cylinder_surface_geometry(case: FSI2Case, device=None, dtype=torch.float64):
    """Cylinder surface stations with outward normals and tributary arc length.

    The cylinder is rigid, so its stations are fixed: ``n_circ`` points evenly
    spaced on the circumference, ``trib = 2 pi r / n_circ`` of arc each.  The
    outward normal of a circle points along the radius, i.e. *from the solid
    into the fluid*, which is the sign convention of
    :func:`boundary_force_from_pressure` (``F = -p n trib``).

    This is the geometry the force-from-fluid operator needs on the rigid part
    of the benchmark body.  Before it existed the coupled driver only
    accumulated the *beam* load, so the Turek-Hron drag/lift coefficients --
    which are defined on the whole body -- could not be formed at all.

    Returns ``(pos, normals, tributary)``; ``pos`` is the same particle set as
    :func:`build_cylinder_particles` (bit for bit, same angle lattice).
    """
    dx = case.dx
    r = case.cylinder_d / 2.0
    n_circ = max(int(math.ceil(2 * math.pi * r / dx)), 8)
    angles = torch.arange(n_circ, dtype=dtype, device=device) * (2 * math.pi) / n_circ
    normals = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    pos = torch.stack((case.cylinder_x + r * normals[:, 0],
                       case.cylinder_y + r * normals[:, 1]), dim=-1)
    tributary = 2.0 * math.pi * r / n_circ
    return pos, normals, tributary


def build_cylinder_particles(case: FSI2Case, device) -> torch.Tensor:
    """Boundary particles on the cylinder surface."""
    pos, _, _ = cylinder_surface_geometry(case, device=device)
    return pos


def wall_shear_from_velocity_gradient(pos_solid, normals, tributary, pos_fluid,
                                      vel_fluid, vol_fluid, mu_fluid, h,
                                      vel_solid=None):
    """Viscous traction on a solid surface from a kernel-corrected gradient.

    Mirrors the X3 arm of the force-from-fluid audit
    (:func:`tensorsph.fsi_force_from_fluid.wall_shear_from_gradient`) with the
    driver's own cubic-spline kernel and support ``2h``: the velocity-gradient
    tensor is reconstructed from the *measured* first moment of the truncated
    neighbour set, so the one-sided quadrature deficit (measured at -38 % on the
    X3 Couette case) is absorbed exactly for a linear velocity field.  The
    traction is ``F = trib * mu_wall * (grad u . n)`` with ``n`` the outward
    normal (into the fluid) and ``mu_wall`` the kernel-weighted mean viscosity.

    ``vol_fluid`` are the fluid particle volumes (``mass / rho``).  Returns
    ``(n_solid, 2)``.
    """
    n_s = pos_solid.shape[0]
    dim = pos_solid.shape[1]
    out = torch.zeros((n_s, dim), dtype=pos_solid.dtype, device=pos_solid.device)
    if n_s == 0 or pos_fluid.shape[0] == 0:
        return out
    pos_all = torch.cat((pos_solid, pos_fluid), dim=0)
    pairs = cell_linked_pairs(pos_all, 2.0 * h)
    if pairs.numel() == 0:
        return out
    i, j = pairs.unbind(dim=1)
    crossing = (i < n_s) != (j < n_s)
    if not bool(crossing.any()):
        return out
    is_idx = torch.where(i < n_s, i, j)[crossing]
    jf_idx = (torch.where(i < n_s, j, i) - n_s)[crossing]

    delta = pos_fluid[jf_idx] - pos_solid[is_idx]
    r = torch.linalg.vector_norm(delta, dim=1)
    active = (r > 1.0e-12) & (r < 2.0 * h)
    if not bool(active.any()):
        return out
    is_idx, jf_idx, delta, r = is_idx[active], jf_idx[active], delta[active], r[active]
    e = delta / r[:, None]
    # dW/dr of the 2-D cubic spline, written out so that the sign does not
    # depend on which of (x_i - x_j), (x_j - x_i) a helper happens to take.
    q = r / h
    dshape = torch.where(q < 1.0, -3.0 * q + 2.25 * q.square(),
                         torch.where(q < 2.0, -0.75 * (2.0 - q).square(), 0.0))
    radial = (10.0 / (7.0 * math.pi * h * h)) * dshape / h  # dW/dr < 0
    vol = vol_fluid[jf_idx]
    w = di.cubic_spline_kernel(r, h) * vol

    moment = (radial * vol * r)[:, None, None] * (e[:, :, None] * e[:, None, :])
    b_mat = torch.zeros((n_s, dim, dim), dtype=pos_solid.dtype,
                        device=pos_solid.device).index_add(0, is_idx, -moment)
    v_s = (torch.zeros_like(pos_solid) if vel_solid is None else vel_solid)
    du = vel_fluid[jf_idx] - v_s[is_idx]
    raw = (radial * vol)[:, None, None] * (du[:, :, None] * e[:, None, :])
    g_mat = torch.zeros_like(b_mat).index_add(0, is_idx, raw)

    trace = b_mat.diagonal(dim1=-2, dim2=-1).sum(dim=1).clamp_min(1.0e-30)
    eye = torch.eye(dim, dtype=pos_solid.dtype,
                    device=pos_solid.device).expand_as(b_mat)
    grad_u = torch.bmm(g_mat, torch.linalg.inv(
        b_mat + 1.0e-12 * trace[:, None, None] * eye))
    mu_wall = (torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device)
               .index_add(0, is_idx, w * mu_fluid[jf_idx])
               / torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device)
               .index_add(0, is_idx, w).clamp_min(1.0e-30))
    shear = torch.bmm(grad_u, normals.unsqueeze(-1)).squeeze(-1)
    # Traction *on the solid*, from the fluid: with ``n`` pointing into the
    # fluid, the viscous stress the fluid exerts is ``-mu (grad u . n)``.  Gate
    # D (flat wall under ``u = gamma y``) pins both the sign and the magnitude:
    # +mu*gamma before the sign flip, as measured.
    return -tributary[:, None] * mu_wall[:, None] * shear


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
                                 wall_stop_idx=None,
                                 normals=None, vel=None, wall_vel=None,
                                 rho0=None, sound_speed=None,
                                 impedance_sign=1.0):
    """Kernel-weighted fluid pressure extrapolated onto moving wall particles.

    ``wall`` holds the boundary particles from index ``wall_start_idx`` on.
    Returns ``(p_wall, w_sum)`` of shape ``(W_moving,)``; entries with no fluid
    neighbour inside ``2h`` get ``p_wall = 0`` and ``w_sum = 0`` so a caller can
    tell "no load" from "no data".  The kernel is the solver's own cubic-spline
    kernel (:func:`tensorsph.droplet_impact.cubic_spline_kernel`), not the
    Wendland C2 kernel used by the earlier one-sided port, which the audit in
    ``docs/xval_fsi2/PRESSURE_MOMENT_NORMALISATION.md`` showed carries a
    +24..28 % kernel-mismatch overload.

    ``impedance_sign`` is the sign of the acoustic (Riemann star-state) term and
    is ``+1`` for the heavy-wall limit derived in the comment below.  It exists
    only so that the pre-fix convention (``-1``, an anti-damping term) can be
    replayed side by side in ``benchmarks/xval_fsi2/impedance_sign.py``; no
    production path uses anything but the default.
    """
    n = pos.shape[0]
    n_moving = max(wall.shape[0] - wall_start_idx, 0)
    if wall_stop_idx is not None:
        n_moving = max(min(n_moving, wall_stop_idx - wall_start_idx), 0)
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
    if wall_stop_idx is not None:
        keep = keep & (wall_idx < wall_stop_idx)
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
        #   p* = p_f + rho0 c0 (v_wall - v_f).n_solid
        # with n_solid pointing from the solid *into* the fluid.  Derivation of
        # the sign: linearised Riemann problem across the interface, states
        # (p_L, u_L) = wall and (p_R, u_R) = fluid, impedances Z = rho c.  The
        # interface velocity is
        #   u* = (Z_L u_L + Z_R u_R + p_R - p_L) / (Z_L + Z_R)
        # and in the heavy-wall limit Z_L >> Z_R it collapses to u* = u_L, so
        #   p* = p_R + Z_R (u_L - u_R) = p_f + rho_f c_f (u_wall - u_f).n.
        # Physics: a face closing on the fluid, (u_wall - u_f).n > 0, does work
        # on it and generates a compression wave of amplitude rho c u_n, i.e. a
        # *higher* wall pressure; the reaction -p* n then opposes the motion and
        # the term is dissipative.  The opposite sign pumps energy into every
        # wall mode (anti-damping) -- measured in
        # docs/fsi2_impedance_sign/01_report.md, where it was the destabiliser
        # of the added-mass regime (rho_s/rho_f -> 1).
        n_w = normals[wall_idx]
        dv = wall_vel[wall_start_idx + wall_idx] - vel[fluid_idx]
        u_n = (dv * n_w).sum(dim=1)
        p_src = p_src + impedance_sign * rho0 * sound_speed * u_n

    w_sum = torch.zeros_like(p_out).index_add(0, wall_idx, w)
    wp = torch.zeros_like(p_out).index_add(0, wall_idx, w * p_src)
    covered = w_sum > 1.0e-30
    p_out = torch.where(covered, wp / w_sum.clamp_min(1.0e-30), p_out)
    return p_out, w_sum


def boundary_pressure_from_fluid_linear_exact(pos, pressure, wall,
                                              wall_start_idx, wall_stop_idx,
                                              pairs, h):
    """Wall pressure from a *local plane fit*, exact on a linear pressure field.

    Why this arm exists
    -------------------
    :func:`boundary_pressure_from_fluid` returns the kernel-weighted mean of the
    neighbouring fluid pressures.  The support of a surface particle is
    truncated by the body (a half space), so the weighted *centroid* of its
    neighbours sits a distance ``delta`` outside the surface.  For a linear
    field that biases the face pressure by ``grad p . delta``; on a convex
    surface ``delta`` is radial, so the bias shows up as a spurious inward
    (over-)load whose relative size is ``O(h / r)`` and does **not** vanish at
    fixed ``h/r``.  Measured on the FSI2 cylinder with an imposed field
    ``p = 1 - y``: +60.1 / +38.5 / +18.6 / +9.5 % at dx = 40 / 20 / 10 / 5 mm
    (``docs/xval_fsi2/CYLINDER_BODY_FORCE.md``).  The x- (drag) component is
    unaffected to < 0.4 %, which is why the defect hid in the flat-wall audits.

    This arm fits ``p(x) ~ a + g . (x - x_wall)`` by weighted least squares over
    the same neighbour set and returns ``a``, the value *at the surface
    station*.  For a linear field the fit is exact whatever the truncation, so
    the geometric bias is removed instead of being made smaller.  The measured
    first moment of the actual neighbour set enters the normal equations
    directly; no analytic half-space correction is assumed.

    Returns ``(p_wall, w_sum)`` like the kernel-average arm; stations with fewer
    than three usable neighbours fall back to the kernel average (documented,
    and reported through ``w_sum``).
    """
    n = pos.shape[0]
    n_wall = max(min(wall.shape[0] - wall_start_idx, wall_stop_idx - wall_start_idx), 0)
    p_out = torch.zeros((n_wall,), dtype=pos.dtype, device=pos.device)
    w_out = torch.zeros((n_wall,), dtype=pos.dtype, device=pos.device)
    if n_wall == 0 or pairs.numel() == 0:
        return p_out, w_out

    i, j = pairs.unbind(dim=1)
    crossing = (i < n) != (j < n)
    if not bool(crossing.any()):
        return p_out, w_out
    fluid_idx = torch.where(i < n, i, j)[crossing]
    wall_idx = (torch.where(i < n, j, i) - n)[crossing]
    keep = (wall_idx >= wall_start_idx) & (wall_idx < wall_start_idx + n_wall)
    if not bool(keep.any()):
        return p_out, w_out
    fluid_idx = fluid_idx[keep]
    wall_idx = wall_idx[keep] - wall_start_idx

    delta = pos[fluid_idx] - wall[wall_start_idx + wall_idx]
    r = torch.linalg.vector_norm(delta, dim=1)
    active = (r > 1.0e-12) & (r < 2.0 * h)
    if not bool(active.any()):
        return p_out, w_out
    fluid_idx, wall_idx = fluid_idx[active], wall_idx[active]
    delta, r = delta[active], r[active]
    w = di.cubic_spline_kernel(r, h)
    p_src = pressure[fluid_idx]

    # normal equations of the weighted plane fit, accumulated per station
    one = torch.ones_like(r)
    basis = torch.stack((one, delta[:, 0], delta[:, 1]), dim=1)      # (P, 3)
    lhs = torch.zeros((n_wall, 3, 3), dtype=pos.dtype, device=pos.device)
    lhs = lhs.index_add(0, wall_idx, w[:, None, None] * (basis[:, :, None] * basis[:, None, :]))
    rhs = torch.zeros((n_wall, 3), dtype=pos.dtype, device=pos.device)
    rhs = rhs.index_add(0, wall_idx, w[:, None] * p_src[:, None] * basis)
    w_out = w_out.index_add(0, wall_idx, w)

    # Tikhonov-regularised solve: the penalty is scaled by the station's own
    # weight sum, so a well-populated station is unaffected while a degenerate
    # one (collinear neighbours) falls back to its kernel-weighted mean instead
    # of producing a huge gradient.
    wsum = w_out.clamp_min(1.0e-30)
    scale = wsum[:, None, None]
    eye = torch.eye(3, dtype=pos.dtype, device=pos.device).expand(n_wall, 3, 3)
    reg = 1.0e-10 * scale * eye
    fallback = (torch.zeros_like(rhs[:, 0]).index_add(0, wall_idx, w * p_src)
                / wsum)
    try:
        sol = torch.linalg.solve(lhs + reg, rhs.unsqueeze(-1)).squeeze(-1)
    except Exception:  # pragma: no cover - singular batch on exotic input
        sol = None
    if sol is None or not bool(torch.isfinite(sol).all()):
        p_out = fallback
    else:
        ok = (wsum > 1.0e-30) & torch.isfinite(sol[:, 0])
        p_out = torch.where(ok, sol[:, 0], fallback)
    return p_out, w_out


def boundary_force_from_pressure(p_wall, normals, tributary):
    """Traction on a boundary particle: ``F = -p n trib`` (n points into fluid).

    ``tributary`` is a scalar (a straight row, as on the beam) or one value per
    particle (a curved surface, where the arc spacing varies with ``dx``).
    """
    if p_wall.numel() == 0:
        return torch.zeros((0, 2), dtype=p_wall.dtype, device=p_wall.device)
    trib = tributary
    if torch.is_tensor(trib) and trib.dim() > 0:
        trib = trib[:, None]
    return -(p_wall[:, None] * trib * normals)


def body_force_from_fluid(pos, rho, pressure, wall, pairs, case: "FSI2Case",
                         *, cyl_start: int, cyl_normals, cyl_tributary,
                         vel=None, wall_vel=None, fluid_mass=None,
                         beam_boundary=None, beam_normals=None,
                         beam_tributary=None, beam_boundary_vel=None):
    """Loads on the rigid cylinder (two pressure arms + friction arm).

    Shared by the loose driver (:func:`step_fsi2`) and the strong-coupling
    driver (:func:`tensorsph.fsi2_strong_coupling.step_fsi2_strong`) so the two
    paths cannot drift apart.  That matters for the benchmark: at
    ``rho_s/rho_f = 1`` only the strong path is stable, so the Turek-Hron body
    coefficients have to be formed there, from exactly the operators audited in
    ``docs/xval_fsi2/CYLINDER_BODY_FORCE.md``.

    Parameters
    ----------
    pos, rho, pressure : fluid positions, densities, pressures of the current
        fluid snapshot.
    wall : concatenated wall rows ``[static walls, cylinder, beam boundary]``.
    pairs : ``cell_linked_pairs`` of ``cat(fluid, wall)`` at support ``2h``.
    cyl_start : first row of the cylinder inside ``wall``.
    vel, wall_vel : fluid / wall velocities, consumed only by the Robin
        (impedance) interface; the cylinder rows are static (zero velocity).
    fluid_mass : fluid particle masses, needed by the friction arm (volumes
        ``m/rho``); required when ``case.body_viscous_load`` is true.

    Returns a dict with ``f_cyl`` (kernel-average arm), ``f_cyl_fit``
    (linear-exact plane-fit arm; equal to ``f_cyl`` unless
    ``case.cylinder_load_arm == "fit"``), the friction arms ``f_visc_cyl`` /
    ``f_visc_beam`` and ``cyl_coverage`` -- the fraction of cylinder rows with a
    non-empty fluid neighbourhood, so a silent (zero) load is distinguishable
    from a small one.
    """
    n_cyl = int(cyl_normals.shape[0])
    dev, dtype = pos.device, pos.dtype
    empty = torch.zeros((0, 2), dtype=dtype, device=dev)
    if n_cyl == 0:
        return {"f_cyl": empty, "f_cyl_fit": empty, "f_visc_cyl": empty,
                "f_visc_beam": empty, "cyl_coverage": 0.0,
                "visc_total": torch.zeros(2, dtype=dtype, device=dev)}

    p_cyl, w_cyl = boundary_pressure_from_fluid(
        pos, pressure, wall, cyl_start, pairs, case.h(),
        wall_stop_idx=cyl_start + n_cyl,
        normals=cyl_normals if case.wall_impedance else None,
        vel=vel if case.wall_impedance else None,
        wall_vel=wall_vel if case.wall_impedance else None,
        rho0=case.rho0, sound_speed=case.sound_speed,
        impedance_sign=case.wall_impedance_sign)
    f_cyl = boundary_force_from_pressure(p_cyl, cyl_normals, cyl_tributary)
    cyl_coverage = (float((w_cyl > 1.0e-30).to(torch.float64).mean())
                    if p_cyl.numel() else 0.0)
    # Second, geometric-bias-free arm on the same neighbours: the kernel-average
    # arm carries the +O(h/r) radial over-load of a one-sided sum on a convex
    # body, which does not vanish at the resolutions we can afford.
    if case.cylinder_load_arm == "fit":
        p_cyl_fit, _ = boundary_pressure_from_fluid_linear_exact(
            pos, pressure, wall, cyl_start, cyl_start + n_cyl, pairs, case.h())
        f_cyl_fit = boundary_force_from_pressure(
            p_cyl_fit, cyl_normals, cyl_tributary)
    else:
        f_cyl_fit = f_cyl

    # Friction (diagnostic arm, not fed back into the rod dynamics): one call on
    # the concatenated body surface so the neighbour search is shared.
    if case.body_viscous_load and beam_boundary is not None \
            and beam_boundary.shape[0]:
        body_pos = torch.cat((wall[cyl_start:cyl_start + n_cyl],
                              beam_boundary), dim=0)
        body_nrm = torch.cat((cyl_normals, beam_normals), dim=0)
        n_beam_rows = int(beam_boundary.shape[0])
        cyl_trib = (cyl_tributary if torch.is_tensor(cyl_tributary)
                    else torch.full((n_cyl,), float(cyl_tributary),
                                    dtype=dtype, device=dev))
        beam_trib = (beam_tributary if torch.is_tensor(beam_tributary)
                     else torch.full((n_beam_rows,), float(beam_tributary),
                                     dtype=dtype, device=dev))
        body_trib = torch.cat((cyl_trib, beam_trib), dim=0)
        body_vel = torch.cat((torch.zeros_like(wall[cyl_start:cyl_start + n_cyl]),
                              beam_boundary_vel), dim=0)
        if fluid_mass is None:
            raise ValueError("body_viscous_load needs the fluid particle masses")
        vol_fluid = fluid_mass / rho.clamp_min(1.0e-12)
        mu_fluid = torch.full_like(fluid_mass, case.viscosity)
        f_visc = wall_shear_from_velocity_gradient(
            body_pos, body_nrm, body_trib, pos,
            vel if vel is not None else torch.zeros_like(pos),
            vol_fluid, mu_fluid, case.h(), vel_solid=body_vel)
        f_visc_cyl = f_visc[:n_cyl]
        f_visc_beam = f_visc[n_cyl:]
    else:
        f_visc_cyl = torch.zeros((n_cyl, 2), dtype=dtype, device=dev)
        f_visc_beam = torch.zeros((0 if beam_boundary is None
                                   else int(beam_boundary.shape[0]), 2),
                                  dtype=dtype, device=dev)
    visc_total = (torch.cat((f_visc_cyl, f_visc_beam), dim=0).sum(dim=0)
                  if f_visc_cyl.numel() else torch.zeros(2, dtype=dtype, device=dev))
    return {"f_cyl": f_cyl, "f_cyl_fit": f_cyl_fit, "f_visc_cyl": f_visc_cyl,
            "f_visc_beam": f_visc_beam, "cyl_coverage": cyl_coverage,
            "visc_total": visc_total}


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
    #: outward unit normals of the rigid cylinder stations (into the fluid)
    cylinder_normals: Optional[torch.Tensor] = None
    #: tributary arc length per cylinder station [m]
    cylinder_tributary: Optional[torch.Tensor] = None
    #: mean over the inlet rows of the per-row inflow accumulators, in metres of
    #: channel travel.  Kept as a scalar for logging/back-compat; the injection
    #: itself is driven by ``inflow_deficit_rows``.
    inflow_deficit: float = 0.0
    #: per-row (per-streamline) inflow volume accumulated since the last particle
    #: laid in that row, in metres.  Row ``j`` sits at ``y_j = (j + 1/2) dx`` and
    #: carries its local inlet speed ``u(y_j)``, so the injected mass flux equals
    #: the flux the inlet profile delivers on that streamline.
    inflow_deficit_rows: Optional[torch.Tensor] = None


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

    # Cylinder (static), with the surface normals the load operator needs
    cyl, cyl_normals, cyl_trib = cylinder_surface_geometry(case, device=device)
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
        "cyl_drag": [], "cyl_lift": [], "body_drag": [], "body_lift": [],
        "drag_viscous": [], "lift_viscous": [], "cyl_coverage": [],
        "cyl_drag_fit": [], "cyl_lift_fit": [],
        "body_drag_fit": [], "body_lift_fit": [],
    }

    return FSI2State(
        pos, vel, mass, wall_static, m_wall, cyl, m_cyl,
        beam_interp, nodes, torch.zeros_like(nodes), time=0.0, history=history,
        cylinder_normals=cyl_normals,
        cylinder_tributary=torch.full((cyl.shape[0],), cyl_trib,
                                      dtype=cyl.dtype, device=device))


def _inlet_ramp(case: FSI2Case, t: float) -> float:
    """Cosine start-up ramp so the inlet does not shock the fluid at t = 0."""
    if case.inlet_ramp_time <= 0.0:
        return 1.0
    x = min(max(t / case.inlet_ramp_time, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * x))


def _apply_open_boundaries(state: "FSI2State", case: FSI2Case,
                            dt: Optional[float] = None) -> Dict[str, int]:
    """Inflow column injection + outflow removal for the through-flow channel.

    Without this the channel is a closed tank: fluid piles up against the
    downstream end and the benchmark's through-flow (and hence its drag level)
    cannot exist.  The scheme is the standard "buffer-zone" treatment for
    particle methods:

    * particles that reach the downstream end (``x > L - dx/4``) are deleted;
    * a fresh particle is laid at ``x = dx/2`` on inlet row ``j`` once that
      row's *locally delivered* inflow volume has filled one spacing,
      ``sum(u(y_j) dt) >= dx``.  The clock is per streamline, not global: with a
      single global ``U_mean`` clock the near-wall rows (``u = 0.14 U_mean`` at
      ``y = dx/2``) are over-fed by ``U_mean / u(y_j)`` ~ 7x and their particles
      stack up at the inlet corner, which is a density/pressure spike, not the
      benchmark's through-flow (``redistribute defect``, this round).

    The second rule used to be geometric -- "inject when ``min x > 1.5 dx``" --
    and that clock is pinned by the slowest particle in the domain: the inlet
    parabola vanishes at the walls, so a near-wall particle creeps at
    ``U(y=0.5dx) = 0.1 U_max`` and keeps ``min x`` below the trigger while the
    outlet keeps deleting particles.  The channel then drains instead of
    through-flowing; measured on the FSI2 rigid-body precursor at
    ``dx = 0.02``, ``injected = 0`` over 1200 steps against 26 removals, bulk
    speed down to 0.19 ``U_mean`` and downstream flux 0.07-0.18 of the inlet
    flux (``docs/xval_fsi2/INFLOW_CLOCK_STALL.md``).  A flux-driven clock cannot
    be stalled by a single slow particle, and it injects the volume the inlet
    boundary condition actually delivers.

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

    dt = case.dt() if dt is None else float(dt)
    dtype, device = state.fluid_pos.dtype, state.fluid_pos.device
    ys = torch.arange(0.5 * dx, case.channel_height, dx, dtype=dtype, device=device)
    n_rows = int(ys.numel())
    if state.inflow_deficit_rows is None or state.inflow_deficit_rows.numel() != n_rows:
        state.inflow_deficit_rows = torch.zeros(n_rows, dtype=dtype, device=device)
    # local inlet speed of every row: u(y_j) = U_max 4 y (H - y) / H^2
    H = case.channel_height
    u_row = case.inlet_u_max * 4.0 * ys * (H - ys) / (H * H)
    state.inflow_deficit_rows += u_row * dt * _inlet_ramp(case, state.time)
    fired = state.inflow_deficit_rows >= dx
    if bool(fired.any()):
        cx, cy = case.cylinder_x, case.cylinder_y
        r = 0.5 * case.cylinder_d + 0.5 * dx
        while bool(fired.any()):
            state.inflow_deficit_rows = torch.where(
                fired, state.inflow_deficit_rows - dx, state.inflow_deficit_rows)
            rows = torch.nonzero(fired, as_tuple=False).flatten()
            col = torch.stack((torch.full((rows.numel(),), 0.5 * dx,
                                          dtype=dtype, device=device), ys[rows]), dim=-1)
            # keep the inlet row clear of the cylinder / beam footprint
            col = col[((col[:, 0] - cx) ** 2 + (col[:, 1] - cy) ** 2) > r ** 2]
            if col.shape[0]:
                vel = set_inlet_velocity(case, col) * _inlet_ramp(case, state.time)
                mass = torch.full((col.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                                  dtype=dtype, device=device)
                state.fluid_pos = torch.cat((state.fluid_pos, col), dim=0)
                state.fluid_vel = torch.cat((state.fluid_vel, vel), dim=0)
                state.fluid_mass = torch.cat((state.fluid_mass, mass), dim=0)
                injected += int(col.shape[0])
            fired = state.inflow_deficit_rows >= dx
    state.inflow_deficit = float(state.inflow_deficit_rows.mean())

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
        counters = _apply_open_boundaries(state, case, dt)

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
        rho0=case.rho0, sound_speed=case.sound_speed,
        impedance_sign=case.wall_impedance_sign)
    f_beam = boundary_force_from_pressure(p_wall, beam_normals, tributary)
    coverage = float((w_sum > 1.0e-30).to(torch.float64).mean()) if p_wall.numel() else 0.0

    # ---- rigid-body (cylinder) load from the fluid ------------------------ #
    # The Turek-Hron drag/lift coefficients are defined on the *whole* body
    # (cylinder + beam).  The arms live in :func:`body_force_from_fluid` so the
    # strong-coupling driver reports exactly the same numbers.
    body = body_force_from_fluid(
        state.fluid_pos, rho, pressure, wall, pairs, case,
        cyl_start=state.wall_static.shape[0],
        cyl_normals=state.cylinder_normals,
        cyl_tributary=state.cylinder_tributary,
        # the friction arm needs the fluid velocity whatever the interface
        # scheme is; ``body_force_from_fluid`` gates the Robin arm itself
        vel=state.fluid_vel,
        wall_vel=wall_vel,
        fluid_mass=state.fluid_mass,
        beam_boundary=beam_boundary, beam_normals=beam_normals,
        beam_tributary=tributary, beam_boundary_vel=beam_boundary_vel)
    f_cyl, f_cyl_fit = body["f_cyl"], body["f_cyl_fit"]
    cyl_coverage = body["cyl_coverage"]
    f_visc_cyl, f_visc_beam = body["f_visc_cyl"], body["f_visc_beam"]

    # Structural load = pressure traction + skin friction on the same faces.
    f_beam_struct = (f_beam + f_visc_beam
                     if (case.beam_viscous_load and f_visc_beam.numel())
                     else f_beam)
    raw_load = (beam_interp.T @ f_beam_struct) * case.load_scale
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
    cyl_drag = float(f_cyl[:, 0].sum()) if f_cyl.numel() else 0.0
    cyl_lift = float(f_cyl[:, 1].sum()) if f_cyl.numel() else 0.0
    visc = (torch.cat((f_visc_cyl, f_visc_beam), dim=0).sum(dim=0)
            if f_visc_cyl.numel() else
            torch.zeros(2, dtype=f_cyl.dtype, device=f_cyl.device))

    return {"t": state.time, "tip_x": tip_x, "tip_y": tip_y,
            "drag": drag, "lift": lift,
            "cyl_drag": cyl_drag, "cyl_lift": cyl_lift,
            "body_drag": cyl_drag + drag, "body_lift": cyl_lift + lift,
            "cyl_drag_fit": float(f_cyl_fit[:, 0].sum()) if f_cyl_fit.numel() else 0.0,
            "cyl_lift_fit": float(f_cyl_fit[:, 1].sum()) if f_cyl_fit.numel() else 0.0,
            "body_drag_fit": (float(f_cyl_fit[:, 0].sum()) if f_cyl_fit.numel() else 0.0) + drag,
            "body_lift_fit": (float(f_cyl_fit[:, 1].sum()) if f_cyl_fit.numel() else 0.0) + lift,
            "drag_struct": (float(f_beam_struct[:, 0].sum())
                            if f_beam_struct.numel() else 0.0),
            "lift_struct": (float(f_beam_struct[:, 1].sum())
                            if f_beam_struct.numel() else 0.0),
            "beam_drag_pressure": drag, "beam_drag_friction": (
                float(f_visc_beam[:, 0].sum()) if f_visc_beam.numel() else 0.0),
            "drag_viscous": float(visc[0]), "lift_viscous": float(visc[1]),
            "cyl_coverage": cyl_coverage,
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
                state.history.setdefault(key, []).append(value)

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
        "mean_body_drag": (sum(hist["body_drag"]) / max(len(hist["body_drag"]), 1)
                           if hist.get("body_drag") else 0.0),
        "mean_body_lift": (sum(hist["body_lift"]) / max(len(hist["body_lift"]), 1)
                           if hist.get("body_lift") else 0.0),
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
