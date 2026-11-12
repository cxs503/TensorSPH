"""Elastic-gate dam-break FSI in 2-D (Antoci, Gallati & Sibilla 2007 case).

Physical problem
----------------
A rectangular water column stands on the tank floor and is retained by a thin
elastic gate clamped to the floor at its lower end and free at its upper end.
At ``t = 0`` the water is released, loads the gate, the gate bends downstream,
the water overtops the deflected gate and discharges into the dry part of the
tank.  The engineering quantity is the horizontal displacement history of the
gate tip (peak value and phase).

Reference
---------
C. Antoci, M. Gallati, S. Sibilla, "Numerical simulation of fluid-structure
interaction by SPH", *Computers & Structures* **85** (2007) 879-890,
DOI 10.1016/j.compstruc.2007.01.002.

Access status of that reference, checked while this module was written so the
next reader does not repeat it: Unpaywall reports ``is_oa = false`` with no
repository copy, Semantic Scholar reports ``openAccessPdf`` status ``CLOSED``,
OpenAlex reports ``any_repository_has_fulltext = false``.  The experimental
gate-tip curve therefore could **not** be read here; the case parameters below
carry :data:`PARAMETER_PROVENANCE`, and nothing in this module pretends to own
a reference number it does not have.

Solver
------
* fluid: the repository's weakly-compressible SPH engine
  (:mod:`tensorsph.droplet_impact`): Tait equation of state, cubic-spline
  kernel with ``h = h_factor * dx``, density by summation, Adami, Hu & Adams
  (2012) wall boundary treatment, artificial viscosity.
* solid: a discrete elastic rod with stretch and bending energy, plane stress
  per unit width, lumped nodal mass.  Nodal forces are the negative gradient of
  the discrete energy taken by automatic differentiation, so no hand-coded
  stiffness matrix can drift away from the energy it claims to represent.
* coupling: the gate is represented in the fluid by one row of *dynamic
  boundary particles* (spacing ``dx``) whose position and velocity follow the
  rod through a linear interpolation matrix ``W``.  The pressure force the
  fluid exerts on those particles (the same Adami pressure term the fluid's own
  boundary loop uses) is lumped back with ``W^T``, so the fluid->solid transfer
  conserves the total force; the solid->fluid transfer is kinematic.  Time
  stepping is explicit and staggered at the acoustic Courant step.

Verification that does not need the paper: :func:`verify_hydrostatic_load`
(total horizontal load vs ``0.5 rho g h_s^2``) and :func:`verify_cantilever`
(rod tip deflection vs the Euler-Bernoulli cantilever Green's function).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import torch

from . import droplet_impact as di
from .neighbor_search import cell_linked_pairs

PARAMETER_PROVENANCE = (
    "geometry and material data of the Antoci et al. (2007) rig as reproduced "
    "in the SPH/FSI literature; NOT read from the original article, which is "
    "closed access. Verify against the paper before quoting."
)


@dataclass
class ElasticGateCase:
    """Configuration of the elastic-gate dam-break.

    ``x`` runs along the tank (``0 <= x <= tank_length``), ``y`` up from the
    floor.  The water occupies ``0 < x < water_width``, ``0 < y < water_height``;
    the gate is the vertical line ``x = water_width``, clamped at ``y = 0`` and
    free at ``y = gate_height``.
    """

    dx: float = 0.01
    h_factor: float = 1.3
    rho0: float = 1000.0
    sound_speed: float = 20.0
    gamma: float = 7.0
    gravity: float = 9.81
    viscosity: float = 1.0e-3
    max_courant: float = 0.3
    wall_layers: int = 3

    tank_length: float = 0.584
    tank_height: float = 0.45
    water_width: float = 0.146
    water_height: float = 0.292

    gate_height: float = 0.36
    gate_thickness: float = 0.004
    gate_E: float = 1.0e6
    gate_nu: float = 0.0
    gate_rho: float = 2500.0
    n_gate_nodes: int = 25
    #: 1.0 physical.  Kept only as a diagnostic knob: the production run must
    #: not calibrate the fluid->solid transfer (m16257 audit, finding 3.3).
    load_scale: float = 1.0
    #: Fluid->solid load transfer mode: ``"panel"`` extrapolates the fluid
    #: pressure onto the gate face (default, no systematic load deficit) and
    #: ``"pairwise"`` is the legacy Adami wall-force sum, kept as the
    #: before/after baseline of the m16257 audit finding 3.2.
    load_transfer: str = "panel"
    #: ``True`` = the gate is clamped at the floor (physical rig).  ``False``
    #: reproduces the shipped rod, which only penalised the angle between
    #: consecutive segments and therefore had a **pin** at the clamp; it is kept
    #: for the before/after comparison only.
    clamped_base: bool = True

    t_end: float = 0.4
    output_every: int = 5
    device: str = "cpu"

    # -- derived --------------------------------------------------------- #
    def h(self) -> float:
        return self.h_factor * self.dx

    def dt(self) -> float:
        return self.max_courant * self.h() / self.sound_speed

    def fluid_config(self) -> "di.Config":
        return di.Config(
            dx=self.dx, h_factor=self.h_factor, rho0=self.rho0,
            sound_speed=self.sound_speed, gamma=self.gamma, gravity=self.gravity,
            viscosity=self.viscosity, dt=self.dt(), steps=1, output_every=1,
            pool_width=self.tank_length, pool_depth=self.tank_height,
            device=self.device, boundary="wall", wall_layers=self.wall_layers,
            max_courant=self.max_courant, density_calibration=True)

    def gate_stiffness_axial(self) -> float:
        """``E A`` per unit width [N]."""
        return self.gate_E * self.gate_thickness

    def gate_stiffness_bending(self) -> float:
        """``E I`` per unit width [N m^2] (plane stress)."""
        return self.gate_E * self.gate_thickness ** 3 / 12.0

    def gate_area_density(self) -> float:
        """Mass per unit area per unit width [kg/m^2]."""
        return self.gate_rho * self.gate_thickness

    def validate(self) -> None:
        if self.dx <= 0 or self.h_factor <= 0:
            raise ValueError("dx and h_factor must be positive")
        if not (0 < self.water_width < self.tank_length):
            raise ValueError("the water column must fit inside the tank")
        if self.water_height >= self.tank_height:
            raise ValueError("the water level must be below the tank lid")
        if self.gate_height >= self.tank_height:
            raise ValueError("the gate must be shorter than the tank")
        if self.water_width + 4 * self.dx >= self.tank_length:
            raise ValueError("no room downstream of the gate at this dx")
        if min(self.gate_E, self.gate_rho, self.gate_thickness, self.gate_height) <= 0:
            raise ValueError("gate material data must be positive")
        if self.n_gate_nodes < 3:
            raise ValueError("the rod needs at least three nodes")
        l0 = self.gate_height / (self.n_gate_nodes - 1)
        omega = math.sqrt(self.gate_stiffness_axial()
                          / (self.gate_area_density() * l0))
        if self.dt() * omega > 0.5:
            raise ValueError(
                f"gate axial frequency {omega:.0f} rad/s is not resolved by "
                f"dt={self.dt():.2e} s; refine the rod or the fluid lattice")


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #

def cell_centres(extent: float, dx: float, device) -> torch.Tensor:
    count = max(int(math.floor(extent / dx)), 0)
    if count == 0:
        return torch.empty(0, dtype=torch.float64, device=device)
    return (torch.arange(count, dtype=torch.float64, device=device) + 0.5) * dx


def build_fluid(case: ElasticGateCase, device):
    xs = cell_centres(case.water_width, case.dx, device)
    ys = cell_centres(case.water_height, case.dx, device)
    if xs.numel() == 0 or ys.numel() == 0:
        raise ValueError("water column resolved by no particle; reduce dx")
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                      dtype=pos.dtype, device=device)
    return pos, mass


def build_tank_walls(case: ElasticGateCase, device) -> torch.Tensor:
    """``wall_layers`` rows of static boundary particles around the tank box."""
    dx = case.dx
    xs = cell_centres(case.tank_length, dx, device)
    ys = cell_centres(case.tank_height, dx, device)
    parts: List[torch.Tensor] = []
    for k in range(1, case.wall_layers + 1):
        off = k * dx
        parts.append(torch.stack((xs, torch.full_like(xs, -off)), dim=-1))
        parts.append(torch.stack((xs, torch.full_like(xs, case.tank_height + off)),
                                 dim=-1))
        parts.append(torch.stack((torch.full_like(ys, -off), ys), dim=-1))
        parts.append(torch.stack((torch.full_like(ys, case.tank_length + off), ys),
                                 dim=-1))
    return torch.cat(parts, dim=0)


def gate_particle_arcs(case: ElasticGateCase, device) -> torch.Tensor:
    """Arc-length positions (from the clamp) of the gate boundary row."""
    count = max(int(math.floor(case.gate_height / case.dx)), 2)
    s = (torch.arange(count, dtype=torch.float64, device=device) + 0.5) * case.dx
    return s[s < case.gate_height]


def interpolation_matrix(arcs: torch.Tensor, n_nodes: int, length: float
                         ) -> torch.Tensor:
    """Linear arc-length weights, shape ``(len(arcs), n_nodes)``, rows sum to 1."""
    l0 = length / (n_nodes - 1)
    xi = (arcs / l0).clamp(0.0, n_nodes - 1 - 1.0e-12)
    idx = xi.floor().long()
    w = (xi - idx.to(xi.dtype)).clamp(0.0, 1.0)
    mat = torch.zeros((arcs.numel(), n_nodes), dtype=arcs.dtype, device=arcs.device)
    rows = torch.arange(arcs.numel(), device=arcs.device)
    mat[rows, idx] = 1.0 - w
    mat[rows, idx + 1] += w
    return mat


# --------------------------------------------------------------------------- #
# Discrete elastic rod
# --------------------------------------------------------------------------- #

def rod_nodal_mass(case: ElasticGateCase) -> torch.Tensor:
    n = case.n_gate_nodes
    l0 = case.gate_height / (n - 1)
    m = torch.full((n,), case.gate_area_density() * l0, dtype=torch.float64)
    m[0] *= 0.5
    m[-1] *= 0.5
    return m


def rod_energy(nodes: torch.Tensor, case: ElasticGateCase) -> torch.Tensor:
    """Discrete stretch + bending energy (J per unit width)."""
    n = nodes.shape[0]
    l0 = case.gate_height / (n - 1)
    seg = nodes[1:] - nodes[:-1]
    length = torch.linalg.vector_norm(seg, dim=1).clamp_min(1.0e-9)
    ks = case.gate_stiffness_axial() / l0
    e_stretch = 0.5 * ks * (length - l0).square().sum()
    tangent = seg / length[:, None]
    cross = tangent[1:, 0] * tangent[:-1, 1] - tangent[1:, 1] * tangent[:-1, 0]
    dot = (tangent[1:] * tangent[:-1]).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.atan2(cross, dot)
    # Clamped base: the *slope* of the gate at the floor is held vertical.  The
    # shipped rod only penalised the angle between consecutive segments, i.e. it
    # had a **pin** at the clamp; a pin lets the whole gate rotate rigidly about
    # the floor at zero bending cost, which is why the delivered run "folded" to
    # 354 mm (m16257 audit finding 3.3) -- the gate was a hinged flap, not a
    # clamped cantilever.  The clamp hinge is the angle between the first segment
    # and the fixed reference tangent (0, 1).
    kb = case.gate_stiffness_bending() / l0
    if case.clamped_base:
        ref = torch.zeros_like(tangent[0])
        ref[1] = 1.0
        theta0 = torch.atan2(tangent[0, 0] * ref[1] - tangent[0, 1] * ref[0],
                             (tangent[0] * ref).sum().clamp(-1.0, 1.0))
        return e_stretch + 0.5 * kb * (theta0.square() + theta.square().sum())
    return e_stretch + 0.5 * kb * theta.square().sum()


def rod_forces(nodes: torch.Tensor, case: ElasticGateCase,
               nodal_load: Optional[torch.Tensor] = None,
               gravity: bool = True) -> torch.Tensor:
    """Nodal force ``-dE/dx`` + load + gravity; the clamped node gets nothing.

    ``gravity=False`` is required by the static verification: the m16257 audit
    showed that the shipped V2 compared a *self-weight buckled* rod against a
    small-deflection Green's function (the gate's own weight is 348x the Euler
    buckling load of the rod), so its -92 % was a setup artefact.
    """
    work = nodes.detach().clone().requires_grad_(True)
    grad, = torch.autograd.grad(rod_energy(work, case), work)
    force = -grad
    if nodal_load is not None:
        force = force + nodal_load
    if gravity:
        mass = rod_nodal_mass(case).to(nodes.dtype).to(nodes.device)
        weight = torch.zeros_like(nodes)
        weight[:, 1] = -case.gravity * mass
        force = force + weight
    force = force.clone()
    force[0] = 0.0
    return force


def rod_advance(nodes: torch.Tensor, velocities: torch.Tensor,
                nodal_load: torch.Tensor, case: ElasticGateCase, dt: float):
    """One symplectic-Euler step of the rod."""
    mass = rod_nodal_mass(case).to(nodes.dtype).to(nodes.device)[:, None]
    acc = rod_forces(nodes, case, nodal_load) / mass
    velocities = velocities + dt * acc
    velocities = velocities.clone()
    velocities[0] = 0.0
    nodes = nodes + dt * velocities
    nodes = nodes.clone()
    nodes[0] = torch.tensor((case.water_width, 0.0), dtype=nodes.dtype,
                            device=nodes.device)
    return nodes, velocities


def tip_displacement(nodes: torch.Tensor, case: ElasticGateCase):
    return (float(nodes[-1, 0]) - case.water_width,
            float(nodes[-1, 1]) - case.gate_height)


# --------------------------------------------------------------------------- #
# Fluid -> solid load transfer
# --------------------------------------------------------------------------- #

def gate_load(pos, rho, pressure, wall, wall_mass, gate_start, pairs, case):
    """Pressure force on each gate boundary particle [N per unit width].

    Two transfer modes, selected by :attr:`ElasticGateCase.load_transfer`:

    ``"panel"`` (default)
        The gate face is treated as a wall panel: the fluid pressure is
        extrapolated onto the gate particle by a *first-order* weighted least
        squares fit of the neighbouring fluid pressures (a plane in 2-D) and the
        force is that pressure times the tributary length ``dx``.  Because the
        fit reproduces any linear pressure field exactly -- independently of how
        much of the kernel support is cut away by the wall -- this mode carries
        no systematic load deficit at the wet face.  This is the same
        panel/pressure-integration approach used by wall-pressure boundary
        conditions in WCSPH codes, and it is what ``V1`` measures.

    ``"pairwise"`` (legacy, kept for the before/after comparison)
        The Adami, Hu & Adams (2012) pressure term that
        :func:`tensorsph.droplet_impact.rhs` applies to wall particles::

            f_g = m_g sum_f m_f (p_f / rho_f^2 + p_g / rho_g^2) grad_g W

        It is the momentum-symmetric choice, but with fluid on one side only the
        kernel sum loses part of the load: the m16257 audit measured -20.4 % at
        ``dx = 10`` mm (this module reproduces -28.4 % at 12.5 mm, -20.4 % at
        10 mm and -8.3 % at 7.5 mm), and m16214 divided that deficit out with
        ``load_scale = 1.2558``.  It is retained here only as a baseline.
    """
    n = pos.shape[0]
    n_gate = wall.shape[0] - gate_start
    out = torch.zeros((max(n_gate, 0), 2), dtype=pos.dtype, device=pos.device)
    if n_gate <= 0 or pairs.numel() == 0:
        return out
    i, j = pairs.unbind(dim=1)
    crossing = (i < n) != (j < n)
    if not bool(crossing.any()):
        return out
    fluid_idx = torch.where(i < n, i, j)[crossing]
    wall_idx = (torch.where(i < n, j, i) - n)[crossing]
    keep = wall_idx >= gate_start
    if not bool(keep.any()):
        return out
    fluid_idx = fluid_idx[keep]
    wall_idx = wall_idx[keep] - gate_start

    h = case.h()
    delta = pos[fluid_idx] - wall[gate_start + wall_idx]
    r = torch.linalg.vector_norm(delta, dim=1)
    active = (r > 1.0e-12) & (r < 2.0 * h)
    if not bool(active.any()):
        return out
    fluid_idx, wall_idx = fluid_idx[active], wall_idx[active]
    r, delta = r[active], delta[active]
    weight = di.cubic_spline_kernel(r, h)

    if case.load_transfer == "panel":
        # Weighted linear fit p(x) = a0 + a1 (x - x_g) + a2 (y - y_g) per gate
        # particle; the value at the gate particle itself is a0, the face
        # pressure.  A 2-D fit has three unknowns, so the two normal-equation
        # rows/columns beyond the diagonal are assembled explicitly.
        vol_f = di.lattice_mass_unit(case.fluid_config()) / case.rho0
        dx_, dy_ = delta[:, 0], delta[:, 1]
        w = weight * vol_f
        s0 = torch.zeros(n_gate, dtype=pos.dtype, device=pos.device).index_add(0, wall_idx, w)
        s1 = torch.zeros_like(s0).index_add(0, wall_idx, w * dx_)
        s2 = torch.zeros_like(s0).index_add(0, wall_idx, w * dy_)
        s11 = torch.zeros_like(s0).index_add(0, wall_idx, w * dx_ * dx_)
        s12 = torch.zeros_like(s0).index_add(0, wall_idx, w * dx_ * dy_)
        s22 = torch.zeros_like(s0).index_add(0, wall_idx, w * dy_ * dy_)
        b0 = torch.zeros_like(s0).index_add(0, wall_idx, w * pressure[fluid_idx])
        b1 = torch.zeros_like(s0).index_add(0, wall_idx, w * dx_ * pressure[fluid_idx])
        b2 = torch.zeros_like(s0).index_add(0, wall_idx, w * dy_ * pressure[fluid_idx])
        # A = [[s0,s1,s2],[s1,s11,s12],[s2,s12,s22]] assembled per gate particle
        A = torch.zeros((n_gate, 3, 3), dtype=pos.dtype, device=pos.device)
        A[:, 0, 0] = s0; A[:, 0, 1] = s1; A[:, 0, 2] = s2
        A[:, 1, 0] = s1; A[:, 1, 1] = s11; A[:, 1, 2] = s12
        A[:, 2, 0] = s2; A[:, 2, 1] = s12; A[:, 2, 2] = s22
        b = torch.stack((b0, b1, b2), dim=1)
        # Tikhonov-regularised solve: the tangential direction is poorly
        # constrained when the support is cut by the wall, but it does not
        # affect a0, which is the face pressure actually used.
        trace = (s0 + s11 + s22).clamp_min(1.0e-30)
        reg = 1.0e-10 * trace
        eye = torch.eye(3, dtype=pos.dtype, device=pos.device).expand_as(A)
        sol = torch.linalg.solve(A + reg[:, None, None] * eye, b[:, :, None])[:, :, 0]
        p_face = sol[:, 0]
        has_fluid = s0 > 1.0e-12
        p_face = torch.where(has_fluid, p_face, torch.zeros_like(p_face))
        force = torch.zeros_like(out)
        force[:, 0] = p_face * case.dx
        # the wall pushes the fluid the other way; the gate is loaded in +x when
        # the water is on its upstream (x < x_gate) side
        out = out.index_add(0, torch.arange(n_gate, device=pos.device), force)
        return out

    if case.load_transfer != "pairwise":
        raise ValueError(f"unknown load_transfer {case.load_transfer!r}")

    eps = 1.0e-6 * h
    dWdr = (di.cubic_spline_kernel(r + eps, h)
            - di.cubic_spline_kernel((r - eps).clamp_min(1.0e-12), h)) / (2.0 * eps)
    grad = (dWdr / r)[:, None] * delta

    denom = torch.zeros(n_gate, dtype=pos.dtype, device=pos.device).index_add(
        0, wall_idx, weight)
    numer = torch.zeros(n_gate, dtype=pos.dtype, device=pos.device).index_add(
        0, wall_idx, pressure[fluid_idx] * weight)
    p_gate = numer / denom.clamp_min(1.0e-30)

    coeff = (pressure[fluid_idx] / rho[fluid_idx].square()
             + p_gate[wall_idx] / case.rho0 ** 2)
    # m_f must be the *fluid* particle mass; the pre-m16252 code indexed
    # ``wall_mass`` with a fluid index, which only happened to be right because
    # every lattice particle carries the same unit mass.
    m_fluid = rho[fluid_idx] * (di.lattice_mass_unit(case.fluid_config()) / case.rho0)
    contrib = (m_fluid * wall_mass[gate_start + wall_idx] * coeff)[:, None] * grad
    return out.index_add(0, wall_idx, contrib)


# --------------------------------------------------------------------------- #
# Coupled integration
# --------------------------------------------------------------------------- #

@dataclass
class CaseState:
    fluid_pos: torch.Tensor
    fluid_vel: torch.Tensor
    fluid_mass: torch.Tensor
    wall_static: torch.Tensor
    wall_mass_static: torch.Tensor
    interp: torch.Tensor
    nodes: torch.Tensor
    node_vel: torch.Tensor
    time: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=dict)


def initial_state(case: ElasticGateCase) -> CaseState:
    case.validate()
    device = torch.device(case.device)
    pos, mass = build_fluid(case, device)
    wall_static = build_tank_walls(case, device)
    m_wall = torch.full((wall_static.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                        dtype=wall_static.dtype, device=device)
    arcs = gate_particle_arcs(case, device)
    interp = interpolation_matrix(arcs, case.n_gate_nodes, case.gate_height)
    nodes = torch.stack((
        torch.full((case.n_gate_nodes,), case.water_width, dtype=pos.dtype, device=device),
        torch.linspace(0.0, case.gate_height, case.n_gate_nodes, dtype=pos.dtype,
                       device=device)), dim=-1)
    history: Dict[str, List[float]] = {"t": [], "tip_x": [], "tip_y": [],
                                       "gate_force_x": [], "max_speed": [],
                                       "fluid_ke": []}
    return CaseState(pos, torch.zeros_like(pos), mass, wall_static, m_wall, interp,
                     nodes, torch.zeros_like(nodes), 0.0, history)


def step(state: CaseState, case: ElasticGateCase, dt: float) -> Dict[str, float]:
    """Advance the coupled system by one explicit staggered step."""
    cfg = case.fluid_config()
    cfg.dt = dt
    gate_pos = state.interp @ state.nodes
    wall = torch.cat((state.wall_static, gate_pos), dim=0)
    m_unit = di.lattice_mass_unit(cfg)
    wall_mass = torch.cat((state.wall_mass_static,
                           torch.full((gate_pos.shape[0],), m_unit,
                                      dtype=gate_pos.dtype, device=gate_pos.device)))
    wall_vel = torch.cat((torch.zeros_like(state.wall_static),
                          state.interp @ state.node_vel), dim=0)
    all_pos = torch.cat((state.fluid_pos, wall), dim=0)
    pairs = cell_linked_pairs(all_pos, 2.0 * case.h())
    rho, pressure, acc = di.rhs(state.fluid_pos, state.fluid_vel, state.fluid_mass,
                                cfg, pairs=pairs, wall=wall, dt=dt,
                                wall_vel=wall_vel)
    f_gate = gate_load(state.fluid_pos, rho, pressure, wall, wall_mass,
                       state.wall_static.shape[0], pairs, case)
    nodal_load = (state.interp.T @ f_gate) * case.load_scale
    state.nodes, state.node_vel = rod_advance(state.nodes, state.node_vel,
                                              nodal_load, case, dt)

    state.fluid_vel = state.fluid_vel + dt * acc
    state.fluid_pos = state.fluid_pos + dt * state.fluid_vel
    state.fluid_pos[:, 0] = state.fluid_pos[:, 0].clamp(0.0, case.tank_length)
    state.fluid_pos[:, 1] = state.fluid_pos[:, 1].clamp(0.0, case.tank_height)
    state.time += dt

    tip_x, tip_y = tip_displacement(state.nodes, case)
    ke = 0.5 * float((state.fluid_mass * state.fluid_vel.square().sum(dim=1)).sum())
    return {"t": state.time, "tip_x": tip_x, "tip_y": tip_y,
            "gate_force_x": float(f_gate[:, 0].sum()) if f_gate.numel() else 0.0,
            "max_speed": float(state.fluid_vel.norm(dim=1).max()),
            "fluid_ke": ke}


def run(case: ElasticGateCase, state: Optional[CaseState] = None):
    """Integrate to ``case.t_end``; return the state and peak metrics."""
    if state is None:
        state = initial_state(case)
    dt = case.dt()
    n_steps = max(int(math.ceil(case.t_end / dt)), 1)
    for k in range(n_steps):
        row = step(state, case, dt)
        if k % max(case.output_every, 1) == 0 or k == n_steps - 1:
            for key, value in row.items():
                state.history[key].append(value)
    hist = state.history
    tip = hist["tip_x"]
    peak = max(tip) if tip else 0.0
    peak_time = hist["t"][tip.index(peak)] if tip else 0.0
    return state, {
        "peak_tip_x": peak, "peak_time": peak_time,
        "final_tip_x": tip[-1] if tip else 0.0,
        "min_tip_x": min(tip) if tip else 0.0,
        "n_fluid": int(state.fluid_pos.shape[0]), "n_steps": n_steps,
        "dt": dt, "dx": case.dx, "h": case.h(), "history": hist,
    }


# --------------------------------------------------------------------------- #
# Verification ladder that does not need the paper
# --------------------------------------------------------------------------- #

def verify_hydrostatic_load(case: ElasticGateCase) -> Dict[str, float]:
    """V1: SPH-path load on the gate vs the analytic ``0.5 rho g h_s^2``.

    The fluid is put at rest with the analytic hydrostatic pressure and the
    transfer of :func:`gate_load` is exercised exactly as in the coupled run, so
    the check measures the lattice/kernel geometry of the load transfer, not
    the pressure field.
    """
    state = initial_state(case)
    pos = state.fluid_pos
    rho = torch.full((pos.shape[0],), case.rho0, dtype=pos.dtype, device=pos.device)
    p = (case.rho0 * case.gravity * (case.water_height - pos[:, 1])).clamp_min(0.0)
    wall = torch.cat((state.wall_static, state.interp @ state.nodes), dim=0)
    wall_mass = torch.cat((state.wall_mass_static,
                           torch.full((wall.shape[0] - state.wall_static.shape[0],),
                                      di.lattice_mass_unit(case.fluid_config()),
                                      dtype=pos.dtype, device=pos.device)))
    pairs = cell_linked_pairs(torch.cat((pos, wall), dim=0), 2.0 * case.h())
    f = gate_load(pos, rho, p, wall, wall_mass, state.wall_static.shape[0], pairs, case)
    got = float(f[:, 0].sum())
    exact = 0.5 * case.rho0 * case.gravity * case.water_height ** 2
    return {"load_sph": got, "load_exact": exact,
            "rel_error": (got - exact) / exact}


def eb_tip_distributed(load_per_length, height: float, EI: float,
                       n_quad: int = 2001) -> float:
    """Euler-Bernoulli cantilever tip deflection under ``q(a)`` by Green's function.

    Point load ``P`` at ``a`` gives tip deflection ``P a^2 (3H - a) / (6 EI)``;
    the distributed load is integrated with the trapezoidal rule.  Independent
    of the SPH and rod code (small-deflection linear beam theory).
    """
    a = torch.linspace(0.0, height, n_quad, dtype=torch.float64)
    q = (load_per_length(a) if callable(load_per_length)
         else torch.as_tensor(load_per_length, dtype=torch.float64))
    kernel = a.square() * (3.0 * height - a) / (6.0 * EI)
    return float(torch.trapezoid(q * kernel, a))


def eb_tip_point_load(load: float, height: float, EI: float) -> float:
    """``P H^3 / (3 EI)`` -- tip deflection of a cantilever under a tip load."""
    return load * height ** 3 / (3.0 * EI)


def elastica_tip(load: float, height: float, EI: float) -> Dict[str, float]:
    """Large-deflection cantilever under a *tip* load: exact elastica reference.

    Independent of the discrete rod: the Kirchhoff elastica is integrated as an
    ODE in arc length with ``theta`` measured from the initial (undeflected)
    axis of the cantilever::

        EI dtheta/ds = M(s) = P * (x_tip - x(s)),
        dx/ds = cos(theta),  dy/ds = sin(theta),

    with ``theta(0) = 0``.  ``x_tip`` is the shooting unknown; the free-end
    condition ``M(L) = 0`` requires ``x(L) = x_tip``.  This is the classical
    Bisshopp & Drucker (1945) problem, solved here by quadrature instead of
    tabulated values, so no reference number is transcribed.
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def integrate(x_tip):
        def rhs(_s, y):
            theta, x, _yy = y
            return [load * (x_tip - x) / EI, np.cos(theta), np.sin(theta)]
        sol = solve_ivp(rhs, (0.0, height), [0.0, 0.0, 0.0], rtol=1e-11,
                        atol=1e-13, dense_output=True)
        return sol.y[1, -1], sol.y[2, -1], sol.y[0, -1]

    def miss(x_tip):
        x_end, _y, _t = integrate(x_tip)
        return x_end - x_tip

    lo, hi = 0.0, 50.0 * height
    # bracket the root: the tip x must sit between the undeflected and the
    # fully folded (quarter-circle) configurations
    while miss(hi) > 0.0 and hi < 1.0e3 * height:
        hi *= 2.0
    x_tip = brentq(miss, lo, hi, xtol=1e-13)
    x_end, y_end, theta_end = integrate(x_tip)
    return {"tip_x": x_end, "tip_y": y_end, "tip_angle": theta_end,
            "x_tip_self_consistent": float(miss(x_tip))}


def rod_static_equilibrium(case: ElasticGateCase, nodal_load: Optional[torch.Tensor],
                           gravity: bool = False, gtol: float = 1.0e-12
                           ) -> torch.Tensor:
    """Static equilibrium of the rod as a *minimisation* of the discrete energy.

    The clamped node is frozen and the free coordinates are relaxed with
    L-BFGS-B (``scipy.optimize``) on

        Pi = E_rod - load . x (+ m g y),

    both non-dimensionalised by ``H`` and by ``EI`` so the optimiser sees
    O(1) numbers.  The gradient is the same automatic-differentiation gradient
    :func:`rod_forces` uses, so the equilibrium is the minimum of exactly the
    energy the rod integrates.

    This replaces the shipped V2, which time-marched with a fixed ``0.995``
    damping: the m16257 audit showed that schedule had not converged at all
    (changing only the damping flipped the sign of the deflection), so its
    -92 % was a setup artefact rather than a property of the rod.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = case.n_gate_nodes
    H = case.gate_height
    scale = case.gate_stiffness_bending()      # EI  [N m]
    z0 = torch.tensor(nodes_initial(case), dtype=torch.float64) / H

    def expand(flat):
        z = z0.clone()
        z[1:] = torch.as_tensor(np.asarray(flat).reshape(n - 1, 2), dtype=torch.float64)
        return z * H

    def objective(flat):
        nodes = expand(flat).requires_grad_(True)
        energy = rod_energy(nodes, case) / scale
        if nodal_load is not None:
            energy = energy - (nodal_load * nodes).sum() / scale
        if gravity:
            mass = rod_nodal_mass(case).to(torch.float64)
            energy = energy + (mass * case.gravity * nodes[:, 1]).sum() / scale
        grad, = torch.autograd.grad(energy, nodes)
        return float(energy), (grad[1:] * (H / scale)).flatten().numpy()

    result = minimize(objective, z0[1:].flatten().numpy(), jac=True, method="L-BFGS-B",
                      options={"gtol": gtol, "ftol": 0.0, "maxiter": 20000,
                               "maxfun": 60000, "maxcor": 40})
    nodes = expand(result.x)
    nodes[0] = torch.tensor((case.water_width, 0.0), dtype=torch.float64)
    return nodes


def nodes_initial(case: ElasticGateCase):
    return [[case.water_width, case.gate_height * k / (case.n_gate_nodes - 1)]
            for k in range(case.n_gate_nodes)]


def rod_nodal_load_from_line(case: ElasticGateCase, q_per_length) -> torch.Tensor:
    """Nodal load equivalent of a line load ``q_per_length(a)`` [N/m]."""
    l0 = case.gate_height / (case.n_gate_nodes - 1)
    a = torch.linspace(0.0, case.gate_height, case.n_gate_nodes, dtype=torch.float64)
    q = (q_per_length(a) if callable(q_per_length)
         else torch.as_tensor(q_per_length, dtype=torch.float64).expand_as(a))
    load = torch.zeros((case.n_gate_nodes, 2), dtype=torch.float64)
    load[:, 0] = q * l0
    load[0] *= 0.5
    load[-1] *= 0.5
    return load


def verify_cantilever(case: ElasticGateCase, stiff_E: float = 1.0e7
                      ) -> Dict[str, object]:
    """V2/V3: the rod against *two* independent beam references.

    A0  small deflection, tip point load, ``P H^3 / (3 EI)``;
    A1  small deflection, hydrostatic-like distributed load, Green's function
        ``int q(a) a^2 (3H-a)/(6EI) da``;
    A2  **large** deflection (``P L^2 / EI = 10``), the Kirchhoff elastica solved
        as an ODE -- this is the regime the real gate lives in, and the regime
        the shipped verification never tested.  The rod stands along ``+y`` and
        is loaded along ``+x`` while :func:`elastica_tip` runs along its own
        ``+x`` with a transverse load along ``+y``; the two configurations are
        congruent, so the *transverse* tip deflection must be compared with
        ``tip_y_elastica`` and the *axial* shortening with ``H - tip_x``.  The
        delivered A2 (m16214-m16352) paired the transverse deflection with the
        elastica's axial coordinate and reported the mismatch as an +86.8 %
        "rod accuracy gap"; m16454 fixed the frame and the rod is inside 2.6 %
        transverse / 3.7 % axial / 0.6 % slope at ``n = 25``.

    All three run with ``gravity = 0`` and by energy minimisation, so the check
    compares constitutive/geometry code against beam theory only.
    """
    EI = case.gate_stiffness_bending()
    H = case.gate_height
    out: Dict[str, object] = {"EI": EI, "EA": case.gate_stiffness_axial(),
                              "rho_A": case.gate_area_density()}

    def _run(load_nodes, gravity=False):
        nodes = rod_static_equilibrium(case, load_nodes, gravity=gravity)
        return float(nodes[-1, 0]) - case.water_width, float(nodes[-1, 1]) - H

    # A0: tip point load, small deflection (P H^2 / (2 EI) = 1e-3)
    P = 1.0e-3 * 2.0 * EI / H ** 2
    load = torch.zeros((case.n_gate_nodes, 2), dtype=torch.float64)
    load[-1, 0] = P
    got, _ = _run(load)
    exact = eb_tip_point_load(P, H, EI)
    out["A0_point_load"] = {"P": P, "tip_rod": got, "tip_exact": exact,
                            "rel_error": (got - exact) / exact}

    # A1: distributed hydrostatic load, small deflection (scaled by 1e-6)
    scale = 1.0e-6
    q = lambda a: scale * case.rho0 * case.gravity * torch.clamp(
        torch.as_tensor(case.water_height, dtype=torch.float64) - a, min=0.0)
    got, _ = _run(rod_nodal_load_from_line(case, q))
    exact = eb_tip_distributed(q, H, EI)
    out["A1_distributed"] = {"load_scale": scale, "tip_rod": got,
                             "tip_exact": exact, "rel_error": (got - exact) / exact}

    # A2: large deflection against the elastica, *in the same frame*
    P = 10.0 * EI / H ** 2          # P H^2 / EI = 10
    load = torch.zeros((case.n_gate_nodes, 2), dtype=torch.float64)
    load[-1, 0] = P
    nodes = rod_static_equilibrium(case, load, gravity=False)
    tip_x = float(nodes[-1, 0]) - case.water_width       # transverse deflection
    tip_y = float(nodes[-1, 1]) - H                      # -(axial shortening)
    seg = nodes[-1] - nodes[-2]
    slope_rod = float(math.atan2(float(seg[0]), float(seg[1])))   # from +y axis
    ref = elastica_tip(P, H, EI)
    trans_ref = float(ref["tip_y"])
    axial_ref = H - float(ref["tip_x"])
    slope_ref = float(ref["tip_angle"])
    out["A2_elastica"] = {
        "P": P, "P_L2_over_EI": P * H ** 2 / EI,
        "frame": ("rod stands along +y and is loaded along +x; elastica_tip "
                  "runs along its own +x with a transverse load along +y, so "
                  "the rod's transverse tip deflection maps to tip_y_elastica "
                  "and its axial shortening (-dy) to H - tip_x_elastica"),
        "tip_transverse_rod": tip_x, "tip_transverse_elastica": trans_ref,
        "tip_axial_shortening_rod": -tip_y,
        "tip_axial_shortening_elastica": axial_ref,
        "tip_slope_rod": slope_rod, "tip_slope_elastica": slope_ref,
        # the primary case metric: the gate-tip deflection
        "rel_error": (tip_x - trans_ref) / trans_ref,
        "axial_rel_error": (-tip_y - axial_ref) / axial_ref,
        "slope_rel_error": (slope_rod - slope_ref) / slope_ref,
        # historical diagnostic (m16214..m16352): the delivered A2 compared the
        # rod's *transverse* deflection with the elastica's *axial* coordinate
        # (x_tip_rod - x_tip_elastica) and read that mismatch as an +86.8 %
        # rod-accuracy gap.  Kept so the before/after stays auditable; it is
        # never a pass/fail gate here.
        "frame_mismatch_rel_error": (tip_x - float(ref["tip_x"]))
                                    / float(ref["tip_x"]),
    }
    return out


def verify_gate_load_path(case: ElasticGateCase, stiff_E: float = 1.0e7
                          ) -> Dict[str, float]:
    """V4 end-to-end: panel transfer + ``W^T`` lumping + rod vs beam theory.

    The fluid is frozen at its hydrostatic pressure, the gate is made stiff so
    that beam theory applies, and the *whole* fluid->solid path of the coupled
    run (:func:`gate_load` -> ``interp^T`` -> :func:`rod_static_equilibrium`) is
    driven to equilibrium.  Reference: the Euler-Bernoulli tip deflection of a
    cantilever under ``p(a) = rho g (h_s - a)``.
    """
    stiff = ElasticGateCase(**{**case.__dict__, "gate_E": stiff_E})
    state = initial_state(stiff)
    pos = state.fluid_pos
    rho = torch.full((pos.shape[0],), stiff.rho0, dtype=pos.dtype, device=pos.device)
    p = (stiff.rho0 * stiff.gravity * (stiff.water_height - pos[:, 1])).clamp_min(0.0)
    gate = state.interp @ state.nodes
    wall = torch.cat((state.wall_static, gate), dim=0)
    wall_mass = torch.cat((
        state.wall_mass_static,
        torch.full((gate.shape[0],), di.lattice_mass_unit(stiff.fluid_config()),
                   dtype=pos.dtype, device=pos.device)))
    pairs = cell_linked_pairs(torch.cat((pos, wall), dim=0), 2.0 * stiff.h())
    f_gate = gate_load(pos, rho, p, wall, wall_mass, state.wall_static.shape[0],
                       pairs, stiff)
    nodal = state.interp.T @ f_gate
    nodes = rod_static_equilibrium(stiff, nodal, gravity=False)
    got = float(nodes[-1, 0]) - stiff.water_width
    EI = stiff.gate_stiffness_bending()
    exact = eb_tip_distributed(
        lambda a: stiff.rho0 * stiff.gravity * torch.clamp(
            torch.as_tensor(stiff.water_height, dtype=torch.float64) - a, min=0.0),
        stiff.gate_height, EI)
    return {"total_load": float(f_gate[:, 0].sum()), "tip_rod": got,
            "tip_exact": exact, "rel_error": (got - exact) / exact,
            "stiff_E": stiff_E, "gate_E": stiff_E}
