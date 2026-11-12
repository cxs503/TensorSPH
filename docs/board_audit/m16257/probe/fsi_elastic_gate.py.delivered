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
    load_scale: float = 1.0      # 1.0 physical; other values only in V2

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
    kb = case.gate_stiffness_bending() / l0
    return e_stretch + 0.5 * kb * theta.square().sum()


def rod_forces(nodes: torch.Tensor, case: ElasticGateCase,
               nodal_load: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Nodal force ``-dE/dx`` + load + gravity; the clamped node gets nothing."""
    work = nodes.detach().clone().requires_grad_(True)
    grad, = torch.autograd.grad(rod_energy(work, case), work)
    force = -grad
    if nodal_load is not None:
        force = force + nodal_load
    mass = rod_nodal_mass(case).to(nodes.dtype).to(nodes.device)
    gravity = torch.zeros_like(nodes)
    gravity[:, 1] = -case.gravity * mass
    force = force + gravity
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

    Mirrors the Adami et al. (2012) pressure term that
    :func:`tensorsph.droplet_impact.rhs` applies to wall particles::

        f_g = m_g sum_f m_f (p_f / rho_f^2 + p_g / rho_g^2) grad_g W

    with ``p_g`` the kernel-weighted pressure of the neighbouring fluid (a
    first-order extrapolation to the gate face) and ``rho_g = rho0``.  Only
    fluid/gate pairs contribute.
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

    eps = 1.0e-6 * h
    dWdr = (di.cubic_spline_kernel(r + eps, h)
            - di.cubic_spline_kernel((r - eps).clamp_min(1.0e-12), h)) / (2.0 * eps)
    grad = (dWdr / r)[:, None] * delta

    weight = di.cubic_spline_kernel(r, h)
    denom = torch.zeros(n_gate, dtype=pos.dtype, device=pos.device).index_add(
        0, wall_idx, weight)
    numer = torch.zeros(n_gate, dtype=pos.dtype, device=pos.device).index_add(
        0, wall_idx, pressure[fluid_idx] * weight)
    p_gate = numer / denom.clamp_min(1.0e-30)

    coeff = (pressure[fluid_idx] / rho[fluid_idx].square()
             + p_gate[wall_idx] / case.rho0 ** 2)
    contrib = (wall_mass[fluid_idx] * wall_mass[gate_start + wall_idx]
               * coeff)[:, None] * grad
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


def cantilever_tip_deflection(load_per_length, height: float, EI: float,
                              n_quad: int = 2001) -> float:
    """Tip deflection of a cantilever under ``q(a)`` by Green's function.

    Point load ``P`` at ``a`` gives tip deflection ``P a^2 (3H - a) / (6 EI)``;
    the distributed load is integrated with the trapezoidal rule.  Independent
    of the SPH and rod code.
    """
    a = torch.linspace(0.0, height, n_quad, dtype=torch.float64)
    q = (load_per_length(a) if callable(load_per_length)
         else torch.as_tensor(load_per_length, dtype=torch.float64))
    kernel = a.square() * (3.0 * height - a) / (6.0 * EI)
    return float(torch.trapezoid(q * kernel, a))


def verify_cantilever(case: ElasticGateCase, load_scale: float = 1.0e-5,
                      n_steps: int = 30000, damping: float = 0.995
                      ) -> Dict[str, float]:
    """V2: rod tip deflection under a small hydrostatic load vs Euler-Bernoulli.

    The physical rod (``E = 1e6 Pa``) is loaded with ``s * rho g (h_s - a)``
    where ``s = load_scale`` keeps the deflection inside the small-deflection
    range where the analytic cantilever is valid, and relaxed to static
    equilibrium with velocity damping.  Reference: the point-load Green's
    function ``a^2 (3H - a) / (6 EI)`` integrated over the loaded length, which
    shares no code with the SPH or the rod.
    """
    state = initial_state(case)
    dt = case.dt()
    nodes, vel = state.nodes, state.node_vel
    l0 = case.gate_height / (case.n_gate_nodes - 1)
    q_mag = load_scale * case.rho0 * case.gravity
    q_node = torch.zeros_like(nodes)
    q_node[:, 0] = q_mag * torch.clamp(case.water_height - nodes[:, 1], min=0.0) * l0
    for _ in range(n_steps):
        mass = rod_nodal_mass(case).to(nodes.dtype)[:, None]
        acc = rod_forces(nodes, case, q_node) / mass
        vel = damping * (vel + dt * acc)
        vel = vel.clone(); vel[0] = 0.0
        nodes = nodes + dt * vel
        nodes = nodes.clone()
        nodes[0] = torch.tensor((case.water_width, 0.0), dtype=nodes.dtype)
    got = float(nodes[-1, 0]) - case.water_width
    exact = cantilever_tip_deflection(
        lambda a: q_mag * torch.clamp(case.water_height - a, min=0.0),
        case.gate_height, case.gate_stiffness_bending())
    return {"tip_rod": got, "tip_exact": exact, "load_scale": load_scale,
            "rel_error": (got - exact) / exact if exact else float("nan")}
