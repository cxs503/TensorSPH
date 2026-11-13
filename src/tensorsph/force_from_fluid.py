"""Force from fluid: per-boundary-particle reaction loads and the momentum budget.

Why this module exists
----------------------
Three different ways of answering "what force does the fluid exert on the
structure?" are used in this repository (and in the SPH FSI literature,
including the ``fsi2`` elastic-gate/force-from-fluid practice of
Xiangyu-Hu/SPHinXsys):

``pair`` route
    Sum, over fluid/wall kernel pairs, the reaction to the pressure (and, with
    ``wall_friction > 0``, viscous) force the pair applies to the fluid. Newton's
    third law makes this the exact discrete load; it is what a coupled solid
    actually receives. Returned by :func:`tensorsph.droplet_impact.rhs` with
    ``split=True`` as ``wall_reaction`` (one force per wall particle).
``panel`` route
    Fit the wall-face pressure (Adami, Hu & Adams 2012, eq. 11) and integrate it
    over the face, ``p_face * dx``. This is the route the elastic-gate FSI case
    uses (:func:`tensorsph.fsi_elastic_gate.gate_load`); on a vertical gate it
    reproduces the analytic hydrostatic thrust to a fraction of a percent while
    the raw ``pair`` route was measured 28 % low (docs/fsi_elastic_gate/03_report.md).
``momentum`` route
    The control-volume identity of the whole fluid body,

        d/dt sum_i m_i u_i  =  sum_i m_i g  +  F_wall->fluid,
        F_wall->fluid = -sum_wall wall_reaction,

    which needs no knowledge of the wall geometry at all. It is the arbiter:
    if the fluid's own momentum change is not accounted for by gravity plus the
    reaction it reports, then *any* fluid-side load (pair or panel route) is
    built on a bookkeeping error.

:class:`MomentumBudget` accumulates that identity step by step,
:func:`run_gate_case` drives the rigid-gate variant of the ``fsi2`` dam-break
geometry while recording all three, and :func:`hydrostatic_reference` supplies
the two closed-form numbers the static case must reproduce:

* horizontal thrust on a vertical wall of wetted height ``H``: ``0.5 rho g H^2``
  per unit width,
* total vertical reaction: the weight of the fluid, ``rho g V`` per unit width.

Both are geometry facts, not measurements, so nothing here borrows a reference
number from a paper.

What the budget identity does and does not prove
------------------------------------------------
The identity is exact *by construction* for the discrete operators as long as
(a) every fluid/fluid pair term is antisymmetric, (b) nothing else changes the
velocity (boundary clamping, particle shifting, XSPH), and (c) no momentum is
injected or removed by a post-step correction. :meth:`MomentumBudget.step`
accumulates the left-endpoint impulse ``dt * a_n``, which is precisely the
semi-implicit Euler velocity update, so a clean run closes the budget at
round-off; the measured residual is therefore a *detector* for (b) and (c) and
for a missing term in the reaction sum, not a tuning knob. Case sensitivity to
it is measured in ``benchmarks/force_from_fluid/run_budget.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import torch

from .dam_break import DamBreakConfig, make_column, validate_case
from .droplet_impact import apply_boundary, resolve_device, rhs, wall_lattice
from .neighbor_search import PairListCache

__all__ = [
    "GateCaseConfig",
    "wall_groups",
    "gate_lattice",
    "MomentumBudget",
    "hydrostatic_reference",
    "run_gate_case",
    "group_force",
    "panel_load",
    "FACE_NORMALS",
]


@dataclass
class GateCaseConfig(DamBreakConfig):
    """Dam break against a vertical rigid gate (the ``fsi2`` geometry).

    The tank is the ``DamBreakConfig`` tank; the water column is retained by a
    static gate made of wall particles standing at the right face of the column.
    ``gate_height`` defaults to ``column_height`` (the column is exactly as tall
    as the gate, as in Antoci et al. 2007 and in the SPHinXsys ``fsi2`` case).
    """

    gate_height: Optional[float] = None
    gate_layers: int = 2
    # Distance of the first gate layer from the column face, in dx.
    gate_offset: float = 0.5

    def gate_top(self) -> float:
        return (float(self.column_height) if self.gate_height is None
                else float(self.gate_height))


def gate_lattice(cfg: GateCaseConfig, device: torch.device) -> torch.Tensor:
    """Wall particles of the rigid gate: ``gate_layers`` vertical columns."""
    dx = cfg.dx
    if cfg.gate_layers < 1:
        raise ValueError("gate_layers must be positive")
    height = cfg.gate_top()
    if height <= 0.0:
        raise ValueError("gate_height must be positive")
    ys = torch.arange(dx * 0.5, height, dx, device=device)
    xs = cfg.column_width + dx * (cfg.gate_offset
                                  + torch.arange(cfg.gate_layers,
                                                 dtype=ys.dtype, device=device))
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def wall_groups(wall: torch.Tensor, cfg: GateCaseConfig,
                gate_start: int) -> Dict[str, torch.Tensor]:
    """Boolean masks over the wall particles, by physical face.

    ``gate`` is last in the wall tensor (appended after
    :func:`tensorsph.droplet_impact.wall_lattice`), so its mask is a slice; the
    remaining faces are classified geometrically with a half-spacing tolerance.
    Floor and the vertical faces overlap in the corners, so the classification
    is ordered: floor first, then the vertical faces on what is left.
    """
    dx = cfg.dx
    nw = wall.shape[0]
    gate = torch.zeros(nw, dtype=torch.bool, device=wall.device)
    gate[gate_start:] = True
    floor = (wall[:, 1] < dx * 0.5) & ~gate
    left = (wall[:, 0] < dx * 0.5) & ~gate & ~floor
    right = (wall[:, 0] > cfg.pool_width - dx * 0.5) & ~gate & ~floor
    classified = gate | floor | left | right
    other = ~classified
    # The panel route needs one area per unit face, i.e. the layer adjacent to
    # the fluid only: the second and third wall layers sit behind the first and
    # summing over them would count the same face area three times.
    gate_face = gate & (wall[:, 0] < cfg.column_width + dx * (cfg.gate_offset + 1.0))
    floor_face = floor & (wall[:, 1] > -dx)
    left_face = left & (wall[:, 0] > -dx)
    right_face = right & (wall[:, 0] < cfg.pool_width + dx)
    return {"gate": gate, "floor": floor, "left": left, "right": right,
            "other": other, "gate_face": gate_face, "floor_face": floor_face,
            "left_face": left_face, "right_face": right_face}


#: Outward normal of each wall face, i.e. the direction of the force the fluid
#: exerts on that face (the fluid sits on the side opposite to the normal).
FACE_NORMALS = {
    "gate": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "floor": (0.0, -1.0),
}


def panel_load(face_pressure: torch.Tensor, cfg: "GateCaseConfig",
               groups: Dict[str, torch.Tensor],
               normals: Optional[Dict[str, Tuple[float, ...]]] = None
               ) -> Dict[str, torch.Tensor]:
    """Surface-integral (``panel``) route: ``sum_j p_face,j * dx * n`` per face.

    ``face_pressure`` is the wall-face pressure the solver already extrapolates
    (Adami et al. 2012, eq. 11), one value per wall particle.  Each wall particle
    represents an area ``dx`` per unit width, so the force on the face is the
    pressure sum times ``dx`` along the face normal.  This is the same family as
    :func:`tensorsph.fsi_elastic_gate.gate_load`, which fits the pressure plane
    by weighted least squares instead of using eq. 11 directly.
    """
    normals = FACE_NORMALS if normals is None else normals
    out: Dict[str, torch.Tensor] = {}
    for name in normals:
        mask = groups.get(name + "_face", groups.get(name))
        if mask is None:
            continue
        normal = torch.tensor(normals[name], dtype=face_pressure.dtype,
                             device=face_pressure.device)
        total = face_pressure[mask].sum() * cfg.dx
        out[name] = total * normal
    return out


def group_force(wall_reaction: torch.Tensor,
                groups: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Sum the per-particle reaction over each group (force *on* the structure)."""
    return {name: wall_reaction[mask].sum(dim=0)
            for name, mask in groups.items()}


def hydrostatic_reference(cfg: GateCaseConfig) -> Dict[str, float]:
    """Closed-form hydrostatic loads for the initial column, per unit width.

    The column is bounded by the left wall and the gate, both wetted to
    ``column_height``; the free surface is at rest, so the wall pressure is
    ``rho g (H - y)`` and the horizontal thrust on *each* vertical face is
    ``0.5 rho g H^2``.  The whole fluid body is supported by the floor, so the
    vertical reaction is the fluid weight ``rho g A`` with ``A`` the column area.
    """
    h = float(cfg.column_height)
    area = float(cfg.column_width) * h
    return {
        "wetted_height": h,
        "column_area": area,
        "vertical_wall_thrust": 0.5 * cfg.rho0 * cfg.gravity * h * h,
        "fluid_weight": cfg.rho0 * cfg.gravity * area,
        "fluid_mass": cfg.rho0 * area,
    }


class MomentumBudget:
    """Step-by-step control-volume momentum budget of the fluid body.

    ``weight`` is the body force ``M g``.  Each :meth:`step` adds the
    left-endpoint impulse ``dt * F_wall->fluid`` (the force the walls exert on
    the fluid, ``-sum(wall_reaction)``), which is the impulse the semi-implicit
    Euler update actually applies.  After the run,

    * :attr:`momentum_change` -- measured ``P(t) - P(0)``,
    * :attr:`impulse_gravity` + :attr:`impulse_reaction` -- the conservative
      account of the same quantity,
    * :meth:`residual` -- their difference.  Round-off on a clean run; a leak
      when the velocity field is modified outside the force assembly.
    """

    def __init__(self, gravity: float, total_mass: float, velocity: torch.Tensor,
                 recirculation: Optional[Dict[str, torch.Tensor]] = None):
        self.gravity = float(gravity)
        self.total_mass = float(total_mass)
        dtype = velocity.dtype if velocity.numel() else torch.float64
        device = velocity.device
        self.weight = torch.tensor((0.0, -self.gravity * self.total_mass),
                                   dtype=dtype, device=device)
        self.dim = velocity.shape[1] if velocity.dim() > 1 else 2
        if self.dim != 2:
            self.weight = torch.zeros(self.dim, dtype=dtype, device=device)
            self.weight[-1] = -self.gravity * self.total_mass
        self.initial_momentum = torch.zeros(self.dim, dtype=dtype, device=device)
        self.momentum = self.initial_momentum.clone()
        self.impulse_gravity = torch.zeros_like(self.initial_momentum)
        self.impulse_reaction = torch.zeros_like(self.initial_momentum)
        self.time = 0.0

    # -- recording ---------------------------------------------------------
    def observe(self, velocity: torch.Tensor, mass: torch.Tensor) -> None:
        """Record the current momentum (call once per state, before stepping)."""
        self.momentum = (mass[:, None] * velocity).sum(dim=0)

    def step(self, wall_reaction: torch.Tensor, dt: float) -> None:
        """Accumulate one left-endpoint impulse from the wall reaction."""
        force_on_fluid = -wall_reaction.sum(dim=0)
        self.impulse_reaction = self.impulse_reaction + dt * force_on_fluid
        self.impulse_gravity = self.impulse_gravity + dt * self.weight
        self.time += float(dt)

    # -- results -----------------------------------------------------------
    @property
    def momentum_change(self) -> torch.Tensor:
        return self.momentum - self.initial_momentum

    @property
    def residual(self) -> torch.Tensor:
        return (self.momentum_change
                - (self.impulse_gravity + self.impulse_reaction))

    def summary(self) -> Dict[str, float]:
        change = self.momentum_change
        accounted = self.impulse_gravity + self.impulse_reaction
        scale = float(torch.linalg.vector_norm(accounted)) 
        res = self.residual
        rel = float(torch.linalg.vector_norm(res)) / scale if scale > 0 else float("inf")
        return {
            "time": self.time,
            "momentum_change_x": float(change[0]),
            "momentum_change_y": float(change[1]),
            "impulse_reaction_x": float(self.impulse_reaction[0]),
            "impulse_reaction_y": float(self.impulse_reaction[1]),
            "impulse_gravity_y": float(self.impulse_gravity[1]),
            "residual_norm": float(torch.linalg.vector_norm(res)),
            "accounted_norm": scale,
            "residual_relative": rel,
        }


def run_gate_case(cfg: GateCaseConfig, *, device: Optional[torch.device] = None,
                  record_every: int = 1, gate: bool = True) -> Dict[str, object]:
    """Integrate the rigid-gate dam break and audit the fluid-side load routes.

    The time loop is the one of :func:`tensorsph.dam_break.run_case` (fixed
    semi-implicit Euler step, explicit boundary clamp as a safety net,
    pair-list cache), extended with ``split=True`` so every step yields the
    per-wall-particle reaction.  Set ``gate=False`` to run the plain dam break
    (no gate particles) -- the budget must close just the same, which is what
    makes it a check of the solver rather than of the gate.
    """
    validate_case(cfg)
    dev = resolve_device(cfg.device) if device is None else device
    top = cfg.tank_top()
    positions, mass = make_column(cfg, dev)
    base_wall = wall_lattice(cfg, dev, top=top)
    gate_wall = gate_lattice(cfg, dev) if gate else base_wall[:0]
    wall = torch.cat((base_wall, gate_wall), dim=0) if gate_wall.numel() else base_wall
    gate_start = base_wall.shape[0] if gate_wall.numel() else wall.shape[0]
    if gate_wall.numel():
        # A gate that does not line up with the fluid lattice leaves a gap wider
        # than one spacing, the fluid/wall support collapses and the gate load
        # silently falls to ~0 (measured: 0.0003 of the analytic thrust at
        # dx = 0.03 with a 1.67 dx gap). Refuse the case instead.
        gap = float(gate_wall[:, 0].min() - positions[:, 0].max())
        if not (0.2 * cfg.dx <= gap <= 1.2 * cfg.dx):
            raise ValueError(
                f"gate face is {gap / cfg.dx:.2f} dx from the last fluid column; "
                "choose a dx that divides column_width so the gap is one "
                "spacing (the column lattice is floor based)")
    groups = wall_groups(wall, cfg, gate_start)

    velocity = torch.zeros_like(positions)
    initial_mass = float(mass.sum())
    fluid_count = positions.shape[0]
    h = cfg.h_factor * cfg.dx
    cache = (PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                           fluid_count=fluid_count)
             if cfg.pair_skin_factor > 0 else None)
    all_positions = torch.cat((positions, wall), dim=0)

    budget = MomentumBudget(cfg.gravity, initial_mass, velocity)
    budget.observe(velocity, mass)

    times: List[float] = []
    momentum: List[torch.Tensor] = []
    group_forces: Dict[str, List[float]] = {name: [] for name in groups}
    panel_forces: Dict[str, List[float]] = {name: [] for name in groups
                                            if name in FACE_NORMALS}
    reaction_forces: Dict[str, List[float]] = {name: [] for name in groups
                                               if name in FACE_NORMALS}
    front: List[float] = []
    mass_history: List[float] = []
    escaped = 0
    dt = cfg.dt
    elapsed = 0.0
    peak_compression = torch.zeros((), dtype=torch.float64, device=dev)

    def record(step: int, rho: torch.Tensor) -> None:
        obs = group_force(wall_reaction, groups)
        for name in groups:
            group_forces[name].append(obs[name].tolist())
        panel = panel_load(wall_loads["face_pressure"], cfg, groups)
        for name in panel:
            panel_forces[name].append(panel[name].tolist())
            reaction_forces[name].append(obs[name].tolist())
        times.append(elapsed)
        momentum.append((mass[:, None] * velocity).sum(dim=0).tolist())
        front.append(float(positions[:, 0].max().detach()))
        mass_history.append(float(mass.sum()))

    for step in range(cfg.steps + 1):
        if cache is not None:
            all_positions[:fluid_count] = positions
            pairs, _ = cache.get(all_positions)
        else:
            pairs = None
        rho, _, acc, wall_loads = rhs(positions, velocity, mass, cfg,
                                      pairs=pairs, wall=wall, dt=dt, split=True)
        wall_reaction = wall_loads["reaction"]
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        peak_compression = torch.maximum(peak_compression,
                                         rho.max().double() / cfg.rho0 - 1.0)
        if step % record_every == 0:
            record(step, rho)
        if step == cfg.steps:
            budget.observe(velocity, mass)
            break
        budget.observe(velocity, mass)
        budget.step(wall_reaction, dt)
        velocity = velocity + dt * acc
        positions = positions + dt * velocity
        escaped += apply_boundary(positions, velocity, cfg, top=top)
        elapsed += dt

    summary = budget.summary()
    summary.update({
        "steps": cfg.steps,
        "dt": cfg.dt,
        "particle_count": int(fluid_count),
        "wall_count": int(wall.shape[0]),
        "gate_count": int(groups["gate"].sum()) if groups["gate"].numel() else 0,
        "mass_initial": initial_mass,
        "mass_final": float(mass.sum()),
        "mass_drift_relative": (float(mass.sum()) - initial_mass) / initial_mass,
        "peak_rho_compression": float(peak_compression),
        "boundary_clamp_events": int(escaped),
        "final_front_x": front[-1] if front else float("nan"),
        "diffusion": cfg.density_diffusion,
        "wall_friction": cfg.wall_friction,
        "gate": bool(gate),
    })
    return {
        "summary": summary,
        "times": times,
        "momentum": momentum,
        "group_forces": group_forces,
        "panel_forces": panel_forces,
        "reaction_forces": reaction_forces,
        "front": front,
        "mass": mass_history,
        "budget": budget,
        "config": cfg,
        "wall": wall,
        "groups": groups,
    }