"""2-D dam-break benchmark case (B2).

A rectangular column of water of width ``a`` and height ``2a`` is released
against a dry floor inside a tank. The leading-edge position is tracked as a
function of time and reported in the non-dimensional form used by the classical
dam-break experiments, ``Z/a`` against ``tau = t sqrt(2 g / a)`` (Martin &
Moyce, 1952).

Scope of the checks, stated up front:

* this module and ``benchmarks/dam_break/run.py`` measure mass conservation,
  boundedness, and **resolution convergence** of the front trajectory;
* the Ritter shallow-water solution is reported as a reference curve, but the
  case is a finite reservoir with a free-slip floor, so agreement is not
  expected and is not used as a pass criterion;
* **no digitized experimental table is bundled**, so no physical-validation
  claim is made. See ``docs/benchmarks/dam-break-2d.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .droplet_impact import (
    Config,
    apply_boundary,
    boundary_box,
    hydrostatic_density,
    resolve_device,
    rhs,
    wall_lattice,
)
from .neighbor_search import PairListCache


@dataclass
class DamBreakConfig(Config):
    """Dam-break case parameters.

    The tank occupies ``[0, tank_width] x [0, tank_height]``; the water column
    occupies ``[0, column_width] x [0, column_height]`` and is at rest at
    ``t = 0``. ``pool_width`` is reused as the tank width so the wall and
    boundary helpers apply unchanged.
    """

    dx: float = 0.01
    tank_height: float = 0.75
    column_width: float = 0.2
    column_height: float = 0.4
    pool_width: float = 2.0
    pool_depth: float = 0.75          # only used as a fallback domain height
    dt: float = 1.0e-4
    steps: int = 6000
    output_every: int = 100
    device: str = "cpu"
    hydrostatic_init: bool = True

    def tank_top(self) -> float:
        return float(self.tank_height)

    def dt_from_courant(self) -> float:
        return (self.max_courant * self.h_factor * self.dx / self.sound_speed)


def validate_case(cfg: DamBreakConfig) -> None:
    if cfg.boundary != "wall":
        raise ValueError(
            "the dam-break case integrates static wall particles only; "
            f"boundary={cfg.boundary!r} would be silently ignored")
    if cfg.dx <= 0 or cfg.h_factor <= 0:
        raise ValueError("dx and h_factor must be positive")
    if cfg.column_width <= 0 or cfg.column_height <= 0:
        raise ValueError("column dimensions must be positive")
    if cfg.column_width > cfg.pool_width or cfg.column_height > cfg.tank_height:
        raise ValueError("the column must fit inside the tank")
    if cfg.sound_speed <= 0 or cfg.rho0 <= 0 or cfg.gamma <= 0:
        raise ValueError("rho0, sound_speed, and gamma must be positive")
    courant = cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx)
    if courant > cfg.max_courant * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} gives acoustic Courant {courant:.3f} > max_courant "
            f"{cfg.max_courant:g}; use dt <= {cfg.dt_from_courant():.3e} s")


def make_column(cfg: DamBreakConfig, device: torch.device):
    """Particles filling the initial water column, resting on the floor."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, cfg.column_width, dx, device=device)
    ys = torch.arange(dx * 0.5, cfg.column_height, dx, device=device)
    if xs.numel() == 0 or ys.numel() == 0:
        raise ValueError("column discretization produced no particles")
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    if cfg.hydrostatic_init:
        mass = hydrostatic_density(positions[:, 1], cfg, surface=cfg.column_height
                                   ) * dx * dx
    else:
        mass = torch.full((positions.shape[0],), cfg.rho0 * dx * dx, device=device)
    return positions, mass


def front_position(pos: torch.Tensor, cfg: DamBreakConfig) -> float:
    """Leading-edge x coordinate of the water (the wall particles are excluded)."""
    if pos.numel() == 0:
        return 0.0
    return float(pos[:, 0].max())


def ritter_front(time: float, cfg: DamBreakConfig) -> float:
    """Ritter shallow-water leading edge on a dry bed, measured from x = 0.

    ``Z(t) = 2 sqrt(g H) t`` for a reservoir that is effectively semi-infinite.
    Reported as a reference only: the reservoir here is finite and the column is
    not shallow.
    """
    return 2.0 * math.sqrt(cfg.gravity * cfg.column_height) * time


@dataclass
class DamBreakRun:
    times: list[float]
    front: list[float]
    kinetic_energy: list[float]
    potential_energy: list[float]
    mass: float
    escaped: int
    rebuilds: int
    max_speed: float
    max_density_compression: float
    compression_before_wall_impact: float
    wall_impact_time: float | None
    initial_mass: float
    final_mass: float
    final_positions: torch.Tensor

    @property
    def mass_drift(self) -> float:
        """Difference between the initial and final total mass.

        Particle masses are constants here, so this is expected to be exactly
        zero. It is returned rather than asserted because the benchmark gate is
        supposed to measure it instead of assuming it.
        """
        return self.final_mass - self.initial_mass

    def dimensionless(self, cfg: DamBreakConfig):
        scale = cfg.column_width
        return ([t * math.sqrt(2.0 * cfg.gravity / cfg.column_width) for t in self.times],
                [z / scale for z in self.front])


def run_case(cfg: DamBreakConfig, pair_skin_factor: float | None = None):
    """Integrate the dam break and return the measured trajectory."""
    validate_case(cfg)
    device = resolve_device(cfg.device)
    top = cfg.tank_top()
    positions, mass = make_column(cfg, device)
    wall = wall_lattice(cfg, device, top=top)
    velocity = torch.zeros_like(positions)
    initial_mass = float(mass.sum())

    h = cfg.h_factor * cfg.dx
    fluid_count = positions.shape[0]
    skin = cfg.pair_skin_factor if pair_skin_factor is None else pair_skin_factor
    cache = (PairListCache(2.0 * h, skin * h, fluid_count=fluid_count)
             if skin > 0 else None)
    all_positions = torch.cat((positions, wall), dim=0)

    times: list[float] = []
    front: list[float] = []
    kinetic: list[float] = []
    potential: list[float] = []
    escaped = 0
    max_speed = 0.0
    # Compression only: rarefaction at the free surface is expected in SPH
    # density summation and is handled by the zero-pressure clamp. Both maxima
    # are accumulated every step, not only at the output steps, so the recorded
    # extremum cannot hide between two diagnostic rows.
    compression_peak = torch.zeros((), dtype=positions.dtype, device=device)
    speed_peak = torch.zeros((), dtype=positions.dtype, device=device)
    # The peak compression over the whole run is dominated by the surge hitting
    # the far wall, not by the bulk flow. Keep the pre-impact maximum separate so
    # the two effects can be told apart.
    impact_limit = cfg.pool_width - 3.0 * cfg.dx
    compression_at_impact = None
    wall_impact_time = None

    for step in range(cfg.steps + 1):
        if cache is None:
            pairs = None
        else:
            if wall is not None:
                all_positions[:fluid_count] = positions
            pairs, _ = cache.get(all_positions)
        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs, wall=wall)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        compression_peak = torch.maximum(compression_peak,
                                         (density / cfg.rho0).max() - 1.0)
        if wall_impact_time is None and float(positions[:, 0].max()) >= impact_limit:
            wall_impact_time = step * cfg.dt
            compression_at_impact = float(compression_peak)
        if velocity.numel():
            speed_peak = torch.maximum(
                speed_peak, torch.linalg.vector_norm(velocity, dim=-1).max())

        if step % cfg.output_every == 0 or step == cfg.steps:
            speed = float(torch.linalg.vector_norm(velocity, dim=-1).max())
            max_speed = max(max_speed, speed)
            times.append(step * cfg.dt)
            front.append(front_position(positions, cfg))
            kinetic.append(float(0.5 * (mass[:, None] * velocity.square()).sum()))
            potential.append(float((mass * cfg.gravity * positions[:, 1]).sum()))

        if step == cfg.steps:
            break
        velocity = velocity + cfg.dt * acc
        positions = positions + cfg.dt * velocity
        escaped += apply_boundary(positions, velocity, cfg, top=top)

    # The column starts at rest: its potential energy above the floor is the
    # energy budget available to the flow.
    return DamBreakRun(
        times=times, front=front, kinetic_energy=kinetic,
        potential_energy=potential, mass=initial_mass, escaped=escaped,
        rebuilds=0 if cache is None else cache.rebuilds,
        max_speed=max(max_speed, float(speed_peak)),
        max_density_compression=float(compression_peak),
        compression_before_wall_impact=float(
            compression_peak if compression_at_impact is None else compression_at_impact),
        wall_impact_time=wall_impact_time,
        initial_mass=initial_mass, final_mass=float(mass.sum()),
        final_positions=positions.detach().clone(),
    )


__all__ = [
    "DamBreakConfig", "DamBreakRun", "front_position", "make_column",
    "ritter_front", "run_case", "validate_case", "boundary_box",
]
