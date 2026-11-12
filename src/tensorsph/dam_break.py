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

from dataclasses import dataclass, field
import math

import torch

from .particle_shifting import shift_positions
from .droplet_impact import (
    Config,
    apply_boundary,
    boundary_box,
    cell_centres,
    hydrostatic_density,
    lattice_mass_unit,
    next_dt,
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
    # Particle shifting (Lind, Xu, Stansby & Rogers 2012; implemented in
    # ``tensorsph.particle_shifting``). Every ``shift_every`` steps the fluid
    # positions are relaxed down ``grad C`` with the free-surface shift projected
    # onto the surface tangent. 0 disables it, so every published B2 number is
    # bit-identical to the pre-shifting solver.
    shift_every: int = 0
    shift_strength: float = 1.0
    shift_limit: float = 0.25
    # Reference speed of the shift coefficient. None = max(current speed,
    # 0.05 c0): a resting column has v_max -> 0, and a shift that vanishes at
    # rest cannot relax the lattice, which is precisely defect F1
    # (docs/verification_startup_quiescence.md).
    shift_v_ref: float | None = None

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
    # Floor-based cell count (see droplet_impact.cell_centres): the last row
    # and column stay a full spacing from the tank walls even when
    # column_width/dx or column_height/dx is not an integer.  For integer
    # ratios the lattice is identical to the historical arange layout, so all
    # published benchmark numbers are untouched.
    xs = cell_centres(cfg.column_width, dx, device)
    ys = cell_centres(cfg.column_height, dx, device)
    if xs.numel() == 0 or ys.numel() == 0:
        raise ValueError("column discretization produced no particles")
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    # Lattice mass unit, calibrated unless ``Config.density_calibration`` is
    # off (droplet_impact.lattice_mass_unit).  The flag was documented on
    # Config but this case ignored it: the column started with the raw
    # ``rho0 dx^2``, i.e. with the lattice kernel-sum bias ``b`` of the chosen
    # h/dx baked in as a spurious pressure.  See
    # docs/verification_case_lattice_mass_calibration.md.
    mass_unit = lattice_mass_unit(cfg)
    if cfg.hydrostatic_init:
        mass = hydrostatic_density(positions[:, 1], cfg, surface=cfg.column_height
                                   ) * (mass_unit / cfg.rho0)
    else:
        mass = torch.full((positions.shape[0],), mass_unit, device=device)
    return positions, mass


def front_position(pos: torch.Tensor, cfg: DamBreakConfig) -> float:
    """Leading-edge x coordinate of the water (the wall particles are excluded)."""
    if pos.numel() == 0:
        return 0.0
    return float(pos[:, 0].max())


def ritter_front(time: float, cfg: DamBreakConfig) -> float:
    """Ritter shallow-water leading edge on a dry bed, measured from the dam.

    ``Z(t) = 2 sqrt(g H) t`` for a reservoir that is effectively semi-infinite.
    Reported as a reference only: the reservoir here is finite and the column is
    not shallow.

    **Frame reminder.** This is ``Z`` in the self-similar frame, i.e. the
    distance travelled *from the initial column face*. The solver's
    :func:`front_position` reports the lab-frame coordinate ``x_max >= a``, so
    the two are only comparable as ``front_position - column_width`` against
    ``ritter_front``. Comparing ``front_position`` itself against this function
    is a frame error worth one column width in every sample.

    Non-dimensionalised with ``tau = t sqrt(2 g / a)`` the same reference is
    :func:`ritter_front_over_a`.
    """
    return 2.0 * math.sqrt(cfg.gravity * cfg.column_height) * time


def ritter_height_profile(x: float, time: float,
                          cfg: DamBreakConfig) -> float | None:
    """Exact Ritter free surface ``h(x, t)`` on a dry bed, ``x`` from the dam.

    With ``c = sqrt(g H)`` the solution on ``-c t <= x <= 2 c t`` is the
    rarefaction ``h = (2c - x/t)^2 / (9 g)``, ``u = (2/3)(c + x/t)``; upstream of
    the rarefaction tail the reservoir is still at ``H`` and downstream of the
    front it is dry. ``None`` is returned before release.

    Landmarks (pinned in ``tests/test_ritter_reference.py``): ``h(-c t) = H``,
    ``h(0, t) = 4H/9`` (the dam does *not* keep its depth), ``h(2 c t) = 0``, and
    the Riemann invariant ``u + 2 sqrt(gh) = 2c`` holds across the whole fan.
    """
    if time <= 0.0:
        return None
    c = math.sqrt(cfg.gravity * cfg.column_height)
    if x < -c * time:
        return cfg.column_height
    if x > 2.0 * c * time:
        return 0.0
    return (2.0 * c - x / time) ** 2 / (9.0 * cfg.gravity)


def ritter_front_over_a(tau: float) -> float:
    """Ritter leading edge in the experiment's ``Z/a`` non-dimensionalisation.

    With ``H = 2a`` and ``tau = t sqrt(2 g / a)`` the identity is exactly
    ``Z/a = 2 tau`` (``2 sqrt(2 g a) t / a``). Using ``Z/a = tau`` instead, as
    ``benchmarks/dam_break/experimental_validation.py`` did until this audit,
    halves the reference and silently changes every reported Ritter error.
    """
    return 2.0 * tau


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
    steps_taken: int = 0
    simulated_time: float = 0.0
    dt_min: float = 0.0
    dt_max: float = 0.0
    # ``dt (c0 + |v|) / h`` accumulated per step. The nominal Courant number of
    # a fixed-step run is ``dt c0 / h``; this is the number the stability
    # condition actually bounds, and the two differ by |v|/c0.
    max_acoustic_courant: float = 0.0
    # ``(time, positions, velocities)`` captured at the requested
    # ``snapshot_times``.
    # Used by the shallow-water (Ritter) verification, which needs the depth
    # field at intermediate times and not only the trajectory.
    snapshots: list = field(default_factory=list)

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


def run_case(cfg: DamBreakConfig, pair_skin_factor: float | None = None,
             snapshot_times=None):
    """Integrate the dam break and return the measured trajectory.

    ``snapshot_times`` (seconds, increasing) asks for the fluid positions to be
    captured at those times, in ``DamBreakRun.snapshots``. Positions only: the
    trajectory above is unchanged by it.
    """
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

    snap_targets = sorted(float(t) for t in (snapshot_times or []))
    snapshots: list = []
    snap_next = 0
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
    compression_peak = torch.zeros((), dtype=torch.float64, device=device)
    speed_peak = torch.zeros((), dtype=positions.dtype, device=device)
    courant_peak = torch.zeros((), dtype=torch.float64, device=device)
    # The peak compression over the whole run is dominated by the surge hitting
    # the far wall, not by the bulk flow. Keep the pre-impact maximum separate so
    # the two effects can be told apart.
    impact_limit = cfg.pool_width - 3.0 * cfg.dx
    compression_at_impact = None
    wall_impact_time = None

    dt = cfg.dt
    elapsed = 0.0
    dt_min = dt_max = dt
    # Same semantics as ``droplet_impact.run``: ``cfg.steps`` at ``cfg.dt`` is
    # the requested simulated time, so an adaptive run takes as many steps as
    # that time needs instead of stopping short.
    target_time = cfg.steps * cfg.dt
    time_tolerance = 1.0e-12 * max(1.0, target_time)
    max_steps = cfg.steps if not cfg.adaptive_dt else 4 * cfg.steps + 1000

    for step in range(max_steps + 1):
        if cache is None:
            pairs = None
        else:
            if wall is not None:
                all_positions[:fluid_count] = positions
            pairs, _ = cache.get(all_positions)
        step_dt = dt if not cfg.adaptive_dt else min(dt, target_time - elapsed)
        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs, wall=wall,
                              dt=step_dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        # Reduce first, rescale in float64: ``(rho/rho0).max() - 1`` in the
        # field's own precision loses ~5e-4 relative at 1.0, which is the size
        # of the quantity being measured near hydrostatic equilibrium.
        compression_peak = torch.maximum(compression_peak,
                                         density.max().double() / cfg.rho0 - 1.0)
        speed_now = (torch.linalg.vector_norm(velocity, dim=-1) if velocity.numel()
                     else torch.zeros(0, dtype=positions.dtype, device=device))
        if speed_now.numel():
            speed_peak = torch.maximum(speed_peak, speed_now.max())
        speed_scalar = float(speed_now.max()) if speed_now.numel() else 0.0
        courant_peak = torch.maximum(
            courant_peak, (torch.tensor(step_dt * (cfg.sound_speed + speed_scalar)
                                        / (cfg.h_factor * cfg.dx),
                                        dtype=torch.float64, device=device)))
        if wall_impact_time is None and float(positions[:, 0].max()) >= impact_limit:
            wall_impact_time = elapsed
            compression_at_impact = float(compression_peak)

        final = (step == cfg.steps if not cfg.adaptive_dt
                 else elapsed >= target_time - time_tolerance)
        if step % cfg.output_every == 0 or final:
            speed = speed_scalar
            max_speed = max(max_speed, speed)
            times.append(elapsed)
            front.append(front_position(positions, cfg))
            while (snap_next < len(snap_targets)
                   and elapsed >= snap_targets[snap_next] - 1.0e-12):
                snapshots.append((elapsed, positions.detach().clone(),
                                  velocity.detach().clone()))
                snap_next += 1
            kinetic.append(float(0.5 * (mass[:, None] * velocity.square()).sum()))
            potential.append(float((mass * cfg.gravity * positions[:, 1]).sum()))

        if final:
            break
        velocity = velocity + step_dt * acc
        positions = positions + step_dt * velocity
        escaped += apply_boundary(positions, velocity, cfg, top=top)
        if cfg.shift_every and step > 0 and step % cfg.shift_every == 0:
            v_ref = (cfg.shift_v_ref if cfg.shift_v_ref is not None
                     else max(speed_scalar, 0.05 * cfg.sound_speed))
            # Volumes come from the summation density of this step; the shift
            # moves positions only, so mass and velocity are untouched.
            volume = mass / density
            positions = shift_positions(positions, volume, h, dt=step_dt,
                                        v_ref=v_ref,
                                        strength=cfg.shift_strength,
                                        limit=cfg.shift_limit)
            escaped += apply_boundary(positions, velocity, cfg, top=top)
        elapsed += step_dt
        if cfg.adaptive_dt:
            dt = next_dt(cfg, dt, speed_scalar)
            dt_min = min(dt_min, dt)
            dt_max = max(dt_max, dt)

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
        steps_taken=step, simulated_time=elapsed,
        dt_min=dt_min, dt_max=dt_max,
        max_acoustic_courant=float(courant_peak),
        snapshots=snapshots,
    )


__all__ = [
    "DamBreakConfig", "DamBreakRun", "front_position", "make_column",
    "ritter_front", "ritter_front_over_a", "ritter_height_profile",
    "run_case", "validate_case", "boundary_box",
]
