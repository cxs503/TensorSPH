"""3-D dam-break benchmark case.

A rectangular column of water of width ``a``, depth ``b`` (spanwise) and height
``2a`` is released against a dry floor inside a tank.  The leading-edge position
is tracked in the streamwise (x) direction and compared against the 2-D Ritter
shallow-water solution on the spanwise mid-plane.  Three-dimensional effects
(side-wall friction, spanwise spreading) are quantified as the deviation from
the 2-D reference.

Non-dimensionalisation follows the classical dam-break experiments:
``Z/a`` against ``tau = t sqrt(2 g / a)`` (Martin & Moyce, 1952).

Scope of the checks, stated up front:

* this module measures mass conservation, boundedness, total-energy drift,
  and the front trajectory vs the 2-D Ritter reference on the mid-plane;
* the 3-D front is expected to lag the 2-D Ritter curve because of side-wall
  friction and spanwise spreading; the deviation is reported, not penalised;
* total energy drift must stay below 1 % of the initial potential energy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .droplet_impact import (
    Config,
    cell_centres,
    hydrostatic_density,
    next_dt,
    resolve_device,
    tait_pressure,
    density_diffusion_rate,
    free_surface_pressure,
    VISCOUS_LAPLACIAN_FACTOR,
)
from .neighbor_search import PairListCache, cell_linked_pairs


# --------------------------------------------------------------------------- #
# 3-D cubic spline kernel
# --------------------------------------------------------------------------- #

def cubic_spline_kernel_3d(r: torch.Tensor, h: float) -> torch.Tensor:
    """3-D cubic spline kernel W(r,h), normalised by 1/(pi*h**3).

    The polynomial shape is the same as the 2-D kernel (Monaghan 1992):
        q < 1:  1 - 3/2 q^2 + 3/4 q^3
        1<=q<2: 1/4 (2 - q)^3
        q>=2:   0
    but the normalisation constant is 1/(pi h^3) for three dimensions.
    """
    q = r / h
    sigma = 1.0 / (math.pi * h * h * h)
    first = 1.0 - 1.5 * q.square() + 0.75 * q.pow(3)
    second = 0.25 * (2.0 - q).clamp_min(0).pow(3)
    shape = torch.where(q < 1.0, first, torch.where(q < 2.0, second, 0.0))
    return sigma * shape.clamp_min(0.0)


def cubic_spline_gradient_3d(
    delta: torch.Tensor, h: float, eps: float = 1.0e-12
) -> torch.Tensor:
    """Gradient of the 3-D cubic spline kernel w.r.t. x_i for delta_ij = x_i - x_j."""
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    sigma = 1.0 / (math.pi * h * h * h)
    dshape = torch.where(
        q < 1.0,
        -3.0 * q + 2.25 * q.square(),
        torch.where(q < 2.0, -0.75 * (2.0 - q).square(), 0.0),
    )
    scale = sigma * dshape / (h * r.clamp_min(eps))
    return delta * scale.unsqueeze(-1)


def pair_kernels_3d(delta: torch.Tensor, h: float, eps: float = 1.0e-12):
    """Return ``(W, grad W)`` for a batch of pair separations in 3-D."""
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    sigma = 1.0 / (math.pi * h * h * h)
    first = 1.0 - 1.5 * q.square() + 0.75 * q.pow(3)
    second = 0.25 * (2.0 - q).clamp_min(0).pow(3)
    shape = torch.where(q < 1.0, first, torch.where(q < 2.0, second, 0.0))
    dshape = torch.where(
        q < 1.0,
        -3.0 * q + 2.25 * q.square(),
        torch.where(q < 2.0, -0.75 * (2.0 - q).square(), 0.0),
    )
    weight = sigma * shape.clamp_min(0.0)
    scale = sigma * dshape / (h * r.clamp_min(eps))
    return weight, delta * scale.unsqueeze(-1)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class DamBreak3DConfig(Config):
    """3-D dam-break case parameters.

    The tank occupies ``[0, tank_width] x [0, tank_depth] x [0, tank_height]``;
    the water column occupies ``[0, column_width] x [0, column_depth] x
    [0, column_height]`` and is at rest at ``t = 0``.
    """

    dx: float = 0.02
    tank_height: float = 0.6
    tank_depth: float = 0.3          # spanwise (z) extent
    column_width: float = 0.15       # streamwise (x) extent
    column_depth: float = 0.15       # spanwise (z) extent
    column_height: float = 0.3       # vertical (y) extent
    pool_width: float = 1.0          # tank x-extent (reused for wall helpers)
    pool_depth: float = 0.6          # tank y-extent (fallback)
    dt: float = 5.0e-5
    steps: int = 4000
    output_every: int = 100
    device: str = "cpu"
    hydrostatic_init: bool = True

    def tank_top(self) -> float:
        return float(self.tank_height)

    def dt_from_courant(self) -> float:
        return (self.max_courant * self.h_factor * self.dx / self.sound_speed)


def validate_case(cfg: DamBreak3DConfig) -> None:
    if cfg.boundary != "wall":
        raise ValueError(
            "the 3-D dam-break case integrates static wall particles only; "
            f"boundary={cfg.boundary!r} would be silently ignored")
    if cfg.dx <= 0 or cfg.h_factor <= 0:
        raise ValueError("dx and h_factor must be positive")
    if cfg.column_width <= 0 or cfg.column_height <= 0 or cfg.column_depth <= 0:
        raise ValueError("column dimensions must be positive")
    if cfg.column_width > cfg.pool_width:
        raise ValueError("column_width must fit inside the tank")
    if cfg.column_height > cfg.tank_height:
        raise ValueError("column_height must fit inside the tank")
    if cfg.column_depth > cfg.tank_depth:
        raise ValueError("column_depth must fit inside the tank")
    if cfg.sound_speed <= 0 or cfg.rho0 <= 0 or cfg.gamma <= 0:
        raise ValueError("rho0, sound_speed, and gamma must be positive")
    courant = cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx)
    if courant > cfg.max_courant * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} gives acoustic Courant {courant:.3f} > max_courant "
            f"{cfg.max_courant:g}; use dt <= {cfg.dt_from_courant():.3e} s")


# --------------------------------------------------------------------------- #
# Particle initialisation
# --------------------------------------------------------------------------- #

def make_column_3d(cfg: DamBreak3DConfig, device: torch.device,
                   seed: int | None = None):
    """Particles filling the initial 3-D water column, resting on the floor.

    If ``seed`` is given, a small random perturbation (amplitude ``dx * 1e-4``)
    is added to break perfect lattice symmetry.  This is useful for testing
    reproducibility: the same seed gives the same initial state.
    """
    dx = cfg.dx
    xs = cell_centres(cfg.column_width, dx, device)
    ys = cell_centres(cfg.column_height, dx, device)
    zs = cell_centres(cfg.column_depth, dx, device)
    if xs.numel() == 0 or ys.numel() == 0 or zs.numel() == 0:
        raise ValueError("column discretization produced no particles")
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1)

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(positions.shape, generator=gen, device=device) * dx * 1e-4
        positions = positions + noise

    # Calibrated 3-D lattice mass (tensorsph.threed.kernel3): the 3-D cubic
    # lattice needs the 3-D-normalised kernel for this, so it cannot reuse the
    # 2-D helper.  ``density_calibration`` on Config used to be ignored here.
    mass_unit = cfg.rho0 * dx * dx * dx
    if getattr(cfg, "density_calibration", True):
        from .threed.kernel3 import calibrated_particle_mass
        mass_unit = calibrated_particle_mass(dx, cfg.h_factor * dx, cfg.rho0)
    if cfg.hydrostatic_init:
        mass = hydrostatic_density(positions[:, 1], cfg,
                                   surface=cfg.column_height) * (mass_unit / cfg.rho0)
    else:
        mass = torch.full((positions.shape[0],), mass_unit, device=device)
    return positions, mass


# --------------------------------------------------------------------------- #
# 3-D wall lattice
# --------------------------------------------------------------------------- #

def wall_lattice_3d(cfg: DamBreak3DConfig, device: torch.device,
                    top: float | None = None) -> torch.Tensor:
    """Static wall particles on the bottom and all four side walls.

    Three layers with the fluid lattice spacing, placed half a spacing outside
    the fluid domain.  Walls: bottom (y=0), left (x=0), right (x=tank_width),
    front (z=0), back (z=tank_depth).  No top wall (free surface).
    """
    if cfg.wall_layers < 1:
        raise ValueError("wall_layers must be positive")
    dx = cfg.dx
    layers = cfg.wall_layers
    top = cfg.tank_top() if top is None else float(top)
    tw = cfg.pool_width   # tank x-extent
    td = cfg.tank_depth   # tank z-extent

    parts: list[torch.Tensor] = []

    # Bottom wall: y = -layer*dx, spanning x in [-layers*dx, tw+layers*dx],
    # z in [-layers*dx, td+layers*dx]
    span_x = torch.arange(-(layers - 0.5) * dx,
                          tw + (layers - 0.5) * dx, dx, device=device)
    span_z = torch.arange(-(layers - 0.5) * dx,
                          td + (layers - 0.5) * dx, dx, device=device)
    for layer in range(1, layers + 1):
        y = torch.full((1,), dx * 0.5 - layer * dx, device=device)
        gx, gy, gz = torch.meshgrid(span_x, y, span_z, indexing="xy")
        parts.append(torch.stack(
            (gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1))

    # Left wall: x = -layer*dx, spanning y in [0, top], z in [-layers*dx, td+layers*dx]
    cols_y = torch.arange(dx * 0.5, top, dx, device=device)
    for layer in range(1, layers + 1):
        x = torch.full((1,), dx * 0.5 - layer * dx, device=device)
        gx, gy, gz = torch.meshgrid(x, cols_y, span_z, indexing="xy")
        parts.append(torch.stack(
            (gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1))

    # Right wall: x = tw + layer*dx
    for layer in range(1, layers + 1):
        x = torch.full((1,), tw - dx * 0.5 + layer * dx, device=device)
        gx, gy, gz = torch.meshgrid(x, cols_y, span_z, indexing="xy")
        parts.append(torch.stack(
            (gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1))

    # Front wall: z = -layer*dx, spanning x in [0, tw], y in [0, top]
    span_x_wall = torch.arange(dx * 0.5, tw, dx, device=device)
    for layer in range(1, layers + 1):
        z = torch.full((1,), dx * 0.5 - layer * dx, device=device)
        gx, gy, gz = torch.meshgrid(span_x_wall, cols_y, z, indexing="xy")
        parts.append(torch.stack(
            (gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1))

    # Back wall: z = td + layer*dx
    for layer in range(1, layers + 1):
        z = torch.full((1,), td - dx * 0.5 + layer * dx, device=device)
        gx, gy, gz = torch.meshgrid(span_x_wall, cols_y, z, indexing="xy")
        parts.append(torch.stack(
            (gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1))

    return torch.cat(parts, dim=0)


# --------------------------------------------------------------------------- #
# 3-D RHS (right-hand side of the ODE)
# --------------------------------------------------------------------------- #

def rhs_3d(pos, vel, mass, cfg: DamBreak3DConfig, pairs=None, wall=None,
           dt=None):
    """Compute density, pressure, and acceleration for a 3-D particle set.

    Same physics as the 2-D ``rhs`` but with the 3-D kernel.
    """
    n = pos.shape[0]
    nw = wall.shape[0] if wall is not None else 0
    h = cfg.h_factor * cfg.dx
    step_dt = dt if dt is not None else cfg.dt
    gravity = torch.tensor([0.0, -cfg.gravity, 0.0], dtype=pos.dtype,
                           device=pos.device)

    if wall is not None:
        all_pos = torch.cat((pos, wall), dim=0)
        all_mass = torch.cat((mass, torch.zeros(nw, dtype=mass.dtype,
                                                device=mass.device)))
    else:
        all_pos = pos
        all_mass = mass

    if pairs is not None:
        i, j = pairs[:, 0], pairs[:, 1]
    else:
        pairs = cell_linked_pairs(all_pos, 2.0 * h)
        i, j = pairs[:, 0], pairs[:, 1]

    delta = all_pos[i] - all_pos[j]
    distance = torch.linalg.vector_norm(delta, dim=1)
    wij, grad_w = pair_kernels_3d(delta, h)

    rho_all = torch.zeros(n + nw, dtype=pos.dtype, device=pos.device)
    rho_all = rho_all.index_add(0, i, all_mass[j] * wij)
    rho_all = rho_all.index_add(0, j, all_mass[i] * wij)
    rho = rho_all[:n].clamp_min(0.1 * cfg.rho0)

    if cfg.density_diffusion > 0.0:
        fluid_only = (i < n) & (j < n)
        radial = -(delta[fluid_only] * grad_w[fluid_only]).sum(dim=1)
        rate = density_diffusion_rate(
            rho, mass, distance[fluid_only], radial,
            (i[fluid_only], j[fluid_only]), cfg)
        rho = (rho + step_dt * rate).clamp_min(0.1 * cfg.rho0)

    pressure = tait_pressure(rho, cfg)
    if cfg.surface_pressure_closure:
        pressure = free_surface_pressure(pos, rho, pressure, delta, wij, i, j, cfg)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
        i_fluid = i < n
        crossing = i_fluid != (j < n)
        fluid_idx = torch.where(i_fluid, i, j)
        wall_idx = (torch.where(i_fluid, j, i) - n).clamp_(0, nw - 1)
        weight = torch.where(crossing, wij, torch.zeros_like(wij))
        zeros = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
        pressure_ext = torch.cat((pressure, zeros))
        numerator = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
        denominator = torch.zeros(nw, dtype=pos.dtype, device=pos.device)

        dy = torch.where(i_fluid, -delta[:, 1], delta[:, 1])
        body = -cfg.gravity * dy
        source = pressure_ext[fluid_idx]
        if cfg.gravity != 0.0:
            source = source + torch.cat((rho, zeros))[fluid_idx] * body
        numerator = numerator.index_add(0, wall_idx, source * weight)
        denominator = denominator.index_add(0, wall_idx, weight)
        wall_pressure = (numerator / denominator.clamp_min(1.0e-30)).clamp_min(0.0)
        wall_density = (cfg.rho0 * (wall_pressure / bulk + 1.0).pow(1.0 / cfg.gamma)
                        ).clamp_min(cfg.rho0)
        pressure_all = torch.cat((pressure, wall_pressure))
        density_all = torch.cat((rho, wall_density))
    else:
        pressure_all, density_all = pressure, rho

    if nw:
        all_vel = torch.cat((vel, torch.zeros((nw, pos.shape[1]),
                                              dtype=pos.dtype, device=pos.device)))
    else:
        all_vel = vel

    pressure_term = (pressure_all[i] / density_all[i].square()
                     + pressure_all[j] / density_all[j].square())
    pair_pressure = pressure_term[:, None] * grad_w

    acc = torch.zeros((n + nw, pos.shape[1]), dtype=pos.dtype, device=pos.device)
    acc = acc.index_add(0, i, -all_mass[j, None] * pair_pressure)
    acc = acc.index_add(0, j, all_mass[i, None] * pair_pressure)
    acc_pressure = acc[:n]

    # Viscosity
    fluid_pair = (i < n) & (j < n)
    if nw and cfg.wall_friction > 0.0:
        crossing = torch.logical_xor(i < n, j < n)
        blend = (fluid_pair.to(pos.dtype)
                 + cfg.wall_friction * crossing.to(pos.dtype))
        visc_scale = (VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity * blend)[:, None]
    elif nw:
        visc_scale = (VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity
                      * fluid_pair.to(pos.dtype))[:, None]
    else:
        visc_scale = VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity
    dv = all_vel[i] - all_vel[j]
    denom = distance.square() + 0.01 * h * h
    laplace_weight = torch.relu(-(delta * grad_w).sum(dim=1)) / denom
    visc_i = (-visc_scale * dv
              * (all_mass[j] / density_all[j] * laplace_weight)[:, None])
    visc_j = (visc_scale * dv
              * (all_mass[i] / density_all[i] * laplace_weight)[:, None])
    acc_visc = torch.zeros((n + nw, pos.shape[1]), dtype=pos.dtype,
                           device=pos.device)
    acc_visc = acc_visc.index_add(0, i, visc_i)
    acc_visc = acc_visc.index_add(0, j, visc_j)

    return rho, pressure, acc_pressure + acc_visc[:n] + gravity


# --------------------------------------------------------------------------- #
# 3-D boundary
# --------------------------------------------------------------------------- #

def apply_boundary_3d(pos, vel, cfg: DamBreak3DConfig,
                      top: float | None = None) -> int:
    """Keep particles inside the 3-D box and count crossings."""
    tw = cfg.pool_width
    td = cfg.tank_depth
    th = cfg.tank_top() if top is None else float(top)
    tolerance = 0.25 * cfg.dx

    low_x = pos[:, 0] < -tolerance
    high_x = pos[:, 0] > tw + tolerance
    low_y = pos[:, 1] < -tolerance
    high_y = pos[:, 1] > th + tolerance
    low_z = pos[:, 2] < -tolerance
    high_z = pos[:, 2] > td + tolerance

    pos[:, 0] = pos[:, 0].clamp(0.0, tw)
    pos[:, 1] = pos[:, 1].clamp(0.0, th)
    pos[:, 2] = pos[:, 2].clamp(0.0, td)

    vel[:, 0] = torch.where(low_x | high_x, torch.zeros_like(vel[:, 0]), vel[:, 0])
    vel[:, 1] = torch.where(low_y | high_y, torch.zeros_like(vel[:, 1]), vel[:, 1])
    vel[:, 2] = torch.where(low_z | high_z, torch.zeros_like(vel[:, 2]), vel[:, 2])

    return int((low_x | high_x | low_y | high_y | low_z | high_z).sum().item())


# --------------------------------------------------------------------------- #
# Front position and Ritter reference
# --------------------------------------------------------------------------- #

def front_position_3d(pos: torch.Tensor, cfg: DamBreak3DConfig) -> float:
    """Leading-edge x coordinate of the water (fluid particles only)."""
    if pos.numel() == 0:
        return 0.0
    return float(pos[:, 0].max())


def ritter_front(time: float, cfg: DamBreak3DConfig) -> float:
    """Ritter shallow-water leading edge on a dry bed, measured from the dam.

    ``Z(t) = 2 sqrt(g H) t``.  Reported as a reference only: the 3-D front
    is expected to lag because of side-wall friction and spanwise spreading.
    """
    return 2.0 * math.sqrt(cfg.gravity * cfg.column_height) * time


def ritter_front_over_a(tau: float) -> float:
    """Ritter leading edge in the experiment's ``Z/a`` non-dimensionalisation.

    With ``H = 2a`` and ``tau = t sqrt(2 g / a)`` the identity is exactly
    ``Z/a = 2 tau``.
    """
    return 2.0 * tau


# --------------------------------------------------------------------------- #
# Run dataclass
# --------------------------------------------------------------------------- #

@dataclass
class DamBreak3DRun:
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
    initial_energy: float = 0.0
    final_energy: float = 0.0
    steps_taken: int = 0
    simulated_time: float = 0.0
    dt_min: float = 0.0
    dt_max: float = 0.0
    max_acoustic_courant: float = 0.0
    snapshots: list = field(default_factory=list)

    @property
    def mass_drift(self) -> float:
        return self.final_mass - self.initial_mass

    @property
    def energy_drift(self) -> float:
        """Total energy drift as a fraction of the initial potential energy."""
        if self.initial_energy == 0.0:
            return 0.0
        return (self.final_energy - self.initial_energy) / self.initial_energy

    def dimensionless(self, cfg: DamBreak3DConfig):
        scale = cfg.column_width
        return ([t * math.sqrt(2.0 * cfg.gravity / cfg.column_width)
                 for t in self.times],
                [z / scale for z in self.front])


# --------------------------------------------------------------------------- #
# run_case
# --------------------------------------------------------------------------- #

def run_case(cfg: DamBreak3DConfig, pair_skin_factor: float | None = None,
             snapshot_times=None, seed: int | None = None):
    """Integrate the 3-D dam break and return the measured trajectory.

    ``snapshot_times`` (seconds, increasing) asks for the fluid positions to be
    captured at those times, in ``DamBreak3DRun.snapshots``.
    ``seed`` is passed to :func:`make_column_3d` for reproducible perturbations.
    """
    validate_case(cfg)
    device = resolve_device(cfg.device)
    top = cfg.tank_top()
    positions, mass = make_column_3d(cfg, device, seed=seed)
    wall = wall_lattice_3d(cfg, device, top=top)
    velocity = torch.zeros_like(positions)
    initial_mass = float(mass.sum())

    # Initial potential energy: sum(m_i * g * y_i)
    initial_pe = float((mass * cfg.gravity * positions[:, 1]).sum())

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
    compression_peak = torch.zeros((), dtype=torch.float64, device=device)
    speed_peak = torch.zeros((), dtype=positions.dtype, device=device)
    courant_peak = torch.zeros((), dtype=torch.float64, device=device)

    impact_limit = cfg.pool_width - 3.0 * cfg.dx
    compression_at_impact = None
    wall_impact_time = None

    dt = cfg.dt
    elapsed = 0.0
    dt_min = dt_max = dt
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
        density, _, acc = rhs_3d(positions, velocity, mass, cfg, pairs=pairs,
                                 wall=wall, dt=step_dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")

        compression_peak = torch.maximum(
            compression_peak, density.max().double() / cfg.rho0 - 1.0)
        speed_now = (torch.linalg.vector_norm(velocity, dim=-1)
                     if velocity.numel()
                     else torch.zeros(0, dtype=positions.dtype, device=device))
        if speed_now.numel():
            speed_peak = torch.maximum(speed_peak, speed_now.max())
        speed_scalar = float(speed_now.max()) if speed_now.numel() else 0.0
        courant_peak = torch.maximum(
            courant_peak,
            torch.tensor(step_dt * (cfg.sound_speed + speed_scalar)
                         / (cfg.h_factor * cfg.dx),
                         dtype=torch.float64, device=device))

        if wall_impact_time is None and float(positions[:, 0].max()) >= impact_limit:
            wall_impact_time = elapsed
            compression_at_impact = float(compression_peak)

        final = (step == cfg.steps if not cfg.adaptive_dt
                 else elapsed >= target_time - time_tolerance)
        if step % cfg.output_every == 0 or final:
            max_speed = max(max_speed, speed_scalar)
            times.append(elapsed)
            front.append(front_position_3d(positions, cfg))
            while (snap_next < len(snap_targets)
                   and elapsed >= snap_targets[snap_next] - 1.0e-12):
                snapshots.append((elapsed, positions.detach().clone(),
                                  velocity.detach().clone()))
                snap_next += 1
            ke = float(0.5 * (mass[:, None] * velocity.square()).sum())
            pe = float((mass * cfg.gravity * positions[:, 1]).sum())
            kinetic.append(ke)
            potential.append(pe)

        if final:
            break
        velocity = velocity + step_dt * acc
        positions = positions + step_dt * velocity
        escaped += apply_boundary_3d(positions, velocity, cfg, top=top)
        elapsed += step_dt
        if cfg.adaptive_dt:
            dt = next_dt(cfg, dt, speed_scalar)
            dt_min = min(dt_min, dt)
            dt_max = max(dt_max, dt)

    final_ke = float(0.5 * (mass[:, None] * velocity.square()).sum())
    final_pe = float((mass * cfg.gravity * positions[:, 1]).sum())
    final_energy = final_ke + final_pe

    return DamBreak3DRun(
        times=times, front=front, kinetic_energy=kinetic,
        potential_energy=potential, mass=initial_mass, escaped=escaped,
        rebuilds=0 if cache is None else cache.rebuilds,
        max_speed=max(max_speed, float(speed_peak)),
        max_density_compression=float(compression_peak),
        compression_before_wall_impact=float(
            compression_peak if compression_at_impact is None
            else compression_at_impact),
        wall_impact_time=wall_impact_time,
        initial_mass=initial_mass, final_mass=float(mass.sum()),
        final_positions=positions.detach().clone(),
        initial_energy=initial_pe, final_energy=final_energy,
        steps_taken=step, simulated_time=elapsed,
        dt_min=dt_min, dt_max=dt_max,
        max_acoustic_courant=float(courant_peak),
        snapshots=snapshots,
    )


__all__ = [
    "DamBreak3DConfig", "DamBreak3DRun", "front_position_3d", "make_column_3d",
    "ritter_front", "ritter_front_over_a", "run_case", "validate_case",
    "wall_lattice_3d", "rhs_3d", "apply_boundary_3d",
    "cubic_spline_kernel_3d", "cubic_spline_gradient_3d", "pair_kernels_3d",
]
