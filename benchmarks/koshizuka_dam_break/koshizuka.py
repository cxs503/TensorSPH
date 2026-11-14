"""Koshizuka & Oka (1996) dam-break pressure peak benchmark.

Reference:
    Koshizuka, S. & Oka, Y. (1996). Moving-particle semi-implicit method for
    fragmentation of incompressible fluid. *Nuclear Science and Engineering*,
    123(3), 421-434. DOI: 10.13182/NSE96-A24205.

Geometry (all lengths in metres, tank frame, floor at y = 0):
    tank length:       0.584
    tank height:       0.30   (must accommodate 0.28 m column)
    column width (a):  0.14
    column height:     0.28   (2a)
    probe location:    x = 0.584 m (right wall), y = 0.024 m from floor

Reference pressure peak (from Koshizuka & Oka 1996, Fig. 15):
    P_max ≈ 2700 Pa at the probe (y = 0.024 m)
    Tolerance: <12% per task specification.

Note on geometry: The original paper (Fig. 10) shows a tank that accommodates
the 0.28 m high water column. Some secondary sources quote 0.14 m tank height,
but this is inconsistent with the column height. We use 0.30 m to ensure the
column fits with a small margin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from tensorsph.dam_break import (
    DamBreakConfig,
    DamBreakRun,
    validate_case,
    make_column,
    front_position,
)
from tensorsph.droplet_impact import (
    resolve_device,
    rhs,
    wall_lattice,
    tait_pressure,
    cubic_spline_kernel,
    pair_kernels,
    apply_boundary,
    next_dt,
)
from tensorsph.neighbor_search import PairListCache


@dataclass
class KoshizukaConfig(DamBreakConfig):
    """Koshizuka dam-break case parameters.

    The tank occupies [0, tank_width] x [0, tank_height]; the water column
    occupies [0, column_width] x [0, column_height] and is at rest at t = 0.
    """

    # Koshizuka geometry
    tank_width: float = 0.584
    tank_height: float = 0.30
    column_width: float = 0.14
    column_height: float = 0.28
    pool_width: float = 0.584
    pool_depth: float = 0.30

    # Pressure probe on the right wall
    probe_y: float = 0.024       # height above floor (m)
    probe_halfwidth: float = 1.5  # in units of dx, for averaging

    # Time stepping
    dx: float = 0.005            # particle spacing (m)
    sound_speed: float = 30.0    # m/s, higher than default for lower compressibility
    max_courant: float = 0.2
    dt: float = 5.0e-5
    steps: int = 8000            # ~0.4 s simulated time
    output_every: int = 10

    # Physics
    hydrostatic_init: bool = True
    wall_friction: float = 0.0   # free slip
    density_diffusion: float = 0.1  # mild density diffusion for stability
    adaptive_dt: bool = True

    def tank_top(self) -> float:
        return self.tank_height

    def dt_from_courant(self) -> float:
        return self.max_courant * self.h_factor * self.dx / self.sound_speed


@dataclass
class KoshizukaRun:
    """Results from a Koshizuka dam-break run."""

    times: list[float] = field(default_factory=list)
    front: list[float] = field(default_factory=list)
    pressure_probe: list[float] = field(default_factory=list)  # pressure at probe (Pa)
    pressure_probe_times: list[float] = field(default_factory=list)

    # Derived quantities
    peak_pressure: float = float("nan")
    peak_pressure_time: float = float("nan")

    # Standard diagnostics
    mass: float = 0.0
    initial_mass: float = 0.0
    final_mass: float = 0.0
    escaped: int = 0
    max_speed: float = 0.0
    max_density_compression: float = 0.0
    wall_impact_time: float | None = None
    steps_taken: int = 0
    simulated_time: float = 0.0
    dt_min: float = 0.0
    dt_max: float = 0.0
    max_acoustic_courant: float = 0.0

    @property
    def mass_drift(self) -> float:
        return self.final_mass - self.initial_mass


def wall_pressure_at_probe(positions, velocity, mass, density, cfg: KoshizukaConfig,
                           wall, wall_density, device: torch.device) -> float:
    """Compute pressure at a probe point on the right wall using SPH interpolation.

    The probe is at (tank_width, probe_y). We find wall particles near the probe
    and interpolate their pressure using the kernel.
    """
    probe_x = cfg.tank_width
    probe_y = cfg.probe_y
    h = cfg.h_factor * cfg.dx

    # Find wall particles near the probe
    dx_wall = wall[:, 0] - probe_x
    dy_wall = wall[:, 1] - probe_y
    dist_sq = dx_wall ** 2 + dy_wall ** 2
    mask = dist_sq <= (2.0 * h) ** 2  # within 2h

    if not mask.any():
        return 0.0

    # Get wall particle pressures from their density via Tait EOS
    wall_rho = wall_density[mask]
    wall_p = tait_pressure(wall_rho, cfg)

    # Kernel-weighted average
    r = torch.sqrt(dist_sq[mask])
    q = r / h
    w = cubic_spline_kernel(q, cfg.h_factor)

    if w.sum() < 1e-30:
        return 0.0

    return float((w * wall_p).sum() / w.sum())


def _compute_wall_density(positions, density, mass, cfg, wall, pairs):
    """Compute wall particle density using Adami et al. (2012) extrapolation.

    pairs is a tensor of shape (pair_count, 2) with indices into [fluid..., wall...].
    """
    n_wall = wall.shape[0]
    n_fluid = positions.shape[0]
    if n_wall == 0:
        return torch.zeros(0, device=positions.device)

    h = cfg.h_factor * cfg.dx
    wall_density = torch.full((n_wall,), cfg.rho0, device=positions.device)

    if pairs is not None and pairs.numel() > 0:
        # pairs is (N, 2) tensor
        i = pairs[:, 0]
        j = pairs[:, 1]

        # Find fluid-wall pairs: i < n_fluid, j >= n_fluid
        fw_mask = (i < n_fluid) & (j >= n_fluid)

        if fw_mask.any():
            i_fw = i[fw_mask]
            j_fw = j[fw_mask] - n_fluid  # wall index

            delta = positions[i_fw] - wall[j_fw]
            r = torch.linalg.vector_norm(delta, dim=-1)
            q = r / h
            w = cubic_spline_kernel(q, cfg.h_factor)

            # Adami extrapolation: rho_w = rho0 * (p_i / B + 1)^(1/gamma)
            p_i = tait_pressure(density[i_fw], cfg)
            B = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
            rho_w = cfg.rho0 * (p_i / B + 1.0) ** (1.0 / cfg.gamma)

            # Kernel-weighted average per wall particle using scatter
            w_sum = torch.zeros(n_wall, device=positions.device)
            w_rho = torch.zeros(n_wall, device=positions.device)

            w_sum.scatter_add_(0, j_fw, w)
            w_rho.scatter_add_(0, j_fw, w * rho_w)

            valid = w_sum > 1e-30
            wall_density[valid] = w_rho[valid] / w_sum[valid]

    return wall_density


def run_koshizuka(cfg: KoshizukaConfig) -> KoshizukaRun:
    """Run the Koshizuka dam-break case and return pressure history."""

    validate_case(cfg)
    device = resolve_device(cfg.device)
    top = cfg.tank_top()

    positions, mass = make_column(cfg, device)
    wall = wall_lattice(cfg, device, top=top)
    velocity = torch.zeros_like(positions)
    initial_mass = float(mass.sum())

    h = cfg.h_factor * cfg.dx
    fluid_count = positions.shape[0]
    skin = cfg.pair_skin_factor
    cache = PairListCache(2.0 * h, skin * h, fluid_count=fluid_count) if skin > 0 else None
    all_positions = torch.cat((positions, wall), dim=0)

    times: list[float] = []
    front: list[float] = []
    pressure_probe: list[float] = []
    pressure_probe_times: list[float] = []

    escaped = 0
    max_speed = 0.0
    compression_peak = torch.zeros((), dtype=torch.float64, device=device)
    speed_peak = torch.zeros((), dtype=positions.dtype, device=device)
    courant_peak = torch.zeros((), dtype=torch.float64, device=device)

    impact_limit = cfg.tank_width - 3.0 * cfg.dx
    wall_impact_time = None

    dt = cfg.dt
    elapsed = 0.0
    dt_min = dt_max = dt
    target_time = cfg.steps * cfg.dt
    time_tolerance = 1.0e-12 * max(1.0, target_time)
    max_steps = cfg.steps if not cfg.adaptive_dt else 4 * cfg.steps + 1000

    peak_pressure = 0.0
    peak_pressure_time = 0.0

    for step in range(max_steps + 1):
        if cache is None:
            pairs = None
        else:
            all_positions[:fluid_count] = positions
            pairs, _ = cache.get(all_positions)

        step_dt = dt if not cfg.adaptive_dt else min(dt, target_time - elapsed)

        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs, wall=wall,
                              dt=step_dt)

        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")

        # Compute wall density and pressure for probe measurement
        wall_density = _compute_wall_density(positions, density, mass, cfg, wall, pairs)

        # Measure pressure at probe
        p_probe = wall_pressure_at_probe(positions, velocity, mass, density, cfg,
                                         wall, wall_density, device)

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

        # Track peak pressure
        if p_probe > peak_pressure:
            peak_pressure = p_probe
            peak_pressure_time = elapsed

        final = (step == cfg.steps if not cfg.adaptive_dt
                 else elapsed >= target_time - time_tolerance)

        if step % cfg.output_every == 0 or final:
            times.append(elapsed)
            front.append(front_position(positions, cfg))
            pressure_probe.append(p_probe)
            pressure_probe_times.append(elapsed)
            max_speed = max(max_speed, speed_scalar)

        if final:
            break

        velocity = velocity + step_dt * acc
        positions = positions + step_dt * velocity
        escaped += apply_boundary(positions, velocity, cfg, top=top)
        elapsed += step_dt

        if cfg.adaptive_dt:
            dt = next_dt(cfg, dt, speed_scalar)
            dt_min = min(dt_min, dt)
            dt_max = max(dt_max, dt)

    return KoshizukaRun(
        times=times,
        front=front,
        pressure_probe=pressure_probe,
        pressure_probe_times=pressure_probe_times,
        peak_pressure=peak_pressure,
        peak_pressure_time=peak_pressure_time,
        mass=initial_mass,
        initial_mass=initial_mass,
        final_mass=float(mass.sum()),
        escaped=escaped,
        max_speed=max_speed,
        max_density_compression=float(compression_peak),
        wall_impact_time=wall_impact_time,
        steps_taken=step,
        simulated_time=elapsed,
        dt_min=dt_min,
        dt_max=dt_max,
        max_acoustic_courant=float(courant_peak),
    )


# Reference value from Koshizuka & Oka (1996), Fig. 15
KOSHIZUKA_REF_PRESSURE_PA = 2700.0  # Pa, central estimate
KOSHIZUKA_REF_PRESSURE_UNCERTAINTY = 0.12  # 12% tolerance per task spec
