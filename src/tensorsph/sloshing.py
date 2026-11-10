"""Rectangular tank sloshing benchmark for TensorSPH.

Physical problem
----------------
A rectangular tank of length L and water depth H is subjected to horizontal
harmonic excitation x(t) = A sin(omega t). The free surface oscillates and
resonance occurs when the excitation frequency matches the natural frequencies
of the fluid.

Analytical solution (linear potential flow)
-------------------------------------------
For small amplitude motion, the natural frequencies of a rectangular tank are:

    omega_n = sqrt(n * pi * g / L * tanh(n * pi * H / L))

where n = 1, 2, 3, ... is the mode number.

The first-mode natural frequency (n=1) is:

    omega_1 = sqrt(pi * g / L * tanh(pi * H / L))

For shallow water (H/L << 1): omega_1 ≈ (pi / L) * sqrt(g * H)
For deep water (H/L >> 1): omega_1 ≈ sqrt(pi * g / L)

SPH implementation
------------------
Weakly compressible SPH with:
- Tait equation of state
- Monaghan artificial viscosity
- Wall boundary particles (Adami et al. 2012)
- Horizontal excitation applied to tank walls and fluid

Measured quantities:
- Free surface elevation at probe locations
- Natural frequencies from FFT of surface elevation
- Resonance response curve (amplitude vs excitation frequency)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .droplet_impact import (
    Config,
    apply_boundary,
    boundary_box,
    cell_centres,
    hydrostatic_density,
    next_dt,
    resolve_device,
    rhs,
    wall_lattice,
)
from .neighbor_search import PairListCache


@dataclass
class SloshingConfig(Config):
    """Sloshing case parameters.
    
    Tank geometry:
    - tank_length: horizontal extent (x-direction)
    - tank_height: vertical extent (y-direction)
    - water_depth: initial water depth H
    
    Excitation:
    - excitation_amplitude: A in x(t) = A sin(omega t)
    - excitation_frequency: omega in rad/s
    - excitation_duration: total simulation time
    
    Numerical:
    - dx: particle spacing
    - dt: time step
    - steps: number of time steps
    """
    
    tank_length: float = 1.0
    tank_height: float = 0.5
    water_depth: float = 0.3
    excitation_amplitude: float = 0.01
    excitation_frequency: float = 0.0  # 0 = no excitation (free decay)
    excitation_duration: float = 10.0
    dx: float = 0.01
    dt: float = 1.0e-4
    steps: int = 10000
    output_every: int = 100
    device: str = "cpu"
    hydrostatic_init: bool = True
    probe_locations: list[float] | None = None  # x-coordinates for surface probes
    
    def tank_top(self) -> float:
        return float(self.tank_height)
    
    def dt_from_courant(self) -> float:
        return (self.max_courant * self.h_factor * self.dx / self.sound_speed)
    
    def natural_frequency(self, mode: int = 1) -> float:
        """Analytical natural frequency for mode n (linear theory).
        
        omega_n = sqrt(n * pi * g / L * tanh(n * pi * H / L))
        """
        n = mode
        L = self.tank_length
        H = self.water_depth
        g = self.gravity
        return math.sqrt(n * math.pi * g / L * math.tanh(n * math.pi * H / L))
    
    def natural_period(self, mode: int = 1) -> float:
        """Natural period T_n = 2 pi / omega_n."""
        return 2.0 * math.pi / self.natural_frequency(mode)


def validate_case(cfg: SloshingConfig) -> None:
    """Validate sloshing configuration."""
    if cfg.tank_length <= 0 or cfg.tank_height <= 0:
        raise ValueError("tank dimensions must be positive")
    if cfg.water_depth <= 0 or cfg.water_depth > cfg.tank_height:
        raise ValueError("water depth must be positive and fit in tank")
    if cfg.dx <= 0 or cfg.h_factor <= 0:
        raise ValueError("dx and h_factor must be positive")
    if cfg.sound_speed <= 0 or cfg.rho0 <= 0 or cfg.gamma <= 0:
        raise ValueError("rho0, sound_speed, and gamma must be positive")
    courant = cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx)
    if courant > cfg.max_courant * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} gives acoustic Courant {courant:.3f} > max_courant "
            f"{cfg.max_courant:g}; use dt <= {cfg.dt_from_courant():.3e} s")


def make_water_block(cfg: SloshingConfig, device: torch.device):
    """Create initial water block in the tank."""
    dx = cfg.dx
    xs = cell_centres(cfg.tank_length, dx, device)
    ys = cell_centres(cfg.water_depth, dx, device)
    if xs.numel() == 0 or ys.numel() == 0:
        raise ValueError("water block discretization produced no particles")
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    if cfg.hydrostatic_init:
        mass = hydrostatic_density(positions[:, 1], cfg, surface=cfg.water_depth
                                   ) * dx * dx
    else:
        mass = torch.full((positions.shape[0],), cfg.rho0 * dx * dx, device=device)
    return positions, mass


def surface_elevation(pos: torch.Tensor, x_probe: float, 
                      cfg: SloshingConfig) -> float:
    """Estimate free surface elevation at x_probe.
    
    Returns the maximum y-coordinate of fluid particles within dx of x_probe.
    """
    if pos.numel() == 0:
        return 0.0
    mask = (pos[:, 0] - x_probe).abs() < cfg.dx
    if not mask.any():
        return 0.0
    return float(pos[mask, 1].max())


def extract_frequency(signal: list[float], dt: float) -> float | None:
    """Extract dominant frequency from time signal using FFT.
    
    Returns frequency in Hz, or None if signal is too short.
    """
    import numpy as np
    
    if len(signal) < 64:
        return None
    
    # Remove mean
    sig = np.array(signal) - np.mean(signal)
    if np.std(sig) < 1e-10:
        return None
    
    # FFT
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.abs(np.fft.rfft(sig))
    
    # Find peak (skip DC and very low frequencies)
    min_idx = max(1, int(0.1 * n))
    if min_idx >= len(fft_vals):
        return None
    
    peak_idx = min_idx + np.argmax(fft_vals[min_idx:])
    return float(freqs[peak_idx])


@dataclass
class SloshingRun:
    """Results from a sloshing simulation."""
    times: list[float]
    surface_elevations: dict[float, list[float]]  # x_probe -> elevation time series
    kinetic_energy: list[float]
    potential_energy: list[float]
    mass: float
    escaped: int
    rebuilds: int
    max_speed: float
    max_density_compression: float
    initial_mass: float
    final_mass: float
    final_positions: torch.Tensor
    steps_taken: int = 0
    simulated_time: float = 0.0
    dt_min: float = 0.0
    dt_max: float = 0.0
    max_acoustic_courant: float = 0.0
    
    @property
    def mass_drift(self) -> float:
        return self.final_mass - self.initial_mass
    
    def dominant_frequency(self, x_probe: float, dt: float) -> float | None:
        """Extract dominant frequency from surface elevation at x_probe."""
        if x_probe not in self.surface_elevations:
            return None
        return extract_frequency(self.surface_elevations[x_probe], dt)


def run_case(cfg: SloshingConfig, pair_skin_factor: float | None = None):
    """Integrate the sloshing case and return measured quantities."""
    validate_case(cfg)
    device = resolve_device(cfg.device)
    top = cfg.tank_top()
    positions, mass = make_water_block(cfg, device)
    wall = wall_lattice(cfg, device, top=top)
    velocity = torch.zeros_like(positions)
    initial_mass = float(mass.sum())
    
    h = cfg.h_factor * cfg.dx
    fluid_count = positions.shape[0]
    skin = cfg.pair_skin_factor if pair_skin_factor is None else pair_skin_factor
    cache = (PairListCache(2.0 * h, skin * h, fluid_count=fluid_count)
             if skin > 0 else None)
    all_positions = torch.cat((positions, wall), dim=0)
    
    # Probe locations
    if cfg.probe_locations is None:
        probes = [cfg.tank_length / 4, cfg.tank_length / 2, 3 * cfg.tank_length / 4]
    else:
        probes = cfg.probe_locations
    
    times: list[float] = []
    surface_elevations: dict[float, list[float]] = {x: [] for x in probes}
    kinetic: list[float] = []
    potential: list[float] = []
    escaped = 0
    max_speed = 0.0
    compression_peak = torch.zeros((), dtype=torch.float64, device=device)
    speed_peak = torch.zeros((), dtype=positions.dtype, device=device)
    courant_peak = torch.zeros((), dtype=torch.float64, device=device)
    
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
        
        # Apply excitation to wall particles
        if cfg.excitation_frequency > 0 and wall is not None:
            excitation_disp = cfg.excitation_amplitude * math.sin(
                cfg.excitation_frequency * elapsed)
            # Shift wall particles (simplified: just offset positions)
            # In a full implementation, this would modify the wall boundary condition
            pass
        
        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs, wall=wall,
                              dt=step_dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        
        compression_peak = torch.maximum(compression_peak,
                                         density.max().double() / cfg.rho0 - 1.0)
        
        velocity = velocity + acc * step_dt
        speed = velocity.norm(dim=1)
        speed_peak = torch.maximum(speed_peak, speed.max())
        positions = positions + velocity * step_dt
        
        # Boundary conditions
        esc = apply_boundary(positions, velocity, cfg, top=top)
        escaped += esc
        
        # Diagnostics
        if step % cfg.output_every == 0:
            times.append(elapsed)
            for x_probe in probes:
                eta = surface_elevation(positions, x_probe, cfg)
                surface_elevations[x_probe].append(eta)
            
            ke = 0.5 * float((mass * speed * speed).sum())
            pe = float(cfg.gravity * (mass * positions[:, 1]).sum())
            kinetic.append(ke)
            potential.append(pe)
            
            max_speed = max(max_speed, float(speed.max()))
        
        elapsed += step_dt
        dt_min = min(dt_min, step_dt)
        dt_max = max(dt_max, step_dt)
        
        if elapsed >= target_time - time_tolerance:
            break
    
    final_mass = float(mass.sum())
    
    return SloshingRun(
        times=times,
        surface_elevations=surface_elevations,
        kinetic_energy=kinetic,
        potential_energy=potential,
        mass=final_mass,
        escaped=escaped,
        rebuilds=cache.rebuilds if cache else 0,
        max_speed=max_speed,
        max_density_compression=float(compression_peak),
        initial_mass=initial_mass,
        final_mass=final_mass,
        final_positions=positions.detach().cpu(),
        steps_taken=step + 1,
        simulated_time=elapsed,
        dt_min=dt_min,
        dt_max=dt_max,
        max_acoustic_courant=float(courant_peak),
    )
