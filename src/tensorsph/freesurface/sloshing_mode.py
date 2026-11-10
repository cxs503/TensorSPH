"""First-mode sloshing in a rectangular tank, released from rest.

Physical problem
----------------
A rectangular tank of length ``L`` holds water of depth ``H``. The free surface
is displaced as a pure first mode, ``eta(x, 0) = A cos(pi x / L)``, and released
from rest. Linear potential-flow theory gives the natural frequencies

    omega_n = sqrt(n pi g / L * tanh(n pi H / L))

so the first mode oscillates at ``omega_1`` and, in the inviscid limit, forever.
An SPH tank has numerical dissipation, so the measured record is a decaying
sinusoid: **the frequency is the pass/fail quantity, the decay rate is the
reported quality indicator** (it says how much of the wave energy the
discretisation eats per period, which is what a sloshing load case cares about).

Why released from rest, not forced
----------------------------------
The existing :mod:`tensorsph.sloshing` case applies a horizontal tank
acceleration and looks for the resonance peak; that measures the response of the
whole system and needs a frequency sweep. Releasing a pure mode from rest
measures the same eigenvalue in a single run, with no forcing model, no
wavemaker transfer function and no sweep -- one run, one number, one analytic
reference. The two are complementary: this module supplies the anchor, the
forced case supplies the engineering response curve.

Measurement
-----------
The surface elevation is projected onto the first mode,
``a1(t) = (2/L) integral eta(x,t) cos(pi x / L) dx`` (:func:`tensorsph.freesurface.
profiles.first_mode_amplitude`). A single gauge cannot do this job: the SPH free
surface at one probe scatters by a fraction of the particle spacing, which for a
2 cm wave at 1.25 cm spacing is the amplitude itself. The projection over ~50
columns averages that noise down. The frequency then comes from mean crossings
of ``a1(t)`` and the decay from the peak envelope.

Honest scope
------------
* ``A / H = 0.08`` and ``A / L = 0.03``: small enough for the linear eigenvalue to
  be the reference (the second-order frequency shift is ``O((A/H)^2)``).
* weak compressibility costs ``O(g H / c_s^2)`` of the frequency, which is why the
  case reports that ratio instead of assuming it is negligible;
* the damping measured here is *numerical*, not physical: it depends on the
  kernel, the viscosity setting and the resolution, and it is reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .. import droplet_impact
from ..droplet_impact import Config
from ..neighbor_search import PairListCache
from . import profiles

__all__ = [
    "SloshingModeCase",
    "SloshingModeResult",
    "analytic_omega",
    "make_mode_lattice",
    "run_sloshing_mode",
]

GRAVITY = 9.81


def analytic_omega(tank_length: float, depth: float, mode: int = 1,
                   gravity: float = GRAVITY) -> float:
    """``omega_n = sqrt(n pi g / L tanh(n pi H / L))`` (linear potential flow)."""
    n = int(mode)
    if n < 1:
        raise ValueError("mode number must be >= 1")
    if tank_length <= 0.0 or depth <= 0.0:
        raise ValueError("tank length and depth must be positive")
    k = n * math.pi / tank_length
    return math.sqrt(k * gravity * math.tanh(k * depth))


@dataclass
class SloshingModeCase:
    tank_length: float = 0.60
    water_depth: float = 0.25
    tank_height: float = 0.40
    amplitude: float = 0.02          # surface displacement at the wall, m
    cells_per_depth: float = 20.0
    sound_speed_ratio: float = 80.0  # c_s / sqrt(g H)
    viscosity: float = 1.0e-3
    density_diffusion: float = 0.0
    wall_layers: int = 3
    periods: float = 3.0             # simulated time in first-mode periods
    output_per_period: int = 40
    device: str = "cpu"
    gravity: float = GRAVITY
    label: str = ""

    @property
    def dx(self) -> float:
        return float(self.water_depth / self.cells_per_depth)

    @property
    def omega1(self) -> float:
        return analytic_omega(self.tank_length, self.water_depth, 1, self.gravity)

    @property
    def period1(self) -> float:
        return 2.0 * math.pi / self.omega1

    @property
    def sound_speed(self) -> float:
        return float(self.sound_speed_ratio * math.sqrt(self.gravity * self.water_depth))

    @property
    def compressibility(self) -> float:
        """``g H / c_s^2``: the relative frequency bias to expect from Tait."""
        return float(self.gravity * self.water_depth / self.sound_speed ** 2)

    @property
    def dt(self) -> float:
        return float(0.25 * 1.3 * self.dx / self.sound_speed)

    @property
    def end_time(self) -> float:
        return float(self.periods * self.period1)

    @property
    def steps(self) -> int:
        return int(math.ceil(self.end_time / self.dt))

    def config(self) -> Config:
        cfg = droplet_impact.Config(
            dx=self.dx,
            rho0=1000.0,
            sound_speed=self.sound_speed,
            gravity=self.gravity,
            viscosity=self.viscosity,
            dt=self.dt,
            steps=self.steps,
            output_every=max(1, int(round(self.period1 / self.output_per_period / self.dt))),
            pool_width=self.tank_length,
            pool_depth=self.tank_height,
            wall_layers=self.wall_layers,
            density_diffusion=self.density_diffusion,
            device=self.device,
            hydrostatic_init=True,
        )
        return cfg

    def elevation(self, x: torch.Tensor) -> torch.Tensor:
        """Initial first-mode displacement ``eta(x,0) = A cos(pi x / L)``."""
        return self.amplitude * torch.cos(math.pi * torch.as_tensor(x) / self.tank_length)

    def surface(self, x: torch.Tensor) -> torch.Tensor:
        """Water-surface height ``H + eta(x, 0)`` above the tank floor."""
        return self.water_depth + self.elevation(x)


def make_mode_lattice(cfg: Config, case: SloshingModeCase
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Particle lattice under the displaced free surface, hydrostatic in mass."""
    dx = case.dx
    device = torch.device(cfg.device)
    xs = droplet_impact.cell_centres(case.tank_length, dx, device)
    eta = case.surface(xs)
    positions = []
    masses = []
    for x, surface in zip(xs.tolist(), eta.tolist()):
        if surface <= 1.5 * dx:
            continue                        # column too shallow to hold a particle
        # Cell centres from the floor up to the local surface.
        ys = torch.arange(0.5 * dx, surface, dx, dtype=xs.dtype, device=device)
        if ys.numel() == 0:
            continue
        column = torch.stack((torch.full_like(ys, x), ys), dim=-1)
        positions.append(column)
        masses.append(droplet_impact.hydrostatic_density(ys, cfg, surface=surface) * dx * dx)
    if not positions:
        raise ValueError("the displaced surface produced no particles")
    return torch.cat(positions, dim=0), torch.cat(masses, dim=0)


@dataclass
class SloshingModeResult:
    case: SloshingModeCase
    times: list = field(default_factory=list)
    modal: list = field(default_factory=list)       # a1(t), m
    surface_max: list = field(default_factory=list)  # max elevation, m
    kinetic: list = field(default_factory=list)
    potential: list = field(default_factory=list)
    mass: float = 0.0
    mass_final: float = 0.0
    escaped: int = 0
    steps_taken: int = 0
    simulated_time: float = 0.0
    max_speed: float = 0.0
    max_density_compression: float = 0.0
    wall_clock: float = 0.0
    particle_count: int = 0
    profiles_store: list = field(default_factory=list)

    @property
    def mass_drift(self) -> float:
        return abs(self.mass_final - self.mass) / max(self.mass, 1e-30)

    def summary(self) -> dict:
        """Frequency, damping and the analytic reference, as plain numbers."""
        omega = float("nan")
        decay = float("nan")
        frequency_error = float("nan")
        try:
            omega = profiles.oscillation_frequency(self.times, self.modal)
            frequency_error = abs(omega - self.case.omega1) / self.case.omega1
        except ValueError:
            pass
        decay = profiles.peak_decay_rate(self.times, self.modal)
        amplitude = max((abs(v) for v in self.modal), default=float("nan"))
        return {
            "label": self.case.label,
            "tank_length": self.case.tank_length,
            "water_depth": self.case.water_depth,
            "amplitude": self.case.amplitude,
            "dx": self.case.dx,
            "cells_per_depth": self.case.cells_per_depth,
            "sound_speed": self.case.sound_speed,
            "compressibility": self.case.compressibility,
            "dt": self.case.dt,
            "steps": self.steps_taken,
            "particles": self.particle_count,
            "simulated_time": self.simulated_time,
            "wall_clock": self.wall_clock,
            "omega_analytic": self.case.omega1,
            "omega_measured": omega,
            "frequency_error": frequency_error,
            "modal_amplitude_initial": self.case.amplitude,
            "modal_amplitude_peak": amplitude,
            "decay_rate": decay,
            "decay_per_period": (decay * self.case.period1
                                 if decay == decay else float("nan")),
            "max_speed": self.max_speed,
            "max_density_compression": self.max_density_compression,
            "escaped": self.escaped,
            "mass_drift": self.mass_drift,
            "times": [float(v) for v in self.times],
            "modal": [float(v) for v in self.modal],
        }


def run_sloshing_mode(case: SloshingModeCase) -> SloshingModeResult:
    """Integrate the released first mode and record its modal amplitude."""
    import time as _time

    cfg = case.config()
    device = droplet_impact.resolve_device(cfg.device)
    cfg.device = str(device)
    top = float(case.tank_height)
    positions, mass = make_mode_lattice(cfg, case)
    wall = droplet_impact.wall_lattice(cfg, device, top=top)
    velocity = torch.zeros_like(positions)
    initial_mass = float(mass.sum())

    h = cfg.h_factor * cfg.dx
    fluid_count = positions.shape[0]
    skin = cfg.pair_skin_factor
    cache = PairListCache(2.0 * h, skin * h, fluid_count=fluid_count)
    all_positions = torch.cat((positions, wall), dim=0)

    # one strip per particle column: the modal projection is a Riemann sum,
    # and a strip twice as wide as the particle spacing aliases the
    # short-wavelength (grid-scale) part of the surface into mode 1.
    slab = case.dx
    centres = profiles.slab_centres(0.0, case.tank_length, slab)

    result = SloshingModeResult(case=case, mass=initial_mass,
                                particle_count=int(fluid_count))
    dt = cfg.dt
    elapsed = 0.0
    max_speed = 0.0
    compression_peak = torch.zeros((), dtype=torch.float64, device=device)
    started = _time.perf_counter()
    step = 0
    for step in range(cfg.steps + 1):
        all_positions[:fluid_count] = positions
        pairs, _ = cache.get(all_positions)
        density, _, acc = droplet_impact.rhs(positions, velocity, mass, cfg,
                                             pairs=pairs, wall=wall, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        compression_peak = torch.maximum(compression_peak,
                                         density.max().double() / cfg.rho0 - 1.0)
        speed_now = (torch.linalg.vector_norm(velocity, dim=-1) if velocity.numel()
                     else torch.zeros(0, device=device))
        speed_scalar = float(speed_now.max()) if speed_now.numel() else 0.0
        max_speed = max(max_speed, speed_scalar)

        final = step == cfg.steps
        if step % cfg.output_every == 0 or final:
            eta = _surface_elevation(positions, centres, slab, case.dx)
            # Remove the mode-0 (uniform heave) component: the tank conserves
            # volume, so a uniform offset of the measured surface is pure
            # measurement bias (the topmost-particle gauge is biased by the
            # crest/trough shape), and it would leak into the mode-1 projection
            # as soon as any strip is dry.
            finite = torch.isfinite(eta)
            if bool(finite.any()):
                eta = eta.clone()
                eta[finite] = eta[finite] - eta[finite].mean()
            result.times.append(float(elapsed))
            result.modal.append(profiles.first_mode_amplitude(centres, eta,
                                                              case.tank_length, slab))
            finite_eta = eta[torch.isfinite(eta)]
            result.surface_max.append(float(finite_eta.max()) if finite_eta.numel()
                                      else float("nan"))
            result.kinetic.append(float(0.5 * (mass[:, None] * velocity.square()).sum()))
            result.potential.append(float((mass * cfg.gravity * positions[:, 1]).sum()))
            result.profiles_store.append((float(elapsed), centres.tolist(),
                                          eta.tolist()))
        if final:
            break

        velocity = velocity + dt * acc
        positions = positions + dt * velocity
        result.escaped += int(droplet_impact.apply_boundary(positions, velocity, cfg,
                                                            top=top))
        elapsed += dt
        dt = min(cfg.dt, cfg.max_courant * h / (cfg.sound_speed + speed_scalar))

    result.mass_final = float(mass.sum())
    result.steps_taken = int(step)
    result.simulated_time = float(elapsed)
    result.max_speed = max_speed
    result.max_density_compression = float(compression_peak)
    result.wall_clock = float(_time.perf_counter() - started)
    return result


def _surface_elevation(positions: torch.Tensor, centres: torch.Tensor,
                       slab: float, dx: float) -> torch.Tensor:
    """Topmost particle per strip, corrected by half a spacing to the surface."""
    x = positions[:, 0]
    y = positions[:, 1]
    idx = torch.floor((x - 0.0) / slab).long().clamp(0, centres.numel() - 1)
    eta = torch.full((centres.numel(),), float("nan"), dtype=torch.float64)
    top = torch.full((centres.numel(),), -1.0e30, dtype=torch.float64)
    top.scatter_reduce_(0, idx, y.double(), reduce="amax", include_self=True)
    filled = top > -1.0e29
    eta[filled] = top[filled] + 0.5 * dx
    return eta
