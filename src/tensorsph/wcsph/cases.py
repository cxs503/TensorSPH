"""Benchmark cases for the WCSPH verification chain.

Four rungs, each a self-contained initial/boundary value problem:

``particle_pair``
    Two particles of equal mass fly at each other in free space (no gravity,
    no boundary).  The pressure force is the only interaction, so the run
    measures the two discrete invariants the scheme is supposed to have:
    exact total-momentum conservation and bounded energy error.

``dam_break_2d``
    Martin & Moyce (1952) geometry: a column of width ``a`` and height ``2a``
    released against a dry bed.  The leading edge is tracked both as the
    furthest particle (``front_max``, the historical definition) and as the
    furthest particle of the *bed layer* (``front_toe``), which is what the
    experiment measured -- a flying droplet can only inflate the former.

``dam_break_3d``
    the same column, now of finite width, so the front feels the side walls.

``sloshing_2d``
    a rectangular tank of length ``L`` and depth ``H`` forced horizontally in
    its own frame with ``a_x(t) = -A omega^2 sin(omega t)`` (the non-inertial
    formulation: walls are at rest and the fluid carries the apparent body
    force).  Wall pressure is read from the dynamic boundary particles, which
    is exactly the physical quantity a pressure sensor in the tank records.

All cases are deterministic: no random numbers, no fitted constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from . import kernels
from .solver import (Solver, WCSPHConfig, stable_dt, tait_pressure)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def lattice_box(lower, upper, dx: float, dim: int) -> torch.Tensor:
    """Cell-centred lattice filling ``[lower, upper]`` (last cell may be short)."""
    axes = []
    for lo, hi in zip(lower, upper):
        n = int(math.floor((hi - lo) / dx + 1.0e-9))
        axes.append(lo + dx * (torch.arange(n, dtype=torch.float64) + 0.5))
    if dim == 2:
        gx, gy = torch.meshgrid(axes[0], axes[1], indexing="xy")
        return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1)


def box_walls(lower, upper, dx: float, dim: int, layers: int = 3) -> torch.Tensor:
    """Static boundary lattice on the walls of a box (``layers`` rows thick).

    Layers are placed half a spacing outside the fluid domain, as in
    ``tensorsph.droplet_impact.wall_lattice``.
    """
    lower = list(lower)
    upper = list(upper)
    parts = []
    for axis in range(dim):
        other = [a for a in range(dim) if a != axis]
        for side, sign in ((lower[axis], -1.0), (upper[axis], +1.0)):
            for layer in range(1, layers + 1):
                coord = side + sign * (layer - 0.5) * dx
                ranges = []
                for a in range(dim):
                    if a == axis:
                        ranges.append(torch.tensor([coord], dtype=torch.float64))
                    else:
                        lo = lower[a] - (layers - 0.5) * dx
                        hi = upper[a] + (layers - 0.5) * dx
                        n = int(math.floor((hi - lo) / dx + 1.0e-9))
                        ranges.append(lo + dx * (torch.arange(n, dtype=torch.float64)
                                                 + 0.5))
                if dim == 2:
                    g0, g1 = torch.meshgrid(ranges[0], ranges[1], indexing="xy")
                    parts.append(torch.stack((g0.reshape(-1), g1.reshape(-1)), -1))
                else:
                    g0, g1, g2 = torch.meshgrid(ranges[0], ranges[1], ranges[2],
                                                indexing="xy")
                    parts.append(torch.stack((g0.reshape(-1), g1.reshape(-1),
                                              g2.reshape(-1)), -1))
        del other
    # the per-face lattices overlap along the box edges and corners, and the
    # two faces compute the shared coordinate from different expressions, so
    # the duplicates differ in the last bits: quantise to a fraction of the
    # spacing before taking the union, otherwise the leftover twin adds its
    # mass twice to every fluid density sum within its support
    stacked = torch.cat(parts, dim=0)
    key = torch.round(stacked / dx * 1.0e6)
    _, inverse = torch.unique(key, dim=0, return_inverse=True)
    first = torch.zeros(int(inverse.max()) + 1, dtype=torch.long)
    first[inverse.flip(0)] = torch.arange(inverse.numel() - 1, -1, -1)
    return stacked[first]


def front_positions(pos: torch.Tensor, dx: float, bed_rows: float = 1.5) -> dict:
    """Leading-edge x of the water: all particles and the bed layer only."""
    if pos.numel() == 0:
        return {"front_max": 0.0, "front_toe": 0.0}
    return {
        "front_max": float(pos[:, 0].max()),
        "front_toe": float(pos[pos[:, 1] < bed_rows * dx, 0].max())
        if bool((pos[:, 1] < bed_rows * dx).any()) else float(pos[:, 0].max()),
    }


class Counter:
    """Counts fluid particles that left the tank through a wall."""

    def __init__(self, lower, upper):
        self.lower = torch.tensor(lower, dtype=torch.float64)
        self.upper = torch.tensor(upper, dtype=torch.float64)
        self.count = 0

    def __call__(self, solver: Solver) -> None:
        pos = solver.state.pos
        if pos.numel() == 0:
            return
        out = ((pos < self.lower).any(dim=1) | (pos > self.upper).any(dim=1))
        self.count += int(out.sum())


# --------------------------------------------------------------------------- #
# rung 1: particle pair
# --------------------------------------------------------------------------- #

@dataclass
class PairRun:
    times: list[float]
    momentum: list[torch.Tensor]
    energy: list[float]
    separation: list[float]
    mass: float
    dt: float
    steps: int

    @property
    def momentum_drift(self) -> float:
        p0 = self.momentum[0]
        return float(max((p - p0).abs().max() for p in self.momentum)
                     / p0.abs().max().clamp_min(1.0e-300))

    @property
    def energy_drift(self) -> float:
        e0 = self.energy[0]
        return float(abs(max(self.energy) - e0) / abs(e0))


def particle_pair(v0: float = 0.5, c0: float = 20.0, alpha_av: float = 0.0,
                  separation: float = 1.0, dx: float = 0.01,
                  steps: int = 4000, dim: int = 2,
                  kernel: str = "quintic", cfl: float = 0.25) -> PairRun:
    """Two equal particles approach head-on; measure momentum and energy.

    The pair mass is calibrated so that the summation density of the isolated
    pair equals ``rho0`` at the initial separation -- without that the Tait
    pressure would clamp at zero and the particles would never interact.
    """
    cfg = WCSPHConfig(dim=dim, dx=dx, c0=c0, alpha_av=alpha_av, gravity=0.0,
                      boundary="none", kernel=kernel, cfl=cfl)
    cfg.validate()
    r0 = separation * cfg.h0
    axis = torch.zeros(dim, dtype=torch.float64)
    axis[0] = 1.0
    pos = torch.stack((-0.5 * r0 * axis, 0.5 * r0 * axis))
    vel = torch.stack((v0 * axis, -v0 * axis))
    w_sum = float(kernels.kernel_value(
        torch.tensor([0.0, r0], dtype=torch.float64), float(cfg.h0), dim,
        kernel).sum())
    m = torch.full((2,), cfg.rho0 / w_sum)
    solver = Solver(cfg, pos, vel, m)
    # equilibrium check: the initial density must be rho0, else the pressures
    # are clamped and the collision is not the one being measured
    solver.force(0.0)
    if abs(float(solver.state.rho[0]) / cfg.rho0 - 1.0) > 1.0e-9:
        raise RuntimeError("pair mass calibration failed")

    times, mom, en, sep = [0.0], [solver.momentum().clone()], [solver.total_energy()], [
        float(torch.linalg.vector_norm(pos[0] - pos[1]))]
    for _ in range(steps):
        dt = stable_dt(cfg, solver.state.max_speed)
        solver.step(dt)
        times.append(solver.state.time)
        mom.append(solver.momentum().clone())
        en.append(solver.total_energy())
        sep.append(float(torch.linalg.vector_norm(
            solver.state.pos[0] - solver.state.pos[1])))
    return PairRun(times, mom, en, sep, float(m.sum()), dt, steps)


# --------------------------------------------------------------------------- #
# rung 2/3: dam break
# --------------------------------------------------------------------------- #

@dataclass
class DamBreakConfig:
    column_width: float = 0.2
    column_height: float = 0.4
    tank_length: float = 2.0
    tank_height: float = 0.6
    width: float = 0.2            # 3-D only
    dx: float = 0.01
    c0: float = 25.0
    alpha_av: float = 0.1
    h_factor: float = 1.3
    kernel: str = "quintic"
    t_end: float = 0.5
    sample_times: tuple[float, ...] = (0.1, 0.2, 0.3)
    profile_bins: int = 200
    wall_layers: int = 3
    h_adaptive: bool = False

    @property
    def dim(self) -> int:
        return 3 if self.width > 0 and self.width != 0.2 else 2


@dataclass
class DamBreakRun:
    times: list[float]
    front_max: list[float]
    front_toe: list[float]
    kinetic: list[float]
    potential: list[float]
    mass: float
    escaped: int
    steps: int
    simulated_time: float
    dt: float
    max_speed: float
    max_compression: float
    max_courant: float
    particles: int
    profiles: dict = field(default_factory=dict)   # time -> depth profile
    profile_edges: list = field(default_factory=list)


def depth_profile(pos: torch.Tensor, mass: torch.Tensor, rho: torch.Tensor,
                  length: float, bins: int = 200, dim: int = 2,
                  span: float | None = None) -> list:
    """Volume per unit width (2-D) / per unit area (3-D) in each x bin.

    ``h(x) = sum_{i in bin} (m_i / rho_i) / (dx_bin * width)`` -- a smooth,
    particle-order-independent free-surface measure that two different
    resolutions can be compared on without matching particles.
    """
    edges = torch.linspace(0.0, length, bins + 1, dtype=torch.float64)
    idx = torch.bucketize(pos[:, 0].contiguous(), edges) - 1
    idx = idx.clamp(0, bins - 1)
    acc = torch.zeros(bins, dtype=torch.float64).index_add(0, idx, mass / rho)
    width = length / bins
    if dim == 3:
        extent = float((pos[:, 2].max() - pos[:, 2].min()).clamp_min(1.0e-9)) \
            if span is None else float(span)
        width = width * extent
    return (acc / width).tolist()


def dam_break(cfg: DamBreakConfig, dim: int = 2,
              wall_clock_limit: float | None = None) -> DamBreakRun:
    """Integrate the dam-break case and track the leading edge."""
    import time as _time

    lower = [0.0, 0.0] + ([0.0] if dim == 3 else [])
    upper = ([cfg.tank_length, cfg.tank_height, cfg.width] if dim == 3
             else [cfg.tank_length, cfg.tank_height])
    fluid_lo = [0.0, 0.0] + ([0.0] if dim == 3 else [])
    fluid_hi = ([cfg.column_width, cfg.column_height, cfg.width] if dim == 3
                else [cfg.column_width, cfg.column_height])
    pos = lattice_box(fluid_lo, fluid_hi, cfg.dx, dim)
    wall = box_walls(lower, upper, cfg.dx, dim, cfg.wall_layers)
    s_cfg = WCSPHConfig(dim=dim, dx=cfg.dx, h_factor=cfg.h_factor,
                        kernel=cfg.kernel, c0=cfg.c0, alpha_av=cfg.alpha_av,
                        gravity=9.81, h_adaptive=cfg.h_adaptive)
    mass = torch.full((pos.shape[0],), s_cfg.rho0 * cfg.dx ** dim)
    vel = torch.zeros_like(pos)
    solver = Solver(s_cfg, pos, vel, mass, wall_pos=wall,
                    box=((lower[0], upper[0], lower[1], upper[1])
                         if dim == 2 else None),
                    box3=((lower[0], upper[0], lower[1], upper[1],
                           lower[2], upper[2]) if dim == 3 else None))
    solver.apply_hydrostatic_mass(cfg.column_height)
    counter = Counter(lower, upper)

    times, fmax, ftoe, ke, pe = [0.0], [0.0], [0.0], [0.0], [
        float(solver.potential_energy())]
    fronts = front_positions(solver.state.pos, cfg.dx)
    fmax[0], ftoe[0] = fronts["front_max"], fronts["front_toe"]
    profiles: dict = {}
    pending = sorted(cfg.sample_times)
    start = _time.time()
    while solver.state.time < cfg.t_end - 1.0e-15:
        dt = solver.step()
        counter(solver)
        times.append(solver.state.time)
        fronts = front_positions(solver.state.pos, cfg.dx)
        fmax.append(fronts["front_max"])
        ftoe.append(fronts["front_toe"])
        ke.append(solver.kinetic_energy())
        pe.append(solver.potential_energy())
        while pending and solver.state.time >= pending[0] - 1.0e-15:
            t_prof = pending.pop(0)
            profiles[t_prof] = depth_profile(
                solver.state.pos, solver.state.mass, solver.state.rho,
                cfg.tank_length, cfg.profile_bins, dim,
                span=cfg.width if dim == 3 else None)
        if wall_clock_limit and _time.time() - start > wall_clock_limit:
            break
        if dt <= 0.0 or not math.isfinite(dt):
            raise RuntimeError("step size collapsed")
    return DamBreakRun(
        times=times, front_max=fmax, front_toe=ftoe, kinetic=ke, potential=pe,
        mass=float(solver.state.mass.sum()), escaped=counter.count,
        steps=solver.state.steps, simulated_time=solver.state.time,
        dt=float(solver.state.time / max(solver.state.steps, 1)),
        max_speed=solver.state.max_speed,
        max_compression=solver.state.max_compression,
        max_courant=solver.state.max_courant,
        particles=int(solver.state.n_fluid), profiles=profiles,
        profile_edges=[0.0, float(cfg.tank_length)])


def front_at(times: list[float], front: list[float], t: float) -> float | None:
    """Linear interpolation of a tracked front at time ``t`` (clamped)."""
    if not times:
        return None
    if t <= times[0]:
        return float(front[0])
    if t >= times[-1]:
        return None
    for k in range(1, len(times)):
        if times[k] >= t:
            f = (t - times[k - 1]) / (times[k] - times[k - 1])
            return float(front[k - 1] + f * (front[k] - front[k - 1]))
    return None


def relative_front_error(run: DamBreakRun, reference, column_width: float,
                         z_of_t=None) -> dict:
    """Compare the tracked front with ``reference(tau) -> Z/a`` at each time.

    ``Z/a`` is measured from the initial column edge: ``(x_front - a) / a``.
    """
    rows = []
    for t in reference:
        x = front_at(run.times, run.front_toe, t)
        if x is None:
            rows.append({"t": t, "reference": None, "sim": None, "error": None})
            continue
        sim = (x - column_width) / column_width
        ref = z_of_t(t)
        rows.append({"t": t, "reference": ref, "sim": sim,
                     "error": sim - ref,
                     "relative_error": (sim - ref) / ref if ref else None})
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# rung 4: sloshing
# --------------------------------------------------------------------------- #

def natural_frequency(length: float, depth: float, gravity: float = 9.81,
                      mode: int = 1) -> float:
    """``omega_n = sqrt(g k tanh(k H))``, ``k = n pi / L`` (linear theory)."""
    k = mode * math.pi / length
    return math.sqrt(gravity * k * math.tanh(k * depth))


@dataclass
class SloshingConfig:
    length: float = 1.0
    depth: float = 0.5
    dx: float = 0.02
    amplitude: float = 0.01
    frequency: float = 1.0
    c0: float = 25.0
    alpha_av: float = 0.1
    h_factor: float = 1.3
    kernel: str = "quintic"
    t_end: float = 8.0
    sample_every: int = 10
    wall_layers: int = 3
    probe_height: float | None = None     # defaults to half the water depth


@dataclass
class SloshingRun:
    times: list[float]
    eta_wall: list[float]            # free-surface elevation at the end wall
    p_wall_max: list[float]          # max pressure over the end-wall particles
    p_wall_probe: list[float]        # pressure at the probe height
    eta_probe: float
    steps: int
    simulated_time: float
    dt: float
    max_speed: float
    max_compression: float
    max_courant: float
    particles: int
    escaped: int
    periodic: bool = True

    def peak(self, series) -> float:
        return float(max(series)) if series else 0.0

    def amplitude(self, series) -> float:
        return 0.5 * (max(series) - min(series)) if series else 0.0


def sloshing(cfg: SloshingConfig) -> SloshingRun:
    """Forced 2-D tank sloshing; returns the end-wall pressure history."""
    dim = 2
    lower = [0.0, 0.0]
    upper = [cfg.length, cfg.depth]
    pos = lattice_box([0.0, 0.0], [cfg.length, cfg.depth], cfg.dx, dim)
    wall = box_walls(lower, upper, cfg.dx, dim, cfg.wall_layers)
    omega = 2.0 * math.pi * cfg.frequency
    s_cfg = WCSPHConfig(dim=dim, dx=cfg.dx, h_factor=cfg.h_factor,
                        kernel=cfg.kernel, c0=cfg.c0, alpha_av=cfg.alpha_av,
                        gravity=9.81, excitation_amplitude=cfg.amplitude,
                        excitation_omega=omega)
    mass = torch.full((pos.shape[0],), s_cfg.rho0 * cfg.dx ** dim)
    solver = Solver(s_cfg, pos, torch.zeros_like(pos), mass, wall_pos=wall,
                    box=(lower[0], upper[0], lower[1], upper[1]))
    counter = Counter(lower, upper)

    wall_pos = solver.state.wall_pos
    end_wall = wall_pos[:, 0] > cfg.length            # right wall particles
    probe_h = 0.5 * cfg.depth if cfg.probe_height is None else cfg.probe_height
    if bool(end_wall.any()):
        idx = torch.nonzero(end_wall, as_tuple=False).reshape(-1)
        probe_idx = idx[(wall_pos[idx, 1] - probe_h).abs().argmin()]
    else:  # pragma: no cover - geometry always has an end wall
        probe_idx = None

    times: list[float] = []
    eta_wall: list[float] = []
    p_max: list[float] = []
    p_probe: list[float] = []
    tol = 3.0 * cfg.dx
    while solver.state.time < cfg.t_end - 1.0e-15:
        solver.step()
        counter(solver)
        if solver.state.steps % cfg.sample_every == 0:
            times.append(solver.state.time)
            near_wall = solver.state.pos[:, 0] > cfg.length - tol
            eta_wall.append(float(solver.state.pos[near_wall, 1].max())
                            if bool(near_wall.any()) else 0.0)
            p_wall = solver.state.wall_pressure
            if p_wall is None:
                raise RuntimeError("wall pressure missing: no boundary particles")
            p_max.append(float(p_wall[end_wall].max()))
            p_probe.append(float(p_wall[probe_idx]) if probe_idx is not None else 0.0)
    return SloshingRun(
        times=times, eta_wall=eta_wall, p_wall_max=p_max, p_wall_probe=p_probe,
        eta_probe=float(solver.state.pos[:, 1].max()),
        steps=solver.state.steps, simulated_time=solver.state.time,
        dt=float(solver.state.time / max(solver.state.steps, 1)),
        max_speed=solver.state.max_speed,
        max_compression=solver.state.max_compression,
        max_courant=solver.state.max_courant,
        particles=int(solver.state.n_fluid), escaped=counter.count)


__all__ = [
    "DamBreakConfig", "DamBreakRun", "PairRun", "SloshingConfig", "SloshingRun",
    "box_walls", "dam_break", "front_at", "front_positions", "lattice_box",
    "natural_frequency", "particle_pair", "relative_front_error",
    "sloshing", "tait_pressure", "kernels",
]