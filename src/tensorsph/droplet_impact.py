"""Small, transparent 2-D WCSPH droplet-impact prototype (PyTorch).

Physical scope: weakly-compressible SPH with density summation, a Tait equation
of state, a symmetric pressure-force form, regularized laminar viscosity,
gravity, and static wall particles. Execution success is not evidence of
physical accuracy; see ``docs/benchmarks/`` for what has and has not been
verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import platform
import subprocess

import torch

from .neighbor_search import PairListCache, cell_linked_pairs


__version__ = "0.4.0"

# The Morris et al. (1997) discrete Laplacian carries a factor 2:
#   laplacian(v)_i ~= 2 sum_j (m_j / rho_j) (v_i - v_j) . (-r_ij . grad_i W) / (r^2 + eta^2)
# Without it the operator applies exactly half the requested kinematic
# viscosity. That was measured on a smooth sinusoidal field at dx = 0.04, 0.02,
# 0.01 and h/dx = 1.1..1.7: the ratio of applied to analytic Laplacian was
# 0.500 to three decimals in every case, i.e. a normalization constant rather
# than a discretization error. test_viscosity_is_calibrated pins this.
VISCOUS_LAPLACIAN_FACTOR = 2.0


class SimulationUnstableError(FloatingPointError):
    """Raised when a run violates one of its numerical acceptance criteria."""


@dataclass
class Config:
    """Case configuration.

    ``dt`` defaults to a value that satisfies the acoustic Courant limit
    ``dt * c0 / h <= max_courant`` with roughly 20% of margin. ``run`` rejects
    configurations that exceed the limit instead of silently producing an
    unstable run.

    ``hydrostatic_init`` assigns particle masses from the analytical hydrostatic
    density profile instead of a uniform ``rho0 dx^2``. It makes the initial
    state a discrete equilibrium (the pressure gradient cancels gravity) at the
    cost of a slightly larger total mass, because a resting column is compressed.
    It is off by default: measured on B1 it does not tighten the time-averaged
    hydrostatic profile, only the first instant. See
    docs/benchmarks/hydrostatic.md.

    ``wall_friction`` blends the wall's tangential condition: 0 gives free slip
    (viscosity acts between fluid particles only) and 1 gives a no-slip wall
    (viscosity also acts between fluid and wall particles, whose velocity is
    zero). It does nothing without wall particles.

    ``pair_skin_factor`` scales the Verlet skin used to reuse neighbour lists:
    the skin is ``pair_skin_factor * h``, and the list is rebuilt once any
    particle has moved half a skin. Set it to zero to rebuild pairs every step.
    Reuse does not change the result, only the cost: the default 0.25 was chosen
    by measurement on the impact case (see docs/benchmarks/neighbor-search.md).
    """

    dx: float = 0.025
    h_factor: float = 1.3
    rho0: float = 1000.0
    sound_speed: float = 20.0
    gamma: float = 7.0
    gravity: float = 9.81
    viscosity: float = 0.01
    dt: float = 4.0e-4
    steps: int = 750
    output_every: int = 50
    pool_width: float = 1.0
    pool_depth: float = 0.35
    drop_radius: float = 0.10
    drop_center_y: float = 0.49
    device: str = "auto"
    boundary: str = "wall"
    wall_layers: int = 3
    max_courant: float = 0.3
    max_density_error: float = 0.05
    max_mach: float = 0.10
    fail_fast: bool = True
    pair_skin_factor: float = 0.25
    hydrostatic_init: bool = False
    wall_friction: float = 0.0


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #

def cubic_spline_kernel(r: torch.Tensor, h: float) -> torch.Tensor:
    """2-D cubic spline kernel W(r,h), normalized by 10/(7*pi*h**2)."""
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    first = 1.0 - 1.5 * q.square() + 0.75 * q.pow(3)
    second = 0.25 * (2.0 - q).clamp_min(0).pow(3)
    shape = torch.where(q < 1.0, first, torch.where(q < 2.0, second, 0.0))
    return sigma * shape.clamp_min(0.0)


def cubic_spline_gradient(
    delta: torch.Tensor, h: float, eps: float = 1.0e-12
) -> torch.Tensor:
    """Gradient with respect to x_i for delta_ij = x_i - x_j."""
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    dshape = torch.where(
        q < 1.0,
        -3.0 * q + 2.25 * q.square(),
        torch.where(q < 2.0, -0.75 * (2.0 - q).square(), 0.0),
    )
    scale = sigma * dshape / (h * r.clamp_min(eps))
    return delta * scale.unsqueeze(-1)


def tait_pressure(rho: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Tait equation of state, clamped to non-negative pressure.

    Density summation under-counts neighbours at a free surface, so
    ``rho < rho0`` there. The unclamped Tait law would then return a large
    negative (tensile) pressure, which acts as spurious attraction and drives
    the well-known tensile instability. Clamping at zero pressure is the
    standard first-order free-surface treatment for WCSPH; it is not a
    surface-tension model.
    """
    bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
    ratio = (rho / cfg.rho0).clamp_min(0.01)
    return (bulk * (ratio.pow(cfg.gamma) - 1.0)).clamp_min(0.0)


def acoustic_courant(dt: float, cfg: Config, max_speed: float = 0.0) -> float:
    """Acoustic Courant number dt * (c0 + |v|_max) / h."""
    return dt * (cfg.sound_speed + max_speed) / (cfg.h_factor * cfg.dx)


def suggested_dt(cfg: Config, max_courant: float | None = None) -> float:
    """Largest fixed time step allowed by the acoustic Courant criterion."""
    limit = cfg.max_courant if max_courant is None else max_courant
    return limit * cfg.h_factor * cfg.dx / cfg.sound_speed


# --------------------------------------------------------------------------- #
# Case setup
# --------------------------------------------------------------------------- #

def domain_top(cfg: Config) -> float:
    """Vertical extent of the computational box (free-surface headroom)."""
    return cfg.drop_center_y + cfg.drop_radius + 2.0 * cfg.dx


def hydrostatic_density(y: torch.Tensor, cfg: Config, surface: float | None = None
                        ) -> torch.Tensor:
    """Density of a resting column at height ``y``, from inverting the Tait law.

    ``p(y) = rho0 g (H - y)`` with ``p = B ((rho/rho0)^gamma - 1)`` gives
    ``rho(y) = rho0 (1 + rho0 g (H - y) / B)^(1/gamma)``.
    """
    bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
    top = cfg.pool_depth if surface is None else surface
    depth = (top - y).clamp_min(0.0)
    return cfg.rho0 * (1.0 + cfg.rho0 * cfg.gravity * depth / bulk).pow(1.0 / cfg.gamma)


def make_particles(cfg: Config, device: torch.device):
    """Generate a rectangular pool and a circular drop on a regular lattice.

    The arrangement is deterministic: the solver contains no stochastic step,
    so no random seed is required. Returns ``(positions, mass)``.

    With ``cfg.hydrostatic_init`` the pool masses follow the analytical
    hydrostatic density profile; the drop is in free fall and keeps ``rho0``.
    """
    dx = cfg.dx
    if cfg.drop_center_y - cfg.drop_radius <= cfg.pool_depth:
        raise ValueError("drop must start above the pool surface")
    xs = torch.arange(dx * 0.5, cfg.pool_width, dx, device=device)
    ys = torch.arange(dx * 0.5, cfg.pool_depth, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pool = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)

    drop_y = torch.arange(cfg.drop_center_y - cfg.drop_radius,
                          cfg.drop_center_y + cfg.drop_radius, dx, device=device)
    if drop_y.numel() == 0:
        raise ValueError("drop_radius is too small for the particle spacing")
    dxg, dyg = torch.meshgrid(xs, drop_y, indexing="xy")
    candidate = torch.stack((dxg.reshape(-1), dyg.reshape(-1)), dim=-1)
    center = torch.tensor([cfg.pool_width * 0.5, cfg.drop_center_y], device=device)
    drop = candidate[torch.linalg.vector_norm(candidate - center, dim=-1) <= cfg.drop_radius]
    if drop.shape[0] == 0:
        raise ValueError("drop discretization produced no particles")
    positions = torch.cat((pool, drop), dim=0)
    mass = torch.full((positions.shape[0],), cfg.rho0 * dx * dx, device=device)
    if cfg.hydrostatic_init:
        pool_mass = (hydrostatic_density(pool[:, 1], cfg) * dx * dx)
        mass = torch.cat((pool_mass, mass[pool.shape[0]:]), dim=0)
    return positions, mass


def wall_lattice(cfg: Config, device: torch.device,
                 top: float | None = None) -> torch.Tensor:
    """Static dummy wall particles on the bottom and both sides.

    Three layers with the fluid lattice spacing, placed half a spacing outside
    the fluid domain and extended around the corners. Wall particles use an
    extrapolated pressure and a Tait-consistent density (Adami et al., 2012,
    *A generalized wall boundary condition for smoothed particle
    hydrodynamics*). They restore the kernel support that a fluid particle
    misses next to a wall, which is what makes the density summation usable
    near boundaries.
    """
    if cfg.wall_layers < 1:
        raise ValueError("wall_layers must be positive")
    dx, layers = cfg.dx, cfg.wall_layers
    top = domain_top(cfg) if top is None else float(top)
    span_x = torch.arange(-(layers - 0.5) * dx,
                          cfg.pool_width + (layers - 0.5) * dx, dx, device=device)
    columns = torch.arange(dx * 0.5, top, dx, device=device)
    parts = []
    for layer in range(1, layers + 1):
        y = torch.full_like(span_x[:1], dx * 0.5 - layer * dx)
        gx, gy = torch.meshgrid(span_x, y, indexing="xy")
        parts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1))
    for layer in range(1, layers + 1):
        for x in (dx * 0.5 - layer * dx, cfg.pool_width - dx * 0.5 + layer * dx):
            y = columns
            gx, gy = torch.meshgrid(torch.full_like(y[:1], x), y, indexing="xy")
            parts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1))
    return torch.cat(parts, dim=0)


def wall_mass(cfg: Config, count: int, device: torch.device) -> torch.Tensor:
    return torch.full((count,), cfg.rho0 * cfg.dx * cfg.dx, device=device)


# --------------------------------------------------------------------------- #
# Right-hand side
# --------------------------------------------------------------------------- #

def density_from_neighbor_pairs(pos, mass, h: float, pairs=None):
    """Density summation over unordered pairs, including each particle's self term."""
    if pos.ndim != 2 or mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
        raise ValueError("pos must be (N, D) and mass must be (N,)")
    if pos.device != mass.device:
        raise ValueError("pos and mass must be on the same device")
    if not pos.is_floating_point() or not mass.is_floating_point():
        raise TypeError("pos and mass must be floating-point tensors")
    if not math.isfinite(float(h)) or h <= 0:
        raise ValueError("h must be finite and positive")
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be a long tensor with shape (pair_count, 2)")
    if pairs.device != pos.device:
        raise ValueError("pairs and pos must be on the same device")

    n = pos.shape[0]
    zero = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    rho = mass * cubic_spline_kernel(zero, h)
    if pairs.numel() == 0:
        return rho
    i, j = pairs.unbind(dim=1)
    rij = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    wij = cubic_spline_kernel(rij, h)
    rho = rho.index_add(0, i, mass[j] * wij)
    rho = rho.index_add(0, j, mass[i] * wij)
    return rho


def rhs(pos, vel, mass, cfg: Config, pairs=None, wall=None):
    """Compute WCSPH RHS with pair-based compact-support interactions.

    ``wall`` is an optional ``(W, D)`` tensor of static wall particles. When it
    is given, wall particles contribute to the fluid density summation, their
    pressure is extrapolated from neighbouring fluid pressure, and they exert
    pressure forces on the fluid. Viscosity acts between fluid particles and,
    when ``cfg.wall_friction > 0``, also between fluid particles and the static
    wall particles (whose velocity is zero), which imposes a no-slip tangential
    condition.

    Returns ``(rho, pressure, acceleration)`` for the fluid particles.
    """
    h = cfg.h_factor * cfg.dx
    n = pos.shape[0]
    if wall is None:
        all_pos, all_mass, nw = pos, mass, 0
    else:
        if wall.ndim != 2 or wall.shape[1] != pos.shape[1]:
            raise ValueError("wall must be (W, D) with the same D as pos")
        if wall.device != pos.device or wall.dtype != pos.dtype:
            raise ValueError("wall must share pos device and dtype")
        nw = wall.shape[0]
        all_pos = torch.cat((pos, wall), dim=0)
        all_mass = torch.cat((mass, wall_mass(cfg, nw, pos.device)), dim=0)

    if pairs is None:
        pairs = cell_linked_pairs(all_pos, 2.0 * h)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be a long tensor with shape (pair_count, 2)")
    if pairs.device != pos.device:
        raise ValueError("pairs and pos must be on the same device")

    zero = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    rho_all = torch.zeros(n + nw, dtype=pos.dtype, device=pos.device)
    rho_all[:n] = mass * cubic_spline_kernel(zero, h)

    gravity = torch.zeros_like(pos)
    if pos.shape[1] > 1:
        gravity[:, 1] = -cfg.gravity

    if pairs.numel() == 0:
        rho = rho_all[:n].clamp_min(0.1 * cfg.rho0)
        return rho, tait_pressure(rho, cfg), gravity

    i, j = pairs.unbind(dim=1)
    delta = all_pos[i] - all_pos[j]
    distance = torch.linalg.vector_norm(delta, dim=1)
    wij = cubic_spline_kernel(distance, h)
    grad_w = cubic_spline_gradient(delta, h)

    rho_all = rho_all.index_add(0, i, all_mass[j] * wij)
    rho_all = rho_all.index_add(0, j, all_mass[i] * wij)
    rho = rho_all[:n].clamp_min(0.1 * cfg.rho0)
    pressure = tait_pressure(rho, cfg)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
        fluid_first = (i < n) & (j >= n)
        wall_first = (i >= n) & (j < n)
        numerator = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
        denominator = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
        numerator = numerator.index_add(0, j[fluid_first] - n,
                                        pressure[i[fluid_first]] * wij[fluid_first])
        denominator = denominator.index_add(0, j[fluid_first] - n, wij[fluid_first])
        numerator = numerator.index_add(0, i[wall_first] - n,
                                        pressure[j[wall_first]] * wij[wall_first])
        denominator = denominator.index_add(0, i[wall_first] - n, wij[wall_first])
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
    wall_friction_tensor = torch.full((), cfg.wall_friction, dtype=pos.dtype,
                                      device=pos.device)
    ones = torch.ones((), dtype=pos.dtype, device=pos.device)

    pressure_term = (pressure_all[i] / density_all[i].square()
                     + pressure_all[j] / density_all[j].square())
    pair_pressure = pressure_term[:, None] * grad_w

    acc = torch.zeros((n + nw, pos.shape[1]), dtype=pos.dtype, device=pos.device)
    acc = acc.index_add(0, i, -all_mass[j, None] * pair_pressure)
    acc = acc.index_add(0, j, all_mass[i, None] * pair_pressure)
    acc_pressure = acc[:n]

    fluid_pair = (i < n) & (j < n)
    if nw and cfg.wall_friction > 0.0:
        # One side is fluid and the other is wall: extend the viscous stencil
        # into the wall, with the wall at rest. This is the mirror/image
        # treatment of a no-slip boundary in a particle method.
        wall_side = ((i < n) & (j >= n)) | ((i >= n) & (j < n))
        visc_pair = fluid_pair | wall_side
        blend = torch.where(wall_side, wall_friction_tensor, ones)[visc_pair][:, None]
    else:
        visc_pair = fluid_pair
        blend = 1.0
    dv = all_vel[i[visc_pair]] - all_vel[j[visc_pair]]
    pair_distance = distance[visc_pair]
    denom = pair_distance.square() + 0.01 * h * h
    laplace_weight = torch.relu(
        -(delta[visc_pair] * grad_w[visc_pair]).sum(dim=1)) / denom
    coefficient = VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity * blend
    visc_i = (-coefficient * dv
              * (all_mass[j[visc_pair]] / density_all[j[visc_pair]]
                 * laplace_weight)[:, None])
    visc_j = (coefficient * dv
              * (all_mass[i[visc_pair]] / density_all[i[visc_pair]]
                 * laplace_weight)[:, None])
    acc_visc = torch.zeros((n + nw, pos.shape[1]), dtype=pos.dtype,
                           device=pos.device)
    acc_visc = acc_visc.index_add(0, i[visc_pair], visc_i)
    acc_visc = acc_visc.index_add(0, j[visc_pair], visc_j)
    return rho, pressure, acc_pressure + acc_visc[:n] + gravity


def rhs_all_pairs(pos, vel, mass, cfg: Config):
    """Dense O(N^2) reference implementation used for regression comparisons.

    This is an independent formulation of the same physics with full pairwise
    matrices. It does not support wall particles; it exists so that changes to
    the production (neighbour-pair) path can be checked against a second
    implementation.
    """
    h = cfg.h_factor * cfg.dx
    delta = pos[:, None, :] - pos[None, :, :]
    r = torch.linalg.vector_norm(delta, dim=-1)
    w = cubic_spline_kernel(r, h)
    rho = (w * mass[None, :]).sum(dim=1).clamp_min(0.1 * cfg.rho0)
    pressure = tait_pressure(rho, cfg)
    grad_w = cubic_spline_gradient(delta, h)
    pressure_term = (
        pressure[:, None] / rho[:, None].square()
        + pressure[None, :] / rho[None, :].square()
    )
    acc_pressure = -(mass[None, :, None] * pressure_term[:, :, None] * grad_w).sum(dim=1)

    # Symmetric, regularized laminar-viscosity approximation.
    dv = vel[:, None, :] - vel[None, :, :]
    denom = r.square() + 0.01 * h * h
    laplace_weight = torch.relu(-(delta * grad_w).sum(dim=-1)) / denom
    mu_weight = (mass[None, :] / rho[None, :]) * laplace_weight
    acc_visc = -(VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity
                 * (dv * mu_weight[:, :, None]).sum(dim=1))
    gravity = torch.zeros_like(pos)
    if pos.shape[1] > 1:
        gravity[:, 1] = -cfg.gravity
    return rho, pressure, acc_pressure + acc_visc + gravity


# --------------------------------------------------------------------------- #
# Boundary handling
# --------------------------------------------------------------------------- #

def boundary_box(cfg: Config, top: float | None = None):
    """Physical extents of the fluid box as ``(x_min, x_max, y_min, y_max)``."""
    height = domain_top(cfg) if top is None else float(top)
    if cfg.boundary == "wall":
        return 0.0, cfg.pool_width, 0.0, height
    margin = 0.5 * cfg.dx
    return margin, cfg.pool_width - margin, margin, height


def apply_boundary(pos, vel, cfg: Config, top: float | None = None) -> int:
    """Keep particles inside the box and count crossings.

    The clamp is a safety net, not a boundary condition. With
    ``boundary="wall"`` the wall particles are expected to hold the fluid, so a
    non-zero return value means the wall model was bypassed. The normal
    velocity component is zeroed on contact, which avoids injecting energy
    into the fluid; tangential motion is left free (free slip).
    """
    x_min, x_max, y_min, y_max = boundary_box(cfg, top)
    tolerance = 0.25 * cfg.dx  # small excursions are normal wall repulsion
    low_x = pos[:, 0] < x_min - tolerance
    high_x = pos[:, 0] > x_max + tolerance
    low_y = pos[:, 1] < y_min - tolerance
    high_y = pos[:, 1] > y_max + tolerance
    pos[:, 0] = pos[:, 0].clamp(x_min, x_max)
    pos[:, 1] = pos[:, 1].clamp(y_min, y_max)
    vel[:, 0] = torch.where(low_x | high_x, torch.zeros_like(vel[:, 0]), vel[:, 0])
    vel[:, 1] = torch.where(low_y | high_y, torch.zeros_like(vel[:, 1]), vel[:, 1])
    return int((low_x | high_x | low_y | high_y).sum().item())


# --------------------------------------------------------------------------- #
# Run orchestration
# --------------------------------------------------------------------------- #

def validate_config(cfg: Config) -> None:
    if cfg.dx <= 0 or cfg.dt <= 0 or cfg.h_factor <= 0:
        raise ValueError("dx, dt, and h_factor must be positive")
    if cfg.rho0 <= 0 or cfg.sound_speed <= 0 or cfg.gamma <= 0:
        raise ValueError("rho0, sound_speed, and gamma must be positive")
    if cfg.steps < 0 or cfg.output_every <= 0:
        raise ValueError("steps must be non-negative and output_every must be positive")
    if cfg.pool_width <= 0 or cfg.pool_depth <= 0 or cfg.drop_radius <= 0:
        raise ValueError("pool dimensions and drop_radius must be positive")
    if cfg.boundary not in ("wall", "clamp"):
        raise ValueError("boundary must be 'wall' or 'clamp'")
    if not 0.0 < cfg.max_courant <= 1.0:
        raise ValueError("max_courant must be in (0, 1]")
    if cfg.max_density_error <= 0 or cfg.max_mach <= 0:
        raise ValueError("max_density_error and max_mach must be positive")
    if cfg.pair_skin_factor < 0:
        raise ValueError("pair_skin_factor must be non-negative")
    if not 0.0 <= cfg.wall_friction <= 1.0:
        raise ValueError("wall_friction must be in [0, 1]")
    courant = acoustic_courant(cfg.dt, cfg)
    if courant > cfg.max_courant * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} gives acoustic Courant {courant:.3f} > max_courant "
            f"{cfg.max_courant:g}; use dt <= {suggested_dt(cfg):.3e} s"
        )


def nominal_fall_time(cfg: Config, pos: torch.Tensor) -> float | None:
    """Free-fall time from the drop's lowest row to the pool surface.

    Derived from the discretised particles rather than the nominal radii, so the
    value matches what the run actually does.
    """
    if pos.numel() == 0 or pos.shape[1] < 2:
        return None
    pool = pos[pos[:, 1] <= cfg.pool_depth]
    drop = pos[pos[:, 1] > cfg.pool_depth]
    if pool.numel() == 0 or drop.numel() == 0:
        return None
    gap = float(drop[:, 1].min() - pool[:, 1].max())
    if gap <= 0.0:
        return 0.0
    return math.sqrt(2.0 * gap / cfg.gravity)


def _source_revision() -> dict | None:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True,
                                timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, check=True,
                                    timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None
    return {"git_commit": commit, "worktree_dirty": dirty}


def evaluate_acceptance(rows, cfg: Config, boundary_events: int) -> list[dict]:
    """Evaluate the run against explicit, recorded numerical criteria."""
    worst_compression = max(row["rho_max"] / cfg.rho0 - 1.0 for row in rows)
    worst_mach = max(row["max_speed"] for row in rows) / cfg.sound_speed
    worst_courant = max(row["courant_number"] for row in rows)
    finite = all(all(math.isfinite(value) for value in row.values()) for row in rows)

    def criterion(identifier, metric, measured, threshold, unit, comparison):
        if comparison == ">=":
            passed = measured >= threshold
        else:
            passed = measured <= threshold
        return {
            "id": identifier, "metric": metric, "measured": measured,
            "threshold": threshold, "unit": unit, "comparison": comparison,
            "result": "pass" if passed else "fail",
        }

    criteria = [
        criterion("finite_state", "all diagnostics finite", 1.0 if finite else 0.0,
                  1.0, "bool", ">="),
        criterion("density_compression", "max(rho/rho0 - 1)", worst_compression,
                  cfg.max_density_error, "1", "<="),
        criterion("mach_number", "max|v| / c0", worst_mach, cfg.max_mach, "1", "<="),
        criterion("acoustic_courant", "max dt(c0+|v|)/h", worst_courant,
                  cfg.max_courant, "1", "<="),
    ]
    escape = criterion("boundary_escape", "particles clamped beyond the box",
                       float(boundary_events), 0.0, "count", "<=")
    if cfg.boundary == "clamp":
        # The coordinate clamp *is* the boundary condition here, so exceeding
        # the box is expected behaviour rather than a failure.
        escape["result"] = "not_evaluated"
    criteria.append(escape)
    return criteria


def _write_manifest(out: Path, cfg: Config, device: torch.device, initial_n: int,
                    rows, criteria, boundary_events: int, wall_count: int,
                    failure: str | None, fall_time: float | None,
                    cache: PairListCache | None = None) -> dict:
    numerical = ("fail" if any(c["result"] == "fail" for c in criteria)
                 else "pass")
    metadata = {
        "case": "droplet_impact_2d_wcsph",
        "solver": {"name": "tensorsph", "version": __version__,
                   "revision": _source_revision()},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "device": str(device)},
        "initial_particle_count": int(initial_n),
        "wall_particle_count": int(wall_count),
        "config": cfg.__dict__,
        "derived": {"smoothing_length": cfg.h_factor * cfg.dx,
                    "suggested_dt": suggested_dt(cfg),
                    "acoustic_courant": acoustic_courant(cfg.dt, cfg),
                    "drop_free_fall_time": fall_time,
                    "simulated_time": cfg.steps * cfg.dt},
        "statuses": {
            "software_execution": "failed" if failure else "success",
            "numerical_verification": numerical,
            "physical_validation": "not_evaluated",
        },
        "acceptance": {"criteria": criteria},
        "performance": {
            "pair_skin": 0.0 if cache is None else cache.skin,
            "pair_rebuilds": 0 if cache is None else cache.rebuilds,
            "pair_reuses": 0 if cache is None else cache.reuses,
            "mean_steps_per_rebuild": (
                None if cache is None or cache.rebuilds == 0
                else (cache.rebuilds + cache.reuses) / cache.rebuilds),
        },
        "diagnostics": {"rows": len(rows),
                        "boundary_clamp_events": int(boundary_events),
                        "first_step": rows[0] if rows else None,
                        "last_step": rows[-1] if rows else None},
        "failure": failure,
        "notes": [
            "Execution success is not evidence of physical accuracy.",
            "No surface-tension model is included; free-surface pressure is "
            "clamped to zero, which suppresses tensile instability but is not "
            "a capillary model.",
            "Walls are static dummy particles (Adami et al., 2012) with free-slip "
            "viscosity. Hydrostatic accuracy is measured in "
            "docs/benchmarks/hydrostatic.md, not assumed.",
            "physical_validation is not_evaluated: no reference experiment is "
            "compared against yet.",
        ],
    }
    text = json.dumps(metadata, indent=2)
    (out / "run_manifest.json").write_text(text)
    return metadata


def run(cfg: Config, output: str | Path):
    """Run the case, write artifacts, and enforce numerical acceptance gates.

    Artifacts (diagnostics CSV, snapshots, run manifest, plot) are written even
    when a criterion fails, so a failed run can be inspected. If
    ``cfg.fail_fast`` is set, a violated criterion raises
    :class:`SimulationUnstableError` after the artifacts are on disk.
    """
    validate_config(cfg)
    device = resolve_device(cfg.device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    pos, mass = make_particles(cfg, device)
    x_min, x_max, y_min, y_max = boundary_box(cfg)
    if (pos[:, 0].min() < x_min or pos[:, 0].max() > x_max
            or pos[:, 1].min() < y_min or pos[:, 1].max() > y_max):
        raise ValueError("initial particle arrangement leaves the computational box")
    wall = wall_lattice(cfg, device) if cfg.boundary == "wall" else None

    h = cfg.h_factor * cfg.dx
    cache = (PairListCache(2.0 * h, cfg.pair_skin_factor * h)
             if cfg.pair_skin_factor > 0 else None)

    vel = torch.zeros_like(pos)
    pos_initial = pos.clone()
    initial_n = pos.shape[0]
    rows: list[dict] = []
    snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    boundary_events = 0
    failure: str | None = None

    for step in range(cfg.steps + 1):
        if cache is None:
            pairs = None
        else:
            pairs, _ = cache.get(torch.cat((pos, wall), dim=0) if wall is not None
                                 else pos)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall, pairs=pairs)
        state_finite = bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()
                            and torch.isfinite(rho).all() and torch.isfinite(acc).all())
        if not state_finite and failure is None:
            failure = f"non-finite state at step {step}"

        if step % cfg.output_every == 0 or step == cfg.steps:
            speed = torch.linalg.vector_norm(vel, dim=-1)
            accel = torch.linalg.vector_norm(acc, dim=-1)
            momentum = (mass[:, None] * vel).sum(dim=0)
            max_speed = float(speed.max()) if speed.numel() else 0.0
            rows.append({
                "step": step, "time": step * cfg.dt,
                "particle_count": int(pos.shape[0]),
                "mass_total": float(mass.sum()),
                "momentum_x": float(momentum[0]),
                "momentum_y": float(momentum[1]) if momentum.numel() > 1 else 0.0,
                "rho_min": float(rho.min()), "rho_max": float(rho.max()),
                "rho_mean": float(rho.mean()),
                "rho_max_relative_error": float((rho / cfg.rho0 - 1.0).abs().max()),
                "pressure_min": float(pressure.min()),
                "pressure_max": float(pressure.max()),
                "max_speed": max_speed,
                "max_acceleration": float(accel.max()) if accel.numel() else 0.0,
                "mach_number": max_speed / cfg.sound_speed,
                "courant_number": acoustic_courant(cfg.dt, cfg, max_speed),
                "kinetic_energy": float(0.5 * (mass[:, None] * vel.square()).sum()),
                "y_min": float(pos[:, 1].min()), "y_max": float(pos[:, 1].max()),
            })
            snapshots.append((step, pos.detach().clone(), rho.detach().clone()))

        if step == cfg.steps or not state_finite:
            break
        # Semi-implicit Euler with explicit boundary handling.
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        boundary_events += apply_boundary(pos, vel, cfg)

    criteria = evaluate_acceptance(rows, cfg, boundary_events)
    _write_manifest(out, cfg, device, initial_n, rows, criteria, boundary_events,
                    0 if wall is None else wall.shape[0], failure,
                    nominal_fall_time(cfg, pos_initial), cache)

    with (out / "diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for step, xyz, density in snapshots:
        torch.save({"step": step, "time": step * cfg.dt, "position": xyz,
                    "density": density, "boundary": cfg.boundary},
                   out / f"particles_{step:06d}.pt")
    _plot(out, cfg, snapshots)

    if cfg.fail_fast:
        failed = [c for c in criteria if c["result"] == "fail"]
        if failure or failed:
            detail = failure or ", ".join(
                f"{c['id']}={c['measured']:.4g} (limit {c['threshold']:g})"
                for c in failed)
            raise SimulationUnstableError(
                f"numerical acceptance failed: {detail}; artifacts in {out}")
    return rows


def _plot(out: Path, cfg: Config, snapshots) -> None:
    if len(snapshots) > 12:  # keep the panel width readable for long runs
        stride = math.ceil(len(snapshots) / 12)
        snapshots = snapshots[::stride]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    _, _, _, y_max = boundary_box(cfg)
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4),
                             squeeze=False)
    sc = None
    for ax, (step, xyz, density) in zip(axes[0], snapshots):
        sc = ax.scatter(xyz[:, 0].tolist(), xyz[:, 1].tolist(),
                        c=density.tolist(), s=5, cmap="viridis",
                        vmin=0.9 * cfg.rho0, vmax=1.05 * cfg.rho0)
        ax.set_title(f"t={step * cfg.dt:.4f} s")
        ax.set_xlim(0, cfg.pool_width)
        ax.set_ylim(0, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="density (kg/m³)")
    fig.savefig(out / "particle_snapshots.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

