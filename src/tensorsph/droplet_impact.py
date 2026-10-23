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
import warnings

import torch

from .neighbor_search import PairListCache, cell_linked_pairs


__version__ = "0.5.0"

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
    """2-D cubic spline kernel W(r,h), normalized by 10/(7*pi*h**2).

    Valid for two spatial dimensions only: the normalisation ``10/(7 pi h^2)``
    is the 2-D one. ``rhs`` and ``run`` reject other dimensions instead of
    silently applying this normalisation to a 3-D cloud.
    """
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


def pair_kernels(delta: torch.Tensor, h: float, eps: float = 1.0e-12):
    """Return ``(W, grad W)`` for a batch of pair separations.

    A hot loop needs both the kernel value and its gradient for the same pairs.
    Calling the two public functions separately evaluates the radius, the
    normalised distance ``q`` and the polynomial branch twice. This helper
    shares them and is used by :func:`rhs`; the public functions remain the
    readable single-purpose entry points and are what the tests pin.
    """
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
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


def drop_wall_wall_pairs(pairs: torch.Tensor, fluid_count: int) -> torch.Tensor:
    """Remove pairs in which both particles are wall particles.

    Positions are ordered fluid-first, wall-after, so a pair ``(i, j)`` with
    ``i >= fluid_count`` and ``j >= fluid_count`` touches no fluid particle.
    Such a pair can only contribute to wall density and wall pressure, and both
    of those are discarded: the wall pressure is extrapolated *from* the fluid,
    and the force on a static particle is never integrated. Keeping them is
    therefore pure waste -- 23 % of the pairs in the default impact case.
    """
    if pairs.numel() == 0:
        return pairs
    i, j = pairs.unbind(dim=1)
    usable = (i < fluid_count) | (j < fluid_count)
    if bool(usable.all()):
        return pairs
    return pairs[usable]


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
    if pos.shape[1] != 2:
        raise ValueError(
            f"rhs implements 2-D smoothing only, got {pos.shape[1]} dimensions")
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
        # A freshly built list still contains wall-wall pairs; drop them before
        # paying for them. A cached list is filtered when it is built.
        pairs = drop_wall_wall_pairs(pairs, n)
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
    wij, grad_w = pair_kernels(delta, h)

    rho_all = rho_all.index_add(0, i, all_mass[j] * wij)
    rho_all = rho_all.index_add(0, j, all_mass[i] * wij)
    rho = rho_all[:n].clamp_min(0.1 * cfg.rho0)
    pressure = tait_pressure(rho, cfg)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
        # Only fluid/wall pairs extrapolate wall pressure. Instead of selecting
        # those pairs with boolean masks (which costs 12 full pair-level copies
        # in this function), gather with a clamped index and zero the weight of
        # every non-crossing pair: identical numbers, fewer kernel launches.
        i_fluid = i < n
        crossing = i_fluid != (j < n)
        fluid_idx = torch.where(i_fluid, i, j)
        wall_idx = (torch.where(i_fluid, j, i) - n).clamp_(0, nw - 1)
        weight = torch.where(crossing, wij, torch.zeros_like(wij))
        pressure_ext = torch.cat((pressure, torch.zeros(nw, dtype=pos.dtype,
                                                       device=pos.device)))
        numerator = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
        denominator = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
        numerator = numerator.index_add(0, wall_idx,
                                        pressure_ext[fluid_idx] * weight)
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

    # Viscosity acts on fluid-fluid pairs and, when a no-slip wall is asked
    # for, on fluid-wall pairs with a blend factor. Wall-wall pairs never
    # contribute (the wall is static), so the mask only has to separate
    # fluid-fluid from fluid-wall.
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
    viscosity_limit = viscous_dt_limit(cfg)
    if cfg.dt > viscosity_limit * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} exceeds the viscous limit {viscosity_limit:.3e} s "
            f"(viscosity={cfg.viscosity:g}, h={cfg.h_factor * cfg.dx:g}); the "
            f"explicit Laplacian is unstable above it"
        )
    gravity_limit = gravity_dt_limit(cfg)
    if cfg.dt > gravity_limit * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} exceeds the gravity limit {gravity_limit:.3e} s"
        )


def viscous_dt_limit(cfg: Config, safety: float = 0.125) -> float:
    """Largest step the explicit viscous term tolerates: ``safety*h^2/nu``.

    The acoustic Courant condition alone does not bound the viscous operator.
    A configuration can satisfy it and still diverge: with ``viscosity = 1`` and
    the default ``dt`` the impact case reaches 53 % density compression and
    Mach 1.2, and ``viscosity = 2`` reaches 1.6e4 % before fail-fast stops it.
    The Morris/ESPHS form used here is a standard explicit Laplacian, for which
    ``dt <= 0.125 h^2 / nu`` is the usual stability bound.
    """
    h = cfg.h_factor * cfg.dx
    if cfg.viscosity <= 0.0:
        return float("inf")
    return safety * h * h / cfg.viscosity


def gravity_dt_limit(cfg: Config, safety: float = 0.25) -> float:
    """Largest step that still resolves free fall over one smoothing length."""
    if cfg.gravity <= 0.0:
        return float("inf")
    return safety * math.sqrt(cfg.h_factor * cfg.dx / cfg.gravity)


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


def evaluate_acceptance(rows, cfg: Config, boundary_events: int,
                        extremes: dict | None = None) -> list[dict]:
    """Evaluate the run against explicit, recorded numerical criteria.

    ``extremes`` are maxima accumulated over **every** integration step:
    ``peak_rho_compression``, ``peak_speed`` and ``peak_courant``. Diagnostics
    are only written every ``output_every`` steps, so judging stability from
    ``rows`` alone can miss an overshoot that happens between two output
    points. When ``extremes`` is omitted (a caller that only has diagnostics),
    the values fall back to the sampled rows and the criterion records that
    population.
    """
    sampled_compression = max(row["rho_max"] / cfg.rho0 - 1.0 for row in rows)
    sampled_speed = max(row["max_speed"] for row in rows)
    sampled_courant = max(row["courant_number"] for row in rows)
    if extremes is None:
        population = "sampled_rows"
        worst_compression, worst_speed = sampled_compression, sampled_speed
        worst_courant = sampled_courant
    else:
        population = "all_steps"
        worst_compression = float(extremes["peak_rho_compression"])
        worst_speed = float(extremes["peak_speed"])
        worst_courant = float(extremes["peak_courant"])
    worst_mach = worst_speed / cfg.sound_speed
    finite = all(all(math.isfinite(value) for value in row.values()) for row in rows)

    def criterion(identifier, metric, measured, threshold, unit, comparison,
                  scope=population):
        if comparison == ">=":
            passed = measured >= threshold
        else:
            passed = measured <= threshold
        return {
            "id": identifier, "metric": metric, "measured": measured,
            "threshold": threshold, "unit": unit, "comparison": comparison,
            "population": scope, "result": "pass" if passed else "fail",
        }

    criteria = [
        criterion("finite_state", "all diagnostics finite", 1.0 if finite else 0.0,
                  1.0, "bool", ">=", "sampled_rows"),
        criterion("density_compression", "max(rho/rho0 - 1)", worst_compression,
                  cfg.max_density_error, "1", "<="),
        criterion("mach_number", "max|v| / c0", worst_mach, cfg.max_mach, "1", "<="),
        criterion("acoustic_courant", "max dt(c0+|v|)/h", worst_courant,
                  cfg.max_courant, "1", "<="),
    ]
    escape = criterion("boundary_escape", "particles clamped beyond the box",
                       float(boundary_events), 0.0, "count", "<=",
                       "all_steps")
    if cfg.boundary == "clamp":
        # The coordinate clamp *is* the boundary condition here, so exceeding
        # the box is expected behaviour rather than a failure.
        escape["result"] = "not_evaluated"
    criteria.append(escape)
    return criteria


def _write_manifest(out: Path, cfg: Config, device: torch.device, initial_n: int,
                    rows, criteria, boundary_events: int, wall_count: int,
                    failure: str | None, fall_time: float | None,
                    cache: PairListCache | None = None,
                    extremes: dict | None = None) -> dict:
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
            "wall_wall_pairs_pruned": 0 if cache is None else cache.pruned,
        },
        "sampling": {
            "diagnostic_rows": len(rows),
            "output_every": cfg.output_every,
            "extrema_population": ("all_steps" if extremes is not None
                                   else "sampled_rows"),
            "extrema": extremes,
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


def run(cfg: Config, output: str | Path, plot: bool = True):
    """Run the case, write artifacts, and enforce numerical acceptance gates.

    Artifacts (diagnostics CSV, snapshots, run manifest, plot) are written even
    when a criterion fails, so a failed run can be inspected. If
    ``cfg.fail_fast`` is set, a violated criterion raises
    :class:`SimulationUnstableError` after the artifacts are on disk.

    Snapshots are written as they are produced and the plot reads back a
    subsample, so memory stays independent of ``steps``. Rendering the plot
    costs about 1.5 s (matplotlib import plus save); ``plot=False`` skips it
    when only the numbers matter.
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
    initial_n = pos.shape[0]
    cache = (PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                           fluid_count=initial_n)
             if cfg.pair_skin_factor > 0 else None)
    # One buffer for the fluid and the wall particles: rebuilt never, rewritten
    # every step, so the solver does not allocate two concatenations per step.
    all_pos = (torch.cat((pos, wall), dim=0) if wall is not None else pos)

    vel = torch.zeros_like(pos)
    pos_initial = pos.clone()
    rows: list[dict] = []
    snapshot_steps: list[int] = []
    boundary_events = 0
    failure: str | None = None
    # Maxima over every step, not just the steps that write a diagnostic row.
    dtype, dev = pos.dtype, pos.device
    peak_compression = torch.zeros((), dtype=dtype, device=dev)
    peak_speed = torch.zeros((), dtype=dtype, device=dev)

    for step in range(cfg.steps + 1):
        if cache is not None:
            if wall is not None:
                all_pos[:initial_n] = pos
            pairs, _ = cache.get(all_pos)
        else:
            pairs = None
        rho, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall, pairs=pairs)
        state_finite = bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()
                            and torch.isfinite(rho).all() and torch.isfinite(acc).all())
        if not state_finite and failure is None:
            failure = f"non-finite state at step {step}"
        peak_compression = torch.maximum(peak_compression,
                                         (rho / cfg.rho0).max() - 1.0)
        if vel.numel():
            peak_speed = torch.maximum(peak_speed,
                                       torch.linalg.vector_norm(vel, dim=-1).max())

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
            torch.save({"step": step, "time": step * cfg.dt,
                        "position": pos.detach().cpu(), "density": rho.detach().cpu(),
                        "boundary": cfg.boundary},
                       out / f"particles_{step:06d}.pt")
            snapshot_steps.append(step)

        if step == cfg.steps or not state_finite:
            break
        # Semi-implicit Euler with explicit boundary handling.
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        boundary_events += apply_boundary(pos, vel, cfg)

    extremes = {
        "peak_rho_compression": float(peak_compression),
        "peak_speed": float(peak_speed),
        "peak_courant": acoustic_courant(cfg.dt, cfg, float(peak_speed)),
    }
    criteria = evaluate_acceptance(rows, cfg, boundary_events, extremes=extremes)
    _write_manifest(out, cfg, device, initial_n, rows, criteria, boundary_events,
                    0 if wall is None else wall.shape[0], failure,
                    nominal_fall_time(cfg, pos_initial), cache, extremes)

    with (out / "diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        _plot(out, cfg, snapshot_steps)

    if cfg.fail_fast:
        failed = [c for c in criteria if c["result"] == "fail"]
        if failure or failed:
            detail = failure or ", ".join(
                f"{c['id']}={c['measured']:.4g} (limit {c['threshold']:g})"
                for c in failed)
            raise SimulationUnstableError(
                f"numerical acceptance failed: {detail}; artifacts in {out}")
    return rows


def _plot(out: Path, cfg: Config, steps, max_frames: int = 12) -> None:
    """Render up to ``max_frames`` snapshot files that are already on disk."""
    if not steps:
        return
    steps = list(steps)
    if len(steps) > max_frames:  # keep the panel width readable for long runs
        stride = math.ceil(len(steps) / max_frames)
        steps = steps[::stride]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed; skipping particle_snapshots.png. "
            "Install it with: pip install \"tensorsph[plot]\"",
            RuntimeWarning, stacklevel=2)
        return
    frames = []
    for step in steps:
        payload = torch.load(out / f"particles_{step:06d}.pt")
        frames.append((step, payload["position"], payload["density"]))
    _, _, _, y_max = boundary_box(cfg)
    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4),
                             squeeze=False)
    sc = None
    for ax, (step, xyz, density) in zip(axes[0], frames):
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

