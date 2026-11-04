"""Acceptance gates for the C2 initialisation mass calibration.

The discrete SPH sum on a regular lattice is not exactly one: with the
lattice-constant mass the interior sums to ``rho0 * (1 + b)``.  On the
hexagonal lattice (the Laplace / oscillating-droplet line) ``b ~ +5.9e-3``
is a *positive* bias, i.e. a 2.4 kPa spurious initial pressure at
``c0 = 20 m/s`` -- 6.7x the Young-Laplace signal of the verification
droplet.  On the square lattice of the impact case the bias is a small
negative deficit the Tait clamp hides.  ``calibrate_masses`` removes the
offset with one pass ``m -> m / (1 + b)``.

The platform acceptance bar for the calibrated state is the backlog P1
criterion: initial ``p_max / (rho0 c0^2) < 1e-4``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    make_particles,
    tait_pressure,
)
from tensorsph.initialization import (
    calibrate_droplet,
    calibrate_masses,
    circular_core_mask,
    rectangular_core_mask,
)
from tensorsph.neighbor_search import cell_linked_pairs

RHO0 = 1000.0
C0 = 20.0
GAMMA = 7.0
H_FACTOR = 1.3


def hex_droplet(radius: float, dx: float, rho0: float):
    """Hexagonal-packed circular drop centred on the origin."""
    dy = dx * math.sqrt(3.0) / 2.0
    ys = torch.arange(-radius, radius + 0.5 * dy, dy)
    xs = torch.arange(-radius, radius + 0.5 * dx, dx)
    pts = []
    for row, y in enumerate(ys):
        offset = 0.5 * dx if (row % 2) else 0.0
        xr = xs + offset
        keep = xr.square() + y.square() <= radius * radius
        if keep.any():
            pts.append(torch.stack([xr[keep], y.expand_as(xr)[keep]], dim=1))
    pos = torch.cat(pts, dim=0)
    mass = torch.full((pos.shape[0],), rho0 * dx * dy)
    return pos, mass


def test_hex_droplet_calibration_meets_the_pressure_bar():
    """Backlog P1 acceptance: initial p_max/(rho0 c0^2) < 1e-4 after the fix.

    Before the calibration the same drop starts at
    ``p = B[(1+b)^gamma - 1] ~ 2.4 kPa`` (ratio ~ 6e-3), i.e. the gate fails
    by ~60x.  After the rescale the interior is rho0 to machine precision.
    """
    radius, dx = 0.02, 0.001
    h = H_FACTOR * dx
    pos, mass = hex_droplet(radius, dx, RHO0)
    cfg = Config(dx=dx, h_factor=H_FACTOR, rho0=RHO0, sound_speed=C0,
                  gamma=GAMMA)
    pairs = cell_linked_pairs(pos, 2.0 * h)

    rho0_ = density_from_neighbor_pairs(pos, mass, h, pairs)
    p_before = tait_pressure(rho0_, cfg)
    ratio_before = float(p_before.max()) / (RHO0 * C0 ** 2)
    assert ratio_before > 1.0e-4, (
        "the uncalibrated drop no longer carries the C2 spurious pressure; "
        "the lattice or kernel changed, re-derive the bias")

    center = torch.zeros(2)
    mass_cal, b = calibrate_droplet(pos, mass, h, RHO0, center, radius,
                                    pairs=pairs)
    assert b > 0.0, "hexagonal lattice bias is expected positive"
    rho_cal = density_from_neighbor_pairs(pos, mass_cal, h, pairs)
    core = circular_core_mask(pos, center, radius, 2.0 * h)
    assert abs(float(rho_cal[core].mean() / RHO0) - 1.0) < 1.0e-6
    p_cal = tait_pressure(rho_cal, cfg)
    ratio_after = float(p_cal.max()) / (RHO0 * C0 ** 2)
    assert ratio_after < 1.0e-4, f"p_max/(rho0 c0^2) = {ratio_after:.3e}"


def test_square_pool_calibration_is_a_noop_within_tolerance():
    """The impact case's square lattice has b ~ -5e-5: calibration must not
    perturb the run, and the interior must sum to rho0."""
    cfg = Config()
    dev = torch.device("cpu")
    pos, mass = make_particles(cfg, dev)  # calibrate_init=True by default
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    n_pool = int((pos[:, 1] < cfg.pool_depth).sum())
    pool = pos[:n_pool]
    core = rectangular_core_mask(pool, cfg.pool_width, cfg.pool_depth, 2.0 * h)
    assert bool(core.any())
    assert abs(float(rho[:n_pool][core].mean() / cfg.rho0) - 1.0) < 1.0e-6
    p = tait_pressure(rho, cfg)
    assert float(p.max()) / (cfg.rho0 * cfg.sound_speed ** 2) < 1.0e-4


def test_make_particles_uncalibrated_keeps_the_lattice_mass():
    """calibrate_init=False must reproduce the pre-fix lattice mass exactly,
    so pre-calibration published numbers stay reproducible."""
    cfg = Config(calibrate_init=False)
    dev = torch.device("cpu")
    pos, mass = make_particles(cfg, dev)
    assert torch.allclose(mass, torch.full_like(mass, cfg.rho0 * cfg.dx ** 2))


def test_calibration_rejects_an_empty_or_surface_core():
    pos, mass = hex_droplet(0.02, 0.001, RHO0)
    h = H_FACTOR * 0.001
    center = torch.zeros(2)
    empty = torch.zeros(pos.shape[0], dtype=torch.bool)
    with pytest.raises(ValueError, match="no particles"):
        calibrate_masses(pos, mass, h, RHO0, empty)
    # a core restricted to the outer shell sits on the free surface and
    # measures a large deficit -> the guard must refuse to rescale
    r = torch.linalg.vector_norm(pos, dim=1)
    shell = r > (0.02 - 1.5 * h)
    assert bool(shell.any())
    with pytest.raises(ValueError, match="not small"):
        calibrate_masses(pos, mass, h, RHO0, shell)


def test_calibration_preserves_relative_mass_distribution():
    """The rescale is uniform: a non-uniform (hydrostatic-like) mass profile
    keeps its shape, only the overall factor changes."""
    pos, mass = hex_droplet(0.02, 0.001, RHO0)
    h = H_FACTOR * 0.001
    profile = 1.0 + 0.01 * pos[:, 1] / 0.02  # linear, like a hydrostatic column
    m = mass * profile
    center = torch.zeros(2)
    m_cal, b = calibrate_droplet(pos, m, h, RHO0, center, 0.02)
    ratio = m_cal / m
    assert torch.allclose(ratio, torch.full_like(ratio, 1.0 / (1.0 + b)))
