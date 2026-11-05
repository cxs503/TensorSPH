"""Unit gates for the free-surface pressure closure (Adami, Hu & Adams 2012).

The closure extrapolates pressure onto under-supported ("free surface")
particles from their *interior* neighbours, the same way the static wall
pressure is extrapolated from the fluid.  It only matters physically inside a
full SPH run, but its exactness properties are verifiable in milliseconds on a
lattice and are what makes it safe to switch on:

1. a constant pressure field is reproduced exactly -- numerator and denominator
   are the same kernel-weighted set, so the kernel sum never has to be one;
2. with the body-force term a hydrostatic field ``p = p0 - rho0 g y`` is a fixed
   point of the operator, which is what makes it an extrapolation and not an
   average;
3. interior particles (complete support) are returned bit-identically, so the
   operator can never perturb the solved interior;
4. a particle whose neighbours are *all* surface particles keeps its Tait
   pressure: nothing is invented out of an empty sum;
5. wall pairs carry no pressure and change nothing.

The end-to-end Young-Laplace acceptance gate lives in
``tests/test_laplace_pressure_floor.py`` (``test_laplace_jump_within_15pct``).
"""
from __future__ import annotations

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    free_surface_pressure_closure,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs


def _lattice(nx: int, ny: int, dx: float):
    xs = torch.arange(nx, dtype=torch.float64) * dx
    ys = torch.arange(ny, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)


def _cfg(dx: float = 0.002, **kwargs) -> Config:
    base = dict(dx=dx, h_factor=1.3, rho0=1000.0, sound_speed=20.0, gamma=7.0,
                gravity=0.0, free_surface_pressure=True)
    base.update(kwargs)
    return Config(**base)


def test_constant_pressure_field_is_reproduced_exactly():
    """p = const survives the closure, including on incomplete support."""
    cfg = _cfg()
    h = cfg.h_factor * cfg.dx
    pos = _lattice(25, 25, cfg.dx)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                       dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    # the lattice bias makes every particle "incomplete" to a degree, so the
    # whole field goes through the extrapolation, not just the rim
    p0 = torch.full_like(rho, 123.0)
    closed = free_surface_pressure_closure(pos, rho, p0, pairs, cfg, h=h)
    torch.testing.assert_close(closed, p0, rtol=1e-12, atol=1e-10)


def test_hydrostatic_field_is_reproduced_exactly_with_the_body_force_term():
    """p(y) = p0 - rho0 g y is a fixed point of the closure."""
    cfg = _cfg(gravity=9.81)
    h = cfg.h_factor * cfg.dx
    pos = _lattice(25, 25, cfg.dx)
    _mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                      dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = torch.full((pos.shape[0],), cfg.rho0, dtype=pos.dtype)
    pressure = torch.clamp(5.0e3 - cfg.rho0 * cfg.gravity * pos[:, 1], min=0.0)

    closed = free_surface_pressure_closure(pos, rho, pressure, pairs, cfg, h=h)
    torch.testing.assert_close(closed, pressure, rtol=1e-10, atol=1e-6)


def test_interior_particles_are_bit_exact():
    """Above the density threshold the returned field is the input field."""
    cfg = _cfg()
    h = cfg.h_factor * cfg.dx
    pos = _lattice(20, 20, cfg.dx)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                      dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    pressure = tait_pressure(rho, cfg)

    closed = free_surface_pressure_closure(pos, rho, pressure, pairs, cfg, h=h)
    interior = rho >= cfg.free_surface_rho_threshold * cfg.rho0
    assert bool(interior.any())
    assert torch.equal(closed[interior], pressure[interior])


def test_a_surface_particle_without_interior_neighbours_keeps_its_pressure():
    """Two mutually-neighbouring surface particles: no source, no invention."""
    cfg = _cfg(dx=0.002)
    h = cfg.h_factor * cfg.dx
    pos = torch.tensor([[0.0, 0.0], [0.6 * h, 0.0]], dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = torch.full((2,), 0.90 * cfg.rho0, dtype=pos.dtype)  # both non-interior
    pressure = tait_pressure(rho, cfg)
    assert float(pressure.abs().max()) == 0.0

    closed = free_surface_pressure_closure(pos, rho, pressure, pairs, cfg, h=h)
    assert torch.equal(closed, pressure)


def test_wall_pairs_carry_no_pressure_and_do_not_change_the_fluid_result():
    """Pairs with an index >= n (wall) get zero weight: identical numbers."""
    cfg = _cfg()
    h = cfg.h_factor * cfg.dx
    fluid = _lattice(18, 18, cfg.dx)
    mass = torch.full((fluid.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                      dtype=fluid.dtype)
    pairs_fluid = cell_linked_pairs(fluid, 2.0 * h)
    rho = density_from_neighbor_pairs(fluid, mass, h, pairs_fluid)
    pressure = tait_pressure(rho, cfg)
    base = free_surface_pressure_closure(fluid, rho, pressure, pairs_fluid, cfg,
                                         h=h)

    wall = _lattice(18, 2, cfg.dx) + torch.tensor([0.0, -2 * cfg.dx],
                                                  dtype=fluid.dtype)
    all_pos = torch.cat((fluid, wall), dim=0)
    pairs_all = cell_linked_pairs(all_pos, 2.0 * h)
    # drop wall-wall pairs the way ``rhs`` does, keep fluid-wall pairs
    i, j = pairs_all.unbind(dim=1)
    pairs_all = pairs_all[(i < fluid.shape[0]) | (j < fluid.shape[0])]
    with_wall = free_surface_pressure_closure(all_pos, rho, pressure, pairs_all,
                                              cfg, h=h)
    torch.testing.assert_close(with_wall, base, rtol=1e-12, atol=1e-10)


def test_open_shell_pressure_is_lifted_from_zero():
    """A drop's zero-pressure skin becomes a pressure-bearing interface.

    ``tait_pressure`` clamps at p >= 0, so the deficit shell (rho/rho0 < 1) is
    pressure-free and the Young-Laplace jump cannot be held across it. The
    closure gives those particles a finite fraction of the interior pressure,
    which is the whole point of the operator. The measured lift is ~15 % of
    ``p_core`` at 0.85-0.95R for dx = 2 mm -- partial by design: the source set
    is the *interior* neighbours and the shell is several particles thick.
    """
    dx = 0.002
    radius = 0.02
    cfg = _cfg(dx=dx)
    h = cfg.h_factor * dx
    n = int(2 * radius / dx) + 3
    pos = _lattice(n, n, dx)
    pos = pos - pos.mean(dim=0, keepdim=True)
    pos = pos[torch.linalg.vector_norm(pos, dim=1) <= radius]
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    # sum-density, renormalised to the interior value and compressed by the
    # Young-Laplace requirement: this is the *state* the closure has to read
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    core = r < 0.5
    rho = rho / float(rho[core].mean()) * cfg.rho0 * (1.0 + 8.98e-4)
    pressure = tait_pressure(rho, cfg)

    shell = (r >= 0.80) & (r < 0.95)
    assert bool(shell.any())
    # the honest starting point: the shell carries no pressure at all
    assert float(pressure[shell].mean()) == 0.0
    p_core = float(pressure[core].mean())
    assert p_core == pytest.approx(360.0, rel=0.02)      # sigma/R

    closed = free_surface_pressure_closure(pos, rho, pressure, pairs, cfg, h=h)
    shell_frac = float(closed[shell].mean()) / p_core
    assert 0.05 < shell_frac < 0.40, shell_frac
    assert bool(torch.isfinite(closed).all())
    # and the closure must not create a pressure above the interior value
    assert float(closed.max()) <= p_core * 1.05


def test_flag_defaults_off_and_the_rhs_path_only_activates_on_request():
    """Backwards compatibility: the closure is opt-in."""
    assert Config().free_surface_pressure is False
    assert Config(free_surface_pressure=True).free_surface_pressure is True
    assert 0.0 < Config().free_surface_rho_threshold < 1.0

def test_rhs_with_a_wall_and_the_closure_stays_finite():
    """End-to-end gate for the code path that first crashed on wall pairs.

    ``rhs`` passes the *concatenated* (fluid + wall) positions to the closure
    while the pressure field is fluid-sized; an unclamped wall index used to
    raise ``IndexError`` here. One call is enough to pin it.
    """
    from tensorsph.droplet_impact import rhs

    dx = 0.01
    cfg = _cfg(dx=dx, gravity=9.81, sound_speed=20.0,
               surface_tension_coefficient=0.0, wall_friction=0.0)
    _h = cfg.h_factor * dx
    pos = _lattice(8, 8, dx)
    pos = pos[pos[:, 1] < 0.07]
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, dtype=pos.dtype)
    wall = torch.stack(
        (torch.arange(-1, 9, dtype=torch.float64) * dx,
         torch.full((10,), -0.5 * dx, dtype=torch.float64)), dim=1)
    vel = torch.zeros_like(pos)

    rho, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall, dt=1e-5)
    for t in (rho, pressure, acc):
        assert bool(torch.isfinite(t).all())
    assert float(pressure.min()) >= 0.0
    assert float(pressure.max()) > 0.0        # gravity compresses the column

    # the closure must not change the *wall-free interior* result
    cfg_off = _cfg(dx=dx, gravity=9.81, sound_speed=20.0,
                   surface_tension_coefficient=0.0, wall_friction=0.0,
                   free_surface_pressure=False)
    rho0_, p0_, a0_ = rhs(pos, vel, mass, cfg_off, wall=wall, dt=1e-5)
    interior = rho0_ >= cfg.free_surface_rho_threshold * cfg.rho0
    assert bool(interior.any())
    assert torch.equal(pressure[interior], p0_[interior])
