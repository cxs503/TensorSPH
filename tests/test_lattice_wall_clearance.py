"""Regression tests for the fluid-lattice / wall-lattice clearance.

``make_particles`` used to build its columns with ``arange(dx/2, W, dx)``, which
emits ``ceil(W/dx)`` columns -- a lattice whose total span exceeds the tank
whenever ``W/dx`` is not an integer. At ``dx = 0.006`` the overhanging last
column landed at ``x = 0.999``, which is exactly a column of the right-hand wall
lattice (anchored at ``-(layers - 0.5) dx``). A fluid particle sitting on a wall
particle counts it at full kernel weight, so the run started at 17 % over-density
and produced a 3.4 m/s transient, which the scaling benchmark recorded as a
"resolution" failure. See ``docs/benchmarks/resolution-stability.md``.
"""
import torch

from tensorsph.droplet_impact import Config, make_particles, rhs, wall_lattice


def test_no_fluid_particle_coincides_with_a_wall_particle():
    """Every fluid particle keeps at least half a spacing from the wall lattice."""
    for dx in (0.025, 0.006):
        cfg = Config(dx=dx)
        pos, _ = make_particles(cfg, torch.device("cpu"))
        wall = wall_lattice(cfg, torch.device("cpu"))
        gap = float(torch.cdist(pos.double(), wall.double()).min())
        assert gap >= 0.5 * dx, (dx, gap)


def test_lattice_never_overfills_the_tank():
    """The fluid lattice must fit inside the tank for every spacing."""
    for dx in (0.025, 0.02, 0.009, 0.006, 0.0037):
        cfg = Config(dx=dx)
        pos, _ = make_particles(cfg, torch.device("cpu"))
        assert float(pos[:, 0].max()) + 0.5 * dx <= cfg.pool_width + 1e-3, dx
        assert float(pos[:, 1].max()) + 0.5 * dx <= cfg.pool_depth + 1e-3, dx


def test_initial_density_error_at_fine_spacing():
    """dx = 0.006 must start below the 5 % density gate (it started at 17 %)."""
    cfg = Config(dx=0.006, device="cpu", steps=1, output_every=1)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    wall = wall_lattice(cfg, torch.device("cpu"))
    vel = torch.zeros_like(pos)
    rho, _, _ = rhs(pos, vel, mass, cfg, wall=wall)
    compression = float(rho.max()) / cfg.rho0 - 1.0
    assert compression < cfg.max_density_error, compression
