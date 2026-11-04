"""Regression tests for the fluid-lattice / wall-lattice clearance.

``make_particles`` used to build its columns with ``arange(dx * 0.5, W, dx)``,
which emits ``ceil(W/dx)`` columns: the last column's *centre* stays below
``W`` while its particles reach ``(ceil(W/dx) - 0.5) dx``, so the lattice
overhangs the tank whenever ``W/dx`` is not an integer. The static wall layer is
anchored half a spacing *outside* the tank (``W + dx/2``), so an overhanging
column sits less than one spacing from the wall instead of about 1.5. At
``dx = 0.006`` (``1/0.006 = 166.67``) the last fluid column landed at
``x = 0.999``, 0.67 of a spacing from the wall column at ``x = 1.003``; a fluid
particle that close to a wall particle counts it at almost full kernel weight,
so the run started 17 % over-dense (gate 5 %) and produced a 3.4 m/s transient
that the scaling benchmark recorded as a "resolution" failure.

The fix is a cell count of ``floor(extent/dx)``
(:func:`tensorsph.droplet_impact.cell_centres`), which keeps the far column at
``(frac + 1) dx`` from the wall with ``frac = W - floor(W/dx) dx``. The tests
below pin the geometry it produces and the initial density error that follows
from it.

The pool and the drop are checked separately on purpose: the pool must fit
inside the tank, while the drop starts *above* the pool surface by
construction (``make_particles`` rejects configurations where it would not), so
a containment test over ``positions`` as a whole would be testing the wrong
thing -- the drop is at ``y = 0.49 +- 0.10`` by definition.
"""
import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cell_centres,
    domain_top,
    make_particles,
    pool_lattice,
    rhs,
    wall_lattice,
)

SPACINGS = (0.025, 0.02, 0.009, 0.006, 0.0037)


def test_no_fluid_particle_coincides_with_a_wall_particle():
    """Every pool particle keeps at least half a spacing from the wall lattice.

    Measured minimum for the configurations below: exactly ``1.00 dx`` (the
    bottom layer sits a full spacing below the lowest fluid row). The bound is
    kept at ``0.5 dx`` because that is the geometric requirement -- below it two
    particles are in the same kernel centre region -- and it stays valid if a
    configuration ever places the lattice against a wall more tightly.
    """
    for dx in (0.025, 0.02, 0.009, 0.006):
        cfg = Config(dx=dx)
        pool = pool_lattice(cfg, torch.device("cpu"))
        wall = wall_lattice(cfg, torch.device("cpu"))
        gap = float(torch.cdist(pool.double(), wall.double()).min())
        assert gap >= 0.5 * dx, (dx, gap)


def test_pool_lattice_never_overfills_the_tank():
    """The pool lattice must fit inside ``[0, W] x [0, depth]`` for every dx."""
    for dx in SPACINGS:
        cfg = Config(dx=dx)
        pool = pool_lattice(cfg, torch.device("cpu"))
        # every particle (centre) plus its own half-width stays inside the tank
        assert float(pool[:, 0].min()) - 0.5 * dx >= -1e-3, dx
        assert float(pool[:, 0].max()) + 0.5 * dx <= cfg.pool_width + 1e-3, dx
        assert float(pool[:, 1].min()) - 0.5 * dx >= -1e-3, dx
        assert float(pool[:, 1].max()) + 0.5 * dx <= cfg.pool_depth + 1e-3, dx


def test_cell_centres_count_is_the_floor_of_the_ratio():
    """``cell_centres`` emits ``floor(extent/dx)`` cells, centred in the extent.

    This is the arithmetic form of the overhang bug: ``ceil`` cells put the last
    centre above ``extent - dx/2`` and, for ``extent/dx = 166.67``, only
    ``0.33 dx`` from the far boundary. The identity is checked on extents whose
    ratio is deliberately non-integer (including one just below an integer,
    ``0.21/0.07``, where a naive floor would drop a whole row).
    """
    for extent, dx in ((1.0, 0.006), (0.35, 0.006), (1.0, 0.009),
                       (1.0, 0.0037), (0.21, 0.07), (1.0, 0.025), (1.0, 0.02)):
        x = cell_centres(extent, dx, torch.device("cpu"))
        count = int(torch.floor(torch.tensor(extent / dx + 1e-9)))
        assert x.numel() == count, (extent, dx, x.numel(), count)
        assert float(x[0]) == pytest.approx(0.5 * dx, rel=1e-6, abs=1e-9), (extent, dx)
        assert float(x[-1]) + 0.5 * dx <= extent + 1e-6, (extent, dx)
        # the far cell centre keeps at least half a spacing from the boundary
        assert extent - float(x[-1]) >= 0.5 * dx - 1e-6, (extent, dx)


def test_drop_lattice_sits_above_the_pool_and_inside_the_tank():
    """The drop may leave the pool, but not the tank."""
    for dx in (0.025, 0.009):
        cfg = Config(dx=dx)
        pos, _ = make_particles(cfg, torch.device("cpu"))
        drop = pos[pos[:, 1] > cfg.pool_depth]
        assert drop.shape[0] > 0
        # above the pool surface (the impact case needs it) and inside the box
        assert float(drop[:, 1].min()) > cfg.pool_depth, dx
        assert float(drop[:, 1].max()) + 0.5 * dx <= domain_top(cfg) + 1e-3, dx
        assert float(drop[:, 0].min()) - 0.5 * dx >= -1e-3, dx
        assert float(drop[:, 0].max()) + 0.5 * dx <= cfg.pool_width + 1e-3, dx


def test_initial_density_error_at_fine_spacing():
    """The initial state must satisfy the density gate at every spacing.

    ``rhs`` is evaluated once on the assembled initial state, so this measures
    the density summation's own error, not the solver's. Measured maximum
    compression after the fix: 0.00 % at ``dx = 0.006, 0.009, 0.0037`` (the free
    surface is *deficient*, by 37..42 %, which is what the p < 0 clamp is for).
    Before the fix ``dx = 0.006`` read 17.0 %.
    """
    for dx in (0.006, 0.009, 0.0037):
        cfg = Config(dx=dx, device="cpu", steps=1, output_every=1)
        pos, mass = make_particles(cfg, torch.device("cpu"))
        wall = wall_lattice(cfg, torch.device("cpu"))
        vel = torch.zeros_like(pos)
        rho, _, _ = rhs(pos, vel, mass, cfg, wall=wall)
        compression = float(rho.max()) / cfg.rho0 - 1.0
        assert compression < cfg.max_density_error, (dx, compression)
