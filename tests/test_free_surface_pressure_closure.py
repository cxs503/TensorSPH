"""Free-surface pressure closure (Adami, Ihmsen & Adams 2012, Sec. 3.3).

``tait_pressure`` clamps at ``p >= 0``. A particle that has lost neighbours to
the void has ``rho < rho0`` and is therefore clamped to exactly zero, so the
whole density-deficit shell sits at zero pressure and the interior profile has
to reach that zero over the shell. ``Config.free_surface_pressure_ratio`` turns
on the closure that replaces the clamped value by a kernel-weighted
extrapolation of the *interior* pressure.

These tests pin the operator's invariants (what it touches and what it does
not), that the two implementations -- neighbour-pair and dense -- agree, and
that the feature is off by default. What it does to the end-to-end
Young-Laplace check is measured in ``benchmarks/laplace_pressure_floor/`` and
gated in ``tests/test_laplace_pressure_floor.py``.
"""
import math

import torch

from tensorsph.droplet_impact import (
    Config,
    pair_kernels,
    rhs,
    rhs_all_pairs,
    shell_pressure,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs


def _hex_drop(radius=0.04, dx=0.004, rho0=1000.0):
    """Hexagonal lattice inside a circle: a drop with a real free surface."""
    dy = dx * math.sqrt(3.0) / 2.0
    ys = torch.arange(-radius, radius + 0.5 * dy, dy)
    pts = []
    for row, y in enumerate(ys):
        xs = torch.arange(-radius, radius + 0.5 * dx, dx)
        if row % 2:
            xs = xs + 0.5 * dx
        keep = xs.square() + y.square() <= radius * radius
        pts.extend((float(x), float(y)) for x in xs[keep])
    pos = torch.tensor(pts, dtype=torch.float64)
    mass = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=torch.float64)
    return pos, mass


def _chain(rhos, pressures, dx=0.001):
    """A one-dimensional row of particles at ``y = 0`` (2-D, so kernels apply).

    ``rho`` and ``pressure`` are handed in directly, which makes the closure's
    arithmetic checkable without a flow solve.
    """
    n = len(rhos)
    pos = torch.stack((torch.arange(n, dtype=torch.float64) * dx,
                       torch.zeros(n, dtype=torch.float64)), dim=1)
    rho = torch.tensor(rhos, dtype=torch.float64)
    p = torch.tensor(pressures, dtype=torch.float64)
    cfg = Config(dx=dx, rho0=1000.0, sound_speed=20.0, gravity=0.0,
                 free_surface_pressure_ratio=0.99)
    h = cfg.h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    wij, _ = pair_kernels(delta, h)
    return cfg, pos, rho, p, i, j, wij, delta


def test_closure_is_off_by_default():
    assert Config().free_surface_pressure_ratio == 0.0


def test_clamped_tait_puts_the_whole_deficit_shell_at_zero():
    """The defect the closure exists to remove, stated as a test."""
    cfg = Config(rho0=1000.0, sound_speed=20.0)
    rho = torch.tensor([0.85, 0.95, 1.0, 1.0063], dtype=torch.float64) * 1000.0
    p = tait_pressure(rho, cfg)
    assert float(p[0]) == 0.0 and float(p[1]) == 0.0
    assert float(p[2]) == 0.0            # rho == rho0 is still zero pressure
    assert float(p[3]) > 0.0


def test_closure_lifts_surface_particles_and_leaves_interior_alone():
    cfg, pos, rho, p, i, j, wij, delta = _chain(
        [0.95e3, 0.95e3, 1.0e3, 1.0e3, 0.95e3],
        [0.0, 0.0, 100.0, 120.0, 0.0])
    out = shell_pressure(pos, rho, p, i, j, wij, delta, cfg)

    # interior (rho >= ratio * rho0) is untouched, bit for bit
    assert float(out[2]) == 100.0 and float(out[3]) == 120.0
    # every flagged particle is now positive but never exceeds the interior
    for k in (0, 1, 4):
        assert 0.0 < float(out[k]) <= 120.0
    # closer to the interior means closer to the interior pressure
    assert float(out[1]) > float(out[0]) > 0.0
    # particle 4 has interior particle 3 one spacing away, particle 0 only has
    # particle 2 at two spacings: the nearer source must read higher
    assert float(out[4]) > float(out[0])


def test_closure_ignores_flagged_neighbours_as_sources():
    """A surface particle must not average another surface particle's artefact."""
    cfg, pos, rho, p, i, j, wij, delta = _chain([0.95e3, 0.95e3], [0.0, 0.0])
    out = shell_pressure(pos, rho, p, i, j, wij, delta, cfg)
    # no interior neighbour at all -> the Tait value is kept, not invented
    assert torch.equal(out, p)


def test_pair_and_dense_closure_agree_on_a_drop():
    """The neighbour-pair path and the dense reference must give the same field.

    This is also what catches an unordered pair list being accumulated in one
    direction only (the first version of the closure did exactly that).
    """
    pos, mass = _hex_drop()
    cfg = Config(dx=0.004, rho0=1000.0, sound_speed=20.0, gravity=9.81,
                 free_surface_pressure_ratio=0.99)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    vel = torch.zeros_like(pos)
    rho_p, p_pair, acc_pair = rhs(pos, vel, mass, cfg, pairs=pairs)
    rho_d, p_dense, acc_dense = rhs_all_pairs(pos, vel, mass, cfg)
    assert torch.allclose(rho_p, rho_d, rtol=1e-10, atol=1e-8)
    assert torch.allclose(p_pair, p_dense, rtol=1e-10, atol=1e-8)
    scale = float(acc_dense.abs().max())
    assert scale > 0.0
    assert float((acc_pair - acc_dense).abs().max()) / scale < 1e-8


def test_closure_fills_the_zero_pressure_shell():
    """With the closure on, fewer particles are pinned at exactly zero."""
    pos, mass = _hex_drop()
    cfg_kwargs = dict(dx=0.004, rho0=1000.0, sound_speed=20.0, gravity=9.81)
    off = Config(**cfg_kwargs)
    on = Config(**cfg_kwargs, free_surface_pressure_ratio=0.99)
    h = off.h_factor * off.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    vel = torch.zeros_like(pos)
    _, p_off, _ = rhs(pos, vel, mass, off, pairs=pairs)
    _, p_on, _ = rhs(pos, vel, mass, on, pairs=pairs)
    zero_off = int((p_off == 0).sum())
    zero_on = int((p_on == 0).sum())
    assert zero_off > 0
    assert zero_on < zero_off
    # the closure propagates the interior pressure outwards; it does not raise
    # the peak, which is set by the interior compression
    assert float(p_on.max()) == float(p_off.max())