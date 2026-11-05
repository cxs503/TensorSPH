"""Regression: dam_break.make_column lattice clearance (m3087).

Same defect class as the droplet pool overhang cleared in m3117
(``docs/benchmarks/resolution-stability.md``): ``arange(dx/2, extent, dx)``
emits ``ceil(extent/dx)`` cells, so whenever ``frac(extent/dx) > 1/2`` the
last column hangs over the boundary and sits ``< 1 dx`` from the static wall
layer, starting the run over-dense.  ``make_column`` now builds the lattice
with ``droplet_impact.cell_centres`` (floor-based count).

For every published configuration (integer ``extent/dx``) the particle count
is unchanged and the layout agrees with the historical arange layout to
within the arange step-accumulation error (~1e-8 m, i.e. ~1e-6 dx), so the
published benchmark numbers are untouched.
"""
import torch

from tensorsph.dam_break import DamBreakConfig, make_column
from tensorsph.droplet_impact import rhs, wall_lattice


def _old_lattice(extent, dx):
    return torch.arange(dx * 0.5, extent, dx)


def _wall_centre_distance(cfg, positions):
    """Centre-to-centre distance, last fluid column -> first right-wall layer
    (wall layer anchored at ``pool_width + dx/2``)."""
    return cfg.pool_width + cfg.dx * 0.5 - float(positions[:, 0].max())


def test_integer_ratio_lattice_unchanged():
    """All published configurations keep their particle count and layout."""
    for dx, cw, ch in ((0.02, 0.2, 0.4), (0.04, 0.2, 0.4), (0.01, 0.2, 0.4),
                       (0.02, 1.0, 0.1), (0.04, 1.0, 0.1)):
        cfg = DamBreakConfig(dx=dx, pool_width=2.0, column_width=cw,
                             column_height=ch, tank_height=0.75,
                             sound_speed=25.0)
        pos, _ = make_column(cfg, torch.device("cpu"))
        xs, ys = _old_lattice(cw, dx), _old_lattice(ch, dx)
        assert pos.shape[0] == xs.numel() * ys.numel()
        assert float(pos[:, 0].max()) <= cw
        assert float(pos[:, 1].max()) <= ch
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        old = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
        assert (pos - old).abs().max() < 1e-5 * dx


def test_non_integer_full_width_column_stays_clear_of_the_wall():
    """frac(W/dx) = 2/3 used to lift a 167th column to 0.67 dx from the wall
    (17 % initial compression); the floor-based count keeps >= 1 dx."""
    cfg = DamBreakConfig(dx=0.006, pool_width=1.0, column_width=1.0,
                         column_height=0.4, tank_height=0.75, sound_speed=25.0)
    dev = torch.device("cpu")
    pos, mass = make_column(cfg, dev)
    assert pos.shape[0] == 166 * 66
    # the historical arange layout would have put the last column at 0.999
    xs_old = _old_lattice(cfg.column_width, cfg.dx)
    assert xs_old.numel() == 167
    assert _wall_centre_distance(cfg, pos) >= cfg.dx
    assert _wall_centre_distance(cfg, pos) > \
        cfg.pool_width + cfg.dx * 0.5 - float(xs_old.max())
    # one rhs evaluation: the initial state must not start over-dense
    wall = wall_lattice(cfg, dev, top=cfg.tank_top())
    vel = torch.zeros_like(pos)
    density, _, _ = rhs(pos, vel, mass, cfg, pairs=None, wall=wall, dt=cfg.dt)
    assert float(density.max().double() / cfg.rho0 - 1.0) < cfg.max_density_error
