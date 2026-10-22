import torch

from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    boundary_box,
    domain_top,
    make_particles,
    rhs,
    wall_lattice,
)


def _cfg(**overrides):
    base = dict(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
                drop_center_y=0.25, device="cpu")
    base.update(overrides)
    return Config(**base)


def test_wall_particles_lie_outside_the_fluid_box():
    cfg = _cfg()
    wall = wall_lattice(cfg, torch.device("cpu"))
    assert wall.shape[0] > 0
    outside = ((wall[:, 0] < 0) | (wall[:, 0] > cfg.pool_width)
               | (wall[:, 1] < 0))
    assert outside.all(), "wall particles must sit outside the fluid domain"
    assert wall[:, 1].min() < 0 and wall[:, 0].min() < 0
    assert wall[:, 0].max() > cfg.pool_width


def test_wall_has_requested_number_of_layers():
    cfg = _cfg(wall_layers=2, dx=0.01)
    wall = wall_lattice(cfg, torch.device("cpu"))
    dx = cfg.dx
    # Distinct negative x values on the left must form exactly `layers` columns.
    left = sorted({round(float(x), 6) for x in wall[wall[:, 0] < -1e-9][:, 0]})
    assert len([x for x in left if x > -1e-9 - 2 * dx]) == 2


def test_initial_arrangement_is_inside_the_box():
    cfg = _cfg()
    pos, mass = make_particles(cfg, torch.device("cpu"))
    x_min, x_max, y_min, y_max = boundary_box(cfg)
    assert pos[:, 0].min() >= x_min and pos[:, 0].max() <= x_max
    assert pos[:, 1].min() >= y_min and pos[:, 1].max() <= y_max
    assert (mass > 0).all()


def test_domain_top_clears_the_drop():
    cfg = _cfg()
    pos, _ = make_particles(cfg, torch.device("cpu"))
    assert pos[:, 1].max() < domain_top(cfg)


def test_apply_boundary_clamps_and_counts_crossings():
    cfg = _cfg()
    pos = torch.tensor([[0.2, 0.2], [-0.5, 0.2], [0.2, -0.5]], dtype=torch.float32)
    vel = torch.tensor([[1.0, 1.0], [-2.0, 1.0], [1.0, -3.0]], dtype=torch.float32)
    events = apply_boundary(pos, vel, cfg)
    assert events == 2
    x_min, x_max, y_min, y_max = boundary_box(cfg)
    assert pos[:, 0].min() >= x_min and pos[:, 0].max() <= x_max
    assert pos[:, 1].min() >= y_min and pos[:, 1].max() <= y_max
    # normal velocity is zeroed, tangential velocity is kept (free slip)
    assert vel[1, 0] == 0.0 and vel[1, 1] == 1.0
    assert vel[2, 1] == 0.0 and vel[2, 0] == 1.0
    assert torch.equal(vel[0], torch.tensor([1.0, 1.0]))


def test_pool_with_walls_stays_bounded_and_does_not_clump(tmp_path):
    """A short pool-only run must stay finite, bounded, and free of coincident particles."""
    cfg = _cfg(dt=2.0e-4, steps=400, output_every=100)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    keep = pos[:, 1] <= cfg.pool_depth
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    wall = wall_lattice(cfg, torch.device("cpu"))
    vel = torch.zeros_like(pos)
    for _ in range(cfg.steps):
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        assert torch.isfinite(acc).all()
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        assert apply_boundary(pos, vel, cfg) == 0
    speed = torch.linalg.vector_norm(vel, dim=-1)
    assert float(speed.max()) < 5.0
    distance = torch.linalg.vector_norm(pos[:, None, :] - pos[None, :, :], dim=-1)
    distance += torch.eye(pos.shape[0]) * 1.0e9
    assert float(distance.min()) > 0.05 * cfg.dx, "particles must not collapse onto each other"
