import pytest
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


def _shear_setup(friction, dx=0.02, height=0.2, viscosity=0.05, width=0.4):
    """A slab sheared over a bottom wall, with no side constraints."""
    import math
    cfg = Config(dx=dx, pool_width=width, pool_depth=height, drop_radius=2 * dx,
                 drop_center_y=height + 0.5, gravity=0.0, viscosity=viscosity,
                 wall_friction=friction, device="cpu")
    xs = torch.arange(dx * 0.5, width, dx)
    ys = torch.arange(dx * 0.5, height, dx)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx)
    wall = wall_lattice(cfg, torch.device("cpu"), top=height + 2 * dx)
    wall = wall[wall[:, 1] < 0.0]  # bottom row only
    return cfg, pos, mass, wall, math.pi / (2.0 * height)


def test_free_slip_exerts_no_tangential_force():
    cfg, pos, mass, wall, _ = _shear_setup(0.0)
    vel = torch.zeros_like(pos)
    vel[:, 0] = 0.1  # uniform tangential motion
    _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
    assert float(acc[:, 0].abs().max()) == 0.0


def test_no_slip_exerts_tangential_force_that_scales_with_friction():
    def force(friction):
        cfg, pos, mass, wall, _ = _shear_setup(friction)
        vel = torch.zeros_like(pos)
        vel[:, 0] = 0.1
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        return float(acc[:, 0].abs().max()), float(acc[:, 0].min())

    half, _ = force(0.5)
    full, most_negative = force(1.0)
    assert full > 0.0, "a no-slip wall must oppose tangential motion"
    assert most_negative < 0.0, "the force must act against the flow direction"
    assert half == pytest.approx(0.5 * full, rel=1e-5)


def test_no_slip_reproduces_the_stokes_shear_decay_rate():
    """Quarter-wave shear over a no-slip floor decays like exp(-nu (pi/2H)^2 t)."""
    import math

    dx, height, viscosity = 0.02, 0.2, 0.05
    cfg, pos, mass, wall, k = _shear_setup(1.0, dx=dx, height=height,
                                          viscosity=viscosity)
    vel = torch.zeros_like(pos)
    vel[:, 0] = 0.1 * torch.sin(k * pos[:, 1])
    initial = float(0.5 * (mass[:, None] * vel.square()).sum())

    samples = []
    steps = int(0.25 / cfg.dt)
    for step in range(steps + 1):
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        if step % 10 == 0:
            samples.append((step * cfg.dt,
                            float(0.5 * (mass[:, None] * vel.square()).sum())))
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel

    points = [(t, math.log(e / initial)) for t, e in samples
              if 0.5 * initial >= e >= 0.02 * initial]
    assert len(points) >= 3, "the mode must decay enough to fit a rate"
    ts = [p[0] for p in points]
    ls = [p[1] for p in points]
    mean_t = sum(ts) / len(ts)
    mean_l = sum(ls) / len(ls)
    slope = (sum((a - mean_t) * (b - mean_l) for a, b in zip(ts, ls))
             / sum((a - mean_t) ** 2 for a in ts))
    measured = -slope / 2.0
    analytic = viscosity * k * k
    # Discretisation error at this resolution: measured 0.84 of analytic.
    assert 0.70 <= measured / analytic <= 1.05, (measured, analytic)
