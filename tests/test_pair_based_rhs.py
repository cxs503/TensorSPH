import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
    rhs,
    rhs_all_pairs,
    wall_lattice,
)


@pytest.mark.parametrize("seed", [3, 19])
def test_pair_density_matches_dense_reference(seed):
    generator = torch.Generator().manual_seed(seed)
    pos = torch.randn((83, 2), generator=generator, dtype=torch.float64)
    pos[1] = pos[0]
    mass = torch.rand((83,), generator=generator, dtype=torch.float64) + 0.1
    h = 0.37
    distance = torch.linalg.vector_norm(pos[:, None, :] - pos[None, :, :], dim=-1)
    expected = (cubic_spline_kernel(distance, h) * mass[None, :]).sum(dim=1)
    actual = density_from_neighbor_pairs(pos, mass, h)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [5, 23])
def test_pair_rhs_matches_dense_reference(seed):
    generator = torch.Generator().manual_seed(seed)
    cfg = Config(dx=0.1, h_factor=1.3, rho0=1000.0, sound_speed=8.0,
                 gamma=7.0, gravity=9.81, viscosity=0.03, device="cpu")
    pos = torch.randn((61, 2), generator=generator, dtype=torch.float64) * 0.12
    vel = torch.randn((61, 2), generator=generator, dtype=torch.float64) * 0.2
    mass = (torch.rand((61,), generator=generator, dtype=torch.float64) + 0.2) * 0.01

    dense = rhs_all_pairs(pos, vel, mass, cfg)
    paired = rhs(pos, vel, mass, cfg)
    for actual, expected in zip(paired, dense):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_pair_rhs_empty_pair_list_returns_gravity_only():
    cfg = Config(dx=0.1, h_factor=1.0, device="cpu")
    pos = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float64)
    vel = torch.zeros_like(pos)
    mass = torch.ones(2, dtype=torch.float64)
    pairs = torch.empty((0, 2), dtype=torch.long)
    rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs)
    assert torch.isfinite(rho).all()
    assert torch.isfinite(pressure).all()
    torch.testing.assert_close(acc[:, 0], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(acc[:, 1], torch.full((2,), -cfg.gravity, dtype=torch.float64))


def test_wall_outside_support_changes_nothing():
    """A wall farther away than 2h must not perturb the solution at all."""
    cfg = Config(dx=0.1, h_factor=1.0, device="cpu")
    generator = torch.Generator().manual_seed(11)
    pos = torch.rand((40, 2), generator=generator, dtype=torch.float64)
    vel = torch.rand((40, 2), generator=generator, dtype=torch.float64) * 0.1
    mass = torch.full((40,), 0.01, dtype=torch.float64)
    far_wall = torch.tensor([[-5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]],
                            dtype=torch.float64)
    without = rhs(pos, vel, mass, cfg)
    with_wall = rhs(pos, vel, mass, cfg, wall=far_wall)
    for actual, expected in zip(with_wall, without):
        assert torch.equal(actual, expected)


def test_wall_restores_kernel_support():
    """Density next to a wall is closer to rho0 when wall particles are present."""
    cfg = Config(dx=0.05, h_factor=1.3, device="cpu")
    wall = wall_lattice(cfg, torch.device("cpu"))
    # A column of fluid particles sitting just above the bottom wall.
    ys = torch.arange(cfg.dx * 0.5, 0.4, cfg.dx, dtype=torch.float32)
    pos = torch.stack((torch.full_like(ys, 0.5), ys), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx)
    vel = torch.zeros_like(pos)
    rho_free, _, _ = rhs(pos, vel, mass, cfg)
    rho_wall, _, _ = rhs(pos, vel, mass, cfg, wall=wall)
    bottom = 0  # lowest fluid particle, i.e. the one closest to the wall
    assert rho_wall[bottom] > rho_free[bottom]
    assert abs(rho_wall[bottom] / cfg.rho0 - 1.0) < abs(rho_free[bottom] / cfg.rho0 - 1.0)


def test_wall_supports_a_compressed_lattice():
    """Wall particles must push back: the bottom row accelerates less downward."""
    cfg = Config(dx=0.05, h_factor=1.3, device="cpu")
    wall = wall_lattice(cfg, torch.device("cpu"))
    spacing = 0.9 * cfg.dx  # over-packed, so the lattice carries real pressure
    xs = torch.arange(cfg.dx * 0.5, 0.4, spacing)
    ys = torch.arange(cfg.dx * 0.5, 0.25, spacing)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx)
    vel = torch.zeros_like(pos)
    bottom = pos[:, 1] < cfg.dx

    _, pressure_free, acc_free = rhs(pos, vel, mass, cfg)
    _, pressure_wall, acc_wall = rhs(pos, vel, mass, cfg, wall=wall)
    assert (pressure_free >= 0).all() and (pressure_wall >= 0).all()
    assert torch.isfinite(acc_wall).all()
    # Without walls the bottom row is unsupported and pulled down hard; the wall
    # must reduce that acceleration substantially.
    assert acc_wall[bottom, 1].mean() > acc_free[bottom, 1].mean() + 50.0


def test_viscosity_is_calibrated():
    """The discrete operator must apply the viscosity that was requested.

    A smooth shear field u_x = sin(pi y / 2H) has laplacian -k^2 u with
    k = pi / 2H. With gravity off, a uniform lattice and a low sound speed
    (pressure identically zero), the measured acceleration divided by the
    analytic laplacian must equal the nominal kinematic viscosity. Before the
    Morris factor 2 was restored this ratio was 0.5 at every resolution.
    """
    import math
    from tensorsph.droplet_impact import VISCOUS_LAPLACIAN_FACTOR

    assert VISCOUS_LAPLACIAN_FACTOR == 2.0
    cfg = Config(dx=0.05, h_factor=1.3, viscosity=1.0, gravity=0.0,
                 sound_speed=0.1, device="cpu")
    width, height = 60 * cfg.dx, 40 * cfg.dx
    xs = torch.arange(cfg.dx * 0.5, width, cfg.dx)
    ys = torch.arange(cfg.dx * 0.5, height, cfg.dx)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx)
    k = math.pi / (2.0 * height)
    vel = torch.zeros_like(pos)
    vel[:, 0] = torch.sin(k * pos[:, 1])

    rho, pressure, acc = rhs(pos, vel, mass, cfg)
    assert float(pressure.abs().max()) == 0.0, "the test field must be pressure free"

    interior = ((pos[:, 1] > 6 * cfg.dx) & (pos[:, 1] < height - 6 * cfg.dx)
                & (pos[:, 0] > 6 * cfg.dx) & (pos[:, 0] < width - 6 * cfg.dx))
    laplacian = -(k * k) * vel[interior, 0]
    ratio = acc[interior, 0] / laplacian
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-3, atol=1e-3)
