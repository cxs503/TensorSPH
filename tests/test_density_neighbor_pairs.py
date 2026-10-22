import pytest
import torch

from tensorsph.droplet_impact import cubic_spline_kernel, density_from_neighbor_pairs, rhs, Config


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("seed", [3, 19])
def test_pair_density_matches_all_pairs_reference(dimension, seed):
    generator = torch.Generator().manual_seed(seed)
    pos = torch.randn((83, dimension), generator=generator, dtype=torch.float64)
    pos[1] = pos[0]
    mass = torch.rand((83,), generator=generator, dtype=torch.float64) + 0.1
    h = 0.37

    delta = pos[:, None, :] - pos[None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    expected = (cubic_spline_kernel(distance, h) * mass[None, :]).sum(dim=1)
    actual = density_from_neighbor_pairs(pos, mass, h)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_pair_density_handles_empty_and_single_particle():
    empty_pos = torch.empty((0, 2), dtype=torch.float64)
    empty_mass = torch.empty((0,), dtype=torch.float64)
    assert density_from_neighbor_pairs(empty_pos, empty_mass, 0.2).shape == (0,)

    pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    mass = torch.tensor([2.5], dtype=torch.float64)
    actual = density_from_neighbor_pairs(pos, mass, 0.2)
    expected = mass * cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), 0.2)
    torch.testing.assert_close(actual, expected)


def test_rhs_remains_finite_with_neighbor_density():
    cfg = Config(dx=0.05, steps=0, pool_width=0.4, pool_depth=0.15,
                 drop_radius=0.05, drop_center_y=0.25, device="cpu")
    from tensorsph.droplet_impact import make_particles
    pos, _, mass = make_particles(cfg, torch.device("cpu"))
    vel = torch.zeros_like(pos)
    rho, pressure, acc = rhs(pos, vel, mass, cfg)
    assert torch.isfinite(rho).all()
    assert torch.isfinite(pressure).all()
    assert torch.isfinite(acc).all()


@pytest.mark.parametrize("seed", [5, 23])
def test_pair_rhs_matches_dense_reference(seed):
    from tensorsph.droplet_impact import rhs_all_pairs

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
