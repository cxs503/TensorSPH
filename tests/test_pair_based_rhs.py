import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
    rhs,
    rhs_all_pairs,
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
