import pytest
import torch
from tensorsph.neighbor_search import (brute_force_pairs, cell_linked_pairs,
                                        cell_linked_pairs_host)


@pytest.mark.parametrize("positions,cutoff", [
    (torch.empty((0, 2), dtype=torch.float64), 1.0),
    (torch.tensor([[0.0, 0.0]], dtype=torch.float64), 1.0),
    (torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64), 1.0),
    (torch.tensor([[-1.0, -1.0], [-0.5, -1.0], [0.0, 0.0], [2.0, 2.0]], dtype=torch.float64), 0.5),
])
def test_cell_linked_matches_brute_force_for_edge_cases(positions, cutoff):
    assert torch.equal(cell_linked_pairs(positions, cutoff), brute_force_pairs(positions, cutoff))


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_cell_linked_matches_brute_force_random_clouds(dimension, seed):
    generator = torch.Generator().manual_seed(seed)
    positions = 4.0 * torch.randn((137, dimension), generator=generator, dtype=torch.float64)
    positions[1] = positions[0]
    positions[2] = positions[0] + torch.nn.functional.one_hot(
        torch.tensor(0), num_classes=dimension
    ).to(torch.float64) * 0.25
    actual = cell_linked_pairs(positions, 0.25)
    expected = brute_force_pairs(positions, 0.25)
    assert torch.equal(actual, expected)
    assert actual.shape[1] == 2
    assert torch.all(actual[:, 0] < actual[:, 1])
    if actual.numel():
        assert torch.all(actual[1:, 0] >= actual[:-1, 0])
        same_first = actual[1:, 0] == actual[:-1, 0]
        assert torch.all(actual[1:, 1][same_first] > actual[:-1, 1][same_first])


def test_pairs_return_on_input_device():
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0]], device="cpu")
    assert cell_linked_pairs(positions, 0.2).device == positions.device


@pytest.mark.parametrize("positions,cutoff,error", [
    (torch.zeros((2, 2)), 0.0, ValueError),
    (torch.zeros((2, 2)), float("inf"), ValueError),
    (torch.zeros((2, 2, 2)), 1.0, ValueError),
    (torch.tensor([[float("nan"), 0.0]]), 1.0, ValueError),
    (torch.ones((2, 2), dtype=torch.int64), 1.0, TypeError),
])
def test_invalid_inputs_are_rejected(positions, cutoff, error):
    with pytest.raises(error):
        cell_linked_pairs(positions, cutoff)


def test_device_search_matches_host_reference_on_large_cloud():
    """The vectorized search must agree with the independent host implementation."""
    generator = torch.Generator().manual_seed(21)
    for dimension in (1, 2, 3):
        positions = torch.rand((400, dimension), generator=generator,
                               dtype=torch.float64)
        for cutoff in (0.05, 0.1, 0.3):
            assert torch.equal(cell_linked_pairs(positions, cutoff),
                               cell_linked_pairs_host(positions, cutoff))


def test_device_search_handles_single_cell_and_empty_cells():
    clustered = torch.full((30, 2), 2.5, dtype=torch.float64)
    clustered[0] = torch.tensor([-4.0, 7.0], dtype=torch.float64)
    assert torch.equal(cell_linked_pairs(clustered, 0.2),
                       brute_force_pairs(clustered, 0.2))
    assert cell_linked_pairs(clustered, 0.2).shape[0] == 29 * 28 // 2


def test_pair_cache_reuse_matches_fresh_pairs():
    """Reusing an inflated pair list must not change the forces."""
    from tensorsph.droplet_impact import Config, make_particles, rhs
    from tensorsph.neighbor_search import PairListCache

    cfg = Config(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
                 drop_center_y=0.25, device="cpu")
    positions, mass = make_particles(cfg, torch.device("cpu"))
    velocity = torch.zeros_like(positions)
    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, 0.25 * h)

    pairs_cached, rebuilt = cache.get(positions)
    assert rebuilt and cache.rebuilds == 1
    cached = rhs(positions, velocity, mass, cfg, pairs=pairs_cached)
    fresh = rhs(positions, velocity, mass, cfg)
    for actual, expected in zip(cached, fresh):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    # A tiny displacement must not trigger a rebuild, a large one must.
    nudged = positions + 1.0e-6
    _, rebuilt = cache.get(nudged)
    assert not rebuilt and cache.reuses == 1
    moved = positions + 2.0 * h
    _, rebuilt = cache.get(moved)
    assert rebuilt and cache.rebuilds == 2


def test_pair_cache_displacement_and_validation():
    from tensorsph.neighbor_search import PairListCache

    cache = PairListCache(0.5, 0.1)
    assert cache.displacement(torch.zeros((4, 2))) == float("inf")
    cache.get(torch.zeros((4, 2)))
    assert cache.displacement(torch.zeros((4, 2))) == 0.0
    moved = torch.zeros((4, 2))
    moved[2] = torch.tensor([0.3, 0.4])  # norm 0.5
    assert cache.displacement(moved) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        PairListCache(0.0, 0.1)
    with pytest.raises(ValueError):
        PairListCache(0.5, -0.1)
