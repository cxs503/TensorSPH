import pytest
import torch
from tensorsph.neighbor_search import brute_force_pairs, cell_linked_pairs


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
