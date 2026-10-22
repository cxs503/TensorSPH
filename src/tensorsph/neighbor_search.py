"""Exact compact-support particle-pair search implementations."""
from __future__ import annotations
from itertools import product
import math
import torch


def _validate(positions: torch.Tensor, cutoff: float) -> tuple[int, int]:
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim != 2:
        raise ValueError("positions must have shape (particle_count, dimension)")
    if positions.shape[1] < 1:
        raise ValueError("positions must have at least one spatial dimension")
    if not positions.is_floating_point():
        raise TypeError("positions must use a floating-point dtype")
    if not torch.isfinite(positions).all().item():
        raise ValueError("positions must contain only finite values")
    if not math.isfinite(float(cutoff)) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    return int(positions.shape[0]), int(positions.shape[1])


def brute_force_pairs(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Return sorted unordered pairs (i, j), i < j, within the cutoff.

    Pairs exactly at the cutoff are included. Output is shaped (pair_count, 2)
    and remains on the input device.
    """
    n, _ = _validate(positions, cutoff)
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device)
    ij = torch.triu_indices(n, n, offset=1, device=positions.device)
    distance_squared = (positions[ij[0]] - positions[ij[1]]).square().sum(dim=1)
    return ij[:, distance_squared <= float(cutoff) ** 2].T.contiguous()


def cell_linked_pairs(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Find exact unordered pairs using a uniform linked-cell grid.

    Cell edge length equals cutoff, so only adjacent cells can contain a pair
    within support. Supports arbitrary dimensions, negative coordinates, empty
    inputs, and CPU/CUDA tensors. Coordinates are copied to CPU for binning;
    this is a correctness baseline, not a GPU performance implementation.
    """
    n, dim = _validate(positions, cutoff)
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device)

    coords = positions.detach().cpu().tolist()
    cells: dict[tuple[int, ...], list[int]] = {}
    inv_cell = 1.0 / float(cutoff)
    for index, point in enumerate(coords):
        key = tuple(math.floor(value * inv_cell) for value in point)
        cells.setdefault(key, []).append(index)

    offsets = tuple(product((-1, 0, 1), repeat=dim))
    cutoff_squared = float(cutoff) ** 2
    pairs: list[tuple[int, int]] = []
    for key, members in cells.items():
        for offset in offsets:
            neighbor_key = tuple(k + d for k, d in zip(key, offset))
            other_members = cells.get(neighbor_key)
            if other_members is None:
                continue
            for i in members:
                for j in other_members:
                    if i >= j:
                        continue
                    d2 = sum((coords[i][axis] - coords[j][axis]) ** 2
                             for axis in range(dim))
                    if d2 <= cutoff_squared:
                        pairs.append((i, j))

    pairs.sort()
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device)
    return torch.tensor(pairs, dtype=torch.long, device=positions.device)
