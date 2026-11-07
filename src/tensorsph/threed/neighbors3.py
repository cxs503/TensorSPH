"""Neighbour search for 3-D clouds: cubic box walls and periodic images.

``tensorsph.neighbor_search.cell_linked_pairs`` is already dimension agnostic, so
an open (wall-bounded) 3-D cloud needs nothing new and this module just forwards
to it.  What is missing there is the **periodic** variant: in a triply periodic
box the pair displacement has to be the minimum image, and a pair whose partners
sit either side of the boundary is otherwise never found because their cells are
``ncell - 1`` apart.

Reference implementation for the tests is :func:`brute_force_pairs_min_image`,
which shares no code with the cell-based search.

All functions are device resident (no host round-trip), take ``positions`` of
shape ``(N, D)`` with coordinates anywhere inside ``[0, box)`` for the periodic
variants, and return unique unordered pairs ``(i, j)`` with ``i < j`` sorted
lexicographically -- the same contract as the 2-D search, so the operators do not
care which one produced the list.
"""

from __future__ import annotations

from itertools import product

import torch

from ..neighbor_search import cell_linked_pairs

__all__ = [
    "box_pairs",
    "periodic_pairs",
    "brute_force_pairs_min_image",
    "minimum_image",
]


def box_pairs(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Unordered pairs in an open cloud (delegates to the generic 2-D search).

    Kept as a named entry point so 3-D callers have one import site and so the
    dimensionality is stated where it is used rather than assumed.
    """
    return cell_linked_pairs(positions, cutoff)


def minimum_image(delta: torch.Tensor, box) -> torch.Tensor:
    """Minimum-image displacement for ``delta`` in a rectangular periodic box.

    ``box`` is a scalar (cubic box) or a sequence of ``dim`` edge lengths.  Valid
    for ``|delta| < box / 2`` per axis, which the cutoff guarantees whenever
    ``2 h < L / 2`` -- the condition the shear-wave benchmark is configured for.
    """
    box_t = _box_tensor(box, delta)
    return delta - box_t * torch.round(delta / box_t)


def _box_tensor(box, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or per-axis box to shape ``(1, dim)``."""
    dim = like.shape[-1]
    out = torch.as_tensor(box, dtype=like.dtype, device=like.device).reshape(-1)
    if out.numel() == 1:
        out = out.expand(dim)
    if out.numel() != dim:
        raise ValueError(f"box must have 1 or {dim} edges, got {out.numel()}")
    return out.reshape(1, dim)


def _empty(device, dtype=torch.int64) -> torch.Tensor:
    return torch.zeros((0, 2), dtype=dtype, device=device)


def periodic_pairs(positions: torch.Tensor, box, cutoff: float) -> torch.Tensor:
    """Unordered pairs in a triply periodic rectangular box.

    The box is cut into ``ncell`` cells per axis with ``ncell >= 3`` and cell
    width ``<= cutoff``, so a pair inside the support always lies in the same or
    an adjacent cell *modulo the period*.  Cell indices are wrapped with
    ``remainder``, particle keys are hashed into a sorted unique array, and the
    3^D neighbour cells are expanded in one batched pass -- the same structure as
    the shipped 2-D search, with the wrap folded into the key instead of being
    handled by ghost particles (which would cost 27x memory in 3-D).
    """
    if positions.dim() != 2:
        raise ValueError("positions must be (N, D)")
    n, dim = positions.shape
    if n < 2:
        return _empty(positions.device)
    if any(float(b) <= 0.0 for b in torch.as_tensor(box).reshape(-1).tolist()):
        raise ValueError("box edges must be positive")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must be finite")

    device = positions.device
    box_t = _box_tensor(box, positions).reshape(dim)
    ncell = torch.floor(box_t / float(cutoff)).to(torch.int64).clamp_min(3)
    width = (box_t / ncell.to(positions.dtype)).reshape(1, dim)

    cells = torch.floor(positions / width).to(torch.int64).remainder(ncell)
    strides = torch.ones(dim, dtype=torch.int64, device=device)
    for axis in range(dim - 2, -1, -1):
        strides[axis] = strides[axis + 1] * ncell[axis + 1]
    keys = (cells * strides).sum(dim=1)

    unique_keys, inverse, counts = torch.unique(keys, return_inverse=True,
                                                return_counts=True)
    order = torch.argsort(inverse, stable=True)
    starts = torch.cumsum(counts, 0) - counts

    offsets = torch.tensor(list(product((-1, 0, 1), repeat=dim)),
                           dtype=torch.int64, device=device)
    neighbour = (cells.unsqueeze(1) + offsets.unsqueeze(0)).remainder(ncell)
    neighbour_keys = (neighbour * strides).sum(dim=2)                  # (n, M)
    slot = torch.searchsorted(unique_keys, neighbour_keys.reshape(-1))
    slot = slot.clamp_max(unique_keys.numel() - 1).reshape_as(neighbour_keys)
    found = unique_keys[slot] == neighbour_keys
    sources, which = found.nonzero(as_tuple=True)
    if sources.numel() == 0:
        return _empty(device)

    slots = slot[sources, which]
    occupancy = counts[slots]
    total = int(occupancy.sum())
    if total == 0:
        return _empty(device)
    first = starts[slots]
    spans = torch.repeat_interleave(first, occupancy)
    within = torch.arange(total, device=device) - torch.repeat_interleave(
        torch.cumsum(occupancy, 0) - occupancy, occupancy)
    i = torch.repeat_interleave(sources, occupancy)
    j = order[spans + within]

    keep = i < j
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return _empty(device)
    delta = minimum_image(positions[i] - positions[j], box_t)
    keep = delta.square().sum(dim=1) <= float(cutoff) ** 2
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return _empty(device)
    lexicographic = i * n + j
    sorted_index = torch.argsort(lexicographic)
    return torch.stack((i[sorted_index], j[sorted_index]), dim=1).contiguous()


def brute_force_pairs_min_image(positions: torch.Tensor, box, cutoff: float,
                                chunk: int = 512) -> torch.Tensor:
    """Reference implementation: O(N^2), minimum image, no shared machinery."""
    n, dim = positions.shape
    if n < 2:
        return _empty(positions.device)
    box_t = _box_tensor(box, positions).reshape(dim)
    cutoff_f = float(cutoff)
    out_i, out_j = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        delta = positions[start:stop].unsqueeze(1) - positions.unsqueeze(0)
        delta = minimum_image(delta, box_t)
        close = delta.square().sum(dim=-1) <= cutoff_f * cutoff_f
        rows, cols = close.nonzero(as_tuple=True)
        rows = rows + start
        keep = rows < cols
        out_i.append(rows[keep])
        out_j.append(cols[keep])
    i = torch.cat(out_i)
    j = torch.cat(out_j)
    if i.numel() == 0:
        return _empty(positions.device)
    lexicographic = i * n + j
    sorted_index = torch.argsort(lexicographic)
    return torch.stack((i[sorted_index], j[sorted_index]), dim=1).contiguous()
