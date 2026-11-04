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


def _empty_pairs(device) -> torch.Tensor:
    return torch.empty((0, 2), dtype=torch.long, device=device)


def brute_force_pairs(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Return sorted unordered pairs (i, j), i < j, within the cutoff.

    Pairs exactly at the cutoff are included. Output is shaped (pair_count, 2)
    and remains on the input device. This is the reference implementation: it
    makes no assumptions about the distribution and is O(N^2) in memory.
    """
    n, _ = _validate(positions, cutoff)
    if n < 2:
        return _empty_pairs(positions.device)
    ij = torch.triu_indices(n, n, offset=1, device=positions.device)
    distance_squared = (positions[ij[0]] - positions[ij[1]]).square().sum(dim=1)
    return ij[:, distance_squared <= float(cutoff) ** 2].T.contiguous()


def cell_linked_pairs(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Find exact unordered pairs using a uniform linked-cell grid.

    Cell edge length equals the cutoff, so only adjacent cells can hold a pair
    within support. Binning, candidate expansion, and filtering all run on the
    input device: there is no host round-trip, so the search is usable on CUDA.

    Supports arbitrary dimensions, negative coordinates, empty inputs, and
    CPU/CUDA tensors. Returns unique unordered pairs ``(i, j)`` with ``i < j``
    sorted lexicographically, identical to :func:`brute_force_pairs`.

    Complexity is O(N * 3^D * occupancy) candidates instead of O(N^2). Worst
    case (every particle in one cell) degenerates to O(N^2), which is the same
    bound as the brute-force reference.
    """
    n, dim = _validate(positions, cutoff)
    if n < 2:
        return _empty_pairs(positions.device)

    device = positions.device
    cutoff_f = float(cutoff)
    length = positions.detach() / cutoff_f
    cells = torch.floor(length).to(torch.int64)
    cells = cells - cells.amin(dim=0)
    extent = cells.amax(dim=0) + 1

    strides = torch.ones(dim, dtype=torch.int64, device=device)
    for axis in range(dim - 2, -1, -1):
        strides[axis] = strides[axis + 1] * extent[axis + 1]
    keys = (cells * strides).sum(dim=1)

    unique_keys, inverse, counts = torch.unique(keys, return_inverse=True,
                                                return_counts=True)
    # Group particles by cell; stable sort keeps original order inside a cell.
    order = torch.argsort(inverse, stable=True)
    starts = torch.cumsum(counts, 0) - counts

    # All 3^D neighbour cells are queried in one batched pass: a Python loop
    # over offsets costs more than the search itself at small particle counts.
    offsets = torch.tensor(list(product((-1, 0, 1), repeat=dim)),
                           dtype=torch.int64, device=device)
    neighbour = cells.unsqueeze(1) + offsets.unsqueeze(0)              # (n, M, D)
    inside = ((neighbour >= 0) & (neighbour < extent)).all(dim=2)      # (n, M)
    neighbour_keys = (neighbour.clamp_min(0) * strides).sum(dim=2)     # (n, M)
    slot = torch.searchsorted(unique_keys, neighbour_keys.reshape(-1))
    slot = slot.clamp_max(unique_keys.numel() - 1).reshape_as(inside)
    found = inside & (unique_keys[slot] == neighbour_keys)
    sources, which = found.nonzero(as_tuple=True)
    if sources.numel() == 0:
        return _empty_pairs(device)

    slots = slot[sources, which]
    occupancy = counts[slots]
    total = int(occupancy.sum())
    if total == 0:
        return _empty_pairs(device)
    first = starts[slots]
    # For every (source particle, neighbour cell) pair, emit that cell's range.
    spans = torch.repeat_interleave(first, occupancy)
    within = torch.arange(total, device=device) - torch.repeat_interleave(
        torch.cumsum(occupancy, 0) - occupancy, occupancy)
    i = torch.repeat_interleave(sources, occupancy)
    j = order[spans + within]

    # Each unordered pair is generated exactly once as (i, j) with i < j.
    keep = i < j
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return _empty_pairs(device)
    distance_squared = (positions[i] - positions[j]).square().sum(dim=1)
    keep = distance_squared <= cutoff_f * cutoff_f
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return _empty_pairs(device)
    lexicographic = i * n + j
    sorted_index = torch.argsort(lexicographic)
    return torch.stack((i[sorted_index], j[sorted_index]), dim=1).contiguous()


def cell_linked_pairs_host(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Host-side linked-cell search, kept as a cross-check for the device version.

    This is the previous implementation: coordinates are copied to the host and
    bins are built with Python loops. It is correct but slow, and it exists so
    that the vectorized search can be compared against an independent algorithm
    rather than only against brute force.
    """
    n, dim = _validate(positions, cutoff)
    if n < 2:
        return _empty_pairs(positions.device)

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

    if not pairs:
        return _empty_pairs(positions.device)
    pairs.sort()
    return torch.tensor(pairs, dtype=torch.long, device=positions.device)


class PairListCache:
    """Reuse an inflated pair list while particles move less than half the skin.

    A list built at ``cutoff + skin`` stays valid at ``cutoff`` as long as no
    particle has moved more than ``skin / 2`` since the build: the separation of
    any pair can then change by at most ``skin``. Pairs beyond the true cutoff
    contribute exactly zero kernel weight, so reuse does not change the physics;
    it only avoids rebuilding a set that is still correct.

    This is the standard Verlet-skin strategy. It matters because particles in a
    weakly-compressible flow move a small fraction of a cell per step: rebuilding
    the list every step costs far more than the force evaluation that uses it.

    ``fluid_count`` optionally prunes the list at build time: positions are
    assumed to be ordered fluid-first (the order :func:`tensorsph.droplet_impact.rhs`
    already assumes for ``pos`` followed by ``wall``), and pairs in which both
    indices are at or beyond it are dropped. They cannot affect any fluid
    quantity, so pruning changes nothing but the cost; ``pruned`` reports how
    many were dropped.
    """

    def __init__(self, cutoff: float, skin: float, fluid_count: int | None = None):
        if not math.isfinite(float(cutoff)) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not math.isfinite(float(skin)) or skin < 0:
            raise ValueError("skin must be finite and non-negative")
        if fluid_count is not None and fluid_count < 0:
            raise ValueError("fluid_count must be non-negative")
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.fluid_count = fluid_count
        self._pairs: torch.Tensor | None = None
        self._anchor: torch.Tensor | None = None
        self.rebuilds = 0
        self.reuses = 0
        self.pruned = 0

    def displacement(self, positions: torch.Tensor) -> float:
        if self._anchor is None or self._anchor.shape != positions.shape:
            return float("inf")
        return float(torch.linalg.vector_norm(positions - self._anchor, dim=-1).max())

    def _build(self, positions: torch.Tensor) -> torch.Tensor:
        pairs = cell_linked_pairs(positions, self.cutoff + self.skin)
        if self.fluid_count is not None and pairs.numel():
            # Positions are ordered fluid-first: drop pairs that touch no fluid
            # particle once here rather than on every step that reuses them.
            i, j = pairs.unbind(dim=1)
            keep = (i < self.fluid_count) | (j < self.fluid_count)
            self.pruned += int(keep.numel() - keep.sum())
            pairs = pairs[keep]
        return pairs

    def get(self, positions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return ``(pairs, rebuilt)`` for the current positions."""
        if self._pairs is None or self.displacement(positions) > 0.5 * self.skin:
            self._pairs = self._build(positions)
            self._anchor = positions.detach().clone()
            self.rebuilds += 1
            return self._pairs, True
        self.reuses += 1
        return self._pairs, False
