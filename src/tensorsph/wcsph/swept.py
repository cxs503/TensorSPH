"""Summation density evaluated through the execution policy's padded launch.

The kernel is the one SPHinXsys' ``generic_for`` pattern is built for: a
per-particle loop whose trip count (the neighbour count) is not known at launch
time.  Two implementations of the same discrete sum are provided:

``summation_density_vectorised``
    TensorSPH's existing idiom -- one ``index_add`` over the whole pair list.
    This is the *reference* arm: the same discrete sum the repository already
    ships in ``tensorsph.wcsph.solver.Solver._density_only``.
``summation_density_swept``
    The ported arm: the pair list is reorganised into a ragged per-particle
    (CSR) form and walked in padded work-groups whose tail is masked, exactly
    like the OSS device launch.  ``mask_tail=False`` reproduces the antipattern
    the OSS mask exists to prevent -- the padded work items re-do the last
    particle's neighbour sum -- and is kept so that the failure mode can be
    measured instead of asserted.

Both arms share the kernel and the mass vector, so the only difference between
them is the launch geometry.
"""
from __future__ import annotations

import torch

from ..execution_policy import execution_instance, iterate_blocks, work_item_counts
from . import kernels

__all__ = [
    "summation_density_vectorised",
    "summation_density_naive",
    "build_csr",
    "summation_density_swept",
    "sweep_diagnostics",
]


def build_csr(pairs: torch.Tensor, n: int, device: torch.device | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric ragged adjacency: ``(indptr, indices)`` with ``indptr`` length ``n+1``.

    ``pairs`` holds unordered ``(i, j)`` with ``i < j``; the density sum needs
    both directions, so the list is mirrored and sorted by row.  A row is one
    particle's work item in the OSS pattern, and its length is the ragged trip
    count the padded launch has to tolerate.
    """
    dev = pairs.device if device is None else device
    if pairs.numel() == 0:
        return (torch.zeros(n + 1, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))
    pairs = pairs.to(dev)
    mirrored = torch.cat((pairs, pairs.flip(1)), dim=0)
    order = torch.argsort(mirrored[:, 0], stable=True)
    cols = mirrored[order, 1]
    counts = torch.bincount(mirrored[:, 0], minlength=n)
    indptr = torch.zeros(n + 1, dtype=torch.long, device=dev)
    torch.cumsum(counts, 0, out=indptr[1:])
    return indptr, cols


def summation_density_vectorised(
    pos: torch.Tensor,
    mass: torch.Tensor,
    h: torch.Tensor | float,
    dim: int,
    pairs: torch.Tensor,
    *,
    kernel: str = "quintic",
) -> torch.Tensor:
    """Reference arm: whole-array ``index_add`` over the pair list."""
    n = pos.shape[0]
    rho = mass * kernels.self_contribution(h, dim, kernel)
    if pairs.numel():
        i, j = pairs.unbind(dim=1)
        w, _ = kernels.kernel_pairs(pos[i] - pos[j],
                                    h[i] if torch.is_tensor(h) else h, dim, kernel)
        rho = rho.index_add(0, i, mass[j] * w)
        rho = rho.index_add(0, j, mass[i] * w)
    return rho[:n]


def summation_density_naive(
    pos: torch.Tensor,
    mass: torch.Tensor,
    h: torch.Tensor | float,
    dim: int,
    pairs: torch.Tensor,
    *,
    kernel: str = "quintic",
) -> torch.Tensor:
    """Independent reference: scalar per-particle loop, no tensor reduction.

    Scalar work with Python-level accumulation, so it is only usable on small
    cases -- but it shares no code path with either arm (no ``index_add``, no
    block sweep), which is what makes it a usable third opinion for the
    cross-validation table.
    """
    n = int(pos.shape[0])
    tensor_h = torch.is_tensor(h)
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for i, j in pairs.tolist():
        neighbours[i].append(j)
        neighbours[j].append(i)
    mass_list = mass.tolist()
    rho = []
    for i in range(n):
        h_i = h[i:i + 1] if tensor_h else torch.as_tensor(h)
        acc = float(mass_list[i]) * float(kernels.self_contribution(h_i, dim, kernel)[0])
        for j in neighbours[i]:
            w = kernels.kernel_value(torch.linalg.vector_norm(pos[i] - pos[j]).reshape(1),
                                     h_i, dim, kernel)
            acc += float(mass_list[j]) * float(w[0])
        rho.append(acc)
    return torch.tensor(rho, dtype=pos.dtype, device=pos.device)


def summation_density_swept(
    pos: torch.Tensor,
    mass: torch.Tensor,
    h: torch.Tensor | float,
    dim: int,
    pairs: torch.Tensor,
    *,
    kernel: str = "quintic",
    local_size: int | None = None,
    mask_tail: bool = True,
    bound: int | None = None,
) -> torch.Tensor:
    """Ported arm: padded work-groups of particles, tail masked.

    Parameters
    ----------
    local_size
        Work-group size; ``None`` uses the execution instance's (the OSS
        ``min(max_work_group_size, 64)`` clamp unless it was set).
    mask_tail
        ``True`` (default) applies the ``if (global_id < loop_bound)`` guard.
        ``False`` is the negative control: padded work items are clamped onto
        particle ``bound - 1`` and their contribution is *not* dropped.
    bound
        Loop bound of the launch, i.e. the number of *swept* rows; ``None``
        sweeps every row.  The OSS main loop is bounded by the fluid particle
        count (boundary particles carry their own dynamics), so passing
        ``n_fluid`` is the faithful choice when the particle set is
        ``fluid + boundary``.  Rows at or beyond ``bound`` keep their self term
        only, and the partner set is never truncated.
    """
    n = int(pos.shape[0])
    if local_size is None:
        local_size = execution_instance.work_group_size
    if bound is None:
        bound = n
    bound = int(bound)
    if not 0 <= bound <= n:
        raise ValueError("bound must satisfy 0 <= bound <= particle count")
    device = pos.device
    indptr, indices = build_csr(pairs, n, device)

    rho = mass * kernels.self_contribution(h, dim, kernel)
    for block in iterate_blocks(bound, local_size, device=device):
        rows = block.indices.clamp_max(max(bound - 1, 0))   # padded rows -> last row
        counts = indptr[rows + 1] - indptr[rows]
        total = int(counts.sum().item())
        if total == 0:
            continue
        starts = torch.repeat_interleave(indptr[rows], counts)
        # position of every pair inside its own row
        local_offset = (torch.arange(total, dtype=torch.long, device=device)
                        - torch.repeat_interleave(torch.cumsum(counts, 0) - counts,
                                                  counts))
        cols = indices[starts + local_offset]
        row_of_pair = torch.repeat_interleave(rows, counts)
        h_row = h[row_of_pair] if torch.is_tensor(h) else h
        w, _ = kernels.kernel_pairs(pos[row_of_pair] - pos[cols], h_row, dim, kernel)
        contrib = mass[cols] * w
        if mask_tail:
            # the ported ``if (global_id < loop_bound)``: padded rows contribute 0
            pair_mask = torch.repeat_interleave(block.mask, counts)
            contrib = contrib * pair_mask.to(contrib.dtype)
        rho = rho.index_add(0, row_of_pair, contrib)
    return rho[:n]


def sweep_diagnostics(particle_count: int, local_size: int | None = None) -> dict:
    """Launch bookkeeping for a particle sweep (padded / idle work items)."""
    return work_item_counts(particle_count, local_size)
