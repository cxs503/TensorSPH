# Neighbor-search baseline

This increment adds two exact compact-support pair enumerators:

- `brute_force_pairs`: all-pairs reference implementation.
- `cell_linked_pairs`: uniform linked-cell implementation with cell edge equal to the cutoff.

Both return unique unordered pairs `(i, j)` with `i < j`, sorted lexicographically. Pairs exactly at the cutoff are included. The implementation supports 1-D, 2-D, and 3-D positions, negative coordinates, empty inputs, and arbitrary particle ordering.

## Why a linked-cell grid?

For a uniform support radius, a cell whose edge equals the cutoff only needs to inspect its own cell and adjacent cells. In 2-D this is at most 9 cells; in 3-D it is at most 27. This reduces candidate comparisons for reasonably uniform particle distributions, although worst-case clustering can still be quadratic.

This follows the same broad spatial-binning idea used by mature SPH packages such as [PySPH's linked-list and box-sort neighbor search](https://pysph.readthedocs.io/en/main/reference/nnps.html). TensorSPH's implementation is independent and deliberately small so its behavior can be tested against a brute-force oracle.

## Important current limitation

The current implementation copies positions to CPU to build bins and returns indices on the original device. It is a correctness-first reference, **not yet a GPU-resident or production-performance implementation**. The first solver integration now uses these pairs for the WCSPH density sum, including the self-kernel contribution and symmetric endpoint accumulation. The pressure and viscous force calculations still use dense all-pairs tensors, so this is a staged integration rather than an end-to-end complexity reduction. Exact density equivalence is tested against the all-pairs reference. GPU-resident binning and pair-based force accumulation remain future work.

## Run tests and compare

```bash
python -m pytest -q
python benchmarks/neighbor_search/compare.py --particles 1000 --dimension 2 --device cpu
python benchmarks/neighbor_search/compare.py --particles 1000 --dimension 3 --device cpu
```

The comparison script prints pair counts, exact equivalence, and observed timings. Results depend on particle distribution, particle count, hardware, and synchronization. Do not generalize a single run into a performance claim.

## Acceptance gates

1. Exact pair equality with brute force in deterministic random 1-D/2-D/3-D tests.
2. Explicit coverage for empty inputs, one particle, duplicate coordinates, negative coordinates, cutoff equality, and invalid input.
3. CI green before solver integration.
4. Separate future performance work: avoid CPU transfer, integrate pair lists into density/force accumulation, profile memory, and benchmark multiple particle counts.
