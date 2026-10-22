# Neighbor-search baseline

This increment adds two exact compact-support pair enumerators:

- `brute_force_pairs`: all-pairs reference implementation.
- `cell_linked_pairs`: uniform linked-cell implementation with cell edge equal to the cutoff.

Both return unique unordered pairs `(i, j)` with `i < j`, sorted lexicographically. Pairs exactly at the cutoff are included. The implementation supports 1-D, 2-D, and 3-D positions, negative coordinates, empty inputs, and arbitrary particle ordering.

## Why a linked-cell grid?

For a uniform support radius, a cell whose edge equals the cutoff only needs to inspect its own cell and adjacent cells. In 2-D this is at most 9 cells; in 3-D it is at most 27. This reduces candidate comparisons for reasonably uniform particle distributions, although worst-case clustering can still be quadratic.

This follows the same broad spatial-binning idea used by mature SPH packages such as [PySPH's linked-list and box-sort neighbor search](https://pysph.readthedocs.io/en/main/reference/nnps.html). TensorSPH's implementation is independent and deliberately small so its behavior can be tested against a brute-force oracle.

## Important current limitation

The current implementation copies positions to CPU to build bins and returns indices on the original device. It is a correctness-first reference, **not yet a GPU-resident or production-performance implementation**.

Measured on this repository (2 CPU cores), 608 particles, 5357 pairs, 17.6 neighbours per particle:

| stage | time per step |
|---|---|
| `cell_linked_pairs` (CPU binning + Python loops) | 7.1 ms |
| `rhs` given a precomputed pair list | 1.2 ms |
| `rhs` rebuilding pairs every step | 8.6 ms |

Pair construction is therefore **83 % of the step**, and at 1000 particles linked-cell is
*slower* than brute force (21.5 ms vs 18.2 ms, ratio 0.85). The RHS speedup over the dense
reference (1.8x at 256 particles, 3.5x at 512, 4.6x at 1024) comes from the O(N k) versus
O(N^2) interaction count, not from a fast search. Do not quote it as a search speedup.

The solver uses one shared pair list for density summation, pressure acceleration, and viscous acceleration. The original dense implementation is retained as `rhs_all_pairs` for regression comparisons. Pair accumulation changes floating-point summation order, so comparisons use explicit numerical tolerances.

## Run tests and compare

```bash
python -m pytest -q
python benchmarks/neighbor_search/compare.py --particles 1000 --dimension 2 --device cpu
python benchmarks/neighbor_search/compare.py --particles 1000 --dimension 3 --device cpu
python benchmarks/neighbor_search/compare_rhs.py --particles 1000 --device cpu
python benchmarks/neighbor_search/compare_rhs.py --sizes 256,512,1024 --device cpu
```

The comparison script prints pair counts, exact equivalence, and observed timings. Results depend on particle distribution, particle count, hardware, and synchronization. Do not generalize a single run into a performance claim.

## Acceptance gates

1. Exact pair equality with brute force in deterministic random 1-D/2-D/3-D tests. **Met.**
2. Explicit coverage for empty inputs, one particle, duplicate coordinates, negative coordinates, cutoff equality, and invalid input. **Met.**
3. CI green before solver integration. **Met.**
4. Compare dense and pair-based RHS values before accepting changes to the force operators. **Met** (`rtol=2e-10` in float64, plus a check that a wall outside the support changes nothing).
5. Benchmark multiple particle counts and devices; avoid CPU transfer and profile peak memory before making performance claims. **Partially met**: multiple counts are measured, but the CPU transfer is still there, so no performance claim is made.

## Open item

The next change here should be a device-resident neighbour search (vectorized binning or a
sort-based approach). Until pair construction leaves the host, the solver's per-step cost is
set by Python looping, and GPU execution cannot help.


The current WCSPH kernel normalization and droplet-impact RHS are 2-D. The neighbor-pair enumeration itself supports 1-D/2-D/3-D, but this does not imply that the physical SPH kernel/RHS is dimension-generalized. Use the multi-size RHS sweep to record numerical agreement and timing on the target machine; no universal speedup is assumed.
