# Neighbor-search baseline

This increment provides two exact compact-support pair enumerators plus a cache
that keeps the list valid while particles move:

- `brute_force_pairs`: all-pairs reference implementation, O(N²) memory, no assumptions.
- `cell_linked_pairs`: uniform linked-cell implementation with cell edge equal to the cutoff, vectorized and **device-resident**.
- `cell_linked_pairs_host`: the previous host-side binning (copy to CPU plus Python loops), retained as an independent algorithm to cross-check the vectorized search.
- `PairListCache`: Verlet-skin reuse of the pair list, optionally pruning
  wall-wall pairs at build time (`fluid_count=`).

All enumerators return unique unordered pairs `(i, j)` with `i < j`, sorted
lexicographically, and include pairs exactly at the cutoff. They support 1-D,
2-D, and 3-D positions, negative coordinates, empty inputs, arbitrary particle
ordering, and CPU/CUDA tensors.

## Why a linked-cell grid?

For a uniform support radius, a cell whose edge equals the cutoff only needs to
inspect its own cell and adjacent cells. In 2-D this is at most 9 cells; in 3-D
at most 27. This reduces candidate comparisons for reasonably uniform particle
distributions, although worst-case clustering (everything in one cell) still
degenerates to quadratic, the same bound as the reference.

This follows the same broad spatial-binning idea used by mature SPH packages
such as [PySPH's linked-list and box-sort neighbor search](https://pysph.readthedocs.io/en/main/reference/nnps.html).
TensorSPH's implementation is independent and deliberately small so its
behaviour can be tested against a brute-force oracle.

## Vectorization replaced the host round-trip

The previous version copied positions to the host (`positions.cpu().tolist()`)
and built bins with Python loops. That made the search the dominant cost of a
step and made it *slower than brute force* at 1000 particles.

The current implementation stays on the input device:

1. bin with `floor(x / cutoff)`;
2. turn the multi-index into a linear key and take `torch.unique(..., return_inverse, return_counts)`;
3. group particles by cell with a stable `argsort`;
4. query all 3^D neighbour cells in **one batched `searchsorted`**, then expand
   each (particle, neighbour cell) hit with `repeat_interleave`;
5. keep `i < j`, filter by true distance, and sort lexicographically.

Step 4 matters: a Python loop over offsets cost more than the search itself at
small particle counts, so the offsets are batched into a single tensor pass.

## Measured

Two CPU cores, uniform random points in the unit square, `cutoff = 0.1`,
best of 3. All three implementations return the **identical** pair set in every
configuration below.

| particles | pairs | brute force | host binning | vectorized | vs host | vs brute |
|---|---|---|---|---|---|---|
| 400 | 2 250 | 2.8 ms | 4.9 ms | **1.5 ms** | 3.2x | 1.8x |
| 1 000 | 14 378 | 16.4 ms | 27.3 ms | **5.5 ms** | 4.9x | 3.0x |
| 2 000 | 56 897 | 66.0 ms | 96.0 ms | **17.2 ms** | 5.6x | 3.8x |
| 5 000 | 357 911 | 641.2 ms | 602.3 ms | **120.0 ms** | 5.0x | 5.3x |

3-D, 2 000 particles, `cutoff = 0.1`: brute force 104.8 ms, host 47.9 ms,
vectorized **12.4 ms** (8.5x and 3.9x).

Before this change, linked-cell was 0.85x the speed of brute force at 1000
particles. It is now 3.0x.

## Pruning pairs that cannot contribute

When positions are ordered fluid-first, wall-after — the order `rhs` already
assumes — a pair with both indices at or beyond the fluid count touches no
fluid particle. It contributes only to wall density and wall pressure, and both
are discarded (the wall pressure is extrapolated *from* the fluid, and a static
particle's acceleration is never integrated).

In the default impact case that is **1 823 of 7 918 pairs, 23 %**, and dropping
them reduces the force evaluation from 2.62 ms to 1.85 ms together with the
masking changes described in [`../changes-0.5.0.md`](../changes-0.5.0.md). The
pruning is exact: `test_dropping_wall_wall_pairs_changes_nothing` compares
`rho`, `pressure` and `acc` against the unpruned list with `torch.equal`.

`PairListCache(cutoff, skin, fluid_count=n)` performs the prune once per rebuild,
so reused lists stay pruned.

## The skin cache matters more than the search

Particles in a weakly-compressible flow move a tiny fraction of a cell per step:
in the impact case, 0.4 mm per step against a 65 mm cell. Rebuilding the list
every step is therefore wasted work.

`PairListCache` builds at `cutoff + skin` and rebuilds only when some particle
has moved more than `skin / 2`, which bounds the change in any pair separation
by `skin`. Pairs beyond the true cutoff carry exactly zero kernel weight, so
reuse changes the cost, not the physics — verified by bit-comparable results and
by `test_pair_cache_reuse_matches_fresh_pairs`.

Impact case, 608 fluid + 291 wall particles, 750 steps:

| skin | step time | rebuilds | steps per rebuild | final kinetic energy |
|---|---|---|---|---|
| 0 (rebuild every step) | 7.50 ms | 750 | 1 | 15.607620 |
| **0.25 h** (default) | **4.35 ms** | 66 | 11 | 15.607620 |
| 0.5 h | 4.97 ms | 34 | 22 | 15.607620 |
| 1.0 h | 5.84 ms | 17 | 44 | 15.607620 |

The kinetic energy is identical to six digits across all skins, confirming that
reuse does not perturb the solution. A larger skin saves rebuilds but inflates
the pair list that every force operator then walks, so 0.25 h is the measured
optimum for this case; `Config.pair_skin_factor` exposes the choice.

Combined, the default 750-step impact run, including artifact writing and
plotting, went from **14.2 s to 5.2 s** (2.8x) with unchanged results. The
solver step alone went from 7.50 ms to 4.35 ms (1.72x); the rest of the run is
CSV, `torch.save`, and a matplotlib figure.

## Run tests and compare

```bash
python -m pytest -q
python benchmarks/neighbor_search/compare.py --particles 1000 --dimension 2
python benchmarks/neighbor_search/compare.py --particles 2000 --dimension 3
python benchmarks/neighbor_search/compare_rhs.py --sizes 256,512,1024
```

The comparison script prints pair counts, exact equivalence against brute force,
and observed timings. Results depend on particle distribution, particle count,
hardware, and synchronization. Do not generalize a single run into a
performance claim.

## Acceptance gates

1. Exact pair equality with brute force in deterministic random 1-D/2-D/3-D tests. **Met.**
2. Explicit coverage for empty inputs, one particle, duplicate coordinates, negative coordinates, cutoff equality, and invalid input. **Met.**
3. Exact agreement between the vectorized search and the independent host implementation on 400-particle clouds in 1-D/2-D/3-D. **Met.**
4. Compare dense and pair-based RHS values before accepting changes to the force operators. **Met** (`rtol=2e-10` in float64, plus a check that a wall outside the support changes nothing).
5. Reuse must not change the physics. **Met** (cache reuse matches fresh pairs; kinetic energy identical to six digits across skin sizes).

## Remaining caveat

CUDA execution is exercised by construction (no host round-trip and no Python
loop over particles) but has not been benchmarked on a GPU in this repository.
The microbenchmark numbers above are CPU. Peak memory grows with the number of
candidate pairs, which is bounded by clustering; a pathological distribution can
still allocate O(N²) candidates.
