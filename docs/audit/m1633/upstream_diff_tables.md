Generated: 2026-11-01T09:20:33Z · HEAD `400db02` on `agent/m1633`

### T1. Remote branch divergence vs local HEAD

| remote branch | ahead (unmerged) | behind (stale by) | class |
|---|---:|---:|---|
| `origin/feat/wcsph-droplet-impact` | 46 | 0 | merged |
| `origin/docs/sph-validation-roadmap` | 42 | 0 | merged |
| `origin/feat/cell-linked-neighbor-search` | 35 | 8 | diverged |
| `origin/feat/pair-based-wcsph-forces` | 28 | 3 | diverged |
| `origin/bench/rhs-scaling-validation` | 24 | 0 | merged |
| `origin/arena/2cbcd8cc-tensorsph` | 22 | 1 | diverged |
| `origin/arena/be36f406-tensorsph` | 17 | 0 | merged |
| `origin/arena/aa52281d-tensorsph` | 11 | 0 | merged |
| `origin/agent/m239` | 9 | 0 | merged |
| `origin/agent/m276` | 7 | 0 | merged |
| `origin/agent/m277` | 4 | 0 | merged |
| `origin/agent/m281` | 4 | 0 | merged |
| `origin` | 0 | 0 | merged |
| `origin/main` | 0 | 0 | merged |

### T2. Unmerged commit inventory (patch-id `git cherry` verdict)

| branch | sha | date | subject | patch in HEAD? |
|---|---|---|---|---|
| `origin/arena/2cbcd8cc-tensorsph` | `2c33033` | 2026-10-22 | perf: device-resident neighbour search, skin cache, and a dam-break case | yes (equivalent) |
| `origin/feat/cell-linked-neighbor-search` | `61aba32` | 2026-10-22 | docs: document pair-based force accumulation and reference tests | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `20013fd` | 2026-10-22 | feat: export dense RHS regression reference | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `b56e559` | 2026-10-22 | test: compare pair-based WCSPH RHS with dense reference | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `de6b57e` | 2026-10-22 | feat: compute WCSPH forces from neighbor pairs | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `346856d` | 2026-10-22 | docs: record staged WCSPH density integration | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `3f298c1` | 2026-10-22 | feat: export neighbor-based density routine | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `2076a9a` | 2026-10-22 | test: verify neighbor-pair density against all-pairs reference | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `33a659b` | 2026-10-22 | feat: compute WCSPH density from compact-support pairs | no (novel text) |
| `origin/feat/pair-based-wcsph-forces` | `9a96287` | 2026-10-22 | docs: add RHS scaling sweep and clarify 2D kernel scope | yes (equivalent) |
| `origin/feat/pair-based-wcsph-forces` | `a0f5bc9` | 2026-10-22 | test: scope WCSPH density equivalence to its 2D kernel | yes (equivalent) |
| `origin/feat/pair-based-wcsph-forces` | `7eb7096` | 2026-10-22 | bench: support RHS scaling sweeps across particle counts | yes (equivalent) |

### T3. Merge-ability against HEAD

| branch | stale by (commits) | merge clean? | conflicted files |
|---|---:|---|---|
| `origin/arena/2cbcd8cc-tensorsph` | 22 | CONFLICT | 15 |
| `origin/feat/cell-linked-neighbor-search` | 35 | CONFLICT | 3 |
| `origin/feat/pair-based-wcsph-forces` | 28 | CONFLICT | 2 |

### T4. Semantic coverage of branch-introduced symbols in HEAD

| branch | new defs | present in HEAD | absent from HEAD |
|---|---:|---:|---|
| `origin/arena/2cbcd8cc-tensorsph` | 18 | 17 | small_case |
| `origin/feat/cell-linked-neighbor-search` | 3 | 3 | — |
| `origin/feat/pair-based-wcsph-forces` | 0 | 0 | — |

### T4b. Name-level absence explained by API rename

| branch | branch symbol | closest HEAD symbol | ratio |
|---|---|---|---:|
| `origin/arena/2cbcd8cc-tensorsph` | `small_case` | `wall_mass` | 0.632 |

### T5. Branch test functions with no HEAD counterpart

| branch | unmatched test functions |
|---|---|
| `origin/arena/2cbcd8cc-tensorsph` | — |
| `origin/feat/cell-linked-neighbor-search` | `test_pair_density_handles_empty_and_single_particle` (~`test_pair_density_handles_degenerate_inputs`, 0.702), `test_pair_density_matches_all_pairs_reference` (~`test_pair_density_matches_dense_reference`, 0.86), `test_rhs_remains_finite_with_neighbor_density` (~`test_rhs_rejects_non_two_dimensional_input`, 0.552) |
| `origin/feat/pair-based-wcsph-forces` | — |

### T6. HEAD top-15 commit themes

| theme | commits |
|---|---:|
| merge-integration | 6 |
| gpu-scaling | 2 |
| validation-gate | 2 |
| audit-review | 1 |
| free-surface | 1 |
| surface-tension | 1 |
| time-integration | 1 |
| wall-boundary | 1 |
