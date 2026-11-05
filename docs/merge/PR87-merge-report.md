# PR #87 merge report — `agent/m2191` → `main`

Mission m3791. Repository TensorSPH. Merge of the 0.9.0 surface-tension
colour-function work (PR #87) into `main`, resolving the textual conflicts by
semantic union: **nothing from either side was dropped**, and where the two
sides genuinely disagreed about which model is shipped, the union keeps both
models behind an explicit switch and records the disagreement in the docs.

| item | value |
|---|---|
| branch merged | `agent/m2191` |
| old SHA (PR head as merged) | `8c19ac0782ba6f4d3920b211b55e9db6dde2c573` |
| `main` merged into | `a6610546dc1d96c400ce1c2777a3a3b0580c09b5` |
| merge base | `400db02cfa49fe26ecc0fd5e62ab04bc52099008` |
| merge commit | see this commit (`git log -1 docs/merge/PR87-merge-report.md`) |
| conflict resolution | 8 conflicted paths, all resolved by hand (no `-X ours/theirs`, no overlay) |

## 1. Why the branches diverged

`main` moved 195 files past the merge base (the m2399→m3387 surface-tension
programme: the `csf_consistent` module, the Young–Laplace gates, the static
droplet, the curvature-fit default, plus a repo-wide lint pass). `agent/m2191`
moved 14 files: a colour-function study (`neighbor_count` / `kernel_sum` /
`support_sum`), three new `Config` fields, its own Laplace benchmark, tests and
docs. Both sides independently added a file at the same two paths
(`benchmarks/surface_tension/laplace.py`,
`tests/test_surface_tension_laplace.py`) — an add/add conflict on different
content, i.e. exactly the case a mechanical overlay loses silently.

## 2. Conflict inventory and per-file resolution

| # | path | conflict | resolution |
|---|---|---|---|
| 1 | `src/tensorsph/__init__.py` | content: one import each | **Union**: both `apply_contact_angle_boundary` and `capillary_dt_limit` imported; `kernel_sum_color_field` added to the public list. |
| 2 | `src/tensorsph/droplet_impact.py` | content: adjacent `Config` fields | **Union**: `contact_angle` / `csf_scheme` / `csf_curvature` (main) **and** `surface_tension_color` / `surface_tension_smoothing` / `surface_tension_delta_scale` (PR) all kept; `validate_config` validates all of them and accepts `csf_scheme="colour"`. |
| 3 | `src/tensorsph/surface_tension.py` | two content blocks | **Union of both model families** in one dispatcher; the auto-merged import line had dropped PR's `cubic_spline_kernel` (fixed); the name `kernel_sum_color` collided between main's `csf_consistent` variant and PR's local-density variant (PR's renamed `kernel_sum_color_field`, both exported, both documented). |
| 4 | `tests/test_surface_tension_laplace.py` | add/add, two different suites | **Union of both suites in one file** (disjoint test names): main's gauge tests + strict `xfail` Laplace gate against `benchmarks/surface_tension/laplace.py`, plus PR's colour-function tests (support-sum normal, unsmoothed negative control, neighbour-count negative control, delta-scale linearity, end-to-end droplet pressure). One module docstring covers both. Unused import (`support_vector`) not carried over (F401). |
| 5 | `benchmarks/surface_tension/laplace.py` | add/add, two different 420-line scripts | **Both kept**: main's script stays at this path (main's tests import it by path and `docs/benchmarks/surface-tension-laplace.md` cites it); PR's script moved verbatim to **`benchmarks/surface_tension/laplace_colour.py`**, with its 3 textual references updated (`docs/benchmarks/surface-tension.md`, `docs/changes-0.9.0.md`, `benchmarks/surface_tension/gen_docs.py`). Outputs do not collide (main → `artifacts/surface_tension_laplace.json`, PR → `benchmarks/surface_tension/results/`). |
| 6 | `benchmarks/static_droplet/run.py` | 5 content blocks | **Union**: main's lint pass, superseded header and honest JSON fields **plus** PR's `model` argument (colour function), its dx-scaled acceleration scale, mean-cosine direction criterion and colour-model reporting. The script now selects the colour family explicitly (`csf_scheme="colour"`) instead of silently falling into the new default scheme; the status stays `WITHDRAWN` with `status_before_withdrawal` carrying the sanity-check verdict. |
| 7 | `benchmarks/static_droplet/validation_report.json` | content: status block | **Union**: PR's measured record (`support_sum`, mean cosine 0.999, acc ratio 0.179, both criteria) *and* main's withdrawal (`status: WITHDRAWN`, `status_before_withdrawal: PASS`, `withdrawal_reason`, `superseded_by`, `docs`). Regenerable by the merged script. |
| 8 | `docs/algorithm-gap-analysis.md` | content: B3 status paragraph | **Union**: main's "measured and failed" numbers for the merged neighbour-count model *and* the 0.9.0 colour-function numbers (curvature 4.3 %, pressure 27–47 %, δ integral 0.67), each pointing at its own benchmark and evidence doc; the oscillating-droplet leg stays open. |

### 2.1 The one substantive design decision

`main` made `csf_scheme="consistent"` the default; `agent/m2191` made a colour
function (`surface_tension_color`) the model selector. Naively merged, every
PR config would silently run the *consistent* model while its report claimed
`support_sum`. The resolution is an explicit precedence rule, documented in the
`compute_surface_tension_force` docstring:

1. an explicit `scheme` argument wins;
2. otherwise a non-default `cfg.surface_tension_color` routes to the
   colour-function path (the 0.9.0 feature);
3. otherwise `cfg.csf_scheme` (default `"consistent"`).

`"legacy"` is kept as the historical neighbour-count behaviour (it pins the
colour to `neighbor_count`, so the alias cannot silently start measuring
something else); `"colour"` is the general colour-function path. `main`'s
legacy branch was consolidated into `surface_tension_field(model="neighbor_count")`,
which is the same formulae — verified by main's pinned legacy numbers still
passing (`tests/test_laplace_pressure.py`, legacy ratio < 0.4) and by 48
surface-tension/contact-angle/wall tests passing unchanged.

## 3. Verification

| check | command | result |
|---|---|---|
| lint (CI gate) | `python -m ruff check .` | **pass** (all checks passed; one per-file E501 ignore for the markdown-emitting generator) |
| AST/import sanity | `python -c "import tensorsph"` | **pass** |
| surface-tension + CSF + contact angle + wall | `pytest tests/test_surface_tension.py tests/test_csf_consistent.py tests/test_contact_angle.py tests/test_wall_pressure.py` | **48 passed** |
| Laplace family (main + PR suites) | `pytest tests/test_surface_tension_laplace.py tests/test_laplace_pressure.py tests/test_laplace_law.py tests/test_laplace_validation.py tests/test_laplace_verification.py tests/test_young_laplace.py` | see `artifacts/merge/pr87-test-log.txt` |
| full suite | `pytest -q` | see `artifacts/merge/pr87-full-test-log.txt` |

The two strict `xfail` tests inherited from `main` still fail as xfail (the
shipped consistent scheme still does not hold `dp = sigma/R`); they are not
converted into passes by this merge, and the docs say so.

## 4. Known limitations of this merge

- The two Laplace benchmarks remain separate files (`laplace.py` = gate for the
  shipped scheme, `laplace_colour.py` = colour-function study). Merging them
  into one CLI was not attempted: their `main()`s write different artefacts and
  both are cited by docs and tests.
- The static-droplet report is a hand-merged union of two runs at different
  model settings; only its `setup`/`measured` block comes from one run
  (`support_sum`), which the `colour_model` field states explicitly.
- `agent/m2191`'s public name `kernel_sum_color` now refers to the
  `csf_consistent` variant (signature takes `rho0`); the PR's local-density
  variant is exported as `kernel_sum_color_field`.