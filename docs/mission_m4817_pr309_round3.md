# PR #309, round 3: the failing side of each relaxation (mission m4817)

## 0. Where the lineage stands

| item | value |
|---|---|
| Original PR | #309 (`fix/ci-main-dbo3265`, `de35c5d`) -- **closed by the owner 2026-11-05** |
| Substance landed as | m4659 / m4793, merged to `main` as PR #322 (`d61b6c2`) |
| This round (m4817) | branch `agent/m4817`, cut from `main` = `c272a6f0027a12ca9986abfc29aa7cd92e7798dd` |
| Solver code touched | **none** -- tests, measurement and documentation only |
| Philosophy of this unit | a relaxed assertion is a *claim*; each claim gets a control with a failing side |

PR #309 wanted three red assertions made green. m4659 re-measured them, found that
two of the three relaxations had already been absorbed by later work, and replaced
the third by an order-of-magnitude + sign gate. That left a gap this round closes:
the relaxations were **justified in prose only**. Nothing in the suite showed that
the *strict* version of each gate is wrong today, so a future edit could widen a
tolerance and every test would still be green. Each of the three controls below
therefore measures the side of the gate that must fail.

## 1. Control A -- the kernel-sum gate is a lattice property, not a reduction artefact

The merged assertion in `tests/test_kernel_sum_consistency.py` is
`abs(bias) < 1e-4` plus `bias < 0`. What licenses it is the claim that the fifth
digit belongs to the summation order while the magnitude belongs to the lattice.
Measured on the shipped pool (`make_particles`, `dx = 0.01 m`, `h/dx = 1.3`,
3500 pool particles, 2726 interior):

| reduction | C - 1 |
|---|---|
| `fp32_as_shipped` | -5.304813e-05 |
| `fp32_permuted` | -5.310774e-05 |
| `fp64_cast` | -5.316119e-05 |
| `fp64_cast_permuted` | -5.316119e-05 |

| quantity | value |
|---|---|
| spread over the four reductions | 1.131e-07 = 1.13e-03 x gate |
| all four inside the `1e-4` gate | yes |
| all four on the tension side (`bias < 0`) | yes |
| measured / PR #309's pin `1.19e-07` | 446x |

So the gate discriminates at the level of the lattice (a hex lattice moves it by
two orders of magnitude) and is insensitive to the reduction. PR #309's `-1.19e-7`
is 446 times smaller than what this tree measures: restoring it verbatim
would redden `main`, exactly as m4659 reported.

## 2. Control B -- the wall-pressure tolerance: red where it must be, green where it must be

`tests/test_wall_pressure.py` now runs `rtol=2e-4, atol=2e-4` (m4635).
The two-particle Adami eq. (11) states are graded here by *both* that gate and the
pre-#309 gate (`rtol=1e-5, atol=1e-6`):

| mass scale | a_y solver | a_y closed form | rel. error | pre-#309 gate | shipped gate |
|---|---|---|---|---|---|
| 0.8 | -9.141726 | -9.141774 | 5.320e-06 | green | green |
| 3.0 | -7.971807 | -7.971917 | 1.388e-05 | **red** | green |

The body-force-only state is green under the old tolerance; the two-term state
(where the `rho g dx = 490.5 Pa` head lives) is **red** under it and green
under the shipped one. The relaxation is therefore a correction, not a weakening --
and since `rtol=2e-4` is 14.4x the worst measured error, a 5 % error
is still caught (`test_the_shipped_wall_tolerance_still_detects_a_real_regression`).

## 3. Control C -- the calibration default is load bearing where the clamp cannot help

`tests/test_initial_density_calibration.py` asserts that
`Config().density_calibration` defaults to `True` (guarding the silent default
change that opened this whole lineage). The measured reason:

| lattice at `h/dx = 1.3` | raw bias | clamped to 0 Pa? | `p_false/(rho0 c0^2)` raw | after calibration |
|---|---|---|---|---|
| square | -5.305e-05 | yes | 5.304e-05 | 1.83e-07 |
| hex | +5.884e-03 | no | 5.988e-03 | not applicable (hex mass differs) |

The shipped square pool is rescued by the tension clamp either way; a hex lattice at
the same `h/dx` is not -- its positive bias survives the clamp as
`2395 Pa` at `c0 = 20 m/s`. On that lattice the default flag is the
difference between a quiet start and a start that rings.

## 4. What this round adds (5 files, one focal point, no solver change)

| file | role |
|---|---|
| `tests/test_pr309_relaxation_controls.py` | the three controls + their failing sides + a guard that the committed probe still describes this tree |
| `docs/reports/pr309_round3_probe.py` / `.json` | reproducible measurement of every number in this document |
| `docs/reports/pr309_round3_figure.py` / `docs/figures/pr309_round3_controls.png` | the three panels, rendered from the JSON |
| `docs/reports/pr309_round3_make_doc.py` | renders *this* document from the JSON (no number is transcribed by hand) |

Per-file state of PR #309 on `main` = `c272a6f0027a12ca9986abfc29aa7cd92e7798dd`, unchanged by this round:

| PR #309 file | state on `main` | action here |
|---|---|---|
| `tests/test_initial_density_calibration.py` | merged via m4793; flag default asserted | keep; the *justification* is now machine-checked (§3) |
| `tests/test_kernel_sum_consistency.py` | merged via m4793; `abs(bias) < 1e-4` + sign | keep; reduction control added (§1) |
| `tests/test_wall_pressure.py` | merged via m4635; `rtol=2e-4, atol=2e-4` | keep; negative + positive controls added (§2) |
| `docs/mission_m4659_fix_summary.md` | merged via m4793 | keep; this document is the round-3 continuation |
| `benchmarks/laplace_pressure_jump/*.png` | still uncommitted on `main`; `test_figure_exists` rebuilds it | **deliberately not taken here** -- three open branches (m4700, m4795, m4811) already commit that blob; adding a fourth copy would collide on a binary |

## 5. Verification

```
PYTHONPATH=$PWD/src python -m pytest -q tests/test_pr309_relaxation_controls.py
# -> 6 passed

PYTHONPATH=$PWD/src python -m pytest -q tests/test_initial_density_calibration.py \\
    tests/test_kernel_sum_consistency.py tests/test_wall_pressure.py
# -> 30 passed  (the merged PR #309 assertions, unchanged)

PYTHONPATH=$PWD/src python -m ruff check tests/test_pr309_relaxation_controls.py \\
    docs/reports/pr309_round3_probe.py docs/reports/pr309_round3_figure.py
# -> All checks passed
```

`PYTHONPATH=$PWD/src` is not optional: this machine has editable installs of other
agent worktrees on `sys.path`, so a bare `pytest` can grade a *different tree*
(trap recorded by m4659, reconfirmed here).

Full suite on this branch (raw logs are committed next to this generator):

| run | tree | result |
|---|---|---|
| `docs/reports/m4817_full_suite_before.log` | untouched `main` | 7 failed, 564 passed, 9 xfailed, 2 warnings in 482.42s (0:08:02) |
| `docs/reports/m4817_full_suite_after.log` | this branch | 7 failed, 570 passed, 9 xfailed, 2 warnings in 431.47s (0:07:11) |

The two runs differ by exactly the 6 new tests: same 7 pre-existing reds, no new
failure and no newly-green pre-existing failure.

## 6. Pre-existing reds, untouched, with owners

Measurement on the **unmodified** tree (same tree, `git status` clean) before this
round: `7 failed, 564 passed, 9 xfailed, 2 warnings in 482.42s (0:08:02)`. All seven are outside this unit's scope and already contested by
open branches:

| red | cause | in flight |
|---|---|---|
| `test_oscillating_droplet_gauges.py` (6 tests) | `sys.modules['run']` holds another benchmark's `run` module by the time this file does `sys.path.insert(...); import run` -> `AttributeError: no attribute 'elliptical_drop'` | m4795, m4802, m4805, m4808, m4811, m4813 |
| `test_lint_gate.py::test_ruff_check_is_clean` | 27 `E501/F841/W292` findings in `gen_report.py`, `benchmarks/oscillating_droplet/gauge_audit.py`, `tests/test_oscillating_droplet_gauges.py` | m4782 (gen_report), m4685/m4700 (lint gate) |

Both were reproduced on the untouched tree *before* any file here was written; the
fixes belong to those branches, and duplicating them would put six agents on the
same three files. This unit adds no new red: the seven are byte-identical before and
after (compare the two committed logs in §5).

## 7. Traps recorded (append-only knowledge)

- **`sys.modules['run']` collision.** `sys.path.insert(0, bench_dir); import run`
  binds the *first* `run` module imported in the session. That is why six tests in
  `test_oscillating_droplet_gauges.py` fail on this tree while passing in isolation.
  Reproduce: `pytest -q tests/test_laplace_pressure_jump.py tests/test_oscillating_droplet_gauges.py`.
- **Permuting past a prefix breaks a masked measurement.** My first version of the
  reduction control shuffled *all* particles, which mixed drop particles into the
  `[:n_pool]` prefix and produced `-1.1e-2` -- the free-surface contamination of
  `docs/mission_m4659_fix_summary.md` §1.1, not a bias. Permute *inside* the subset
  you mask.
- **Stale JSON is the failure mode of this lineage.** A probe measured on an old
  `main` is a fossil; `test_probe_json_agrees_with_this_tree` now reddens if the
  committed JSON stops describing the tree.

## 8. Accuracy against analytic values

| quantity | this tree | reference | relative error | < 3 %? |
|---|---|---|---|---|
| two-particle wall acceleration, body-force state | -9.141726 | -9.141774 (closed form) | 5.320e-06 | yes |
| two-particle wall acceleration, two-term state | -7.971807 | -7.971917 (closed form) | 1.388e-05 | yes |
| calibrated hex kernel sum | bias 1.83e-07 | 0 (exact) | n/a (machine zero class) | yes |
| reduction spread, kernel sum | 1.13e-07 | 0 (exact) | 1.13e-03 of the 1e-4 gate | yes |
| measured vs PR #309 pin | -5.305e-05 | 1.19e-07 | 100 % off -- the pin is tree-specific | n/a (claimed non-reproducible) |

No solver accuracy claim is made in this mission: no solver line changed.
