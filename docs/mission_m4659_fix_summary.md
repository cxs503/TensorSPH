# Mission m4659 (re-opened): what the closed PR #309 tried to fix, re-measured

## 0. Status

| item | value |
|---|---|
| Original PR | #309 (`fix/ci-main-dbo3265`, commit `de35c5d`), **closed by owner 2026-11-05** |
| Re-opened as | m4793, branch `agent/m4793`, re-cut from `main` = `799310b` |
| Scope | the three test assertions PR #309 touched + the mission summary it shipped |
| Solver code touched | **none** |
| Measured on | loongarch64, torch 2.7.1, default dtype torch.float32 |

PR #309 was opened against `db0c3265`, where three assertions in the test suite were
red. Between then and the current `main` (`799310b`, i.e. after m4635/m4665/m4685)
both the solver and the tests moved, and two of the three relaxations were
**independently absorbed** by that later work. Re-validating the PR verbatim was
therefore not an option: one of its three assertions (`abs(bias) < 1e-5`) would now
be a *red* test on `main` (the measured bias is 5.305e-05,
five times larger). This document records the re-measurement and what was kept.

## 1. Re-measurement on current `main`

Raw evidence: [`docs/reports/m4659_probe.json`](reports/m4659_probe.json), produced by
[`docs/reports/m4659_probe.py`](reports/m4659_probe.py) — the tables below are generated
from that JSON, no number is copied by hand.

### 1.1 Kernel-sum bias of the shipped square lattice (`make_particles`, h = 1.3 dx)

| quantity | value |
|---|---|
| dx, h/dx | 0.01 m, 1.3 |
| pool particles / interior (`2h` from every face) | 3500 / 2726 |
| C - 1 on the interior mask | **-5.305e-05** |
| C - 1 on the whole bounding box (the masked trap) | -1.092e-02 |
| PR #309 claim | -1.190e-07 (not reproducible) |
| PR #309 proposed bound `abs(bias) < 1e-5` | **would fail** (measured is 5.3x larger) |
| m2971 acceptance gate | 1e-4 (measured margin: 1.9x) |

The bias sits on the tension side, so the tension-only clamp turns it into 0 Pa; what
matters physically is the order of magnitude and the sign, both stable, not the fifth
digit, which moves with the summation order. `-1.19e-7` was a property of the
`db0c3265` tree, not of the lattice.

### 1.2 Lattice sweep, raw vs calibrated

| h/dx | lattice | bias raw | bias calibrated | \|p_false\|/(rho0 c0^2) after calibration |
|---|---|---|---|---|
| 1.0 | square | 8.618e-04 | 0.000e+00 | 0.000e+00 |
| 1.1 | square | 1.834e-03 | 0.000e+00 | 0.000e+00 |
| 1.2 | square | -2.427e-04 | 4.441e-16 | 4.441e-16 |
| 1.3 | square | -5.323e-05 | -2.220e-16 | 2.220e-16 |
| 1.4 | square | 2.079e-03 | 0.000e+00 | 0.000e+00 |
| 1.5 | square | 3.440e-03 | -2.220e-16 | 2.220e-16 |
| 1.6 | square | 2.975e-03 | 0.000e+00 | 0.000e+00 |
| 1.8 | square | 8.003e-04 | -6.661e-16 | 6.661e-16 |
| 1.0 | hex | -4.120e-03 | -2.220e-16 | 2.220e-16 |
| 1.1 | hex | -1.686e-03 | 0.000e+00 | 0.000e+00 |
| 1.2 | hex | 3.225e-03 | -2.220e-16 | 2.220e-16 |
| 1.3 | hex | 5.884e-03 | -3.331e-16 | 3.331e-16 |
| 1.4 | hex | 4.171e-03 | 4.441e-16 | 4.441e-16 |
| 1.5 | hex | 1.718e-03 | 4.441e-16 | 4.441e-16 |
| 1.6 | hex | 1.863e-04 | 2.220e-16 | 2.220e-16 |
| 1.8 | hex | -1.026e-04 | 0.000e+00 | 0.000e+00 |

Calibration drives every entry to machine zero (max |bias| = 6.661e-16), which is
the reproducible part of the claim: the gate `|p_false|/(rho0 c0^2) < 1e-4` is met with
>10 orders of margin after `calibrate_mass`, and the *uncalibrated* lattice is what the
pinned value describes.

### 1.3 Wall pressure, Adami et al. (2012) eq. (11) two-particle state

| mass scale | p_f [Pa] | p_w [Pa] | a_y solver | a_y closed form | rel. error |
|---|---|---|---|---|---|
| 0.8 | 0.0 | 165.475 | -9.141726 | -9.141774 | 5.320e-06 |
| 3.0 | 0.0 | 455.829 | -7.971807 | -7.971917 | 1.388e-05 |

`rho0 g dx = 490.5 Pa` is the head the extrapolation term adds; the
body-force-only case (0.8) is recovered to 5.320e-06 and the two-term case
(3.0) to 1.388e-05 — the latter is **above** the original
`rtol=1e-5, atol=1e-6` gate, which is exactly why PR #309 relaxed it. `main` already
carries `rtol=2e-4, atol=2e-4` (m4635), leaving a
14x margin.

### 1.4 Initial particle mass and the calibration flag

| quantity | value |
|---|---|
| `Config().density_calibration` default | True |
| m_off / (rho0 dx^2) | 1.0000000000 |
| m_on / m_off | 1.0000532866 |
| analytic `calibration_factor` | 1.0000532339 |
| relative deviation | 5.267e-08 (inside `torch.allclose` default 1e-5) |
| particles in the test configuration | 1266 |

Because the flag defaults to **True**, the "uncalibrated `rho0 dx^2`" branch of this
test only exists when it is requested explicitly — which is the change PR #309 made and
`main` already contains.

## 2. Decision per file of PR #309

| file | PR #309 change | state on `main` = `799310b` | action here |
|---|---|---|---|
| `tests/test_initial_density_calibration.py` | pass `density_calibration=False` | already present (via m4635) | keep; document why + assert the default is True |
| `tests/test_kernel_sum_consistency.py` | relax `-5.32e-5` pin to `abs(bias) < 1e-5` | pin still exact, still green locally | **replace the digit pin by order-of-magnitude + sign** (PR's `1e-5` bound would be red) |
| `tests/test_wall_pressure.py` | `rtol=2e-4, atol=2e-4` | already present (via m4635) | keep; record the measured errors behind it |
| `docs/mission_m4659_fix_summary.md` | 52-line summary | absent | rewritten as this re-validated document |
| `benchmarks/laplace_pressure_jump/*.png` | generated the missing figure | still missing (unrelated red, see §5) | **out of scope**, overlaps the unmerged m4700 fix |

## 3. What this PR changes

1. `tests/test_kernel_sum_consistency.py` — the brittle
   `bias == pytest.approx(-5.32e-5, rel=5e-2)` becomes `abs(bias) < 1e-4` (the m2971
   gate) plus `bias < 0.0` (the tension side, which is what makes the clamp produce
   0 Pa). Same physics claim, no dependence on the fifth significant digit.
2. `tests/test_initial_density_calibration.py` — comment stating that the flag defaults
   to True, plus `assert Config().density_calibration is True`, so the root cause of the
   original red (silent default change) cannot come back unnoticed.
3. `tests/test_wall_pressure.py` — comment recording the measured 5.3e-6 / 1.4e-5
   agreement and the margin the 2e-4 tolerance buys. Tolerance unchanged.
4. `docs/reports/m4659_probe.py` + `.json` — reproducible measurement of every number
   above; `docs/reports/m4659_make_figure.py` + `docs/figures/m4659_revalidation.png` —
   the three panels of this document.

## 4. Verification

```
# touched tests
PYTHONPATH=$PWD/src python -m pytest -q tests/test_initial_density_calibration.py \
    tests/test_kernel_sum_consistency.py tests/test_wall_pressure.py
# -> 30 passed

# whole suite (see docs/reports for the log discipline of this mission)
PYTHONPATH=$PWD/src python -m pytest -q
# -> 4 failed, 494 passed, 9 xfailed in 350.91s
#    the 4 reds are the two pre-existing groups listed in section 5

# lint gate (CI runs the same)
python -m ruff check tests docs/reports
# -> all checks passed
```

## 5. Traps, limits, and what is deliberately not done

- **Worktree shadowing (found here).** This machine has editable installs of *other*
  agent worktrees on `sys.path` (`.pth` files). `import tensorsph` without
  `PYTHONPATH=$PWD/src` resolves to `/root/tp-agent/work/m4635/src/tensorsph`, i.e. a
  test run can silently grade a different tree. Every command above sets
  `PYTHONPATH=$PWD/src`; without it the numbers in this document are not reproducible.
  This is the most likely explanation for the "different tree, different bias" episode
  in PR #309.
- **Architecture.** All numbers are loongarch64 / torch 2.7.1 / float32. The digit-level
  values are platform-dependent by nature; the assertions this PR leaves behind are
  not (order of magnitude, sign, and the 1e-4 gate).
- **Two pre-existing reds on `main`, untouched here.** `test_laplace_pressure_jump.py::test_figure_exists`
  (the committed PNG is missing) and
  `test_oscillating_droplet_lamb.py::TestSolverRegression::test_mode_oscillates_near_the_capillary_frequency`
  (arch-scoped mode-2 pin). Both are reproducible on the unmodified tree and both are
  already the subject of other work (m4700 restores the Laplace figure and scopes the
  mode-2 pin); duplicating them here would collide with those branches.
- **No solver code was changed**, so no accuracy claim is made about the solver in this
  mission. The errors that matter here are asserted against analytic values:
  5.320e-06 and 1.388e-05 against the closed-form wall
  acceleration (<3%), 5.267e-08 against the analytic calibration
  factor (<3%), and 5.305e-05 against the exact-zero
  calibrated kernel sum (i.e. the *raw* lattice bias, not an error of the solver).
