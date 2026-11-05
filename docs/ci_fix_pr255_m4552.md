# CI Fix for PR #255 (TensorSPH)

## Mission: m4552
## Date: 2026-11-05

## Problem
PR #255 (branch agent/m4350) had CI red due to pytest failures. The root cause was that PR #243 introduced `density_calibration=True` as the new Config default, which changes the particle mass from `rho0*dx^2` to the calibrated value `rho0/S` (where S is the discrete kernel sum).

## Failing Tests
1. `test_initial_density_calibration.py::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass`
2. `test_kernel_sum_consistency.py::test_shipped_square_lattice_carries_no_bias`
3. `test_wall_pressure.py::test_wall_pressure_matches_the_published_extrapolation[3.0]`
4. `test_laplace_pressure_jump.py::test_figure_exists`

## Fixes Applied

### 1. test_initial_density_calibration.py
Added `density_calibration=False` to the Config in `test_make_particles_calibration_flag_is_off_by_default_and_scales_mass` to isolate the `calibrate_initial_mass` flag behavior. The test expects the base mass to be `rho0*dx^2`, which requires `density_calibration=False`.

### 2. test_kernel_sum_consistency.py
Added `density_calibration=False` to the Config in `test_shipped_square_lattice_carries_no_bias` to measure the raw lattice bias. The test is checking the uncalibrated lattice bias value.

### 3. test_wall_pressure.py
- Imported `wall_mass` function from `tensorsph.droplet_impact`
- Changed `closed_form` to use `wall_mass(cfg, 1, pos.device)` instead of `cfg.rho0 * dx * dx` to match the calibrated wall mass used by `rhs()`

### 4. test_laplace_pressure_jump.py
Changed `test_figure_exists` to `test_figure_renders_from_the_committed_report` since binary PNG files are stripped by the delivery gate. The new test renders the figure from the committed `validation_report.json` and verifies the PNG output.

## Verification
All four tests now pass:
```bash
PYTHONPATH=/root/tp-agent/work/m4350/src:$PYTHONPATH python -m pytest tests/test_initial_density_calibration.py tests/test_kernel_sum_consistency.py tests/test_wall_pressure.py tests/test_laplace_pressure_jump.py -v
```

## Commit
Commit `69289eb` on branch `agent/m4350`:
- 4 files changed, 33 insertions(+), 8 deletions(-)

## Notes
- The fix aligns with the approach taken by mission m4486 (commit `ec6503b`)
- The `density_calibration=True` default is intentional to eliminate spurious startup pressure (2.4 kPa at c0=20 m/s)
- Tests that need to measure the raw lattice bias or isolate specific flag behaviors must explicitly set `density_calibration=False`
