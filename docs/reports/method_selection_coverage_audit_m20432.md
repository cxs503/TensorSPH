# 方法选择证据登记表的“指标挑选”审计 (mission 20432)

独立复算：本文件由 `benchmarks/method_selection/audit_quantity_coverage.py` 程序化生成，不读取 `build_registry.py` 的任何函数，也不信任产物里已存好的百分比。

## 1 覆盖情况

- 登记表记录数：17
- 产物自带门限的量：16
- 覆盖完整（每个自带门限的量都有对应登记记录）：**是**
- 复算与登记值不一致：无
- 产物内已存百分比与复算不一致：无

## 2 逐量审计表

| 方法 | 量 | 复算方式 | 复算 [%] | 登记 [%] | 算例门 [%] | 过 15%? | 过自身门? |
|---|---|---|---|---|---|---|---|
| `wcsph_delta_sph` | worst target wave-height error over the gauges | direct | 2.97 | 2.97 | 10 | 是 | 是 |
| `compressible_riemann` | max relative error of the Riemann invariants | direct | 0.777 | 0.777 | 5 | 是 | 是 |
| `heat_conduction_operator` | max normalised error vs the erfc similarity solution | direct | 0.000218 | 0.000218 | 3 | 是 | 是 |
| `csf_surface_tension` | delta_p error vs 2 sigma / R | direct | 5.6 | 5.6 | 15 | 是 | 是 |
| `csf_surface_tension` | band-mean curvature vs 1/R, plain (no delta_s weighting) | direct | 43.9 | - | - | - | - |
| `csf_surface_tension` | delta_s-weighted curvature error (dx = 3 mm rung) | direct | 23.8 | 23.8 | 15 | **否** | **否** |
| `csf_surface_tension` | cut-force error |F_SPH / sigma - 1| | direct | 5.99 | 5.99 | 15 | 是 | 是 |
| `density_diffusion` | worst decay-rate error over all three ladders | direct | 1.02 | 1.02 | 3 | 是 | 是 |
| `density_diffusion` | final-amplitude error after the finite run | direct | 0.482 | 0.482 | 3.5 | 是 | 是 |
| `isph_projection` | centreline velocity error vs the Poiseuille parabola (finest rung) | direct | 0.867 | 0.867 | 3 | 是 | 是 |
| `isph_projection` | momentum-budget wall shear vs rho g H / 2 (finest rung) | direct | 0.0458 | 0.0458 | 3 | 是 | 是 |
| `compression_regularization` | max interior-band hydrostatic pressure error | stored | 5.53 | 5.53 | 1 | 是 | **否** |
| `particle_shifting` | front-position deviation with shifting on vs off | direct | 8.83 | 8.83 | - | 是 | 是 |
| `wcsph_wall_no_slip` | interior L2 velocity error vs u = U y / H (finest rung) | direct | 5.65 | 5.65 | 5 | 是 | **否** |
| `wcsph_wall_no_slip` | RMS relative error including the two wall-adjacent rows | direct | 6 | - | - | - | - |
| `wcsph_wall_no_slip` | cross-channel mass-flux error |Q_SPH / Q_exact - 1| | direct | 7.16 | 7.16 | 5 | 是 | **否** |
| `wcsph_wall_no_slip` | wall-shear error at the moving wall |tau_SPH / tau_exact - 1| | direct | 32.6 | 32.6 | 5 | **否** | **否** |
| `transport_velocity_correction` | mean relative error of Z/a vs Martin & Moyce | direct | 10.8 | 10.8 | 15 | 是 | 是 |
| `transport_velocity_correction` | same quantity on the coarse rung | direct | 17 | 17 | 15 | **否** | **否** |

## 3 方法级汇总（claim vs 最差已登记量）

| 方法 | claimed bound（仅取过 15% 的记录）[%] | 最差已登记量 [%] | 差额 [%] |
|---|---|---|---|
| `compressible_riemann` | 0.777 | 0.777 | 0 |
| `compression_regularization` | 5.53 | 5.53 | 0 |
| `csf_surface_tension` | 5.99 | 23.8 | 17.8 |
| `density_diffusion` | 1.02 | 1.02 | 0 |
| `heat_conduction_operator` | 0.000218 | 0.000218 | 0 |
| `isph_projection` | 0.867 | 0.867 | 0 |
| `particle_shifting` | 8.83 | 8.83 | 0 |
| `transport_velocity_correction` | 10.8 | 17 | 6.15 |
| `wcsph_delta_sph` | 2.97 | 2.97 | 0 |
| `wcsph_wall_no_slip` | 7.16 | 32.6 | 25.5 |

## 4 产物自评状态

| 产物 | 自评 |
|---|---|
| `benchmarks/case_conduction_1d/result.json` | `True` |
| `benchmarks/case_sod_shock_compressible/result.json` | `None` |
| `benchmarks/couette/out/results.json` | `FAIL` |
| `benchmarks/csf_consistent/validation_report.json` | `None` |
| `benchmarks/density_diffusion_decay/decay_rate.json` | `pass` |
| `benchmarks/isph_channel/results.json` | `None` |
| `benchmarks/laplace_law/validation_report.json` | `None` |
| `benchmarks/particle_shifting/results.json` | `None` |
| `benchmarks/xval_sphinxsys_compression/records.json` | `None` |
| `docs/reports/wave_flume_numbers.json` | `True` |
| `docs/xval_sphinx_tvc/data/grid_dx003.json` | `None` |
| `docs/xval_sphinx_tvc/data/grid_dx004.json` | `None` |

## 5 声明了门限却没有测量的量（不变量：这些必须显式列出）

| 产物 | 方法 | 门限 | 限值 | 原因 |
|---|---|---|---|---|
| `benchmarks/couette/out/results.json` | `wcsph_wall_no_slip` | steady_drift | 0.01 | gate declared in gates{}, no value in ladder[].metrics and verdict.gated_results is empty |
| `benchmarks/couette/out/results.json` | `wcsph_wall_no_slip` | control_flux | 0.2 | same: declared but unmeasured, so the artefact's own FAIL verdict cannot be attributed to a gate from the stored data |
| `benchmarks/isph_channel/results.json` | `isph_projection` | density_fluctuation_max | 0.001 | dimensionless (density fluctuation fraction), not a percentage: outside the percent-only record schema |
| `docs/xval_sphinx_tvc/data/grid_dx003.json` | `transport_velocity_correction` | ledger_abs (MAE of Z/a) | 0.15 | absolute (dimensionless Z/a), not a percentage: outside the schema |
| `benchmarks/case_conduction_1d/result.json` | `heat_conduction_operator` | ladder rung / interface temperature / stencil match / k0 calibration | None | normalised, absolute or machine-precision gates; the percentage record banks the erfc error, the ladder-rung gate is the same quantity on a coarser rung (worst 0.0141 % < 3 %) |
