# TensorSPH 验证断言账本（generated — 勿手改）

> 由 `benchmarks/verification_ledger/build_ledger.py` 从 `benchmarks/**/*.json` 程序化生成；`tests/test_verification_ledger.py` 在重生成结果与提交副本不一致时失败。表内每个数字都能按「工件 → JSON 路径」溯源，零手抄。

## 1. 为什么需要这本账

同一个物理量（σ=7.2 N/m、R=20 mm 的 Young–Laplace 压差，解析值 360 Pa）在仓里有 5 份互相矛盾的机器可读证据，口径/容差/状态词汇各不相同：有的用 `p_core − p_band`（旧双标尺）、有的用绝对压强时间平均、有的拿 `p_center` 对真空当压差、有的用 `p_plateau − p_ambient`。这些数字被 `docs/` 跨文引用，却没有任何一处说明**哪个是当前值、哪个已作废、哪个被自己的量具 disqualify**。本账本把这件事机器化：

1. 每条 claim =（工件 + JSON 路径 + 读数 + 参考值 + 守护测试 + 欠账指针）；
2. 分级规则全仓唯一（§2），**不由各工件自述决定**；
3. §5 的 findings 记录证据本身的可检测缺陷：退化量具、自述 PASS 高于容差、同一物理量多读数、未登记工件、散文里未标注的作废数字。

## 2. 分级规则（机裁）

| 等级 | 判据 | 含义 |
|---|---|---|
| PASS | \|rel err\| ≤ 3% | 论文级 |
| MARGINAL | 3% < \|rel err\| ≤ 15% | 过平台线、不够论文级 |
| WITHHELD | \|rel err\| > 15%，工件未给出收敛性证伪 | 测得出但不可采信 |
| FAIL | \|rel err\| > 15% 且工件自证不收敛/不满足 | 已证伪 |
| WITHDRAWN | 工件被仓自身撤销（`superseded_by` / `withdrawn`） | 不得再引用 |

量具退化（`gauge_degenerate`：压差用真空当环境压）**一律** WITHHELD，无论其 rel error 多小。

## 3. 汇总

- claims **40** ／ 物理量 **20** ／ findings **19**
- 分级分布：PASS 7，MARGINAL 8，WITHHELD 13，FAIL 10，WITHDRAWN 2
- findings 分类：credit_from_failing_artifact 6，declared_vs_derived 1，inadmissible_run 3，quantity_spread 4，run_validity 4，uncovered_benchmark 1
- **credit 资格**：7 条 PASS / 8 条 MARGINAL，其中 **6** 条取自「工件自述 verdict = fail」的运行集（best-of-N / 阶梯不全，不可作为能力验证引用）；另有 6 条 credit 的工件根本没有声明 verdict（等级只能被读成「该读数自身合规」，不是「该能力被验证」）。

## 4. 断言表

| claim | 物理量 / 口径 | 参考 | 实测 | rel err | 等级 | 工件 → JSON 路径 | 守护测试 |
|---|---|---|---|---|---|---|---|
| `colour_surface.delta_s_integral` | colour-surface normalisation, integral |grad c| dr (exact: 1) | 1 - | 0.754545 - | -24.55 % | **WITHHELD** | `benchmarks/csf_consistent/curvature_audit.json` → `verdicts.delta_s_integral_dn` | `tests/test_csf_consistent.py` |
| `csf_force.sigma_eff_over_sigma.dx5mm` | effective surface tension from the half-droplet force integral, sigma_eff/sigma | 1 - | 0.704004 - | -29.60 % | **FAIL** | `benchmarks/laplace_pressure/validation_report.json` → `cases[0]` | `tests/test_laplace_pressure.py` |
| `curvature_operator.delta_s_weighted.dx3mm` | delta_s-weighted curvature kappa vs analytic 1/R (R = 20 mm) | 50 1/m | 64.1315 1/m | +28.26 % | **WITHHELD** | `benchmarks/csf_consistent/curvature_audit.json` → `rows[0].variants[0]` | `tests/test_csf_curvature_audit.py` |
| `dam_break.martin_moyce.mae` | dam-break front position vs Martin & Moyce (1952), mean abs error in Z/a | limit 0.15 | 0.572757 | — | **FAIL** | `benchmarks/dam_break/resolution_ladder.json` → `gates.experimental_comparison` | `tests/test_dam_break_experimental.py` |
| `density_diffusion.decay_rate.worst_of_ladder` | decay rate of a plane density wave vs the analytic gamma = D k^2, D = delta c0 h (Antuono, Colagrossi & Marrone 2012) | limit 0.03 | 0.010208 | — | **PASS** | `benchmarks/density_diffusion_decay/decay_rate.json` → `max_rate_relative_error` | `tests/test_density_diffusion_decay.py::test_the_diffusive_term_decays_a_plane_wave_at_the_analytic_rate` |
| `density_diffusion.finite_time.amplitude` | mode amplitude after a finite integration time vs rho0 A exp(-gamma t) | limit 0.035 | 0.00482 | — | **PASS** | `benchmarks/density_diffusion_decay/decay_rate.json` → `finite_time` | `tests/test_density_diffusion_decay.py::test_the_mode_amplitude_follows_the_exponential_over_a_finite_time` |
| `effective_sigma.sigma_eff_over_sigma.dx5mm` | effective surface tension from the half-droplet force integral, sigma_eff/sigma (R = 40 mm, curvature estimator: quadratic interface fit) | 1 - | 0.710625 - | -28.94 % | **FAIL** | `benchmarks/effective_sigma/out/report.json` → `rows[0].ratio` | `tests/test_effective_sigma.py::test_current_default_delivers_the_documented_ratio` |
| `laplace_droplet.merged_csf.8cases` | dp = sigma/R (2-D), sigma=0.072 N/m, R=0.1 m -> 0.72 Pa | 1 - | 3.19723 - | +219.72 % | **FAIL** | `benchmarks/laplace_droplet/results.json` → `verdict` | `tests/test_laplace_droplet.py::test_laplace_jump_current_state` |
| `laplace_report.laplace_pressure.Rdx10` | pressure balance of a static droplet at R/dx = 10 (code units) | 1.44 - | 1.82627 - | +26.82 % | **FAIL** | `benchmarks/laplace/report.json` → `checks.R/dx=10:laplace_pressure` | `tests/test_laplace_pressure.py` |
| `oscillating_droplet.frequency.lamb` | fundamental frequency of an inviscid 2-D droplet (Lamb omega_2) | 73.4847 rad/s | 41.4979 rad/s | -43.53 % | **WITHHELD** | `benchmarks/oscillating_droplet/validation_report.json` → `verdict.rows[0]` | `tests/test_oscillating_droplet.py` |
| `paper_uncertainty.band.dx3mm` | Young-Laplace jump (2-D), sigma=7.2 N/m, R=20 mm -> 360 Pa | 360 Pa | 249.626 Pa | -30.66 % | **WITHHELD** | `benchmarks/paper_laplace_uncertainty/out/results.json` → `rows[0].metrics.band` | `tests/test_paper_laplace_uncertainty.py` |
| `poiseuille.centre_velocity.finest` | centre velocity of a body-force-driven plane channel at steady state | 1.22625 m/s | 1.24476 m/s | +1.51 % | **PASS** | `benchmarks/poiseuille/out/results.json` → `results[2].fit.u_centre_measured` | `tests/test_poiseuille.py::test_steady_channel_flow_develops_the_poiseuille_parabola` |
| `poiseuille.wall_shear_ratio.finest` | no-slip wall shear stress of the same channel, relative to rho g H / 2 | 1 - | 0.502508 - | -49.75 % | **WITHHELD** | `benchmarks/poiseuille/out/results.json` → `results[2].wall_shear_ratio` | `tests/test_poiseuille.py::test_steady_channel_flow_develops_the_poiseuille_parabola` |
| `quiescence.flume_resting_noise.v_late_rms` | residual velocity of a flume at rest: RMS over the last quarter of max_i |v_i| on interior particles (m/s) | limit 0.06 | 0.079759 | — | **FAIL** | `benchmarks/startup_quiescence/report.json` → `findings[0]` | `tests/test_startup_quiescence.py` |
| `static_droplet.force_direction.WITHDRAWN` | sign/shape of the CSF force on a static droplet (not a physical gate) | 1 - | 0.173549 - | -82.65 % | **WITHDRAWN** | `benchmarks/static_droplet/validation_report.json` → `validation` | `tests/test_laplace_validation.py` |
| `surface_tension_ladder.laplace_error_finest` | Laplace error of the surface-tension-only ladder, finest level | 0.15 fraction | 2.51981 fraction | +251.98 % | **FAIL** | `benchmarks/surface_tension/laplace_ladder.json` → `gates.laplace_error_finest` | `tests/test_surface_tension_laplace.py` |
| `threed_hydrostatic.bottom_pressure.adami_best` | bottom pressure of a 3-D column at rest, p = rho0 g H | 1520.55 Pa | 1478.77 Pa | -2.75 % | **PASS** | `benchmarks/threed_hydrostatic/report.json` → `runs[0].pressure` | `tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria` |
| `threed_hydrostatic.bottom_pressure.adami_worst` | bottom pressure of a 3-D column at rest, p = rho0 g H | 1520.55 Pa | 1387.58 Pa | -8.74 % | **MARGINAL** | `benchmarks/threed_hydrostatic/report.json` → `runs[2].pressure` | `tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria` |
| `threed_hydrostatic.bottom_pressure.tait_wall_control` | bottom pressure of a 3-D column at rest, p = rho0 g H | 1520.55 Pa | 1136.93 Pa | -25.23 % | **WITHHELD** | `benchmarks/threed_hydrostatic/report.json` → `runs[3].pressure` | `tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria` |
| `threed_hydrostatic.profile.max_rel_error_core.adami_best` | worst per-layer relative error over the interior pressure profile of a 3-D column at rest | 0 % | 43.1741 % | +43.17 % | **FAIL** | `benchmarks/threed_hydrostatic/report.json` → `runs[0].pressure` | `tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria` |
| `threed_hydrostatic.profile.mean_rel_error_core.adami_best` | mean relative error of the interior (non-bottom, non-surface) pressure profile of a 3-D column at rest | 0 % | -11.9227 % | -11.92 % | **MARGINAL** | `benchmarks/threed_hydrostatic/report.json` → `runs[0].pressure` | `tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria` |
| `threed_kernel.integral_of_W` | volume integral of the 3-D cubic-spline kernel over its support | 1 - | 1 - | +0.00 % | **PASS** | `benchmarks/threed_kernel/report.json` → `quadrature.kernel_integral` | `tests/test_threed_kernel.py::test_kernel_integrates_to_one` |
| `threed_shear.decay_rate.finest` | decay exponent of u = A sin(ky) sin(kz) exp(-2 nu k^2 t), periodic 3-D cube, nu = 0.05 m^2/s, k = 2 pi / m | 3.94784 1/s | 3.84214 1/s | -2.68 % | **PASS** | `benchmarks/threed_shear/report.json` → `resolution_ladder[3].decay` | `tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria` |
| `threed_shear.decay_rate.raw_laplacian` | decay exponent of u = A sin(ky) sin(kz) exp(-2 nu k^2 t), periodic 3-D cube, nu = 0.05 m^2/s, k = 2 pi / m | 3.94784 1/s | 4.40032 1/s | +11.46 % | **MARGINAL** | `benchmarks/threed_shear/report.json` → `raw_vs_renormalised.raw.decay` | `tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria` |
| `threed_shear.decay_rate.worst_of_ladder` | decay exponent of u = A sin(ky) sin(kz) exp(-2 nu k^2 t), periodic 3-D cube, nu = 0.05 m^2/s, k = 2 pi / m | 3.94784 1/s | 3.44962 1/s | -12.62 % | **MARGINAL** | `benchmarks/threed_shear/report.json` → `resolution_ladder[0].decay` | `tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria` |
| `two_phase_laplace.smoothed_colour.dx2.5mm` | Laplace error on a closed two-phase interface (periodic, no free surface) | limit 0.15 | 0.046079 | — | **MARGINAL** | `benchmarks/surface_tension_laplace/validation_report.json` → `criteria.two_phase_laplace_error_at_dx_0.0025_lt_15pct` | `tests/test_surface_tension_laplace.py` |
| `young_laplace.absolute_tail.best_of_6` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 237.636 Pa | -33.99 % | **WITHHELD** | `benchmarks/laplace_pressure_floor/report.json` → `runs[2].absolute_time_average_tail` | `tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct` |
| `young_laplace.closure.closure_on_all_donors.dx1.5mm_c20` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 269.069 Pa | -25.26 % | **FAIL** | `benchmarks/laplace_pressure_closure/raw/relax_ladder_dx1.5mm_c20.json` → `cases[1]` | `tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder` |
| `young_laplace.closure.closure_on_all_donors.dx2mm_c20` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 287.473 Pa | -20.15 % | **FAIL** | `benchmarks/laplace_pressure_closure/raw/relax_ladder.json` → `cases[1]` | `tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder` |
| `young_laplace.closure.closure_on_interior_layers2.dx1.5mm_c20` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 347.73 Pa | -3.41 % | **MARGINAL** | `benchmarks/laplace_pressure_closure/raw/relax_ladder_dx1.5mm_c20.json` → `cases[2]` | `tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder` |
| `young_laplace.closure.closure_on_interior_layers2.dx2mm_c10` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 345.481 Pa | -4.03 % | **MARGINAL** | `benchmarks/laplace_pressure_closure/raw/relax_ladder_dx2mm_c10.json` → `cases[1]` | `tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder` |
| `young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 391.805 Pa | +8.83 % | **WITHHELD** | `benchmarks/laplace_pressure_closure/raw/relax_ladder.json` → `cases[2]` | `tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder` |
| `young_laplace.closure.closure_on_interior_layers2.dx2mm_c5` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 377.482 Pa | +4.86 % | **WITHHELD** | `benchmarks/laplace_pressure_closure/raw/relax_ladder_dx2mm_c5.json` → `cases[1]` | `tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder` |
| `young_laplace.closure.closure_on_interior_layers4.dx2mm_c20` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 391.805 Pa | +8.83 % | **WITHDRAWN** | `benchmarks/laplace_pressure_closure/raw/relax_ladder.json` → `cases[3]` | `tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder` |
| `young_laplace.csf_consistent.dx2mm` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 77.1912 Pa | -78.56 % | **WITHHELD** | `benchmarks/csf_consistent/validation_report.json` → `relax.laplace` | `tests/test_csf_consistent.py` |
| `young_laplace.csf_curvature_fit.dx2mm` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 44.5997 Pa | -87.61 % | **WITHHELD** | `benchmarks/csf_curvature_fit/validation_report.json` → `relax[0].laplace` | `tests/test_csf_curvature_fit.py` |
| `young_laplace.force_balance.flux_integral.dx1.5mm` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 333.145 Pa | -7.46 % | **MARGINAL** | `benchmarks/laplace_pressure_closure/report_force_balance.json` → `cases[1].force_side` | `tests/test_young_laplace_closure_ledger.py::test_force_balance_integral_is_registered_and_paper_grade` |
| `young_laplace.force_balance.flux_integral.dx2mm` | Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m | 360 Pa | 359.078 Pa | -0.26 % | **PASS** | `benchmarks/laplace_pressure_closure/report_force_balance.json` → `cases[0].force_side` | `tests/test_young_laplace_closure_ledger.py::test_force_balance_integral_is_registered_and_paper_grade` |
| `young_laplace.laplace_law.vacuum_gauge` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 380.158 Pa | +5.60 % | **WITHHELD** | `benchmarks/laplace_law/validation_report.json` → `laplace` | `tests/test_laplace_law.py` |
| `young_laplace.plateau_definition` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 530.313 Pa | +47.31 % | **WITHHELD** | `benchmarks/laplace_plateau/validation_report.json` → `verdict` | `tests/test_laplace_plateau.py` |

## 5. findings（证据缺陷清单）

- **declared_vs_derived**（`high`）`young_laplace.laplace_law.vacuum_gauge` — artifact declares PASS, ledger derives WITHHELD: vacuum-gauge readout (additive |p| offset, not a physical dp)
- **run_validity**（`high`）`oscillating_droplet.frequency.lamb` — validity gate says INVALID (containment / Mach / estimator / fit / null-control): the run had stopped being a liquid before the reading was taken, so the reading is graded WITHHELD instead of FAIL — see docs/verification_claim_validity.md
- **run_validity**（`high`）`young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` — validity gate says INVALID (containment / Mach / estimator / fit / null-control): the run had stopped being a liquid before the reading was taken, so the reading is graded WITHHELD instead of FAIL — see docs/verification_claim_validity.md
- **run_validity**（`high`）`young_laplace.closure.closure_on_interior_layers2.dx2mm_c5` — validity gate says INVALID (containment / Mach / estimator / fit / null-control): the run had stopped being a liquid before the reading was taken, so the reading is graded WITHHELD instead of FAIL — see docs/verification_claim_validity.md
- **run_validity**（`high`）`young_laplace.closure.closure_on_interior_layers4.dx2mm_c20` — validity gate says INVALID (containment / Mach / estimator / fit / null-control): the run had stopped being a liquid before the reading was taken, so the reading is graded WITHHELD instead of FAIL — see docs/verification_claim_validity.md
- **inadmissible_run**（`high`）`young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` — +8.83 % is inside the 15 % band, but the run violates the admissibility gate (v_max/v_ref < 0.1 (Adami, Ihmsen & Adams 2012): 0.388129 >= 0.1): the residual spurious current, not the physics, sets the agreement, so the reading is graded WITHHELD and may not be quoted as a pass
- **inadmissible_run**（`high`）`young_laplace.closure.closure_on_interior_layers2.dx2mm_c5` — +4.86 % is inside the 15 % band, but the run violates the admissibility gate (v_max/v_ref < 0.1 (Adami, Ihmsen & Adams 2012): 0.103857 >= 0.1): the residual spurious current, not the physics, sets the agreement, so the reading is graded WITHHELD and may not be quoted as a pass
- **inadmissible_run**（`high`）`young_laplace.closure.closure_on_interior_layers4.dx2mm_c20` — +8.83 % is inside the 15 % band, but the run violates the admissibility gate (v_max/v_ref < 0.1 (Adami, Ihmsen & Adams 2012): 0.388129 >= 0.1): the residual spurious current, not the physics, sets the agreement, so the reading is graded WITHHELD and may not be quoted as a pass
- **credit_from_failing_artifact**（`high`）`threed_hydrostatic.bottom_pressure.adami_best` — -2.75 % is credited PASS, but the artifact `benchmarks/threed_hydrostatic/report.json` declares `verdict = fail` (failing criteria: bottom_pressure_within_15pct_all, centre_of_mass_drift_below_1pct, column_settled_window_drift_below_2pct, compressibility_helps_as_predicted, interior_pressure_within_15pct_best): this is a best-of-N / finest-rung selection out of a run set that does not pass, so it may not be quoted as verification of the capability -- see docs/verification_ledger_overcredit_m6510.md
- **credit_from_failing_artifact**（`high`）`threed_hydrostatic.bottom_pressure.adami_worst` — -8.74 % is credited MARGINAL, but the artifact `benchmarks/threed_hydrostatic/report.json` declares `verdict = fail` (failing criteria: bottom_pressure_within_15pct_all, centre_of_mass_drift_below_1pct, column_settled_window_drift_below_2pct, compressibility_helps_as_predicted, interior_pressure_within_15pct_best): this is a best-of-N / finest-rung selection out of a run set that does not pass, so it may not be quoted as verification of the capability -- see docs/verification_ledger_overcredit_m6510.md
- **credit_from_failing_artifact**（`high`）`threed_hydrostatic.profile.mean_rel_error_core.adami_best` — -11.92 % is credited MARGINAL, but the artifact `benchmarks/threed_hydrostatic/report.json` declares `verdict = fail` (failing criteria: bottom_pressure_within_15pct_all, centre_of_mass_drift_below_1pct, column_settled_window_drift_below_2pct, compressibility_helps_as_predicted, interior_pressure_within_15pct_best): this is a best-of-N / finest-rung selection out of a run set that does not pass, so it may not be quoted as verification of the capability -- see docs/verification_ledger_overcredit_m6510.md
- **credit_from_failing_artifact**（`high`）`threed_shear.decay_rate.finest` — -2.68 % is credited PASS, but the artifact `benchmarks/threed_shear/report.json` declares `verdict = fail` (missing cases: h1.8): this is a best-of-N / finest-rung selection out of a run set that does not pass, so it may not be quoted as verification of the capability -- see docs/verification_ledger_overcredit_m6510.md
- **credit_from_failing_artifact**（`high`）`threed_shear.decay_rate.raw_laplacian` — +11.46 % is credited MARGINAL, but the artifact `benchmarks/threed_shear/report.json` declares `verdict = fail` (missing cases: h1.8): this is a best-of-N / finest-rung selection out of a run set that does not pass, so it may not be quoted as verification of the capability -- see docs/verification_ledger_overcredit_m6510.md
- **credit_from_failing_artifact**（`high`）`threed_shear.decay_rate.worst_of_ladder` — -12.62 % is credited MARGINAL, but the artifact `benchmarks/threed_shear/report.json` declares `verdict = fail` (missing cases: h1.8): this is a best-of-N / finest-rung selection out of a run set that does not pass, so it may not be quoted as verification of the capability -- see docs/verification_ledger_overcredit_m6510.md
- **quantity_spread**（`high`）`threed_hydrostatic_bottom_pressure` — 3 active readings of one quantity span -25.2 % … -2.7 % (spread 22.5 pt): threed_hydrostatic.bottom_pressure.adami_best=-2.7 %, threed_hydrostatic.bottom_pressure.adami_worst=-8.7 %, threed_hydrostatic.bottom_pressure.tait_wall_control=-25.2 % — definitions differ (gauge / scheme / resolution), so no single number may be quoted without naming its definition.
- **quantity_spread**（`high`）`threed_shear_decay_rate_2nuk2` — 3 active readings of one quantity span -12.6 % … +11.5 % (spread 24.1 pt): threed_shear.decay_rate.finest=-2.7 %, threed_shear.decay_rate.raw_laplacian=+11.5 %, threed_shear.decay_rate.worst_of_ladder=-12.6 % — definitions differ (gauge / scheme / resolution), so no single number may be quoted without naming its definition.
- **quantity_spread**（`high`）`young_laplace_jump_sigma0.072_R0.1` — 2 active readings of one quantity span +219.7 % … +252.0 % (spread 32.3 pt): laplace_droplet.merged_csf.8cases=+219.7 %, surface_tension_ladder.laplace_error_finest=+252.0 % — definitions differ (gauge / scheme / resolution), so no single number may be quoted without naming its definition.
- **quantity_spread**（`high`）`young_laplace_jump_sigma7.2_R0.02` — 14 active readings of one quantity span -87.6 % … +47.3 % (spread 134.9 pt): paper_uncertainty.band.dx3mm=-30.7 %, young_laplace.absolute_tail.best_of_6=-34.0 %, young_laplace.closure.closure_on_all_donors.dx1.5mm_c20=-25.3 %, young_laplace.closure.closure_on_all_donors.dx2mm_c20=-20.1 %, young_laplace.closure.closure_on_interior_layers2.dx1.5mm_c20=-3.4 %, young_laplace.closure.closure_on_interior_layers2.dx2mm_c10=-4.0 %, young_laplace.closure.closure_on_interior_layers2.dx2mm_c20=+8.8 %, young_laplace.closure.closure_on_interior_layers2.dx2mm_c5=+4.9 %, young_laplace.csf_consistent.dx2mm=-78.6 %, young_laplace.csf_curvature_fit.dx2mm=-87.6 %, young_laplace.force_balance.flux_integral.dx1.5mm=-7.5 %, young_laplace.force_balance.flux_integral.dx2mm=-0.3 %, young_laplace.laplace_law.vacuum_gauge=+5.6 %, young_laplace.plateau_definition=+47.3 % — definitions differ (gauge / scheme / resolution), so no single number may be quoted without naming its definition.
- **uncovered_benchmark**（`medium`）`benchmarks/ice_floe_added_mass` — benchmark harness (run.py) has no ledger claim and no waiver: nothing it produces is graded, so a regression there is invisible and its published numbers are not reproducible from the repo

## 6. 读数明细（口径决定数字）

### `colour_surface.delta_s_integral` — WITHHELD

- 物理量：colour-surface normalisation, integral |grad c| dr (exact: 1)
- 参考：1 -（CSF requires the surface delta to integrate to one）
- 实测：0.754545 -
  - 口径：coarsest level of the audit ladder (dx = 3 mm)
- 相对误差：-24.55 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/csf_consistent/curvature_audit.json` → `verdicts.delta_s_integral_dn`
  - 守护测试：`tests/test_csf_consistent.py`
- 欠账指针：`docs/verification_ledger.md`
- 注：Root cause of the force-path deficit: the delta function carries only 0.755 of unit mass, and the section that does exist is ~28 % overweight, so the cut force comes out ~0.70 sigma. This is the quantity to fix, not the pressure assembly.

### `csf_force.sigma_eff_over_sigma.dx5mm` — FAIL

- 物理量：effective surface tension from the half-droplet force integral, sigma_eff/sigma
- 参考：1 -（sigma_eff = sigma (force integral / 2 sigma)）
- 实测：0.704004 -
  - 口径：|F_half| / (2 sigma) at R/h = 6.15
- 相对误差：-29.60 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure/validation_report.json` → `cases[0]`
  - 守护测试：`tests/test_laplace_pressure.py`
- 欠账指针：`docs/verification_ledger.md`
- 注：Force path is 30 % low while being mirror/rotation symmetric to 1e-14 (gates pass): the deficit lives in the colour-surface delta, not in the discrete operators.

### `curvature_operator.delta_s_weighted.dx3mm` — WITHHELD

- 物理量：delta_s-weighted curvature kappa vs analytic 1/R (R = 20 mm)
- 参考：50 1/m（analytic curvature of a circle）
- 实测：64.1315 1/m
  - 口径：delta_s-weighted kappa, V0 baseline (m2447: kappa from sharp colour)
- 相对误差：+28.26 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/csf_consistent/curvature_audit.json` → `rows[0].variants[0]`
  - 守护测试：`tests/test_csf_curvature_audit.py`
- 欠账指针：`backlog_update_m2971.md#P1`
- 注：Shipped operator: 21.5-28.3 % high over the ladder and non-monotone (monotone_convergence_of_shipped=False); the exact-normal control is also outside 15 % at the coarsest levels, so part of the error is the delta_s weighting itself, not the fit.

### `dam_break.martin_moyce.mae` — FAIL

- 物理量：dam-break front position vs Martin & Moyce (1952), mean abs error in Z/a
- 参考：0.15 Z/a（acceptance limit on the mean absolute error）
- 实测：0.572757 Z/a
  - 口径：mean absolute error over 20 digitised points
- 相对误差：—（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/dam_break/resolution_ladder.json` → `gates.experimental_comparison`
  - 守护测试：`tests/test_dam_break_experimental.py`
- 欠账指针：`docs/verification_ledger.md`
- 注：RMSE 0.699 Z/a, max 1.618 Z/a; fraction inside experimental uncertainty 0.00. Wave breaking is a 3-D/viscous/non-hydrostatic problem: this is not a surface-tension defect.

### `density_diffusion.decay_rate.worst_of_ladder` — PASS

- 物理量：decay rate of a plane density wave vs the analytic gamma = D k^2, D = delta c0 h (Antuono, Colagrossi & Marrone 2012)
- 参考：0.03 fraction（gates.rate_relative_error on |gamma_measured/gamma_theory - 1|）
- 实测：0.010208 fraction
  - 口径：worst |gamma_measured/gamma_theory - 1| over the resolution / wavenumber / smoothing-length ladders
- 相对误差：—（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**PASS**
- 工件自身 verdict：**pass**（`verdict`）
- 溯源：`benchmarks/density_diffusion_decay/decay_rate.json` → `max_rate_relative_error`
  - 守护测试：`tests/test_density_diffusion_decay.py::test_the_diffusive_term_decays_a_plane_wave_at_the_analytic_rate`
- 欠账指针：`docs/verification_density_diffusion_decay.md`
- 注：D = delta c0 h is not fitted: delta, c0 and h = h_factor dx are solver inputs, so a sign flip, a factor of two or a dx/h confusion would miss by far more than the gate. Before this registration the term had only conservation and smoothing tests, either of which such a defect would pass.

### `density_diffusion.finite_time.amplitude` — PASS

- 物理量：mode amplitude after a finite integration time vs rho0 A exp(-gamma t)
- 参考：0.035 fraction（gates.finite_time_relative_error）
- 实测：0.00482 fraction
  - 口径：|rho(x,t)/rho0 - 1 - A exp(-gamma t)| amplitude error after the run
- 相对误差：—（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**PASS**
- 工件自身 verdict：**pass**（`verdict`）
- 溯源：`benchmarks/density_diffusion_decay/decay_rate.json` → `finite_time`
  - 守护测试：`tests/test_density_diffusion_decay.py::test_the_mode_amplitude_follows_the_exponential_over_a_finite_time`
- 欠账指针：`docs/verification_density_diffusion_decay.md`

### `effective_sigma.sigma_eff_over_sigma.dx5mm` — FAIL

- 物理量：effective surface tension from the half-droplet force integral, sigma_eff/sigma (R = 40 mm, curvature estimator: quadratic interface fit)
- 参考：1 -（sigma_eff = sigma (|F_half| / (2 sigma))）
- 实测：0.710625 -
  - 口径：|F_half| / (2 sigma) at R/h = 6.15
- 相对误差：-28.94 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/effective_sigma/out/report.json` → `rows[0].ratio`
  - 守护测试：`tests/test_effective_sigma.py::test_current_default_delivers_the_documented_ratio`
- 欠账指针：`docs/verification_ledger.md`
- 注：Independent replication of claim csf_force.sigma_eff_over_sigma.dx5mm at twice the radius with a different curvature estimator: the two agree to ~1 pt, so the ~30 % force deficit is not a radius or operator artefact. Same root cause as colour_surface.delta_s_integral (the colour-surface delta carries ~0.75 of unit mass).

### `laplace_droplet.merged_csf.8cases` — FAIL

- 物理量：dp = sigma/R (2-D), sigma=0.072 N/m, R=0.1 m -> 0.72 Pa
- 参考：1 -（dp_model/dp_theory = 1）
- 实测：3.19723 -
  - 口径：worst of 8 lattice/jitter cases (0 of 8 within 15 %)
- 相对误差：+219.72 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：**fail**（`verdict`）
- 溯源：`benchmarks/laplace_droplet/results.json` → `verdict`
  - 守护测试：`tests/test_laplace_droplet.py::test_laplace_jump_current_state`
- 欠账指针：`docs/verification_ledger.md`
- 注：Legacy verdict of the merged CSF model: no case within tolerance and no convergence under refinement. NOTE the fixture-level tests in tests/test_laplace_droplet.py now record 0.70-0.95 of sigma/R on the same geometry, i.e. this artifact predates the consistent-scheme refactor even though it has no superseded_by pointer.

### `laplace_report.laplace_pressure.Rdx10` — FAIL

- 物理量：pressure balance of a static droplet at R/dx = 10 (code units)
- 参考：1.44 -（sigma/R in code units）
- 实测：1.82627 -
  - 口径：measured dp in code units
- 相对误差：+26.82 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace/report.json` → `checks.R/dx=10:laplace_pressure`
  - 守护测试：`tests/test_laplace_pressure.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Platform-level verdict at generation time: 9 of 11 criteria failed.

### `oscillating_droplet.frequency.lamb` — WITHHELD

- 物理量：fundamental frequency of an inviscid 2-D droplet (Lamb omega_2)
- 参考：73.4847 rad/s（Lamb (1932) omega^2 = n(n^2-1) sigma/(rho R^3), n = 2）
- 实测：41.4979 rad/s
  - 口径：FFT peak of the semi-axis oscillation, dx = 2 mm
- 相对误差：-43.53 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**WITHHELD**
- 工件自身 verdict：**fail**（`verdict`）
- 溯源：`benchmarks/oscillating_droplet/validation_report.json` → `verdict.rows[0]`
  - 守护测试：`tests/test_oscillating_droplet.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：The coupling to the force path does NOT hold for this number: with sigma_eff/sigma = 0.7040 the capillary scaling omega ~ sqrt(sigma_eff) predicts -16.10 %, while the quoted reading is -43.53 % (residual 27.43 pt > 15 pt). The reason is upstream: the recorded run is dead — rms radius 3531 R against its own 1.3 R gate, peak speed Mach 9092 — so the reading is graded WITHHELD (evidence insufficient), not FAIL (measured defect). The one sibling rung that stays coherent (dx = 2 mm) does reproduce the prediction, see docs/verification_claim_validity.md.

### `paper_uncertainty.band.dx3mm` — WITHHELD

- 物理量：Young-Laplace jump (2-D), sigma=7.2 N/m, R=20 mm -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：249.626 Pa
  - 口径：band definition, mean of 5 realisations
- 相对误差：-30.66 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/paper_laplace_uncertainty/out/results.json` → `rows[0].metrics.band`
  - 守护测试：`tests/test_paper_laplace_uncertainty.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Largest R/dx = 6.7 of the sweep; 95 % CI +/-9.45 pt.

### `poiseuille.centre_velocity.finest` — PASS

- 物理量：centre velocity of a body-force-driven plane channel at steady state
- 参考：1.22625 m/s（u_max = g H^2 / (8 nu), H = 0.1 m, nu = 0.01 m^2/s, g = 9.81 m/s^2）
- 实测：1.24476 m/s
  - 口径：parabola fitted to the time-averaged profile at H/dx = 16
- 相对误差：+1.51 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`None`；本账等级：**PASS**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/poiseuille/out/results.json` → `results[2].fit.u_centre_measured`
  - 守护测试：`tests/test_poiseuille.py::test_steady_channel_flow_develops_the_poiseuille_parabola`
- 欠账指针：`docs/verification_ledger_coverage_m6453.md`

### `poiseuille.wall_shear_ratio.finest` — WITHHELD

- 物理量：no-slip wall shear stress of the same channel, relative to rho g H / 2
- 参考：1 -（tau_wall = rho nu du/dy|_wall = rho g H / 2 = 490.5 Pa）
- 实测：0.502508 -
  - 口径：measured wall friction divided by the analytic value at H/dx = 16
- 相对误差：-49.75 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`None`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/poiseuille/out/results.json` → `results[2].wall_shear_ratio`
  - 守护测试：`tests/test_poiseuille.py::test_steady_channel_flow_develops_the_poiseuille_parabola`
- 欠账指针：`docs/verification_ledger_coverage_m6453.md`
- 注：WITHHELD rather than FAIL: the two readings inside one artifact disagree. The measured centre velocity implies tau = 4 rho nu u_max / H = 498 Pa, while the wall-band readout says 246 Pa, so the wall-band estimator (not necessarily the flow) carries the deficit and the number may not be quoted as a physical gate.

### `quiescence.flume_resting_noise.v_late_rms` — FAIL

- 物理量：residual velocity of a flume at rest: RMS over the last quarter of max_i |v_i| on interior particles (m/s)
- 参考：0.06 m/s（gate_parameters.v_floor_m_s (resting-noise ceiling)）
- 实测：0.079759 m/s
  - 口径：shipped flume resting column, paddle frozen at rest (0.0798 m/s interior RMS)
- 相对误差：—（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/startup_quiescence/report.json` → `findings[0]`
  - 守护测试：`tests/test_startup_quiescence.py`
- 欠账指针：`docs/verification_startup_quiescence.md`
- 注：OPEN defect register entry F1: the shipped resting flume reads 0.0798 m/s against a 0.06 m/s ceiling, i.e. 60 % of the wave orbital velocity (signal-to-noise ~1.7:1). The ablation 'flume resting column (paddle frozen at rest), overrides density_diffusion=0.0,viscosity=0.01' reaches 0.0072 m/s (0.09x calmer), so the noise is controlled by density diffusion + viscosity, not by the boundary model — see docs/verification_startup_quiescence.md §5.

### `static_droplet.force_direction.WITHDRAWN` — WITHDRAWN

- 物理量：sign/shape of the CSF force on a static droplet (not a physical gate)
- 参考：1 -（no physical reference: the criterion was self-referential）
- 实测：0.173549 -
  - 口径：average radial dot product of the surface-particle force
- 相对误差：-82.65 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHDRAWN`；本账等级：**WITHDRAWN**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/static_droplet/validation_report.json` → `validation`
  - 守护测试：`tests/test_laplace_validation.py`
- 欠账指针：`backlog_update_m2971.md#P2`；被 `benchmarks/laplace_pressure/validation_report.json` 取代
- 注：Kept on purpose: this artifact used to PASS with a criterion no experiment could falsify, and was withdrawn after audit. Recording it stops the old PASS from being re-cited.

### `surface_tension_ladder.laplace_error_finest` — FAIL

- 物理量：Laplace error of the surface-tension-only ladder, finest level
- 参考：0.15 fraction（acceptance limit on |dp_model/dp_theory - 1|）
- 实测：2.51981 fraction
  - 口径：finest-level relative error
- 相对误差：+251.98 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：**fail**（`verdict`）
- 溯源：`benchmarks/surface_tension/laplace_ladder.json` → `gates.laplace_error_finest`
  - 守护测试：`tests/test_surface_tension_laplace.py`
- 欠账指针：`docs/verification_ledger.md`
- 注：Observed convergence order -1.633 (negative: the error grows under refinement) -> this ladder's operator is not convergent.

### `threed_hydrostatic.bottom_pressure.adami_best` — PASS

- 物理量：bottom pressure of a 3-D column at rest, p = rho0 g H
- 参考：1520.55 Pa（rho0 g H, H = nz dx）
- 实测：1478.77 Pa
  - 口径：mean p of the deepest fluid layer; adami-c0_20-nz16, dx = 0.01 m, 1846 steps
- 相对误差：-2.75 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**PASS**
- 工件自身 verdict：**fail**（`verdict`；未通过项：bottom_pressure_within_15pct_all、centre_of_mass_drift_below_1pct、column_settled_window_drift_below_2pct、compressibility_helps_as_predicted、interior_pressure_within_15pct_best）
- 溯源：`benchmarks/threed_hydrostatic/report.json` → `runs[0].pressure`
  - 守护测试：`tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：Per-layer relative errors alternate between ~-6 % and up to -43 % toward the free surface: a particle lattice supports an odd-even mode in the pressure field. The bottom layer, which is what holds the load, is quoted here; the odd-even amplitude is recorded in the artifact so the two are never conflated.

### `threed_hydrostatic.bottom_pressure.adami_worst` — MARGINAL

- 物理量：bottom pressure of a 3-D column at rest, p = rho0 g H
- 参考：1520.55 Pa（rho0 g H, H = nz dx）
- 实测：1387.58 Pa
  - 口径：worst of the 5 Adami runs on the bottom layer; adami-c0_5-nz16, dx = 0.01 m
- 相对误差：-8.74 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**MARGINAL**
- 工件自身 verdict：**fail**（`verdict`；未通过项：bottom_pressure_within_15pct_all、centre_of_mass_drift_below_1pct、column_settled_window_drift_below_2pct、compressibility_helps_as_predicted、interior_pressure_within_15pct_best）
- 溯源：`benchmarks/threed_hydrostatic/report.json` → `runs[2].pressure`
  - 守护测试：`tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：Complement of `threed_hydrostatic.bottom_pressure.adami_best`: the 5 Adami runs span -8.74 % to -2.75 % on the same collapse quantity, so the single best value is a selection, not a property of the wall model alone (it tracks c0 and therefore the density error, as Adami et al. 2012 predict).

### `threed_hydrostatic.bottom_pressure.tait_wall_control` — WITHHELD

- 物理量：bottom pressure of a 3-D column at rest, p = rho0 g H
- 参考：1520.55 Pa（rho0 g H, H = nz dx）
- 实测：1136.93 Pa
  - 口径：same case with the wall pressure taken from the wall's own summation density instead of the Adami extrapolation (tait-c0_10-nz16)
- 相对误差：-25.23 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：**fail**（`verdict`；未通过项：bottom_pressure_within_15pct_all、centre_of_mass_drift_below_1pct、column_settled_window_drift_below_2pct、compressibility_helps_as_predicted、interior_pressure_within_15pct_best）
- 溯源：`benchmarks/threed_hydrostatic/report.json` → `runs[3].pressure`
  - 守护测试：`tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：Control run, expected to fail: without the Adami extrapolation the wall under-pressurises, the column keeps sinking and the settling-window drift reaches tens of percent. Recorded so the wall model is never assumed to be optional decoration.

### `threed_hydrostatic.profile.max_rel_error_core.adami_best` — FAIL

- 物理量：worst per-layer relative error over the interior pressure profile of a 3-D column at rest
- 参考：0 %（exact profile p = rho0 g (H - z): target error 0 %）
- 实测：43.1741 %
  - 口径：max over the 13 core layers; adami-c0_20-nz16
- 相对误差：+43.17 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：**fail**（`verdict`；未通过项：bottom_pressure_within_15pct_all、centre_of_mass_drift_below_1pct、column_settled_window_drift_below_2pct、compressibility_helps_as_predicted、interior_pressure_within_15pct_best）
- 溯源：`benchmarks/threed_hydrostatic/report.json` → `runs[0].pressure`
  - 守护测试：`tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：The odd-even pressure mode of the particle lattice: adjacent layers alternate between ~-6 % and up to -43 %, so any single-layer reading of this benchmark is a sample of a mode, not a converged profile.

### `threed_hydrostatic.profile.mean_rel_error_core.adami_best` — MARGINAL

- 物理量：mean relative error of the interior (non-bottom, non-surface) pressure profile of a 3-D column at rest
- 参考：0 %（exact profile p = rho0 g (H - z): target error 0 %）
- 实测：-11.9227 %
  - 口径：mean of the per-layer relative errors over the 13 core layers; adami-c0_20-nz16
- 相对误差：-11.92 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**MARGINAL**
- 工件自身 verdict：**fail**（`verdict`；未通过项：bottom_pressure_within_15pct_all、centre_of_mass_drift_below_1pct、column_settled_window_drift_below_2pct、compressibility_helps_as_predicted、interior_pressure_within_15pct_best）
- 溯源：`benchmarks/threed_hydrostatic/report.json` → `runs[0].pressure`
  - 守护测试：`tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：The bottom-layer reading above is one point of a profile that is NOT within the band everywhere: the interior is systematically low, which is why the artifact's own criterion `interior_pressure_within_15pct_best` is false. Registered so the PASS on the bottom layer cannot be read as 'the 3-D hydrostatic profile is verified'.

### `threed_kernel.integral_of_W` — PASS

- 物理量：volume integral of the 3-D cubic-spline kernel over its support
- 参考：1 -（int W dV = 1 (cubic-spline normalisation)）
- 实测：1 -
  - 口径：quadrature of W over the support at h = 0.013
- 相对误差：+0.00 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**PASS**
- 工件自身 verdict：**pass**（`verdict`）
- 溯源：`benchmarks/threed_kernel/report.json` → `quadrature.kernel_integral`
  - 守护测试：`tests/test_threed_kernel.py::test_kernel_integrates_to_one`
- 欠账指针：`docs/verification_threed.md`
- 注：Second moment of laplacian W = 6.000000000 against the analytic 6, and laplacian W(0) matches its analytic limit to 1e-12 relative: this is the one quantity in the repo that is verified at round-off level rather than on the 3 % ladder.

### `threed_shear.decay_rate.finest` — PASS

- 物理量：decay exponent of u = A sin(ky) sin(kz) exp(-2 nu k^2 t), periodic 3-D cube, nu = 0.05 m^2/s, k = 2 pi / m
- 参考：3.94784 1/s（2 nu k^2 (exact)）
- 实测：3.84214 1/s
  - 口径：least-squares slope of ln a(t), a = 4 <u_x sin(ky) sin(kz)>; n = 24^3 particles, k h = 0.340, second-moment renormalised viscous operator
- 相对误差：-2.68 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**PASS**
- 工件自身 verdict：**fail**（`verdict`；缺失用例：h1.8）
- 溯源：`benchmarks/threed_shear/report.json` → `resolution_ladder[3].decay`
  - 守护测试：`tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：Renormalising the Laplacian by its discrete second moment (K = 2.347 instead of the continuum 2 on a cubic lattice with h = 1.3 dx) is what brings this inside the 3 % band.

### `threed_shear.decay_rate.raw_laplacian` — MARGINAL

- 物理量：decay exponent of u = A sin(ky) sin(kz) exp(-2 nu k^2 t), periodic 3-D cube, nu = 0.05 m^2/s, k = 2 pi / m
- 参考：3.94784 1/s（2 nu k^2 (exact)）
- 实测：4.40032 1/s
  - 口径：same run with the raw second-derivative Laplacian (n = 20^3, k h = 0.408)
- 相对误差：+11.46 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**MARGINAL**
- 工件自身 verdict：**fail**（`verdict`；缺失用例：h1.8）
- 溯源：`benchmarks/threed_shear/report.json` → `raw_vs_renormalised.raw.decay`
  - 守护测试：`tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：The raw operator is 17 % too strong for a locally quadratic field on a cubic lattice; the shear wave converts that into a decay-rate bias of the same order. Kept as the A/B partner.

### `threed_shear.decay_rate.worst_of_ladder` — MARGINAL

- 物理量：decay exponent of u = A sin(ky) sin(kz) exp(-2 nu k^2 t), periodic 3-D cube, nu = 0.05 m^2/s, k = 2 pi / m
- 参考：3.94784 1/s（2 nu k^2 (exact)）
- 实测：3.44962 1/s
  - 口径：coarsest rung of the same ladder, n = 12^3 particles, k h = 0.681, second-moment renormalised viscous operator
- 相对误差：-12.62 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**MARGINAL**
- 工件自身 verdict：**fail**（`verdict`；缺失用例：h1.8）
- 溯源：`benchmarks/threed_shear/report.json` → `resolution_ladder[0].decay`
  - 守护测试：`tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria`
- 欠账指针：`docs/verification_threed.md`
- 注：Complement of `threed_shear.decay_rate.finest`: the ladder runs -12.62 % (n = 12) to -2.68 % (n = 24). Quoting only the finest rung hides that the coarsest one is outside the paper band, and the artifact's own h/dx sweep is incomplete (see missing_cases).

### `two_phase_laplace.smoothed_colour.dx2.5mm` — MARGINAL

- 物理量：Laplace error on a closed two-phase interface (periodic, no free surface)
- 参考：0 fraction（exact Laplace balance on the closed interface）
- 实测：0.046079 fraction
  - 口径：|dp_measured/dp_theory - 1| at dx = 2.5 mm, smoothed colour field
- 相对误差：—（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**MARGINAL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/surface_tension_laplace/validation_report.json` → `criteria.two_phase_laplace_error_at_dx_0.0025_lt_15pct`
  - 守护测试：`tests/test_surface_tension_laplace.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Decisive control: with a resolved closed interface the very same CSF chain is 4-5 % accurate at both dx = 2.5 and 5 mm and the error shrinks under refinement, so the operator is not the defect. The one-sided free-surface readings above are.

### `young_laplace.absolute_tail.best_of_6` — WITHHELD

- 物理量：dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：237.636 Pa
  - 口径：p_core - p_ambient (absolute pressure, no vacuum gauge), tail time average; dx=0.002 m, c0=5.0 m/s
- 相对误差：-33.99 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_floor/report.json` → `runs[2].absolute_time_average_tail`
  - 守护测试：`tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Best of 6 runs (c0 20/5 x dx 0.002/0.0015 plus controls). Reproduces backlog P0: the residual is the free-surface pressure closure, not the curvature operator.

### `young_laplace.closure.closure_on_all_donors.dx1.5mm_c20` — FAIL

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：269.069 Pa
  - 口径：absolute <p_core> over the capillary tail average, closure_on_all_donors, dx = 1.5 mm, c0 = 20 m/s, 642 particles, 1710 steps
- 相对误差：-25.26 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/raw/relax_ladder_dx1.5mm_c20.json` → `cases[1]`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired into ``droplet_impact.rhs``. The closure-off control of the same lattice reads -38.10 %, i.e. this variant recovers +12.84 points of the deficit. Out of band at v_max/v_ref = 0.019. Overall the closure is PROVISIONAL: it is admissible and in band at two of the four tested (dx, c0) points, not at all of them.

### `young_laplace.closure.closure_on_all_donors.dx2mm_c20` — FAIL

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：287.473 Pa
  - 口径：absolute <p_core> over the capillary tail average, closure_on_all_donors, dx = 2 mm, c0 = 20 m/s, 362 particles, 1283 steps
- 相对误差：-20.15 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/raw/relax_ladder.json` → `cases[1]`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired into ``droplet_impact.rhs``. The closure-off control of the same lattice reads -37.88 %, i.e. this variant recovers +17.73 points of the deficit. Out of band at v_max/v_ref = 0.020. Overall the closure is PROVISIONAL: it is admissible and in band at two of the four tested (dx, c0) points, not at all of them.

### `young_laplace.closure.closure_on_interior_layers2.dx1.5mm_c20` — MARGINAL

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：347.73 Pa
  - 口径：absolute <p_core> over the capillary tail average, closure_on_interior_layers2, dx = 1.5 mm, c0 = 20 m/s, 642 particles, 1710 steps
- 相对误差：-3.41 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**MARGINAL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/raw/relax_ladder_dx1.5mm_c20.json` → `cases[2]`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired into ``droplet_impact.rhs``. The closure-off control of the same lattice reads -38.10 %, i.e. this variant recovers +34.69 points of the deficit. Admissible (v_max/v_ref = 0.090 < 0.1) and inside the 15 % band, but outside the 3 % paper band. Overall the closure is PROVISIONAL: it is admissible and in band at two of the four tested (dx, c0) points, not at all of them.

### `young_laplace.closure.closure_on_interior_layers2.dx2mm_c10` — MARGINAL

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：345.481 Pa
  - 口径：absolute <p_core> over the capillary tail average, closure_on_interior_layers2, dx = 2 mm, c0 = 10 m/s, 362 particles, 642 steps
- 相对误差：-4.03 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**MARGINAL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/raw/relax_ladder_dx2mm_c10.json` → `cases[1]`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired into ``droplet_impact.rhs``. The closure-off control of the same lattice reads -37.23 %, i.e. this variant recovers +33.20 points of the deficit. Admissible (v_max/v_ref = 0.060 < 0.1) and inside the 15 % band, but outside the 3 % paper band. Overall the closure is PROVISIONAL: it is admissible and in band at two of the four tested (dx, c0) points, not at all of them.

### `young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` — WITHHELD

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：391.805 Pa
  - 口径：absolute <p_core> over the capillary tail average, closure_on_interior_layers2, dx = 2 mm, c0 = 20 m/s, 362 particles, 1283 steps
- 相对误差：+8.83 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/raw/relax_ladder.json` → `cases[2]`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired into ``droplet_impact.rhs``. The closure-off control of the same lattice reads -37.88 %, i.e. this variant recovers +29.04 points of the deficit. INADMISSIBLE: residual spurious current v_max/v_ref = 0.388 >= 0.1, so the drop was not at rest when the reading was taken; the number is recorded but not credited. Overall the closure is PROVISIONAL: it is admissible and in band at two of the four tested (dx, c0) points, not at all of them.

### `young_laplace.closure.closure_on_interior_layers2.dx2mm_c5` — WITHHELD

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：377.482 Pa
  - 口径：absolute <p_core> over the capillary tail average, closure_on_interior_layers2, dx = 2 mm, c0 = 5 m/s, 362 particles, 592 steps
- 相对误差：+4.86 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/raw/relax_ladder_dx2mm_c5.json` → `cases[1]`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired into ``droplet_impact.rhs``. The closure-off control of the same lattice reads -33.99 %, i.e. this variant recovers +29.13 points of the deficit. INADMISSIBLE: residual spurious current v_max/v_ref = 0.104 >= 0.1, so the drop was not at rest when the reading was taken; the number is recorded but not credited. Overall the closure is PROVISIONAL: it is admissible and in band at two of the four tested (dx, c0) points, not at all of them.

### `young_laplace.closure.closure_on_interior_layers4.dx2mm_c20` — WITHDRAWN

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：391.805 Pa
  - 口径：absolute <p_core> over the capillary tail average, closure_on_interior_layers4, dx = 2 mm, c0 = 20 m/s, 362 particles, 1283 steps
- 相对误差：+8.83 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHDRAWN**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/raw/relax_ladder.json` → `cases[3]`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_closure_claim_tracks_the_raw_ladder`
- 欠账指针：`backlog_update_m2971.md#P0`；被 `young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` 取代
- 注：Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired into ``droplet_impact.rhs``. The closure-off control of the same lattice reads -37.88 %, i.e. this variant recovers +29.04 points of the deficit. INADMISSIBLE: residual spurious current v_max/v_ref = 0.388 >= 0.1, so the drop was not at rest when the reading was taken; the number is recorded but not credited. Overall the closure is PROVISIONAL: it is admissible and in band at two of the four tested (dx, c0) points, not at all of them. Bit-identical to `young_laplace.closure.closure_on_interior_layers2.dx2mm_c20`: extra donor layers do not change the donor set at this resolution, so this row is kept as provenance only and does not count as independent evidence.

### `young_laplace.csf_consistent.dx2mm` — WITHHELD

- 物理量：dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：77.1912 Pa
  - 口径：p_center - p_interface_band (two-gauge readout, band at 0.80-0.95 R)
- 相对误差：-78.56 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/csf_consistent/validation_report.json` → `relax.laplace`
  - 守护测试：`tests/test_csf_consistent.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：This artifact's historic value (-75.6 %) is the figure quoted in docs/verification_surface_tension_consistent.md; it has been regenerated since (see SUPERSEDED_NUMBERS in the generator).

### `young_laplace.csf_curvature_fit.dx2mm` — WITHHELD

- 物理量：dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：44.5997 Pa
  - 口径：p_center - p_interface_band, MLS curvature operator
- 相对误差：-87.61 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/csf_curvature_fit/validation_report.json` → `relax[0].laplace`
  - 守护测试：`tests/test_csf_curvature_fit.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Same pressure-closure deficit through a different curvature operator.

### `young_laplace.force_balance.flux_integral.dx1.5mm` — MARGINAL

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：333.145 Pa
  - 口径：radial flux integral -int_0^R rho a_r dr of a single CSF evaluation on the calibrated lattice (no time stepping)
- 相对误差：-7.46 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**MARGINAL**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/report_force_balance.json` → `cases[1].force_side`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_force_balance_integral_is_registered_and_paper_grade`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Bypasses the pressure field entirely, so it separates the two halves of the Young-Laplace deficit: the surface-tension force already delivers sigma/R to -7.46 %, hence the historical -38 % lived in the free surface pressure assembly (the closure), not in the force model. Bins 400/800 agree to 0.01 % (``bin_sensitivity``), so this is not a quadrature artefact.

### `young_laplace.force_balance.flux_integral.dx2mm` — PASS

- 物理量：Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：359.078 Pa
  - 口径：radial flux integral -int_0^R rho a_r dr of a single CSF evaluation on the calibrated lattice (no time stepping)
- 相对误差：-0.26 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**PASS**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_pressure_closure/report_force_balance.json` → `cases[0].force_side`
  - 守护测试：`tests/test_young_laplace_closure_ledger.py::test_force_balance_integral_is_registered_and_paper_grade`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Bypasses the pressure field entirely, so it separates the two halves of the Young-Laplace deficit: the surface-tension force already delivers sigma/R to -0.26 %, hence the historical -38 % lived in the free surface pressure assembly (the closure), not in the force model. Bins 400/800 agree to 0.01 % (``bin_sensitivity``), so this is not a quadrature artefact.

### `young_laplace.laplace_law.vacuum_gauge` — WITHHELD

- 物理量：dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：380.158 Pa
  - 口径：p_center - p_interface_band with p_interface_band = 0.0 Pa (i.e. p_center - vacuum, not p_core - p_ambient)
- 相对误差：+5.60 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**WITHHELD**
- 工件自身 verdict：未声明
- 溯源：`benchmarks/laplace_law/validation_report.json` → `laplace`
  - 守护测试：`tests/test_laplace_law.py`
- 欠账指针：`docs/verification_ledger.md`
- 注：Declares within_15pct=True, but the readout subtracts a zero ambient pressure: the core pressure of a wcSPH droplet carries a ~rho c0^2 additive offset, so the +5.6 % is not comparable with the -37.9 % absolute reading above and must not be cited as a pass.

### `young_laplace.plateau_definition` — WITHHELD

- 物理量：dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：530.313 Pa
  - 口径：p_plateau(r<0.6R) - p_ambient(r>1.0R), worst of the ladder
- 相对误差：+47.31 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 工件自身 verdict：**fail**（`verdict`）
- 溯源：`benchmarks/laplace_plateau/validation_report.json` → `verdict`
  - 守护测试：`tests/test_laplace_plateau.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Plateau settles at ~0.55 sigma/R at every resolution -> systematic, not resolution limited. Kernel-sum bias of the static lattice: 0.588 % (P1 in backlog_update_m2971.md).

## 7. 怎么用 / 怎么维护

```bash
# 新增或刷新任意 benchmark 工件后，重生成账本与图
python benchmarks/verification_ledger/build_ledger.py --figure
# 只检查是否漂移（CI 语义，退出码 1 = 有漂移）
python benchmarks/verification_ledger/build_ledger.py --check
```

纪律：

1. **不要手改本文件**，改生成器；重生成即得。
2. 新增 benchmark 工具必须登记 claim（否则 `unregistered_artifact` finding），或在 `benchmarks/verification_ledger/waivers.json` 里显式豁免并写理由。
3. 引用任何验证数字必须写明口径（本表 §6 的 `definition` 字段）；`WITHHELD`/`FAIL`/`WITHDRAWN` 的数字不得作为当前精度引用。
4. 散文里出现 `SUPERSEDED_NUMBERS` 中的作废数字必须带历史标记，否则 finding。
