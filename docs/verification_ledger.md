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

- claims **16** ／ 物理量 **10** ／ findings **12**
- 分级分布：MARGINAL 1，WITHHELD 8，FAIL 6，WITHDRAWN 1
- findings 分类：declared_vs_derived 1，quantity_spread 2，unmarked_superseded_number 1，unregistered_artifact 8

## 4. 断言表

| claim | 物理量 / 口径 | 参考 | 实测 | rel err | 等级 | 工件 → JSON 路径 | 守护测试 |
|---|---|---|---|---|---|---|---|
| `colour_surface.delta_s_integral` | colour-surface normalisation, integral |grad c| dr (exact: 1) | 1 - | 0.754545 - | -24.55 % | **WITHHELD** | `benchmarks/csf_consistent/curvature_audit.json` → `verdicts.delta_s_integral_dn` | `tests/test_csf_consistent.py` |
| `csf_force.sigma_eff_over_sigma.dx5mm` | effective surface tension from the half-droplet force integral, sigma_eff/sigma | 1 - | 0.704004 - | -29.60 % | **FAIL** | `benchmarks/laplace_pressure/validation_report.json` → `cases[0]` | `tests/test_laplace_pressure.py` |
| `curvature_operator.delta_s_weighted.dx3mm` | delta_s-weighted curvature kappa vs analytic 1/R (R = 20 mm) | 50 1/m | 64.1315 1/m | +28.26 % | **WITHHELD** | `benchmarks/csf_consistent/curvature_audit.json` → `rows[0].variants[0]` | `tests/test_csf_curvature_audit.py` |
| `dam_break.martin_moyce.mae` | dam-break front position vs Martin & Moyce (1952), mean abs error in Z/a | limit 0.15 | 0.572757 | — | **FAIL** | `benchmarks/dam_break/resolution_ladder.json` → `gates.experimental_comparison` | `tests/test_dam_break_experimental.py` |
| `laplace_droplet.merged_csf.8cases` | dp = sigma/R (2-D), sigma=0.072 N/m, R=0.1 m -> 0.72 Pa | 1 - | 3.19723 - | +219.72 % | **FAIL** | `benchmarks/laplace_droplet/results.json` → `verdict` | `tests/test_laplace_droplet.py::test_laplace_jump_current_state` |
| `laplace_report.laplace_pressure.Rdx10` | pressure balance of a static droplet at R/dx = 10 (code units) | 1.44 - | 1.82627 - | +26.82 % | **FAIL** | `benchmarks/laplace/report.json` → `checks.R/dx=10:laplace_pressure` | `tests/test_laplace_pressure.py` |
| `oscillating_droplet.frequency.lamb` | fundamental frequency of an inviscid 2-D droplet (Lamb omega_2) | 73.4847 rad/s | 41.4979 rad/s | -43.53 % | **FAIL** | `benchmarks/oscillating_droplet/validation_report.json` → `verdict.rows[0]` | `tests/test_oscillating_droplet.py` |
| `paper_uncertainty.band.dx3mm` | Young-Laplace jump (2-D), sigma=7.2 N/m, R=20 mm -> 360 Pa | 360 Pa | 249.626 Pa | -30.66 % | **WITHHELD** | `benchmarks/paper_laplace_uncertainty/out/results.json` → `rows[0].metrics.band` | `tests/test_paper_laplace_uncertainty.py` |
| `static_droplet.force_direction.WITHDRAWN` | sign/shape of the CSF force on a static droplet (not a physical gate) | 1 - | 0.173549 - | -82.65 % | **WITHDRAWN** | `benchmarks/static_droplet/validation_report.json` → `validation` | `tests/test_laplace_validation.py` |
| `surface_tension_ladder.laplace_error_finest` | Laplace error of the surface-tension-only ladder, finest level | 0.15 fraction | 2.51981 fraction | +251.98 % | **FAIL** | `benchmarks/surface_tension/laplace_ladder.json` → `gates.laplace_error_finest` | `tests/test_surface_tension_laplace.py` |
| `two_phase_laplace.smoothed_colour.dx2.5mm` | Laplace error on a closed two-phase interface (periodic, no free surface) | limit 0.15 | 0.046079 | — | **MARGINAL** | `benchmarks/surface_tension_laplace/validation_report.json` → `criteria.two_phase_laplace_error_at_dx_0.0025_lt_15pct` | `tests/test_surface_tension_laplace.py` |
| `young_laplace.absolute_tail.best_of_6` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 237.636 Pa | -33.99 % | **WITHHELD** | `benchmarks/laplace_pressure_floor/report.json` → `runs[2].absolute_time_average_tail` | `tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct` |
| `young_laplace.csf_consistent.dx2mm` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 77.1912 Pa | -78.56 % | **WITHHELD** | `benchmarks/csf_consistent/validation_report.json` → `relax.laplace` | `tests/test_csf_consistent.py` |
| `young_laplace.csf_curvature_fit.dx2mm` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 44.5997 Pa | -87.61 % | **WITHHELD** | `benchmarks/csf_curvature_fit/validation_report.json` → `relax[0].laplace` | `tests/test_csf_curvature_fit.py` |
| `young_laplace.laplace_law.vacuum_gauge` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 380.158 Pa | +5.60 % | **WITHHELD** | `benchmarks/laplace_law/validation_report.json` → `laplace` | `tests/test_laplace_law.py` |
| `young_laplace.plateau_definition` | dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa | 360 Pa | 530.313 Pa | +47.31 % | **WITHHELD** | `benchmarks/laplace_plateau/validation_report.json` → `verdict` | `tests/test_laplace_plateau.py` |

## 5. findings（证据缺陷清单）

- **declared_vs_derived**（`high`）`young_laplace.laplace_law.vacuum_gauge` — artifact declares PASS, ledger derives WITHHELD: vacuum-gauge readout (additive |p| offset, not a physical dp)
- **quantity_spread**（`high`）`young_laplace_jump_sigma0.072_R0.1` — 2 active readings of one quantity span +219.7 % … +252.0 % (spread 32.3 pt): laplace_droplet.merged_csf.8cases=+219.7 %, surface_tension_ladder.laplace_error_finest=+252.0 % — definitions differ (gauge / scheme / resolution), so no single number may be quoted without naming its definition.
- **quantity_spread**（`high`）`young_laplace_jump_sigma7.2_R0.02` — 6 active readings of one quantity span -87.6 % … +47.3 % (spread 134.9 pt): paper_uncertainty.band.dx3mm=-30.7 %, young_laplace.absolute_tail.best_of_6=-34.0 %, young_laplace.csf_consistent.dx2mm=-78.6 %, young_laplace.csf_curvature_fit.dx2mm=-87.6 %, young_laplace.laplace_law.vacuum_gauge=+5.6 %, young_laplace.plateau_definition=+47.3 % — definitions differ (gauge / scheme / resolution), so no single number may be quoted without naming its definition.
- **unregistered_artifact**（`medium`）`benchmarks/conservation/report.json` — machine-readable evidence file not referenced by any ledger claim
- **unregistered_artifact**（`medium`）`benchmarks/contact_angle_geometry/results.json` — machine-readable evidence file not referenced by any ledger claim
- **unregistered_artifact**（`medium`）`benchmarks/dam_break/reference_ladder.json` — machine-readable evidence file not referenced by any ledger claim
- **unregistered_artifact**（`medium`）`benchmarks/init_consistency/report.json` — machine-readable evidence file not referenced by any ledger claim
- **unregistered_artifact**（`medium`）`benchmarks/initial_density_bias/report.json` — machine-readable evidence file not referenced by any ledger claim
- **unregistered_artifact**（`medium`）`benchmarks/laplace_pressure_closure/raw/relax_ladder.json` — machine-readable evidence file not referenced by any ledger claim
- **unregistered_artifact**（`medium`）`benchmarks/laplace_pressure_jump/validation_report.json` — machine-readable evidence file not referenced by any ledger claim
- **unregistered_artifact**（`medium`）`benchmarks/lattice_equilibrium/report.json` — machine-readable evidence file not referenced by any ledger claim
- **unmarked_superseded_number**（`medium`）`docs/verification_surface_tension_consistent.md:158` — quotes -75.6% without a historical marker (csf_consistent relaxed readout was regenerated after the consistent-scheme refactor (the artifact now reports -78.6 %); current value in docs/verification_ledger.md)

## 6. 读数明细（口径决定数字）

### `colour_surface.delta_s_integral` — WITHHELD

- 物理量：colour-surface normalisation, integral |grad c| dr (exact: 1)
- 参考：1 -（CSF requires the surface delta to integrate to one）
- 实测：0.754545 -
  - 口径：coarsest level of the audit ladder (dx = 3 mm)
- 相对误差：-24.55 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
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
- 溯源：`benchmarks/dam_break/resolution_ladder.json` → `gates.experimental_comparison`
  - 守护测试：`tests/test_dam_break_experimental.py`
- 欠账指针：`docs/verification_ledger.md`
- 注：RMSE 0.699 Z/a, max 1.618 Z/a; fraction inside experimental uncertainty 0.00. Wave breaking is a 3-D/viscous/non-hydrostatic problem: this is not a surface-tension defect.

### `laplace_droplet.merged_csf.8cases` — FAIL

- 物理量：dp = sigma/R (2-D), sigma=0.072 N/m, R=0.1 m -> 0.72 Pa
- 参考：1 -（dp_model/dp_theory = 1）
- 实测：3.19723 -
  - 口径：worst of 8 lattice/jitter cases (0 of 8 within 15 %)
- 相对误差：+219.72 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
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
- 溯源：`benchmarks/laplace/report.json` → `checks.R/dx=10:laplace_pressure`
  - 守护测试：`tests/test_laplace_pressure.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Platform-level verdict at generation time: 9 of 11 criteria failed.

### `oscillating_droplet.frequency.lamb` — FAIL

- 物理量：fundamental frequency of an inviscid 2-D droplet (Lamb omega_2)
- 参考：73.4847 rad/s（Lamb (1932) omega^2 = n(n^2-1) sigma/(rho R^3), n = 2）
- 实测：41.4979 rad/s
  - 口径：FFT peak of the semi-axis oscillation, dx = 2 mm
- 相对误差：-43.53 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`FAIL`；本账等级：**FAIL**
- 溯源：`benchmarks/oscillating_droplet/validation_report.json` → `verdict.rows[0]`
  - 守护测试：`tests/test_oscillating_droplet.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Restoring force comes from the same surface-tension path: a -43 % frequency is consistent with sigma_eff/sigma ~ 0.7 (omega ~ sqrt(sigma)), which is exactly the force-path finding below.

### `paper_uncertainty.band.dx3mm` — WITHHELD

- 物理量：Young-Laplace jump (2-D), sigma=7.2 N/m, R=20 mm -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：249.626 Pa
  - 口径：band definition, mean of 5 realisations
- 相对误差：-30.66 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
- 溯源：`benchmarks/paper_laplace_uncertainty/out/results.json` → `rows[0].metrics.band`
  - 守护测试：`tests/test_paper_laplace_uncertainty.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Largest R/dx = 6.7 of the sweep; 95 % CI +/-9.45 pt.

### `static_droplet.force_direction.WITHDRAWN` — WITHDRAWN

- 物理量：sign/shape of the CSF force on a static droplet (not a physical gate)
- 参考：1 -（no physical reference: the criterion was self-referential）
- 实测：0.173549 -
  - 口径：average radial dot product of the surface-particle force
- 相对误差：-82.65 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHDRAWN`；本账等级：**WITHDRAWN**
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
- 溯源：`benchmarks/surface_tension/laplace_ladder.json` → `gates.laplace_error_finest`
  - 守护测试：`tests/test_surface_tension_laplace.py`
- 欠账指针：`docs/verification_ledger.md`
- 注：Observed convergence order -1.633 (negative: the error grows under refinement) -> this ladder's operator is not convergent.

### `two_phase_laplace.smoothed_colour.dx2.5mm` — MARGINAL

- 物理量：Laplace error on a closed two-phase interface (periodic, no free surface)
- 参考：0 fraction（exact Laplace balance on the closed interface）
- 实测：0.046079 fraction
  - 口径：|dp_measured/dp_theory - 1| at dx = 2.5 mm, smoothed colour field
- 相对误差：—（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**MARGINAL**
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
- 溯源：`benchmarks/laplace_pressure_floor/report.json` → `runs[2].absolute_time_average_tail`
  - 守护测试：`tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Best of 6 runs (c0 20/5 x dx 0.002/0.0015 plus controls). Reproduces backlog P0: the residual is the free-surface pressure closure, not the curvature operator.

### `young_laplace.csf_consistent.dx2mm` — WITHHELD

- 物理量：dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：77.1912 Pa
  - 口径：p_center - p_interface_band (two-gauge readout, band at 0.80-0.95 R)
- 相对误差：-78.56 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`WITHHELD`；本账等级：**WITHHELD**
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
- 溯源：`benchmarks/csf_curvature_fit/validation_report.json` → `relax[0].laplace`
  - 守护测试：`tests/test_csf_curvature_fit.py`
- 欠账指针：`backlog_update_m2971.md#P0`
- 注：Same pressure-closure deficit through a different curvature operator.

### `young_laplace.laplace_law.vacuum_gauge` — WITHHELD

- 物理量：dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa
- 参考：360 Pa（Young-Laplace sigma/R）
- 实测：380.158 Pa
  - 口径：p_center - p_interface_band with p_interface_band = 0.0 Pa (i.e. p_center - vacuum, not p_core - p_ambient)
- 相对误差：+5.60 %（平台线 ±15%，论文级 ±3%）
- 工件自述状态：`PASS`；本账等级：**WITHHELD**
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
