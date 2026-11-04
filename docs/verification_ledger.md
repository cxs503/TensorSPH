# 接口压差验证台账（跨 benchmark 家族 · 机器核对）

> 本文件由 `benchmarks/verification_ledger/build.py` 从仓库内已提交的 `*.json` 工件**程序化注入**生成（零手抄）。改数字请改工件或 `ledger.py` 的口径登记表后重新生成；`tests/test_verification_ledger.py` 会在台账 / 工件 / 本文档三者漂移时报红。

## 0. 结论摘要

- 覆盖 **11 个家族 / 67 条读数**，来自 11 个已提交工件，归入 15 种测量口径。
- Δp 共 58 条读数，只有 7 条 <15%、0 条 <3%；当前口径（sanctioned）最优一条是 `floor#run2#absolute_time_average_tail`：**-34.0%**。
- 唯一那条跨自由面且 <15% 的读数（`laplace_law#relaxed#0`）经三项对照检验后**不能作为自由面压强闭合有效的证据**：工件不可复现（+5.6% vs 重跑 +8.2%）、同代码路径细化一档即翻到 -105.0%、且其减去的上带压强在粗网格上恰被钳为 0（0.0 Pa → 细网格 299.1 Pa）。
- **判定：Δp 维持 WITHHELD。** 台账把 m2971 的 P0 归因（自由面压强闭合是唯一门槛项）从推理升级为可计数、可复算的事实，并正式退役 `center_minus_band_instant` 口径。

## 1. 为什么要有这本台账

同一个物理量（弯曲界面压差 Δp = σ/R）在本仓 2026-10 → 2026-11 的历史里被先后引用为 −76%、−107%、−87.6%、+5.6%、−34%。它们互相矛盾的主因不是求解器变了，而是**测量口径**变了（减的是哪两个压力、单时刻还是时间平均）。仓库原先没有任何机制阻止后来的 agent 把已废弃口径的数字当成当前精度引用——`backlog_update_m2971.md`（P2.1）要求的正是这台机器闸门。

本台账把每个家族的**每一条读数**登记为一行，并给每一行贴上口径标签；口径标签带 `sanctioned`（可否作为当前精度引用）与理由。当前标准口径为「绝对压强 + 时间平均」（m2971）。

## 2. 口径登记表（单一事实来源）

| 口径 key | 相减的操作数 | 时间处理 | 跨自由面 | 可引用为当前精度 | 理由 |
|---|---|---|---|---|---|
| `abs_tail_avg` | p_core - 0 (absolute) | time_average | 是 | **是** | m2971 canonical: absolute pressure, plateau-segment time average（6 条） |
| `abs_plateau_avg` | p_plateau - p_ambient (absolute) | time_average | 是 | **是** | m2971 canonical: plateau mean over a time-averaged window（3 条） |
| `abs_instant` | p_core - 0 (absolute) | instant | 是 | 否（历史口径） | single time sample; the initial spurious pressure has not decayed（6 条） |
| `bandA_instant` | p_core - p(0.80-0.95 R) | instant | 是 | 否（历史口径） | upper band sits in the surface density-deficit layer: biases dp low（6 条） |
| `bandB_instant` | p_core - p(0.60-0.95 R) | instant | 是 | 否（历史口径） | wider version of bandA; same density-deficit contamination（6 条） |
| `two_phase_abs_instant` | p_in - p_out (two-phase, smoothed colour) | instant | 否 | 否（历史口径） | both sides wetted, so no free-surface closure is exercised; still a single instant（3 条） |
| `two_phase_sharp_instant` | p_in - p_out (two-phase, sharp colour step) | instant | 否 | 否（历史口径） | as two_phase_abs_instant, but with an under-resolved colour step（3 条） |
| `legacy_gradient_instant` | p_in - p_out (legacy gradient colour field) | instant | 否 | 否（历史口径） | pre-CSF colour gradient; kept only as a regression floor（1 条） |
| `one_sided_free_surface` | p_core - p_ambient (one-sided kernel) | instant | 是 | 否（历史口径） | one-sided kernel truncation at the free surface（2 条） |
| `shell_band_relax` | p_core - p_shell (dS-weighted shell) | relaxed | 是 | 否（历史口径） | m277/m2174 convention; the shell band sits inside the interface（5 条） |
| `center_minus_band_instant` | p_center - p_band(interface) | instant | 是 | 否（历史口径） | band pressure can be exactly 0 (degenerate), so the reading silently becomes absolute（1 条） |
| `center_minus_band_relax` | p_center - p_band(interface) | relaxed | 是 | 否（历史口径） | relaxed form of center_minus_band_instant; band location drifts with dx（4 条） |
| `band_ensemble_instant` | p_core - p_band, 5 lattice phases | instant | 是 | 否（历史口径） | ensemble over lattice phase at fixed t = 1 t_sigma; still an instant（5 条） |
| `legacy_status_quo` | legacy static-probe dp | instant | 是 | 否（历史口径） | pre-CSF status-quo probe, superseded by the consistent scheme（7 条） |
| `cut_force_half` | F_cut / (2 sigma L) | instant | 是 | 否（历史口径） | half-domain cut force, not a pressure difference (auxiliary quantity)（9 条） |

## 3. 读数总表

`Δ%` = (实测 − 解析)/解析，门限取平台标准 3% / 15%。全部数值由工件注入。

| # | 家族 | 读数 id | 口径 | dx (m) | 实测 | 解析 | Δ% | 门限 |
|---|---|---|---|---|---|---|---|---|
| 1 | laplace_cut_force | `cut_force#case0` | `cut_force_half` | 0.0050 | 0.7040 | 1.0000 | -29.6 | FAIL |
| 2 | laplace_cut_force | `cut_force#case1` | `cut_force_half` | 0.0050 | 0.9194 | 1.0000 | -8.1 | pass<15% |
| 3 | laplace_cut_force | `cut_force#case2` | `cut_force_half` | 0.0050 | 1.0874 | 1.0000 | +8.7 | pass<15% |
| 4 | laplace_cut_force | `cut_force#case3` | `cut_force_half` | 0.0025 | 1.0874 | 1.0000 | +8.7 | pass<15% |
| 5 | laplace_cut_force | `cut_force#case4` | `cut_force_half` | 0.0025 | 1.4991 | 1.0000 | +49.9 | FAIL |
| 6 | laplace_cut_force | `cut_force#case5` | `cut_force_half` | 0.0025 | 1.8784 | 1.0000 | +87.8 | FAIL |
| 7 | laplace_cut_force | `cut_force#case6` | `cut_force_half` | 0.0013 | 1.8784 | 1.0000 | +87.8 | FAIL |
| 8 | laplace_cut_force | `cut_force#case7` | `cut_force_half` | 0.0013 | 2.5764 | 1.0000 | +157.6 | FAIL |
| 9 | laplace_cut_force | `cut_force#case8` | `cut_force_half` | 0.0013 | 3.3400 | 1.0000 | +234.0 | FAIL |
| 10 | csf_consistent | `csf_consistent#relaxed#0` | `center_minus_band_relax` | 0.0020 | 77.1912 | 360.0000 | -78.6 | FAIL |
| 11 | csf_curvature_fit | `csf_curv_fit#relaxed#0` | `center_minus_band_relax` | 0.0020 | 44.5997 | 360.0000 | -87.6 | FAIL |
| 12 | csf_curvature_fit | `csf_curv_fit#relaxed#1` | `center_minus_band_relax` | 0.0020 | -24.2528 | 360.0000 | -106.7 | FAIL |
| 13 | csf_curvature_fit | `csf_curv_fit#relaxed#2` | `center_minus_band_relax` | 0.0020 | 43.5333 | 360.0000 | -87.9 | FAIL |
| 14 | laplace_law | `laplace_law#relaxed#0` | `center_minus_band_instant` | 0.0040 | 380.1575 | 360.0000 | +5.6 | pass<15% |
| 15 | laplace_plateau | `plateau#ladder0` | `abs_plateau_avg` | 0.0040 | 199.6250 | 360.0000 | -44.5 | FAIL |
| 16 | laplace_plateau | `plateau#ladder1` | `abs_plateau_avg` | 0.0020 | 198.9759 | 360.0000 | -44.7 | FAIL |
| 17 | laplace_plateau | `plateau#ladder2` | `abs_plateau_avg` | 0.0015 | 189.6871 | 360.0000 | -47.3 | FAIL |
| 18 | laplace_pressure_floor | `floor#run0#absolute_time_average_tail` | `abs_tail_avg` | 0.0020 | 223.6462 | 360.0000 | -37.9 | FAIL |
| 19 | laplace_pressure_floor | `floor#run0#definition_absolute` | `abs_instant` | 0.0020 | 208.5009 | 360.0000 | -42.1 | FAIL |
| 20 | laplace_pressure_floor | `floor#run0#definition_bandA` | `bandA_instant` | 0.0020 | 11.4405 | 360.0000 | -96.8 | FAIL |
| 21 | laplace_pressure_floor | `floor#run0#definition_bandB` | `bandB_instant` | 0.0020 | 25.5550 | 360.0000 | -92.9 | FAIL |
| 22 | laplace_pressure_floor | `floor#run1#absolute_time_average_tail` | `abs_tail_avg` | 0.0020 | 225.9711 | 360.0000 | -37.2 | FAIL |
| 23 | laplace_pressure_floor | `floor#run1#definition_absolute` | `abs_instant` | 0.0020 | 178.1732 | 360.0000 | -50.5 | FAIL |
| 24 | laplace_pressure_floor | `floor#run1#definition_bandA` | `bandA_instant` | 0.0020 | -3.6946 | 360.0000 | -101.0 | FAIL |
| 25 | laplace_pressure_floor | `floor#run1#definition_bandB` | `bandB_instant` | 0.0020 | 7.4568 | 360.0000 | -97.9 | FAIL |
| 26 | laplace_pressure_floor | `floor#run2#absolute_time_average_tail` | `abs_tail_avg` | 0.0020 | 237.6365 | 360.0000 | -34.0 | FAIL |
| 27 | laplace_pressure_floor | `floor#run2#definition_absolute` | `abs_instant` | 0.0020 | 203.0073 | 360.0000 | -43.6 | FAIL |
| 28 | laplace_pressure_floor | `floor#run2#definition_bandA` | `bandA_instant` | 0.0020 | 43.0804 | 360.0000 | -88.0 | FAIL |
| 29 | laplace_pressure_floor | `floor#run2#definition_bandB` | `bandB_instant` | 0.0020 | 23.6726 | 360.0000 | -93.4 | FAIL |
| 30 | laplace_pressure_floor | `floor#run3#absolute_time_average_tail` | `abs_tail_avg` | 0.0015 | 222.8507 | 360.0000 | -38.1 | FAIL |
| 31 | laplace_pressure_floor | `floor#run3#definition_absolute` | `abs_instant` | 0.0015 | 203.2152 | 360.0000 | -43.6 | FAIL |
| 32 | laplace_pressure_floor | `floor#run3#definition_bandA` | `bandA_instant` | 0.0015 | 32.7236 | 360.0000 | -90.9 | FAIL |
| 33 | laplace_pressure_floor | `floor#run3#definition_bandB` | `bandB_instant` | 0.0015 | -49.9348 | 360.0000 | -113.9 | FAIL |
| 34 | laplace_pressure_floor | `floor#run4#absolute_time_average_tail` | `abs_tail_avg` | 0.0015 | 229.7097 | 360.0000 | -36.2 | FAIL |
| 35 | laplace_pressure_floor | `floor#run4#definition_absolute` | `abs_instant` | 0.0015 | 242.4686 | 360.0000 | -32.6 | FAIL |
| 36 | laplace_pressure_floor | `floor#run4#definition_bandA` | `bandA_instant` | 0.0015 | 65.3451 | 360.0000 | -81.8 | FAIL |
| 37 | laplace_pressure_floor | `floor#run4#definition_bandB` | `bandB_instant` | 0.0015 | -15.8977 | 360.0000 | -104.4 | FAIL |
| 38 | laplace_pressure_floor | `floor#run5#absolute_time_average_tail` | `abs_tail_avg` | 0.0015 | 225.3194 | 360.0000 | -37.4 | FAIL |
| 39 | laplace_pressure_floor | `floor#run5#definition_absolute` | `abs_instant` | 0.0015 | 179.5915 | 360.0000 | -50.1 | FAIL |
| 40 | laplace_pressure_floor | `floor#run5#definition_bandA` | `bandA_instant` | 0.0015 | 22.1329 | 360.0000 | -93.9 | FAIL |
| 41 | laplace_pressure_floor | `floor#run5#definition_bandB` | `bandB_instant` | 0.0015 | -22.8415 | 360.0000 | -106.3 | FAIL |
| 42 | laplace_status_quo | `legacy#R/dx=10:laplace_pressure` | `legacy_status_quo` | 0.0050 | 1.8263 | 1.4400 | +26.8 | FAIL |
| 43 | laplace_status_quo | `legacy#R/dx=20:laplace_pressure` | `legacy_status_quo` | 0.0025 | 1.3838 | 1.4400 | -3.9 | pass<15% |
| 44 | laplace_status_quo | `legacy#R/dx=40:laplace_pressure` | `legacy_status_quo` | 0.0013 | 5.7039 | 1.4400 | +296.1 | FAIL |
| 45 | paper_laplace_uncertainty | `paper#row0` | `band_ensemble_instant` | 0.0030 | 249.6263 | 360.0000 | -30.7 | FAIL |
| 46 | paper_laplace_uncertainty | `paper#row1` | `band_ensemble_instant` | 0.0025 | 102.7330 | 360.0000 | -71.5 | FAIL |
| 47 | paper_laplace_uncertainty | `paper#row2` | `band_ensemble_instant` | 0.0020 | 51.2888 | 360.0000 | -85.8 | FAIL |
| 48 | paper_laplace_uncertainty | `paper#row3` | `band_ensemble_instant` | 0.0015 | 41.0707 | 360.0000 | -88.6 | FAIL |
| 49 | paper_laplace_uncertainty | `paper#row4` | `band_ensemble_instant` | 0.0010 | 8.9522 | 360.0000 | -97.5 | FAIL |
| 50 | static_droplet | `static#row0` | `shell_band_relax` | 0.0050 | 0.3847 | 1.4400 | -73.3 | FAIL |
| 51 | static_droplet | `static#row1` | `shell_band_relax` | 0.0050 | 0.2472 | 0.9600 | -74.3 | FAIL |
| 52 | static_droplet | `static#row2` | `shell_band_relax` | 0.0050 | 0.1931 | 0.7200 | -73.2 | FAIL |
| 53 | static_droplet | `static#row3` | `shell_band_relax` | 0.0050 | 0.1597 | 0.5760 | -72.3 | FAIL |
| 54 | static_droplet | `static#row4` | `shell_band_relax` | 0.0050 | 0.1302 | 0.4800 | -72.9 | FAIL |
| 55 | surface_tension_ladder | `ladder#level0` | `legacy_status_quo` | 0.0063 | 0.3486 | 0.7200 | -51.6 | FAIL |
| 56 | surface_tension_ladder | `ladder#level1` | `legacy_status_quo` | 0.0031 | 0.7729 | 0.7200 | +7.4 | pass<15% |
| 57 | surface_tension_ladder | `ladder#level2` | `legacy_status_quo` | 0.0016 | 1.6559 | 0.7200 | +130.0 | FAIL |
| 58 | surface_tension_ladder | `ladder#level3` | `legacy_status_quo` | 0.0010 | 2.5343 | 0.7200 | +252.0 | FAIL |
| 59 | surface_tension_laplace | `stlaplace#legacy_gradient#0` | `legacy_gradient_instant` | 0.0050 | 0.0293 | 0.7200 | -95.9 | FAIL |
| 60 | surface_tension_laplace | `stlaplace#one_sided_free_surface#0` | `one_sided_free_surface` | 0.0100 | 0.1130 | 0.7200 | -84.3 | FAIL |
| 61 | surface_tension_laplace | `stlaplace#one_sided_free_surface#1` | `one_sided_free_surface` | 0.0050 | 0.1972 | 0.7200 | -72.6 | FAIL |
| 62 | surface_tension_laplace | `stlaplace#two_phase_sweep_sharp_step_color#0` | `two_phase_sharp_instant` | 0.0100 | 0.6090 | 0.7200 | -15.4 | FAIL |
| 63 | surface_tension_laplace | `stlaplace#two_phase_sweep_sharp_step_color#1` | `two_phase_sharp_instant` | 0.0050 | 0.6512 | 0.7200 | -9.6 | pass<15% |
| 64 | surface_tension_laplace | `stlaplace#two_phase_sweep_sharp_step_color#2` | `two_phase_sharp_instant` | 0.0025 | 0.5138 | 0.7200 | -28.6 | FAIL |
| 65 | surface_tension_laplace | `stlaplace#two_phase_sweep_smoothed_color#0` | `two_phase_abs_instant` | 0.0100 | 0.6467 | 0.7200 | -10.2 | pass<15% |
| 66 | surface_tension_laplace | `stlaplace#two_phase_sweep_smoothed_color#1` | `two_phase_abs_instant` | 0.0050 | 0.6885 | 0.7200 | -4.4 | pass<15% |
| 67 | surface_tension_laplace | `stlaplace#two_phase_sweep_smoothed_color#2` | `two_phase_abs_instant` | 0.0025 | 0.6868 | 0.7200 | -4.6 | pass<15% |

## 4. 同一算例、四种口径：口径能把误差挪多远

`laplace_pressure_floor` 在 6 个算例上同时给出 4 种口径的读数，因此可以**在物理完全相同**的前提下隔离口径的影响：

| 算例（dx, 说明） | abs_instant | bandA | bandB | abs_tail_avg（当前口径） | 口径极差 (pp) |
|---|---|---|---|---|---|
| 0.0015 / c0=20.0 | - | - | - | - | 75.8 |
| 0.0015 / c0=10.0 | - | - | - | - | 71.8 |
| 0.0015 / c0=5.0 | - | - | - | - | 68.9 |
| 0.002 / c0=10.0 | - | - | - | - | 63.8 |
| 0.002 / c0=5.0 | - | - | - | - | 59.4 |
| 0.002 / c0=20.0 | - | - | - | - | 58.9 |

最大口径极差 **75.8 pp**（算例 `laplace_pressure_floor / 0.0015 / c0=20.0`：同一算例读数从 -113.9 到 -38.1）。口径选择本身就能吞掉或放大整个误差，这就是历史引用互相矛盾的量级来源。

## 5. 汇总与误差判定

| 量 | 家族数 | 读数 | 中位 Δ% | 最差 abs(Δ%) | <15% | <3% |
|---|---|---|---|---|---|---|
| Laplace pressure jump dp = sigma/R | 10 | 58 | -51.0 | 296.1 | 7/58 | 0/58 |
| half-domain cut force ratio F_cut/(2 sigma L) | 1 | 9 | +49.9 | 234.0 | 3/9 | 0/9 |

### 5.1 Δp 是否 <15%？（逐项作答）

- 当前口径（sanctioned）读数 9 条：abs(Δ%) 最小 **34.0%**、中位 -37.9、最差 47.3%；全部 <15% ？ **否**。
- 最优 sanctioned 读数：`floor#run2#absolute_time_average_tail`（laplace_pressure_floor / `abs_tail_avg`）实测 237.6 Pa vs 解析 360.0 Pa → **Δ = -34.0%**。
- 全台账是否存在 abs(Δ%) < 3% 的 Δp 读数？**否**。
- **结论：Δp 证据等级维持 WITHHELD** —— 当前口径最优点仍差 -34.0%，未达平台 15% 门限。

### 5.2 自由面是唯一分离变量（但有一条例外必须解释）

- 跨自由面读数 51 条：abs(Δ%) 最小 3.9%，其中仅 **3** 条 <15%（逐条见下表与第 6 节）。
- 不跨自由面（两侧皆有流体，双相算例）读数 7 条：abs(Δ%) 最小 4.4%，其中 4 条 <15%（占该子集 57%）。

跨自由面且 <15% 的读数共 **3** 条。它们能否推翻 P0 归因，取决于两项检验：(a) 是否分辨率稳定；(b) 读数是否真由表面张力产生。第 6 节逐条作答，结论是**两项都不满足**，因此不推翻。

| 读数 | 家族 | 口径 | Δ% | dx (m) | 分辨率检验 |
|---|---|---|---|---|---|
| `laplace_law#relaxed#0` | laplace_law | `center_minus_band_instant` | +5.60 | 0.0040 | 家族内只有单点，需外部对照（见 6.1–6.3） |
| `legacy#R/dx=20:laplace_pressure` | laplace_status_quo | `legacy_status_quo` | -3.90 | 0.0025 | 零穿越：同家族 4 档分辨率上 |Δ%| 先降后升（随细化发散） |
| `ladder#level1` | surface_tension_ladder | `legacy_status_quo` | +7.35 | 0.0031 | 零穿越：同家族 4 档分辨率上 |Δ%| 先降后升（随细化发散） |

### 5.3 辅助量：切割力比 F_cut/(2σL)（9 条，中位 +49.9%，最差 abs(Δ%) 234.0%）

切割力是**半域积分**量，不经过自由面压强闭合，因此它单独达标也不能证明 Δp 达标；它落在门限附近说明「力 → 压强」的传递环节才是缺口。该量仅作辅助登记。

## 6. 来源审计：3 条 <15% 的读数经得起检验吗

### 6.1–6.3 被审对象 A：`laplace_law#relaxed#0`（工件 `benchmarks/laplace_law/validation_report.json`），登记的读数为 +5.6%。三项对照全部复跑该家族**自己的**代码路径 `benchmarks/laplace_law/run.py::relax`（R = 0.02 m，σ = 7.2 N/m，c₀ = 20 m/s，t = 0.89 t_σ）：

### Q1 可复现性 —— 不通过

- 重跑同一设置得 **+8.2%**（389.4 Pa），工件登记 +5.6%（380.2 Pa），相差 **2.57 pp**。
- 且 `run.py` 的输出版本是 `{"instant": [...], "relax": {...}}`，而该工件顶层为 `setup/reference/laplace/history`、`dx = 0.004`、380 步 —— 它不是当前代码的输出，是旧代码状态的摘要。**该工件不可复现，属陈旧快照。**

### Q2 σ → 0 对照 —— 通过（读数确由表面张力产生）

- σ 置为 1e-09 N/m、其余完全不变：Δp = **0.0 Pa**（σ = 7.2 时 389.4 Pa）。
- 初始核和偏置压强（t=0 时 2395 Pa，即审计 C2 的同一量）在 σ→0 下 50 步内即衰减到 0 Pa，而 σ = 7.2 时核心压强被持续维持。
- 因此 +8% 这一读数**不是**初始化伪压强的残留，它是 CSF 力真实压出来的核心过压。问题不在「是否为数值噪声」，而在下一问。

### Q3 分辨率稳定性 —— 不通过（决定性）

| 算例 | R/dx | p_core (Pa) | p_band (Pa) | Δp (Pa) | Δ% | <15% |
|---|---|---|---|---|---|---|
| `sigma7_rdx5` | 5 | 389.4 | 0.0 | +389.4 | +8.17 | 是 |
| `sigma7_rdx10` | 10 | 281.0 | 299.1 | -18.1 | -105.03 | 否 |

同一条代码路径、同一瞬时口径、同一物理时刻，仅把 R/dx 从 5 细化到 10，读数从 **+8.17%** 翻到 **-105.03%**。机械原因很清楚：粗网格上被减去的上带压强（p(0.80–0.95R)）被 Tait 钳位**恰为 0**（0.0 Pa），于是「减上带」这一步退化成「绝对压强」；细网格上该带进入受压区变为 299.1 Pa，减法把信号本身减掉了。

**判定：该读数不作数。** 它是「退化操作数 + 粗网格」的巧合，不能作为自由面压强闭合有效的证据；`center_minus_band_instant` 口径据此正式退役（见第 2 节）。P0 归因（自由面压强闭合是唯一门槛项）在台账里保持成立：

- 当前口径（绝对压强 + 时间平均）的 9 条读数全部在门限外（最优 34.0%）；
- 不跨自由面（双相、无自由面闭合问题）的读数里已有 4/7 条 <15%，说明曲率算子与 δ_s 已不是瓶颈。

### 6.4 其余通过者：同家族的分辨率阶梯自己否定了它们

**`legacy#R/dx=20:laplace_pressure`**（`benchmarks/laplace/report.json`，登记 -3.90%）：把同工件内的分辨率阶梯全部列出来（数值同样由工件注入）：

| dx (m) | 实测 Δp | 解析 Δp | Δ% |
|---|---|---|---|
| 0.00125 | 5.7039 | 1.440 | +296.11 |
| 0.00250 | 1.3838 | 1.440 | -3.90 |
| 0.00500 | 1.8263 | 1.440 | +26.82 |

符号误差从 +296.1%（最细网格）单调变到 +26.8%（最粗网格），abs(Δ%) 的最小值 3.90% 落在**内部一档**：这是随细化发散的曲线穿过零点的位置（判定：`zero_crossing`），不是收敛点。以该档为「验证通过」引用，细化后必然失效。

**`ladder#level1`**（`benchmarks/surface_tension/laplace_ladder.json`，登记 +7.35%）：把同工件内的分辨率阶梯全部列出来（数值同样由工件注入）：

| dx (m) | 实测 Δp | 解析 Δp | Δ% |
|---|---|---|---|
| 0.00104 | 2.5343 | 0.720 | +251.98 |
| 0.00156 | 1.6559 | 0.720 | +129.98 |
| 0.00313 | 0.7729 | 0.720 | +7.35 |
| 0.00625 | 0.3486 | 0.720 | -51.59 |

符号误差从 +252.0%（最细网格）单调变到 -51.6%（最粗网格），abs(Δ%) 的最小值 7.35% 落在**内部一档**：这是随细化发散的曲线穿过零点的位置（判定：`zero_crossing`），不是收敛点。以该档为「验证通过」引用，细化后必然失效。

## 7. 注意点 / 局限

1. **工件是历史快照**：台账只读 `benchmarks/**` 已提交的 json，不重跑仿真。陈旧工件会被如实登记（第 6 节是范例），行级 `source` 字段给出可追溯路径。
2. **口径标签人工登记、机器校验**：`sanctioned` 由 `CONVENTIONS` 表一次性给出，读数继承；新工件必须同时登记口径，否则 `collect()` 直接 KeyError。
3. **第 6 节的三项对照是实跑的**（非台账提取），原始快照留在 `benchmarks/verification_ledger/snapshot_*.json`；本机 load15 高时重跑约 3 分钟。
4. **未纳入**：
   - `benchmarks/oscillating_droplet/validation_report.json`：droplet oscillation frequency, not a pressure jump (quantity = omega)
   - `benchmarks/dam_break/*.json`：free-surface flow validation, no Laplace reference value
   - `benchmarks/csf_consistent/validation_report.json:instant`：curvature and cut force at t=0 only via a different operator chain; its relaxed dp entry is in scope, the instant curvature rows are not
5. 台账不判断模型好坏，只回答「这条数字是什么口径下测的、现在还能不能引用」。

## 8. 复现

```bash
python benchmarks/verification_ledger/build.py          # 重建 ledger.json / 本文档 / 图
python benchmarks/verification_ledger/build.py --check   # 只校验漂移（CI 用）
python benchmarks/verification_ledger/provenance_control.py --run   # 重跑第 6 节三项对照
python -m pytest tests/test_verification_ledger.py -q
```

![verification ledger](benchmarks/verification_ledger/ledger.png)

