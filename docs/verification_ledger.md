# Young-Laplace / 表面张力证据总账（机器校验）

本文件由 `benchmarks/evidence_ledger/ledger.py` 从仓库内已提交的 JSON 工件生成，
**没有手抄数字**；`tests/test_verification_ledger.py` 校验本文件与工件一致、历史引用仍在原文档中、且分级符合平台判据。重新生成：`PYTHONPATH=src python benchmarks/evidence_ledger/ledger.py`。

分级规则（平台标准）：|误差| ≤ 3 % → **REAL (<3 %)**；≤ 15 % → **REAL (<15 %)**；> 15 % → **WITHHELD (>15 %)**。所有误差均为 "(实测-基准)/基准"。


## ① 全部压差测量（按年代排序）

| # | 工件 | 年代/方案 | 量 | 实测 | 基准 | 误差 | 分级 | 读数口径 |
|---|---|---|---|---|---|---|---|---|
| 1 | `benchmarks/laplace_law/validation_report.json` | m2174 CSF (delta_s-weighted) | Delta p [Pa] | 380.2 | 360 | +5.6 % | **REAL (<15 %)** | p_center - p(interface band); the band measured 0 Pa, so this is an absolute read-out |
| 2 | `benchmarks/laplace_droplet/results.json` | merged CSF, instrument A | dp_model / dp_theory, worst of 8 cases | -- | -- | +219.7 % | **WITHHELD (>15 %)** | instrument-A read-out of the merged CSF model (pre-refactor) |
| 3 | `benchmarks/csf_consistent/validation_report.json` | consistent CSF (MLS operators) | Delta p [Pa] | 77.19 | 360 | -78.6 % | **WITHHELD (>15 %)** | p_center - p_interface_band (historical band read-out) |
| 4 | `benchmarks/csf_curvature_fit/validation_report.json` | consistent CSF + fit_quadratic curvature | Delta p [Pa] | 44.6 | 360 | -87.6 % | **WITHHELD (>15 %)** | p_center - p_interface_band (historical band read-out) |
| 5 | `benchmarks/surface_tension_laplace/validation_report.json` | one-sided free surface | Delta p, one-sided free surface (worst dx) | 0.113 | -- | +84.3 % | **WITHHELD (>15 %)** | colour jump only partly resolved (no outside particles) |
| 6 | `benchmarks/laplace_plateau/validation_report.json` | relaxation plateau | Delta p [Pa] (worst resolution) | -- | -- | +47.3 % | **WITHHELD (>15 %)** | p_plateau(r<0.6R) - p_ambient(r>1.0R) vs sigma/R |
| 7 | `benchmarks/laplace_pressure/validation_report.json` | cut-force identity | sigma_eff / sigma - 1, cut-force identity (best case) | -- | -- | -8.1 % | **REAL (<15 %)** | global cut-force path, best of 9 (radius, dx) cases |
| 8 | `benchmarks/laplace_pressure_floor/report.json` | same runs as below, historical read-out | Delta p [Pa], band read-out p_core - p(0.80-0.95R) | -3.695 | 360 | -101.0 % | **WITHHELD (>15 %)** | same runs, historical band read-out (contaminated by the free-surface density deficit) |
| 9 | `benchmarks/laplace_pressure_floor/report.json` | same runs, C1-corrected read-out | Delta p [Pa], absolute + tail average (worst run) | 222.9 | 360 | -38.1 % | **WITHHELD (>15 %)** | p_core - 0, time-averaged over the last 40 % of the window |

### 读数口径的代价（C1）

`benchmarks/laplace_pressure_floor/report.json` 的 6 个算例（c0 × 分辨率阶梯）用**同一份流场**同时读两种口径：

- 历史口径 `p_core − p(0.80–0.95R)`：平均 **-92.1 %**；
- 绝对压强 `p_core − 0` + 时间平均：平均 **-36.8 %**。
- 口径本身贡献 **-55.3 个百分点**（基准 σ/R = 360 Pa）。

因此上表中所有标注 "band read-out" 的行，其数值**系统性偏负**，不能与 C1 校正后的行直接比较；它们的历史价值是证明离散方案演进的相对改善，而不是当前精度。


## ② 历史引用 vs 工件（漂移检查）

| 文档 | 文中引用 | 文内存在 | 工件当前值 | 差 | 判定 | 口径 |
|---|---|---|---|---|---|---|
| `verification_surface_tension_laplace.md` | `-93.6%` | 是 | 仓库内无对应字段 | -- | 无法定位：文档是历史快照 | band read-out of the legacy colour heuristic; the run that produced it is no longer shipped as a field |
| `verification_surface_tension_consistent.md` | `-75.6%` | 是 | `validation_report.json` = -78.6 % | -3.0 pt | **DRIFT**（不可复现） | band read-out of the consistent CSF |
| `verification_surface_tension_curvature.md` | `-87.6%` | 是 | `validation_report.json` = -87.6 % | -0.0 pt | 一致（但口径为 band） | band read-out of the fit_quadratic curvature operator |

### 结论（引用纪律）

1. 上述三份 `verification_surface_tension_*.md` 里引用的 −93.6 % / −75.6 % / −87.6 % 一律视为**历史 band 口径值**，不是当前精度，不得作为现状引用。
2. 当前可用于引用的压差结论只有 C1 校正后的行（ledger ① 中 definition 为 "absolute" 的行）：仍 >15 %，即 **WITHHELD**，残差归属自由面压强闭合 （`backlog_update_m2971.md` P0）。

3. 本轮自动检查发现 `verification_surface_tension_consistent.md`（引 -75.6%，工件 -78.6 %） 已与自身工件不一致：文档是历史快照，工件随后被重跑过。该差已登记在本表，不在文档里静默修改。


## ③ 未结项

- **P0 自由面压强闭合**（`backlog_update_m2971.md`）：本 ledger 不含其数值——该实现由并行在飞单负责，本单刻意不重复实现、也不预写它的结果。落地后需向 `ARTIFACTS` 追加一行（closure ladder），本表即自动纳入。
- **表面张力 3D / 多相**：仓库内无工件，故不在表内（无证据不列表）。


## ④ 局限

- 本表只覆盖**压差/切割面力**这一族量；曲率、伪流、δ_s 归一化等判据仍在各 benchmark 自己的 validation_report.json 内，未纳入本表。
- 各行的参考量并不统一（有的按 σ/R，有的按 2σ 切割力，有的按归一化压强），表中 "基准" 列已如实给出该行自己的基准；跨行比较只在同 metric 内有效。
- `benchmarks/laplace/report.json` 的 `checks` 只存偏差量、未记单位，无法诚实地归一化成 (实测-基准)/基准，故未纳入本表。
- 本表不重跑仿真，因此无法发现"工件本身已过期于当前代码"的情况；它发现的是"文档与工件不一致"。

