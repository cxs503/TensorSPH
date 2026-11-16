# 超 15% 登记记录逐条裁定（mission 22787）

本文件由 `benchmarks/method_selection/adjudicate_over_tolerance.py` 程序化生成；所有数字都从产物/登记表读出，没有任何手抄值。

- 登记表记录数：30
- 审计侧完整性失败：无
- 超 15% 记录数：7（其中 accuracy 类 6 条）
- 覆盖率审计只复算了其中 5 条；审计目录之外的超限记录：['x_conduction_2d_one_d_calibration', 'x_csf_laplace_quasistatic']
- `resolution_attributable`：无
- `resolution_window`：['x_transport_velocity_coarse']
- `resolution_pair_only`：['x_couette_wall_shear']
- `reading_mobility`：['x_couette_viscous_time_mobility_h8']
- `genuine_accuracy_failure`：['x_csf_curvature', 'x_couette_viscous_time_top_h16', 'x_csf_laplace_quasistatic']
- `documented_trap_arm`：['x_conduction_2d_one_d_calibration']
- 决策泄漏（严格模式下仍在 15% 标准内被推荐）：无
- 被推荐方法同时挂着「记录在案的陷阱支」：['x_conduction_2d_one_d_calibration']（不是泄漏：陷阱支不承载出厂配置的精度主张，但会污染按方法聚合的最差值）
- 被推荐方法同时挂着「读数机动性」记录：无

“决策角色”一列区分两种完全不同的情形：超 15% 的记录**本身**就是排除某方法的依据，还是该方法因为另一条（更细档/更强）记录被排除、本记录只是并存。

## 逐条裁定

| 记录 | 方法 | 登记类别 | 登记 [%] | 复算 [%] | 裁定 | 决策角色 |
|---|---|---|---|---|---|---|
| `x_conduction_2d_one_d_calibration` | heat_conduction_operator | accuracy | 835 | 835 | **documented_trap_arm** | 严格模式首选 |
| `x_csf_curvature` | csf_surface_tension | accuracy | 23.8 | 23.8 | **genuine_accuracy_failure** | capillary droplet (Laplace law)：该方法被排除，但引用的最差记录是 45.7%（不是本记录） |
| `x_couette_wall_shear` | wcsph_wall_no_slip | accuracy | 32.6 | 32.6 | **resolution_pair_only** | planar Couette channel：该方法被排除，但引用的最差记录是 17.1%（不是本记录） |
| `x_couette_viscous_time_mobility_h8` | wcsph_wall_no_slip | robustness | 18.3 | 18.3 | **reading_mobility** | planar Couette channel：该方法被排除，但引用的最差记录是 17.1%（不是本记录） |
| `x_couette_viscous_time_top_h16` | wcsph_wall_no_slip | accuracy | 17.1 | 17.1 | **genuine_accuracy_failure** | planar Couette channel：本记录即排除所引用的最差记录 |
| `x_csf_laplace_quasistatic` | csf_surface_tension | accuracy | 45.7 | 未独立复算（stored only） | **genuine_accuracy_failure** | capillary droplet (Laplace law)：本记录即排除所引用的最差记录 |
| `x_transport_velocity_coarse` | transport_velocity_correction | accuracy | 17 | 17 | **resolution_window** | breaking wave over a slope：本记录即排除所引用的最差记录 |

## 分辨率响应（resolution_* 类别的证据）

一个「粗档 > 15% ≥ 细档」的配对只是与收敛相容，不是收敛的证据。`resolution_ladder_guard.py` 从仓库自带的完整分辨率阶梯重新判级：阶梯 ≥3 档、误差随加密不增、最细档在标准内，才给 `resolution_attributable`；阶梯回落或最细档重新超限给 `resolution_window`（通过区间是标定窗口的性质，不是收敛）；档数不足给 `resolution_pair_only`（单调性未测）。

- `x_couette_wall_shear` → **resolution_pair_only**：阶梯 {'8': 4.338367283344287, '16': 0.47810226678850043}（H/dx），最细档 0.48%，窗口 {'min': 8.0, 'max': 16.0}
  - only 2 rung(s) committed: a pair is consistent with convergence but cannot demonstrate it
### `x_couette_wall_shear` vs `x_couette_wall_shear_design`

- 参数：H/dx，粗 `8` → 细 `16`
  - benchmarks/couette/out/results.json: 8 → 32.6 %
  - benchmarks/couette/out_design/results.json: 8 → 4.34 %, 16 → 0.478 %

- `x_transport_velocity_coarse` → **resolution_window**：阶梯 {'0.04': 17.059208092684724, '0.03': 10.646706528059385, '0.02': 12.480203376603916, '0.01': 16.38573465320491}（dx [m]），最细档 16.39%，窗口 {'min': 0.02, 'max': 0.03}
  - 4 rungs; the pass covers only the measured window, not the refinement sequence: error rises under refinement (0.03->0.02: +1.83 pp, 0.02->0.01: +3.91 pp); finest rung 0.01 measures 16.39% > 15%
### `x_transport_velocity_coarse` vs `x_transport_velocity`

- 参数：dx [m]，粗 `0.04` → 细 `0.03`
  - registry records (both grids): 0.03 → 10.8 %, 0.04 → 17 %

## 覆盖率审计的盲区

以下记录的超限值从未被覆盖率审计独立复算过（不在其量目录内），因此任何“审计清单已覆盖全部超限记录”的说法都不成立：

- `x_conduction_2d_one_d_calibration`：835 %，裁定 **documented_trap_arm**，来源 `benchmarks/case_conduction_2d/result.json`
- `x_csf_laplace_quasistatic`：45.7 %，裁定 **genuine_accuracy_failure**，来源 `benchmarks/csf_consistent/relax_quasistatic.json`

## 判定的成立条件与反证

- `reading_mobility`：登记记录的 `kind` 必须是 `robustness`，且其值是同一读数的极差（本脚本从 sweeps 产物重算并比对）。
- `resolution_attributable` / `resolution_window` / `resolution_pair_only`：配对记录必须是同一方法的同一量，且粗档 > 15% ≥ 细档，再由完整阶梯定级；配对不成立时脚本把问题写进 `problems`（测试会因此变红）。
- `genuine_accuracy_failure`：没有更细档的同类记录落在 15% 以内；本脚本仍从原始产物独立复算一次，避免把陈旧数字当作失败证据。
