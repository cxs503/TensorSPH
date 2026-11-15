# XVAL-SPH-METHOD-SELECTION — 推荐上界到底站在几条独立记录上：证据深度与逐条抽出脆弱性审计（m21797）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ **任务**: m21797（自找活·验证欠账）

## ① 案例介绍

方法选择器（`src/tensorsph/method_selection.py`）对每个算例画像给出一份 SPH 格式推荐（主方案 + 修饰项）与一个精度上界。此前两轮复核问的是「数字对不对」（m20432 独立复算记录值）和「上界是否覆盖被推荐格式的全部在册记录」（m21537 严格模式）。**没有人问过：这些上界站在几条*独立*记录上。**

登记表共 30 条记录、10 个有证据的格式、13 个不同的基准族。若某格式的全部在册精度记录都来自同一个基准族，则该上界只是一次测量，删掉那条记录推荐就会静默变成 `WITHHELD`——这正是本单要量的东西。

七个标准画像取自 `benchmarks/method_selection/build_registry.py::PROFILES`（规则波水槽、Sod 激波管、毛细液滴、平面 Couette 槽道、Gresho 涡、一维瞬态导热、斜坡破碎波）。

## ② 怎么做的

1. **证据深度（独立复算）**：审计脚本直接读 `docs/reports/method_selection_registry.json`，自己判 live 精度记录（`kind=='accuracy'` 且 `status!='refuted'`），自己把 `CASE-X (variant)` 归并成基准族 `case-x`；**不复用**路由器的任何函数。然后把结果与路由器 `Decision.evidence_depth` 逐字段比对，不一致即硬失败。
2. **逐条抽出（leave-one-record-out）**：对每个画像，把登记表里 26 条记录**逐条删掉一条**重跑一次严格模式选择（`strict_bound=True`，即产出已发布推荐的那个模式），比较主方案 / 主方案状态 / 是否 verified / 修饰项 / 上界。**能改变推荐的记录**记为「承重记录」（load-bearing）。
3. **不变量（本脚本自己断言）**：删掉 `refuted` 记录或 `robustness`（conditioning）记录必须**什么都不变**——它们不属于任何精度声明。任何违反都会被写进 `inert_removals` 并置 `invariants_ok=false`。
4. 全部数字由本脚本写入 JSON/MD/PNG，零手抄；`--check` 与磁盘上的文件逐字节比对，是测试套件用的防漂移门。

## ③ 计算结果 —— 证据深度（逐格式）

| 格式 | 记录数 | live 精度记录 | 基准族 | 最差 [%] | 最好 [%] | 极差 [pp] | 单一来源 |
|---|---|---|---|---|---|---|---|
| `compressible_riemann` | 2 | 2 | 2 | 0.7768 | 0.4142 | 0.3626 | 否 |
| `compression_regularization` | 1 | 1 | 1 | 5.533 | 5.533 | — | **是** |
| `csf_surface_tension` | 4 | 4 | 1 | 45.69 | 5.599 | 40.09 | **是** |
| `density_diffusion` | 2 | 2 | 1 | 1.021 | 0.482 | 0.5389 | **是** |
| `heat_conduction_operator` | 3 | 2 | 2 | 0.1504 | 0.0002181 | 0.1502 | 否 |
| `isph_projection` | 2 | 2 | 1 | 0.8666 | 0.04583 | 0.8208 | **是** |
| `particle_shifting` | 1 | 1 | 1 | 8.833 | 8.833 | — | **是** |
| `transport_velocity_correction` | 2 | 2 | 1 | 16.96 | 10.81 | 6.152 | **是** |
| `wcsph_delta_sph` | 1 | 1 | 1 | 2.968 | 2.968 | — | **是** |
| `wcsph_wall_no_slip` | 12 | 7 | 2 | 17.08 | 0.4781 | 16.6 | 否 |

**结论**：10 个格式里 **7 个是单一来源**（`compression_regularization`, `csf_surface_tension`, `density_diffusion`, `isph_projection`, `particle_shifting`, `transport_velocity_correction`, `wcsph_delta_sph`）。按业主标准 15 % 判级，这些上界属于**一次测量的数据点**，不属于经独立复现的 bound。

## ④ 逐画像：承重记录（删掉就变）

| 画像 | 严格模式主方案 | 上界 [%] | 承重记录数 | 变化 |
|---|---|---|---|---|
| regular wave in a flume | `wcsph_delta_sph` | 5.533 | 2 | 删 `compression_regularization` → modifiers/withheld；删 `wcsph_delta_sph` → primary/primary_status/verified/modifiers |
| Sod shock tube | `compressible_riemann` | 0.7768 | 1 | 删 `compressible_riemann` → 上界变 |
| planar Couette channel | `isph_projection` | 0.8666 | 2 | 删 `isph_projection` → 上界变；删 `wcsph_wall_no_slip` → withheld |
| 1-D transient conduction | `heat_conduction_operator` | 0.1504 | 1 | 删 `heat_conduction_operator` → 上界变 |
| breaking wave over a slope | `wcsph_delta_sph` | 8.833 | 4 | 删 `compression_regularization` → modifiers/withheld；删 `particle_shifting` → modifiers/withheld；删 `transport_velocity_correction` → modifiers/withheld；删 `wcsph_delta_sph` → primary/primary_status/verified/modifiers/withheld |

删掉 `refuted` / `robustness` 记录造成的推荐变化：**0** 处（不变量 成立）。

## ④b 逐条抽出的连带发现：证据缺失时的「偷换求解器」

抽掉某个 regime 自己的记录后，选择器会不会改口推荐一个物理上不适用的格式？本脚本直接做这个探测（两种模式都探）：

| regime | 被抽掉的格式 | 模式 | 结果主方案 | verified | regime 锁定 |
|---|---|---|---|---|---|
| `shock` | `compressible_riemann` | lenient | （无） | 否 | 是 |
| `shock` | `compressible_riemann` | strict | （无） | 否 | 是 |
| `capillary` | `csf_surface_tension` | lenient | （无） | 否 | 是 |
| `capillary` | `csf_surface_tension` | strict | （无） | 否 | 是 |

**修复前**上表两行 shock/lenient 复现为 `isph_projection`（不可压投影求解器拿去算 M=0.9 激波管），capillary/lenient 复现为 `wcsph_delta_sph`（没有表面张力项去算毛细主导液滴）——即「记录一缺就静默换一个物理上不适用的求解器」。修复：`EXCLUSIVE_RULES`（`shock`/`capillary`）一旦触发而候选全部不可用，规则walk 立即停止并留空，不向更宽泛的规则降级；回归测试 `tests/test_method_selection_fragility.py::test_shock_rule_never_falls_through_to_a_projection_solver` 与 `...::test_capillary_rule_never_falls_through_to_surface_tension_free_wcsph` 钉住。

## ⑤ 顺带发现并修掉的缺陷

- m21537 的 `benchmarks/method_selection/strict_bound_profiles.py` 把登记表里已是百分数的值又乘了 100：表格里 `csf_surface_tension` 的上界印成 2379.93 %、破碎波画像印成 883.26 %/1696.26 %，而卡片正文写的是 23.8 %/8.83 %。同一脚本的 PNG 也按 100 倍绘制。本单把 `_fmt_pct` 与绘图改为直接用登记值，并加回归测试：MD 表格里印的百分数必须能反解回 JSON 值（相对差 < 1e-6）。

## ⑤ 注意点与适用范围

* 「单一来源」不是错误，它是对**证据强度**的诚实标注：一个算例两条分辨率臂仍只是同一个基准族，跨族复现才算独立。
* 逐条抽出只测「删一条记录」这一种扰动；它不判断记录本身是否可信（那是 m20432 的复算）也不判断上界是否越线（m21537）。
* 严格模式下被剔除的格式不进 `endorsed`，因此它的记录不参与承重统计——这是有意的：审计对象是**已发布推荐**。
* 本单没有新的物理算例，也就没有新的「与解析解/文献对标的物理量」；误差判定只能落在**复算一致性**上（见 ⑥）。

## ⑥ 误差判定：复算是否一致

| 关键量 | 基准 | 本单值 | 差异 | 是否 < 3 % |
|---|---|---|---|---|
| router `evidence_depth` vs 独立复算（10 格式 × 7 字段） | 逐字段相等 | 不一致 0 处 | 0 | 是（要求严格相等） |
| 承重统计对不变量（refuted/robustness 记录不可承重） | 期望 0 | 0 | 0 | 是（要求严格相等） |
| 推荐上界是否越过业主标准 | 15 % | 见 ④ 表 | — | 上界均 ≤ 15 %（严格模式定义所致） |

**明确回答**：「物理量误差 < 3 %」这一项在本单**不适用**——本单不产生新的物理量，只对已有登记表做证据强度与稳健性审计；我不声称任何物理精度结论。可答的是复算一致性，见上表（要求严格相等，已满足）。

