> **后续修正（m22957）**：本文档中 `resolution_attributable` 的两条裁定已被`docs/xval_method_selection_resolution_ladder_guard_m22957.md` 修正——一档配对不构成收敛证据。`x_transport_velocity_coarse` 现为 `resolution_window`（四位阶梯非单调，最细档 16.39% 重新超限），`x_couette_wall_shear` 现为`resolution_pair_only`（仅两档）。其余裁定不变，决策无泄漏、无变化。

# 方法选择账本的两类开口：审计误报 + 超 15% 记录逐条裁定（m22787）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ **任务**: m22787（自找活·验证欠账）
**类型**: 验证欠账清偿（adversarial-review 口径）｜ **产物**: 代码修复 + 新裁定脚本 + 16 条守卫测试 + 文档 + 图

## ① 案例介绍（问题从哪来）

锚点结论的载体是 `docs/reports/method_selection_registry.json`（30 条记录，每条 = 某方法某次测量的误差），
而 `benchmarks/method_selection/audit_quantity_coverage.py`（m20432）是对它的**独立复算审计**。
该审计输出两个"开口"并停在那里：

| 开口 | m20432 输出 | 之前是否被裁定 |
|---|---|---|
| `integrity_failures` | `['x_couette_design']`（登记值与独立复算不一致） | 否（m22608 明确留下） |
| `records_over_owner_tolerance` | 5 条记录 > 业主 15% 标准 | 否（m22608 明确留下） |

这两类开口的危险在于双向误读：一份"超限清单"既可能被读成「该方法被否掉」，也可能被读成
「账本会自我报警所以可信」。两者都不成立——除非逐条知道**这个量是什么、在哪个分辨率上读的、
以及是否有已公布的决策真的踩在它上面**。

## ② 怎么做的（方法与关键设置）

1. **先复现**：在干净检出上重跑两个生产者 `--check`（均绿），确认开口是已合并状态而非我引入。
2. **定位 integrity 误报**：`x_couette_design` 的独立复算用 `rows[1:-1]`（按**位置**切行），
   而算例自身在 `benchmarks/couette/run.py:profile_metrics` 里把"内部带"定义为
   `0.15 <= y_over_H <= 0.85`（按**位置在通道中的高度**）。两者只在 H/dx = 8 时巧合一致
   （两侧各 1 行落在带外），H/dx = 16 时两侧各 2 行落在带外 → 复算 2.851 % vs 登记 2.933 %。
   **是审计脚本的缺陷，不是产物的缺陷。** 修复：`r_couette_l2` 改按声明带取行，
   并保留一条负控制测试钉住"位置切片在 H/dx=16 必不一致"。
3. **把清单从"审计所能复算的量"扩到"账本全部记录"**：业主的账本是 registry，审计只是它的一个视图。
   按 `measured_pct > 15` 从 registry 重取超限集合，并显式报告**审计盲区**
   （`audit_coverage_gap`：超限但不在审计量目录内的记录）。
4. **逐条裁定（只用数据，不用标签）**，脚本 `benchmarks/method_selection/adjudicate_over_tolerance.py`：
   - `reading_mobility`：登记 `kind == robustness`，其值是同一读数在扫描内的**极差**，从 sweeps 产物重算并比对；
   - `resolution_attributable`：同一方法的同一量在**更细档** ≤ 15%（配对显式登记 `RESOLUTION_PAIRS`，
     由 `_check_pairing` 做数据校验：同方法、粗档 > 15 ≥ 细档、量的措辞一致），并在图上给出分辨率响应；
   - `genuine_accuracy_failure`：以上皆非，从原始产物独立复算一次；审计目录外的记录标 `stored only`（诚实降级）；
   - `documented_trap_arm`：产物自己把该支称为陷阱支，并在同一文件里给出出厂支的误差。
5. **决策联动**：读 `method_selection_strict_bound_m21537.json`，对每条记录回答
   "它本身是不是某次排除所引用的最差记录"（`cites_this_record`），并检查是否存在
   **泄漏**：严格模式下仍在 15% 标准内被推荐、而它同时带着超限的精度主张。
6. **守卫**：新增 `tests/test_method_selection_open_flags_adjudication.py`（16 条），覆盖
   带定义、负控制、集合覆盖、分类的数据依据、配对校验、陷阱支、泄漏、决策引用覆盖、两个生产者 `--check` 新鲜度。

## ③ 计算结果（程序化注入，来自 `docs/reports/method_selection_open_flags_adjudication_m22787.json`）

### 3.1 审计开口清零与盲区

| 项 | 修复前 | 修复后 |
|---|---|---|
| `integrity_failures` | `['x_couette_design']` | `[]` |
| `stored_vs_recomputed_mismatches` | `['x_couette_design']` | `[]` |
| 超 15% 记录数（registry 口径） | 审计只报 5 | **7**（审计盲区 2 条） |

### 3.2 七条超限记录的裁定

| 记录 | 方法 | 类别 | 登记 [%] | 独立复算 [%] | 裁定 | 决策角色 |
|---|---|---|---|---|---|---|
| `x_conduction_2d_one_d_calibration` | heat_conduction_operator | accuracy | 834.6 | 834.6（陷阱支；出厂支 2.42） | `documented_trap_arm` | 不承载决策；但该方法是 1-D 瞬态导热剖面的首选 |
| `x_csf_laplace_quasistatic` | csf_surface_tension | accuracy | 45.69 | 未独立复算（stored only） | `genuine_accuracy_failure` | 毛细剖面排除的**最差记录**（`cites_this_record`） |
| `x_couette_wall_shear` | wcsph_wall_no_slip | accuracy | 32.65 | 32.65 | `resolution_attributable`（H/dx 8→16：32.65→0.478） | 不是排除依据（排除引用的是 17.08 那条） |
| `x_csf_curvature` | csf_surface_tension | accuracy | 23.80 | 23.80 | `genuine_accuracy_failure` | 不是排除依据（引用的是 45.69） |
| `x_couette_viscous_time_mobility_h8` | wcsph_wall_no_slip | robustness | 18.26 | 18.26（极差重算一致） | `reading_mobility` | 不是排除依据 |
| `x_couette_viscous_time_top_h16` | wcsph_wall_no_slip | accuracy | 17.08 | 17.08 | `genuine_accuracy_failure`（H/dx 8→16：14.20→17.08，随加密**变差**） | **本记录即排除依据** |
| `x_transport_velocity_coarse` | transport_velocity_correction | accuracy | 16.96 | 16.96（dx 0.04→0.03：16.96→10.81） | `resolution_attributable` | **本记录即排除依据**（破波剖面 modifier） |

### 3.3 判决

- **泄漏检查：无泄漏。** 没有任何严格模式决策在 15% 标准内推荐一个同时带着**活精度主张** > 15% 的方法。
- 被推荐方法 `heat_conduction_operator` 同时挂着 834.6% 的陷阱支记录（`endorsed_with_trap_arm`）：
  陷阱支是产物自认的负控制，不承载出厂支的精度主张，**但会污染按方法聚合的最差值**。
- **两条排除的真实理由不同**：`x_couette_viscous_time_top_h16`（17.08%）随加密变差 →
  排除稳健；`x_transport_velocity_coarse`（16.96%）在更细档 10.81% < 15% →
  排除是"在 dx=0.04 m 上未取证"，而不是"该方法过不了 15% 标准"。这条差别此前只写在记录措辞里，
  现在是脚本判定 + 图 + 守卫。

## ④ 结果图

`docs/figures/method_selection_m22787/open_flags.png`：
左图 = 7 条超限记录的登记误差按裁定类别着色，15% 标准线；右图 = 两条 `resolution_attributable`
记录在各自算例的分辨率响应（Couette 底壁剪切 8→32.65 %、16→0.478 %；TVC 前缘 0.04→16.96 %、0.03→10.81 %）。
两条曲线都跨越 15% 线，这是"归因于分辨率"的图形证据。

## ⑤ 注意点（踩过的坑 / 局限）

- **`x_csf_laplace_quasistatic` 未能独立复算**（`stored only`）：它的量是"4–6 t_sigma 窗口内 bulk 压强对真空"，
  产物 `benchmarks/csf_consistent/relax_quasistatic.json` 未提供可重算的 measured/reference 原始对。
  因此它是"按产物自述的精度主张超限"，不是"已被独立复算证伪"。这条不确定性**没有被抹平**。
- **审计盲区是本次新发现**：审计量目录（28 条）比账本（30 条）少 4 个 extractor，
  其中 2 条超限（834.6%、45.69%）。9 条审计结论此前默认"清单即全集"，这个默认是错的。
- 我没有改任何**已公布决策**（严格/宽松模式的选择、排除），也没有把 `x_transport_velocity_coarse`
  的排除"修正"为推荐：那是业主的策略问题（最差记录规则 vs 分辨率取证），不是审计可以自行裁决的。
- `resolution_attributable` 的配对是**显式登记**的两对，脚本用数据校验其成立；
  它不是自动配对算法，不会自己发现新的粗/细档对——新增配对必须人工登记并由守卫校验。

## ⑥ 误差判定（对基准逐项）

| 关键量 | 值 | 基准 | 误差 | <3 % ? |
|---|---|---|---|---|
| Couette H/dx=16 内部 L2（复算 vs 产物登记） | 2.93278 % | 2.93278 %（登记） | 3e-15 % | **是** |
| 位置切片（旧缺陷）vs 登记 | 2.85142 % | 2.93278 % | 2.77 % | 否（正是本次修复的缺陷） |
| 底壁剪切 H/dx=16 | 0.478 % | 0（解析 Couette） | 0.478 % | **是** |
| 底壁剪切 H/dx=8 | 32.65 % | 0（解析） | 32.65 % | 否（判别为分辨率症状） |
| 移动壁剪切 H/dx=16 稳态末 | 17.08 % | 0（解析） | 17.08 % | 否（真实系统性偏差，随加密变差 → 排除成立） |
| TVC 前缘平均相对误差 dx=0.03 m | 10.81 % | Martin & Moyce 数字化点 | 10.81 % | 否（<15 % 标准，未达 3 %） |
| 机动性极差（H/dx=8，重算 vs 登记） | 18.2551 % | 18.2551 %（登记） | <1e-9 % | **是** |

**锚点验收（<15 %）**：本次工作不改变任何方法选择结论；账本里 7 条超限记录中，
4 条经裁定不是活精度主张或归因于分辨率，2 条（45.69 %、17.08 %）是真实的 >15 % 证据且已被
严格模式 WITHHOLD/排除，1 条（834.6 %）是记录在案的陷阱支。**结论：已在严格模式下的推荐没有一个
依赖 >15 % 的记录**；TVC 的排除是分辨率取证问题，需业主裁决。