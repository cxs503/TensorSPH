# 3-D 静水柱的静态一致性审计（零仿真）

**性质**: 验证欠账（复核已合并能力）｜**改动**: `benchmarks/hydrostatic_consistency/`（新 harness `run.py` / `make_doc.py` / `lid_experiment.py`）、`src/tensorsph/threed/solver.py`（新增 `run_hydrostatic(lid=...)`，默认行为不变）、`tests/test_hydrostatic_consistency.py`。本文所有数字由 `report.json` 经 `make_doc.py` 注入，零手抄。

## 1. 案例与问题

`benchmarks/threed_hydrostatic` 积分一个 8×8×16 的静止水柱，参考解 `p(z) = rho0 g (H - z)`。其已提交工件自判 `verdict = fail`：13 个核心层平均相对误差 **-17.58 %**、最差层 **-99.54 %**；账本 `docs/verification_ledger.md` 把这些读数记 FAIL/MARGINAL，但**没有任何机制定位误差来源**。一次 3692 步 float64 运行约 30 min/例（本机实测 0.56 s/步 × 2880 粒子），靠重跑做探测既贵又混杂：算子、密度闭合、自由面钳位、未收敛窗口混在一个读数里。

本审计只做能精确做且免费的那一半：把**精确解**喂给仓库自己的算子。方案若静水一致，残差应恒为 0。

```
p_i   = max(0, rho0 g (H - z_i))       # 解析压强，含壁面粒子
rho_i = tait_density(p_i)               # 与 Tait 一致的密度
a_i   = pressure_acceleration(pos, m, rho, p, h, pairs)[:, 2] - g
```

网格：`h/dx ∈ {1.3, 1.5}` × 壁厚 `{1,2,3}` 层 × 自由面 `{带盖, 开口}` × 壁面压强来源 `{解析, Adami 外推}`；Adami 变体把**解析流体场**喂给外推算子，测的是算子而不是启动瞬态。

## 2. 结果

### 2.1 静态不平衡 `a_z/g`（负号 = 支撑不足）

| h/dx | 壁厚 | 自由面 | 壁面压强 | 第0层 | 第8层 | 第14层 | 第15层 |
|---|---|---|---|---|---|---|---|
| 1.3 | 1 | 带盖 | Adami | -153.9493 | -1.5329 | -0.9582 | -0.8665 |
| 1.3 | 1 | 开口 | Adami | -153.9493 | -1.5329 | -0.9582 | -0.8665 |
| 1.3 | 2 | 带盖 | Adami | -1.7076 | -1.1504 | -0.5735 | -4.9816 |
| 1.3 | 2 | 开口 | Adami | -1.7076 | -1.1504 | -0.5735 | -0.4813 |
| 1.3 | 3 | 带盖 | Adami | -1.7076 | -1.1504 | -0.5735 | -4.9816 |
| 1.3 | 3 | 开口 | Adami | -1.7076 | -1.1504 | -0.5735 | -0.4813 |
| 1.5 | 1 | 带盖 | Adami | -294.5227 | -1.0915 | -0.5143 | -0.4264 |
| 1.5 | 1 | 开口 | Adami | -294.5227 | -1.0915 | -0.5143 | -0.4264 |
| 1.5 | 2 | 带盖 | Adami | -0.5080 | +0.0504 | +0.6343 | -7.9778 |
| 1.5 | 2 | 开口 | Adami | -0.5080 | +0.0504 | +0.6343 | +0.7232 |
| 1.5 | 3 | 带盖 | Adami | -0.5080 | +0.0504 | +0.6343 | -7.9778 |
| 1.5 | 3 | 开口 | Adami | -0.5080 | +0.0504 | +0.6343 | +0.7232 |

**① 壁厚是硬要求。** 1 层壁时壁邻层残差 -153.9493 % g，2 层壁 -1.7076 % g —— 相差 **90×**。h = 1.3 dx 时核半径 2h = 2.6 dx，一层壁填不满。生产值 `wall_layers=2` 必要且充分；h = 1.5 dx 下差距更大，因为 2h = 3 dx。

**② 自由面盖是伪边界（缺陷 DW-1）。** `wall_shell` 封的是**六个面**，所以 `run_hydrostatic` 默认配置在自由面之上还有两层壁粒子（z = H, H + dx），而 2h = 2.6 dx —— 正好落在最上层流体的支撑内。去掉它们，最上层残差从 -4.9816 % g 降到 -0.4813 % g（**10.3×**）；h = 1.5 dx 时甚至变号（-7.9778 → +0.7232 % g）。`run_hydrostatic` 的 docstring 写的是「底面与四个侧面」，实现与文档矛盾。

**③ 算子的地板是 1 % 量级。** 即使给精确解，内部（第8层）残差 -1.150 % g、壁邻层 -1.708 % g —— 压强梯度算子没有再现性修正，格点偏置就在这个量级。这既是任何静水读数的下限，也是它在 2-D/3-D 全部算例里的固有量。

### 2.2 与已提交读数的对撞

| 量 | 值 |
|---|---|
| 已提交工件 | `benchmarks/threed_hydrostatic/report.json`（adami-c0_20-nz16）|
| 工件自判 | fail；未过判据 ['interior_pressure_within_15pct_best', 'bottom_pressure_within_15pct_all', 'column_settled_window_drift_below_2pct', 'centre_of_mass_drift_below_1pct', 'compressibility_helps_as_predicted'] |
| 实测层面平均相对误差 | **-17.58 %**（15 层）|
| 静态不平衡隐含的平衡修正 | **+1.40 %** |
| 符号一致层数 | 0 / 15 |
| 两者均方根差 | 1.70 百分点 |
| 工件自报收敛标志 | {'centre_of_mass_drift_below_1pct': False, 'column_settled_window_drift_below_2pct': False} |

静态不平衡要求压强**上调** +1.40 %，而实测每一层都**偏低**（符号一致 0/15），量级差一个数量级以上。**「3-D 静水剖面误差 = 压强算子的静水不一致性」这一假设被证伪。** 残余偏差必来自别的机制，而工件自己报告的 `column_settled_window_drift_below_2pct = false` / `centre_of_mass_drift_below_1pct = false` 说明读数取在一个尚未停稳的窗口里。

### 2.3 自由面那一层是钳位问题，不是算子问题

| c0 [m/s] | 自由面核和亏缺 [%] | 首层静水压缩 [%] | 亏缺/信号 | 等价钳位压强 [Pa] | 首层解析压强 [Pa] | 钳位/信号 |
|---|---|---|---|---|---|---|
| 5 | 0.698 | 0.3924 | 1.8 | -171.0 | 98.1 | 1.7 |
| 10 | 0.698 | 0.0981 | 7.1 | -683.9 | 98.1 | 7.0 |
| 20 | 0.698 | 0.0245 | 28.5 | -2735.7 | 98.1 | 27.9 |
| 40 | 0.698 | 0.0061 | 113.9 | -10942.6 | 98.1 | 111.5 |

自由面核和亏缺是**格点属性**（不随 c0 变），静水压缩信号按 `1/c0^2` 缩小：c0 越大，钳位吃掉的比例越高。所以最上层将近 -100 % 的读数**不是**算子错，是 wcSPH 在这个格点/声速组合下**读不出**那一层的压强。初始格点的核和亏缺只在最上层出现（带盖时被盖填满、亏缺为 0），这正是 15 层里只有最上层塌到 -99.5 % 的结构性原因。

## 3. 判定

| 判据 | 结果 |
|---|---|
| `interior_lattice_bias_in_documented_band` | PASS |
| `one_wall_layer_at_least_20x_worse` | PASS |
| `open_top_cuts_surface_residual_5x` | PASS |
| `static_correction_has_opposite_sign_to_measurement` | PASS |
| `static_imbalance_below_measured_mean` | PASS |
| `surface_clamp_dominates_signal_10x` | PASS |
| `surface_clamp_ratio_grows_with_c0` | PASS |

benchmark 自判：**pass**。

**是否 < 3 %？** 本审计的判据是结构性与符号性断言（残差倍数、符号一致性、钳位比），不是对某个物理量的精度断言，故 3 % 判据不直接适用。唯一可量化的精度结论是：静水剖面读数的算子地板 ≈ 1.71 % g（< 3 %），但它比实测偏差小一个数量级 —— 所以**不能**把实测 FAIL 归因于算子。

## 4. 注意点与未完成

1. **DW-1 的修复默认关闭。** `run_hydrostatic(lid=False)` 是新增选项，默认 `True` 保持已提交工件的含义不变。`tests/test_hydrostatic_consistency.py::test_free_surface_face_has_no_wall_particles` 用 `xfail(strict=True)` 把「自由面之上不该有壁粒子」钉成待修事实：修好后它会转红，强制更新账本。
2. **开口重跑的验证已递交集群等待调度**（`cj958` 带盖 / `cj959` 开口，`adami-c0_20-nz16`、t_end = 0.3 s、1846 步 ≈ 15 min/例）。静态审计只能证明盖子给最上层注入 10× 的伪力，**不能**预测剖面误差会改善多少 —— 那需要完整重跑，本机 load15 ≈ 100，按容量纪律不在本机起。
3. **内部格点偏置**（第8层量级见上表）是压强梯度算子无再现性修正的固有量；本单只量化并钉住，没有动算子。
4. 参考：Adami, Ihmsen & Adams (2012) 壁面压强外推；SPH 静水一致性作为压强梯度的必要条件（Morris 1997 的边界压强梯度偏置讨论）。

5. 图：`benchmarks/hydrostatic_consistency/figures/hydrostatic_consistency.png`（左：带盖/开口的残差剖面；中：运行读数 vs 静态预测；右：钳位比随 c0）。
