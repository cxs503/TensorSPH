# 初始和密度晶格偏置 `b` 的闭式解、实测验证与质量校准（TensorSPH / m3209）

**物理背景**：本仓用「密度求和」（summation density）而不是连续性方程积分来获得
`rho`。求和是对核函数的**晶格求积**，因此在完整支撑（complete support）的粒子
上并不精确等于 `rho0`，而是

```
rho_i / rho0 = 1 + b,      b = m * S / rho0 - 1,   S = W(0) + sum_{j!=i} W(r_ij,h)
```

`b` 只由 `h/dx` 与晶格几何决定，与 `dx`、`rho0`、`c0`、物体形状无关。经 Tait 状态
方程，它变成一个**均匀的虚假初始压强**
`p_spurious = B[(1+b)^gamma - 1]`，`B = rho0 c0^2/gamma`。

这解释了 `backlog_update_m2971.md`（P1）里「`b ~ 5.9e-3`、2395 Pa」的来历，并把
它从一次手工观测升级为可预测、可验证、可校准的闭式量；同时给出该 P1 的适用范围
（下表显示它并非对每个算例都成立）。

## 1. 案例与参数

| 项 | 值 |
|---|---|
| 液滴半径 R | 0.02 m |
| 表面张力 sigma | 7.2 N/m（水的 10 倍，缩短毛细时间） |
| 参考密度 rho0 | 1000.0 kg/m^3 |
| Tait 指数 gamma | 7.0 |
| 核支撑 h/dx | 1.3（2-D cubic spline） |
| 分辨率 dx | 0.002 m |
| 体模量 B(c0=20 m/s) | 57142.9 Pa |
| Laplace 压差 sigma/R | 360.0 Pa |
| 毛细时间 t_sigma | 0.0333 s |
| 液滴晶格 | hex（benchmarks/laplace_law 的生成器） |
| 水池晶格 | square（`make_particles`/`cell_centres`） |
| m2971 报告的偏置 / 压强 | 0.00588 / 2395.0 Pa |

## 2. 方法

1. `src/tensorsph/initial_density.py`：`lattice_bias(dx,h,kind)` 在截断到核支撑之外
   的有限晶格上直接求 `W(0)+sum W`，得到闭式 `b`；`calibrate_mass` 施加 `m/(1+b)`。
2. **独立实测**：`measure_complete_support_bias` 用 `density_from_neighbor_pairs`
   （另一条代码路径）在真实粒子集上取 `rho` 的最大值——完整支撑粒子的和密度在
   规则晶格上平移不变，且没有别的粒子能达到它，故最大值即 `rho0(1+b)`。
3. 两条路线的对比就是本报告的判据：闭式解必须复现实测（G1/G2a），而不是反过来
   用实测去拟合闭式。
4. 时间域：同一液滴（CSF consistent + fit_quadratic 曲率）跑 1.5 个毛细时间，
   分别以未校准 / 校准质量初始化，比较核心压强与虚假流动。

## 3. 结果

### 3.1 闭式 vs 实测（关键表）

| 位形 | 实测 b | 闭式 b | 相对差 | 完整支撑粒子数 | 判据 |
|---|---|---|---|---|---|
| hex 液滴（真实生成器） | +5.8835e-03 | +5.8835e-03 | 1.13e-13 | 235 | pass (G1 <1e-9) |
| 理想 square 晶格（float64） | -5.3231e-05 | -5.3231e-05 | 3.13e-11 | — | pass (G2a <1e-9) |
| `make_particles` 水池（float32） | -5.0659e-05 | -5.3231e-05 | 4.83% | 9 | pass (G2 <10%) |

hex 的闭式解与独立实测相差 `1.1e-13`，即 m2971 手工测到的
`5.88e-3` 现在有了完整解释（不是数值噪声，是晶格求积误差）。square 水池的 5.0%
残差**已归因**：`cell_centres` 用 float32 生成晶格，间距抖动 ~3e-6 相对量级，而
`b~5e-5` 的求积误差对间距敏感；float64 理想晶格上残差降到 `3.1e-11`。

### 3.2 `b` 随 `h/dx` 变化（两种晶格）

| h/dx | square b | square p_spur@c0=20 | hex b | hex p_spur@c0=20 |
|---|---|---|---|---|
| 1.10 | +1.8340e-03 |   +737.6 Pa | -1.6858e-03 |   -670.9 Pa |
| 1.20 | -2.4269e-04 |    -97.0 Pa | +3.2247e-03 |  +1302.4 Pa |
| 1.30 | -5.3231e-05 |    -21.3 Pa | +5.8835e-03 |  +2395.4 Pa |
| 1.45 | +2.9698e-03 |  +1198.6 Pa | +2.8648e-03 |  +1155.8 Pa |
| 1.60 | +2.9751e-03 |  +1200.7 Pa | +1.8632e-04 |    +74.6 Pa |

两个结论：(a) `b` **不是**随细化单调下降的误差，而是随 `h/dx` 变号振荡的求积误差；
(b) 本仓默认的 square 晶格在 `h/dx=1.3` 恰好接近零点（`b=-5.3231e-05`），
所以 **m2971 P1 的「任何 wcSPH 算例都带 2.4 kPa」只在 hex 晶格上成立**——这是对
该文件的一处更正，证据即上表。

### 3.3 压强代价与声速

| c0 [m/s] | p_max [Pa] | B [Pa] | p_max/(rho0 c0^2) | p_max/(sigma/R) |
|---|---|---|---|---|
| 20 | 2395.4 | 57143 | 5.9884e-03 | 6.65 |
| 10 | 598.8 | 14286 | 5.9884e-03 | 1.66 |
| 5 | 149.7 | 3571 | 5.9884e-03 | 0.42 |

`p_max/(rho0 c0^2)` 在 c0 阶梯上不变（G6 pass），即偏置是纯几何量、压强按 `c0^2`
缩放，与 m2971「b 与 c0 无关」的观察一致。降到 c0=5 m/s 确实把虚假压强降到
150 Pa，但仍为 `sigma/R` 的 0.42 倍——降声速不是出路（与 m2971 的 C4 结论一致）。

### 3.4 质量校准前后

| 量 | 未校准 | 校准后 |
|---|---|---|
| 核心 `rho/rho0 - 1` | +5.8835e-03 | +0.0000e+00 |
| `p_max` @c0=20 [Pa] | 2395.36 | 1.776e-10 |
| `p_max/(rho0 c0^2)` | 5.9884e-03 | 4.441e-16 |
| 校准因子 1/(1+b) | 1 | 0.994150891 |

校准是**标量**，所以质量分布形状（均匀、或 `hydrostatic_density` 的剖面）保持不变，
只是把「晶格常数质量 `rho0 dx^2` 并不代表 `rho0`」这一事实除掉。总质量随之变化
`0.58%`——这正是修正量本身。

### 3.5 时间域：校准前后的液滴弛豫

| 量 | 未校准 | 校准 |
|---|---|---|
| 初始核心压强 [Pa] | 2395.36 | 7.599e-11 |
| t=1.5 t_sigma 核心压强 [Pa] | 199.7 | 208.2 |
| 尾部 40% 时间平均核心压强 [Pa] | 236.7 | 229.3 |
| max|v|/v_ref 峰值 | 0.122 | 0.115 |
| max|v|/v_ref 终值 | 0.032 | 0.030 |
| 步数 / dt [s] | 600 / 8.333e-05 | 600 / 8.333e-05 |

![initial density bias](benchmarks/initial_density_bias/initial_density_bias.png)

图注：左上 `b(h/dx)` 闭式曲线与两处实测点（星号）；右上同一偏置换算成的虚假压强，
红虚线为 `sigma/R=360 Pa`；左下虚假压强 vs `c0`（`c0^2` 缩放）；右下弛豫过程核心
压强（校准后与 `b=0` 重合，未校准从 2395 Pa 起跳）。

## 4. 误差判定

| 关键量 | 基准 | 本实现误差 | 是否 <3% |
|---|---|---|---|
| hex 晶格 `b` | 独立实测（`density_from_neighbor_pairs`） | 1.13e-13 | 是 |
| 理想 square 晶格 `b` | 独立实测 | 3.13e-11 | 是 |
| 未校准 `p_max`@c0=20 | m2971 的 2395 Pa | 1.49e-04 | 是 |
| 实测 `b` | m2971 的 5.88e-3 | 5.99e-04 | 是 |
| `make_particles` 水池 `b` | 闭式（理想晶格） | 4.83% | **否（5.0%）** |
| 校准后 `p_max/(rho0 c0^2)` | 0 | 4.4e-16 | 是 |

唯一超过 3% 的项是 `make_particles` 水池的 `b`，且已定位到 float32 晶格抖动
（`cell_centres`），不是模型误差：同一实测方法在 float64 理想晶格上给出 3.1e-11。

## 5. 注意点 / 局限

- **校准时总质量变化 `1/(1+b)`**：对 `b<0` 的晶格（本仓 square 默认）质量变大，
  对 hex 变小。若某算例总质量本身是标定量（如按注入体积给定），需重新核对——
  本仓各 benchmark 均由 `rho0` 与几何派生质量，不属此列。
- **不修自由面**：部分支撑粒子的密度亏缺（`rho<rho0`，压强被钳到 0）是另一个缺陷，
  由 `benchmarks/laplace_pressure_floor` 与自由面压强闭合线负责；质量校准只消掉
  「完整支撑区的均匀偏置」，两者的误差不可混算。
- **默认关闭**：`Config.calibrate_initial_mass = False` 保持已发表数字逐位可复现；
  需要用它的算例显式打开（hex 类算例建议必开）。关闭时 `make_particles` 的质量与
  历史版本逐位相同，有测试钉住。
- **半解析性**：闭式解在有限晶格上求和，`n_half` 需覆盖 2h 支撑（实现自动取
  `ceil(2h/dx)+2`）；若 `h/dx` 极端大（>3），求和域需同步放大。
- 时间域证据只跑到 1.5 个毛细时间，未做长时间统计；核心压强的**稳态**归因
  （C3 自由面闭合）不在本单范围。

## 6. 判据汇总（机器裁决）

| 判据 | 结果 |
|---|---|
| G1_analytic_matches_hex_drop | pass |
| G2_analytic_matches_square_pool | pass |
| G2a_analytic_matches_ideal_square_lattice | pass |
| G3_bias_depends_on_h_over_dx_only | pass |
| G4_calibration_zeroes_bias_and_pressure | pass |
| G5_reproduces_m2971_offset | pass |
| G6_pressure_cost_scales_as_c0_squared | pass |
| **全部通过** | **是** |

复现：`PYTHONPATH=src python benchmarks/initial_density_bias/run.py`（约 40 s，
`--quick` 更快），文档由 `make_doc.py` 从 `report.json` 注入生成；测试
`tests/test_initial_density_calibration.py`（16 passed）。
