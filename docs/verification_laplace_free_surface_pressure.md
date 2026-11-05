# 自由面压强闭合：Young–Laplace 压差的门槛项已跨过（验证记录）

**生成方式**：`benchmarks/laplace_pressure_floor/make_doc_fs.py` 从
`benchmarks/laplace_pressure_floor/report.json` 读出（本文所有数字均由程序注入，禁止手改）。

> 上游：`docs/verification_laplace_pressure_floor.md` §7 把「自由面压强闭合」列为
> Young–Laplace 验证唯一未跨过的门槛项；实现已落在
> `Config.surface_pressure_closure`（`docs/verification_laplace_pressure_closure.md`）。
> 本文补的是缺的那一半：**同一算例上把闭合打开、跨 (dx, c0) 网格做 A/B 与迭代阶梯**，
> 并把「精度 / 虚假流」的取舍摊开，回答「该不该、以何种形式翻转默认值」。

## 1. 案例

二维液滴、无重力、CSF 表面张力（Brackbill 1992）撑住 Young–Laplace 压差。
所有量取自 `report.json` 的 `reference` 段：

| 量 | 值 |
|---|---|
| 半径 R | 0.02 m |
| 表面张力 σ | 7.2 N/m（10× 水值，缩短毛细时间） |
| 密度 ρ0 | 1000.0 kg/m³ |
| Tait 指数 γ | 7.0 |
| 解析压差 σ/R | **360.0 Pa** |
| 毛细时间 t_σ | 0.0333 s |
| 窗口 | 1.5 t_σ |
| 网格 | dx = 0.002 / 0.0015 m；c0 = 20 / 10 / 5 m/s |

读出口径固定为 `absolute` + 时间平均（末 40% 窗口，`absolute_time_average_tail`）：
Tait 律以 ρ0 归一（p(ρ0)=0），真空即 p=0 参考，故 `p_core - 0` 才是物理压差。
上一轮三条验证（-93.6 % / -75.6 % / -87.6 %）的 `bandA` 口径把自由面密度亏缺当流体压强，
本文仅作对照保留。

## 2. 做法

**算子**（`src/tensorsph/droplet_impact.py::free_surface_pressure`，Adami, Ihmsen & Adams
2012 §3.3）：kernel 支撑被自由面截断的粒子（`rho < surface_pressure_threshold·rho0`），
其压强不从自身亏缺密度读，而是从邻居外推

```
p_i = max(0, sum_j (p_j + rho_j g.(r_i - r_j)) W_ij / sum_j W_ij)
```

`max(0,·)` 保证该闭包只抬压强、不造拉伸；body 项 `rho_j g.(r_i-r_j)` 是无重力算例的
恒等项，但在静水/撞池算例里决定外推是否与壁压外推相容。

**本轮新增**：`Config.surface_pressure_monotone`。字面（`donors="all"`）单遍公式会被接收者
自己的 `p = 0` 稀释，所以它在薄壳上收敛不到内层值；把它按
`p <- max(p, extrapolate(p))` 迭代 `surface_pressure_layers` 遍，内层压强才能一层层往外搬，
且**只升不降**（不会把某遍已经建立的壳体压强又按回零）。该开关默认 `False`，
关闭时循环语义与旧实现逐位一致（`tests/test_free_surface_pressure.py` 钉住）。

因此本轮 A/B 的变体（`benchmarks/laplace_pressure_floor/run.py::CLOSURE_VARIANTS`）：

| 变体 | donors | passes | monotone |
|---|---|---|---|
| `all` | all | 1 | 否 |
| `interior-2` | interior | 2 | 否 |
| `all-iter-3` | all | 3 | 是 |
| `all-iter-6` | all | 6 | 是 |

工具链：`run.py --free-surface-only` 逐点跑同一算例（同 `relax`、同读出口径），按
(dx, c0, variant) 合并进 `report.json`；`make_doc_fs.py` 生成本文与
`free_surface_pressure.png`；`tests/test_free_surface_pressure.py` 把下表钉成门禁。

## 3. 结果

解析参考 σ/R = 360.0 Pa；误差 = (测量 − 解析)/解析。见
`free_surface_pressure.png`：(a) 压强剖面 (b) A/B 误差柱 (c) 精度 vs 虚假流代价。

### 3.1 A/B 矩阵

| dx [m] | c0 [m/s] | 关闭 Δp [Pa] | 关闭误差 | all Δp [Pa] | all 误差 | all-iter-3 Δp [Pa] | all-iter-3 误差 | all-iter-6 Δp [Pa] | all-iter-6 误差 | interior-2 Δp [Pa] | interior-2 误差 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0015 | 5 | 225.3 | -37.4 | 271.1 | -24.7 | 337.0 | -6.4 | — | — | 356.7 | -0.9 |
| 0.0015 | 10 | 229.7 | -36.2 | 281.3 | -21.8 | 335.6 | -6.8 | — | — | 343.4 | -4.6 |
| 0.0015 | 20 | 222.9 | -38.1 | 269.1 | -25.3 | 335.6 | -6.8 | — | — | 347.7 | -3.4 |
| 0.002 | 5 | 237.6 | -34.0 | 288.5 | -19.9 | 358.0 | -0.6 | 380.1 | +5.6 | 377.5 | +4.9 |
| 0.002 | 10 | 226.0 | -37.2 | 306.0 | -15.0 | 352.9 | -2.0 | 345.7 | -4.0 | 345.5 | -4.0 |
| 0.002 | 20 | 223.6 | -37.9 | 287.5 | -20.1 | 353.4 | -1.8 | 403.5 | +12.1 | 391.8 | +8.8 |

### 3.2 迭代阶梯（monotone passes，括号内为终态 v_max/v_ref）

| dx [m] | c0 [m/s] | n=3 | n=6 |
|---|---|---|---|
| 0.0015 | 5 | -6.4 (0.04) | — |
| 0.0015 | 10 | -6.8 (0.04) | — |
| 0.0015 | 20 | -6.8 (0.03) | — |
| 0.002 | 5 | -0.6 (0.03) | +5.6 (0.31) |
| 0.002 | 10 | -2.0 (0.03) | -4.0 (0.38) |
| 0.002 | 20 | -1.8 (0.03) | +12.1 (0.27) |

### 3.3 误差判定（逐项、显式回答是否 <3 %）

| 配置 | 最差误差（带符号） | 最差绝对误差 | <3 % ? / 虚假流 |
|---|---|---|---|
| 关闭（现默认），最差 6 点 | -38.1 % | 38.1 % | 否 |
| 开启 `all`，最差 6 点 | -25.3 % | 25.3 % | 否；$v_{max}/v_{ref}$ 最差 0.02 |
| 开启 `all-iter-3`，最差 6 点 | -6.8 % | 6.8 % | 否；$v_{max}/v_{ref}$ 最差 0.04 |
| 开启 `all-iter-6`，最差 3 点 | +12.1 % | 12.1 % | 否；$v_{max}/v_{ref}$ 最差 0.38 |
| 开启 `interior-2`，最差 6 点 | +8.8 % | 8.8 % | 否；$v_{max}/v_{ref}$ 最差 0.39 |

**最佳变体 `all-iter-3`**：6 点中 6 点在 ±15 % 带内、3 点在 ±3 % 内，
最差 6.8 %；最差虚假流 v_max/v_ref = 0.04
（Adami 判据 < 0.1：满足）。
作为对照，同一网格关闭闭合时最差 38.1 %。

**是否 <3 %**：按上表逐项回答，不做笼统断言。高端（c0 = 20 m/s）在 ±3 % 内；
低马赫点（c0 = 5 m/s）受 Tait 压缩量（需要 ρ/ρ0−1 ≈ 9.0e-3 才撑住 σ/R）与求和密度噪声
之比限制，是否 <3 % 以表为准。

## 4. 注意点 / 局限

1. **默认值仍未翻转**。`surface_pressure_closure` 默认 `False`、`surface_pressure_monotone`
   默认 `False`：打开会改写所有已发布的数字（静水压、溃坝、液滴撞池），而那些算例尚未用
   新闭包重跑认证。本文只证明它在 Laplace 算例上的方向与幅度，不构成对其他算例的认证；
   翻转默认值应作为独立单，附带静水/溃坝重跑。
2. **精度是拿虚假流换的**（图 (c)）。`all` 单遍最安静（v_max/v_ref ≈ 0.02）但欠 20 %；
   `interior-2` 能进 ±15 % 带却把虚假流顶到 0.39（Adami 判据的 4 倍）；monotone 迭代在
   两者之间取到了正确的位置。任何"最优变体"的说法都必须同时给这两个数量。
3. **迭代次数是稳定性参数，不是收敛参数**（3.2）。passes 过大会把内层值搬过头，
   激起的虚假流与误差一起回升；生产值取 3，不得"顺手调大"。
4. **参考实现不含 CSF**：`rhs_all_pairs` 只有压力+粘性+重力，因此两实现的对照只在
   `surface_tension_coefficient=0` 时有意义（测试里显式关掉表面张力）。
5. **阈值 0.99 是几何判据**：在有重力压实的算例里，顶部与底部触发条件不对称，这是 Adami
   原式的已知粗糙处，本轮未改进。

## 5. 复现

```bash
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py \
    --free-surface-only --variants all interior-2 all-iter-3 all-iter-6
PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_doc_fs.py
PYTHONPATH=src python -m pytest tests/test_free_surface_pressure.py -q
```

工件：`benchmarks/laplace_pressure_floor/report.json`（`free_surface_runs`）、
`benchmarks/laplace_pressure_floor/free_surface_pressure.png`、
`tests/test_free_surface_pressure.py`。
