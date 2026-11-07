# 零外力静水的启动伪流：谁真的在动，动多少

**范围**: TensorSPH ｜ **性质**: 验证欠账（把 backlog 的建议变成可复算的判据）
｜ **数据**: `benchmarks/startup_quiescence/report.json`（本文件全部数字由
`benchmarks/startup_quiescence/make_docs.py` 从该 JSON 注入，无手抄）

## 1. 案例与物理背景

`backlog_update_m3211.md` P2.2 提出：把「零外力静止水池的自生流速」当作任何
质量单位／自由面／压强闭合改动的冒烟判据。该条只给了一个数（`h/dx = 1.5`、
未标定质量时 **0.096 m/s**），并把它归因于晶格质量偏置。本单回到这个归因本身，
把一次性数字做成可复算的基准，并区分**静态度量**与**动态度量**：

| 名词 | 定义 |
|---|---|
| 内区（interior） | 距任一壁面/自由面 ≥ 3h 的粒子（见 §5 为何不是 2h） |
| `initial_pressure_deviation` | `max_i |p_i - rho0 g (H - y_i)| / (rho0 g H)`，t=0，内区 |
| `transition_band_pressure_deviation` | 自由面下 2h–3h 壳层内的同一度量（单列，不并入判据） |
| `v_peak` / `interior_v_late_rms` | 全粒子 / 内区 `max_i |v_i|` 的峰值，以及末四分之一时段的 RMS |

算例（无任何外力、无入射波、活塞冻结在静位）：

* **pool**：`droplet_impact.make_particles`，自由面水池 0.60 m × 0.24 m，`dx = 20 mm`，
  `h/dx ∈ {1.3, 1.5}` × `density_calibration ∈ {True, False}`，模拟 0.10 s。
* **flume**：`wave_flume.flume_water_lattice` 静止水柱 0.60 m × 0.20 m，`dx = 10 mm`，
  `h/dx = 1.3`，地板+下游壁+冻结活塞，模拟 0.06 s；再做粘度/密度扩散消融。

## 2. 怎么做的

1. `benchmarks/startup_quiescence/run.py` 直接调用生产 RHS（`droplet_impact.rhs`）与
   生成的初始状态构造函数，不复制求解器代码；每次采样记录全粒子与内区的 `max|v|`。
2. t=0 先用 `rhs` 取一次场量，得到初始压强相对解析静水压的偏差与初始加速度。
3. 判据分两层：**契约判据**（`gates`，必须全绿）与**未决发现**（`findings`，登记缺陷）。
   一个刚暴露出来的缺陷不应该被塞进契约判据里让基准永久变红，也不该被删掉。

## 3. 计算结果

### 3.1 池（静态度量随质量标定而变，动态度量几乎不变）

| h/dx | density_calibration | 内区粒子 | 初始压强偏差 | 2h–3h 壳层 | a0 max (m/s²) | a0 内区 (m/s²) | v_peak (m/s) | 内区 v_late_rms (m/s) |
|---|---|---|---|---|---|---|---|---|
| 1.3 | True | 88 | 1.384e-04 | 0.1250 | 16.45 | 0.09 | 0.0805 | 0.0208 |
| 1.3 | False | 88 | 0.0094 | 0.1250 | 16.45 | 0.08 | 0.0818 | 0.0210 |
| 1.5 | True | 60 | 1.715e-04 | 0.1250 | 16.46 | 0.01 | 0.1235 | 0.0375 |
| 1.5 | False | 60 | 0.6037 | 0.5956 | 33.69 | 0.22 | 0.1424 | 0.0458 |

`g = 9.81 m/s²`，`sqrt(gH) = 1.534 m/s`；`a0` 峰值出现在池底
第 2 行（`y ≈ 0.030 m`），**不在自由面**：静止水池的启动瞬态由壁面压力外推/kernel 补偿驱动，
不是自由面。

### 3.2 波槽静止水柱（消融）

| 配置 | 初始压强偏差 | v_peak (m/s) | 内区 v_late_rms (m/s) | Mach (`v/c0`) |
|---|---|---|---|---|
| shipped | 8.733e-04 | 0.1431 | 0.0798 | 0.0027 |
| shipped, overrides viscosity=0.01 | 8.733e-04 | 0.0608 | 0.0079 | 2.618e-04 |
| shipped, overrides density_diffusion=0.0 | 9.280e-04 | 0.0880 | 0.0472 | 0.0016 |
| shipped, overrides density_diffusion=0.0,viscosity=0.01 | 9.280e-04 | 0.0513 | 0.0072 | 2.387e-04 |

| 消融（波槽静止水柱） | v_peak (m/s) | 内区 v_late_rms (m/s) |
|---|---|---|
| shipped | 0.1431 | 0.0798 |
| shipped, overrides viscosity=0.01 | 0.0608 | 0.0079 |
| shipped, overrides density_diffusion=0.0 | 0.0880 | 0.0472 |
| shipped, overrides density_diffusion=0.0,viscosity=0.01 | 0.0513 | 0.0072 |

波槽信号参照（Airy，一阶）：深 `0.2` m、波高 `0.03` m、
周期 `0.8` s → `k = 7.076 1/m`、
`omega = 7.854 rad/s`、自由面轨道速度
`u_orbital = 0.133 m/s`。

### 3.3 判据与未决发现

| 判据 | 读取 | 实测 | 限值（方向） | 结果 |
|---|---|---|---|---|
| `G1_initial_state_quality` | pool h/dx=1.5, density_calibration=True | 1.715e-04 | 0.0010 (upper) | pass |
| `G2_sensitivity` | pool h/dx=1.5 uncalibrated / calibrated | 3519.1616 | 10.0000 (lower) | pass |
| `G3_noise_floor` | pool h/dx=1.5, density_calibration=True | 0.0375 | 0.0600 (upper) | pass |
| `G4_flume_initial_state` | flume resting column (paddle frozen at rest) | 8.733e-04 | 0.0010 (upper) | pass |

| 发现 | 状态 | 读取 | 实测 | 限值 | 实测/u_orbital |
|---|---|---|---|---|---|
| `F1_flume_resting_noise` | OPEN | flume resting column (paddle frozen at rest) | 0.0798 | 0.0600 | 0.60 |

## 4. 结果图

![startup velocity series](benchmarks/startup_quiescence/figures/startup_v_series.png)

左：自由面水池（无外力）的 `max|v|` 时程——标定与否的差异在动力学里几乎看不见；
右：波槽静止水柱的消融——虚线是 `u_orbital`。

![initial state quality](benchmarks/startup_quiescence/figures/initial_state_quality.png)

静态度量：`density_calibration` 把初始压强偏差从 `6.0e-1`（`h/dx=1.5` 未标定）
压到 `1.7e-4`，横幅是 1e-3 判据线。

## 5. 注意点（踩过的坑与适用范围）

1. **2h 不够。** 最初按 kernel 支撑 `2h` 划内区，波槽在自由面下方 2h–3h 的壳层里
   读到 `3.9e-2` 的压强偏差，直接把 `h/dx=1.3` 的池（`1.4e-4`）与波槽（`3.9e-2`）
   拉成两个量级。改成 3h 后波槽降到 `8.7e-4`，壳层单列成
   `transition_band_pressure_deviation`。**稳态压强的内区判据要用 3h。**
2. **静态度量和动态度量回答不同的问题。** 质量标定（m3211）确实把初始压强偏差
   压到 `1e-4` 级（G1/G2，灵敏度比 `3519`），但它对启动
   瞬态的峰值几乎没影响（`h/dx=1.3`：`0.0805` vs 未标定
   `0.0818` m/s）。
   所以 m3211 P2.2 的「自生流速」不能作为质量单位的判据，只能作为**相对回归探针**；
   它量的主要是壁面/自由面的启动瞬态。
3. **静水不是"应该完全不动"。** 弱可压缩 SPH 的静水本身有 O(Mach) 的声学振铃，
   本基准量到的内区动态底噪（`h/dx=1.3`）为 `0.0208` m/s，
   即 `sqrt(gH)` 的 `1.4 %`。
   任何要解析更小速度的工况都必须先把这个底噪压下去（加数值粘度/更长的沉降段），
   或用相对判据（相对同一配置的基线）而不是绝对门限。
4. **波槽的 F1 是这类问题的实例**：默认 `viscosity = 1e-6`（水）下，静止水柱的噪声在
   0.2 s 窗口内不衰减（`benchmarks/startup_quiescence` 的消融：加 `viscosity = 0.01`
   落到 `0.0079` m/s）。这是"是否该给波槽加数值粘度"的工程决策，不是本基准该自动改的东西。
5. 适用范围：2-D、单相、`h/dx ∈ [1.3, 1.5]`、`dx ∈ [10, 20] mm`、`c0 ∈ [20, 30] m/s`；
   3-D 与多相未测。

## 6. 误差判定

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 内区初始压强（池 h/dx=1.3，标定） | `rho0 g (H-y)` | `1.384e-04`·`rho0 g H` | `0.014 %` | **是** |
| 内区初始压强（池 h/dx=1.5，标定） | 同上 | `1.715e-04`·`rho0 g H` | `0.017 %` | **是** |
| 内区初始压强（波槽，标定） | 同上 | `8.733e-04`·`rho0 g H` | `0.087 %` | **是** |
| 内区初始压强（未标定 h/dx=1.5） | 同上 | `0.6037`·`rho0 g H` | `60.4 %` | **否**（这正是缺陷本身） |
| 动态度量 | 无公认验收阈值 | — | — | **无基准**：不存在「多静才算静」的文献判据，故 `G3` 只是钉在今日实测值之上的回归包络，不是精度验收；不声明 <3% |

动态底噪最坏情况 `0.0375` m/s（`pool h/dx=1.5, density_calibration=True`），
其与 `u_orbital` 的比值为 §3.3 的 F1 行；该量级意味着 **H=3 cm 级波浪的静水基线噪声
不可忽略**，这也是 `docs/verification_wave_flume.md` 里仍列为未验证的原因之一。
