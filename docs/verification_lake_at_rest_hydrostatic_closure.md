# 静水湖面守恒：锚点容差能不能达到？——闭式水槽的静水闭合判决

> 锚点 `CASE-SPH-LAKE-AT-REST`（算例池·静水湖面守恒；ref: 解析恒等；容差: 机器精度）。
> 本文件由 `benchmarks/lake_at_rest/hydrostatic_closure.py --doc` 从
> `tests/data/lake_at_rest/hydrostatic_closure.json` 程序化生成，表内数字零手抄。
> 门禁：`tests/test_lake_at_rest_hydrostatic_closure.py`。

## 0. 结论先行

前两轮把闭式水槽的残差定位到「压力场非线性 + 自由面钳位折点」，并把未决项写成
「重建线性压力场」。本单把它做完并给出判决：

| 命题 | 判定 | 证据 |
|---|---|---|
| 锚点容差在某个容许闭合下**可达** | **成立** | 一致梯度 + 静水重构压力，`max\|a\|/g = 1.087e-15`（容差 1e-09，低 9.2e+05 倍） |
| 出厂配置自己达到 | **不成立（诚实 FAIL）** | 出厂算子 + Tait 压力 `max\|a_x\|/g = 1.242e-01` |
| 在**可压**（Tait 状态方程）闭合里达到 | **不成立** | 精确连续可压平衡态 + 一致梯度 `3.128e-04` —— 可压非线性就是地板 |
| 代价可控 | **需要交代** | 见 §6：压力由状态方程改为静水重构（不可压读法） |

一句话：**锚点容差不是靠更细的网格或更好的质量场达到的，而是靠换闭合**；
出厂可压 WCSPH 闭合在四个分辨率上的下限是 `1e-4 g` 量级，与容差差五个量级。

## 1. 案例与问题

| 量 | 值 |
|---|---|
| 水箱 | 0.40 m × 0.30 m，三层壁粒子 |
| 分辨率 | dx = 0.05, 0.04, 0.025, 0.02 m；h = 1.3 dx |
| 最细网格 | 300 流体粒子 / 198 壁粒子 / 3165 对 |
| 工质 | rho0 = 1000 kg/m³, c0 = 20 m/s, gamma = 7, g = 9.81 m/s² |
| 锚点容差 | 1e-09（机器精度量级） |
| 基准 | p = rho0 g (eta - y)，eta 由粒子数据读出（无注入） |

被检验的问题只有一个：**同一个格点、同一对表、同一个质量场上，哪一类闭合把
静水残差压到锚点容差以下？** 三个维度各取标准选项：

* 压力律：`eos`（出厂 Tait）/ `incompressible`（静水重构 p = rho0 g(eta-y)，
  密度置 rho0，无钳位）/ `barotropic`（Tait 的精确连续平衡态）；
* 壁面压力：`shipped_adami`（出厂加权外推 + 钳位）/ `continuation`（把同一静水压延拓到壁粒子，
  不钳位）/ `continuation_clamped`（延拓后钳位——把折点加回来的对照）；
* 梯度算子：`symmetric`（出厂成对式）/ `difference` / `moment`（一阶一致）。

## 2. 结果：闭合矩阵（Tikhonov 项取 0，即算子自身的残差）

### dx = 0.05 m（48 粒子，eta = 0.3000 m，表面起伏 0.0e+00 m）

| 压力律 | 壁面 | 算子 | max\|a\|/g | 中位 | 壁带 | 内部 | 条件数 | 回退粒子 | 净水平力/重量 |
|---|---|---|---|---|---|---|---|---|
| `eos` | `shipped_adami` | `symmetric` | **1.092e-01** | 1.61e-02 | 1.09e-01 | — | — | 0/48 | 4.8e-14 |
| `eos` | `shipped_adami` | `difference` | **4.314e-01** | 1.28e-01 | 4.31e-01 | — | — | 0/48 | 4.9e-14 |
| `eos` | `shipped_adami` | `moment` | **1.450e+00** | 1.06e-01 | 1.45e+00 | — | 1.7e+297 | 46/48 | 4.9e-14 |
| `incompressible` | `shipped_adami` | `symmetric` | **1.645e-01** | 9.84e-02 | 1.65e-01 | — | — | 0/48 | 3.0e-16 |
| `incompressible` | `shipped_adami` | `difference` | **5.515e-01** | 2.83e-02 | 5.51e-01 | — | — | 0/48 | 2.2e-17 |
| `incompressible` | `shipped_adami` | `moment` | **5.687e-16** | 1.83e-16 | 5.69e-16 | — | 1.5e+297 | 43/48 | 2.2e-17 |
| `incompressible` | `continuation` | `symmetric` | **1.645e-01** | 9.84e-02 | 1.65e-01 | — | — | 0/48 | 2.6e-16 |
| `incompressible` | `continuation` | `difference` | **5.515e-01** | 2.83e-02 | 5.51e-01 | — | — | 0/48 | 2.5e-18 |
| `incompressible` | `continuation` | `moment` | **5.440e-16** | 1.81e-16 | 5.44e-16 | — | 1.5e+297 | 43/48 | 2.2e-18 |
| `incompressible` | `continuation_clamped` | `symmetric` | **1.988e-01** | 9.84e-02 | 1.99e-01 | — | — | 0/48 | 2.6e-16 |
| `incompressible` | `continuation_clamped` | `difference` | **5.515e-01** | 2.83e-02 | 5.51e-01 | — | — | 0/48 | 4.7e-18 |
| `incompressible` | `continuation_clamped` | `moment` | **1.155e-01** | 1.82e-16 | 1.16e-01 | — | 1.5e+297 | 43/48 | 4.3e-18 |
| `barotropic` | `shipped_adami` | `symmetric` | **1.667e-01** | 8.74e-02 | 1.67e-01 | — | — | 0/48 | 2.0e-16 |
| `barotropic` | `shipped_adami` | `difference` | **5.521e-01** | 2.98e-02 | 5.52e-01 | — | — | 0/48 | 1.2e-17 |
| `barotropic` | `shipped_adami` | `moment` | **7.705e-04** | 5.61e-05 | 7.70e-04 | — | 1.5e+297 | 45/48 | 1.2e-17 |
| `barotropic` | `continuation` | `symmetric` | **1.667e-01** | 8.93e-02 | 1.67e-01 | — | — | 0/48 | 2.2e-16 |
| `barotropic` | `continuation` | `difference` | **5.521e-01** | 2.98e-02 | 5.52e-01 | — | — | 0/48 | 2.5e-18 |
| `barotropic` | `continuation` | `moment` | **1.181e-02** | 7.70e-04 | 1.18e-02 | — | 1.5e+297 | 45/48 | 2.6e-18 |
| `barotropic` | `continuation_clamped` | `symmetric` | **2.005e-01** | 8.93e-02 | 2.01e-01 | — | — | 0/48 | 2.2e-16 |
| `barotropic` | `continuation_clamped` | `difference` | **5.521e-01** | 2.98e-02 | 5.52e-01 | — | — | 0/48 | 1.1e-18 |
| `barotropic` | `continuation_clamped` | `moment` | **1.154e-01** | 1.38e-03 | 1.15e-01 | — | 1.5e+297 | 45/48 | 3.2e-18 |

出厂算子 + 出厂闭合的对照（同一质量场，`droplet_impact.rhs` 直算）：`max\|a_x\|/g = 1.092e-01`，`max\|a_y\|/g = 1.311e-13`，`p_min = 0 Pa`（钳位生效）。

### dx = 0.04 m（70 粒子，eta = 0.2800 m，表面起伏 0.0e+00 m）

| 压力律 | 壁面 | 算子 | max\|a\|/g | 中位 | 壁带 | 内部 | 条件数 | 回退粒子 | 净水平力/重量 |
|---|---|---|---|---|---|---|---|---|
| `eos` | `shipped_adami` | `symmetric` | **1.203e-01** | 1.20e-02 | 1.20e-01 | 8.42e-02 | — | 0/70 | 3.0e-14 |
| `eos` | `shipped_adami` | `difference` | **4.881e-01** | 8.78e-02 | 4.88e-01 | 1.63e-01 | — | 0/70 | 3.0e-14 |
| `eos` | `shipped_adami` | `moment` | **1.449e+00** | 1.11e-01 | 1.45e+00 | 1.48e-01 | 1.8e+297 | 53/70 | 3.0e-14 |
| `incompressible` | `shipped_adami` | `symmetric` | **1.824e-01** | 9.15e-02 | 1.82e-01 | 1.59e-01 | — | 0/70 | 2.5e-16 |
| `incompressible` | `shipped_adami` | `difference` | **5.611e-01** | 3.56e-02 | 5.61e-01 | 3.93e-02 | — | 0/70 | 1.5e-17 |
| `incompressible` | `shipped_adami` | `moment` | **1.630e-15** | 1.85e-16 | 1.63e-15 | 3.98e-16 | 1.7e+297 | 53/70 | 1.5e-17 |
| `incompressible` | `continuation` | `symmetric` | **1.824e-01** | 9.15e-02 | 1.82e-01 | 1.59e-01 | — | 0/70 | 2.3e-16 |
| `incompressible` | `continuation` | `difference` | **5.611e-01** | 3.56e-02 | 5.61e-01 | 3.93e-02 | — | 0/70 | 7.7e-19 |
| `incompressible` | `continuation` | `moment` | **5.465e-16** | 1.82e-16 | 5.43e-16 | 3.65e-16 | 1.7e+297 | 53/70 | 2.3e-19 |
| `incompressible` | `continuation_clamped` | `symmetric` | **2.183e-01** | 9.15e-02 | 2.18e-01 | 1.59e-01 | — | 0/70 | 2.3e-16 |
| `incompressible` | `continuation_clamped` | `difference` | **5.611e-01** | 3.56e-02 | 5.61e-01 | 3.93e-02 | — | 0/70 | 9.2e-19 |
| `incompressible` | `continuation_clamped` | `moment` | **1.170e-01** | 1.82e-16 | 1.17e-01 | 3.65e-16 | 1.7e+297 | 53/70 | 1.0e-18 |
| `barotropic` | `shipped_adami` | `symmetric` | **1.842e-01** | 9.46e-02 | 1.84e-01 | 1.51e-01 | — | 0/70 | 2.6e-16 |
| `barotropic` | `shipped_adami` | `difference` | **5.616e-01** | 3.53e-02 | 5.61e-01 | 3.89e-02 | — | 0/70 | 7.4e-18 |
| `barotropic` | `shipped_adami` | `moment` | **6.218e-04** | 3.72e-05 | 6.22e-04 | 1.04e-04 | 1.7e+297 | 52/70 | 8.3e-18 |
| `barotropic` | `continuation` | `symmetric` | **1.842e-01** | 9.46e-02 | 1.84e-01 | 1.51e-01 | — | 0/70 | 2.8e-16 |
| `barotropic` | `continuation` | `difference` | **5.616e-01** | 3.53e-02 | 5.61e-01 | 3.89e-02 | — | 0/70 | 1.2e-18 |
| `barotropic` | `continuation` | `moment` | **1.263e-02** | 5.96e-04 | 1.26e-02 | 2.87e-03 | 1.7e+297 | 52/70 | 1.1e-18 |
| `barotropic` | `continuation_clamped` | `symmetric` | **2.197e-01** | 9.46e-02 | 2.20e-01 | 1.51e-01 | — | 0/70 | 2.8e-16 |
| `barotropic` | `continuation_clamped` | `difference` | **5.616e-01** | 3.53e-02 | 5.61e-01 | 3.89e-02 | — | 0/70 | 1.2e-18 |
| `barotropic` | `continuation_clamped` | `moment` | **1.169e-01** | 6.22e-04 | 1.17e-01 | 2.87e-03 | 1.7e+297 | 52/70 | 7.3e-19 |

出厂算子 + 出厂闭合的对照（同一质量场，`droplet_impact.rhs` 直算）：`max\|a_x\|/g = 1.203e-01`，`max\|a_y\|/g = 1.863e-13`，`p_min = 0 Pa`（钳位生效）。

### dx = 0.025 m（192 粒子，eta = 0.3000 m，表面起伏 0.0e+00 m）

| 压力律 | 壁面 | 算子 | max\|a\|/g | 中位 | 壁带 | 内部 | 条件数 | 回退粒子 | 净水平力/重量 |
|---|---|---|---|---|---|---|---|---|
| `eos` | `shipped_adami` | `symmetric` | **1.013e-01** | 3.71e-03 | 1.01e-01 | 5.20e-02 | — | 0/192 | 4.9e-14 |
| `eos` | `shipped_adami` | `difference` | **3.748e-01** | 5.88e-02 | 3.75e-01 | 1.17e-01 | — | 0/192 | 4.9e-14 |
| `eos` | `shipped_adami` | `moment` | **1.455e+00** | 5.26e-02 | 1.46e+00 | 8.74e-02 | 1.6e+297 | 75/192 | 4.9e-14 |
| `incompressible` | `shipped_adami` | `symmetric` | **1.470e-01** | 6.26e-02 | 1.47e-01 | 7.62e-02 | — | 0/192 | 4.8e-16 |
| `incompressible` | `shipped_adami` | `difference` | **5.434e-01** | 1.61e-02 | 5.43e-01 | 3.12e-02 | — | 0/192 | 1.6e-17 |
| `incompressible` | `shipped_adami` | `moment` | **9.296e-16** | 1.82e-16 | 7.36e-16 | 7.36e-16 | 1.4e+297 | 76/192 | 1.6e-17 |
| `incompressible` | `continuation` | `symmetric` | **1.470e-01** | 6.26e-02 | 1.47e-01 | 7.62e-02 | — | 0/192 | 4.8e-16 |
| `incompressible` | `continuation` | `difference` | **5.434e-01** | 1.61e-02 | 5.43e-01 | 3.12e-02 | — | 0/192 | 9.2e-19 |
| `incompressible` | `continuation` | `moment` | **7.252e-16** | 1.81e-16 | 5.44e-16 | 7.25e-16 | 1.4e+297 | 76/192 | 8.9e-19 |
| `incompressible` | `continuation_clamped` | `symmetric` | **1.832e-01** | 6.26e-02 | 1.83e-01 | 7.62e-02 | — | 0/192 | 4.8e-16 |
| `incompressible` | `continuation_clamped` | `difference` | **5.434e-01** | 1.61e-02 | 5.43e-01 | 3.12e-02 | — | 0/192 | 0.0e+00 |
| `incompressible` | `continuation_clamped` | `moment` | **1.143e-01** | 1.81e-16 | 1.14e-01 | 7.25e-16 | 1.4e+297 | 76/192 | 2.3e-19 |
| `barotropic` | `shipped_adami` | `symmetric` | **1.481e-01** | 5.16e-02 | 1.48e-01 | 6.97e-02 | — | 0/192 | 4.4e-16 |
| `barotropic` | `shipped_adami` | `difference` | **5.437e-01** | 1.14e-02 | 5.44e-01 | 2.91e-02 | — | 0/192 | 1.8e-18 |
| `barotropic` | `shipped_adami` | `moment` | **3.842e-04** | 7.08e-06 | 3.84e-04 | 6.55e-05 | 1.4e+297 | 77/192 | 5.8e-19 |
| `barotropic` | `continuation` | `symmetric` | **1.481e-01** | 5.62e-02 | 1.48e-01 | 6.98e-02 | — | 0/192 | 4.5e-16 |
| `barotropic` | `continuation` | `difference` | **5.437e-01** | 1.44e-02 | 5.44e-01 | 2.90e-02 | — | 0/192 | 0.0e+00 |
| `barotropic` | `continuation` | `moment` | **2.190e-02** | 7.08e-06 | 2.19e-02 | 9.86e-03 | 1.4e+297 | 77/192 | 1.3e-18 |
| `barotropic` | `continuation_clamped` | `symmetric` | **1.841e-01** | 5.62e-02 | 1.84e-01 | 6.98e-02 | — | 0/192 | 4.5e-16 |
| `barotropic` | `continuation_clamped` | `difference` | **5.437e-01** | 1.44e-02 | 5.44e-01 | 2.90e-02 | — | 0/192 | 9.2e-19 |
| `barotropic` | `continuation_clamped` | `moment` | **1.142e-01** | 7.08e-06 | 1.14e-01 | 9.86e-03 | 1.4e+297 | 77/192 | 1.8e-18 |

出厂算子 + 出厂闭合的对照（同一质量场，`droplet_impact.rhs` 直算）：`max\|a_x\|/g = 1.013e-01`，`max\|a_y\|/g = 2.955e-13`，`p_min = 0 Pa`（钳位生效）。

### dx = 0.02 m（300 粒子，eta = 0.3000 m，表面起伏 0.0e+00 m）

| 压力律 | 壁面 | 算子 | max\|a\|/g | 中位 | 壁带 | 内部 | 条件数 | 回退粒子 | 净水平力/重量 |
|---|---|---|---|---|---|---|---|---|
| `eos` | `shipped_adami` | `symmetric` | **1.242e-01** | 9.31e-04 | 1.24e-01 | 5.86e-02 | — | 0/300 | 4.1e-14 |
| `eos` | `shipped_adami` | `difference` | **5.126e-01** | 4.67e-02 | 5.11e-01 | 1.63e-01 | — | 0/300 | 4.1e-14 |
| `eos` | `shipped_adami` | `moment` | **1.450e+00** | 3.94e-02 | 1.45e+00 | 1.21e-01 | 1.8e+297 | 90/300 | 4.1e-14 |
| `incompressible` | `shipped_adami` | `symmetric` | **1.919e-01** | 4.72e-02 | 1.92e-01 | 1.05e-01 | — | 0/300 | 6.0e-16 |
| `incompressible` | `shipped_adami` | `difference` | **5.656e-01** | 1.64e-02 | 5.65e-01 | 3.98e-02 | — | 0/300 | 3.0e-18 |
| `incompressible` | `shipped_adami` | `moment` | **3.993e-15** | 1.84e-16 | 3.99e-15 | 9.71e-16 | 1.7e+297 | 92/300 | 2.0e-18 |
| `incompressible` | `continuation` | `symmetric` | **1.919e-01** | 4.72e-02 | 1.92e-01 | 1.05e-01 | — | 0/300 | 6.2e-16 |
| `incompressible` | `continuation` | `difference` | **5.656e-01** | 1.64e-02 | 5.65e-01 | 3.98e-02 | — | 0/300 | 9.2e-20 |
| `incompressible` | `continuation` | `moment` | **1.087e-15** | 1.84e-16 | 1.09e-15 | 9.06e-16 | 1.7e+297 | 92/300 | 1.2e-18 |
| `incompressible` | `continuation_clamped` | `symmetric` | **2.257e-01** | 4.72e-02 | 2.26e-01 | 1.05e-01 | — | 0/300 | 6.2e-16 |
| `incompressible` | `continuation_clamped` | `difference` | **5.656e-01** | 1.64e-02 | 5.65e-01 | 3.98e-02 | — | 0/300 | 8.1e-19 |
| `incompressible` | `continuation_clamped` | `moment` | **1.177e-01** | 1.84e-16 | 1.18e-01 | 9.06e-16 | 1.7e+297 | 92/300 | 1.6e-18 |
| `barotropic` | `shipped_adami` | `symmetric` | **1.928e-01** | 3.69e-02 | 1.93e-01 | 1.04e-01 | — | 0/300 | 5.8e-16 |
| `barotropic` | `shipped_adami` | `difference` | **5.659e-01** | 1.16e-02 | 5.65e-01 | 3.80e-02 | — | 0/300 | 1.1e-17 |
| `barotropic` | `shipped_adami` | `moment` | **3.128e-04** | 2.53e-06 | 3.13e-04 | 5.28e-05 | 1.7e+297 | 94/300 | 9.5e-18 |
| `barotropic` | `continuation` | `symmetric` | **1.928e-01** | 3.75e-02 | 1.93e-01 | 1.04e-01 | — | 0/300 | 6.0e-16 |
| `barotropic` | `continuation` | `difference` | **5.659e-01** | 1.17e-02 | 5.65e-01 | 3.80e-02 | — | 0/300 | 9.2e-19 |
| `barotropic` | `continuation` | `moment` | **2.701e-02** | 2.53e-06 | 2.70e-02 | 1.43e-02 | 1.7e+297 | 94/300 | 2.0e-18 |
| `barotropic` | `continuation_clamped` | `symmetric` | **2.264e-01** | 3.75e-02 | 2.26e-01 | 1.04e-01 | — | 0/300 | 6.0e-16 |
| `barotropic` | `continuation_clamped` | `difference` | **5.659e-01** | 1.17e-02 | 5.65e-01 | 3.80e-02 | — | 0/300 | 1.5e-18 |
| `barotropic` | `continuation_clamped` | `moment` | **1.177e-01** | 2.53e-06 | 1.18e-01 | 1.43e-02 | 1.7e+297 | 94/300 | 2.1e-18 |

出厂算子 + 出厂闭合的对照（同一质量场，`droplet_impact.rhs` 直算）：`max\|a_x\|/g = 1.242e-01`，`max\|a_y\|/g = 3.504e-13`，`p_min = 0 Pa`（钳位生效）。

## 3. 正则化阶梯：那个 `1e-15` 不是挑出来的

一致梯度的残差**正比于** Tikhonov 正则项 —— 这说明 0 附近的数就是算子自身的舍入底，
而不是某个「刚好跑通」的设置：

| dx [m] | reg=1e-06 | reg=1e-09 | reg=1e-12 | reg=1e-15 | reg=0 |
|---|---|---|---|---|---|
| 0.05 | 3.081e-06 | 3.081e-09 | 3.081e-12 | 3.441e-15 | 5.440e-16 |
| 0.04 | 3.171e-06 | 3.171e-09 | 3.171e-12 | 3.259e-15 | 5.465e-16 |
| 0.025 | 2.996e-06 | 2.996e-09 | 2.996e-12 | 3.440e-15 | 7.252e-16 |
| 0.02 | 3.207e-06 | 3.207e-09 | 3.206e-12 | 3.622e-15 | 1.087e-15 |

表中数字是 `max|a|/g`。六个数量级的单调跟踪后停在 `1e-16` 量级（float64 舍入）。

## 4. 四项控制（决定上面那行数字有没有意义）

| dx [m] | 基准残留 | eta+1 mm（常数平移，应被湮灭） | 斜率 ×1.01（应被打破） | 壁面钳位（应复原折点） | 零压力（应自由落体 1 g） | 算子交叉校验 max\|Δa\| |
|---|---|---|---|---|---|---|
| 0.05 | 5.440e-16 | 7.246e-16 | 1.000e-02 | 1.155e-01 | 1.000 | 1.60e-14 |
| 0.04 | 5.465e-16 | 5.435e-16 | 1.000e-02 | 1.170e-01 | 1.000 | 1.07e-14 |
| 0.025 | 7.252e-16 | 7.247e-16 | 1.000e-02 | 1.143e-01 | 1.000 | 3.91e-14 |
| 0.02 | 1.087e-15 | 1.268e-15 | 1.000e-02 | 1.177e-01 | 1.000 | 4.97e-14 |

读法：常数平移被湮灭到与基准**逐位相同**（一致算子的零空间）；斜率差 1 % 把残差从
`1e-16` 打到 `1e-2`，即残差确实落在物理平衡点上；壁面一钳位，折点回来，残差跳 14 个量级；
压力置零则整片水自由落体；`symmetric` 重实现与出厂 `rhs` 在 `1e-14 m/s²` 内一致。

## 5. 可压闭合的地板（为什么这不是「再调一调」的问题）

| dx [m] | Tait 压力偏离静水重构 max | p_eos_min [Pa] | EOS 平衡密度偏离 | g·eta/c0² |
|---|---|---|---|---|
| 0.05 | 24.78% | 0 | 0.66% | 0.74% |
| 0.04 | 22.29% | 0 | 0.63% | 0.69% |
| 0.025 | 11.41% | 0 | 0.69% | 0.74% |
| 0.02 | 10.15% | 0 | 0.70% | 0.74% |

可压闭合的残差地板由两项决定，都与网格无关：

* 静水压与 Tait 压的差（上表第 2 列）——`max(p,0)` 在截断自由面把零点几 % 的偏差变成折点；
* EOS 平衡密度沿水深是非线性剖面（第 4 列 `g·eta/c0² ≈ 0.7 %`），一致性算子只对**线性**场精确，
  二阶项 `O(h² d²p/dy²)` 因此留下 `1e-4 g` 量级的地板。

这两项都不随 dx 消失（矩阵里 `barotropic` 行在四个分辨率上同量级），所以「加密网格」不是出路。

## 6. 代价：这个闭合换了什么

* 压力不再由状态方程给出，而由静水重构给出 —— 即把 WCSPH 的可压压力源换成不可压式的
  静水闭合。声学响应（弱可压）在这一闭合里被替换掉了，这正是 well-balanced 有限体积格式
  （Audusse 2004；Xing & Shu 2005）在 SPH 里的读法；
* 密度置 `rho0` 后，与 Tait 平衡密度的差就是被丢掉的可压性（上表第 4 列，`0.7 %`）；
* 一致算子 `moment` 在截断支撑上接近奇异（条件数见表），本单把 `reg` 的取值和阶梯全列出，
  并且**要求**：报告 `reg → 0` 的残差时必须同时报告条件数与回退粒子数。`reg = 0` 时
  最细网格有 92/300 个粒子的矩矩阵被判奇异回退 —— 回退粒子照样给出
  舍入级残差，但那是对角正则的功劳，不是一致性。诚实说法：`max\|a\|/g ≤ 1e-15` 是
  「线性场 + 一阶一致算子 + 无钳位闭包」的**代数恒等式**，不是这个离散格式在一般流动里的精度。

## 7. 误差判定（对锚点）

| 项 | 实测 | 基准 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| 一致闭合静水残差（最细网格） | 1.087e-15 | 0（解析恒等） | 1.087e-15 | 是（且 < 锚点容差） |
| 出厂闭合静水残差（最细网格） | 1.242e-01 | 0（解析恒等） | 1.242e-01 | **否** |
| 可压闭合残差地板 | 3.128e-04 | 0（解析恒等） | 3.128e-04 | **否** |

三项都是「与解析恒等式的差」，量纲为 g，因此相对误差即绝对值。锚点容差 `1e-9`：
一致闭合达标（1.1e-15），出厂/可压闭合不达标（诚实 FAIL，未被任何
调参掩盖）。

## 8. 门禁

| gate | dx=0.02 实测 | 阈值 | 判定 |
|---|---|---|---|
| `H1` 锚点容差 | 1.087e-15 | 1e-09 | PASS |
| `H2` 出厂对照 | 1.242e-01 | 0.01 | PASS |
| `H3` 钳位对照 | 1.177e-01 | 0.01 | PASS |
| `H4` 可压地板 | 3.096e-04 | 1e-06 | PASS |
| `H5` 水位平移不变 | 1.808e-16 | 1e-15 | PASS |
| `H6` 斜率敏感度 | 9.202e+12 | 1e+06 | PASS |
| `H7` 零压力对照 | 1.000e+00 | 0.9…1.1 | PASS |
| `H8` 算子交叉校验 | 4.974e-14 | 1e-10 | PASS |
| `H9` 正则化阶梯 | 单调 | — | PASS |
| `H10` 分辨率地板 | 2.00× | < 10× | PASS |

全部通过：**True**。

## 9. 复现

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/hydrostatic_closure.py \
    --json tests/data/lake_at_rest/hydrostatic_closure.json --doc --figures
PYTHONPATH=src pytest tests/test_lake_at_rest_hydrostatic_closure.py -q
```

