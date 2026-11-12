# CASE-HYDROSTATIC 静态初始态误差分析

**日期**: 2026-11-12  
**范围**: TensorSPH `src/tensorsph/case_hydrostatic.py`  
**发现**: 静态初始态（t=0）压强剖面误差 10-28%，远超 1% 门禁

## 1. 问题描述

`benchmarks/case_hydrostatic/run.py --config all --static` 产生的静态记录显示：

| dx [m] | N | max_band_error_pct | 是否 < 1% |
|---|---|---|---|
| 0.04 | 250 | 28.22% | 否 |
| 0.025 | 640 | 14.94% | 否 |
| 0.02 | 1000 | 10.22% | 否 |

误差**不随加密收敛**（28% → 15% → 10%），说明不是分辨率问题，而是系统性偏置。

## 2. 根因分析

### 2.1 求和密度截断

SPH 求和密度 `rho_i = sum_j m_j W_ij` 在以下两处被截断：

1. **自由面附近**: 核支持半径 `2h` 内的真空侧没有粒子，导致密度低估
2. **壁面附近**: 壁面粒子只贡献排斥力，不参与求和密度（除非显式加入）

`hydrostatic_density()` 从 Tait 状态方程反推密度：
```
rho(y) = rho0 * (1 + rho0 * g * (H - y) / B)^(1/gamma)
```
其中 `B = rho0 * c0^2 / gamma`。在 `c0 = 20 m/s, gamma = 7` 下，`B ≈ 57143 Pa`。

底部压强 `p = rho0 * g * H = 1000 * 9.81 * 0.4 = 3924 Pa`，对应的密度比为：
```
rho/rho0 = (1 + 3924/57143)^(1/7) = 1.0094
```
即底部密度只比 `rho0` 高 0.94%。

### 2.2 误差来源分解

静态误差 = 求和密度截断误差 + 晶格离散误差 + EOS 非线性误差

在 `h/dx = 1.3` 下，正方形晶格的核和偏置为 `+5.88e-3`（见 `lattice_kernel_sum`），
即求和密度比 `rho0` 高 0.588%。这个偏置在底部（高压区）和顶部（低压区）是相同的，
但压强误差是相对的：

- 底部 `p_ref = 3924 Pa`，密度偏置 0.588% → 压强偏置约 0.588% * gamma ≈ 4.1%
- 顶部附近 `p_ref` 很小，同样的密度偏置导致更大的相对误差

这解释了为什么误差在顶部带最大（10-28%），而在底部较小。

### 2.3 为什么动态时间平均能改善

动态运行中，粒子会轻微移动，求和密度的截断误差被时间平均平滑。此外，
压力波动（fluctuation）在 8 秒窗口内被平均掉，留下更接近解析值的均值。

## 3. 与文献对照

Adami et al. (2012) JCP 231:7057 的静水壁面测试报告：
- 使用 WSPH 求和密度 + 壁面粒子贡献
- 压强误差 < 2%（在充分分辨率下）
- 关键区别：他们的壁面粒子**参与求和密度**

TensorSPH 当前的 `wall_lattice` 只用于边界排斥，不参与求和密度计算。
这是误差的主要来源。

## 4. 改进建议

### 4.1 壁面粒子参与求和密度（高优先级）

修改 `rhs()` 或求和密度计算，使壁面粒子贡献到流体粒子的求和密度中。
参考 Adami et al. (2012) 的壁面处理。

### 4.2 背景压强/无散修正（中优先级）

使用背景压强（background pressure）或无散修正（divergence-free correction）
来减少自由面附近的密度噪声。

### 4.3 静态判据调整（临时）

在壁面粒子参与求和密度之前，静态判据应放宽到 15%（与动态判据一致），
或仅使用动态时间平均作为门禁。

## 5. 当前状态

- `runs/case_hydrostatic/report.json` 已创建（仅含静态数据）
- 动态运行 `window_8s` 因超时未完成
- 测试 `test_shipped_gate_is_the_predeclared_config_and_within_limit` 因缺少动态数据而失败

## 6. 复现命令

```bash
# 静态记录（快）
PYTHONPATH=src python benchmarks/case_hydrostatic/run.py --config all --static

# 动态门禁运行（慢，~120s+）
PYTHONPATH=src python benchmarks/case_hydrostatic/run.py --config window_8s

# 聚合与出图
PYTHONPATH=src python benchmarks/case_hydrostatic/run.py --aggregate --figure

# 测试
PYTHONPATH=src python -m pytest tests/test_case_hydrostatic.py -v
```
