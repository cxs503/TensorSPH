# CASE-SPH-OSC-DROP-FREQ: 振荡液滴频率验证报告

## 1. 案例介绍

### 物理背景
振荡液滴是表面张力驱动流动的经典解析基准。Lord Rayleigh (1879) 推导了无粘液滴
小振幅振荡的精确频率公式，该公式仅依赖于表面张力系数 σ、密度 ρ、平衡半径 R 和
模态数 m，不涉及任何经验参数。

### 几何与参数
| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 表面张力系数 | σ | 0.072 | N/m |
| 密度 | ρ | 1000 | kg/m³ |
| 平衡半径 | R | 0.05 | m |
| 模态数 | m | 2 (四极矩) | — |
| 初始扰动幅值 | ε | 0.02, 0.05 | — |

### Rayleigh 解析解
对于 2-D 液滴（液柱），精确振荡频率为：

$$\omega_m^2 = \frac{\sigma \cdot m \cdot (m^2 - 1)}{\rho \cdot R^3}$$

代入上述参数（m=2）：
- ω₂ = √(6σ/(ρR³)) = **1.859032 rad/s**
- T₂ = 2π/ω₂ = **3.379816 s**

### 验收判据
**容差 < 6%**：测量频率与解析频率的相对误差绝对值小于 6%。

---

## 2. 方法

### 数值方法
- **求解器**: TensorSPH WCSPH（弱可压缩光滑粒子流体动力学）
- **表面张力**: CSF（连续表面力）模型，Brackbill et al. (1992)
- **曲率估计**: 二次拟合（fit_quadratic）
- **状态方程**: Tait EOS，γ = 7
- **时间积分**: Kick-Drift-Kick Leapfrog，自适应时间步（CFL = 0.25）
- **密度扩散**: δ-SPH（Antuono et al. 2012），δ = 0.1

### 测量方法
模态振幅定义为质量加权的多极矩投影：

$$q_m(t) = \frac{\sum_i m_i r_i^2 \cos(m\theta_i)}{\sum_i m_i r_i^2}$$

对于形状 $r(\theta) = R(1 + \varepsilon \cos m\theta)$，$q_m$ 线性正比于扰动幅值 ε，
其过零点间距为 $T_m/2$，与阻尼包络无关。

### 收敛分析
- **分辨率阶梯**: R/dx = 12, 16, 24
- **通道**: A（标准 CSF）、A_differential（差分通道）、B（替代曲率方案）
- **模态**: m = 2, 3
- **幅值**: ε = 0.02, 0.05
- **分析方法**: 观测收敛阶 + GCI（网格收敛指数）+ ASME V&V 20 验证不确定度

### 工具链
- Python 3.11.6 + PyTorch 2.7.1
- 分析脚本: `benchmarks/oscillating_drop_freq/round7_convergence.py`
- 综合脚本: `benchmarks/oscillating_drop_freq/round8_synthesis.py`

---

## 3. 计算结果

### 解析参考值
| 模态 m | ω_exact (rad/s) | T_exact (s) |
|--------|-----------------|-------------|
| 2 | 1.859032 | 3.379816 |
| 3 | 3.718065 | 1.689908 |

### R/dx = 24 处各通道结果
| 通道 | 模态 | ε | Q(24) | ω 误差 (%) | 门限通过 |
|------|------|-----|-------|------------|----------|
| A | 2 | 0.02 | 0.9689 | -3.11 | ✅ |
| A | 2 | 0.05 | 0.9622 | -3.78 | ✅ |
| A | 3 | 0.02 | 0.9573 | -4.27 | ✅ |
| A | 3 | 0.05 | 0.9573 | -4.27 | ✅ |
| A_differential | 2 | 0.02 | 0.9582 | -4.18 | ✅ |
| A_differential | 2 | 0.05 | 0.9579 | -4.21 | ✅ |
| A_differential | 3 | 0.02 | 0.9572 | -4.28 | ✅ |
| A_differential | 3 | 0.05 | 0.9572 | -4.28 | ✅ |
| B | 2 | 0.02 | 0.9598 | -4.02 | ✅ |
| B | 3 | 0.02 | 0.9595 | -4.05 | ✅ |

**10/10 行通过 6% 门限。**

### 收敛阶分析
| 统计量 | 值 |
|--------|-----|
| 观测阶中位数 p | 1.93 |
| 观测阶范围 | [0.20, 4.00] |
| 一阶收敛假设（R4） | ❌ 被证伪 |
| 单调收敛行数 | 8/10 |

### ASME V&V 20 验证
| 统计量 | 值 |
|--------|-----|
| 验证通过行数 | 3/10 |
| 验证方法 | GCI + 估计器控制偏差 |
| l=3 系统偏差 | ≈ -2.5% |

---

## 4. 结果图

### 收敛图
见 `docs/figures/case_sph_osc_drop_freq_round7_convergence.png`

### 综合图
见 `docs/figures/case_sph_osc_drop_freq_round8_synthesis.png`

---

## 5. 注意点

### 已确认的事实
1. **6% 门限**: R/dx = 24 处所有 10 个通道行均通过
2. **收敛阶**: l=3 模态观测阶 p ≈ 1.93（非 R4 假设的 p=1）
3. **系统偏差**: l=3 模态存在约 -2.5% 的渐近偏差（V&V 20 未通过）
4. **差分通道**: l=2 差分通道（A'）通过 V&V 20，Q_inf ≈ 0.999
5. **一阶假设证伪**: R4 的一阶收敛假设被测量数据证伪

### 开放问题
1. **l=3 偏差来源**: 曲率估计？界面重构？需进一步研究
2. **l=2 非单调性**: 网格对齐效应？需方向系综测试
3. **动态臂不稳定**: 压力+连续性+轴源子系统不稳定（Round 5 记录为 FAIL）
4. **更高分辨率**: R/dx = 32/40 结果未从集群取回（cj4101）
5. **偏差随分辨率变化**: 未在 R/dx > 24 处测试

### 适用范围
- 本验证针对 **2-D 液滴（液柱）**，参考公式为 $\omega_m^2 = \sigma m(m^2-1)/(\rho R^3)$
- 3-D 液滴的 Rayleigh 公式为 $\omega_l^2 = \sigma l(l-1)(l+2)/(\rho R^3)$
- 动态臂（时间推进）当前不可用（发散），验证基于准静态线性响应测量

---

## 6. 误差判定

### 关键量 vs 基准
| 关键量 | 基准值 | 测量值（R/dx=24） | 相对误差 | 是否 < 6% |
|--------|--------|-------------------|----------|-----------|
| ω₂ (A, ε=0.02) | 1.859032 rad/s | 1.801 rad/s | -3.11% | ✅ |
| ω₂ (A, ε=0.05) | 1.859032 rad/s | 1.789 rad/s | -3.78% | ✅ |
| ω₃ (A, ε=0.02) | 3.718065 rad/s | 3.559 rad/s | -4.27% | ✅ |
| Q_inf (l=2, A') | 1.0 | 0.9990 | -0.10% | ✅ |
| Q_inf (l=3, A) | 1.0 | 0.9744 | -2.56% | ❌ (V&V 20) |

### 结论
- **6% 门限**: ✅ 所有 10 行通过
- **渐近验证 (V&V 20)**: ⚠️ 3/10 行通过（l=2 差分通道表现最佳）
- **动态臂**: ❌ 不可用（发散）

**总体 verdict: PARTIAL PASS** — 门限通过但渐近验证不完全，动态臂待修复。

---

## 参考文献

1. Rayleigh, Lord (1879). "On the Capillary Phenomena of Jets". *Proc. R. Soc. Lond.* **29**: 71–97.
2. Lamb, H. (1932). *Hydrodynamics*, 6th ed., Arts. 313–315.
3. Brackbill, J.U., Kothe, D.B., Zemach, C. (1992). "A continuum method for modeling surface tension". *J. Comput. Phys.* **100**: 335–354.
4. Antuono, M., Colagrossi, A., Marrone, S. (2012). "Numerical difficulties in the SPH simulation of free-surface flows". *Proc. MARINE 2011*.
5. Roache, P.J. (1994). "Perspective: A Method for Uniform Reporting of Grid Refinement Studies". *J. Fluids Eng.* **116**: 405–413.
6. ASME V&V 20-2009. "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer".
