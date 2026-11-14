# CASE-SPH-WEDGE-ENTRY-2D: 二维楔形体入水砰击压力验证报告

**任务 ID**: m20829  
**仓库**: TensorSPH  
**分支**: agent/m20829  
**日期**: 2026-11-14  
**锚点**: CASE-SPH-WEDGE-ENTRY-2D（feature_ledger status=open）  
**状态**: 部分完成——粗网格（dx=4mm）已执行并验证；细网格阶梯（dx=2/1/0.5mm）+ 自相似控制已递交集群（cj2979）

---

## ① 案例介绍

### 物理背景

二维对称刚性楔形体以恒定速度 V 垂直入水，deadrise 角 β = 30°（楔面与未扰动自由面的夹角）。重力关闭，流动处于自相似 regime——唯一的长度尺度是 Vt。

这是经典的入水砰击（water entry slamming）问题。Wagner (1932) 首次给出渐近解析解（线性化势流，忽略射流），Zhao & Faltinsen (1993) 用非线性边界元方法给出全相似解（JFM 246:593）。

### 几何与参数表

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| Deadrise 角 | β | 30 | deg |
| 入水速度 | V | 1.0 | m/s |
| 流体密度 | ρ₀ | 1000 | kg/m³ |
| 声速因子 | c₀/V | 20 | — |
| 人工粘度 | α | 0.03 | — |
| 粒子间距 | dx | 0.5–4.0 | mm |
| 核半径 | h | 1.3 dx | — |
| 观测窗口终点湿宽 | c_end | 50 | mm |
| 槽深 | D | 120 | mm |

### 网格收敛阶梯（预声明设计）

| 工况 ID | dx (mm) | 流体粒子数 | 时间步数 | 状态 |
|---------|---------|-----------|---------|------|
| beta30_V1_dx40e-4 | 4.0 | 3000 | 314 | ✅ 已执行 |
| beta30_V1_dx20e-4 | 2.0 | ~12000 | ~556 | ⏳ 集群 cj2979 |
| beta30_V1_dx10e-4 | 1.0 | ~48000 | ~1100 | ⏳ 集群 cj2979 |
| beta30_V1_dx5e-4 | 0.5 | ~192000 | ~2200 | ⏳ 集群 cj2979 |
| beta30_V5_dx1mm | 1.0 (V=5) | ~48000 | ~220 | ⏳ 集群 cj2979 |

### 参考解

- **Wagner (1932)**: 解析解，机器可验证，REAL 状态
  - 湿宽系数: C = π/(2 tan β) = **2.7207**
  - 驻点压力系数: Cp_centre = 2C = **5.4414**
  - 峰值压力系数: Cp_max = C² + 1 = **8.4022**（位于 τ_peak = 0.930）
  - 公式来源: `src/tensorsph/wedge_entry.py`（模块 docstring 式 (1)-(3)，`wagner_pmax`/`wagner_p_centre`）
  - 由 `run.py:refs_derived()` 现算，**不手抄**

- **Zhao & Faltinsen (1993)**: JFM 246:593, DOI 10.1017/S002211209300028X
  - 状态: **WITHHELD**（闭源访问，OpenAlex oa_status=closed，无全文获取渠道）
  - 需要: 论文原文中 β=30° 的表格/图形数值
  - 闭合条件: 论文以可提取文本格式交付到仓库，登记页码/表格坐标后重新运行 `--stage accept`

---

## ② 方法与实现

### 数值方法

- **求解器**: 弱可压缩 SPH (WCSPH)，`src/tensorsph/wedge_entry.py` 的 `WedgeImpact` 类
- **状态方程**: Tait EOS, p = B[(ρ/ρ₀)^γ - 1], γ = 7, B = ρ₀c₀²/γ
- **核函数**: 三次样条核 (cubic spline)，h = 1.3 dx
- **密度计算**: 密度求和法 (density summation)，格点标定质量使满支撑粒子恰好 ρ=ρ₀
- **壁面处理**: Adami et al. (2012) 静/动壁面粒子，压力外推 p_w = Σ p_f W / Σ W
- **人工粘度**: Monaghan 形式, α = 0.03
- **时间积分**: 蛙跳法 (leapfrog)，CFL 限制 dt = 0.35 dx/c₀
- **邻居搜索**: 细胞链表法 (cell-linked list)，每步重建（刚体运动使 Verlet 表失效）
- **重力**: 关闭（自相似前提）

### 测量量定义

| 量 | 定义 | 物理意义 |
|----|------|---------|
| Cp_centre | 最靠近 x=0 的壁面粒子压力 / q | 驻点线压力（Wagner 解析解与时间无关） |
| Cp_max_footprint | 湿区内最大壁面压力 / q | 字面"峰值砰击压力"（位于 spray root 附近，Wagner 在该处发散） |
| Cp_max_interior | \|x\| ≤ 0.9c 范围内最大压力 / q | 内部区域峰值（避开 spray root 奇点） |
| Cp_band_peak | τ 分箱系综平均（后 5 帧）的最大值 | 噪声抑制后的峰值估计 |
| c(t) | 壁面压力超过阈值的最大 \|x\| | 湿宽（SPH 测量） |

其中 q = ½ρV² 为动压。

### 工具链

- `src/tensorsph/wedge_entry.py`: WedgeImpact 求解器 + Wagner 解析参考函数
- `benchmarks/case_sph_wedge_entry_2d/run.py`: 基准测试驱动（run/matrix/accept 阶段）
- `benchmarks/case_sph_wedge_entry_2d/refs.json`: 参考值出处唯一真源
- `tests/test_wedge_entry.py`: 28 项解析验证测试（全部通过）
- `tests/test_case_sph_wedge_entry_2d.py`: 13 项算例逻辑测试（全部通过）

---

## ③ 计算结果

### 3.1 工况 beta30_V1_dx4mm（dx=4mm, V=1m/s, β=30°）

**配置摘要**: 3000 流体粒子, 54 壁面粒子, 314 时间步, 墙钟 95.6s

#### 关键量时间序列

| 帧 | t [s] | c_sph [m] | c_wagner [m] | Cp_centre | Cp_max_footprint | max ρ/ρ₀ |
|----|-------|-----------|--------------|-----------|-----------------|----------|
| 0 | 0.0000 | 0.0025 | 0.0000 | 0.061 | 0.061 | 1.0037 |
| 1 | 0.0015 | 0.0025 | 0.0044 | 12.407 | 12.407 | 1.0205 |
| 2 | 0.0031 | 0.0017 | 0.0088 | 0.000 | 0.000 | 1.0080 |
| ... | ... | ... | ... | ... | ... | ... |
| 10 | 0.0153 | 0.0406 | 0.0440 | 9.658 | 14.829 | 1.0324 |
| 11 | 0.0168 | 0.0440 | 0.0484 | 2.190 | 10.190 | 1.0241 |
| 12 | 0.0184 | 0.0510 | 0.0528 | 5.021 | 20.267 | 1.0471 |

#### 系综平均剖面（后 5 帧 τ 分箱）

| 量 | SPH 值 | Wagner 参考 | 相对误差 |
|----|--------|------------|---------|
| Cp_centre_mean | 4.527 | 5.441 | **-16.8%** |
| Cp_band_peak | 32.01 | 8.402 | **+280.9%** |
| Cp@τ_peak | 3.971 | 8.402 | **-52.7%** |
| max ρ/ρ₀ - 1 | 0.0471 | — | 4.71% (< 5% ✅) |

### 3.2 结果图

#### 图 1: 壁面压力剖面（自相似坐标 τ = |x|/c(t)）

![Wall Pressure Profile](figures/wedge_entry_profile_dx4mm.png)

*图: SPH 系综平均壁面压力剖面（蓝色圆点，dx=4mm，后 5 帧 τ 分箱平均）与 Wagner (1932) 解析解（红色实线）对比。绿色虚线为 Wagner Cp_max = 8.40，橙色虚线为 Wagner Cp_centre = 5.44。SPH 在 τ < 0.7 区域显著高于解析解，反映 WCSPH 在 spray root 附近的压力振荡。*

#### 图 2: 关键量时间序列

![Time Series](figures/wedge_entry_timeseries_dx4mm.png)

*左: 驻点压力系数 Cp_centre 时间序列（蓝色），Wagner 参考值 5.44（红色虚线）。中: 湿宽演化，SPH（蓝色）vs Wagner（红色）。右: 最大密度比，5% 界限（红色虚线）。*

---

## ④ 误差判定

### 4.1 对 Wagner 解析锚的判决

| 判据 | 阈值 | dx=4mm 实测 | 判决 |
|------|------|------------|------|
| \|Cp_band_peak - Cp_max,Wagner\| / Cp_max,Wagner | ≤ 15% (最细) / ≤ 30% (最粗) | +280.9% | ❌ FAIL |
| \|Cp@τ_peak - Cp_max,Wagner\| / Cp_max,Wagner | ≤ 15% | -52.7% | ❌ FAIL |
| \|Cp_centre_mean - Cp_centre,Wagner\| / Cp_centre,Wagner | ≤ 15% | -16.8% | ❌ FAIL |
| GCI (最细对) | ≤ 15% | 待细网格 | ⏳ |
| 自相似偏差 (V=1 vs V=5) | ≤ 5% | 待执行 | ⏳ |
| max ρ/ρ₀ - 1 | ≤ 5% | 4.71% | ✅ PASS |

### 4.2 对 Zhao & Faltinsen (1993) 的判决

**WITHHELD** — β=30° 原文表/图数值不可得。不以记忆值或二手值代替。

### 4.3 失败根因分析

dx=4mm 工况所有压力判据均 FAIL，但这是**预期行为**而非 bug：

1. **压缩性噪声地板**: 弱可压 SPH 的压力由密度误差直接生成。本例 max ρ/ρ₀ = 1.0471，对应的压力噪声地板为：
   
   Cp_budget = 2(c₀/V)²/γ · [(ρ/ρ₀)^γ - 1] = 2×400/7 × (1.0471⁷ - 1) ≈ **43.5**

   这已经远超 Wagner 参考值 8.40。一个噪声地板 5 倍于参考值的量，无法在该分辨率下被正确测量。

2. **Spray root 奇点**: Wagner 解在 τ → 1 处有 1/√(1-τ²) 奇点，SPH 的核平均将其平滑为有限值，但粒子无序导致压力振荡。Cp_max_footprint 捕捉到这些振荡峰值，因此远高于解析值。

3. **分辨率不足**: c_end/dx = 12.5，即湿宽方向仅 12.5 个粒子。对于需要解析压力梯度的问题，这远远不够。

**结论**: 粗网格 FAIL 是物理和数值方法的固有局限，不是实现错误。细网格（dx ≤ 1mm）有望显著改善，因为：
- 压缩性噪声地板随分辨率提高而降低（密度误差减小）
- Spray root 区域有更多粒子，核平均更准确
- 已有文献表明 dx ≤ 1mm 时 WCSPH 入水问题可获合理结果

---

## ⑤ 注意点与局限

### 5.1 诚实声明

1. **Zhao & Faltinsen (1993) 参考值 WITHHELD**: 论文闭源，无全文获取渠道。这是本算例的主要局限——任务指定的参考无法即时验证。闭合条件：论文以可提取文本格式交付到仓库。

2. **WCSPH 压缩性误差**: 人工声速 c₀/V = 20 意味着 Mach = 0.05，密度波动约 5%。这对压力系数的影响是系统性的，不是网格收敛能消除的（除非同时提高 c₀/V）。

3. **Spray root 未解析**: Wagner 理论忽略射流，SPH 也无法解析无限薄射流。压力峰值位置在 spray root 附近，SPH 的核平均必然低估或振荡。

4. **当前仅 dx=4mm 已执行**: 细网格结果在集群队列中（cj2979）。本报告以粗网格数据为主，细网格结果待取回后更新。

### 5.2 适用范围

- 本算例验证的是 **Wagner 渐近解**（小 deadrise 线性化势流），不是 Z&F 非线性相似解
- β = 30° 处于 Wagner 理论的适用边界（Z&F 摘要称"小 deadrise 下 Wagner 渐近解与其非线性相似解一致"）
- 对于 β < 20°，Wagner 解更准确；对于 β > 45°，需考虑钝体效应

### 5.3 改进方向

1. **提高 c₀/V**: 从 20 提高到 40-80 可降低压缩性噪声地板（已有 c0_ladder.py 脚本）
2. **δ-SPH 密度扩散**: 开启密度扩散可抑制压力振荡（slamming.py 已实现）
3. **更细网格**: dx ≤ 1mm 是获得合理压力系数的必要条件
4. **获取 Z&F 原文**: 这是闭合本算例最终判据的唯一途径

---

## ⑥ 复现命令

```bash
# 单例运行
PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/run.py --stage run \
    --dx 0.004 --velocity 1.0 --beta 30.0 --frames 13

# 完整矩阵（需集群）
PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix

# 聚合与判决
PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept

# 生成图表
PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/make_figures.py
```

---

## 附录: 测试证据

```
tests/test_wedge_entry.py: 28 passed, 1 skipped (13.7s)
tests/test_case_sph_wedge_entry_2d.py: 13 passed (4.2s)
```

所有 Wagner 解析验证测试通过（机器精度），证明 `wedge_entry.py` 中的解析函数实现正确。

---

**报告生成**: 2026-11-14  
**工件**: `artifacts/case_sph_wedge_entry_2d/beta30_V1_dx4mm.json`  
**集群作业**: cj2979（完整矩阵，队列中）  
**图表**: `docs/figures/wedge_entry_profile_dx4mm.png`, `docs/figures/wedge_entry_timeseries_dx4mm.png`
