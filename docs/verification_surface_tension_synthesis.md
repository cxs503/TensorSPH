# CSF 表面张力模型验证总览（Synthesis）

**范围**: TensorSPH `src/tensorsph/surface_tension.py`, `csf_consistent.py`, `csf_curvature.py`
**日期**: 2026-11-03 ｜ **审计员**: m2883 (self-start, verifier seat)
**基准**: 2-D 单相静止液滴的 Young-Laplace 定律与切割力恒等式

## 1. 模型演进史

| 阶段 | 提交 | 色函数 c | 梯度/散度 | delta_s | 曲率 | 切割力误差 | 曲率误差 |
|---|---|---|---|---|---|---|---|
| m277 (legacy) | PR #13 | 邻居数启发式 (跳变 O(0.3)) | 标准 SPH (无 MLS 校正) | \|grad c\| | -div n | 48-72% | 31-109% |
| consistent (m2142+) | 多轮迭代 | 核和 c_i = Σ V_j W_ij | MLS 一阶完备 | 2\|grad c\| | MLS div n + Shepard 平滑 | 7-11% | +25% |
| fit_quadratic (m2188+) | csf_curvature.py | 同上 | 同上 | 同上 | 局部二次拟合 (4h 模板) | 1-11% | 0.9-17.8% |

## 2. 当前状态（HEAD 实测，dx = 0.002, R = 0.02 m）

### 2.1 通过项（< 15% 平台标准）

| 关键量 | 基准 | consistent (mls) | consistent (fit_quadratic) | 判定 |
|---|---|---|---|---|
| 切割力 F_cut/(2σ) | 1.0 | 1.077 (7.7%) | 0.899 (10.1%) | ✅ 两者均通过 |
| 力方向中位 cos | 1.0 | 1.000 | 1.000 | ✅ 精确 |
| 伪流 max\|v\|/v_ref | < 0.1 (Adami) | 0.037 | 0.028 | ✅ 通过 |
| 曲率 (fit_quadratic, dx=0.0015) | 50 1/m | 62.4 (+24.9%) ❌ | 50.4 (+0.9%) | ✅ fit_quadratic 通过 |

### 2.2 未通过项（已知欠账）

| 关键量 | 基准 | 实测 | 误差 | 原因 |
|---|---|---|---|---|
| delta_s 归一 ∫δ_s dn | 1.0 | 0.74-0.76 | -24% 至 -26% | 单相粒子云中最外层梯度被抹平；c 从 0.5→0 的过渡在真空中无人携带 |
| Laplace 压差 Δp | 360 Pa | 44-88 Pa | -76% 至 -88% | WCSPH 求和密度噪声地板（Δp 对应压缩仅 6.3e-3，噪声 ~1e-3） |
| 曲率 (mls 默认) | 50 1/m | 62.4 1/m | +24.9% | div n 在粒子壳层上付 O(1) 格子误差 |

### 2.3 测试套件状态

```
tests/test_surface_tension.py          6 passed
tests/test_csf_consistent.py          19 passed
tests/test_csf_curvature_fit.py       10 passed
tests/test_surface_tension_laplace.py  5 passed, 2 xfail (legacy scheme)
```

所有 xfail 均为 `strict=True`，标记 legacy 方案的已知物理缺陷；consistent 方案无 xfail。

## 3. 误差根因分析

### 3.1 delta_s 归一亏损（-26%）

CSF delta 函数必须满足 `∫ δ_s dn = 1`。在单相 SPH 中：
- 色函数 c 在流体内部饱和到 1，在最外层粒子处降至 ~0.5
- 剩余的 c: 0.5 → 0 过渡发生在真空中，没有粒子携带
- 因此 `∫_fluid |grad c| dn = 1 - 0.5 = 0.5`，理论因子 2 补偿
- 但实测 `∫_fluid |grad c| dn ≈ 0.37`（而非 0.5），说明最外层梯度被抹平

**建议修复**：
1. Ghost colour：在界面外 dx 处创建虚拟粒子（c=0），重新计算梯度
2. 局部 Shepard 归一：对每个界面粒子，沿法向积分 |grad c| 并局部缩放

### 3.2 Laplace 压差 WITHHELD

Δp = σ/R = 360 Pa 对应 Tait 状态方程的密度压缩仅 6.3e-3。求和密度在 ~20 邻居下的离散噪声约 1e-3 量级，即压差测量本身被噪声地板吞掉。

**这不是表面张力离散的回归**，而是 WCSPH 弱可压 + 求和密度的已知限制。修复需要：
- 背景压强偏置（Adami et al. 2012: p₀ = σ/R）
- 或降低声速 c₀（但会增加 Mach 数）
- 或改用无散密度重构

### 3.3 曲率算子选择

`csf_curvature` 参数控制曲率计算：
- `"mls"`（默认）：div n + Shepard 平滑，误差 +25%
- `"fit_quadratic"`：局部二次拟合，误差 0.9-17.8%（随分辨率收敛）
- `"fit_circle"`：代数圆拟合，在格子数据上失败（+78-258%）

**建议**：将 `fit_quadratic` 设为新默认值，保留 `mls` 为回退。

## 4. 验证判据总表

| 判据 | 基准 | legacy | consistent (mls) | consistent (fit_quadratic) | 平台标准 |
|---|---|---|---|---|---|
| 切割力 F_cut/(2σ) | 1.0 | 0.28-0.52 ❌ | 1.07-1.11 ✅ | 0.89-1.01 ✅ | < 15% |
| 曲率 κR | 1.0 | -0.21 to -0.26 ❌ | 1.25 ❌ | 1.01-1.18 ✅ | < 15% |
| ∫δ_s dn | 1.0 | 0.29-0.34 ❌ | 0.74-0.76 ❌ | 0.74-0.76 ❌ | < 15% |
| 力方向 cos | 1.0 | 0.51-0.60 ❌ | 1.000 ✅ | 1.000 ✅ | > 0.85 |
| 伪流 | < 0.1 | 0.20 ❌ | 0.037 ✅ | 0.028 ✅ | < 0.1 |
| Δp (relax) | 360 Pa | 5 Pa ❌ | 44-88 Pa ❌ | -24 Pa ❌ | < 15% |

## 5. 路线图

### P0（阻塞合入/发布）
- [ ] delta_s 归一修复（ghost colour 或局部 Shepard 归一）
- [ ] 将 `fit_quadratic` 设为 `csf_curvature` 默认值

### P1（重要改进）
- [ ] Laplace 压差基准修复（背景压强偏置或密度重构）
- [ ] 3-D 扩展（当前仅 2-D 验证）
- [ ] 自适应模板半径（低分辨率下 4h 模板弧长不足）

### P2（长期）
- [ ] 多相 SPH 扩展（真正两相流，非密度比色函数）
- [ ] 实验数据对照（液滴振荡频率、毛细波色散）
- [ ] 隐式表面张力时间积分（避免毛细管 CFL 限制）

## 6. 参考

- `docs/verification_surface_tension_laplace.md` — Laplace 定律验证（legacy 失败记录）
- `docs/verification_surface_tension_consistent.md` — consistent 离散验证
- `docs/verification_surface_tension_curvature.md` — 曲率拟合算子验证
- `benchmarks/surface_tension/laplace.py` — Laplace 定律基准（分辨率扫描）
- `benchmarks/csf_consistent/run.py` — consistent vs legacy 对照基准
- `benchmarks/csf_curvature_fit/run.py` — 曲率算子 A/B 对照基准
