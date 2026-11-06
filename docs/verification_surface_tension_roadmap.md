# TensorSPH 表面张力验证状态与修复路线图

**生成日期**: 2026-11-05  
**数据源**: `docs/verification_ledger.md`（16 claims, 10 physical quantities）  
**范围**: CSF 表面张力模型（`src/tensorsph/csf_consistent.py`, `csf_curvature.py`）

---

## 1. 当前状态总览（机裁分级）

| 分级 | 数量 | 含义 |
|---|---|---|
| MARGINAL | 1 | 两相 Laplace（无自由面）4.6% 误差 |
| WITHHELD | 8 | 可测量但不可采信（>15% 或量具退化） |
| FAIL | 6 | 已证伪（不收敛/自证失败） |
| WITHDRAWN | 1 | 已撤销（static_droplet force_direction） |

**关键结论**: 表面张力模型在**无自由面**的两相周期算例中达到 4.6% 误差（MARGINAL），但在**有自由面**的单相算例中全部 >15%。这证明 CSF 力的离散本身是正确的，问题出在**自由面密度欠计**导致的颜色函数梯度偏置。

---

## 2. 逐项验证状态

### 2.1 力的方向与幅值（已修复）

| 判据 | 状态 | 误差 | 备注 |
|---|---|---|---|
| 力方向 cos(a, r̂) | ✅ PASS | 0.0% | 中位 cos = 1.000 |
| 切割力恒等式 F_cut/(2σ) | ✅ MARGINAL | 7.4–11% | 六角晶格，三档分辨率均 <15% |
| 伪流 max\|v\|/v_ref | ✅ PASS | 0.044 | < 0.1 (Adami 2012) |

### 2.2 曲率（部分修复）

| 方案 | 状态 | 误差 | 备注 |
|---|---|---|---|
| MLS div n | ❌ WITHHELD | +25% | 法向噪声被微分放大 |
| fit_quadratic（默认） | ❌ FAIL | +29% (R/h=6.15) | 随加密恶化：0.7106 → 0.6818 |
| fit_quadratic + color smoothing=1 | ⚠️ MARGINAL | 7–14% | opt-in，粗分辨率仍超门 |

### 2.3 delta_s 归一化（未修复）

| 判据 | 状态 | 实测 | 解析 |
|---|---|---|---|
| ∫ delta_s dn | ❌ WITHHELD | 0.75 | 1.0 |

已应用 2× 因子（单相 c 从 1→0.5，另一半在真空中），但积分仍缺 25%。根因：核和颜色函数在自由面处的离散值不是精确的 0.5，而是 ~0.62（取决于晶格和 h/dx）。

### 2.4 Young-Laplace 压差（未修复）

| 算例 | 状态 | 误差 | 根因 |
|---|---|---|---|
| 两相周期（无自由面） | ✅ MARGINAL | 4.6% | CSF 力正确 |
| 单相液滴（有自由面） | ❌ WITHHELD | -34% ~ -78% | 自由面密度欠计 → Tait 钳位 → 压强跳变被吃掉 |

**关键证据**: `benchmarks/laplace_pressure_jump/` 的声速阶梯实验证明：
- 均值误差与 c0 无关（-37.4% ~ -35.8%）→ 不是 EOS 分辨率地板
- 噪声 std/mean 随 c0 下降（80% → 25%）→ EOS 放大器只影响噪声
- 压强剖面显示 `r/R > 0.9` 层压强被钳到 0（rho/rho0 = 0.874）

---

## 3. 根因分析

### 3.1 问题链

```
自由面邻居缺失
  → 密度求和 rho < rho0（最外层 ~0.75–0.90）
    → Tait EOS 钳位 p = max(0, ...) → 自由面压强恒为 0
      → 颜色函数 c = sum V_j W_ij 在自由面处偏高（~0.62 而非 0.5）
        → |grad c| 偏小 → delta_s 积分 = 0.75 而非 1.0
          → CSF 力幅值偏低 25%
            → Laplace 压差偏低 34–78%
```

### 3.2 为什么两相算例能通过

两相周期算例中，界面两侧都有粒子，颜色函数从 1 平滑过渡到 0，积分精确为 1。没有自由面密度欠计问题。

---

## 4. 修复路线图

### 4.1 P0: δ-SPH 密度扩散（推荐）

**原理**: Antuono et al. (2010) 的 δ-SPH 密度扩散项：
```
D_i = δ h c0 sum_j (rho_j - rho_i) (x_j - x_i) · grad W_ij / |x_j - x_i|² V_j
```
在自由面处补充密度，使 rho/rho0 接近 1，从而：
- 颜色函数在自由面处接近 0.5
- delta_s 积分接近 1.0
- 压强跳变不再被钳位吃掉

**预期效果**: 两相算例已证明 CSF 力正确（4.6%），修复密度后单相算例应能达到 <15%。

**工作量**: 中等（~200 行代码 + 验证）

**风险**: 低（δ-SPH 是成熟方法，已有 `cfg.density_diffusion` 参数）

### 4.2 P1: 背景压强（备选）

**原理**: Adami et al. (2012) 的背景压强偏置：
```
p_i = p0 + B[(rho_i/rho0)^gamma - 1],  p0 = sigma/R
```
使自由面处压强不为 0，从而保留压强跳变。

**预期效果**: 直接解决压强钳位问题，但不解决 delta_s 归一化。

**工作量**: 小（~50 行代码）

**风险**: 中（需要调整 Tait EOS，可能影响其他案例）

### 4.3 P2: 颜色函数修正（长期）

**原理**: 使用 Shepard 平滑的颜色函数计算曲率（`curvature_color_smoothing=1`），同时保持力的锐利场。

**预期效果**: 曲率误差从 +25% 降到 7–14%。

**工作量**: 已完成（opt-in），需设为默认。

**风险**: 低（已有验证数据）

---

## 5. 验收标准

| 判据 | 当前 | 目标 | 门禁测试 |
|---|---|---|---|
| delta_s 积分 | 0.75 | >0.90 | `test_csf_consistent.py` |
| 切割力 F_cut/(2σ) | 0.89–0.93 | >0.95 | `test_effective_sigma.py` |
| Laplace 压差（单相） | -34% ~ -78% | >-15% | `test_laplace_pressure_jump.py` |
| 曲率（fit_quadratic） | +29% | <15% | `test_csf_curvature_fit.py` |

---

## 6. 证据指针

| 文档 | 路径 |
|---|---|
| 验证账本 | `docs/verification_ledger.md` |
| 一致离散验证 | `docs/verification_surface_tension_consistent.md` |
| 曲率偏置归因 | `docs/verification_surface_tension_curvature.md` |
| Laplace 压差复核 | `docs/verification_laplace_pressure_jump.md` |
| m3265 修复报告 | `docs/verification_surface_tension_fix_m3265.md` |
| 基准工件 | `benchmarks/csf_consistent/`, `benchmarks/laplace_pressure_jump/` |

---

## 7. 给下次自找活的建议

1. **最高价值**: 实现 δ-SPH 密度扩散（P0），预期可使 Laplace 压差从 -37% 改善到 <15%
2. **次高价值**: 将 `curvature_color_smoothing=1` 设为默认，曲率误差从 +25% 降到 7–14%
3. **验证欠账**: `test_oscillating_droplet_lamb.py::test_mode_oscillates_near_the_capillary_frequency` 在重负载下超时，需确认是负载问题还是真实失败
4. **实验对照**: dam-break Martin & Moyce (1952) 对照 FAIL（MAE 0.57 vs 限 0.15），需排查
