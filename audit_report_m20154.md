# 交差审计报告：CASE-SPH-OSC-DROP-FREQ · 振荡液滴频率（Rayleigh 线性理论）

> **审计对象**：m20154（CASE-SPH-OSC-DROP-FREQ·算例池·振荡液滴频率·TensorSPH，已 succeeded）
> **审计任务**：20195（交差核验单——非重做）
> **审计日期**：2026-11-14
> **审计人**：board-audit

---

## ① 案例介绍

**物理背景**：半径 R、密度 ρ、表面张力 σ 的无粘液滴，按 `r(θ)=R[1+εP_l(cosθ)]` 变形后作简谐振荡，
频率有闭式解（Rayleigh 1879, *Proc. R. Soc. Lond.* **29**:71；Lamb 1932, *Hydrodynamics* 6th ed., Art. 275）：

```
ω_l² = l(l−1)(l+2)·σ/(ρR³)  ⇒  l=2: ω² = 8σ/(ρR³)
```

**参数表**：

| 量 | 值 |
|---|---|
| R | 0.02 m |
| σ | 7.2 N/m |
| ρ | 1000.0 kg/m³ |
| 模态 l | 2 |
| 分辨率 R/dx | 8, 12 |
| 初始幅值 ε | 0.02, 0.05 |
| 容差 | <6% |

**解析基准**：ω_Rayleigh(l=2) = 84.853 rad/s

---

## ② 方法

**轴对称归约**：Rayleigh 模的方位角波数 m=0，运动是回转面，方程在子午面 (ρ,z) 上精确降维。
曲率 `κ = κ_meridional + n_ρ/ρ`，体积元 `dV = 2πρ dA`。

**主测量（linear-response）**：离散 CSF 算子的模态法向应力投影
```
dp_meas = (2l+1)/(4πR²) · Σ_i (−σ κ_i) δ_s,i dV_i P_l(cosθ_i)
ω_meas/ω_exact = √(dp_meas/dp_exact)
```

**代码指针**：`src/tensorsph/oscillating_drop3d.py`
- `rayleigh_frequency_3d()` / `rayleigh_frequency_2d()` — 解析频率（3D/2D 分支锁）
- `axisym_drop_geometry()` — 回转体几何+体积元
- `axisym_curvature()` — 子午+方位曲率
- `axisym_modal_stress()` — 模态投影与频率比
- `csf3_audit()` — 仓库三维算子审计

**测试**：`tests/test_oscillating_drop_freq.py` — 17 passed

---

## ③ 计算结果

### 3.1 轴对称 linear-response 测量（主测量）

| 模态 l | eps | R/dx | N | ω_Rayleigh (rad/s) | ω_measured (rad/s) | 误差 % |
|---|---|---|---|---|---|---|
| 2 | 0.02 | 8 | 241 | 84.853 | -123.015 | **-244.97** |
| 2 | 0.02 | 12 | 517 | 84.853 | -137.602 | **-262.17** |
| 2 | 0.05 | 8 | 241 | 84.853 | -103.084 | **-221.49** |
| 2 | 0.05 | 12 | 517 | 84.853 | -123.090 | **-245.06** |

**通过行数：0/4**（容差 <6%）

### 3.2 噪声底

| 键 | 噪声等效模态幅值 ε_noise |
|---|---|
| l2_rdx8 | 0.0019 |
| l2_rdx12 | 0.0061 |

噪声等效幅值 ≥ 种子 ε 时，该分辨率的"误差"是点阵噪声而不是物理。

### 3.3 三维点阵算子审计

| 几何 | R/dx | eps | κR | Σδ_s dV/A |
|---|---|---|---|---|
| sphere-3D | 8 | 0 | 0.774 | 0.328 |
| sphere-3D | 8 | 0.03 | 0.919 | 0.342 |
| sphere-3D | 12 | 0 | 0.766 | 0.342 |
| sphere-3D | 12 | 0.03 | 0.964 | 0.365 |

球面 κR 应为 2.0，平面 Σδ_s dV/A 应为 1.0。实测偏离到倍数量级且**不随分辨率收敛**。

---

## ④ 结果图

- `benchmarks/oscillating_drop_freq/fig_frequency_error.png` — 频率误差随分辨率/幅值变化
- `benchmarks/oscillating_drop_freq/fig_noise_floor.png` — 模态噪声底测量
- `benchmarks/oscillating_drop_freq/fig_lattice3d_audit.png` — 三维点阵算子审计

---

## ⑤ 注意点

1. **轴对称归约实现不自洽**：⟨κR⟩≈1.21（球面应为 2.00），方位项 `n_ρ/ρ` 在加权均值里贡献不足。
2. **曲率估计器模态噪声底过高**：ε_noise=0.0019-0.0061，与种子 ε=0.02 同阶。
3. **三维点阵算子不可用**：csf3 的 κR≈0.77（应为 2.0），δ_s 测度 0.33（应为 1.0），不随分辨率收敛。
4. **未跑时间推进**：本单只有 linear-response 测量，动力臂（含离散惯性、寄生流）未做。

---

## ⑥ 误差判定

| 关键量 | 基准值 | 测量值（最佳） | 误差 | 是否<6% |
|---|---|---|---|---|
| ω (l=2, ε=0.05, R/dx=8) | 84.853 rad/s | -103.084 rad/s | -221.49% | ❌ FAIL |
| ω (l=2, ε=0.02, R/dx=12) | 84.853 rad/s | -137.602 rad/s | -262.17% | ❌ FAIL |
| κR (球面) | 2.0 | ~1.21 | -39.5% | ❌ FAIL |
| Σδ_s dV/A (平面) | 1.0 | 0.62-0.63 | -37% | ❌ FAIL |

**结论**：所有关键量均远超 6% 容差。本实现级失败，不能读作 TensorSPH CSF 算子本身的属性。
下一轮需先修算子（曲率-测度自洽的 surface-fit 曲率 + 面积归一 delta），再跑轴对称动力臂。

---

## 判级：L3（系统集成）

| 级别 | 判定 | 理由 |
|---|---|---|
| L0 文档成果 | ✓ | 有完整文档 `docs/case_sph_osc_drop_freq.md` |
| L1 数学/单元验证 | ✓ | 解析参考测试 8 项全部通过 |
| L2 独立功能实现 | ✓ | 真实代码：输入→计算→输出 |
| L3 系统集成 | ✓ | 接入主求解器模块 `src/tensorsph/oscillating_drop3d.py`，17 项测试通过 |
| L4 基准验证 | ❌ | 基准算例完整+误差分析+可复现，但 **0/4 行通过 6% 容差** |
| L5 科研工程成果 | ❌ | 基准未通过 |

**最终判级：L3**

---

## 取证核验三件套

| 项目 | 状态 | 证据 |
|---|---|---|
| run_artifacts 文件 | ✓ | `benchmarks/oscillating_drop_freq/{report.json, fig_*.png, make_report.py}` |
| benchmark ref vs got 数值 | ✓ | report.json 中逐项记录，0/4 通过 |
| repro_cmd 可执行 | ✓ | `PYTHONPATH=src python benchmarks/oscillating_drop_freq/run_freq.py` 验证通过 |
| 测试 | ✓ | `python -m pytest tests/test_oscillating_drop_freq.py -q` → 17 passed |
| 报告六段 | ✓ | ①案例介绍 ②方法 ③参数表 ④对照表 ⑤收敛性 ⑥失败模式 |

---

## 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorSPH', 'm20154 CASE-SPH-OSC-DROP-FREQ', 20195, 3, '<JSON>', 'board-audit');
```

已执行，verdict id=61。
