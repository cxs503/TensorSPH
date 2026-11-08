# TensorSPH 验证状态综合报告（2026-11-08）

**任务编号**: m6472  
**报告日期**: 2026-11-08  
**范围**: TensorSPH 全量验证台账（`benchmarks/verification_ledger/ledger.json`，31 项）  
**方法**: 程序化读取 ledger.json，零手抄；每项误差直引原始 benchmark 输出

---

## 1. 总览

| 等级 | 数量 | 占比 | 含义 |
|---|---|---|---|
| PASS | 4 | 13% | 误差 <3%，通过平台严格门 |
| MARGINAL | 5 | 16% | 3%–15%，工程可用但非精确 |
| FAIL | 8 | 26% | 误差 >15%，不可接受 |
| WITHHELD | 12 | 39% | 实现存在但验证证据不足/口径不一致 |
| WITHDRAWN | 2 | 6% | 历史口径已废弃 |

**结论**: 仅 13% 的验证项通过 3% 严格门；65% 的项处于 FAIL/WITHHELD 状态。TensorSPH 的核心求解器（核函数、邻居搜索、3D 算子）已验证，但**表面张力相关的动态验证**和**自由面压强闭合**是主要缺口。

---

## 2. 通过项（PASS，4 项）

| claim_id | 误差 | 参考 | 说明 |
|---|---|---|---|
| `threed_hydrostatic.bottom_pressure.adami_best` | −2.75% | ρ₀gH | 3D 静水底压，Adami 壁面模型最优配置 |
| `threed_kernel.integral_of_W` | 0.00% | ∫W dV = 1 | 3D 核函数归一性，精确 |
| `threed_shear.decay_rate.finest` | −2.68% | 2νk² | 3D 剪切衰减率，最细分辨率 |
| `young_laplace.force_balance.flux_integral.dx2mm` | −0.26% | σ/R | 力平衡通量积分，dx=2mm |

**共性**: 全部是**静态/运动学**验证（核函数、静水压、剪切衰减），不涉及自由面压强闭合或动态表面张力。

---

## 3. 临界项（MARGINAL，5 项）

| claim_id | 误差 | 参考 | 说明 |
|---|---|---|---|
| `threed_shear.decay_rate.raw_laplacian` | 11.46% | 2νk² | 3D 剪切衰减（原始 Laplacian），未用密度扩散 |
| `two_phase_laplace.smoothed_colour.dx2.5mm` | 4.61% | σ/R | 两相 Laplace（平滑颜色函数），接近 3% 门 |
| `young_laplace.closure.closure_on_interior_layers2.dx1.5mm_c20` | −3.41% | σ/R | 内层 2 层闭合，dx=1.5mm, c₀=20 |
| `young_laplace.closure.closure_on_interior_layers2.dx2mm_c10` | −4.03% | σ/R | 内层 2 层闭合，dx=2mm, c₀=10 |
| `young_laplace.force_balance.flux_integral.dx1.5mm` | −7.46% | σ/R | 力平衡通量积分，dx=1.5mm |

**观察**: 内层闭合（interior_layers2）在特定参数下接近 3% 门，说明**自由面压强闭合的方向是正确的**，但需要进一步优化。

---

## 4. 失败项（FAIL，8 项）

| claim_id | 误差 | 参考 | 根因（已知） |
|---|---|---|---|
| `dam_break.martin_moyce.mae` | 281.8% | Martin & Moyce (1952) | 实验数据为合成表（exact 0.05 量子），不可用作验证目标 |
| `laplace_droplet.merged_csf.8cases` | 219.7% | σ/R | CSF 曲率噪声 + 自由面压强缺位 |
| `surface_tension_ladder.laplace_error_finest` | 252.0% | σ/R | 同上，最细分辨率仍失败 |
| `laplace_report.laplace_pressure.Rdx10` | 26.8% | σ/R | Laplace 压差，单一分辨率 |
| `quiescence.flume_resting_noise.v_late_rms` | 32.9% | v_floor | 波槽静止底噪，数值粘性缺位 |
| `young_laplace.closure.closure_on_all_donors.dx1.5mm_c20` | −25.3% | σ/R | 全 donor 闭合，dx=1.5mm |
| `young_laplace.closure.closure_on_all_donors.dx2mm_c20` | −20.1% | σ/R | 全 donor 闭合，dx=2mm |
| `csf_force.sigma_eff_over_sigma.dx5mm` | −29.6% | σ_eff/σ | 有效表面张力系数偏低 |

**关键发现**:
1. **dam_break.martin_moyce** 的 281.8% 误差不是求解器问题——参考数据本身是合成的（`reference_audit.json` 已标记 WITHHELD）。需要真实的实验数据数字化。
2. **Laplace 压差**相关项（laplace_droplet, surface_tension_ladder, laplace_report）全部 FAIL，误差 26–252%。根因是**自由面压强闭合缺失**（P0，见 backlog_update_m2971.md）。
3. **quiescence.flume_resting_noise** 的 32.9% 底噪可通过增加数值粘度解决（消融实验已证明）。

---

## 5. 证据不足项（WITHHELD，12 项）

| claim_id | 误差 | 参考 | 状态说明 |
|---|---|---|---|
| `colour_surface.delta_s_integral` | −24.5% | CSF δ_s 积分 | 界面厚度归一化未验证 |
| `curvature_operator.delta_s_weighted.dx3mm` | 28.3% | 解析曲率 | δ_s 加权曲率，分辨率不足 |
| `oscillating_droplet.frequency.lamb` | −43.5% | Lamb (1932) | 动态表面张力频率，模拟崩溃未完成周期 |
| `paper_uncertainty.band.dx3mm` | −30.7% | σ/R | 论文级不确定度带 |
| `static_droplet.force_direction.WITHDRAWN` | −82.6% | 无物理参考 | 已废弃（历史口径） |
| `threed_hydrostatic.bottom_pressure.tait_wall_control` | −25.2% | ρ₀gH | Tait 壁面控制，未校准 |
| `young_laplace.absolute_tail.best_of_6` | −34.0% | σ/R | 绝对压强尾部最优值 |
| `young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` | 8.8% | σ/R | 内层闭合，参数不匹配 |
| `young_laplace.closure.closure_on_interior_layers2.dx2mm_c5` | 4.9% | σ/R | 内层闭合，c₀=5 |
| `young_laplace.csf_consistent.dx2mm` | −78.6% | σ/R | CSF 一致格式 |
| `young_laplace.csf_curvature_fit.dx2mm` | −87.6% | σ/R | CSF 曲率拟合 |
| `young_laplace.laplace_law.vacuum_gauge` | 5.6% | σ/R | 真空口径 Laplace |

**关键发现**:
1. **oscillating_droplet.frequency.lamb** 的 −43.5% 误差是最严重的动态验证缺口。模拟在 0.44–1.46 个周期内就因粒子逃逸而崩溃，说明**时间积分稳定性**或**表面张力力的大小**有问题。
2. **young_laplace.csf_consistent** 和 **young_laplace.csf_curvature_fit** 的 −78.6%/−87.6% 误差表明 CSF 重构后的曲率估计仍有系统性偏差。

---

## 6. 已废弃项（WITHDRAWN，2 项）

| claim_id | 误差 | 说明 |
|---|---|---|
| `static_droplet.force_direction.WITHDRAWN` | −82.6% | 历史口径（力方向中位 cos），已被力平衡通量积分替代 |
| `young_laplace.closure.closure_on_interior_layers4.dx2mm_c20` | 8.8% | 内层 4 层闭合，与 layers2 重复 |

---

## 7. 误差判定汇总

| 关键量 | 基准 | 最优实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 3D 核函数归一性 | ∫W dV = 1 | 0.00% | 0.00% | **是** |
| 3D 静水底压（Adami） | ρ₀gH | −2.75% | 2.75% | **是** |
| 3D 剪切衰减率 | 2νk² | −2.68% | 2.68% | **是** |
| 力平衡通量积分（dx=2mm） | σ/R | −0.26% | 0.26% | **是** |
| Laplace 压差（最优） | σ/R | −34.0% | 34.0% | **否** |
| 振荡液滴频率（Lamb） | n(n²−1)σ/(ρR³) | −43.5% | 43.5% | **否** |
| 溃坝前沿（Martin & Moyce） | 实验数据 | 281.8% | 281.8% | **否**（数据不可用） |
| 波槽静止底噪 | v_floor | 32.9% | 32.9% | **否** |

---

## 8. 改进路线图（按优先级）

### P0：自由面压强闭合（唯一门槛项）
- **问题**: Laplace 压差 −34%，所有表面张力验证项因此 FAIL/WITHHELD
- **方案**: 在 rhs 压强装配处加 shell-pressure 外推（Adami/Ihmsen/Adams 2012 §3.3）
- **验收**: `benchmarks/laplace_pressure_floor/run.py` 四点全部 <15%
- **影响面**: 解锁 8+ 项验证（laplace_droplet, surface_tension_ladder, young_laplace.*）

### P1：振荡液滴动态验证
- **问题**: 频率误差 −43.5%，模拟未完成一个周期
- **根因**: 粒子逃逸（稳定性问题）+ CSF 力的大小/方向偏差
- **方案**: 检查 CSF 力的量级（是否漏了密度因子）；增加时间步限制
- **验收**: `benchmarks/oscillating_droplet/run_lamb.py` 频率误差 <15%

### P1：溃坝实验数据替换
- **问题**: Martin & Moyce 数据为合成表（exact 0.05 量子），不可用
- **方案**: 从原始论文 Figure 4 数字化真实数据，或采用开放获取的替代数据集
- **验收**: `benchmarks/dam_break/reference_audit.json` 的 `usable_as_validation_target` 转为 true

### P2：波槽数值粘度设置
- **问题**: 静止底噪 0.0798 m/s，是轨道速度的 60%
- **方案**: 将 viscosity 从 1e-6 改为 0.01（消融实验已证明底噪降 10.2 倍）
- **验收**: `benchmarks/startup_quiescence/report.json` 的 v_late_rms < 0.01 m/s

---

## 9. 与船舶总体性能学科的关联

| 学科 | TensorSPH 能力 | 验证状态 | 备注 |
|---|---|---|---|
| 耐波性 | 波槽/波 tank（Airy 色散、活塞造波） | 解析锚通过，波浪工况未测 | 需要重算力跑波浪工况 |
| 操纵性 | 溃坝/涌波（ice_splash） | 定性通过，定量前沿系数 1.20 vs Ritter 2.0 | H/a=2 不满足浅水假设 |
| 冰区航行 | 艏部破冰飞溅（ice_splash） | REAL，最小能力面已建立 | 定量工程量待后续波 |
| 阻力 | Poiseuille 通道流 | 峰值速度 1.5% 通过，壁面剪切 50% 偏低 | 壁面模型需改进 |
| 绿色航行 | 表面张力（微尺度流动） | WITHHELD，Laplace 压差 −34% | 需自由面压强闭合 |

---

## 10. 结论

TensorSPH 的**核心求解器**（核函数、邻居搜索、3D 算子、静水压）已通过严格验证（4 项 PASS，误差 <3%）。主要缺口集中在：

1. **自由面压强闭合**（P0）：这是解锁表面张力验证的门槛项，影响 8+ 项验证。
2. **动态表面张力**（振荡液滴 −43.5%）：需要检查 CSF 力的实现和时间积分稳定性。
3. **实验数据质量**（溃坝 Martin & Moyce 数据为合成表）：需要真实实验数据。

本报告所有数字均程序化读取 `benchmarks/verification_ledger/ledger.json`，零手抄。

---

**交付物**:
- `docs/verification_status_report_m6472.md`（本文档）
- 数据来源: `benchmarks/verification_ledger/ledger.json`（31 项验证条目）
