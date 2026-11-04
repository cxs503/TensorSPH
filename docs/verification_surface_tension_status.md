# TensorSPH 表面张力验证状态总览

**日期**: 2026-11-04 ｜ **范围**: TensorSPH CSF 表面张力模型 ｜ **版本**: v0.8.0+

## 1. 模型演进

| 阶段 | 提交/任务 | 核心变更 | 验证状态 |
|------|-----------|----------|----------|
| 初始实现 | m277 | 邻居数启发式色函数 + 未修正 SPH 算子 | WITHHELD（压差偏差 77–118%） |
| 一致格式重构 | m2592 | 核和色函数 + MLS 一致梯度/散度 + 接触角 BC | 部分通过（曲率 +25%，压差 −16…−21%） |
| 曲率拟合算子 | m3016/m3020 | `fit_quadratic`/`fit_circle` 局部界面拟合 | 曲率误差降至 +7…+14% |
| 曲率偏置审计 | m3023/m3028 | 内壳饱和粒子偏置识别 + Shepard 平滑选项 | 文档化，可选修复 |
| 陈旧测试清理 | m3330 | 4 个过时 xfail 重写为当前状态断言 | 11/11 PASS |

## 2. 验证判据矩阵

### 2.1 静态 Laplace 压差（B3a）

| 判据 | 解析值 | 实测（一致格式） | 误差 | 判定 |
|------|--------|------------------|------|------|
| Δp = σ/R（R=0.1 m, σ=0.072） | 0.720 Pa | 0.57–0.61 Pa | −16…−21% | **FAIL**（>15%） |
| 曲率 κR（圆界面） | 1.0 | 1.07–1.25 | +7…+25% | **边界**（依赖算子） |
| 色函数积分 ∫\|∇c\|dn | 1.0 | 0.92–0.97 | −3…−8% | PASS |
| 法向径向一致性 | 1.0 | >0.98 | <2% | PASS |
| 伪流 max\|v\|/√(σ/ρR) | <0.1 | 0.05–0.08 | — | PASS |

**残差归因**（m2971）：自由面压强闭合缺失（Tait EOS 钳 p≥0 + 密度亏缺 −6…−34%），
非曲率算子问题。

### 2.2 动态振荡液滴（B3b）

| 判据 | 解析值 | 实测 | 误差 | 判定 |
|------|--------|------|------|------|
| ω₂ = √(6σ/ρR³)（2-D, m=2） | — | 待完整运行 | — | **待验证** |
| 阻尼率（粘性） | — | 待完整运行 | — | **待验证** |

**注**：`benchmarks/oscillating_droplet/run.py` 已实现，完整运行需 >120s（本机 load15≥135 时跳过）。

### 2.3 接触角边界

| 判据 | 实测 | 判定 |
|------|------|------|
| 中性角（π/2）保持梯度 | PASS | ✅ |
| 亲水角倾斜向壁 | PASS | ✅ |
| 疏水角倾斜离壁 | PASS | ✅ |
| 远场粒子不变 | PASS | ✅ |
| 梯度模长守恒 | PASS | ✅ |
| 零 σ 无力 | PASS | ✅ |

**测试**: `tests/test_contact_angle.py` 10/10 PASS。

### 2.4 毛细波频率

| 判据 | 实测 | 判定 |
|------|------|------|
| ω² = σk³/ρ 公式正确性 | PASS（解析验证） | ✅ |
| ω ~ k^(3/2) 标度律 | PASS | ✅ |
| ω ~ √σ 标度律 | PASS | ✅ |
| 毛细时间步限制有限（σ>0） | PASS | ✅ |
| 毛细时间步限制无限（σ=0） | PASS | ✅ |

**注**：完整动力学验证（振幅衰减 vs 解析）因计算量重未在本次完成。

## 3. 严格 xfail 清单（模型修复后自动转 xpass 并失败）

| 测试 | 文件 | 原因 |
|------|------|------|
| `test_laplace_law_is_satisfied` | test_surface_tension_laplace.py | 压差 >15% |
| `test_curvature_matches_exact_interface_curvature` | test_surface_tension_laplace.py | 曲率偏差 |
| `test_curvature_reproduces_circle` | test_laplace_law.py | CSF 曲率 on circle |
| `test_laplace_jump_within_15pct` | test_laplace_pressure_floor.py | 自由面压强闭合 |

## 4. 测试套件状态（轻量子集，2026-11-04）

| 模块 | 通过 | 失败 | xfail | 墙钟 |
|------|------|------|-------|------|
| test_surface_tension.py | 15 | 0 | 0 | ~3s |
| test_surface_tension_laplace.py | 4 | 0 | 2 | ~2s |
| test_laplace_law.py | 8 | 0 | 1 | ~3s |
| test_laplace_pressure_floor.py | 8 | 0 | 1 | ~2s |
| test_laplace_droplet.py | 11 | 0 | 0 | ~16s |
| test_contact_angle.py | 10 | 0 | 0 | ~4s |
| test_csf_consistent.py | 12 | 0 | 0 | ~2s |
| test_csf_curvature_audit.py | 6 | 0 | 0 | ~2s |
| test_csf_curvature_fit.py | 10 | 0 | 0 | ~3s |
| test_laplace_plateau.py | 2 | 0 | 0 | ~1s |
| test_young_laplace.py | 2 | 0 | 0 | ~1s |
| test_laplace_validation.py | 7 | 0 | 1 | ~3s |
| test_laplace_verification.py | 2 | 0 | 3 | ~2s |
| test_paper_laplace_uncertainty.py | 9 | 0 | 0 | ~3s |
| test_free_surface.py | 7 | 0 | 0 | ~2s |
| test_wall_pressure.py | 3 | 0 | 0 | ~1s |
| test_lattice_wall_clearance.py | 5 | 0 | 0 | ~1s |
| test_ritter.py | 10 | 0 | 0 | ~95s |
| test_ritter_reference.py | 9 | 0 | 0 | ~1s |
| test_dam_break_validation.py | 6 | 0 | 0 | ~2s |
| test_dam_break_reference_audit.py | 12 | 0 | 0 | ~3s |
| test_dam_break_experimental.py | 9 | 0 | 0 | ~1s |
| test_dam_break_resolution_ladder.py | 6 | 0 | 0 | ~1s |
| **合计（轻量）** | **166** | **0** | **9** | **~155s** |

**未包含**（计算量重，load15≥135 时跳过）：
- test_capillary_wave.py（完整动力学）
- test_oscillating_droplet.py（时间积分）
- test_dam_break.py（完整求解器运行）
- test_convergence.py（分辨率阶梯）
- test_droplet_impact.py（完整求解器运行）
- test_pair_based_rhs.py（大规模 RHS）
- test_adaptive_dt.py（自适应时间步）
- test_neighbor_search.py（大规模邻居搜索）

## 5. 已知缺口与排期

### P0（门槛项）
- **自由面压强闭合**：`rhs` 压强装配处加 shell-pressure 外推（Adami/Ihmsen/Adams 2012 §3.3）。
  验收：`benchmarks/laplace_pressure_floor/run.py` 四点全部 <15%，
  `test_laplace_jump_within_15pct` 由 xfail 转 xpass。
  **影响**：解决后 Laplace 压差误差从 −34% 降至 <15%。

### P1
- **初始化质量校准**：`make_particles` 按核心区实测核和做 `m → m/(1+b)` 校准。
  验收：初始 `p_max/(ρ0 c0²) < 1e-4`。
- **振荡液滴完整验证**：ω₂ 频率 + 阻尼率 vs 解析解。

### P2（已完成）
- ~~陈旧钉死失败断言清理~~ → m3330 已合入
- ~~验证口径统一~~ → m2971 已文档化

## 6. 参考文献

1. Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335 — CSF 原始论文
2. Adami, Ihmsen & Adams (2012), J. Comput. Phys. 231:7057 — 自由面压强闭合 + 伪流
3. Bonet & Lok (1999), Comput. Methods Appl. Mech. Engrg. 180:97 — MLS 一致算子
4. Rayleigh (1879), Proc. London Math. Soc. 10:4 — 振荡液滴频率
5. Lamb (1932), Hydrodynamics 6th ed. art. 267 — 毛细波/振荡液滴

## 7. 工件指针

| 工件 | 路径 |
|------|------|
| 一致格式 CSF 实现 | `src/tensorsph/csf_consistent.py` |
| 曲率拟合算子 | `src/tensorsph/csf_curvature.py` |
| Laplace 压差基准 | `benchmarks/laplace_pressure_floor/` |
| Laplace 定律基准 | `benchmarks/laplace_law/` |
| 振荡液滴基准 | `benchmarks/oscillating_droplet/` |
| 毛细波基准 | `benchmarks/capillary_wave/` |
| 接触角验证 | `docs/contact_angle_boundary.md` |
| 曲率偏置审计 | `docs/verification_csf_curvature_bias.md` |
| Laplace 压差归因 | `docs/verification_laplace_pressure_floor.md` |
