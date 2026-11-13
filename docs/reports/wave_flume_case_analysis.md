# CASE-WAVE-FLUME 根因分析与改进计划

**日期**: 2026-11-13  
**分析员**: agent/m17154  
**锚点**: CASE-WAVE-FLUME（算例池·规则波生成）  
**判据**: `|H_got - H_ref| / H_ref ≤ 10 %`

## 1. 现状诊断

### 1.1 已有证据（wf_D16.json，H/dx=2.56）

| 指标 | 值 | 判据 | 状态 |
|---|---|---|---|
| 最差波高误差 | +65.47 % | ≤ 10 % | ❌ |
| 反射系数 | 0.824 | 应 < 0.1 | ❌ |
| 波长误差 | 757 % | 应 < 5 % | ❌ |
| μ_meas / μ_theory | -984 / 0.0012 | 应 ≈ 1 | ❌ |

**关键发现**：82% 的波被反射，形成驻波图案，导致下游波高异常增大（从 0.039 增至 0.066）。

### 1.2 根因分析

#### 海绵层吸收效率不足

当前海绵层参数：
- `sponge_length = 0.9 m`（≈ 0.69λ，文献建议 ≥ 1-2λ）
- `sponge_strength = 4.0` → `alpha_max = 5.8 1/s`
- `sponge_exponent = 3.0`（幂次 ramp）

**问题**：海绵仅阻尼速度 `v ← v(1 - α dt)`，不阻尼自由面高程扰动。当波进入海绵层：
1. 速度被阻尼 → 动能被移除
2. 但表面高程扰动保留 → 势能未移除
3. 动能/势能失衡 → 阻抗失配 → 反射

理论累积衰减：`exp(-α_avg × t_transit) = exp(-2.9 × 0.69) ≈ 0.135`，即 86.5% 吸收。
但实测反射 82%，说明速度阻尼 alone 不足以有效吸收波能。

#### 分辨率不足

H/dx = 2.56（波高方向仅 2.56 个粒子），远低于 SPH 自由面分辨率的常规要求（H/dx ≥ 5-10）。

#### 无启停瞬态抑制

wf_D16 未设置 `settle_time`，活塞从 t=0 开始运动，初始瞬态污染了整个波场。

## 2. 改进措施

### 2.1 已提交集群的工况

| 作业号 | 工况 | H/dx | settle | xsph_settle | 预期改善 |
|---|---|---|---|---|---|
| cj1432 | wf_D20 | 3.2 | 1.0s | 0.1 | 分辨率提升 + 瞬态抑制 |
| cj1433 | wf_D32 | 5.12 | 1.0s | 0.1 | 更高分辨率 |

### 2.2 代码改进（待实施）

1. **海绵层增强**：增加密度/压力扰动阻尼项
   ```python
   # 当前：仅阻尼速度
   velocity = velocity * (1.0 - alpha * dt)
   
   # 改进：同时阻尼密度扰动
   density = rho0 + (density - rho0) * (1.0 - alpha * density_damping_factor * dt)
   ```

2. **海绵长度自适应**：根据波长自动设置
   ```python
   sponge_length = max(1.5 * wavelength, 0.9)  # 至少 1.5 个波长
   ```

3. **主动吸收**：启用 `active_absorption` 参数（代码已实现，默认 0.0）
   - 浅水阻抗关系：`v_corr = -gain * sqrt(g/h) * (eta_meas - eta_target)`
   - 建议增益：0.5-0.8

### 2.3 测试改进

- 增加海绵吸收效率的单元测试（合成信号验证）
- 增加反射系数的验收测试（R < 0.1）

## 3. 预期结果

| 指标 | 当前 | 预期（D32 + 改进） |
|---|---|---|
| 波高误差 | 65 % | < 10 % |
| 反射系数 | 0.82 | < 0.15 |
| 波长误差 | 757 % | < 5 % |

## 4. 工件清单

- [x] 集群作业 cj1432（wf_D20.json）
- [x] 集群作业 cj1433（wf_D32.json）
- [ ] 海绵层增强代码（待集群结果确认后实施）
- [ ] 更新 `docs/verification_wave_flume.md`
- [ ] 更新 `docs/reports/wave_flume_case.md`

## 5. 参考文献

- Dean & Dalrymple 1991, *Water Wave Mechanics for Engineers and Scientists*, eq. 6.38
- Mei 1983, *The Applied Dynamics of Ocean Surface Waves*, §4.2
- Putnam & Johnson 1949, *Trans. AGU* 30:295
