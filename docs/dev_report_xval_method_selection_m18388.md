# XVAL-SPH-METHOD-SELECTION-2026-11-12: G2 Reference Verification & bl_route_check Repair

**Anchor**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`  
**Mission**: m18388  
**Date**: 2026-11-13  

## ① 问题陈述

方法选型（`oss_solutions` id=408）将波槽衰减归因锚在层流底边界层阻尼系数
`μ = σ²√(2νσ) / (4 g c_g sinh²kh)` 上。诊断脚本 `benchmarks/xval_method_selection/bl_route_check.py`
本应从 Stokes 边界层数值积分独立复核该闭式，但存在两处缺陷：

1. **`mu_from_dissipation` 缺因子 2**：能量通量 `F = (1/2)ρga²c_g`，当 `a = a₀exp(-μx)` 时
   `dF/dx = -ρga²c_g μ = -D`，故 `μ = D/(ρga²c_g)`。原代码用 `E = (1/8)gH²`（波能）但漏了
   `d(a²)/dx = -2μa²` 的因子 2，导致 μ 被低估 2 倍。
2. **`closed_form_dissipation` 缺 √2**：耗散应为 `D = (1/4)u_b²√(2νω)`，原代码用 `√(νω)`，
   即经典的 "missing sqrt(2)" 转录错误（低 41%）。

## ② 修复

### 2.1 `mu_from_dissipation` 修正

```python
# Before (wrong):
energy = 0.125 * GRAVITY * height * height  # (1/8) g H² = (1/2) g a²
# After (correct):
energy_equiv = 0.25 * GRAVITY * height * height  # g a² = (1/4) g H²
```

推导：`μ = D / (g a² c_g)`，其中 `a = H/2`，故 `g a² = (1/4) g H²`。

### 2.2 `closed_form_dissipation` 修正

```python
# Before (wrong):
return 0.25 * u_b * u_b * math.sqrt(nu * omega)
# After (correct):
return 0.25 * u_b * u_b * math.sqrt(2.0 * nu * omega)
```

## ③ 验证结果

修复后三路线完全一致（T=1.0s, h=0.25m, H=0.04m, ν=1e-6）：

| 路线 | D [m³/s³] | μ [1/m] | 相对 repo 误差 |
|---|---|---|---|
| repo 闭式 | 6.0727e-06 | 1.6525e-03 | 0.00% |
| 重建闭式 | 6.0727e-06 | 1.6525e-03 | 0.00% |
| Stokes 积分 | 6.0726e-06 | 1.6524e-03 | +0.00% |

- 数值求积收敛：nz 200→800、nt 128→512 时 μ 相对变化 +2.50e-04
- **repo 闭式与 Stokes 积分一致（ratio = 1.0000，within 15% gate: True）**

## ④ 对方法选型的意义

G2 衰减参考值 `μ = 1.65e-03 1/m` 经独立数值积分验证正确。对 2 m 传播段的物理衰减预算为
`1 - exp(-μ·2) ≈ 0.33%`，远低于 10% 判据门——实测衰减若超过此值，主要来源为数值耗散而非物理阻尼。

## ⑤ 局限

- 波槽实测记录（3 个 repair cell）已递交集群（cj1909），尚在排队（282 ahead），未取回
- 本报告仅覆盖 G2 参考式验证；G1 波高对标需实测记录
- 2-D 顺直渠、规则波、H/h = 0.16、kh = 1.20（过渡水深）；不外推到破碎波/三维/强非线性

## ⑥ 工件

- `benchmarks/xval_method_selection/bl_route_check.py`（修复后）
- `docs/reports/xval_method_selection_bl_route.json`（更新后数值）
- `tests/test_wave_flume_damping_literature.py`（7 项全通过，独立验证 repo 闭式）
