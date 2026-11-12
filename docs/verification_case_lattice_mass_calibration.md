# 算例层晶格质量校准迁移（m16525）

**结论**：`Config.density_calibration` 自 m3211 起只在求解器路径
（`droplet_impact.make_particles`）生效，**算例层 6 个构建器静默忽略该标志**；
本单把标志在算例层落实，并用解析晶格偏置 `b(h/dx, geometry)` 逐算例门控。

## 1. 缺陷（可机器判定）

`rho_i = sum_j m_j W_ij`。规则晶格上 `m = rho0 dx^2`（2-D 方/六角）或 `rho0 dx^3`（3-D 立方）
时完整支撑粒子的求和密度为 `rho0(1+b)`，`b = m S/rho0 - 1`，只依赖 `(h/dx, 晶格几何)`；
Tait 律把它变成均匀虚假压强 `p = B[(1+b)^gamma - 1]`，与所解物理无关。算例层忽略标志
= 标志无效 + 初始态带偏。

**同时修正两处文档数字**（`initial_density.py` / `droplet_impact.py` 曾称「h=1.3dx 方晶格
b=+5.88e-3」）。实测：

| 晶格 | h/dx=1.3 | h/dx=1.5 |
|---|---|---|
| 方（2-D 生产晶格） | **-5.323e-05** | **+3.440e-03** |
| 六角（Laplace 液滴） | +5.883e-03 | +1.718e-03 |
| 立方（3-D） | -2.738e-03 | — |

+5.88e-3 是**六角**值（m3211 已指出，模块 docstring 未跟进）。方晶格在 1.3 略欠密
（钳位后不可见），在 1.5 过密 0.34 % → 1.35 kPa 虚假压强 @ c0=20 m/s。

## 2. 改动

| 文件 | 改动 |
|---|---|
| `src/tensorsph/dam_break.py` | `make_column` 用 `lattice_mass_unit(cfg)`，静水剖面按 `m_unit/rho0` 缩放 |
| `src/tensorsph/sloshing.py` | 同上（`make_water_block`） |
| `src/tensorsph/dam_break_3d.py` | 3-D 需 3-D 归一核：`threed.kernel3.calibrated_particle_mass` |
| `src/tensorsph/rayleigh_taylor.py` | 两相质量乘 `calibration_factor`（偏置会偏置增长率本量） |
| `src/tensorsph/ice_splash.py` | DEM 导出质量改用运行实际晶格质量单位（原 `rho0 dx^2`） |
| `src/tensorsph/initial_density.py` | 修正 b 的晶格归属；3-D 指到 `threed.kernel3` |
| `tests/test_dam_break.py` | 静水剖面期望改用校准质量单位 |

## 3. 实测（`runs/m16525/case_lattice_mass.json`，dx=0.02）

| 算例 | 标志 | 实测 b | 解析 b |
|---|---|---|---|
| dam_break @1.3 | off → on | -5.280e-05 → +3.05e-07 | -5.323e-05 |
| dam_break @1.5 | off → on | +3.440e-03 → +3.05e-07 | +3.440e-03 |
| sloshing @1.5 | off → on | +3.441e-03 → +1.04e-06 | +3.440e-03 |

质量相对变化：1.3 → **+5.3e-05**（已发布 B1/B2 数字不受影响），1.5 → **-3.4e-03**（被修掉的偏置）。

## 4. 验证

`PYTHONPATH=src pytest tests/test_case_lattice_mass_calibration.py tests/test_dam_break.py tests/test_initial_density_calibration.py -q` → **42 passed**。
新门：G1 实测 b==解析 b（|Δ|<1e-5，1.3/1.5）；G2 校准后 |b|<1e-5（dam_break+sloshing）；
G3 1.3 处质量变化 <1e-4；G4 3-D 用 3-D 核（与 2-D 核差 0.67 %）；G5 docstring 数字与计算一致；
G6 b 与 dx 无关（1e-12）。

## 5. 注意点 / 局限

- 静水初始化时求和密度含物理剖面，故 G1/G2 只在 `hydrostatic_init=False` 上量；
  剖面走「形状保持」门（mass = profile × 单位）。
- `src/tensorsph/wcsph/sod_shock.py` 用 **quintic 核**且显式 `density_calibration=False`：
  其偏置需 quintic 核晶格和，属另一实现，**本单未覆盖（诚实未完成）**。
- 未重跑 B2 三档分辨率 ladder（1.3 处变化 5.3e-5 ≪ front 收敛尺度 0.019 %）。

## 6. 误差判定（是否 <3%）

| 关键量 | 基准 | 误差 | <3% |
|---|---|---|---|
| 实测 b @1.3（方） | 解析 -5.323e-05 | 0.81 % | 是 |
| 实测 b @1.5（方） | 解析 +3.440e-03 | 0.01 % | 是 |
| 校准后残余偏置 | 0 | ≤1.0e-06 | 是 |
| 3-D 校准质量 | `threed.kernel3` 闭式 | 1.3e-08（float32） | 是 |
