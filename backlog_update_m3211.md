# backlog 更新（m3211：晶格离散平衡偏置的根因与校准）

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ 改动：`src/tensorsph/droplet_impact.py` + 新增 benchmark/tests/docs

## 已结清：m2971 P1「初始化质量校准」（并纠正其普适性表述）

`backlog_update_m2971.md` 的 P1 断言「任何 wcSPH 算例在 c0=20 都带 2.4 kPa 初始虚假压强」。
本单把它测成曲线（`benchmarks/lattice_equilibrium/report.json`、`docs/verification_lattice_mass_calibration.md`）：

| 观测 | 结果 |
|---|---|
| 偏置来源 | 纯几何：`b = (晶格质量)·Σ W − 1`，只依赖格点类型与 `h/dx`，与 `c0` 无关 |
| 正方形（生产）@ h/dx=1.3 | `b = −5.3e-5`（Tait 钳位抹掉），**不是** 2.4 kPa |
| 六角（B3 液滴）@ h/dx=1.3 | `b = +5.88e-3` → 2.4 kPa（与 `test_hex_lattice_sum_density_bias_is_systematic` 四位互校） |
| 正方形 @ h/dx=1.5 | `b = +3.44e-3` → **1.39 kPa**；零外力静止水池自生 **0.096 m/s** |
| 校准后（流体+壁面同单位） | 8 个算例 `p_max/(rho0 c0^2) < 1e-4`；hydrostatic t=0 RMS 5.4e-5 → 6.7e-7 |

修复：`calibrated_particle_mass` / `lattice_mass_unit` + `Config.density_calibration`（默认 True）。
默认 `h/dx = 1.3` 下质量只变 5.3e-5，既有算例与门限**无需重定基线**。

## 新 P2

1. `dam_break.py::make_column` 仍用未校准质量（B2 实验门限按现质量标定）；同法校准后需重跑 B2 并记录差异。
2. L4 的「零外力静止水池自生流速」是个好用的启动伪流探针，建议作为后续任何自由面/压强闭合改动的冒烟判据。

## 排期不变

- P0: 自由面压强闭合（`benchmarks/laplace_pressure_floor` 的 C3）——本单与之正交（C2 稳态仅 ±9 pt）。
- 3D 扩展、多相 SPH、dam-break 收敛复现、实验数据对照。
