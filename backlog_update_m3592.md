# m3592：初始化格点密度偏置（P1/C2）结清 + 后续选题

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动类型**: `src/`（1 个文件）+ benchmark/docs/tests

## 已结清：backlog_update_m2971.md 的 P1｜初始化质量校准（C2）

`make_particles` 现在默认做一次**测量的**质量校准：`m → m·rho0/rho_measured`
（满支撑粒子上的核和），墙体虚粒子质量同因子跟随。

| 项 | 结果 |
|---|---|
| 核和实测 vs 解析格点求和 | 6 个算例最大相对误差 **3.05e-7 %**（float32 舍入级） |
| 六边形格点偏置 b | **+5.8836e-3**（dx = 20/10/5 mm 恒定 → 格点性质，不随加密收敛） |
| 正方形格点偏置 b | **−5.323e-5**（冲击算例，p 被 Tait 截断在 0） |
| 六边形 t=0 虚假压强 | 2395 Pa @ c0=20（Laplace 360 Pa 的 6.66 倍）→ 校准后 < 1 Pa |
| P1 验收判据 `p_max/(rho0 c0²) < 1e-4` | 4 个配置全过，最大 8.3e-7 |
| 冲击算例 750 步 A/B | 5 项判据全 pass，差异 < 0.4 % |

工件：`benchmarks/init_density_calibration/`（`run.py`/`make_doc.py`/`report.json`/
`init_density_calibration.png`，16 s 全流程）、`docs/verification_init_density_calibration.md`、
`tests/test_init_density_calibration.py`（16 passed，判据全部锚在解析核和上，
另含「幂等」「全局缩放」「墙质量跟随」与「第二物体不污染最大核和」四条反回归测试）。

**新增 API**：`kernel_consistency_scale` / `calibrate_initial_mass` / `full_support_mask`，
`Config.init_density_calibration`（默认 True）、`Config.init_density_scale`（派生量，写入
`run_manifest.json` 作校准出生证明）。

## 重要更正（对本仓历史结论）

m2971 记录的核心区偏置 `rho/rho0 − 1 = +0.00588` 与本次解析值 `+5.8835e-3` 一致，
**该数正确**；但本单第一版按 `r < 0.5R` 取核心读出的 `−5.45e-4` 是错的
（`2h = 2.6dx = 0.065 m > 0.5R = 0.05 m`，核心本身被截断）。正确值 `−5.32e-5`
与解析 `−5.323e-5` 一致。**凡引用「核心区」数值必须同时给出判据半径与 2h 的关系**。

## 新 P0（未变）：自由面压强闭合

仍是 Laplace 压差 WITHHELD 的唯一门槛项（m2971 已给出判据与实现方案）。

## 新 P1（本单腾出的第一选题）：六边形验证基准的 A/B 复核

`benchmarks/laplace_droplet`、`laplace_law`、`oscillating_droplet` 的格点初始化器仍在
harness 内部、未用本单工具，因此它们当前**带着 +0.588 % 初始预压**运行：

1. 把各自的 `droplet_lattice`/`hexagonal_droplet` 接到 `calibrate_initial_mass`；
2. 重跑弛豫链，A/B `dp_ratio`、`kappa`、伪流 `max|v|/sqrt(sigma/rhoR)`；
3. 预期：伪流指标改善（瞬态源被移除），稳态 Δp 不变（残差仍归 P0）。
   若稳态 Δp 发生变化，说明此前的读数里混着 C2 的贡献 —— 需回写 m2971 的四源分解表。
4. 单算例数十秒至分钟级，需在 load15 < 100 时做。

## 新 P2：`h/dx` 选择指导

`h/dx = 1.5` 时正方形格点偏置变成 **+3.44e-3**（符号翻转），六边形反而降到 +1.7e-3。
若将来放宽 `h_factor`，`init_density_calibration` 必须保持默认开启；解析表见
`docs/verification_init_density_calibration.md` §3.1。