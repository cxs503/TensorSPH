# 初始化质量校准（P1 / m2971 C2）验证记录

**Mission**: m3276 ｜ **日期**: 2026-11-04 ｜ **范围**: TensorSPH
**改动类型**: `src/` 新增能力（默认关闭）+ 测试 + 本文档；未改任何现有代码路径。

## 1. 案例介绍（物理背景）

WCSPH 用密度求和 ``rho_i = sum_j m_j W(r_ij, h)``。规则晶格的离散核和**不精确
等于 1**：取晶格常数质量 ``m = rho0 dx^2``（方格）或 ``rho0 dx dy``（六角格）时，
内部粒子求和密度为 ``rho0 (1 + b)``，``b`` 由核函数与堆积方式决定，与 c0 无关。

- 六角格（Young–Laplace 基准用）：``b = +5.88e-3``，内部**预压缩**。
  c0 = 20 m/s 时对应虚假初压 ``B[(1+b)^gamma - 1] = 2.4 kPa``，
  是静液滴 Laplace 信号 ``sigma/R = 360 Pa`` 的 **6.7 倍**。
- 方格（默认液滴撞池）：``b = -5.2e-5``，可忽略（Tait 钳 0 后 p_max = 0）。

m2971 的误差源分解把 C2（初始化偏置）从「稳态主因」降级，但指出它是**任何
c0=20 wcSPH 算例都带的 2.4 kPa 初始虚假压强**，且当时只在
`benchmarks/laplace_pressure_floor/run.py` 里以本地函数 `calibrated_droplet`
实现，**未进 `src/`**。本单把它提升为 `src/` 通用能力。

## 2. 怎么做的（方法/流程）

`src/tensorsph/droplet_impact.py` 新增：

- `Config.mass_calibration: bool = False`（默认关，保证已发布数字逐位可复现）。
- `lattice_bias(positions, mass, cfg, device) -> float`：求和密度后取
  **内部平台**（`rho >= 0.999 * rho.max()`，核支撑完整、不受自由面亏缺污染）
  的均值，返回 `b = mean(rho_plateau)/rho0 - 1`。
- `calibrate_mass(positions, mass, cfg, device)`：一次 `m -> m/(1+b)`，
  把内部求和密度校准到恰好 `rho0`（机器精度）。
- `make_particles` 在 `cfg.mass_calibration` 为真时调用 `calibrate_mass`。

判据先于动手（机器可裁）：校准后初始 `p_max / (rho0 c0^2) < 1e-4`。

## 3. 计算结果（数字由探针/测试程序化测得，零手抄）

| 量 | 六角格 dx=0.002 | 六角格 dx=0.0015 | 方格（默认撞池） |
|---|---|---|---|
| 偏置 b（内部平台） | +5.8833e-3 | +5.8836e-3 | −5.2e-5 |
| 校准前 p_max | 2395.4 Pa | 2395.4 Pa | 0 Pa |
| 校准前 p_max/(ρ0c0²) | 5.99e-3 | 5.99e-3 | 0 |
| 校准后 core ρ/ρ0 | 1.00000012 | 1.00000000 | 1.0000000 |
| 校准后 p_max/(ρ0c0²) | **4.77e-7** | **2.38e-7** | **7.15e-7** |
| 验收阈值 | 1e-4 | 1e-4 | 1e-4 |

`tests/test_mass_calibration.py`（6 项，全绿）钉住：偏置量级、校准后内部密度
=ρ0、初压达标、未校准态确实预压缩（sanity）、b 与 c0 无关、`make_particles`
开关行为（关=质量不变，开=内部=ρ0 且初压达标）。

## 4. 结果图

纯后端/初始化任务，无场量云图或时间演化曲线可画（校准只改 t=0 的粒子质量，
不产生流场）。误差判定见 §6，以数字表为准。

## 5. 注意点（坑/局限/适用范围）

- **默认关闭**：为保持已发布数字可复现，`mass_calibration` 默认 False。
  需要消除预压缩的算例（如 Laplace/静液滴）需显式开启。
- 平台判据用「内部平台」（核支撑完整区），**不是**几何核心 `r<0.5R`：
  当 `0.5R < 2h` 时（如默认撞池 R=0.10, 2h=0.065）几何核心已受自由面影响，
  会低估/高估 b。平台判据对任意 R/dx 稳健。
- 与 `hydrostatic_init` 语义冲突（后者使池质量非均匀），二者不要同开。
- 只校准流体粒子；壁粒子质量仍为 `rho0 dx^2`（壁压由流体外推，不受影响）。
- 本单**未**改 `benchmarks/laplace_pressure_floor/run.py` 的本地
  `calibrated_droplet`（行为等价，留待后续单统一，避免动已合入基准）。

## 6. 误差判定（逐项 vs 基准，显式回答 <3%）

| 关键量 | 基准/阈值 | 实测 | 是否 <3% |
|---|---|---|---|
| 校准后内部 ρ/ρ0 − 1 | 0（机器精度） | ≤1.2e-7 | 是（≪3%） |
| 校准后初压 p_max/(ρ0c0²) | <1e-4（m2971 判据） | ≤7.2e-7 | 是（≪3%） |
| 偏置 b 与 c0 无关 | 相对差 0 | 0（rel=1e-9 内） | 是 |

无「<3%」意义的连续场量（本单不改流场，只改 t=0 质量）；上表三项均远优于阈值。

## 7. 回归

- 新测试 6/6 通过；droplet 相关 7 文件 73 通过；单元批 93 通过；
  laplace+表面张力批 83 通过。
- 全量套件因机器 load15≈159 超出 run 工具 120s 上限未能一次跑完；
  已覆盖全部被触碰代码路径（`make_particles`/`Config`/密度/压强/边界/自适应步长）。
- **HEAD 既有 3 红**（干净 HEAD 复验确认，非本单引入）：
  `test_laplace_law::test_committed_verdicts_are_self_consistent`
  （relax_dx0.0015.json t/τσ=0.333<0.5）、
  `test_laplace_pressure::test_effective_surface_tension_at_the_documented_resolution`、
  `test_paper_laplace_uncertainty::test_figures_exist`。
