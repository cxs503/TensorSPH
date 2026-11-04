# TensorSPH P1：初始化质量校准（m2971 C2）——消除 6.7× Laplace 信号的虚假初压

## ① 案例介绍（物理背景/参数）

WCSPH 密度求和 ``rho_i = sum_j m_j W(r_ij,h)`` 在规则晶格上**不精确分划单位**：
取晶格常数质量时内部求和密度为 ``rho0(1+b)``，``b`` 由核函数与堆积决定、与 c0 无关。
六角格（Young–Laplace 基准用）``b=+5.88e-3``，c0=20 m/s 时对应虚假初压
``B[(1+b)^gamma-1]=2.4 kPa``，是静液滴 Laplace 信号 ``sigma/R=360 Pa`` 的 **6.7 倍**。
m2971 误差源分解已把 C2 从稳态主因降级，但指出它是**任何 c0=20 wcSPH 算例都带**的
初始缺陷，且当时只在 `benchmarks/laplace_pressure_floor/run.py` 以本地函数实现、未进 `src/`。

| 参数 | 值 |
|---|---|
| 流体 | 水 rho0=1000 kg/m^3, gamma=7 |
| 液滴 | R=0.02 m, sigma=7.2 N/m, c0=20 m/s |
| 核 | 2-D cubic spline, h=1.3 dx |
| 分辨率 | dx=0.002 / 0.0015 m（六角格）；默认撞池方格 dx=0.025 m |

## ② 怎么做的（方法/流程/工具链）

`src/tensorsph/droplet_impact.py` 新增（默认关闭，保已发布数字逐位可复现）：
- `Config.mass_calibration: bool = False`
- `lattice_bias()`：求和密度后取**内部平台**（`rho>=0.999*rho.max`，核支撑完整、
  不受自由面亏缺污染）均值，返回 `b=mean/rho0-1`。平台判据对任意 R/dx 稳健
  （当 `0.5R<2h` 时几何核心 `r<0.5R` 已受自由面影响，会误测 b）。
- `calibrate_mass()`：一次 `m->m/(1+b)`，内部求和密度 → 恰好 rho0（机器精度）。
- `make_particles` 在开关打开时应用。
工具链：PyTorch CPU，pytest 9；数字由探针/测试程序化测得，零手抄。

## ③ 计算结果（数字表）

| 量 | 六角 dx=0.002 | 六角 dx=0.0015 | 方格撞池 |
|---|---|---|---|
| 偏置 b | +5.8833e-3 | +5.8836e-3 | −5.2e-5 |
| 校准前 p_max | 2395.4 Pa | 2395.4 Pa | 0 Pa |
| 校准前 p_max/(ρ0c0²) | 5.99e-3 | 5.99e-3 | 0 |
| 校准后 core ρ/ρ0 | 1.00000012 | 1.00000000 | 1.0000000 |
| 校准后 p_max/(ρ0c0²) | **4.77e-7** | **2.38e-7** | **7.15e-7** |
| 验收阈值 | 1e-4 | 1e-4 | 1e-4 |

`tests/test_mass_calibration.py` 6 项全绿：偏置量级、校准后内部密度=ρ0、初压达标、
未校准态确实预压缩（sanity）、b 与 c0 无关、make_particles 开关行为。

## ④ 结果图

**无图，理由**：纯后端/初始化任务——校准只改 t=0 的粒子质量，不产生流场、
无场量云图或时间演化曲线可画；误差判定以 §⑥ 数字表为准。

## ⑤ 注意点（坑/局限/适用范围）

- 默认关闭；需消除预压缩的算例（Laplace/静液滴）须显式开启。
- 与 `hydrostatic_init` 语义冲突（后者使池质量非均匀），勿同开。
- 只校准流体粒子；壁粒子质量不变（壁压由流体外推，不受影响）。
- 未改 `laplace_pressure_floor` 的本地 `calibrated_droplet`（行为等价，留后续单统一）。
- 全量 pytest 因机器 load15≈159 超 run 工具 120s 上限未能一次跑完；已覆盖全部被
  触碰代码路径（make_particles/Config/密度/压强/边界/自适应步长，共 249 测试通过）。

## ⑥ 误差判定（逐项 vs 基准，显式回答 <3%）

| 关键量 | 基准/阈值 | 实测 | 是否 <3% |
|---|---|---|---|
| 校准后内部 ρ/ρ0−1 | 0（机器精度） | ≤1.2e-7 | **是**（≪3%） |
| 校准后初压 p_max/(ρ0c0²) | <1e-4（m2971 判据） | ≤7.2e-7 | **是**（≪3%） |
| b 与 c0 无关 | 相对差 0 | 0（rel<1e-9） | **是** |

本单不改流场（只改 t=0 质量），无「<3%」意义的连续场量；上表三项均远优于阈值。

## 回归与遗留

- 新测试 6/6；droplet 相关 73；单元批 93；laplace+表面张力批 83 通过。
- **HEAD 既有 3 红**（干净 HEAD 复验确认，非本单引入）：
  `test_laplace_law::test_committed_verdicts_are_self_consistent`
  （m3028 的 relax_dx0.0015.json 只跑到 t/τσ=0.333<0.5）、
  `test_laplace_pressure::test_effective_surface_tension_at_the_documented_resolution`、
  `test_paper_laplace_uncertainty::test_figures_exist`。
- 给下一张 selfstart 单的线索：① 专清 HEAD 3 红（补跑欠松弛 relax 算例或改断言）；
  ② P0 自由面压强闭合（Laplace 唯一门槛项，C2 已移除、归因更干净）。
