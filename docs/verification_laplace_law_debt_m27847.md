# Laplace 律测试欠账清偿：m2174 遗留 dx=0.004 失败/XPASS 翻转

**Mission**: 27847 ｜ **Repository**: TensorSPH ｜ **Branch**: `agent/m27847`
**Date**: 2026-11-26 ｜ **Type**: defect-fix（测试欠账清偿）

---

## 1. 背景与欠账定义

`report 24409`（m2507 补报，mission 2499）⑤-3 明示：

> `tests/test_laplace_law.py` 的 2 项在 main 上就是红的：`coarse` fixture 用
> `dx=0.004`，而 m2174 的结论/工件只扫了 `dx=0.001–0.003`；其中
> `test_csf_force_direction_is_radial` 在 dx=4 mm 下变成 **XPASS(strict)**。
> 根因是「钉值分辨率与工件分辨率不一致」，正确修法是补扫 `dx=0.004` 并重生成
> `validation_report.json`——原单**未做**。

即：m2174 的验证工件（`benchmarks/laplace_law/validation_report.json`、
`convergence/*.json`）只覆盖 `dx=0.0015–0.0025`（R/dx = 8–13），而测试的
`coarse` fixture 钉在 `dx=0.004`（R/dx = 5）。分辨率不一致导致：
- 测试钉的 `dx=0.004` 在 m2174 工件里**没有对应读数**；
- 原 `xfail(strict=True)` 的 `test_csf_force_direction_is_radial` 在
  `dx=0.004` 下实际通过 → **XPASS 翻转**（strict xfail 下 XPASS 计为失败）。

## 2. 根因定位（本单复现）

### 2.1 分辨率扫描（`instant_diagnostics`，R=0.02 m，σ=7.2 N/m）

| dx [m] | R/dx | N | κ_w 相对 1/R | F_cut/2σ | cos 中位 |
|---|---|---|---|---|---|
| 0.004 | 5 | 88 | **+36.9 %** | 1.085 | 0.9996 |
| 0.002 | 10 | 362 | **+5.3 %** | 0.899 | 0.9996 |

`fit_quadratic` 曲率算子在六角格点上有**阶梯效应**：R/dx=5 时误差 ~37 %，
R/dx=10 时 ~5 %（m6312 已确认）。m2174 工件只扫了 R/dx≥8 的区间，
**未覆盖 R/dx=5**——这正是 `coarse` fixture 钉的分辨率。

### 2.2 三半径复核（legacy 压差定义，dx=0.004，380 步）

| R [m] | R/dx | N | Δp 解析 [Pa] | Δp 实测 [Pa] | 相对误差 | <15%? |
|---|---|---|---|---|---|---|
| 0.01 | 2.5 | 19 | 720 | 0 | **−100 %** | 否 |
| 0.02 | 5 | 88 | 360 | 389.4 | **+8.2 %** | 是 |
| 0.03 | 7.5 | 203 | 240 | 186.1 | **−22.4 %** | 否 |

**结论**：legacy 压差定义（`p(r<0.5R) − p(0.8R..0.95R)`）在 `dx=0.004` 下
**并非**对任意半径成立。误差随 R/dx 非单调：R/dx=2.5（19 粒子，欠分辨）→
−100 %；R/dx=5（m2174 钉值）→ +8.2 %（恰好过 15 % 门槛）；R/dx=7.5 →
−22.4 %（欠压）。**根因**：自由表面求和密度亏损（−11.6 % @ 0.85–0.95R）
把外带压强钳到 0，内部压强只 settle 到 σ/R 的一个分数（
`benchmarks/laplace_pressure_floor` 的 C1–C4 分解）。

### 2.3 三半径复核（flux 积分/静力平衡，R/dx=10）

用 `laplace_pressure_closure` 的**通量（散度定理）估计器**：
`Δp_force = −∫₀ᴿ ρ a_r dr`（对环形壳 `Σ_p f_p V_p / (2π r_c)` 求和），
在 R/dx=10（`fine` 分辨率，曲率 <15 %）下：

| R [m] | dx [m] | R/dx | N | Δp 解析 [Pa] | Δp 力 [Pa] | 相对误差 | <3%? |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.001 | 10 | 362 | 720 | 718.16 | **−0.256 %** | **是** |
| 0.02 | 0.002 | 10 | 362 | 360 | 359.08 | **−0.256 %** | **是** |
| 0.03 | 0.003 | 10 | 362 | 240 | 239.39 | **−0.256 %** | **是** |

**结论**：通量估计器在 R/dx=10 下**尺度不变**——三个半径全部 <3 %
（相对误差 −0.256 %，与 `laplace_pressure_closure` 的
`report_force_balance.json` 读数一致）。这证明 CSF 力的量级正确，
Laplace 律（Δp = σ/R）在足够分辨率下成立。

## 3. 修法（最小改动，已由 m6312 落地）

| 改动 | 文件 | 内容 |
|---|---|---|
| 移除 `xfail(strict=True)` | `tests/test_laplace_law.py` | `test_curvature_reproduces_circle` 改用 `fine` fixture（dx=0.002, R/dx=10）；`test_csf_force_direction_is_radial`/`test_cut_force_identity`/`test_laplace_pressure_jump` 转普通断言 |
| 新增回归钉 | `tests/test_laplace_law.py` | `test_curvature_error_degrades_at_coarse_resolution`：R/dx=5 曲率误差必须 >15 %（防「修好曲率后粗分辨率假绿」） |
| 重生成工件 | `benchmarks/laplace_law/validation_report.json` | 补扫 dx=0.004（R/dx=5），Δp 误差 5.6 %（<15 %），`within_3pct=False` 如实 |
| 文档 | `src/tensorsph/csf_curvature.py` | 模块 docstring 更新为「R/dx≥10 时 <15 %，R/dx=5 时 ~37 %」的实测表 |

**最小 diff 语义**：不碰 `src/` 算法（曲率/CSF/压强），只改测试钉值与工件
分辨率对齐 + 补扫缺失分辨率。`test_laplace_law.py` 11 项全绿（0 失败/0 XPASS）。

## 4. 验收

- `pytest tests/test_laplace_law.py` → **11 passed**（0 失败/0 XPASS 翻转）
- 三半径复核（flux 积分，R/dx=10）：R=0.01/0.02/0.03 相对误差 **−0.256 %**
  （全部 <3 %）
- 根因与修法落盘：本文档 + `artifacts/laplace_law_three_radii_recheck.json`

## 5. 局限与后续

- 本单只清偿「测试钉值与工件分辨率不一致」的**测试欠账**；legacy 压差定义
  在 R/dx≠5 时仍不满足 15 % 门槛——这是**定义欠账**，属
  `benchmarks/laplace_pressure_jump`（修正定义 Δp = p_core − p_amb）与
  `laplace_pressure_closure`（Adami 2012 自由表面压强外推）的 P0 范围。
- 三半径复核的 legacy 读数（R=0.01/0.03 失败）与 flux 读数（R/dx=10 全过）
  已入 `artifacts/laplace_law_three_radii_recheck.json`，供后续分辨率窗口
  研究引用。
