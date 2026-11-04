# backlog 更新（m3276：P1 初始化质量校准结清）

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动类型**: `src/` 新增能力（默认关）+ tests + docs

## 已结清：P1 初始化质量校准（m2971 C2）

m2971 账本的 P1 已实现并提升进 `src/tensorsph/droplet_impact.py`：

- `Config.mass_calibration: bool = False`（默认关，保已发布数字可复现）
- `lattice_bias(...)`：内部平台（`rho >= 0.999*rho.max()`）测 b
- `calibrate_mass(...)`：一次 `m -> m/(1+b)`，内部求和密度 → 恰好 rho0
- `make_particles` 在开关打开时应用

验收（m2971 判据，机器可裁）：初始 `p_max/(rho0 c0^2) < 1e-4`
→ 六角格实测 2.4e-7…4.8e-7（校准前 5.99e-3，即 2.4 kPa = 6.7× Laplace 信号）。
工件：`tests/test_mass_calibration.py`（6 passed）、
`docs/verification_mass_calibration.md`。

## 未动 / 留给后续单

- **P0 自由面压强闭合**（Laplace 唯一门槛项）仍未做：需 rhs 加 shell-pressure
  外推（Adami/Ihmsen/Adams 2012 §3.3），验收=laplace_pressure_floor 四点
  `absolute_time_average_tail` <15%。本单已把 C2 从门槛里移除，P0 现在是
  纯 C3 问题，难度未变但归因更干净。
- **HEAD 既有 3 红**（干净 HEAD 复验确认）：
  1. `test_laplace_law::test_committed_verdicts_are_self_consistent`
     —— m3028 提交的 `relax_dx0.0015.json` 只跑到 t/τσ=0.333（欠松弛），
     断言 `>0.5` 不成立。修法二选一：补跑该 relax 算例到 ≥0.5τσ（短算例），
     或把断言改为记录当前状态。
  2. `test_laplace_pressure::test_effective_surface_tension_at_the_documented_resolution`
  3. `test_paper_laplace_uncertainty::test_figures_exist`（缺图文件）
  三者均与 P1 无关，建议下一张 selfstart 单专清「HEAD 红测试」。
- P2 验证口径统一：后续 Laplace 验证一律「绝对压强+时间平均」。

## 环境踩坑

- run 工具 120s 上限 + 机器 load15≈159：全量 pytest（~90s@load118）会超时且
  进程组被整组收割（`&`/`setsid` 均不能幸免）；重测试分批跑（每批 <115s）。
