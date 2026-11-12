# 交差核验审计 · m16239 · CASE-SPH-STOKES-2ND-PROBLEM（TensorSPH）

- 审计单：m16324（board-audit），审计者=独立席位（未参与 m16239 实现）
- 被审计对象：mission 16239 `succeeded`，repo TensorSPH，锚点 `CASE-SPH-STOKES-2ND-PROBLEM`
- 被审计工作树：`/root/tp-agent/work/m16239`，HEAD `b0b8d6a`；同日分支已推 origin `refs/heads/agent/m16239` = `5ffc9f5`
- 基准判据来源：`tensor_case_pool` id=126 `SPH-STOKES-2ND-PROBLEM`
  tol_policy = `<5% | 指标: 速度振幅随壁距衰减 exp(-y/delta) vs 解析 (Stokes 层内 >=5 层粒子)`
- 机器裁判表：`tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-STOKES-2ND-PROBLEM'` → **0 行**（该表全表 27 个 anchor，无本锚点）→ 无机器裁判行，改以 case_pool 126 的 tol_policy 裁定（与 m16237 同类平台缺口）
- 审计产物：`scripts/audit_m16239_stokes_second.py`（只读复算器）→ `runs/audit_m16239/audit_facts.json`；图 `docs/figures/m16324_stokes2_amplitude_decay.png`

## 0. 裁定（一句话）

**L4**：真实 WCSPH 求解 + 逐行振幅误差分析 + 文档化可执行复现命令三件套**实质存在**，判据量 `L2 = 0.0707%` 远优于 5%（70 倍余量），且经本审计从原始分级振幅**独立复算逐位一致**；但交付里存在 **1 项高档证据缺陷（收敛阶梯只提交 1 档、G6 判据在 n=1 下空转）+ 报告/文档未产出 + 1 个错误断言测试**，故同时列出返修项（gate_state 由 maturity_gate 机器裁定）。

## 1. 三件套取证（逐项核，不读结论读文件）

| 三件套 | 是否真存在 | 真实指针 |
|---|---|---|
| ① 实现代码（真算） | 存在 | 库 `src/tensorsph/stokes_second.py:96` `semi_infinite` / `:106` `semi_infinite_amplitude` / `:125` `finite_depth_amplitude` / `:144` `harmonic_fit` / `:186` `verify_symbolic`（sympy 重新推导，非抄写）/ `:244` `verify_numeric`；算例驱动 `benchmarks/stokes_second/run.py:138` `drive`（真时间推进）/ `:327` `evaluate` / `:436` `run_one` |
| ② run_artifacts | **部分存在（有缺陷，F1/F2）** | 工作树 `benchmarks/stokes_second/out/results_ladder.json`（sha256 见 facts）、`results_probe_dx8_3p5p.json`、`results_sensitivity.json`、`amplitude_decay.png`、`field_snapshot.png`；**但 origin 分支 `5ffc9f5` 里 ladder/sensitivity/2 张 png 被门禁滤除（commit 自述「4 个文件被门禁滤除」），仅 `results_probe_dx8_3p5p.json` 入库** |
| ③ repro_cmd | 存在且可执行（实跑过） | `PYTHONPATH=.:src python benchmarks/stokes_second/run.py --probe 8 --periods 5 --measure-periods 2 --c0 4 --out benchmarks/stokes_second/out`（`DESIGN.md:103-106` 第 7 节 Environment and reproducibility）；本审计另行以默认参数独立重跑（见 §4） |

## 2. 基准裁判：got vs ref（唯一裁判=判据，不采信报告）

复算器用 `A(y)=U0·exp(-y/δ)` 直接从**原始逐行 |u|**（`runs[0].rows[*].amplitude`）重算误差，绕开作者自带的 `rel_err_half_space` 字段（防解析器自身出错掩盖）：

| 判据项 | 要求 | 实测（独立复算） | 作者自报 | 一致? | 判定 |
|---|---|---|---|---|---|
| Stokes 层内振幅 L2 相对误差 | < 5% | **0.07070%** | 0.07070% | 逐位一致 | PASS（余量 70×） |
| Stokes 层内行数 | ≥ 5 | **8**（总行 48） | 8 | ✓ | PASS |
| 最差单行误差 | 参考 G2=10% | 0.11606%（y/δ=0.9375） | 0.11606% | ✓ | PASS |
| δ_fit/δ_analytic | ≈1（校核 δ=√(2ν/ω)） | 0.99877 | 0.99877 | ✓ | PASS |
| 首行 A/U0 | vs exp(-Δx/2/δ) | 0.939343 vs 0.939413 | 同 | ✓ | −0.0075% |
| 相位 L2 误差 | 参考 | 0.00512 rad | 同 | ✓ | PASS |
| 整流直流分量 | 参考 | 4.16e-4·U0 | 同 | ✓ | PASS |
| 有限深（H=6δ）L2 | < 5%（G4） | 0.07219% | 同 | ✓ | PASS |

**辩伪性证据**（不是拟合出来的好看曲线）：
- 负控①自由滑移壁：`sensitivity_free_slip_control` → `A_max/U0 = 0.00000`（无剪切生成，符合物理）；
- 负控②静止壁（不振荡）：`sensitivity_static_wall_control` → `A_max/U0 = 0.00000`；
- 线性响应：U0 减半 → L2 0.0706%（与基准 0.0707% 同量级，线性不变）；
- h/Δx 由 1.3 加到 1.9 → L2 恶化到 0.6512%（分辨率/核宽敏感性方向正确）；
- 真时间推进痕迹：`runs[0].steps=7272`, `dt=6.876e-4`, `n_fluid=288`, `max_mach≈1e-3`, trace 25 采样点含振荡 u_max/u_mean 与 wall_clock。

## 3. 发现（按严重度，含反证与复现）

**F1[high｜证据过度声明] 「收敛阶梯」是空的，G6 判据在 n=1 下空转。**
`results_ladder.json` 的 `convergence.resolutions = [8]`、`l2_by_resolution` 长度 1，却同时写出 `G6_not_growing: true, all_gates: true`；而 `run.py:114` 的 `LADDER = (8, 12, 18)`。成因在装配脚本 `scripts/finalize_stokes_second.py:28` `wait_for(paths, limit_s=2.0)` —— 循环 `while time.time()-t0 < 2.0` 但体内 `time.sleep(5.0)`（`:33`），即**最多只能等 2 秒**，`/tmp/p12` 当时未落盘就被跳过；`build_ladder` 又把 `missing_resolutions=['/tmp/p12']` 记下来却照常发布。反证：`/tmp/p12/results_probe.json` 现在存在且 `L2=0.02152%`、5 门全绿 —— 真实存在两档阶梯（0.0707% → 0.02152%，误差比 3.29，观测阶 ≈2.93），只是没进交付。第三档（δ/Δx=18）**从未跑过**。
复现：`python3 scripts/audit_m16239_stokes_second.py`（输出 `shipped_convergence_is_vacuous: true` 与 `reconstructed_ladder`）。

**F2[high｜证据不持久] 交付物被门禁滤除，run_artifacts 只在工作树/tmp 里。**
origin 分支 `5ffc9f5` 的树里 `benchmarks/stokes_second/out/` 仅有 `results_probe_dx8_3p5p.json`；`results_ladder.json`(4.2MB)、`results_sensitivity.json`(10.4MB)、`amplitude_decay.png`、`field_snapshot.png` 均不在库（commit 信息：4 个文件被门禁滤除）。即：**报告主指标所依赖的工件没有进入持久仓**，一旦工作树回收（当前 `worktree_recyclable=false`，尚安全）判据量将无原始工件可核。
复现：`git ls-tree -r --name-only 5ffc9f5 | grep stokes_second`。

**F3[high｜交付缺项] docs 页与平台报告均未产出。**
`benchmarks/stokes_second/make_doc.py:102` 是 `$"the half space..."`（f-string 前缀位置写错，`$` 顶在引号前）→ `python3 -m py_compile` 直接 `SyntaxError`；`docs/` 下无任何 `*stokes_second*` 页；`tensor_reports WHERE mission_id=16239` = **0 行**，且 mission 的 `report_id` 为 NULL、`report_owed=false`（报告纪律钩子没欠账，属平台侧漏挂）。mission 自述的 "docs 与 /tensor/reports 未产出" 与此一致 —— 实现方是**如实申报**了，只是仍被置为 succeeded。
复现：`python3 -m py_compile benchmarks/stokes_second/make_doc.py`；`psql -c "select count(*) from tensor_reports where mission_id=16239"` → 0。

**F4[medium｜溯源失真] 机器溯源块描述的不是实际跑的那次。**
`results_ladder.json` 的 `provenance.case.c0 = 10.0`（Ma 1e-3），但同一文件 `runs[0].sound_speed = 4.0`、`mach_number = 2.5e-3`；即 `provenance["case"]` 取的是模块常量 `run.py:110 SOUND_SPEED=10.0`，不是 CLI 实参。`DESIGN.md:67` 文档写的是 `c0 = 4 m/s`（与实跑一致）→ 文档对、**机器溯源块错**。provenance 亦不含 `periods/measure_periods`（实跑 5 期 / 窗口末 2 期，非默认 6/2）。下游任何"照 provenance 复跑"的人会得到不同构型。

**F5[medium｜测试面假红] 交付的测试里有一个断言本身写错了，且一直红着交付。**
`tests/test_stokes_second.py:83-95` `test_plate_boundary_condition_and_wave_speed` 断言波峰位置 `= δ·ω·t_peak`；但 `u=U0 e^{-y/δ}cos(ωt-y/δ)` 的驻定峰满足 `ωt-y/δ=π/4`，正确峰位是 `y*=δ(ωt-π/4)`。实测 `0.04432243` vs 测试期望 `0.08862269`（差 2 倍），而 vs 正确式 `δ(ωt-π/4)=0.04431135`（差 2.5e-4，纯 4000 点网格分辨）。**是测试错，不是解析解错**；mission checks 里如实写了 "10 passed, 1 failed"，但红灯被带进 succeeded 交付。

**F6[low｜平台缺口] 无机器裁判行。** `tensor_benchmark_cases` 无本锚点行（全表 27 anchor）→ 判据只能取自 case_pool 126 的自由文本，gate 无法机器比对 got/ref。

**F7[low｜集成面] 交付未进 main。** `agent/m16239`(5ffc9f5) 已在 origin，但 `main`(852e941) 的 `benchmarks/` 下无 `stokes_second/`（`git ls-tree HEAD benchmarks/ | grep stokes` 空）→ 基准尚未合并，主仓用户拿不到。

**F8[info｜命名易混] ** `results_probe_dx8_3p5p.json`（3.5 期，L2 0.05285%）与 ladder 的 dx=8（5 期，L2 0.07070%）同为 δ/Δx=8 却是两个数；文件名标了 `3p5p` 但缺 `c0`，易被误读为"同一构型漂移"。

## 4. 独立重执行（cross-check，非重做）

- 提交集群作业 `cj1299`：`stage_dir=/root/tp-agent/work/m16239`，`--probe 8 --periods 6 --measure-periods 2`（**用 run.py 默认 c0=10，刻意与作者 c0=4/5 期错开**，测判据量对构型的鲁棒性）。审计收口时该作业仍在排队（`queued`），结果以 `not_run` 记，不冒充 PASS。
- 已在本地完成的重执行证据：唯一红测 5.6 s 复现（`1 failed, 11 deselected`）；`make_doc.py` 编译复现 `SyntaxError`；ladder 逐位一致性（ladder 的 `runs[0].metrics` 与 `/tmp/p8` 的完全相同，`wall_clock_s 112.611` 同值 → 交付件确由该次运行装配，未事后改数）。

## 5. 级别判定与依据

| 级别要件 | 判定 | 依据 |
|---|---|---|
| L1 数学/单元验证 | 满足 | sympy 重新推导（PDE 残差 0、板面 BC 残差 0、`A(δ)/U0=1/e`）+ 数值再推导（`verify_numeric`，PDE 相对残差 6.2e-7）；`tests/test_stokes_second.py` 11 项覆盖解析面 |
| L2 独立功能实现 | 满足 | `src/tensorsph/stokes_second.py` 输入→计算→输出完整 |
| L3 系统集成 | 满足 | `run.py:76-91` 复用主求解器 `benchmarks.poiseuille.run.build_fluid/make_config`、`tensorsph.droplet_impact.rhs`、`tensorsph.wall_mirror.plane_walls`（`:120` 支持逐面速度的移动壁），288 粒子真推进 7272 步 |
| **L4 基准验证** | **满足（有返修项）** | 完整算例（全 SPH 场 + 双解析参照：半空间 Landau §24 与 H=6δ 有限深精确解）+ 误差分析（L2/最差行/衰减率/相位/δ_fit/截断偏差/敏感性/双负控）+ 可复现（DESIGN.md 第 7 节给命令，本审计复算逐位一致）+ 图 2 张 |
| L5 科研工程成果 | 不满足 | 无新物理/新方法产出，属标准基准复现 |

**L4 的保留项**：①收敛阶未在交付内被证明（F1，n=1 的 G6 空转；真实二档在 /tmp，三档未跑）；②判据量工件未持久入库（F2）；③报告六段未发（F3）。若 gate 认为"收敛阶梯证据"是 L4 的硬要件，则应降到 **L3**（集成与单档基准成立、收敛研究缺证），本审计在 evidence 中显式给出 `level_4_risk` 供机器裁量，不粉饰。

## 6. 返修清单（给 reopened 用，按性价比）

1. `scripts/finalize_stokes_second.py:28-34`：`wait_for` 改为 `time.sleep(min(5, remaining))` 且 `limit_s` 提到 ≥900；`build_ladder` 在 `missing` 非空时**直接 fail**（不得发布空阶梯）；`:74` `dirs` 由 `LADDER` 驱动而非硬编码 `["/tmp/p8","/tmp/p12"]`。
2. 补跑 δ/Δx=18 并重发 ladder（≥3 档才算收敛证据），把 `results_*.json` 压到入库体积（或按档拆分/只存 rows 摘要 + png 单独入库）。
3. 修 `make_doc.py:102` 的 f-string 前缀，产出 docs 页 + 发 `/tensor/reports`（六要素）。
4. `run.py` provenance 的 `case` 段改由实参回填（c0/periods/measure_periods），不用模块常量。
5. 修 `test_stokes_second.py:95` 期望为 `δ(ωt-π/4)`；加一条**把判据量钉住**的回归测试（`amplitude_l2_half_space_pct < 5`，可用 dx=6 短窗），否则基准精度无 CI 守卫。
6. 平台侧：给 `CASE-SPH-STOKES-2ND-PROBLEM` 补 `tensor_benchmark_cases` 行（metric=L2 振幅误差%、tolerance=5），令 gate 有机器裁判；并核查"报告未发仍置 succeeded + report_owed=false"的漏挂。

## 7. 未做（不冒充已做）

- 集群重跑 `cj1299` 未完成（排队中）→ 构型鲁棒性未取得独立新数据；
- δ/Δx=18 未跑（不在审计权限内、属返修项）；
- 未做 loong64/x86 跨架构逐位比较、未做 float32 敏感性；
- 未对 `/tmp/p12` 做独立重跑校验（只读其现存 JSON，其 `wall_clock_s=220.9`、源目录为 /tmp 无持久性）；
- 未评审 `src/tensorsph/wall_mirror.py` 移动壁内核的数值正确性（仅确认集成调用点存在且单测覆盖 ghost 均值＝壁速）。