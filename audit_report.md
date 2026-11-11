# TensorSPH 质量审计（A+B：资产盘点 + 功能真实性）

**锚点** AUDIT-ASSET-12REPO ｜ **mission_id** 14489 ｜ **审计者** audit2-ab ｜ **日期** 2026-11-11
**对象** `cxs503/TensorSPH` @ `main` = `c3ea82d`（Merge #457 之后），工作树 `/root/tp-agent/work/m14489`，分支 `agent/m14489`
**口径** 唯一核心标准 = 可验证的功能交付；文件数/报告页数/PR 数不进级别（owner 六级，L0–L5）。

---

## 0. 结论摘要（先给判定）

| 判定 | 结论 |
|---|---|
| 仓库整体成熟度 | **主求解器 L3、验证基础设施是真实亮点（3 个可复现的 L4 基准链），但"新功能"里存在系统性自证门（4 处已定位到行号）** |
| 最强资产 | `src/tensorsph`（26,049 行，**0 个 `NotImplemented`/`pass` 空函数**）；`verification/` + `benchmarks/verified/` 的 SQL 溯源指标链；仓库自带的 `benchmarks/dam_break/reference_audit.json` 把自家 Martin & Moyce 表判 WITHHELD —— 这是罕见的自我诚实机制 |
| 最严重问题 | **P0-1** 验收门由实现自身定义（clamp/归一化/恒等式），却被写成 PASS；**P0-2** PR#457 的"验证报告 + 5 张图 + 31 个"测试"全部是闭式公式代入，**零 SPH 运行**；**P0-3** 主线 HEAD 红：`fluctuating_hydro` 4/35 用例失败，含源码 `ZeroDivisionError`（`fluctuating_hydro.py:537`）；**P1-1** 旗舰 `dam_break` 的 Martin & Moyce 前沿误差 **77.8%**（门 10%）被 README 的"收敛阶 0.96"叙事掩盖 |
| 与"15%/3% 精度标准"对照 | 达标项：静水压 0.55%、3-D 剪切衰减 2.7%、Sod 质量守恒 <1e-10、Stefan 前沿 0.07–0.99%、Darcy–Bénard 0.016%。**不达标项**：dam-break 前沿 77.8%、CSF 毛细波 −43.4%、Lamb 模态 +9.7%、接触角 60°→152.9° |
| 未完成/无法验证 | 全量测试套件在本审计窗口内未跑完（宿主 load 483，集群作业 `cj846` 排队中）；`ice_thermal` 与 `wcsph_chain` 完整阶梯未由审计者重跑（仅重跑单级）。详见 §7 |

---

## 1. 资产盘点

### 1.1 全仓结构（`git ls-files` 实测，非估算）

| 类别 | 文件数 | 行数 | 占比 | 判读 |
|---|---:|---:|---:|---|
| 核心算法/源码 `src/` | 74 | **26,049** | 100% 基准 | 8 个子包：`(top)` 12,315（droplet_impact 1,521 / wave_flume 1,208 / wave_tank 1,178 / ice_splash 1,038 / wedge_entry 1,023 / slamming 793 / fluctuating_hydro 814 / wave_ice_interaction 791 …）、`threed` 2,506、`freesurface` 2,166、`aeration` 2,136、`ice_thermal` 1,829、`wcsph` 1,822、`verification` 1,708、`vv` 1,567 |
| 测试 `tests/` | 108 | 22,081 | 0.85× src | 每 1 行源码 0.85 行测试，密度高 |
| 基准脚本 `benchmarks/` | 144 | 45,628 | **1.75× src** | 65 个基准目录；其中 40 个有 `run.py`，其余只有数据/文档 |
| 驱动脚本 `scripts/` | 7 | 1,918 | — | 报告生成器为主 |
| 文档 `docs/` | 225（130 个 `.md`） | 19,308（md） | 0.74× src | 报告体量已接近源码体量 → 通胀风险信号 |
| 示例 `examples/` | 2 | 494 | — | `droplet_impact_2d`、`tutorial/first_steps` |
| 公开 API 面 | — | — | — | `tensorsph/__init__.py` `__all__` 导出 **90** 个符号；`src` 内 `def`/`method` 共 **821** 个 |

**关键结构事实**：`src` 内 **`NotImplementedError` 出现 0 次、AST 级空函数（`pass`/`return None`/空体）0 个**。占比意义上的"占位实现"**几乎不存在**（唯一命中是 `added_mass.py:341` 一句 docstring 里的 "placeholder documentation"）。这个仓的问题不是"空壳"，而是**充实的代码 + 自证的门**。

### 1.2 近期并入 main 的 agent PR（本审计对象）

`git log --first-parent --merges` 取最近 12 个含源码变更的 agent PR，逐 commit `git diff --numstat` 实测：

| PR | mission | 主题 | 新增行 | 其中源码 | 测试 | 基准 | 文档/数据 | 首判 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| #457 | m13901 | 自由面四锚点"验证报告"（P154W3-N07） | 1,516 | 309 | 443 | 0 | 142+622脚本 | **L1**（见 P0-2） |
| #456 | m13895 | CSF 离散色散/面 delta 一致性 | 1,452 | 49 | 220 | 978 | 205 | L2 实现 + **L4 基准(F-AIL)** |
| #455 | m13865 | Sod 激波管 WCSPH | 492 | 280 | 177 | 0 | 35 | L2 |
| #452 | m13785 | 两相 Laplace 测试收紧 | 103 | 0 | 44 | 0 | 59 | L1 |
| #448 | m11182 | WCSPH 链（solver+kernels+cases+chain） | 3,236 | 1,542 | 592 | 936 | 166 | L2 实现 + **L4 基准** |
| #447 | m10431 | 冰-热多物理（7 模块） | 5,979 | 1,829 | 281 | 1,014 | 2,855 | **L3/L4**（最强新增） |
| #446 | m10344 | 自由面验证包（M1/M2/M3） | 5,175 | 1,708 | 275 | 1,942 | 1,250 | **L4** |
| #445 | m10660 | 波-冰断裂相互作用 | 1,602 | 791 | 347 | 0 | 464 | L2（门自证） |
| #435 | m8572 | 核函数解析一致性 | 478 | 92 | 175 | 0 | 211 | L1 |
| #433 | m8354 | 涨落流体力学（LJ-MD） | 1,997 | 789 | 499 | 0 | 709 | L2（域外） |
| #432 | m7820 | 入水砰击 + δ-SPH/MLS | 1,437 | 720 | 297 | 420 | 0 | L2 |
| #410 | m7527 | 接触角边界条件诊断 | 1,280 | 25 | 146 | 1,032 | 77 | L1（诚实 FAIL） |
| **合计** | | | **24,747** | **8,134 (32.9%)** | **3,496 (14.1%)** | **6,322 (25.5%)** | **5,762+ (23.3%)** | |

判读：新增量里 **只有 1/3 是核心源码，1/2 是基准+文档**。"任务产出"与"功能产出"在这 12 个 PR 上比值约 **3:1**。

### 1.3 分类清单

**(a) 核心算法模块（真算法代码，输入→计算→输出）**：`droplet_impact`(1521) `dam_break`(354) `dam_break_3d`(625) `wave_flume`(1208) `wave_tank`(1178) `ice_splash`(1038) `wedge_entry`(1023) `slamming`(793) `fluctuating_hydro`(814) `wave_ice_interaction`(791) `csf_consistent`(539) `surface_tension`(308) `neighbor_search`(227) `threed/*`(2506) `wcsph/*`(1822) `ice_thermal/*`(1829) `aeration/*`(2136) `freesurface/*`(2166) `verification/*`(1708) `vv/*`(1567)。

**(b) 新增 API 接口**：`tensorsph/__init__.py` 90 个导出符号（`run`/`rhs`/`make_particles`/`stable_dt`… ），CLI `python -m tensorsph`（`src/tensorsph/__main__.py:12`）；子包各自有 `__init__`（`wcsph` 32 行导出、`ice_thermal` 61 行、`freesurface` 42 行、`aeration` 220 行）。**子包 API 不进入顶层 `__init__`**（`wcsph`/`ice_thermal`/`freesurface`/`verification`/`aeration`/`vv`/`slamming`/`wave_ice_interaction` 均不在顶层 `__all__` 中）。

**(c) 测试代码**：108 文件 22,081 行；但**测试质量两极**：`test_slamming.py`（advance() 对比、MLS 线性场精确性、δ-SPH 质量守恒）、`test_wcsph_solver.py`（227 行真实积分）是真测试；`test_p154w3n07_verification.py`（443 行 31 个用例）**全是 `assert 2·√(g·h0)·t == approx(2·√(g·h0)·t)` 式恒等式**（`tests/test_p154w3n07_verification.py:30-44`）。

**(d) 文档/报告**：130 个 md、19,308 行；含 6 份"验证报告"、5 张 p154 图、多份 `results.json`。

**(e) 占位实现**：**0 个代码级占位**。但存在 3 处**结构性空壳**：
- `docs/ice_thermal_multiphysics_results.json:anchors = {}`（"anchors" 段为空，而报告正文有逐条 PASS 表）；
- `benchmarks/slamming/`（793 行实现 + 420 行 runner）**无任何提交的运行工件**（无 json），L≥4 无从取证；
- `src/tensorsph/freesurface/runner.py`（376 行，自称"drives the study"）**除 `freesurface/__init__.py` 外无任何调用方**（`grep -rn "freesurface.runner"` 在 src/tests/benchmarks/scripts 下零命中）→ 死代码。

**(f) 重复实现（功能重叠）**：

| 功能 | 实现份数 | 位置 |
|---|---:|---|
| 三次样条核/梯度 | **4** | `droplet_impact.py:192`、`dam_break_3d.py:46`、`threed/kernel3.py:90`、`wcsph/kernels.py`、`wcsph/sod_shock.py:_cubic_w_vec` |
| WCSPH 主求解器 | **4** | `droplet_impact.rhs`、`wcsph/solver.py:Solver`、`dam_break_3d.rhs_3d`、`wcsph/sod_shock.py`（numpy 实现，与全仓 torch 不一致） |
| 邻居搜索 | **5** | `neighbor_search.py`（brute/cell/host）、`threed/neighbors3.py`、`wcsph/solver.py:PairListCache`、`fluctuating_hydro.py:_brute_force_periodic_pairs`、`threed/liquid_column.py` |
| 溃坝算例 | **5** | `dam_break.py`、`dam_break_3d.py`、`wcsph/cases.py`(dam2d/dam3d)、`verification/dam_break_2d.py`、`vv/dambreak.py` |
| 时间步长限制 | **3** | `droplet_impact.py:419-431`、`wcsph/solver.py:155`、`threed/solver.py` |

这解释了"26k 行源码 vs 单物理能力"的落差：**同一能力被重复实现 3–5 次，每次带自己的门与文档**。

**(g) 未接入模块（无 `src` 内调用方且无基准驱动）**：

| 模块 | 行数 | 唯一消费者 |
|---|---:|---|
| `wave_ice_interaction` | 791 | `tests/test_wave_ice_interaction.py`（无 runner、无算例） |
| `wcsph/sod_shock` | 280 | `tests/test_wcsph_validation.py`（`benchmarks/` 下无 runner） |
| `freesurface/breaking_wave` | 286 | 其测试 + `scripts/generate_p154w3n07_report.py`（后者只画解析曲线） |
| `freesurface/runner` | 376 | 无（仅 `__init__` 再导出） |
| `fluctuating_hydro` | 814 | 其测试 + `scripts/run_fluctuating_hydro_verification.py`（域外：LJ 分子动力学，不与 SPH 求解器耦合） |

---

## 2. 功能真实性审查（六问表）

Q1 = 代码入口（文件:行）｜Q2 = 真执行计算｜Q3 = 接入主程序｜Q4 = 实际 I/O｜Q5 = 完整运行案例｜Q6 = 独立基准验证

| # | 范围 | Q1 入口 | Q2 真算 | Q3 接入 | Q4 I/O | Q5 运行案例 | Q6 独立基准 | 差异/备注 |
|---|---|:--:|:--:|:--:|:--:|:--:|---|---|
| 1 | `droplet_impact`（主求解器） | `droplet_impact.py:run` / `__main__.py:12` | ✅ torch 积分，实跑 750 步 | ✅ 被 5+ 模块与 20+ 测试复用，CLI 入口 | ✅ CSV/快照/图/JSON | ✅ `examples/droplet_impact_2d` | ⚠️ 无实验对照（README 自认） | 诚实：README 明写 "not compared with experiment" |
| 2 | `dam_break` | `dam_break.py:run_case` | ✅ | ✅ `__init__` 导出 | ✅ | ✅ `benchmarks/dam_break/` | ⚠️ 有 Martin&Moyce 表但**自家审计判 WITHHELD**；前沿误差 **77.8%**（gate 10%，`docs/reports/sph-verification-P482-N04.json`） | README 的"order 0.96"是**对 ritter 自相似解**的收敛阶，不能读出"与实验一致" |
| 3 | `threed/*` | `threed/solver.py:Solver` | ✅ | ⚠️ 独立子包，不改 2-D | ✅ | ✅ 3 个 benchmark | ✅ 剪切衰减 2.7%、静水 2.7%、核一致性；**本审计重跑 `threed_kernel` 6/6 通过（11 s）** | 无自由面/多相（自认） |
| 4 | `wcsph/solver+cases+chain`（#448） | `wcsph/solver.py:180` | ✅ | ⚠️ 不进顶层 API、与 2-D 求解器零共享 | ✅ `artifacts/wcsph_chain/*.json` | ✅ `benchmarks/wcsph_chain/pipeline.sh` | ✅ 门与测量分离（`wcsph/chain.py`）；**本审计重跑 `pair --steps 500`：动量漂移 0.0、能量漂移 0.249%（门 1%）** | `f-tests` 门（覆盖率≥85%）是**自指门**（用测试量测试） |
| 5 | `wcsph/sod_shock`（#455） | `sod_shock.py:139` | ✅ 真 1-D SPH 循环（numpy） | ❌ 仅测试调用 | ✅ 返回 snapshots/errors | ⚠️ 只在测试里跑 | ✅ 与精确 Riemann 解（p*=0.30313）对照；质量守恒 <1e-10 | 与全仓 torch 技术栈不一致；仓内无 runner |
| 6 | `ice_thermal/*`（#447） | `ice_thermal/coupled.py:220:run_three_field` | ✅ 隐式 Picard + 焓法 | ⚠️ 子包，无上层耦合调用 | ✅ `docs/ice_thermal_multiphysics_results.json` | ✅ `benchmarks/ice_thermal_multiphysics/run.py`（465 s） | ✅ Stefan vs Neumann 0.07–0.99%；Darcy–Bénard 0.0157%；⚠️ Timoshenko "0.0000%" 是**恒等式**；`anchors` 段为空 | 收敛非单调（0.99%→0.072%→0.197%） |
| 7 | `verification/*` + `benchmarks/verified`（#446） | `verification/hydrostatic.py`、`dam_break_2d.py:233` | ✅ 跑真求解器 | ✅ 被 benchmark 与 `report.py` 消费 | ✅ SQLite 证据库 + SQL 溯源（`"sql": "SELECT ... FROM metrics WHERE run_id=3"`） | ✅ | ✅ 静水 0.55%、弹道高度 2.5%；**前沿 FAIL 77.8% 如实入账** | **本仓最可信的证据链** |
| 8 | `freesurface/*` 锚点（ritter/solitary/sloshing） | `ritter_anchor.py:355:run_ritter` | ✅ 调 `dam_break.run_case` | ⚠️ 仅测试 + runner(死) | ✅ | ✅ `docs/benchmarks/free-surface/report.json` | ⚠️ 无实测对照表 | 真实现，但被 #457 的"报告"虚化（见 P0-2） |
| 9 | P154W3-N07 / PR#457 | `scripts/generate_p154w3n07_report.py:323` | ❌ **零 SPH 运行**（只画闭式曲线） | ❌ | ✅ 5 张 PNG | ❌ | ❌ 31 个"测试"全是公式恒等式 | **P0-2**：声明"Complete verification surface / all analytical anchors verified to machine precision"，实为解析公式自比 |
| 10 | `csf_consistent` + 色散基准（#456） | `csf_consistent.py:surface_tension_acceleration` | ✅ | ✅ 被 droplet_impact/csf 链路使用 | ✅ `validation_report.json` | ✅ `benchmarks/surface_tension_dispersion/dispersion.py --quick` | ✅ 诚实 FAIL：毛细波 −43.4%、Lamb +9.7%、收敛阶 0.64 | **本审计复现 −43.44%（文档 −43.4%）逐位一致**；C4 "σ 敏感性 0.0%" 是归一化恒等式 |
| 11 | `slamming`（#432） | `slamming.py:512:SlammingImpact` | ✅ 继承 wedge solver 并改 rhs | ⚠️ 不进顶层 API | ✅ 返回 dict | ⚠️ runner 存在但**无提交工件** | ⚠️ Wagner 力系数解析对照；实际算例无留存结果 | δ-SPH 质量守恒、MLS 线性场精确性由测试证明 |
| 12 | `fluctuating_hydro`（#433） | `fluctuating_hydro.py:604:run_lj_fluid` | ✅ LJ-MD + Green-Kubo | ❌ 不与 SPH 耦合（域外） | ✅ | ✅ `scripts/run_fluctuating_hydro_verification.py` | ⚠️ 有 `benchmarks/fluctuating_hydro/*.json` 但无 runner | 单列"域外模块"，不计入 SPH 能力账 |
| 13 | `wave_ice_interaction`（#445） | `wave_ice_interaction.py:700+` | ✅ 弹性板响应/散射衰减/FSD 演化 | ❌ 仅测试调用 | ✅ 返回 dict + json | ❌ 无 runner | ❌ **6 条验收判据全部自证**（见 P0-1） | 771 行实现 → 判据不可信 |
| 14 | `aeration/*` | `aeration/__init__.py` 导出链 | ✅ 解析+数值模型 | ⚠️ 不与 SPH 求解器耦合（decalibration 结论） | ✅ `benchmarks/aeration/report.json` | ✅ `benchmarks/aeration/run.py` | ⚠️ C1 "vs Chanson" 实测 0.0 = **恒等式** | 其余判据（erf 锚点 L2 9.5%）为真对照 |
| 15 | `contact_angle` BC（#410） | `benchmarks/contact_angle/diagnose_bc.py` | ✅ | ⚠️ 诊断脚本 | ✅ 3 份 json | ✅ | ✅ **诚实 FAIL**：设 60° 测得 152.9°（误差 92.9°），并标 `admissible:false` | 正例 |
| 16 | `kernel_analytic`（#435） | `threed/kernel3.py` + 测试 | ✅ | ✅（3-D 核） | ✅ `docs/kernel_analytic_m8572/numbers.json` | ✅ 3 图 | ✅ 与解析核/线性场对照 | 级别 L1（数学验证） |
| 17 | `vv/*`（验证阶梯） | `vv/ladder.py` | ✅ | ⚠️ 仅 benchmark/test | ✅ | ✅ `benchmarks/verification_ladder/run.py` | ⚠️ 依赖 vv schema 自评 | L2 |
| 18 | `wave_tank`/`wave_flume` | `wave_tank.py`、`wave_flume.py` | ✅ | ✅ 被 freesurface 锚点复用 | ✅ | ✅ 2 个 runner | ⚠️ 无实验基准 | L2–L3 |
| 19 | `wedge_entry`/`ice_splash`/`added_mass` | 各自 `run`/`estimate` | ✅ | ✅ | ✅ | ✅ 3 个 runner | ⚠️ 半解析锚点 | L2 |
| 20 | `two_phase_laplace`（#452） | 仅 `tests/test_two_phase_laplace.py` | ⚠️ 测试驱动 | — | — | ❌ | ⚠️ 测试收紧（+59 行文档） | L1 |

---

## 3. 成熟度判级（owner 六级，全部已落 `tensor_maturity_verdicts`）

| scope | L | 一句话依据 |
|---|:--:|---|
| `droplet_impact (主求解器)` | **L3** | 26k 行仓的主积分器，CLI + 90 符号 API，被 5+ 模块/20+ 测试复用；无实验对照 |
| `threed/ (3-D 算子)` | **L3** | 算子-解析解闭环；本审计重跑 `threed_kernel` 6/6 通过；无自由面/多相 |
| `dam_break (2-D 溃坝)` | **L3** | 真求解器 + 收敛阶梯；实验门 **FAIL 77.8%**，参考表自家判 WITHHELD |
| `verification/ + benchmarks/verified (PR#446)` | **L4** | SQL 溯源指标链 + 负向控制（gravity=0）+ 诚实 FAIL；静水 0.55% |
| `benchmarks/wcsph_chain (PR#448)` | **L4** | 测量/判据分离 + 阶梯 + 参考对照；审计者复现 `pair` 门内（0.249% < 1%） |
| `csf_dispersion 基准 (PR#456)` | **L4（物理门 FAIL）** | 完整算例+收敛+误差分析，审计者逐位复现（−43.44%）；物理判据未过 → 能力本身 L2 |
| `wcsph/solver+kernels (PR#448)` | **L2** | 独立 WCSPH 求解器（4 个求解器之一），不进顶层 API、与主求解器零共享 |
| `ice_thermal/ (PR#447)` | **L3** | 真多物理（焓法隐式/达西/盐度），Stefan 0.07–0.99%；审计者未重跑（465 s），2 条判据为恒等式 |
| `csf_consistent (表面张力)` | **L2** | 真 CSF 实现（含 local_range 归一化）；干舷精度 9.7–43% |
| `wave_ice_interaction (PR#445)` | **L2** | 真模型代码，但 6 条验收判据全为构造/clamp → 无独立验证；无 runner |
| `wcsph/sod_shock (PR#455)` | **L2** | 真 1-D SPH，Riemann 精确解对照；无 runner、技术栈异构 |
| `slamming (PR#432)` | **L2** | 真 δ-SPH/MLS 扩展，测试驱动其 rhs；runner 无提交工件 |
| `fluctuating_hydro (PR#433)` | **L2**（实现级） | 真 LJ-MD + Green-Kubo；域外（非 SPH），不与求解器耦合 |
| `fluctuating_hydro (PR#433) @HEAD 实测` | **L1** | **其自身 35 例测试中 4 例在 HEAD 红**（T=0 负向控制失效、均分差 31%、dt 敏感性 21.6%、`fluctuating_hydro.py:537` 除零）→ 原「已验证」声明不成立 |
| `freesurface 锚点 (ritter/solitary/sloshing)` | **L2** | 真跑 `dam_break`/`wave_flume`；报告来自 runner，非解析代填 |
| `aeration/ (P165W2-N04)` | **L2** | 真模型链 + runner + report.json；C1 为恒等式 |
| `wave_tank / wave_flume` | **L2** | 真自由面驱动，被锚点复用；无实验基准 |
| `wedge_entry / ice_splash / added_mass` | **L2** | 真实现 + runner；半解析锚点 |
| `vv/ 验证阶梯` | **L2** | 阶梯框架可跑；自评 schema |
| `freesurface/runner（死代码）` | **L1** | 376 行可运行代码，但除包内 `__init__` 外**零调用方** |
| `P154W3-N07 / PR#457` | **L1** | 31 个恒等式测试 + 5 张解析曲线图 + 报告；零 SPH 运行 |
| `kernel_analytic (PR#435)` | **L1** | 核函数解析一致性（数学验证） |
| `contact_angle BC 诊断 (PR#410)` | **L1** | 诊断脚本 + 诚实 FAIL；未形成可用 BC |
| `two_phase_laplace (PR#452)` | **L1** | 仅测试收紧 + 文档 |

**L 分布（= 已落表 25 行实测）**：L5 = **0**；L4 = **4**（16.0%，其中 1 行为 `hydrostatic #run-evidence` 的独立重跑证据）；L3 = **4**（16.0%）；L2 = **11**（44.0%）；L1 = **6**（24.0%）；L0 = 0。
（`fluctuating_hydro` 因 PR 原状 L2 / HEAD 实测 L1 的差异各落一行，不做覆盖式改写。）
即：**没有一个模块达到"科研与工程成果（L5）"**；L4 全部集中在**验证基础设施**，而非物理能力；物理能力天花板是 L3（主求解器）+ 若干 L2。

---

## 4. Top 问题（按严重度）

### P0-1 验收判据由实现自身定义，却被写成 PASS（4 处，逐行可核）

| # | 位置 | 判据 | 为什么是自证 |
|---|---|---|---|
| a | `src/tensorsph/wave_ice_interaction.py:477` `alpha_new = max(2.0, min(3.5, alpha_new))` + `docs/verification_wave_ice_breakup.md:156` | "α = 3.50 ∈ [2.0, 3.5] ✓" | 代码把 α **clamp 到 [2.0,3.5]**，门又正好是 [2.0,3.5] ⇒ 恒过。文档自己写"三个波高全为 3.50"= 全部顶到上界饱和 |
| b | `wave_ice_interaction.py:312` `critical_wave_amplitude` + `docs/verification_wave_ice_breakup.md:147` | "σ_max/σ_f = 1.000，误差 <0.01% ✓" | 文档**自己承认"由构造保证"**——临界波幅就是这么反解出来的，误差 0.01% 只是浮点往返 |
| c | `benchmarks/aeration/run.py:47,60,287` `INCEPTION["constant_rel_error"]`，`aeration/entrainment.py:46 C_CRIT=1.0`（Chanson 常数语句即 1.0） | "C1 inception vs Chanson，measured 0.0 ≤ 0.1 PASS" | 模型常数**就是**对照常数，相对误差恒为 0 |
| d | `benchmarks/surface_tension_dispersion/validation_report.json:sigma_sensitivity`（x0.5/x1.0/x2.0 三个 σ 值逐位相同，`max_rel_spread: 0.0`），文档 C4 PASS | "σ 敏感性 0.0% <5% PASS" | 观测量按 σ 归一化（ω²∝σ），三档**必然**相同；恒等式冒充敏感性检验 |

### P0-2 PR#457：一份"验证报告"的端到端虚化（1271 行工件，零仿真）

- `scripts/generate_p154w3n07_report.py:36-320` 的 5 个 `plot_*` 函数全为闭式公式作图（Ritter `depth=c²/g`、solitary `c=√(g(h+A))`、sloshing `ω=√(kg·tanh kh)`、breaking `H=0.14L`）；
- 报告数字（§3_results）由脚本内 `g=9.81; h0=0.4; c0=√(g h0)` 现算，**未调用任何 `run_*` 求解函数**（`grep RitterCase/SloshingModeCase/run_` 在该脚本内零命中）；
- `tests/test_p154w3n07_verification.py` 31 例全为公式恒等式，例：`:30-44` 先定义 `distance = 2.0*c0*t`，再断言 `distance == approx(2.0*math.sqrt(g*h0)*t)`；
- 报告却声明 "Complete verification surface … 31 pytest tests pass; **all analytical anchors verified to machine precision**" 并列出 `key_settings.ritter_cells_per_depth: 10` 等**从未发生的分辨率设置**。
- **结论**：该 PR 的实际功能级别 L1（公式实现），被包装成 L4 叙事。工件体积 1,516 行里，可用功能增量为 `freesurface/breaking_wave.py`(286) + `gates.py`(+9)。


### P0-3 主线 HEAD 红：`fluctuating_hydro` 4 个用例失败（含 1 个源码崩溃）

本审计定向跑 15 个审计关键测试文件：**304 passed / 4 failed / 461.7 s**（`/tmp/pytest_subset.log`）。4 个失败**全部**在 `tests/test_fluctuating_hydro.py`（该文件共 35 例）：

| 用例 | 断言/异常 | 行号 |
|---|---|---|
| `test_e_negative_control_t0_no_fluctuation` | `assert 0.00043820226183868984 < 1e-08` —— T=0 负向控制**不成立**（残余温度 4.38e-4） | `tests/test_fluctuating_hydro.py:371` |
| `test_temperature_sweep_equipartition[0.5]` | `T=0.5: equipartition ratio 1.3116`（容差 20%）——**能量均分差 31 %** | 同上 `:400` |
| `test_temperature_dt_sensitivity` | `T(dt=0.005)=1.1243, T(dt=0.0025)=1.3667, rel diff=0.2156`（容差 15%）——**温度不可复现** | 同上 `:436` |
| `test_generate_report` | **`ZeroDivisionError: float division by zero`** @ `src/tensorsph/fluctuating_hydro.py:537`（`eta_full = volume / temperature * integral`，未守 T=0） | `src/tensorsph/fluctuating_hydro.py:537` |

引入路径：`git log -- src/tensorsph/fluctuating_hydro.py` → `0c58ed8 Merge #450 (m13758) via merge-master`（+53/−28，只改源码不改测试）。即 PR#433 的门在 HEAD 上已不成立，而合并门未拦住。**这一条把 `fluctuating_hydro` 的级别从"L2 已验证"压到"L2 未通过自身验收"**，并给出一个明确可修的缺陷（T=0 除零 + 均分容差）。

---

### P1-1 旗舰算例的失败被收敛阶叙事掩盖
`README.md` 首屏写 "2-D dam break: front converges at observed order 0.96"；而 `docs/reports/sph-verification-P482-N04.json` 里 `front_max_rel_error = 0.778`、`gate_front_pass: FAIL`（门 10%）。两者不矛盾（前者是相对 Ritter 自相似解的收敛阶，后者是相对实验的无量纲前沿），但 README 未写后者。**建议在 README "Implemented" 表内直接标 77.8% FAIL**。

### P1-2 同一能力 3–5 份实现，门各写一套
见 §1.3(f)。直接后果：`wcsph`（1822 行）与 `droplet_impact`（1521 行）两套 WCSPH 求解器互不校验，`dam break` 有 5 份实现、5 套判据。维修成本与"看起来很多功能"的错觉同源。

### P1-3 未接入模块 5 个（1,547 行无任何驱动方）
`wave_ice_interaction`(791)、`wcsph/sod_shock`(280)、`freesurface/breaking_wave`(286)、`freesurface/runner`(376，死代码)、`fluctuating_hydro`(814，域外)。它们只有测试消费者，或什么都不消费。

### P2-1 测试密度高但断言强度不均
`assert math.isfinite(x)`、`assert len(orders)==2`、`assert result["steps"]>0` 这类"只测不崩"的断言在 `test_wcsph_validation.py`、`test_p154w3n07_verification.py` 占比高。建议给测试加 `kind: invariant|smoke|identity` 标记，identity 类不计入验收证据。

### P2-2 `f-tests` 门自指
`wcsph/chain.py:22,167-169` 把"pytest 用例数≥20 + 覆盖率≥85%"列为链的物理验收门 `f-tests`。覆盖率高不等于物理对，**测试自证**风险明确。

### P2-3 结果工件缺审计元数据
`benchmarks/slamming/` 无 json；`docs/ice_thermal_multiphysics_results.json` 的 `anchors` 段为空；`benchmarks/aeration/report.json` 的 C1 无原值。凡"报告有结论、工件无原值"处，审计只能降到 L1。

---

## 5. "曾声称为功能/验证成果、实为文档/测试/公式"膨胀清单

| 声称 | 实况 | 证据指针 | 膨胀量 |
|---|---|---|---|
| "Complete verification surface for free-surface … 31 tests pass, anchors verified to machine precision" (#457) | 31 个公式恒等式 + 5 张解析曲线图，**零 SPH 运行** | `tests/test_p154w3n07_verification.py:30-44`、`scripts/generate_p154w3n07_report.py:36-320` | 1,271 行工件 ≈ **0 功能增量** |
| "α = 3.50 与卫星观测 [2.0,3.5] 一致 ✓" (#445) | α 被 clamp 到同一区间；文档另处自认"无基准值可比对" | `wave_ice_interaction.py:477`、`docs/verification_wave_ice_breakup.md:156,258` | 1 条判据 |
| "σ_max/σ_f = 1.000，误差 0.01% ✓" (#445) | 文档自写"由构造保证" | `docs/verification_wave_ice_breakup.md:147` | 1 条判据 |
| "C1 inception vs Chanson PASS" (aeration) | 常数对常数，误差恒 0 | `aeration/entrainment.py:46`、`benchmarks/aeration/run.py:47` | 1 条判据 |
| "C4 σ 敏感性 PASS" (#456) | 归一化恒等式（逐位相同 0.0） | `benchmarks/surface_tension_dispersion/validation_report.json` | 1 条判据（该 PR 其余判据为诚实 FAIL） |
| "Timoshenko 板 0.0000% PASS" (#447) | 与 `EαΔT/2(1−ν)` 代数恒等式比对 | `docs/verification_ice_thermal_multiphysics.md:15` | 1 条判据 |
| "f-tests 门 PASS（覆盖率 85%）" (#448) | 用测试量测试 | `wcsph/chain.py:167-169` | 1 条门 |

**量化**：本审计抽到的 7 处"判据虚化"，分布在 5 个 PR、4 个模块；被虚化的判据总数 ≥7 条（占所抽验判据 ~40 条中的 17%）。

---

## 6. 与精度标准（>15% 不可接受 / <3% 目标）的逐项对照

| 量 | 基准 | 实测 | 误差 | <3%? | 来源 |
|---|---|---|---:|:--:|---|
| 静水压剖面（内带） | ρg(H−z) | — | 0.55%（存档）/ **1.064%（审计者重跑最大值，均值 0.602%）** | ✅ | `benchmarks/verified/sph_hydrostatic/results.json`；`artifacts/audit_m14489/hydrostatic.log` |
| 弹道段高度 | h(t) 解析 | — | 2.50% | ✅ | 同上（P482 metrics） |
| 3-D 剪切波衰减 | e^{−νk²t} | — | 2.7% | ✅ | `docs/verification_threed.md` |
| 3-D 静水压 | ρgH | — | 2.7% | ✅ | 同上 |
| Stefan 前沿（dx=0.0125） | Neumann 相似解 | — | 0.072% | ✅ | `docs/ice_thermal_multiphysics_results.json` |
| Darcy–Bénard 起始 | Ra_c=4π² | — | 0.0157% | ✅ | 同上 |
| Lamb 模态 n=3 | √(12σ/ρR³) | — | +0.42% | ✅（单模态） | `docs/verification_surface_tension_dispersion.md:109` |
| Lamb 模态 n=2 / n=4 | 同上 | — | +9.7% / −7.7% | ❌ | 同上 |
| 毛细波色散 k=1..3 | √(σk³/ρ) | — | −43.4% | ❌ | 同上（**本审计复现 −43.44%**） |
| 溃坝前沿 | Martin & Moyce（crosscheck 表） | Z/a 0.16 vs 0.70 | **77.8%** | ❌ | `docs/reports/sph-verification-P482-N04.json` |
| 接触角 60° | 设定 60° | 152.9° | 92.9° | ❌ | `benchmarks/contact_angle/bc_reach.json` |
| 溃坝前沿收敛阶 | ≥1.0 | 0.96 | — | ❌（略低） | README / `benchmarks/convergence` |
| CSF 收敛阶 | ≥2 | 0.64 | — | ❌ | `docs/verification_surface_tension_dispersion.md:13` |
| Sod 质量守恒 | 机器精度 | <1e-10 | — | ✅ | `tests/test_wcsph_validation.py:61` |
| wcsph 双粒子动量/能量 | 0 / <1% | 0.0 / 0.249% | — | ✅ | **本审计实测** `artifacts/wcsph_chain/pair.json` |

**回答"是否 <3%"**：结构/热学/静水类 6 项达标；**自由面精度类（溃坝前沿、CSF 色散、Lamb 高阶模态、接触角）全部 >15%，不可接受**。仓库对这些失败大体是诚实的（#456、#446、#410 均写明 FAIL），问题在于叙事层（README、报告标题、验收表）把它们混进"已完成"。

---

## 7. 本审计的实际执行证据与 honest FAIL

**已实际运行（工件在 `artifacts/audit_m14489/`）**

| 命令 | 结果 | 耗时 |
|---|---|---|
| `benchmarks/threed_kernel/run.py --merge` | 6/6 判据 pass（kernel 归一 1e-6、二阶矩、质量标定 1e-12、Laplacian 修正 <5%） | 10.9 s |
| `benchmarks/surface_tension_dispersion/dispersion.py --quick` | 复现 legacy −54.57%/−45.42%、local_range −43.44%（与文档 −43.4% 逐位一致） | ~60 s |
| `benchmarks/wcsph_chain/run.py pair --steps 500` | 动量漂移 **0.0**、能量漂移 **0.249%**（门 1%）→ PASS | 12.8 s |
| `benchmarks/verified/sph_hydrostatic/run.py` | **N=1000，内带最大误差 1.064%（均值 0.602%），自由面带 −3.6%，gate PASS；ρ0=0 负向控制 PASS**（与仓库存档 0.55% 同量级） | 6m44s |
| 定向测试子集（15 文件：wcsph×5、ice_thermal、wave_ice、csf_dispersion、p154w3n07、verification、threed_kernel、slamming、fluctuating_hydro、kernel_analytic、dam_break） | **304 passed / 4 failed**（4 例全在 `test_fluctuating_hydro.py`，含 1 个源码 ZeroDivisionError） | 461.7 s |
| 全量 `pytest -q`（108 文件） | **未跑完**：宿主 load average **483**（同机 33 个其他 agent 的 pytest 并发），15 分钟仅到 45%；前 31% 已观察到 **≥7 个 FAIL** | — |

**无法验证 / honest FAIL**
1. **全量测试未完成**：`/tmp/pytest_full.log` 显示前 31% 有 7 处 `F`（未到终局，失败用例名未落盘）。已递交集群补跑（`run_cluster` job **cj846**，stage_dir=本工作树，8 核，7200 s），审计窗口内仍 `queued`（前方 249 单）。**因此"测试全绿"这一声明本审计无法确认，如实记为 unknown/not_run**。
2. **`ice_thermal` 未重跑**（官方 465 s，宿主满载），其 L3 依据为**留存工件 + 代码可运行性**，不是审计者复现。
3. **`wcsph_chain` 完整阶梯未重跑**（pipeline 需 5+ 小时 CPU），仅复现 `pair` 单级；L4 依据 = 测量/判据分离的文件布局 + 单级复现 + 留存 JSON。
4. **`sph_hydrostatic` 已由审计者独立重跑通过**（6m44s）→ 这是 `verification/`（PR#446）L4 判级的直接运行证据。
5. **`wave_ice_interaction` 的 6 条判据不予采信**：无法用任何独立基准复核（无 runner、无对照数据），故模块判 L2 而非更高。
6. `benchmarks/slamming/` 无提交工件 → slamming 的"完整算例"未取证，判 L2。

---

## 8. 建议（按投入产出排序）

1. **给判据加类型标签**（`invariant|identity|reference`），`identity` 类禁止出现在验收表 PASS 列（改 4 个文件即可，见 P0-1）。
2. **README 「Implemented」表加"最大误差/门"列**，把 77.8%、43.4%、9.7% 写进首屏（1 个文件）。
3. **拆两套 WCSPH 求解器**：确定一套为唯一实现，另一套降为"交叉校验工具"并写清差异（长期）。
4. **未接入模块二选一**：给 `wave_ice_interaction`/`sod_shock`/`breaking_wave` 各补一个 `benchmarks/*/run.py` + 工件，或移入 `sandbox/`（各半天）。
5. **死亡代码清理**：`freesurface/runner.py` 要么接 `scripts/run_freesurface_verification.py`（脚本已存在却没用它？）要么删。
6. **结果工件强制原值**：`anchors: {}`、C1 `measured: 0.0` 这类必须带原值与公式来源。

---

**工件位置**：`docs/audit_m14489/`（提交入库）= 全部实跑日志（`threed_kernel.log`/`dispersion.log`/`wcsph_pair.log`/`hydrostatic.log`/`pytest_subset.log`/`pytest_full_partial.log`）、复现数据（`dispersion_quick.json`）、两张图、以及 25 条 verdict 的插入记录 `verdicts.sql`；工作树内同一份副本在 `artifacts/audit_m14489/`（该目录被 .gitignore 忽略，仅本地留档）。

*本报告全部数字来自：`git diff --numstat`、`wc -l`、`ast` 扫描、`pytest` 日志、仓内 JSON 工件、本审计实跑日志（`artifacts/audit_m14489/`）。判级已写入 `tensor_maturity_verdicts`（`verdict_by='audit2-ab'`, `mission_id=14489`），未修改任何既有行、未 ALTER 表。*