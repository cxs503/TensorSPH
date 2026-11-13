# m17452 交差核验（board audit）— XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID

* 审计任务：m17494（`board-audit`）；被审任务：m17452（TensorSPH，状态 succeeded）
* 被审工作树：`/root/tp-agent/work/m17452`，分支 `agent/m17452`，HEAD `e64e0064dd05`（`e64e006 Merge #626 (m17227) via push-merge`）
* 锚点：`XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`；仓：`TensorSPH`
* 结论先行：**级别 L2**，benchmark **FAIL**，disposition = **request changes**（返修，不提级）。报告诚实、数字逐位可复现，但功能实质未过基准。

## 1. 审计对象与独立性

被审对象是 m17452 在 2026-11-13 06:04 结束（`fail_stage=budget_exhausted+forced_finish`，tool_calls 80/80）时留在工作树里的交付物。审计者未参与其开发，只读字节 + 重放命令；`uncertain` 的地方一律标 `not_run`，不写成通过。

**独立性限制（如实声明）**

* 审计者能看清的是**工作树快照**，不是 commit：被审任务 4 条交付路径全部处于 untracked 状态（见 §3 F7），因此不存在可引用的 commit/PR 身份；下面的 sha256 就是对快照的唯一身份绑定。
* 无法运行上游 SPHinXsys 本体（C++/TBB，无编译预算），所以「OSS 参考值」不是跑出来的，而是代数移植 + 解析基准；本审计只能核**移植是否按所引行号实现**与**结果是否达 15%**，不能核 OSS 程序自身行为。
* 注册裁判表对本锚点 **0 行**（见 §2），因此严格意义上没有「机器注册的容差」可对照；审计采用被审任务在跑数**之前**写死的 15%/3%（run.py 模块 docstring），它与 owner 精度标准同值，且审计者未改动它。

## 2. 三件套取证 + 注册裁判

### 2.1 run_artifacts（文件真实存在，sha256 绑定）

| 交付路径 | 存在 | 字节 | sha256(前16) |
|---|---|---|---|
| `src/tensorsph/fsi_force_from_fluid.py` | True | 15546 | `e97df2418671b854` |
| `tests/test_fsi_force_from_fluid.py` | True | 6133 | `3c622ff1f6149994` |
| `benchmarks/xval_fsi2/cases.py` | True | 14818 | `0f7f61821015bd0d` |
| `benchmarks/xval_fsi2/run.py` | True | 9469 | `cda160848d0c6358` |
| `benchmarks/xval_fsi2/make_doc.py` | True | 18865 | `8a4ef529b9954779` |
| `docs/xval_fsi2/GAP-SPH-FSI-BENCHMARK_xval.md` | True | 16679 | `f968ca66621e275f` |
| `artifacts/xval_fsi2/results.json` | True | 24112 | `7ff7c32e981e4cd6` |
| `artifacts/xval_fsi2/fig_error_vs_dx.png` | True | 109338 | `2dd2f51356f98071` |
| `artifacts/xval_fsi2/fig_thickness.png` | True | 32922 | `fb03eacf90e992cf` |

`artifacts/xval_fsi2/oss_src/` 实际归档内容：`BeamObserver_Position_Run_0_result.xml`, `BeamObserver_Position_Run_9_result.xml`, `InsertedBody_TotalViscousForceFromFluid_time_averaged_mean_variance.xml`, `hydrostatic_fsi.cpp`。

### 2.2 注册裁判（tensor_benchmark_cases）

`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID'` → **0 行**（全表 54 例，`anchor_slug ILIKE '%FSI%'` 命中 0 例）。

如实记录：本锚点**没有注册基准用例**，所以「基准判据是唯一裁判」这一条在本单退化为「被审任务自定且跑前声明的判据」：
* 必需 |err| < 15%，目标 < 3%（来源：predeclared in benchmarks/xval_fsi2/run.py docstring）；审计未在任何时刻改动阈值。

### 2.3 repro_cmd（在临时副本里**重放过**，不是看着像能跑）

* `pytest tests/test_fsi_force_from_fluid.py -q` → rc=0，`10 passed, 1 warning in 6.08s`
* `python benchmarks/xval_fsi2/run.py` → rc=0，重写 results.json
* 逐叶比对：542 个数值叶（共 554 叶）**全部一致**；仅 12 个 `timings_s` 叶按预期不同 → 逐位可复现 = **True**

## 3. 判级（L0–L5，按功能实质）

**级别 = L2**。

| 级 | 判据 | 本单 |
|---|---|---|
| L1 | 数学/单元验证 | 达标（不变量测试：符号、dx 不敏感、线性场精确、无邻居零载） |
| L2 | 独立功能实现（真代码：输入→算→输出） | **达标** — 352 行算子 + 复现驱动 |
| L3 | 系统集成（接入主求解器/平台） | **未达标** — production callers outside tests/benchmarks: none；`src/tensorsph/__init__.py` 导出：none (fsi module absent from tensorsph/__init__.py) |
| L4 | 基准验证（完整算例+误差分析+可复现） | **未达标** — 0/21 ported-operator rows within 15% |
| L5 | 科研工程成果 | 不适用 |

即：有真实实现、有可复现证据、有诚实的失败陈述，但**移植算子本身在基准上失败**，且从未被主求解器调用。报告 167 行、两张图、拓扑完整的六段都不改变这一点（级别=功能实质，不是文档厚度）。

## 4. 基准判读：ref vs got 逐项

| case | dx | operator | ref | got | err | <15% | <3% |
|---|---|---|---|---|---|---|---|
| vertical_wall | 0.05 | oss | 19620 | 25033.3 | +27.59% | ✘ | ✘ |
| vertical_wall | 0.05 | oss_riemann | 19620 | 25033.3 | +27.59% | ✘ | ✘ |
| vertical_wall | 0.05 | panel | 19620 | 19620 | -0.00% | ✔ | ✔ |
| vertical_wall | 0.05 | pairwise_legacy | 19620 | -24898.4 | -226.90% | ✘ | ✘ |
| vertical_wall | 0.025 | oss | 19620 | 25105.4 | +27.96% | ✘ | ✘ |
| vertical_wall | 0.025 | oss_riemann | 19620 | 25105.4 | +27.96% | ✘ | ✘ |
| vertical_wall | 0.025 | panel | 19620 | 19620 | -0.00% | ✔ | ✔ |
| vertical_wall | 0.025 | pairwise_legacy | 19620 | -25037 | -227.61% | ✘ | ✘ |
| vertical_wall | 0.0125 | oss | 19620 | 25140.7 | +28.14% | ✘ | ✘ |
| vertical_wall | 0.0125 | oss_riemann | 19620 | 25140.7 | +28.14% | ✘ | ✘ |
| vertical_wall | 0.0125 | panel | 19620 | 19620 | -0.00% | ✔ | ✔ |
| vertical_wall | 0.0125 | pairwise_legacy | 19620 | -25106.3 | -227.96% | ✘ | ✘ |
| horizontal_gate | 0.05 | oss | 19620 | 24373.4 | +24.23% | ✘ | ✘ |
| horizontal_gate | 0.05 | oss_riemann | 19620 | 24373.4 | +24.23% | ✘ | ✘ |
| horizontal_gate | 0.05 | panel | 19620 | 19620 | -0.00% | ✔ | ✔ |
| horizontal_gate | 0.05 | pairwise_legacy | 19620 | -24184.4 | -223.26% | ✘ | ✘ |
| horizontal_gate | 0.025 | oss | 19620 | 24773.1 | +26.26% | ✘ | ✘ |
| horizontal_gate | 0.025 | oss_riemann | 19620 | 24773.1 | +26.26% | ✘ | ✘ |
| horizontal_gate | 0.025 | panel | 19620 | 19620 | -0.00% | ✔ | ✔ |
| horizontal_gate | 0.025 | pairwise_legacy | 19620 | -24677.5 | -225.78% | ✘ | ✘ |
| horizontal_gate | 0.0125 | oss | 19620 | 24974 | +27.29% | ✘ | ✘ |
| horizontal_gate | 0.0125 | oss_riemann | 19620 | 24974 | +27.29% | ✘ | ✘ |
| horizontal_gate | 0.0125 | panel | 19620 | 19620 | -0.00% | ✔ | ✔ |
| horizontal_gate | 0.0125 | pairwise_legacy | 19620 | -24926 | -227.04% | ✘ | ✘ |
| couette_shear | 0.025 | oss_viscous | 0.1 | 0.0614614 | -38.54% | ✘ | ✘ |
| couette_shear | 0.0125 | oss_viscous | 0.1 | 0.061811 | -38.19% | ✘ | ✘ |
| couette_shear | 0.00625 | oss_viscous | 0.1 | 0.0619858 | -38.01% | ✘ | ✘ |
| fsi2_insert_linear | 0.1 | oss | 1.4854 | 3.08282 | +107.54% | ✘ | ✘ |
| fsi2_insert_linear | 0.1 | oss_riemann | 1.4854 | 3.08282 | +107.54% | ✘ | ✘ |
| fsi2_insert_linear | 0.1 | panel | 1.4854 | 1.475 | -0.70% | ✔ | ✔ |
| fsi2_insert_linear | 0.1 | pairwise_legacy | 1.4854 | -2.79096 | -287.89% | ✘ | ✘ |
| fsi2_insert_linear | 0.05 | oss | 1.4854 | 2.72306 | +83.32% | ✘ | ✘ |
| fsi2_insert_linear | 0.05 | oss_riemann | 1.4854 | 2.72306 | +83.32% | ✘ | ✘ |
| fsi2_insert_linear | 0.05 | panel | 1.4854 | 1.78363 | +20.08% | ✘ | ✘ |
| fsi2_insert_linear | 0.05 | pairwise_legacy | 1.4854 | -2.60315 | -275.25% | ✘ | ✘ |
| fsi2_insert_linear | 0.025 | oss | 1.4854 | 2.49036 | +67.66% | ✘ | ✘ |
| fsi2_insert_linear | 0.025 | oss_riemann | 1.4854 | 2.49036 | +67.66% | ✘ | ✘ |
| fsi2_insert_linear | 0.025 | panel | 1.4854 | 1.96453 | +32.26% | ✘ | ✘ |
| fsi2_insert_linear | 0.025 | pairwise_legacy | 1.4854 | -2.43531 | -263.95% | ✘ | ✘ |

（`oss_riemann` 列在本配置下与 `oss` 逐位相同：fsi2 用 `NoRiemannSolver`，Riemann 项本应为零 — 该列是自洽性检查而非独立样本。）

汇总（程序化，零手抄）：

* 移植算子行（oss + oss_riemann + oss_viscous）：0/21 在 15% 内（去重后 9 压力行 + 3 粘性行 = 12 个独立样本，0 通过）
* 压力支误差带：+24.23% … +107.54%
* 粘性支误差带：-38.54% … -38.01%
* panel 臂：7/9 在 15% 内，误差带 -0.70% … +32.26%
* 全部 39 行里 `fluid_state='solved'` 的行数：0

图：`docs/board_audit_m17452/fig_error_vs_threshold.png`（逐行误差柱状图 + 15% 硬线与 3% 目标线，由 `judge.json` 程序化生成）。

## 5. 发现（按严重度，附证据与复现）

### F1 — 阻塞·基准：移植算子 12/12 独立样本超 15%

* 证据：上表 + `evidence/judge.json`；平壁 +24.23…+28.14%，闭合体 +67.66…+107.54%，Couette 粘性 −38.01…−38.54%。且**误差不随 dx 收敛**（+27.59→+27.96→+28.14），说明不是分辨率问题。
* 复现：`python benchmarks/xval_fsi2/run.py`（本审计已在临时副本重跑，逐叶一致）。
* 影响：这是本单唯一的**能力主张**（“把 OSS 传递算子落到 TensorSPH 并在基准上验证”），主张不成立 → 不得计入 L4，maturity gate 应为 reopened。

### F2 — 阻塞·集成：模块无任何生产调用者

* 证据：`grep -rIl fsi_force_from_fluid src/ examples/` 只命中模块自身；`src/tensorsph/__init__.py` 中 `fsi_force` 命中：none (fsi module absent from tensorsph/__init__.py)；文档自己的移植决定是「不要把 panel 换成 OSS 算子」。
* 影响：不进主求解器 = 不是 L3。测试/基准里的调用不算集成。

### F3 — 中等·出处：声称「verbatim OSS 源码已归档」，实际未归档被引文件

`fsi_force_from_fluid.py` docstring 称 *the verbatim files are archived under `artifacts/xval_fsi2/oss_src/`*，并引 `force_on_structure.hpp:55-72,103-124`、`fsi2.cpp`。归档目录里只有 `hydrostatic_fsi.cpp` 与 3 个 XML，**`force_on_structure.hpp` / `fsi2.cpp` 均不存在**（`find` 全树无此名）。
→ 被引行号本地不可核；「零手抄」在 provenance 层面只做到「引用到了」，没做到「字节留档」。

### F4 — 中等·过度声明：跑前设计承诺「有 solved WCSPH 算例」，实际 0 个

`run.py` docstring 写 *One case-family (end-to-end) additionally uses a solved WCSPH field*；而 results.json 全部 39 行 `fluid_state='analytic'`，`solved` 行数 = 0。→ 端到端（求解器在环）被 doc 记为「已含」，实为未做；六段报告第 6 节说明了延迟理由，但设计声明与执行不一致。

### F5 — 中等·方法论：panel 的 −0.00% 是**同义反复**，不能当验证

panel 臂是核权最小二乘平面拟合，解析静水压力场本身就是线性场 → 精确复现是**定义使然**（被审自己的测试 `test_panel_is_exact_on_a_linear_pressure_field` 正是断 this）。同臂在闭合体上 +32.26%（dx=0.025），且随加密变差（−0.70→+20.08→+32.26，越细越差），上述二者都不能作为「算子已验证」的证据。
* 影响：任何把 panel 的 0.00% 计入「TensorSPH 传递算子精度」的结论都是失效的。

### F6 — 低·测试语义：把失败带「钉死」为期望值

`test_oss_operator_signs_and_documented_overload` 断言 `0.20 < err < 0.32`，`test_viscous_branch_...` 断言 `0.55 < got/ref < 0.70`。作为**变更探测器**是合理的，但它不是验证：`10 passed` 不代表能力确认，读者极易误读。审计不把测试通过计入 L4。

### F7 — 低·流程：交付物从未提交，任务以预算耗尽收尾

`git status --porcelain`：
* `?? benchmarks/xval_fsi2/`
* `?? docs/xval_fsi2/`
* `?? src/tensorsph/fsi_force_from_fluid.py`
* `?? tests/test_fsi_force_from_fluid.py`

`head_commit=null`，`fail_stage=budget_exhausted+forced_finish`，80/80 tool_calls。→ 没有 commit/PR 身份，返修必须以工作树快照为起点（本审计已 sha256 绑定，可据此恢复）。

## 6. 未做的检查（标 not_run，不标 passed）

* `not_run` 上游 SPHinXsys 本体编译运行（无 C++ 工具链预算）：因此“OSS 本身在 fsi2 上是否也过冲 28%”未知；被审 doc 也只把它当 hypothesis。
* `not_run` 耦合 fsi2 瞬态（梁尖位移 vs `BeamObserver_Position_Run_0_result.xml`、时均 `TotalViscousForceFromFluid`）——被审明确延迟到 round 2，本轮无可比数字。
* `not_run` 3D / 非均匀粒子分布 / 移动固体（Riemann 项非零）——本审计只覆盖被审已交付的 4 个解析算例族。
* `not_run` 数值稳定性/性能：无求解器在环，无 CFL 时间推进。

## 7. 过强的声明（原文摘出，逐条给判）

| 声明 | 出处 | 判 |
|---|---|---|
| 把 OSS 的 ForceFromFluid 传递算子「落到 TensorSPH」 | mission headline | **部分成立**：算子存在且可运行，但仅作为对照臂（comparison arm），未接入求解器，且文档建议不采用 |
| 「并用解析基准在 4 个算例×分辨率阶梯验证」 | mission headline | **不成立**：验证结果是 12/12 独立样本超 15%，属**证伪**（honest FAIL 有价值，但不是通过） |
| verbatim OSS 源码已归档 `oss_src/` | 模块 docstring | **不成立**（F3） |
| 端到端 solved WCSPH 算例含在内 | run.py docstring | **不成立**（F4） |
| panel 平壁 −0.00% 证明算子精度 | results.json `passes_15` | **不成立/同义反复**（F5） |

## 8. Disposition

**request changes（L2 + 基准 FAIL）**，理由与返修点：

1. 若目标仍是「OSS 算子可用」：必须先解释并消除 +28% 系统偏差（单侧求和 + 单层壳 + h=1.3dx 的组合），或按文档自述改用 OSS 体离散（填充体、dx_s=dx_f/2）再测——否则任何基于该算子的能力声明都无效。
2. 若要进 L3：把算子接到实际 FSI 路径上，并给出至少一个调用点 + 回归用例。
3. F3/F4/F5 三处声明必须改口径（归档被引文件；删掉未做的 solved 声明；把 panel 的线性精确性写成“同义反复检查”而非“验证”）。
4. 交付物必须提交（F7）：现在只有快照，无 commit/PR 身份。

不阻塞的建议：把 `oss_riemann` 的全零差异列显式标注为「自洽性检查」（已被审按其 doc 说明，但结果表未标）；把厚度敏感性（1 层 vs 3 层：oss +27.59% → +66.48%）提升为第一类证据，它比平壁误差更有诊断价值。

## 9. 审计产物清单

本仓 `.gitignore` 忽略 `artifacts/`，所以审计证据同时以**可提交副本**落在 `docs/board_audit_m17452/evidence/`（下表的同名文件），`artifacts/audit_m17452/` 为可重跑再生副本。

* `evidence/manifest.json` — 交付物 sha256/字节/存在性
* `evidence/git_state.json` — 分支/HEAD/porcelain/追踪状态
* `evidence/referee.json` — 注册裁判表查询结果（0 行）
* `evidence/judge.json` — 逐行 ref/got/err/通过门 + 级别规则
* `evidence/repro.json` — 重放记录（554 叶逐位比对）
* `evidence/finding.json` — 结论摘要
* `evidence/verdict.json` — 落表行（verdict id + 完整 evidence）
* `fig_error_vs_threshold.png` — 误差 vs 15% 硬线（逐行柱状）
* `fig_convergence.png` — 误差 vs dx 细化曲线（系统偏差非分辨率）
* `scripts/audit/m17452_board_audit.py` — 一键复跑（`--replay --insert-verdict`）

重跑命令：

```bash
python scripts/audit/m17452_board_audit.py --replay            # 取证+重放+判级
python scripts/audit/build_m17452_doc.py                        # 渲染本文档
python scripts/audit/m17452_board_audit.py --replay --insert-verdict --audit-mission 17494
```
