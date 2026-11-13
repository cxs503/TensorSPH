# M17715 交差核验（board-audit · m17736）

**对象**：m17715 `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`（TensorSPH，status=succeeded，返修轮1 of m17594）
**性质**：交差核验，非重做。判据=tensor_benchmark_cases 唯一裁判 + 三件套（run_artifacts / benchmark ref-vs-got / repro_cmd 可执行）。
**结论**：**L2**（独立功能实现成立，L3 集成调用器实测报错，L4 报告数字与工件自相矛盾且无注册裁判行）→ `tensor_maturity_verdicts#624`（level=2, verdict_by=board-audit），disposition = **request changes**（返修）。

复现审计产物（本仓）：

| 产物 | 内容 |
|---|---|
| `scripts/audit/m17736_board_audit.py` | 一键重跑：快照 + git 态 + 裁判表 + 干净副本复现 + 报告↔工件比对 + 落表 |
| `scripts/audit/m17736_figure.py` | 图 1（本仓 vs 报告 vs 审计记分板） |
| `docs/board_audit_m17715/snapshot/` | 被测 6 个交付文件 + `src/tensorsph/__main__.py` 的字节副本 + sha256 清单 |
| `artifacts/audit_m17715/{delivery,referee,repro,report_consistency,finding}.json` | 每个结论的原始证据 |
| `artifacts/audit_m17715/m17736_audit.png` | 图 1 |

---

## ① 案例与被测对象（物理背景 / 几何 / 参数）

二维静止液滴的 Young-Laplace 压差：投影（不可压）SPH 里压力是 `div u = 0` 的拉格朗日乘子，液滴的静态跳变由 Poisson 源（CSF 体力）承担，而不是由 Tait 状态方程承担。基准量是解析二维律 `dp = sigma/R`（圆：曲率 `1/R` 与 `0`）。

| 量 | 值 | 出处 |
|---|---|---|
| `R` / `sigma` / `rho0` | 0.02 m / 0.072 N/m / 1000 kg/m³ | `results/xval_projection_laplace.json:params` |
| 载体 | 等密度（1:1）蒸汽区，外延到 `R + 0.012 m` | 同上 `outer_m=0.012` |
| 格点 | 六角，`dy = dx*sqrt(3)/2`；外 1.6 dx 层 `p = 0` Dirichlet | `docs/xval_projection_laplace_oss.md:64-78` |
| 阶梯 | `dx = 2.0 / 1.5 / 1.0 mm` → `R/dx = 10 / 13.33 / 20` | `results...json:params.levels_m` |
| 核 / 支持域 | 二维三次样条，`sigma_3 = 10/(7 pi h^2)`，`h/dx = 1.3`（敏感性 1.2 / 1.5） | 同上 |
| 理论值 | `dp_theory = sigma/R = 3.6000 Pa` | 计算得 `0.072/0.02` |
| 判据出处 | 任务目标 text `<15%`（owner 精度标准），报告自设 3 % | mission m17715 goal |

**裁判（关键）**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE'` → **0 行**（表内共 57 行）。即机器裁判**缺位**，唯一在册容差只存在于任务文本与代码常量里（`benchmarks/xval_projection_laplace/run.py:80` `TOL = 0.15`）。见 `artifacts/audit_m17715/referee.json`。

## ② 怎么做的（审计流程）

1. **取证**：`git status`/`rev-list` 定交付态；把 6 个交付文件 + 被改的 `__main__.py` 逐字节拷进本仓 `docs/board_audit_m17715/snapshot/` 并出 sha256 清单（`delivery.json`）。
2. **三件套核验**：工件是否真存在、是否由脚本生成、`repro_cmd` 是否跑得动。
3. **独立复现**：把被测树（排除 `.git` / 旧 `results`）打成干净副本，`PYTHONPATH=src`（a）跑 `pytest tests/test_xval_projection_laplace.py -q`；（b）跑 `benchmarks/xval_projection_laplace/run.py` 全阶梯；（c）跑集成调用器 `python -m tensorsph --case xval_projection_laplace`；（d）`pysph_parity_check()` 对已安装 PySPH 1.0b3.dev0 取算子级数值。
4. **报告↔工件比对**：用正则从 `docs/xval_projection_laplace_oss.md` 抽出 ④ 阶梯表 7 行与 ⑤ 敏感性表 11 行，逐格对 `results.json` 比较（`report_consistency.json`）。
5. **判级 + 落表**：按 L0-L5 定义（功能实质，不看报告厚度）给 level，写 `tensor_maturity_verdicts`（本任务 17736，scope `m17715 XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`）。

## ③ 计算结果（数字表，全部程序化抽取）

**（a）阶梯 `dp` vs 解析 `sigma/R = 3.6000 Pa`——工件侧（我复现的数值，与被测 `results.json` 逐叶相同）**

| arm | R/dx | got dp [Pa] | ref [Pa] | err [%] | 15 % gate | <3 % ? |
|---|---|---|---|---|---|---|
| ours (MLS 曲率) | 10 | 3.63497 | 3.600 | +0.971 | PASS | 是 |
| ours | 13.33 | 3.62182 | 3.600 | +0.606 | PASS | 是 |
| ours | 20 | 3.59232 | 3.600 | −0.213 | PASS | 是 |
| oss (paper ψ) | 10 | 3.74396 | 3.600 | +3.999 | PASS | **否** |
| oss (paper ψ) | 13.33 | 3.72399 | 3.600 | +3.444 | PASS | **否** |
| oss (paper ψ) | 20 | 3.70186 | 3.600 | +2.830 | PASS | 是 |
| oss (shipped ψ=2/(V_i+V_j)) | 10 | 2.684e+22 | 3.600 | +7.45e+23 | FAIL（已知移植陷阱） | 否 |

**（b）报告 ④/⑤ 表 vs 工件（格子级比对）**

| 比对项 | 不一致 / 总行 | 举例 |
|---|---|---|
| ④ 阶梯表 | **4 / 7 行** | ours R/dx=10：报告 3.565 Pa / −0.97 % vs 工件 3.635 Pa / **+0.97 %**（符号+数值镜像，两行同病）；oss R/dx=13.33：3.686 / +2.39 % vs 3.724 / +3.44 %；oss[code]：3.600e14 / +1.00e16 vs 2.684e22 / +7.45e23（差 8 个量级） |
| ④ `kappa*R` 列 | **7 / 7 格** | ours 0.998 / 0.999 / 1.000 vs 工件 1.0111 / 1.0079 / 1.0035 |
| ⑤ 敏感性表 | **5 / 11 行** | `h_factor=1.2`：−0.31 % vs **−2.39 %**；`h_factor=1.5`：+0.15 % vs **+1.66 %**；`fit_quadratic`：−0.18 % vs **−24.46 %**（15 % gate FAIL）；`morris=False`：+5.67 % vs **+2.73 %** |

报告第 6 行声称"every number in this report"由脚本生成、零手抄——对 §④/§⑤ **不成立**。

**（c）三件套 + 集成**

| 检查 | 结果 | 证据 |
|---|---|---|
| run_artifacts 存在 | JSON（9 runs）+ PNG 齐，sha256 记录在案 | `delivery.json:manifest` |
| 工件可再生成 | 干净副本重跑：JSON 逐叶 **0 处差异**（排除 wall-clock），PNG **字节相同** | `repro.json:json_leaf_diff_count=0` |
| 测试 | `8 passed, 1 skipped`（skip = 副本里没有旧 results.json 的"结果一致性"用例；带上旧工件即 9 passed） | `repro.json:pytest` |
| repro_cmd | `PYTHONPATH=src python3 benchmarks/xval_projection_laplace/run.py` rc=0，全阶梯 3 level × 3 arm + 敏感性 + 图 | `repro.json:harness_stdout` |
| **集成调用器** | `python -m tensorsph --case xval_projection_laplace` → **rc=2**，`unrecognized arguments: --case xval_projection_laplace`（内层 harness 的 argparse 抓了 sys.argv）| `repro.json:integration_caller` |
| OSS 算子对拍 | `SmoothedColor 2.2e-16`、`gradC 4.8e-7`、`normal 2.2e-16`、`SY11δ 2.4e-7`、`CSF acc 0.0` 一致；`Curvature_kappa 1.28e9`、`PressureCoeff_diag 1.75e4`、`delta=1/gradC 2.08e9` 未对上 | `repro.json:parity` |
| 交付态 | 被测分支**自有 commit 0 个**（HEAD 6c9939a 反落后 main 两个 merge commit），，6 个交付文件全部 **untracked** | `delivery.json` |

## ④ 结果图

图 1 `artifacts/audit_m17715/m17736_audit.png`：
（a）阶梯 ④：报告表（虚线）vs 工件（实线），可见 ours 前两级符号翻反、oss 中间一级偏离；
（b）敏感性 ⑤：11 行逐项报告值 vs 工件值（symlog），`fit_quadratic` 一项报告 −0.18 % 而工件 −24.46 %；
（c）审计记分板：七项检查 3 通 4 不通（making artifacts / PNG / repro_cmd 通；集成调用器 / 报告=工件 / 注册裁判 / 已提交 不通）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

**坑（被测方）**
1. **集成调用器是死分支**：`src/tensorsph/__main__.py:91-114` 选 `--case xval_projection_laplace` 后 `from benchmarks...run import main` 再 `main()`，而被导入的 `main()` 自己 `argparse.parse_args()` 读 `sys.argv`，于是 `--case` 被当成未知参数直接 `SystemExit(2)`。任何形态（带 `--quick` / `--no-plot`）都失败——m17715 result 里"`__main__.py --case xval_projection_laplace: 集成调用验证`"这句是**不成立的**。
2. **零提交**：所有交付都在工作树里未提交，`agent/m17715` 自有 commit 数为 0（`rev-list --count origin/main..HEAD` = 0）。平台"验收后自动推分支"会推一个空分支，worktree 一清理成果即灭。m17594 的前身分支 `agent/m17594` 携带 `scripts/xval_projection_laplace/` 也未合入 main（`git branch -a --contains cc48564` → 仅 `agent/m17594`）——两轮都把能力留在 main 之外。
3. **报告盖不住回归**：`laplace_isph_case()` 的出厂默认 `curvature_method='fit_quadratic'`（`src/tensorsph/laplace_isph.py:262`）实测 −24.46 %（15 % gate FAIL），harness 只因 `run.py:73` 显式传 `mls` 才过；报告 ⑤ 却写 `fit_quadratic = −0.18 %`。上一轮审计（m17594 F1）已把这个默认值标为待改，本轮未改且被报告掩盖。
4. **移植不等于跑 OSS**：`oss` 臂是 PySPH 算子**在 TensorSPH 内**的重实现（算子级对拍 5/9 一致，3 项未对上），不是端到端跑 PySPH ISPH；参考实现的数值输出并无独立基线。报告 §⑤ 已如实列出未移植项（归一化颜色、自由面边界、求解器、时间推进），这一点是合格的，但"vs PySPH"的措辞不能读成"vs PySPH 跑出来的数"。
5. PySPH 只在遗留 venv `/tmp/xv2`（`oss_projection.py:310` 硬编码路径），干净镜像里对拍腿会退化为 skip——可复现性依赖宿主机残留。

**局限（审计方）**
- 只核核验、不重做：我没有独立重推投影 SPH 的物理公式，`ours` 臂数值的"正确性"以解析 `sigma/R` 为唯一外部锚。
- 单机（loongarch64 宿主）执行，未上集群；未跑全仓 pytest；未做时间推进（静态验证）。
- 报告↔工件比对是**格子级**而非语义级：抽取靠正则，行对齐靠 `arm 前缀 + R/dx`。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 参考 | got | 误差 | <3 %？ |
|---|---|---|---|---|
| ours dp @ R/dx=10 | 3.600 Pa | 3.63497 Pa | +0.971 % | 是 |
| ours dp @ R/dx=13.33 | 3.600 Pa | 3.62182 Pa | +0.606 % | 是 |
| ours dp @ R/dx=20 | 3.600 Pa | 3.59232 Pa | −0.213 % | 是 |
| oss(paper) dp @ R/dx=20 | 3.600 Pa | 3.70186 Pa | +2.830 % | 是（10、13.33 两级 3.999 / 3.444 % > 3 %，报告已如实披露） |
| 报告 ④ 表 vs 工件（ours 3 行） | — | — | 2 行符号+数值不一致 | **否**（这是本次判级的致命项） |
| 报告 ⑤ `fit_quadratic` | 工件 −24.46 % | 报告 −0.18 % | 差 24.3 pp | **否** |

**一句话**：物理量本身对在 <1 %（ours）/ <4 %（移植臂）以内，但**交付形态不合格**——集成调用器跑不通、一个 commit 都没有、报告 ④/⑤ 的 9 处数字与工件不符（含把 −24.46 % 写成 −0.18 %）。级别=L2，`request changes`。

## 判级依据（L0-L5）

| 级别 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 只有说明 | — |
| L1 数学/单元验证 | 公式级验证 | — |
| **L2 独立功能实现** | 真代码 输入→计算→输出 | **成立**：`oss_projection.py` 498 行 11 个公开函数（无 TODO/pass/NotImplemented 占位）+ `run.py` 入口 + 可字节复现的 JSON/PNG 工件 |
| L3 系统集成 | 接入主求解器/平台 | **不成立**：唯一调用点 `__main__.py:91-114` 实测 rc=2 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | **不成立**：报告 ④/⑤ 与工件不符（9 处）、锚点无注册裁判行、零提交（无仓内复现路径） |
| L5 科研工程成果 | 新知+工程闭环 | — |

## 返修清单（交下一轮）

1. `git add` 全部交付（`oss_projection.py`、harness、tests、results、doc）并开 PR 合入 main。
2. 修 argv 打架：harness 暴露 `run(quick=False) -> dict`，`__main__` 直接调；与人机 CLI 解耦。
3. ④/⑤ 表改为由 `results.json` 程序化渲染（`make_doc.py` 已有先例），去掉手抄；顺带修 `kappa*R` 列。
4. 在 `tensor_benchmark_cases` 注册本锚点一行：`metric = dp vs sigma/R`，`tolerance = 15 %`（或报告自设的 3 %，二者择一并写明）。
5. `laplace_isph_case` 默认 `curvature_method` 翻成 `'mls'`（或让默认路径过 gate），否则出厂路径 24 % 偏差。
6. 发 `tensor_reports`（mission 17715 目前 0 行）。