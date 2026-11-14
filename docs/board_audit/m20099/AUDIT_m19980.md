# 交差核验单 m20099 — CASE-SPH-DAMBREAK-PRESSURE / m19980（TensorSPH）

**对象**：`m19980`（溃坝压力峰值 · Koshizuka & Oka 1996 复测，TensorSPH，状态 succeeded / auto-approve dev_acceptance）
**核验方式**：非重做。独立取证 + 独立复跑（本单工作树 m20099），基准判据先行。
**裁定**：**L2**（真实现成立：输入→计算→输出，复跑逐位一致）；**L3/L4 拒绝**（无平台登记、无集成调用者；基准误差 1161% ≫ 12%，收敛性工件为零）。
**证据全部机器生成**：`docs/board_audit/m20099/verify_output.json`（由 `build_evidence.py` 从磁盘工件 + DB 现算，零手抄）。

---

## ① 取证：三件套逐项验收

### 1.1 run_artifacts（真实工件）

| 工件 | 位置 | 状态 |
|---|---|---|
| `result_dx0014.json` | `/root/tp-agent/work/m19980/benchmarks/koshizuka_dam_break/` | 存在，13 684 B，含 237 点压力时程 |
| `pressure_history.png / .pdf` | 同上 | 存在（120 865 B / 20 957 B） |
| `reference.json` | 同上 | 存在，P_ref=2700 Pa，tol=12% |
| `koshizuka.py` / `run.py` | 同上 | 存在（11 410 B / 4 617 B） |
| `result_dx0020.json` | — | **缺失**（报告 ⑤ 的 dx=0.02 行 24 028 Pa 无工件） |
| `results.json`（cj2420 应产出） | — | **缺失**（见 §①.4） |

**交付面缺陷**：被审工作树 `git status --porcelain` = `?? benchmarks/koshizuka_dam_break/`、`?? docs/benchmarks/koshizuka-dam-break-pressure.md`；mission `result.evidence.branch = null`；本单 main（m20099）`benchmarks/` 下 **无** `koshizuka_dam_break` 路径 → 全部交付物**未入库**，只在一次性工作树里存在。

### 1.2 benchmark：ref vs got（唯一裁判）

| 项 | 值 | 出处 |
|---|---|---|
| 参考值 P_ref | 2700 Pa（Koshizuka & Oka 1996, Fig. 15，探针 y=0.024 m） | `reference.json` |
| 容差 | <12% | 任务目标 / `tensor_case_pool` id=188 `tol_policy='<12%'` |
| got（档案） | 34 047.93 Pa | `result_dx0014.json` |
| got（本审计复跑） | 34 047.93 Pa | `audit/m19980_repro/out/rerun_dx0014.json` |
| 误差 | **1161.0%** | 现算 `abs(got-ref)/ref` |
| 判定 | **FAIL（超容差 96.8 倍）** | — |

**登记面核验**：`SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-DAMBREAK-PRESSURE'` = **0 行**；`tensor_case_pool` 中 `SPH-DAMBREAK-PRESSURE`（id=188）状态仍为 **dispatched**，mission_id=19980，tol `<12%`。即：本锚点在"唯一裁判"表里从未登记，机器门禁无行可比对；算例池未 passed。

**报告六段**：`docs/benchmarks/koshizuka-dam-break-pressure.md` 六段（①问题+ref出处 ②方法与实现 ③参数表含 Ma/Fr/Re ④ref-vs-got 对照打勾叉 ⑤收敛性/敏感性 ⑥失败模式+工件+repro_cmd）结构齐全 —— **但第 ⑤ 段有实证缺口，见 §①.4**；第 ④ 段判定自认 FAIL（未粉饰）。

### 1.3 repro_cmd 可执行性（本审计实跑）

```
cd audit/m19980_repro && PYTHONPATH=/root/tp-agent/work/m20099/src:. \
  python3 benchmarks/koshizuka_dam_break/run.py --dx 0.014 --json-out out/rerun_dx0014.json
→ exit 0，29.2 s wall，峰值 34047.9 Pa，误差 1161.0%，Passes tolerance: NO
```
`rerun_vs_archived_rel_delta = 0.0`（逐位一致）→ **可复现成立**。
但报告里给的 repro_cmd 无 `--json-out`，且默认 `steps=int(0.4/dt)+100=3397`，而档案是 4000 步（T=0.485 s）→ **命令不能原样重生成工件**（峰值仍在 0.309 s 处，故得数相同）。缺陷 D3。

### 1.4 "集群运行中"的实证核查（本单最重要发现）

报告 ⑤ 写 `dx=0.01 → (集群运行中)`、⑥ 写 `集群作业 cj2420（完整分辨率研究，排队中）`。实际：

| 来源 | 记录 |
|---|---|
| `cluster_jobs.id=2420` | `state=succeeded, rc=0, note='run rc=0'` |
| `/root/tp-agent/cluster/jobs/cj2420.log` | `{"job_id":"cj2420","node":"n330","rc":1,"output_tail":"bash: line 1: cd: /root/tp-agent/work/m19980: Permission denied"}` |

**该作业在节点上根本没启动（cd 权限拒绝，rc=1），却被队列记成 succeeded/rc=0**；`results.json` 从未产生。因此 dx=0.01/0.007/0.005 三点不存在，报告 ⑤ 的"峰值随分辨率变化不单调"只有 **2 个真实点**（0.02→24 027.8 Pa 已由本审计复跑证实、0.014→34 047.9 Pa），**收敛性/敏感性结论不成立**。

---

## ② 工件自洽性（档案内部矛盾）

从档案 `result_dx0014.json` 的 237 点时程现算（`build_evidence.py`）：

| 量 | 值 |
|---|---|
| 档案 peak_pressure_pa | 34 047.93 Pa |
| 档案 history 最大值 | **30 080.79 Pa** |
| 峰值在时程中"欠计"比例 | **11.65%** |
| 时程采样间隔 | 2.056 ms（每 50 步） |
| 5 ms 滑动平均峰值 | 12 827 Pa @ t=0.284 s |
| 20 ms 滑动平均峰值 | 6 318 Pa @ t=0.290 s |
| 超 ±12% 带的样本占比 | 34/237 = 14.3% |

结论：报告的 34 048 Pa 是**逐步跟踪到的单步声学尖峰**，存档时程（抽样）自身取不到这个数；若按工程惯用的 5 ms/20 ms 窗平滑，得 12.8 kPa / 6.3 kPa。即"压力峰值"缺少测量约定（采样率/平滑窗/探针宽度），而 12% 容差放在这个定义下**原则上不可达**。参照量级：探针处静水压 ρg·y = 235 Pa；ρ·c₀·v（c₀=30, v_max=3.28）= 98 kPa。

---

## ③ 判级（L0–L5）

| 级别 | 判据 | 本单结论 |
|---|---|---|
| L1 | 数学/单元验证 | 通过（实现有真实计算，非硬编码） |
| **L2** | **独立功能实现：输入→计算→输出** | **成立** — `koshizuka.py:124 wall_pressure_at_probe`（Adami 壁面密度外推 + 核权平均）、`:159 _compute_wall_density`、`:207 run_koshizuka`（WCSPH 推进，复用 `tensorsph.droplet_impact` 的 `rhs/apply_boundary/next_dt`）；输入(几何/参数)→计算(时间推进+探针)→输出(json/png/pdf)；本审计双点独立复跑逐位复现 |
| L3 | 系统集成（接入主求解器/平台） | **拒绝** — `grep -rn koshizuka src/ tests/` = 0；`tensor_benchmark_cases` 0 行；case_pool 仍 dispatched；模块未入库 |
| L4 | 基准验证（完整算例+误差分析+可复现） | **拒绝** — 误差 1161% ≫ 12%；收敛性工件为零；工件不可由 repro_cmd 原样重生成 |
| L5 | 科研工程成果 | 拒绝 |

> 级别只认功能实质，不认百页报告；本案报告档质量不低（六段齐、自认 FAIL），但**精度未达标 + 证据未入库**，故停在 L2。

---

## ④ 缺陷清单（按严重度）

| ID | 严重度 | 缺陷 | 证据 |
|---|---|---|---|
| D1 | **high** | cj2420 实为启动失败（rc=1 Permission denied），队列记 succeeded/rc=0，报告 ⑤⑥ 据此写"集群运行中/排队中"；收敛性结论实为 2 点 | `cluster_jobs` id=2420 vs `/root/tp-agent/cluster/jobs/cj2420.log` |
| D2 | **high** | 档案时程最大值(30 080.79) < 报告峰值(34 047.93) 11.65%；峰值对采样/平滑高度敏感（5ms→12.8 kPa）；"peak"无测量约定，12% 判据在该定义下不可达 | `verify_output.json.artifact_self_consistency` |
| D3 | medium | repro_cmd 与原档案不同参（无 `--json-out`，3397 步 vs 4000 步），不能原样重生成工件 | `run.py:57-60` vs `result_dx0014.json.simulated_time` |
| D4 | medium | 交付未入库：branch=null、两处 `??`、main 无此路径、`tensor_reports` 0 行；平台仅 auto-approve dev_acceptance | `git status --porcelain`、`tensor_dev_missions.result` |
| D5 | medium | 契约要求 honest-FAIL 留 `fail_stage`+`failure_reason`，实际 `fail_stage=''`、无 `failure_reason` 键（正文虽详述） | mission result JSON |
| D6 | medium | 基准未登记：`tensor_benchmark_cases` 0 行，机器门禁无行可判；case_pool id=188 停在 dispatched | DB 现查 |
| D7 | low | `probe_halfwidth=1.5` 为死参数（`wall_pressure_at_probe` 用固定 2h 核权），与文档②描述不一致 | `koshizuka.py:69` 无引用 |

**夸大表述**（保留原文、逐条标注）：
1. "压力峰值随分辨率变化不单调，表明数值振荡主导" —— 仅 2 个真实点，第 3 点不存在（D1），不构成敏感性证据。
2. "文献中 WCSPH 模拟同类问题的压力峰值通常在 5000–15000 Pa" —— 无引用、无工件，本审计**未核**（not_run）。
3. 报告 ⑤ `dx=0.01 (集群运行中)` —— 与 cj2420 实况不符（D1）。

## ⑤ 未执行的检查（诚实登记）

- 2700 Pa 的**文献真值复核**：仅能核到 DOI/图号指针，无读图数字化工件（像素坐标表/标尺），离线无法独立复核 → 标 **WITHHELD**，不计入 L4 的"基准"。
- 更高分辨率（dx≤0.01）复跑：本审计**未做**（属重计算，且非本单"交差核验"范围；若要补，应走调度递交）。
- ISPH 对照复跑：未做。

## ⑥ 处置（disposition）

**批准（L2 确认）+ 请求返修（算例维度）**：
1. 登记基准：向 `tensor_benchmark_cases` 插入 `anchor_slug='CASE-SPH-DAMBREAK-PRESSURE'` 一行（metric=壁面压力峰值，tolerance=<12%），并把 case_pool id=188 与之一致；否则门禁永远无裁判。
2. 修 cj2420 记账口径：`rc=1` 必须落为 failed，禁止 succeeded；报告里"运行中"的格子必须在终态后回填真实数字或删除。
3. 峰值测量约定写入 `reference.json`（探针宽度、采样率、是否平滑），并让档案时程能取到所报峰值（提高 output_every 或另存逐步峰值）。
4. 入库：把 `benchmarks/koshizuka_dam_break/`、`docs/benchmarks/koshizuka-dam-break-pressure.md` 提交到分支（当前全部 untracked），`repro_cmd` 带 `--json-out` 且与档案同参数。
5. 精度路线：12% 是跨方法（WCSPH↔MPS）判据，物理上不可达 —— 第二轮应么换不可压缩/ISPH 实现，要么把参考换成 WCSPH 文献值，并在 `reference.json` 记录出处；不要靠"方法差异"话术把 FAIL 说成 pass。

**结论一句话**：实现是真的（L2），复跑逐位一致（repro 成立），基准是硬 FAIL（1161%），报告六段齐且诚实自认失败，但收敛性证据与"集群在跑"不实、交付物未入库、基准未登记 —— 按 L2 记功，算例返修。
---

## ⑦ maturity_gate 预演（本审计自查，防"走过场"）

本单按 `maturity_gate.py` 的 `judge()` 对落表证据做本地预演（同一实现、同一 SCHEME）：

| 声明级 | judge 结果 | 未过项 |
|---|---|---|
| 2（本单） | **gate_level = 2，无缺项 → confirmed** | — |
| 3 | 2 | `integration_callers(grep 无引用)` |
| 4 | 2 | 同上（L4 亦要求 integration_callers） |
| 5 | 2 | `integration_callers`、`external_anchor(工件缺失)` |

即：L3/L4 的拒绝不是审计者的意见，而是机器判据在生态仓 grep 后给出的结论。

**顺带发现的生态缺陷（D8，medium，门禁侧）**：`maturity_gate.callers_verified()` 取
`str(callers[0]).split(":")[0]` 作为 grep 词。若审计者把 `integration_callers` 写成
`"none: 无调用方"` 之类的英文前缀，grep 词退化为 `none`，在 TensorSPH 全仓必然命中 →
**虚假 L3 通过**。本单初稿曾如此书写，预演时发现后改为不可命中的中文描述。建议门禁改为
"调用方必须是可解析的 `path:line` 或 `file.py` 且该文件内确有 import/调用"，而不是先 grep 前缀。
