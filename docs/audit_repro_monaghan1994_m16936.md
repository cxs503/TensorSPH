# 对抗审计：REPRO-SPH-MONAGHAN1994 证据束（m16936）

**审计对象**：m16873 合并进主线的溃坝前沿轨迹复现工件 —— `benchmarks/repro_monaghan_1994/`
（`run.py`、`make_artifacts.py`、`refs.json`、`results/summary.json` 7944 行、
3 个 `trajectory_*.json`、`results/refs_verified.json`）与由其生成的
`docs/repro_monaghan_1994.md`（commit `74ed981`，merge `74ed981` / PR #595）。
**本单性质**：独立复核（adversarial review）+ 缺陷修复 + 缺测补齐，锚点 `REPRO-SPH-MONAGHAN1994`（台账 open）。
**独立性声明**：本审计不读 m16873 的推理过程，只读它的工件、代码与合并后的文档；所有数值由我自己的
插值/度量实现重算（`tests/test_repro_monaghan1994_evidence_contract.py` 内 `_metric`），不复用
`run.py::compare`。审计者可用的证据边界见 §4。

## 1. 序：这份证据束凭什么可信

复现类工件的可信度不是「跑完了」，而是四件事同时成立：参考值有出处、判据先于运行、
数字可从工件重算、文档里的话与工件里的数一致。逐项检查结论：

| 维度 | 结论 | 证据 |
|---|---|---|
| 参考标识 | 合格 | `results/refs_verified.json`：Crossref 按规则命中 DOI `10.1006/jcph.1994.1034`（JCP 110:399-406），OpenAlex `oa_status=closed`；论文数值未填，T4 记 WITHHELD |
| 判据先于运行 | 合格 | `summary.json:protocol` 写明 window τ∈[0.5,3.0]、容差 15 % 幅值、压缩限 0.1，且 `run.py` 顶部常量与之一致 |
| 数字可重算 | **合格（比预期强）** | 我用独立实现从 3 个 `trajectory_*.json` + 两张参考表重算 6 组（3 分辨率×2 参考）MAE/幅值/偏置/幅值，与 `summary.json` 逐值吻合到 1e-9（测试 `test_summary_metrics_rederive_from_committed_trajectories`） |
| 可重跑 | **合格（bit-exact）** | 同机重跑 dx=0.04、0.02 两个算例，与已提交轨迹逐点差 **0.000000** in Z/a，`max|ρ/ρ0−1|` 也一致（0.0192 / 0.0263），见 `results/front_estimator_probe.json` |
| 文档与工件一致 | **不合格 → 已修** | 见 F1 |
| 缺测保护 | **不合格 → 已补** | m16873 的合并**没有附带任何测试**（`git show --stat 74ed981` 无 tests/），文档数字可以静默漂移；见 F3 |

## 2. 发现（按严重度排序）

### F1（Blocker，已修）文档结论与自身表格互相矛盾，且属于「抬级」

`docs/repro_monaghan_1994.md` ⑥ 的表格行写的是
`MAE/幅值 = 4.38 % … 是否 < 3 % = 否`，而同一节末句断言
「**对唯一通过结构审计的参考，前沿轨迹误差在 3 % 以内（回答「是」）**」。
4.38 % 不在 3 % 以内：这是把平台 15 % 验收线的 pass 借用成论文 3 % 档的 pass。
根因不在文档而在**生成器**：`make_artifacts.py:407` 把结论句**硬编码**，而表格的「是/否」
由 `primary['mae_over_amplitude'] < 0.03` 算出——两条路径各有各的真值来源，必然可以互相背离。

* 复现：`git show 74ed981:benchmarks/repro_monaghan_1994/make_artifacts.py | sed -n '393,410p'`，
  对照 `git show 74ed981:docs/repro_monaghan_1994.md | tail -3`。
* 修复：结论句改为由同一变量生成，并显式点名两条线不同：
  `前沿轨迹误差 4.38 %，不在 3 % 以内（回答“否”）；平台验收线（15 % 幅值）为 pass——两条线不是同一条`。
* 守卫：`test_document_verdict_matches_the_artifact_numbers`。**反向对照已做**：把旧断言塞回文档，
  该测试立刻红（1 failed），恢复后 6 passed —— 即守卫测的是不变量，不是「跑通就算过」。

### F2（Major，已补诊断）「趋势 gate 红」被如实报告但没有被解释，读者会把红读成求解器不收敛

诊断值本身是对的：`resolution_trend_vs_independent = fail`。但原因是 **|MAE| 过零**，不是误差不降。
工件里本来就有解释所需的量（`comparisons[case][ref].bias`），文档只印了 MAE：

| dx [m] | MAE/幅值 | 偏置 (sim−ref) | 偏置/幅值 |
|---|---|---|---|
| 0.04 | 4.67 % | −0.125 | −4.67 % |
| 0.02 | 1.65 % | +0.032 | +1.20 % |
| 0.01 | 4.38 % | +0.115 | +4.29 % |

偏置随加密单调上移（+5.86 %、+3.09 % 幅值/级）并**穿过零点**：dx=0.02 是「误差恰好在过零点附近」
的巧合，不是精度更好；最细两级偏置只再变 3.09 % 幅值，残余 4.29 % 幅值是与分辨率无关的固定偏移，
按定义不能归因于离散误差（要么是参考/实验条件差，要么是缺一项物理）。
* 修复：`make_artifacts.py` 的 3.3 节增加偏置列 + 由数据判定的诊断句（单调/翻转/饱和都是算出来的，不是写死的）。
* 守卫：`test_ladder_is_non_monotone_only_because_the_bias_crosses_zero` 同时钉住三件事——
  |MAE| 非单调（观察）、偏置变号（诊断）、gate 仍为 fail（不得为了变绿改归一化）。

### F3（Major，已补）证据束零测试

m16873 是纯工件+文档合并，无 tests/。仓库惯例（`tests/test_verification_ledger.py` 对全仓台账做漂移
检测）在这里缺席：`summary.json` 与轨迹漂移、文档数字与工件漂移、结论句再次被硬编码，全都不会被发现。
* 修复：新增 `tests/test_repro_monaghan1994_evidence_contract.py`（6 测试，5.7 s，纯读工件不起仿真）。

### F4（已证伪的假设）判据量被单颗飞溅粒子决定？

假设 H1：`front_position` 是全体粒子 `x` 的 **max**，细网格若抛出一颗飞溅粒子，前沿读数会被抬高，
阶梯非单调性是估计器伪影。为此新增 `front_estimator_probe.py`：用同一 `DamBreakConfig` 重跑
（不改 `src/`，运行时把模块级 `front_position` 包一层记录全分布），在每个输出时刻记录
max / 99.5% / 99% / 95% 分位与「领先 99.5% 分位 > 2dx 的粒子数」。

结果（`results/front_estimator_probe.json`，dx = 0.04、0.02）：

| dx [m] | 估计器 | MAE/幅值 | 偏置 | 领先 99.5% 分位 >2dx 的粒子数（各时刻最大） |
|---|---|---|---|---|
| 0.04 | `x_max` | 4.67 % | −0.125 | 0 |
| 0.04 | `p995` | 5.38 % | −0.144 | 0 |
| 0.04 | `p99` | 6.09 % | −0.163 | 0 |
| 0.04 | `p95` | 12.43 % | −0.332 | 0 |
| 0.02 | `x_max` | 1.65 % | +0.032 | 0 |
| 0.02 | `p995` | 1.10 % | +0.002 | 0 |
| 0.02 | `p99` | 1.41 % | −0.034 | 0 |
| 0.02 | `p95` | 11.26 % | −0.301 | 0 |

**H1 被证伪**：所有输出时刻都不存在飞溅粒子（计数恒为 0），分位前沿一致地在 max 之后，
估计器换成 99.5% 分位只把 dx=0.02 的误差从 1.65 % 挪到 1.10 %，非单调性依然存在。
也就是说残余差是**求解器/参考的**，不是读数的。这条负面结果比「通过」更有用：
它把后续排查方向从「换估计器」永久排掉。

### F5（Minor，已修）小节编号与「精度上限」误导

- 3.6 节初版被插在 3.4 之前，改为 3.3b。
- 3.2 节 `命中 ±0.05 比例 12 %` 与 3.4 节 `落在双表带内 96 %` 并列出现时容易被读反：前者是
  「仿真 vs 单表的声明不确定度」，后者是「仿真落在**两表之间**」。文档已在 3.4 说明是双表带；
  本审计只登记，不再改动措辞（非阻断）。

## 3. 超出审计范围的判断（不裁决）

* 两份数字化互不自洽（最大分歧 0.851 in Z/a = 17.0× 声明不确定度 ±0.05）是**参考侧**问题，
  m16873 已如实登记 `reference_self_consistency = fail` 并拒绝用合成表判级，本审计同意该处置，
  但**不**裁决哪一份数字化更接近 Martin & Moyce (1952) 原文——原文在仓内不可得。
* 论文自身表/图数值仍 WITHHELD；本次没有、也无法用任何转述值替代（若后续拿到 PDF，
  按 `refs.json:T4.closure_condition` 带坐标转录后重跑 `--stage accept`，不需改求解器）。

## 4. 未做的检查（unknown，不是 passed）

1. **dx = 0.01 的估计器探针未跑完**：本地单跑约 300 s（超过本单单次命令时限），集群递交 cj1378
   排队 527 位后本单主动 kill（同时本机 load15 一度升到 142，按容量纪律不再起重算）。因此 F4 的
   估计器结论只覆盖 0.04/0.02 两级；dx=0.01 仅有已提交 `x_max` 轨迹（其 bit-exact 可重跑性也
   只有 0.04/0.02 两级被实测确认）。登记为**未完成**，不外推。
2. 与 Monaghan (1994) 原文的比较未做（全文不可得，见上）。
3. `benchmarks/dam_break/martin_moyce_1952_crosscheck.json` 的数字化精度未与 1952 原文核对
   （付费墙），只核对到「与 Lethe 官方算例分发值逐值同源」这一层，这是本算例误差地板的下界来源。

## 5. 处置

**request changes → 已改（changes applied）**：F1 Blocker 已修并加反向对照守卫，F2 诊断已程序化补入
文档，F3 守卫测试已补，F4 假设已证伪并留档，F5 编号已修。剩余唯一硬缺口是 dx=0.01 的探针级
（§4.1），如实登记为未完成。**不批准**把本算例的 4.38 % 表述为「3 % 以内」；
4.38 % 对 15 % 平台线为 pass、对 3 % 论文档为否，两者必须分列。

## 6. 复核入口

```bash
PYTHONPATH=src python -m pytest tests/test_repro_monaghan1994_evidence_contract.py -q   # 6 passed
PYTHONPATH=src python benchmarks/repro_monaghan_1994/front_estimator_probe.py --dx 0.04,0.02 --seconds 0.34 --check-committed
PYTHONPATH=src python benchmarks/repro_monaghan_1994/make_artifacts.py                  # 文档/图重生成，无额外 diff
```