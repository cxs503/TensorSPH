# 账本自证审计：被「选择」掩盖的 credit（m6510）

**范围**: TensorSPH ｜ **性质**: 验证欠账（对验证器本身的复核）｜ **改动**: `benchmarks/verification_ledger/build_ledger.py`（新规则 + 4 条未筛选读数）、`ledger.json` / `docs/verification_ledger.md`（重生成）、本文件（由 `make_overcredit_doc.py` 生成）、`tests/test_verification_ledger_overcredit.py`。全部数字由脚本注入，零手抄。

## 1. 案例介绍：账本回答错了哪一个问题

`benchmarks/verification_ledger/` 把仓里每个机器可读验证工件收敛成一张表，按全仓唯一的 3 % / 15 % 阶梯给读数定级。它的分级规则只看两件事：

1. 读数与参考值的相对误差；
2. 该读数是否被自己的量具/有效性门 disqualify（`gauge_degenerate`、`run_validity`、`admissibility`）。

两条都不看的第三件事是：**这条读数是从什么样的运行集里挑出来的**。`threed_shear` 的衰减率取阶梯最细一级（n = 24），同一个工件的 `verdict` 是 `fail`，因为 h/dx = 1.8 的用例根本没跑出来（`missing_cases: [h1.8]`）；`threed_hydrostatic` 的底压取 6 个算例里最好的一个（adami, c0 = 20），同一工件的 `verdict` 也是 `fail`，因为它自己的判据 `interior_pressure_within_15pct_best` 与 `bottom_pressure_within_15pct_all` 都不成立。于是账本给出 PASS，而工件给出 FAIL —— 两个都对，但它们说的不是同一句话：前者说的是「这个读数合规」，后者说的是「这组运行没过」。

## 2. 怎么做的

### 2.1 让工件的自述 verdict 随读数一起走

`build_ledger.py` 新增 `declared_verdict(artifact)`：从工件顶层 `verdict` / `status` / `evidence_grade`（含 dict / bool / `worst_within_15pct` 形态）归一出 `pass` / `fail`，并把工件自身给出的失败原因（`criteria` 里为 false 的项、`missing_cases`）一起记进条目字段 `declared_verdict`。credits（PASS/MARGINAL）另带 `credit_qualified`：`False` = 工件自述失败，`None` 不出现——缺声明记 `True`，但计入 `n_credits_without_declared_verdict` 与合格的 credit 分开统计。

### 2.2 新 finding 规则 `credit_from_failing_artifact`（high）

凡 PASS/MARGINAL 且 `declared_verdict == fail` 的条目，一律进 findings，detail 里点名工件的失败项/缺失用例。规则本身是纯函数 `credit_findings(entries)`，用合成条目做单元测试（只在 PASS/MARGINAL + 自述失败时触发），与读工件解耦。

### 2.3 把被「选择」掉的读数补登记

选择本身不算缺陷，**只登记被选中的那个**才算。同一批工件里未筛选的读数已补进账本：

| 新 claim | 读数 | 补的是什么 |
|---|---|---|
| `threed_shear.decay_rate.worst_of_ladder` | -12.62 % | 同一阶梯最粗一级（对照 finest 的选择偏差） |
| `threed_hydrostatic.bottom_pressure.adami_worst` | -8.74 % | 同一组 Adami 算例最差一个（对照 best） |
| `threed_hydrostatic.profile.mean_rel_error_core.adami_best` | -11.92 % | 同一算例的内区压力剖面均值误差（底压只代表一层） |
| `threed_hydrostatic.profile.max_rel_error_core.adami_best` | +43.17 % | 同一算例的内区逐层最差误差（奇偶模态） |

## 3. 计算结果

### 3.1 取自「自述失败」工件的 credit

| claim | 等级 | rel err | 工件自述 |
|---|---|---|---|
| `threed_hydrostatic.bottom_pressure.adami_best` | **PASS** | -2.75 % | `benchmarks/threed_hydrostatic/report.json` → `verdict = fail`（未通过项 `bottom_pressure_within_15pct_all`、`centre_of_mass_drift_below_1pct`、`column_settled_window_drift_below_2pct`、`compressibility_helps_as_predicted`、`interior_pressure_within_15pct_best`） |
| `threed_hydrostatic.bottom_pressure.adami_worst` | **MARGINAL** | -8.74 % | `benchmarks/threed_hydrostatic/report.json` → `verdict = fail`（未通过项 `bottom_pressure_within_15pct_all`、`centre_of_mass_drift_below_1pct`、`column_settled_window_drift_below_2pct`、`compressibility_helps_as_predicted`、`interior_pressure_within_15pct_best`） |
| `threed_hydrostatic.profile.mean_rel_error_core.adami_best` | **MARGINAL** | -11.92 % | `benchmarks/threed_hydrostatic/report.json` → `verdict = fail`（未通过项 `bottom_pressure_within_15pct_all`、`centre_of_mass_drift_below_1pct`、`column_settled_window_drift_below_2pct`、`compressibility_helps_as_predicted`、`interior_pressure_within_15pct_best`） |
| `threed_shear.decay_rate.finest` | **PASS** | -2.68 % | `benchmarks/threed_shear/report.json` → `verdict = fail`（缺失用例 `h1.8`） |
| `threed_shear.decay_rate.raw_laplacian` | **MARGINAL** | +11.46 % | `benchmarks/threed_shear/report.json` → `verdict = fail`（缺失用例 `h1.8`） |
| `threed_shear.decay_rate.worst_of_ladder` | **MARGINAL** | -12.62 % | `benchmarks/threed_shear/report.json` → `verdict = fail`（缺失用例 `h1.8`） |

合计 **6** 条；另有 **6** 条 credit 的工件根本没有声明 verdict（等级只能读成「该读数合规」，不能读成「该能力被验证」）。工件自述 `fail` 的条目共 **12** 条。

### 3.2 账本规模与分级分布（前 → 后）

| 指标 | 基线（HEAD 提交的 ledger.json） | 本次重生成 |
|---|---|---|
| claims | 36 | 40 |
| 物理量 | 18 | 20 |
| findings | 8 | 19 |
| PASS | 7 | 7 |
| MARGINAL | 5 | 8 |
| FAIL | 9 | 10 |
| WITHHELD | 13 | 13 |
| WITHDRAWN | 2 | 2 |
| findings 分类 | declared_vs_derived 1，inadmissible_run 3，quantity_spread 3，run_validity 1 | credit_from_failing_artifact 6，declared_vs_derived 1，inadmissible_run 3，quantity_spread 4，run_validity 4，uncovered_benchmark 1 |

### 3.3 误差判定（逐项回答是否 < 3%）

| 关键量 | 基准 | 实测 | 误差 | 是否 < 3% |
|---|---|---|---|---|
| `threed_shear.decay_rate.worst_of_ladder` | 3.94784 1/s | 3.44962 1/s | -12.62 % | 否 |
| `threed_hydrostatic.bottom_pressure.adami_worst` | 1520.55 Pa | 1387.58 Pa | -8.74 % | 否 |
| `threed_hydrostatic.profile.mean_rel_error_core.adami_best` | 0 % | -11.9227 % | -11.92 % | 否 |
| `threed_hydrostatic.profile.max_rel_error_core.adami_best` | 0 % | 43.1741 % | +43.17 % | 否 |
| `threed_shear.decay_rate.finest` | 3.94784 1/s | 3.84214 1/s | -2.68 % | **是** |
| `threed_hydrostatic.bottom_pressure.adami_best` | 1520.55 Pa | 1478.77 Pa | -2.75 % | **是** |

分级结论与上面逐项判定一致（PASS 需 ≤ 3 %，MARGINAL ≤ 15 %）：两条 PASS 级读数（-2.68 % / -2.75 %）本身**确实**在 3 % 以内，但它们的工件自述 fail，所以「3-D 剪切衰减/底压已通过验证」这句话不成立——本单没有任何误差被放宽或以更严的口径重测，改的只是**可引用性**。

## 4. 图

本单是证据审计（读 JSON、改规则、加登记），不产生场量；唯一可画的是分级分布柱图，其数值已在 §3.2 表内逐项列出，且已有 `benchmarks/verification_ledger/ledger_grades.png` 由账本自身生成，故不重复出图。

## 5. 注意点 / 局限

1. **没有改任何物理等级**：`credit_from_failing_artifact` 只把「可引用性」显式化，PASS/MARGINAL 的派生规则一字未动，避免审计者顺手改判据。
2. **工件自述 verdict 的粒度是运行集，不是读数**：`threed_hydrostatic` 的 fail 来自内区剖面与沉降判据，与底压那一层 −2.75 % 并不矛盾；所以正确读法是「底压这一层合规、整个验算不通过」，而不是「底压读数错」。
3. **缺声明 ≠ 通过**：`n_credits_without_declared_verdict` 大于 0 说明多份工件的「整体是否通过」根本没被机器表达，这部分欠账要各 benchmark 自己补 verdict 字段。
4. **本单没修的**：`threed_shear` 的 `h1.8` 用例仍然缺失（补齐需 3-D 8000 粒子重算，本机 load15 ≈ 120，按容量纪律不在此单起算力）；`threed_hydrostatic` 的奇偶压力模态（逐层 −6 %…−43 %）仍是未解物理缺陷。两者的关账路径都写在 `docs/verification_threed.md`。
5. **两条旁证（HEAD 上的红，不是本单引入）**：
   - 重生成时 `run_validity` 命中的条目从 1 条变成 4 条（同一批 id 由 `claim_validity` 门判为 INVALID）；
   - HEAD 的 `test_no_benchmark_harness_is_uncovered` 也红：`benchmarks/ice_floe_added_mass/` 有 run.py 但无 claim 无豁免，而 HEAD 提交的 ledger.json 里没有这条 finding。
   两条都指向同一件事：**HEAD 提交的 ledger.json 不能由 HEAD 自己的代码复现，而 `--check` 却报 up-to-date**（anti-drift 机制存在盲区）。本单重生成后「提交副本 == 重生成」由 `test_committed_ledger_is_still_in_sync_after_the_new_rule` 钉住；`--check` 的盲区本身留给后续单。
6. **本机当前有 4 个 HEAD 级红测试**（本单 stash 前后都红，与本单无关）：`test_no_benchmark_harness_is_uncovered`、`test_threed_artifacts::test_shear_report_meets_its_criteria`（h1.8 缺测）、`test_threed_artifacts::test_hydrostatic_report_meets_its_criteria`、`test_doc_pointers::test_every_docs_pointer_resolves_or_is_waived`（`docs/verification_contact_angle_m6453.md` 缺失）。它们是仓库的证据欠账，不因本单消失。

## 6. 怎么用

```bash
python benchmarks/verification_ledger/build_ledger.py            # 重生成账本
python benchmarks/verification_ledger/build_ledger.py --check    # 漂移检查
python benchmarks/verification_ledger/make_overcredit_doc.py     # 重生成本文件
pytest tests/test_verification_ledger.py tests/test_verification_ledger_overcredit.py -q
```

引用纪律：引用 3-D 结果时必须同时给出 rung / 算例名与「工件 verdict」——`threed_shear.decay_rate.finest` 只能写成「n = 24 一级 +11.5 %…−2.7 % 阶梯中最好一级，工件整体 fail（h1.8 缺测）」，不能写成「3-D 粘性算子已验证」。
