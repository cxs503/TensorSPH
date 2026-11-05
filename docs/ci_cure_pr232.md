# PR #232 CI cure：验证账本漂移（`pytest (3.12)` 红）修复

**分支**：`agent/m4109`（PR #232，头 `f092311`）；本次修复提交落在 `agent/m4456`
**修复涉及文件**：`benchmarks/verification_ledger/{build_ledger.py,waivers.json,ledger.json}`、
`docs/verification_ledger.md`、`docs/verification_surface_tension_consistent.md`（1 行）
**日期**：2026-11-05
**复现宿主**：loongarch64 / Python 3.11.6（CI 矩阵为 x86_64 3.10/3.11/3.12，本机无 3.12，见 §4 局限）

## 1. 红在哪：账本 anti-drift 断言，不是 PR 自己引入的

PR #232 相对 main 只改账本族文件（`build_ledger.py` / `ledger.json` / `waivers.json` /
`docs/verification_ledger.md`）与两个 doc。把 PR 头与 main 合并后，在本地复现：

```
$ python benchmarks/verification_ledger/build_ledger.py --check
DRIFT: ledger.json is stale (run build_ledger.py)
DRIFT: docs/verification_ledger.md is stale (run build_ledger.py)

$ python -m pytest -q tests/test_verification_ledger.py        # 修前
6 failed, 14 passed
FAILED ...::test_committed_ledger_json_is_up_to_date
FAILED ...::test_committed_markdown_is_up_to_date
FAILED ...::test_check_mode_reports_no_drift
FAILED ...::test_no_evidence_artifact_is_unregistered
FAILED ...::test_stale_numbers_in_prose_are_marked
FAILED ...::test_markdown_contains_every_claim_and_finding
```

根因与 PR #143 同形（见 `docs/ci_cure_pr143.md`）：账本冻结在旧 main 上生成，之后三条
**机器可读证据工件**随别的分支合并进 main 却从未登记为 claim，coverage 规则因此报
`unregistered_artifact`；另有一处散文仍在无标记地引用**作废读数** `-75.6%`，报
`unmarked_superseded_number`。两类 finding 都被 `tests/test_verification_ledger.py`
的断言直接判红。

| 缺陷 | 工件 / 位置 | 处理方式（本修复） |
|---|---|---|
| `unregistered_artifact` | `benchmarks/laplace_pressure_jump/validation_report.json` | 登记 claim `young_laplace.corrected_definition.dx2mm` |
| `unregistered_artifact` | `benchmarks/contact_angle_geometry/results.json` | 登记 gate claim `contact_angle.fit.spherical_cap.coarse_rung` |
| `unregistered_artifact` | `benchmarks/dam_break/reference_ladder.json` | `waivers.json` 显式豁免（原始逐分辨率梯，读数由 `resolution_ladder.json` 汇总） |
| `unmarked_superseded_number` | `docs/verification_surface_tension_consistent.md:158` | 数字旁补 `historical` + 现行读数 + 账本指针 |

## 2. 怎么修：登记（不是豁免），数字仍由工件程序化读出

`build_ledger.py` 新增 `unregistered_evidence_claims()`，延续仓内既有两种 claim 路径
（`relative_error` / `gate`）：

| claim | JSON 路径（实测） | 参考值 | 实测 | 账本级 rel err | 等级 |
|---|---|---|---|---|---|
| `young_laplace.corrected_definition.dx2mm` | `cases[0].corrected_definition` | 360.3776 Pa（σ/R_eff） | 225.4308 Pa | −37.45 % | FAIL |
| `contact_angle.fit.spherical_cap.coarse_rung` | `cases[0].fit` | 60 deg | 65.6096 deg（超 5 deg gate） | +9.35 %（gate fail） | FAIL |

要点：

1. **口径写进 claim**：`laplace_pressure_jump` 的关键在于「量具在液滴外」——旧 0.80–0.95 R
   带读数仍在液滴内部，工件自身标 `valid_as_jump=false`、rel err −93.72 %；修正口径后
   −37.45 %，与 `young_laplace.absolute_tail.best_of_6`（−34.0 %）同族，说明**残差来自自由面
   压力闭合而非曲率算子**（内部 std/mean 79.8 % 一并在 notes 里落库）。
2. **接触角用 gate 类型**：工件自带 `tolerances.angle_deg = 5 deg`，最粗梯级（dx = 1 mm、
   60 deg）实测 5.61 deg → gate fail，故账本判 FAIL；同一工件 dx ≤ 0.5 mm 的两级分别
   1.07 / 0.07 deg（notes 里由工件读值格式化），即**几何控制随分辨率收敛**，解析球冠与
   边界条件两项检查是精确的（1.9e-14 deg）。
3. `dam_break/reference_ladder.json` 走豁免并写理由：它是 `dam_break.martin_moyce.mae`
   （工件 `resolution_ladder.json`，finest dx = 0.01 m、MAE 0.5728 Z/a vs 限值 0.15）的
   原始梯，且含 bundled / crosscheck 两张参考表的分歧（同一 dx 的 MAE 0.573 vs 0.116），
   该分歧由 `tests/test_dam_break_reference_audit.py` 审计而非再立一条 claim。
4. 随后 `python benchmarks/verification_ledger/build_ledger.py` 重生成账本（零手改，
   生成器即事实源）。

## 3. 验证证据

```
$ python benchmarks/verification_ledger/build_ledger.py --check
ledger up to date: 22 claims, 3 findings                       # exit 0

$ python -m pytest -q tests/test_verification_ledger.py
20 passed                                                      # 修前 6 failed, 14 passed

$ python -m ruff check benchmarks/verification_ledger/build_ledger.py
All checks passed!
```

账本前后对照：

| 指标 | 修前（合并 main 后） | 修后 |
|---|---|---|
| claims | 20 | **22** |
| 物理量 | 14 | **15** |
| findings | 7（3 unregistered + 1 unmarked + 3 原有） | **3**（`declared_vs_derived` 1、`quantity_spread` 2） |
| `unregistered_artifact` | 3 | **0** |
| `unmarked_superseded_number` | 1 | **0** |
| 分级分布 | PASS 3 / MARGINAL 2 / WITHHELD 8 / FAIL 6 / WITHDRAWN 1 | PASS 3 / MARGINAL 2 / WITHHELD 8 / **FAIL 8** / WITHDRAWN 1 |

新增的两条 claim 都是 FAIL，这是**如实记录**（−37.45 %、超 5 deg gate），不是把红改绿：
账本的价值就在于「当前读数不合格」必须留在表里，而循环外的散文不得再引旧数字。

## 4. 局限与未做（诚实说明）

1. **CI 矩阵的 3.12 未在本机复现**：宿主是 loongarch64，只有 Python 3.11.6，仓里 3.12 的
   差异（若有）本单无法直接观测。本单修的是版本无关的账本断言（纯 JSON/文本比较），
   修前在 3.11 上同样 6 failed，说明这条红不依赖 3.12。
2. **本地全量 pytest 仍有 7 条失败，全部在纯 main 上同样失败**（`git archive origin/main`
   独立树复跑逐条对齐）：`test_oscillating_droplet{,_lamb}` 两条 strict-xfail（架构相关，
   `docs/ci_cure_pr143.md` 已登记）、`test_kernel_sum_consistency`、
   `test_wall_pressure[3.0]`、`test_initial_density_calibration`（float32/算子顺序容差）、
   `test_laplace_pressure_jump::test_figure_exists`（图件工件未入库）。它们属 main 既存红、
   与本 PR 无关，按「不得改动无关文件」不触碰，留证待单独任务。
3. 覆盖规则仍是正则子集（`...(validation_report|report|results|ladder|audit)\.json`），
   以 `balance.json` 等结尾的证据文件仍逃过扫描——既存设计缺口，本单只留证不改规则。