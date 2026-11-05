# PR #143 CI cure：验证账本漂移（main 既存红）修复

**分支**：`agent/m3584`（PR #143，仅新增 `docs/verification_csf_curvature_convergence.md`）
**修复提交内容**：`benchmarks/verification_ledger/{build_ledger.py,waivers.json,ledger.json}`、`docs/verification_ledger.md`
**日期**：2026-11-05

## 1. 红在哪：不是 PR 自己引入的

PR #143 相对 main 只多一个 doc 文件，但 CI（pytest 3.10/3.11/3.12 三个矩阵）全红。
在 **纯 main（`733f182`，不含 PR 内容）** 上复现：

```
$ python benchmarks/verification_ledger/build_ledger.py --check
DRIFT: ledger.json is stale (run build_ledger.py)
DRIFT: docs/verification_ledger.md is stale (run build_ledger.py)
```

即：`tests/test_verification_ledger.py` 的 4 条 anti-drift/coverage 断言在 main 上已经是红的，
PR #143 的 CI 只是继承了它。失败根因是 coverage 规则（`unregistered_artifact`）报出 5 个
**已被历次任务提交、但从未登记为 claim** 的机器可读证据工件：

| 工件 | 提交来源 | 处理方式（本修复） |
|---|---|---|
| `benchmarks/conservation/report.json` | m3256 | 登记 gate claim `conservation.ballistic_lag.closed_form` |
| `benchmarks/init_consistency/report.json` | m3114 | 登记 gate claim `initial_consistency.spurious_pressure.calibrated` |
| `benchmarks/initial_density_bias/report.json` | m3209 | 登记 gate claim `initial_density_bias.square_pool.analytic` |
| `benchmarks/lattice_equilibrium/report.json` | m3211 | 登记 gate claim `lattice_equilibrium.hydrostatic.calibrated` |
| `benchmarks/laplace_pressure_closure/raw/relax_ladder.json` | m3151 | `waivers.json` 显式豁免（原始逐 step 松弛梯，读数由 `report_force_balance.json` 汇总；与既有 `laplace_law/convergence/relax_dx*.json` 豁免同一先例） |

## 2. 怎么修：登记而不是豁免（除原始梯）

`build_ledger.py` 新增 `benchmark_report_gate_claims()`，沿用仓内既有的 `claim_type="gate"`
路径：每条 claim 的 **读数、判据限值、pass/fail 都由工件内的字段程序化读取**，无手抄数字：

| claim | JSON 路径（实测） | 工件自述限值 | 账本级 rel err | 等级 |
|---|---|---|---|---|
| `conservation.ballistic_lag.closed_form` | `verdicts.G4_max_ratio_deviation` | 3 %（gate 名 `..._within_3pct`） | 6.2e-9 % | PASS |
| `initial_consistency.spurious_pressure.calibrated` | `summary.worst_calibrated_rel_pressure` | 1e-4（`criterion.expr`） | 6.7e-14 % | PASS |
| `initial_density_bias.square_pool.analytic` | `validation.square_pool.rel_diff` | 0.10（`validation.square_pool.G2_tolerance`） | 4.83 % | MARGINAL |
| `lattice_equilibrium.hydrostatic.calibrated` | `hydrostatic_profile[1].rel_error_max_abs` | 1e-4（`reference.acceptance_ratio`） | 9.5e-5 % | PASS |

然后 `python benchmarks/verification_ledger/build_ledger.py` 重生成 `ledger.json` 与
`docs/verification_ledger.md`（零手改，生成器即事实源）。账本从 16 claims / 3+5 findings
变为 **20 claims / 3 findings**，`unregistered_artifact` 归零。

## 3. 验证证据（本地，loongarch64 宿主，Python 3.11.6）

```
$ PYTHONPATH=src python benchmarks/verification_ledger/build_ledger.py --check
ledger up to date: 20 claims, 3 findings        # exit 0

$ PYTHONPATH=src python -m pytest -q tests/test_verification_ledger.py
20 passed                                        # 修前：4 failed, 16 passed

$ python -m ruff check benchmarks/verification_ledger/build_ledger.py
All checks passed!
```

**未做/未完成的诚实说明**：

1. 本地全量 `pytest` 仍会有 **2 条 `XPASS(strict)`**（`tests/test_oscillating_droplet.py::
   test_recorded_mode2_frequency_deficiency_is_reproduced` /
   `::test_oscillation_frequency_gate_is_not_met`）。这是**宿主架构相关、main 上既存**的红：
   本机 `uname -m = loongarch64`，mode-2 频率误差实测正好落在钉住值 −31.78 % 的容差内；而
   GitHub CI 是 x86_64，同一代码读 +8.64 %，与钉住值不符 → strict xfail 成立、CI 绿。测试
   文件里已把两种架构的实测值写进 `XFAIL_MODE2_ARCH_FRAGILE` 说明，属于 owner 已登记的
   re-baseline 项，本单不触碰（改动它会削弱 owner 明确要求的双向漂移检测）。
2. 覆盖规则仍是正则子集匹配（`...(validation_report|report|results|ladder|audit).json`），
   `benchmarks/laplace_pressure_closure/report_force_balance.json` 这类以 `balance.json`
   结尾的证据文件目前逃过 coverage 扫描——既存缺口，属账本设计范畴，本单只留证不改规则。
3. 本地 `ruff check .` 报 100 个 error，但在纯 main 上数量完全相同（0 差异），非本单引入。