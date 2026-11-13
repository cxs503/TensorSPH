# 验证账本复活：一个缺失工件曾是 55 条断言的盲区（m17086）

**范围**: TensorSPH ｜ **性质**: 验证基础设施缺陷修复（账本可重建性 + 覆盖欠账显式化）｜ **仓**: TensorSPH

## 1. 现象：HEAD 上整本验证账本不可重建

`python benchmarks/verification_ledger/build_ledger.py --check` 在 HEAD **崩溃**：

```
FileNotFoundError: ledger artifact missing: runs/surface_tension_dispersion/report.json
```

`tests/test_verification_ledger.py` 的模块级 fixture 因此全灭：**22 个 ledger 测试全部 error**，
不是"某个断言红"，而是"没有一条断言能跑"。连带 `tests/test_wedge_entry_ledger.py`（读
`runs/wedge_entry_wagner/summary_v2.json`）也是红的。

根因不是笔误，而是**路径类别**：`runs/` 是 campaign 的临时树，交付门禁**不保留**它。
m14849 的 15 个交付文件里恰好有 8 个在 `runs/surface_tension_dispersion/` 下被门禁滤除
（提交信息原文「8 个文件被门禁滤除」），而同一笔提交把两条 CSF 色散断言指向上面的路径。
结果是：**读数留在 `ledger.json` 里，工件却不在仓里** —— 账本自相矛盾、且不可复现。

## 2. 这不是一次孤立事故，而是累积盲区

账本最后一次成功重生成停在 m14849 那一刻。此后 `benchmarks/` 下的 harness 目录从
**62 个涨到 88 个**（26 个新算例），但没有任何一条被登记，也**没有任何一处发出声音**——
因为账本已经跑不动了。修复后第一次重建，账本立刻点名：

| 量 | 修复前（HEAD 提交副本，2026-11-11 冻结） | 修复后（本轮重建） |
|---|---|---|
| claims | 57 | **62** |
| findings | 25 | **19** |
| `uncovered_benchmark` | 5（冻结值，滞后） | 0（已逐条登记理由） |
| `unregistered_artifact` | 1（冻结值） | 0（已逐条登记理由） |
| `missing_artifact`（新规则） | 不存在 | **0** |
| 等级分布 | FAIL 12、MARGINAL 13、PASS 15、WITHDRAWN 2、WITHHELD 15 | FAIL 13、MARGINAL 15、PASS 16、WITHDRAWN 2、WITHHELD 16 |

其中 **5 条断言此前完全不可见**（不是 WITHHELD，是根本不在账上）：wedge_entry 4 条
（Wagner 楔形入水：润湿宽度系数 / 冲击压力 resolved-band / 钝角 45°）＋
`fluctuating_hydro.thermostat.steady_state.n128.uq`（m16346 的稳态温度 UQ）。

## 3. 本轮改了三个真实缺陷（都不是放宽规则）

1. **路径类别**：两个 campaign 工件搬到 harness 旁边（`benchmarks/surface_tension_dispersion/report.json`、
   `benchmarks/wedge_entry/summary_v2.json`）。搬迁是**字节级**的，不重跑、不改数：
   - `report.json` 的 sha256 = `f96406f9…739dde`，与该 campaign 自己写在 `meta.json` 里的
     `report_sha256` **逐位相同**（测试 `test_relocated_campaign_artifact_matches_its_provenance` 机器校验）；
   - `summary_v2.json` 的 `git hash-object` = `922391ec…f4d`，与 m16555 提交里的 blob **同一个对象**。
2. **一个缺失工件不再能让整本账失明**：`_load` 抛 `MissingArtifact`，`build()` 逐 producer 收敛，
   把缺工件降级为一条 `missing_artifact`（severity **high**）finding，其余 61 条照常判级。
   新测试 `test_missing_artifact_is_a_finding_not_a_crash` 用幽灵 producer 钉住这条语义。
3. **两个被崩溃遮住的代码缺陷**：
   - `coverage_closure_claims()` 里 `pathlib.Path(...)` 而 `pathlib` **从未 import**（`NameError`）；
   - `resolve()` 不支持 `ladder[id=L400].c` 这类**按字段选择**的 JSON 路径（wedge 断言全用它）。
     顺带把 `[field=value]` 的实现按"字段选择而非位置"写死——campaign 的 cases/ladder 顺序不稳定，
     位置下标会在重跑后**静默指向另一个算例**。
4. **`declared_verdict` 取最坏的 verdict 味道**：`benchmarks/threed_hydrostatic/report.json` 现在
   `verdict=pass` 而 `verdict_including_probes=fail`。旧逻辑读第一个 key，于是把
   **6 条"挑选型" credit（best-of-N / 最细网格）悄悄升级成了合格**。新逻辑：工件只有在它携带的
   **每一种** verdict 都 pass 时才算自述 PASS。这条改动的方向是**收紧**，needed 的正是 m6510 记录的口径缺陷。

## 4. 覆盖欠账：登记，不是关账

26 个新 harness + 26 个未登记工件，本轮**没有**逐个补 claim（那是 26 个独立的算例级工作），
而是按仓里既有机制（`waivers.json` + `test_every_waiver_has_a_reason`）**逐条登记欠账**，
每条写明类别与关账动作：

- **E1**（harness 交付里根本没有工件，例如 `benchmarks/isph_channel/`、`benchmarks/verification_ladder/`）：
  关账 = 跑 `run.py` → 提交主工件 → 登记 claim。
- **E2**（工件在仓里但无人引用，例如 `benchmarks/kleefsman_obstacle/reference.json`、
  `benchmarks/two_phase_laplace/validation_report.json`）：关账 = 按工件自带的外部锚登记 claim，
  或说明该文件为何不承载物理读数。

waiver 从 61 条 →
**114** 条；同时删掉 4 条指向 `runs/` 的**化石 waiver**（路径交付即失，waiver 指向虚空），
并补回 3 条被丢空的理由（`bc_reach.json`、`raw/ladder_raw.log`、`raw/reach_raw.log`）——
后者在 HEAD 上让 `test_every_waiver_has_a_reason` **本来就红**。

**这些 waiver 没有一条是"能力已验证"**。它们在账本里以"未判级"出现，只表示欠账已被点名。

## 5. findings 清单（本轮重建后的真实状态）

| rule | severity | claim / 路径 |
|---|---|---|
| `declared_vs_derived` | `high` | `aeration.inception_constant.self_referential` |
| `declared_vs_derived` | `high` | `threed_hydrostatic.bottom_pressure.adami_worst` |
| `declared_vs_derived` | `high` | `young_laplace.laplace_law.vacuum_gauge` |
| `run_validity` | `high` | `oscillating_droplet.frequency.lamb` |
| `run_validity` | `high` | `young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` |
| `run_validity` | `high` | `young_laplace.closure.closure_on_interior_layers2.dx2mm_c5` |
| `run_validity` | `high` | `young_laplace.closure.closure_on_interior_layers4.dx2mm_c20` |
| `inadmissible_run` | `high` | `young_laplace.closure.closure_on_interior_layers2.dx2mm_c20` |
| `inadmissible_run` | `high` | `young_laplace.closure.closure_on_interior_layers2.dx2mm_c5` |
| `inadmissible_run` | `high` | `young_laplace.closure.closure_on_interior_layers4.dx2mm_c20` |
| `credit_from_failing_artifact` | `high` | `threed_hydrostatic.bottom_pressure.adami_best` |
| `credit_from_failing_artifact` | `high` | `threed_hydrostatic.profile.mean_rel_error_core.adami_best` |
| `credit_from_failing_artifact` | `high` | `threed_shear.decay_rate.finest` |
| `credit_from_failing_artifact` | `high` | `threed_shear.decay_rate.raw_laplacian` |
| `credit_from_failing_artifact` | `high` | `threed_shear.decay_rate.worst_of_ladder` |
| `quantity_spread` | `high` | `threed_hydrostatic_bottom_pressure` |
| `quantity_spread` | `high` | `threed_shear_decay_rate_2nuk2` |
| `quantity_spread` | `high` | `young_laplace_jump_sigma0.072_R0.1` |
| `quantity_spread` | `high` | `young_laplace_jump_sigma7.2_R0.02` |

## 6. 怎么验证

```bash
python benchmarks/verification_ledger/build_ledger.py --check     # 绿：62 claims / 19 findings
python benchmarks/verification_ledger/make_overcredit_doc.py
python benchmarks/claim_validity/validity.py
pytest tests/test_verification_ledger.py tests/test_verification_ledger_overcredit.py \
       tests/test_claim_validity.py tests/test_wedge_entry_ledger.py -q
# → 57 passed（`test_contact_angle_harness_runs_and_its_instrument_is_repaired` 需实跑 harness，本机未跑）
```

## 7. 注意点 / 局限

1. **搬迁 ≠ 重算**。两个工件是 m14849 / m16555 提交过的内容原样搬家，本轮没有重跑任何 SPH 计算；
   账本的读数因此在语义上与搬迁前一致（色散 C1 245.9 % / C2 9.48 % 仍为 FAIL，楔形 entry 读数不变）。
2. **26+26 条欠账未关**。本轮只做到"点名 + 分类 + 关账动作"，不冒称关账。
3. **`test_contact_angle_harness_runs_and_its_instrument_is_repaired` 未在本机跑**（实跑 SPH harness，
   超出本机容量纪律）；它不依赖本轮的改动。
4. 门禁会滤除 `runs/`：本轮用测试把这条纪律机器化（`test_every_referenced_artifact_path_exists`
   现在显式**禁止** claim 指向 `runs/`），但任何**新** mission 的交付路径仍要靠作者自觉。
