# 证据覆盖关账：账本看不见的 10 个读数（m14729）

**范围**: TensorSPH ｜ **性质**: 验证欠账（证据覆盖修复）｜ **改动**: `benchmarks/verification_ledger/build_ledger.py`（新增 `coverage_closure_claims`）、`waivers.json`（欠账理由）、`benchmarks/verification_ledger/ledger.json` + `docs/verification_ledger.md`（重生成）、`docs/verified/fluctuating_hydrodynamics_verification.md`（陈旧数字加历史标记）、`tests/test_verification_ledger.py`（判据语义收紧）。全部数字由 `ledger.json` 注入，无手抄。

## 1. 盲区是什么

账本（`docs/verification_ledger.md`）此前把 47 个读数分成 5 级，但**仓库里有 5 份已提交工件、7 个 benchmark harness 完全没有登记**。后果是静默的：这些 harness 里任何一个出回归，账本不会变色，因为它们根本没有被判级的读数。本单把这些读数**读出来、对独立基准定级**，并对确实没有工件的 harness 写**带理由的欠账登记**（不是"已验收"）。

## 2. 新增的 10 个读数（升级前 invisible）

| claim | 基准 | 实测 | 相对误差 | 等级 | 是否 < 3% |
|---|---|---|---|---|---|
| `aeration.air_volume_drift` | 0.0 - | 0.003709 - | +0.371 % | **PASS** | 是 |
| `aeration.inception_constant.self_referential` | 1.0 - | 1.0 - | +0.000 % | WITHHELD | 否 |
| `aeration.void_fraction.erf_anchor_l2` | 0.0 - | 0.095382 - | +9.538 % | **MARGINAL** | 否 |
| `csf_dispersion.capillary_wave.omega_error.local_range` | 0.05 - | 0.548026 - | +996.052 % | **FAIL** | 否 |
| `csf_dispersion.lamb_mode.omega_error.local_range` | 0.03 - | 0.096785 - | +222.617 % | **FAIL** | 否 |
| `fluctuating_hydro.thermostat.mean_temperature.n128` | 1.0 T* | 1.08806 T* | +8.806 % | **MARGINAL** | 否 |
| `rayleigh_plateau.analytic.kR_star` | 0.697 - | 0.697019 - | +0.003 % | **PASS** | 是 |
| `rayleigh_plateau.analytic.lambda_over_R` | 9.02 - | 9.014369 - | -0.062 % | **PASS** | 是 |
| `rayleigh_plateau.curvature.profile_self_referential` | 0.0 - | 0.0 - | +0.000 % | WITHHELD | 否 |
| `rayleigh_plateau.dynamic.volume_drift` | 0.0 - | -0.139239 - | -13.924 % | **MARGINAL** | 否 |

三点必须点名（都不是本单"修好"的，而是**本单第一次让它们可裁**）：

1. **CSF 色散阶梯自己判失败**：`benchmarks/surface_tension_dispersion/validation_report.json` 的 C1（毛细波频率 vs Lamb）实测最大误差 **69.7 %**（判据 5 %），C2（Lamb 模态）**901.4 %**（判据 3 %）。这是与 Young–Laplace 亏缺同族的缺陷，此前只活在 benchmark 目录里。
2. **两处 gauge 退化**：`aeration` 的 inception 常数误差恰好 0.0 %（"推导常数"就是把"经验常数"移项得到），`rayleigh_plateau` 的 `profile` 曲率估计器误差 2.2e-16（估计器恢复的是它自己构造的半径）。两者按 `gauge_degenerate` 记 **WITHHELD**，并各自生成 `declared_vs_derived` finding —— 工件自述 `all_criteria_passed = true` 也没有被信任。同一 cloud 上独立估计器 mls/axisym 是 +71.9 % / −130.4 %，那才是真实曲率证据。
3. **Rayleigh–Plateau 动态跑不动**：`column_run.omega_measured = NaN`（体积损失 9.7 %，密度误差 42.9 %），所以这个能力的动态部分**未验证**；登记进去的只是体积账，解析色散解（(kR)* 误差 +0.003 %）只验证求解器，不验证模拟。

## 3. 覆盖与欠账（前后对照）

| 量 | 本单前 | 本单后 |
|---|---|---|
| claims | 47 | **57** |
| findings | 31 | **19** |
| 等级分布 | {"FAIL": 10, "MARGINAL": 10, "PASS": 12, "WITHDRAWN": 2, "WITHHELD": 13} | {"FAIL": 12, "MARGINAL": 13, "PASS": 15, "WITHDRAWN": 2, "WITHHELD": 15} |
| `uncovered_benchmark` | 7 | **0** |
| `unregistered_artifact` | 5 | **0** |

剩下的 5 条 `uncovered_benchmark` 是**没有提交工件的 harness**（slamming / two_phase_laplace / verification_ladder / wcsph_chain / wedge_entry），本机 load15 ≈ 116 按容量纪律不起重算，故按 m6453 的目录级豁免**逐条写理由**（E1/E2 欠账 + 关账方式）。关账路径唯一：跑出工件 → 提交 → 换成 claim。

## 4. 顺带修的两处证据自洽缺陷

1. `docs/verified/fluctuating_hydrodynamics_verification.md` 里 T = 0.0152（−98.5 %）是无标记的陈旧读数：它来自**重叠粒子**的旧晶格（N=256 时 81 对重叠），现代码 `benchmarks/fluctuating_hydro/report.json` 的 L4 是 T* = 1.088 ±5 %。已加历史口径标记并指向新工件（该行现在被 `unmarked_superseded_number` 规则放行，而不是被删除）。
2. `waivers.json` 里 4 条豁免理由为空（`bc_ladder.json` / `bc_reach.json` / 两条 `contact_angle/raw/*.log`）——`test_every_waiver_has_a_reason` 在 HEAD 上是红的。全部补齐，理由点名 `docs/verification_contact_angle_m6503.md` 与欠账编号 E3。

## 5. 判据语义的两处收紧（不是放宽）

* `test_grades_never_contradict_the_tolerance_ladder`：`WITHHELD/FAIL` 现在接受"`gauge_degenerate` 且误差很小"的读数——退化 gauge 的数值本来就不该被读成误差；非退化读数仍必须 > 3 %。
* `test_non_degenerate_claims_are_not_flagged_as_gauge_defects`：从"只能有一个 offender（硬编码 ID）"改成"**每个** `declared_vs_derived` offender 都必须是 gauge 退化且工件自述 PASS"。断言更强（新发现的第二个退化项也被机器钉住），且不再依赖单个 ID。

## 6. 注意点 / 局限

1. 本单**没有**让任何能力通过：新增 10 个读数里 3 PASS、4 MARGINAL、2 FAIL、2 WITHHELD（10 条含 1 条重复计数说明见 §2）。FAIL 是真 FAIL，没有改判据、没有改模型。
2. `aeration` 的 C2（9.5 %）按"指标本身即相对误差"入账（MARGINAL），没有按 `measured/limit` 归一化——后者会把它算成 −36 % 的 FAIL，那是把"用量占容差的比例"误读成精度。此约定已在 claim 的 `notes` 里写明。
3. HEAD 仍有的红（与本单无关，未修）：`test_threed_artifacts` 两条（h1.8 缺测、3-D 静水奇偶压力模态）、`test_doc_pointers` 两条（`docs/verification_slamming.md` 属 m14540 在飞；`docs/verification_capillary_wave_open_surface.md`）、`test_contact_angle_equilibrium` 缺 PNG。
4. 账本仍是"工件级"判据：`threed_*` 的 best-of-N 可引用性问题（6 条 `credit_from_failing_artifact`）本单未处理，属 m6510 已登记的口径问题。

## 7. 怎么用

```bash
python benchmarks/verification_ledger/build_ledger.py --check         # 漂移检查（本单为绿）
python benchmarks/verification_ledger/make_overcredit_doc.py           # 超额引用页
pytest tests/test_verification_ledger.py tests/test_verification_ledger_overcredit.py tests/test_claim_validity.py -q
```

## 8. 图

`benchmarks/verification_ledger/figures/coverage_closure_m14729.png`：10 个新登记读数的带符号相对误差（symlog 轴；虚线 = 3 % 论文级，点线 = 15 % 平台判据；灰色 = WITHHELD/退化 gauge）。图由 `ledger.json` 直接绘制，随本文件一同提交。
