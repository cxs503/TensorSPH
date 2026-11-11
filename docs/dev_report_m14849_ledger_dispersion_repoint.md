# 验证账本指针修复 + CSF 色散读数落地（m14849）

**范围**: TensorSPH ｜ **性质**: 缺陷修复（证据主干回归）+ 验证欠账落地 ｜ **锚点**: NEW（总账无 SPH 条目，见 finish 建议）

## 1. 案例介绍（问题）

`benchmarks/verification_ledger/build_ledger.py` 在 HEAD 上**建不出来**：m14729（#471，已合并）为
`csf_dispersion.capillary_wave.omega_error.local_range` / `csf_dispersion.lamb_mode.omega_error.local_range`
两条 claim 登记的工件是 `benchmarks/surface_tension_dispersion/validation_report.json`，该路径在 m13895
把测量迁进 `tensorsph.surface_tension_dispersion` + `scripts/run_surface_tension_dispersion.py` 时已经退役，
仓库里根本没有这个文件；且 claim 读的 JSON 键（`gates.local_range.C1_max_omega_error_pct` / `C1_within_5pct`）
是迁移前的旧 schema。

后果不是"少两条 claim"，而是 `build()` 抛 `FileNotFoundError` → `tests/test_verification_ledger.py`
**20 个用例全部 setup error + 2 failed**：整个证据账本（57 条 claim / 25 条 finding）在 HEAD 上不可重建，
账本反漂移、判级、覆盖三条守护全部失效。这是一类结构性风险：**一条 claim 指向不存在的文件，就能让全部证据静默失效。**

## 2. 怎么做的

1. **跑出真工件**：`PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --norm local_range --threads 4 --out-dir runs/surface_tension_dispersion`
   （31.8 s，schema `tensorsph.surface_tension_dispersion/1`，含 report/meta/benchmark/summary + 4 张图 + sha256 provenance）。
2. **把两条 claim 重新接到真实工件与当前 schema**（`scripts/m14849_repoint_dispersion_claims.py`，幂等）：
   工件 `runs/surface_tension_dispersion/report.json`，键 `gates.C1_capillary_wave.max_abs_error_pct` /
   `gates.C2_lamb_modes.max_abs_error_pct`；notes 里所有数字都是对工件的 f-string（无手抄）。
3. **加结构性护栏**：`tests/test_verification_ledger.py::test_every_referenced_artifact_path_exists`
   扫描 `build_ledger.py` 里的工件引用，缺文件时**点名路径**失败，而不是塌成 20 个 setup error。
4. 顺带补齐 `waivers.json` 中两条空理由（`bc_ladder.json` 已补；`bc_reach.json` 仍空，见 §5）。
5. 重生成 `ledger.json` + `docs/verification_ledger.md`（57 claims / 25 findings）。

## 3. 计算结果（全部由 `runs/surface_tension_dispersion/report.json` 程序化注入）

| claim | 判据 | 实测 | 倍数超标 | 等级 |
|---|---|---|---|---|
| `csf_dispersion.capillary_wave.omega_error.local_range` | ≤ 5.0 % | **245.92 %** | 49.2× | **FAIL** |
| `csf_dispersion.lamb_mode.omega_error.local_range` | ≤ 3.0 % | **9.48 %** | 3.2× | **FAIL** |
| C3 收敛阶（最小） | ≥ 2.0 | **-0.87** | — | **FAIL**（不收敛） |
| C4 σ 敏感性最大散度 | ≤ 5 % | 0.00 % | — | PASS |

操作员筛选（4 种 δ 归一化 × 曲率格式组合，最细网格）：
毛细波最好 \|ω\| 误差 **93.21 %**（any_pass=False），
Lamb 模态最好 **4.85 %**（any_pass=False）——
**没有任何已交付组合通过波族判据**。

账本级变化：`test_verification_ledger.py` 由 **20 errors + 2 failed** → **0 errors / 2 failed**（余下 2 条为既有欠账，见 §5）。

## 4. 结果图

- `runs/surface_tension_dispersion/figures/capillary_wave_error.png`：毛细波 k=1..3 的 ω 相对误差 vs ppw(24/32/48) 与 5 % 门限。
- `runs/surface_tension_dispersion/figures/lamb_stiffness.png`：Lamb 模态刚度比 vs 解析值。
（图为 harness 自带、随工件提交、sha256 入 `meta.json`；平台报告内嵌同一文件。）

## 5. 注意点 / 局限

1. **没有让任何能力通过**：两条 claim 仍是 FAIL，且我用 C3 的负收敛阶说明这是**算子一致性缺陷**（细化误差不降），
   不是分辨率问题——与 `docs/verification_effective_sigma.md` 的 29 % 有效 σ 亏缺同一族。
2. m14729 报告里引用的 69.7 % / 901.4 % 两个数**不可复现**：其工件未随合并落地（同批 waivers 理由也缺失），
   本单以可复现工件为准（245.9 % / 9.48 %，`--norm local_range`；legacy 归一化下为 198.4 % / 19.3 %）。
   历史数字不覆盖、不删除。
3. 仍未关的既有红（**不是本单引入**）：`test_no_evidence_artifact_is_unregistered`
   （`benchmarks/contact_angle/bc_diagnosis.json` 未登记）、`test_every_waiver_has_a_reason`
   （`bc_reach.json` 理由仍空）、`test_no_benchmark_harness_is_uncovered`
   （slamming / two_phase_laplace / verification_ladder / wcsph_chain / wedge_entry 五台 harness 无工件；
   本单已为它们写入 E1 欠账理由，但覆盖规则仍需逐台跑出工件才能关账）。
4. `--norm local_range` 是本单显式选择，使 claim ID 里的 `local_range` 与工件 `setup.delta_normalization` 一致；
   默认归一化仍为 `legacy`。

## 6. 误差判定

- 关键量 = 毛细波 ω 误差 **245.92 %**（门限 5.0 %）：**否（不满足 <3 %，也不满足 <15 %）**，超门限 49.2 倍。
- 关键量 = Lamb 模态 ω 误差 **9.48 %**（门限 3.0 %）：**否**，超门限 3.2 倍。
- 收敛阶 -0.87 vs 要求 2.0：**否**（负阶，误差随细化增大）。
- 本单的**修复本体**（账本可重建、指针可解析、护栏可失败）由测试机器裁决：`test_every_referenced_artifact_path_exists`
  与 20 条账本用例恢复通过，属"确定性判据"，不涉及数值误差。
