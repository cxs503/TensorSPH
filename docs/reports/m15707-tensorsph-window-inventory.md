# TensorSPH 上游差集盘点（m15707，窗口 = 最近 15 提交）

> 本文件由 `scripts/audit/m15707_window_inventory.py` 生成，数字零手抄；
> 重跑脚本即可复现（`python scripts/audit/m15707_window_inventory.py`）。

## 0. 仓库 / ref / commit（先对表）

| 项 | 值 |
|---|---|
| 仓库 | TensorSPH（remote `origin` = github.com/cxs503/TensorSPH） |
| 被盘点锚点 | `1c9888e` |
| 本报告自身提交 | 不在窗口内（锚点已固定），避免盘点结果随报告提交漂移 |
| tracking upstream | `origin/main` |
| 上游差集（锚点口径） | ahead **0** / behind **0** （`git rev-list --left-right --count`） |
| 窗口 | 最近 15 个提交，覆盖 1 个自然日：2026-11-12 |
| 窗口锚点 | 固定为 `1c9888e`（否则提交本报告后窗口会漂移到报告自身） |

**结论（差集口径）**：本地 HEAD 与上游 `origin/main` 完全一致（ahead 0 / behind 0），
没有可对比的上游提交差集；按任务约定转入第二口径——**窗口主题演化盘点**。

## 1. 变更主题清单（15 提交 → 4 类主题）

| 主题 | 提交数 | 文件数 | 新增行 | 新增测试文件 | 代表 mission |
|---|---|---|---|---|---|
| `solver-capability` | 5 | 58 | 84018 | 5 | m15493, m15445, m15437, m15408, m15396 |
| `cross-audit` | 7 | 44 | 6642 | 0 | m15558, m15489, m15453, m15421, m15419 |
| `case-pool-verification` | 2 | 10 | 2793 | 2 | m15412, m15371 |
| `empty-merge` | 1 | 0 | 0 | 0 | m14500 |

逐提交明细（新旧序）：

| # | commit | date | parents | files | + | − | tests | 目录 | theme | subject |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `3b39225` | 2026-11-12 | 2 | 5 | 1231 | 0 | 1 | benchmarks,docs,tests | `case-pool-verification` | Merge #494 (m15371) via push-merge |
| 2 | `80f3c7f` | 2026-11-12 | 1 | 8 | 2541 | 0 | 1 | benchmarks,docs,src,tests | `solver-capability` | Merge #495 (m15396) via merge-master |
| 3 | `25fc915` | 2026-11-12 | 1 | 13 | 7994 | 0 | 1 | benchmarks,docs,src,tests | `solver-capability` | Merge #496 (m15408) via merge-master |
| 4 | `3be0d41` | 2026-11-12 | 1 | 5 | 1562 | 0 | 1 | benchmarks,docs,tests | `case-pool-verification` | Merge #497 (m15412) via merge-master |
| 5 | `9f6308e` | 2026-11-12 | 1 | 5 | 748 | 0 | 0 | docs,scripts | `cross-audit` | m15419: matured mission deliverables（7 个文件被门禁滤除） |
| 6 | `069c5b1` | 2026-11-12 | 2 | 5 | 748 | 0 | 0 | docs,scripts | `cross-audit` | Merge #498 (m15419) via push-merge |
| 7 | `f5f3f6a` | 2026-11-12 | 1 | 3 | 836 | 0 | 0 | docs,scripts | `cross-audit` | Merge #499 (m15421) via merge-master |
| 8 | `1ad742e` | 2026-11-12 | 1 | 0 | 0 | 0 | 0 | — | `empty-merge` | Merge #500 (m14500) via merge-master |
| 9 | `9a12b6a` | 2026-11-12 | 1 | 23 | 71160 | 0 | 1 | benchmarks,docs,src,tests | `solver-capability` | Merge #501 (m15437) via merge-master |
| 10 | `a293de8` | 2026-11-12 | 1 | 9 | 1275 | 7 | 1 | benchmarks,docs,src,tests | `solver-capability` | Merge #502 (m15445) via merge-master |
| 11 | `19b53bf` | 2026-11-12 | 1 | 7 | 1054 | 0 | 0 | docs,scripts | `cross-audit` | Merge #503 (m15453) via merge-master |
| 12 | `8224568` | 2026-11-12 | 1 | 16 | 2976 | 0 | 0 | audit,docs | `cross-audit` | Merge #504 (m15489) via merge-master |
| 13 | `b717fec` | 2026-11-12 | 1 | 5 | 1048 | 0 | 1 | benchmarks,docs,src,tests | `solver-capability` | Merge #505 (m15493) via merge-master |
| 14 | `55e1bd9` | 2026-11-12 | 1 | 4 | 140 | 0 | 0 | docs | `cross-audit` | m15558: matured mission deliverables |
| 15 | `1c9888e` | 2026-11-12 | 2 | 4 | 140 | 0 | 0 | docs | `cross-audit` | Merge #506 (m15558) via push-merge |

## 2. 算例池误差表（相对解析解，逐项）

`parsed=True` 的行由脚本从入库 JSON 反读；`cited` 行来自入库 markdown（其 run 工件写在被 gitignore 的 `runs/` 下，无 JSON 可反读），已在 source 列标注出处。

| case | variant | level | metric | value [%] | 判定 | source |
|---|---|---|---|---|---|---|
| CASE-LAPLACE-SPH | mls | 1 | dp_error_pct | +0.97 | PASS<3% | `benchmarks/laplace_sph/results.json#ladder_mls` |
| CASE-LAPLACE-SPH | mls | 2 | dp_error_pct | +0.61 | PASS<3% | `benchmarks/laplace_sph/results.json#ladder_mls` |
| CASE-LAPLACE-SPH | mls | 3 | dp_error_pct | -0.21 | PASS<3% | `benchmarks/laplace_sph/results.json#ladder_mls` |
| CASE-LAPLACE-SPH | fit_quadratic | 1 | dp_error_pct | -23.01 | UNACCEPTABLE>15% | `benchmarks/laplace_sph/results.json#ladder_fit_quadratic` |
| CASE-LAPLACE-SPH | fit_quadratic | 2 | dp_error_pct | -23.69 | UNACCEPTABLE>15% | `benchmarks/laplace_sph/results.json#ladder_fit_quadratic` |
| CASE-LAPLACE-SPH | fit_quadratic | 3 | dp_error_pct | -24.46 | UNACCEPTABLE>15% | `benchmarks/laplace_sph/results.json#ladder_fit_quadratic` |
| CASE-POISEUILLE-MIRROR | level0 | 1 | u_max_error_pct | +0.38 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.u_max` |
| CASE-POISEUILLE-MIRROR | level0 | 1 | profile_rms_interior_pct | +2.72 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.rms` |
| CASE-POISEUILLE-MIRROR | level0 | 1 | wall_shear_pct | -0.35 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.shear` |
| CASE-POISEUILLE-MIRROR | level1 | 2 | u_max_error_pct | +0.02 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.u_max` |
| CASE-POISEUILLE-MIRROR | level1 | 2 | profile_rms_interior_pct | +1.23 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.rms` |
| CASE-POISEUILLE-MIRROR | level1 | 2 | wall_shear_pct | -0.38 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.shear` |
| CASE-POISEUILLE-MIRROR | level2 | 3 | u_max_error_pct | -0.25 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.u_max` |
| CASE-POISEUILLE-MIRROR | level2 | 3 | profile_rms_interior_pct | +0.42 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.rms` |
| CASE-POISEUILLE-MIRROR | level2 | 3 | wall_shear_pct | -0.44 | PASS<3% | `benchmarks/poiseuille_mirror/out/results.json#results.shear` |
| CASE-SLOSHING | base cpd12 | 12 | omega_error_pct | +0.96 | PASS<3% | `docs/benchmarks/sloshing-resonance/summary.json#headline` |
| CASE-SLOSHING | refinement | cpd8 | omega_error_pct | +54.60 | UNACCEPTABLE>15% | `docs/benchmarks/sloshing-resonance/summary.json#Q2` |
| CASE-SLOSHING | refinement | cpd12 | omega_error_pct | +0.96 | PASS<3% | `docs/benchmarks/sloshing-resonance/summary.json#Q2` |
| CASE-CAPILLARY-WAVE | mls worst ppw=24 | — | omega_error_pct | -4.22 | ok<15% | cited: `docs/case_capillary_wave_dispersion.md §3` |
| CASE-CAPILLARY-WAVE | fit_quadratic worst ppw=16 | — | omega_error_pct | -24.27 | UNACCEPTABLE>15% | cited: `docs/case_capillary_wave_dispersion.md §3` |
| CASE-COUETTE-SPH | ladder-r24 interior L2 | — | l2_interior_pct | +2.95 | PASS<3% | cited: `docs/reports/m15412-couette-sph.md §3` |
| CASE-COUETTE-SPH | ladder-r16 wall shear (top) | — | wall_shear_ratio_dev_pct | -20.87 | UNACCEPTABLE>15% | cited: `docs/reports/m15412-couette-sph.md §3 (ratio 0.7913)` |

## 3. 关键量 vs 基准：是否 <3%（逐项显式回答）

| 关键量 | 基准 | 误差 | <3%? | 出处 |
|---|---|---|---|---|
| CASE-LAPLACE-SPH / mls / `dp_error_pct` | 解析解 | 0.97% | 是 | `benchmarks/laplace_sph/results.json#ladder_mls` |
| CASE-LAPLACE-SPH / mls / `dp_error_pct` | 解析解 | 0.61% | 是 | `benchmarks/laplace_sph/results.json#ladder_mls` |
| CASE-LAPLACE-SPH / mls / `dp_error_pct` | 解析解 | 0.21% | 是 | `benchmarks/laplace_sph/results.json#ladder_mls` |
| CASE-LAPLACE-SPH / fit_quadratic / `dp_error_pct` | 解析解 | 23.01% | 否 | `benchmarks/laplace_sph/results.json#ladder_fit_quadratic` |
| CASE-LAPLACE-SPH / fit_quadratic / `dp_error_pct` | 解析解 | 23.69% | 否 | `benchmarks/laplace_sph/results.json#ladder_fit_quadratic` |
| CASE-LAPLACE-SPH / fit_quadratic / `dp_error_pct` | 解析解 | 24.46% | 否 | `benchmarks/laplace_sph/results.json#ladder_fit_quadratic` |
| CASE-POISEUILLE-MIRROR / level0 / `u_max_error_pct` | 解析解 | 0.38% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.u_max` |
| CASE-POISEUILLE-MIRROR / level0 / `profile_rms_interior_pct` | 解析解 | 2.72% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.rms` |
| CASE-POISEUILLE-MIRROR / level0 / `wall_shear_pct` | 解析解 | 0.35% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.shear` |
| CASE-POISEUILLE-MIRROR / level1 / `u_max_error_pct` | 解析解 | 0.02% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.u_max` |
| CASE-POISEUILLE-MIRROR / level1 / `profile_rms_interior_pct` | 解析解 | 1.23% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.rms` |
| CASE-POISEUILLE-MIRROR / level1 / `wall_shear_pct` | 解析解 | 0.38% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.shear` |
| CASE-POISEUILLE-MIRROR / level2 / `u_max_error_pct` | 解析解 | 0.25% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.u_max` |
| CASE-POISEUILLE-MIRROR / level2 / `profile_rms_interior_pct` | 解析解 | 0.42% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.rms` |
| CASE-POISEUILLE-MIRROR / level2 / `wall_shear_pct` | 解析解 | 0.44% | 是 | `benchmarks/poiseuille_mirror/out/results.json#results.shear` |
| CASE-SLOSHING / base cpd12 / `omega_error_pct` | 解析解 | 0.96% | 是 | `docs/benchmarks/sloshing-resonance/summary.json#headline` |
| CASE-SLOSHING / refinement / `omega_error_pct` | 解析解 | 54.60% | 否 | `docs/benchmarks/sloshing-resonance/summary.json#Q2` |
| CASE-SLOSHING / refinement / `omega_error_pct` | 解析解 | 0.96% | 是 | `docs/benchmarks/sloshing-resonance/summary.json#Q2` |
| CASE-CAPILLARY-WAVE / mls worst ppw=24 / `omega_error_pct` | 解析解 | 4.22% | 否 | cited `docs/case_capillary_wave_dispersion.md §3` |
| CASE-CAPILLARY-WAVE / fit_quadratic worst ppw=16 / `omega_error_pct` | 解析解 | 24.27% | 否 | cited `docs/case_capillary_wave_dispersion.md §3` |
| CASE-COUETTE-SPH / ladder-r24 interior L2 / `l2_interior_pct` | 解析解 | 2.95% | 是 | cited `docs/reports/m15412-couette-sph.md §3` |
| CASE-COUETTE-SPH / ladder-r16 wall shear (top) / `wall_shear_ratio_dev_pct` | 解析解 | 20.87% | 否 | cited `docs/reports/m15412-couette-sph.md §3 (ratio 0.7913)` |

## 4. 风险标注（台账）

| ID | 级别 | 风险 | 证据 | 建议动作 |
|---|---|---|---|---|
| R1 | medium | Empty merge commit in the window | 1ad742e (m14500) 0 files / 0 lines | Mission m14500 landed no net change; ledger should record it as no-delta instead of counting it as a delivered node. |
| R2 | medium | Documentation share dominates the window | docs/ = 71781 of 93453 inserted lines (76.8%) | Report/audit text grows faster than solver code. Freeze new report variants until the case-pool verdicts are consolidated into one index. |
| R3 | low | Additive-only regime (no consolidation) | 7 deleted lines across 15 commits | Duplicate benchmark trees (laplace*, capillary*, hydrostatic*) accumulate; schedule a deletion pass with owner sign-off (irreversible-ish). |
| R4 | high | Audit evidence is git-ignored and never lands | .gitignore drops artifacts/; runs/ is not committed, so m15419's commit message records '7 files filtered by the gate' (the 7 runs/m15419_audit/*.json) | Evidence JSON must be committed under docs/ or benchmarks/*/out/ (as benchmarks/laplace_sph/results.json already does); 'evidence store empty as committed' makes audits unreproducible. |
| R5 | high | Tests depend on uncommitted run artifacts | tests/test_case_hydrostatic.py::REPORT = runs/case_hydrostatic/report.json; 4 setup errors on a clean checkout (see evidence/clean_checkout_pytest.txt) | Re-run the benchmark inside the fixture or skip with an explicit reason; the current form turns CI red for reasons unrelated to physics correctness. |
| R6 | medium | Rejected variants ship next to the accepted one | 4 committed case points exceed the 15% limit, e.g. CASE-LAPLACE-SPH fit_quadratic -23.01% -- they are the documented negative control. | Keep them (they are the falsified branch) but tag every point with its verdict in the JSON so a downstream reader cannot quote -24% as a result. |
| R7 | low | Push-merge / merge-master double landing | 9f6308e (m15419 deliverables) is merged again by 069c5b1 'via push-merge' | Two landing paths for one mission produce duplicate merge commits; harmless but it inflates the window and confuses the diff-set count. |
| R8 | medium | Benchmark tree fragmentation | 68 directories under benchmarks/ for 84 source modules | Without an index of which benchmark is the authority for each physical claim, reviewers cannot tell a live case from a superseded probe. |
| R9 | low | Branch tracing an unchanged upstream | tracking upstream origin/main: ahead 0, behind 0 (HEAD 1c9888e) | No upstream delta to review, so this inventory falls back to the theme-evolution reading as the task allows. |
| R10 | high | The CI lint gate is red on the current tree | `ruff check .` on the 452 git-tracked .py files (the exact CI step) exits 1 with 1121 violations; top rules: E501 x562; W293 x187; E702 x107 | Either fix/baseline the violations or narrow `[tool.ruff.lint] select`; a gate that is red by default carries no signal for the next landing. |

## 5. 表面规模（基线）

| 量 | 值 |
|---|---|
| benchmark_dirs | 68 |
| src_py | 84 |
| test_py | 124 |
| docs_files | 140 |

## 6. 注意点与局限

1. 窗口内 15 个提交同属一天（一次战役波），因此"主题演化"实为**一波内的分工**，不能读成跨周趋势。
2. 上游差集为 0 只说明 tracking 分支无分叉，不代表没有未 fetch 的远端提交；本报告以已 fetch 的 `origin/main` 为准。
3. 本单**不重算**任何物理量：§2/§3 的数字来自入库工件与入库报告，脚本只做反读与判定，不做求解；重算属算例池自己的验收单。
4. 本单**不修** R5（tests 依赖未入库工件）：改测试可在 CI 变绿的同时掩盖"证据缺失"这一信号，属需 owner 仲裁的口径变更，故只登记并给出修法。
5. `runs/` 未入库（R4）导致 m15419 一类审计的证据链在 main 上不可复现；本报告因此对 4 个 case 只能 cited 而不能 parsed。

## 7. 图

- `docs/figures/m15707/fig1_window_composition.png` — 窗口的新增行按目录分解（对数轴）与增删对比，标注空合并提交。
- `docs/figures/m15707/fig2_case_pool_errors.png` — 算例池误差分布与 3%/15% 判据带。
- `docs/figures/m15707/fig3_resolution_ladders.png` — 三条分辨率阶梯（Laplace / Poiseuille / Sloshing）与判据带。
