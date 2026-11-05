# PR #235 CI red clearance (m4477)

**分支**：`agent/m4112`（PR #235 head `ea90c49`）
**结论**：PR #235 自身引入 0 个新失败；CI 红全部是 main（`b90a8a1`）已有红。
本单在 PR 分支上清掉其中 10 个（3 个根因），剩余 3 个是本机（loong64）专属的浮点/
拟合差异，未在 PR 分支内容里，见 §4。

## 1. 归因证据：红测是继承的，不是 PR 造的

同一组测试在 **main-only 工作树**（`git worktree add /tmp/mainw b90a8a1`，`PYTHONPATH=/tmp/mainw/src`）
与 **merge 后工作树**（main ⊕ PR #235）分别执行，结果逐条一致：

```
main  : 13 failed, 78 passed, 1 xfailed   (/tmp/main_sub.log, 197 s)
merged: 13 failed, 78 passed, 1 xfailed   (/tmp/merged_sub.log, 199 s)
```

失败清单完全相同，说明 PR #235 的 9 个文件改动没有引入新红。
根因只有三类（外加 3 个 loong64-only）。

## 2. 根因 A：`Config.density_calibration` 默认值与其契约相反（3 个红测）

`Config.density_calibration = True`（m3211 引入）让 `lattice_mass_unit()` 默认返回
`rho0 / sum_j W`，于是：

| 红测 | 现象 | 说明 |
|---|---|---|
| `test_initial_density_calibration::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass` | `mass != rho0 dx^2` | 标定默认开启，与「opt-in、已发表数字可复现」的契约冲突 |
| `test_kernel_sum_consistency::test_shipped_square_lattice_carries_no_bias` | 实测 `b = +1.19e-07`，pin `-5.32e-05` | 默认标定把生产格子的微小偏差也抹掉，历史 pin 失效 |
| `test_wall_pressure::test_wall_pressure_matches_the_published_extrapolation[3.0]` | 壁面压强外推基线偏移 | 同上，质量整体 ×(1+5.3e-5) |

**判据（不是拍脑袋）**：生产方形格子 `h = 1.3 dx` 的几何偏差本来就只有
`b = -5.3e-5`（`dx^2 sum_j W - 1`，见 `docs/algorithm-gap-analysis.md`），远低于张力下限；
真正有偏差的是六角格 / `h = 1.5 dx`（`b = +3.4e-3`）。默认开启标定对已发表算例
只带来 5.3e-5 的整体质量缩放、零物理收益，却让三个独立 pin 失效。
因此 `density_calibration` 恢复为**显式 opt-in（默认 False）**，`benchmarks/lattice_equilibrium/run.py`
继续显式 `density_calibration=calibrated` 做 A/B，标定能力一字未删。

验证：翻转后 `test_initial_density_calibration`（16）、`test_kernel_sum_consistency`、
`test_wall_pressure` 全绿，且 `tests/` 里没有任何测试断言「默认 True」。

## 3. 根因 B/C：验证台账漂移与缺失图件（7 个红测）

* **B（6 个）**：`benchmarks/verification_ledger/` 的 committed `ledger.json` /
  `docs/verification_ledger.md` 落后于 `benchmarks/**` 的现状——重新构建后 findings 从
  已提交的 3 条变成 13 条：9 个 `unregistered_artifact` + 1 个 `unmarked_superseded_number`。
  处理：
  - 9 个证据文件以**带理由的 waiver** 登记进 `waivers.json`（每个都写明它属于哪个 claim /
    哪个守卫测试；没有一条是静默豁免）；
  - `docs/verification_surface_tension_consistent.md` §9 的 `-75.6%` 补「历史值，现行读数见
    `docs/verification_ledger.md`」标记；
  - 重新生成 `ledger.json` + `docs/verification_ledger.md`（`build_ledger.py`，16 claims / 3 findings）。
* **C（1 个）**：`tests/test_laplace_pressure_jump.py::test_figure_exists` 要求
  `benchmarks/laplace_pressure_jump/laplace_pressure_jump.png`（>10 kB），仓里没有。
  该图由 `run.py::make_figure(validation_report)` 纯从**已提交的 JSON** 重建（不跑仿真），
  已生成并入库（147 kB），测试一字未改。

## 4. 未清（如实）：3 个 loong64 专属红测

| 测试 | 本机（loong64） | 判定 |
|---|---|---|
| `test_oscillating_droplet.py::test_recorded_mode2_frequency_deficiency_is_reproduced` | XPASS（strict xfail → 记失败），实测 -31.78% 正落在 pin 上 | x86 CI 历史上测 +8.64% → 仍 xfail。属**架构区分**，PR 分支已把 xfail 理由改写为「该算例不可测（能量不守恒，KE 超可用表面能 ~450×）」 |
| `test_oscillating_droplet.py::test_oscillation_frequency_gate_is_not_met` | 同上 XPASS | 同上 |
| `test_oscillating_droplet_lamb.py::TestSolverRegression::test_mode_oscillates_near_the_capillary_frequency` | 61.7% 误差 vs 30% 容差 | 同一 mode-2 算例；PR #235 自己的不稳定审计已证明该算例数值无效，重标定属另一单 |

这 3 条在 **main-only 工作树**上逐条同样失败，与本 PR 无关；本单未改动其物理判据（改容差
= 粉饰）。清除它们需要先给 mode-2 找一个能量守恒的稳定算例，属独立工作项。

## 5. 复现命令

```bash
git worktree add /tmp/mainw b90a8a1            # main baseline
cd /root/tp-agent/work/m4477                   # main ⊕ PR #235 ⊕ 本单修复
PYTHONPATH=src python -m pytest -q -rf                        # 全量
PYTHONPATH=src python -m pytest -q tests/test_verification_ledger.py \
    tests/test_initial_density_calibration.py tests/test_kernel_sum_consistency.py \
    tests/test_laplace_pressure_jump.py "tests/test_wall_pressure.py::test_wall_pressure_matches_the_published_extrapolation"
# -> 63 passed, 1 xfailed
```

注意：本机 `site-packages` 里 `__editable__.tensorsph-0.8.0.pth` 指向**另一个**工作树，
所有命令必须显式 `PYTHONPATH=<本工作树>/src`，否则测的是别人的代码（本单踩过）。