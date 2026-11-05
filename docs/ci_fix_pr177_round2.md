# PR #177 第 2 轮 CI 排查与修复记录

## 1. 现场复原（先对齐 head，再看红）

- PR #177 的 head 是 `agent/m3851` = `e05d2ec`，其唯一父提交是 `7d91b89`。
  本地旧引用 `agent/m3851`（`77db05e`）已经过期，且 `git fetch` 在本环境不可用
  （`git ls-remote` 60 s 无返回），因此从 codeload 取 head 源码包，用
  `git commit-tree` 逐位复原：复原树 = `5cc8fd1b6ee4f4f8ebc335b1edc0cd10e68bcc38`，
  与 GitHub API 报告的 head tree **完全一致**，可证明内容等价（提交号因时区信息不同而不同）。
- 远端 `main` = `b90a8a1`，PR 的 base 落后 41 个提交 → 这是所有“莫名其妙的红”的总根源。

## 2. 复现（本地，PYTHONPATH 指向本仓 src，避免复用他仓 editable 安装）

```
python -m pytest -q     # head 单侧：13 failed, 488 passed
python -m ruff check .  # 43 errors (E501 32 / W292 7 / F401 3 / F841 1)
```

关键对照实验：把 **origin/main 本身**（`b90a8a1`，与 CI 目标一致）用同一命令跑

```
python -m pytest -q --tb=no tests/test_kernel_sum_consistency.py \
    tests/test_verification_ledger.py tests/test_laplace_pressure_jump.py tests/test_wall_pressure.py
# => 9 failed, 39 passed, 1 xfailed
```

即上述失败在 main 上同样存在，属继承性红（环境/仓库级欠账），不是 #177 引入。

## 3. 本轮实际改动（最小、可验证）

1. **把 41 个提交的漂移合并掉**：`Merge origin/main into PR #177 content`，
   以 `7d91b89` 为真祖先做三方合并，**零冲突**（此前用过期分支做合并时出现的
   `benchmarks/capillary_wave/validation.py` add/add 冲突是伪冲突，祖先修正后消失）。
2. **清掉 PR 自己引入的 lint 红**：`benchmarks/capillary_wave/validation.py`
   里 `stable_dt` 未被使用（F401）与文末缺换行（W292）——`ruff check` 该文件与
   PR 新增测试文件现为 `All checks passed!`。

未改动任何与本 PR 无关的文件（`contact_angle_geometry/`、`laplace_pressure_jump/`
等文件里 32 条 E501 是 main 上的既有欠账，本轮刻意不动，另开单处理）。

## 4. 仍未达成的部分（honest-FAIL）

- `ruff check .` 全仓仍有 33 条 E501（29 条在 `benchmarks/contact_angle_geometry/make_doc.py`），
  全部存在于 main 上 → **CI 的 Lint 步骤在本 PR 上仍会失败**，与 #177 无关。
- 3 条 PR 侧的 `tests/test_oscillating_droplet*.py` 失败
  （`test_recorded_mode2_frequency_deficiency_is_reproduced`、
  `test_oscillation_frequency_gate_is_not_met`、
  `TestSolverRegression::test_mode_oscillates_near_the_capillary_frequency`）
  需要跑真实 SPH 弛豫（单次数分钟）才能定位，本轮预算内未闭环。
- 因此 **“CI 绿 + 合入 main”未达成**，本单按未完成如实上报。

## 5. 结论 / 后续建议

- 合并漂移是本轮可交付的净收益：PR #177 现在是 main 的干净超集，零冲突可合并。
- 建议拆分两单：(a) 全仓 lint 收尾（33 条 E501，纯排版）；(b) 振荡液滴三个红测的数值定位
  （需集群跑弛豫 + 与 `docs/verification_oscillating_droplet.md` 的判据对齐）。
