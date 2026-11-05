# m4455 — PR #229 的 CI 红归因与修复（Lint 门，非 dam-break 用例）

## ① 现象：红在 Lint 步骤，不在 pytest

用 GitHub Actions jobs API 读 PR #229（head `12583ba`）的实际步骤结果：

| job | 结论 | 失败步骤 | 其后的 6–16 步 |
|---|---|---|---|
| pytest (3.10) | failure | **step 5 Lint** | 全部 skipped |
| pytest (3.12) | failure | **step 5 Lint** | 全部 skipped |
| pytest (3.11) | （当时未跑完） | — | — |

也就是说：**根本没跑到 `python -m pytest -q`**，红的步骤是 `.github/workflows/*.yml` 里的
`python -m ruff check .`。任务书里"pytest (3.10)/(3.12)=failure"是 job 名，job 名带的是
矩阵版本号，与失败步骤无关——这是本轮第一个要纠正的判断。

对照证据（同一 API，只列 `pytest (3.11)` job）：

| run | 分支 | 失败步骤 | step 6 「Run unit and smoke tests」 |
|---|---|---|---|
| 42024505283 | main (`7d91b89a`) | 5 Lint | skipped |
| 42028993254 | agent/m4131 | 5 Lint | skipped |
| 42018750162 | agent/m3792 | 5 Lint | skipped |
| 42018142459 | agent/m3675 | 5 Lint | skipped |

→ 红不是 PR #229 独有，而是**全仓 Lint 欠账**：main 自己也是红的，任何 PR 一开就红，
`pytest` 之后的所有门（smoke run / manifest 契约 / neighbor 对比 / free-surface / hydrostatic /
dam-break 收敛 / fail-fast）已连续多轮从未执行过。

## ② 复现（本地，忠实还原分支树）

PR #229 只有 5 个文件（`pulls/229/files`）：`benchmarks/dam_break/make_status_doc.py`、
`docs/assets/dam_break_status.png`、`docs/ci_fix_155.md`、`docs/verification_dam_break.md`、
`tests/test_dam_break_status.py`。逐个比对 **GitHub blob sha 与本地 `git hash-object`：5/5 SAME**，
确认本地树 = PR 分支树（无版本漂移）。

本地 `ruff 0.15.0`（CI 侧 `pip install "ruff>=0.6"` 解析到最新版，本机 loongarch64 无 wheel 装不上
新版本，故用本机 0.15.0 作为等价判据）：

| 树 | `ruff check .` |
|---|---|
| 分支基点 `7d91b89a`（= PR #229 拉树时的 main tip） | **113 errors** |
| 当前 main `b90a8a1`（#231/m4131 清过一轮后） | **41 errors** |
| 本分支（修复后） | **0 errors / All checks passed** |

41 条构成：8 条 `W292`(行尾无换行)/`F401`(未用 import)、1 条 `F841`(死变量)、32 条 `E501`。
32 条 `E501` **全部**落在两个文档生成器的 **Markdown 字符串里**（`| ... |` 表格行、正文模板、`$$` 公式行）——
Markdown 表格行必须是单个物理行，折行就等于毁掉渲染结果，所以这不是"代码风格欠账"而是
**规则的适用面错误**。

## ③ 修法（最小、可复核、不降物理判据）

1. `ruff check . --fix` 吃掉 8 条机械问题（EOF 换行、未用 import），零行为变化；
2. `benchmarks/contact_angle_geometry/run.py` 删掉死赋值 `far = d[:n_fluid] > 2.0 * h`
   （全仓 grep `n_far`/`far` 无消费者，`out` 字典也不含它 → 纯死代码）；
3. `pyproject.toml` 增加**文件级作用域**豁免，而不是放宽全局门：
   ```toml
   [tool.ruff.lint.per-file-ignores]
   "benchmarks/*/make_doc.py" = ["E501"]
   ```
   其余所有文件（含 `src/`、`tests/`、其它 benchmark）的 `E501` 照旧生效。
   该文件原有注释就写着 "Correctness-only gate … Formatting is deliberately not enforced"，
   本改动是让配置与其自述一致，而不是新增豁免文化。

## ④ 验证

| 检查 | 命令 | 结果 |
|---|---|---|
| Lint（CI step 5） | `python -m ruff check .` | **All checks passed!**（41→0） |
| PR #229 自带用例 | `pytest tests/test_dam_break_status.py` | **8 passed** |
| 全量单测 | `pytest -q -n 8`（`PYTHONPATH=本树/src`） | 481 passed / 13 failed（见 ⑤） |
| CI step 7 + step 8 | `examples/droplet_impact_2d/run.py --steps 3` + manifest 契约断言 | 跑通，`manifest contract OK: 5 条 criteria 全部有 population` |
| Lint 证据 | `benchmarks/ci_lint_evidence.py` | 见 docs/pr149_lint_evidence.json 机制 |

## ⑤ 注意点 / 未完成

* **13 条单测红与本单无关，但确实在红**：修复前后失败集合**逐条相同**（同一台 loongarch64 机器、
  同一 `PYTHONPATH`），且全部落在本单与 PR #229 都未触碰的文件里：
  `test_verification_ledger`(6)、`test_oscillating_droplet*`(3)、`test_initial_density_calibration`(1)、
  `test_kernel_sum_consistency`(1)、`test_wall_pressure[3.0]`(1)、`test_laplace_pressure_jump::test_figure_exists`(1)。
  其中两条 `XPASS(strict)` 的 docstring 自陈 "mode-2 fit is architecture-fragile
  (x86_64 CI +8.64 % vs loong64 -31.78 %)" —— 本机是 loongarch64，与 CI 的 x86_64 不可互换，
  所以**本地 13 红不能直接判 CI pytest 红**；但也**不能反过来担保 CI pytest 必绿**：
  step 6 被 Lint 挡了多轮，最近一次真实执行记录已不可得。诚实结论：本单只证明了「Lint 门绿灯」，
  pytest 门的状态要等下一次 CI 真实执行（Lint 修好后 step 6 才会第一次跑起来）。
* 被 `tp_pusher` 过滤的 `runs/*`（含本单 `pytest_before`/`pytest_after` 日志）不进 GitHub，属于本地记录。
* `E501` 的 32 条只做**文件级**豁免，没有逐行加 `# noqa`（32 个 `# noqa` 会把表格行本身撑得更长，
  且下次生成器改写又得重打补丁）。若后续要求零豁免，正解是给生成器改成"内容行长"与"源码行长"
  解耦（例如模板外置到 `.md.in` 文件），那是另一单的范围。