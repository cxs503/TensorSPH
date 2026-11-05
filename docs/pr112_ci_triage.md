# PR #112（agent/m3256）CI 无 check runs：根因、修复与验证

**任务**：TensorSPH PR #112 开张后 governor 记录 `checks: no check runs yet`，strict 持链。
**结论一句话**：分支没问题、workflow 也没问题；**#112 自带的三个新文件违反仓库 ruff 门，
唯一一次运行在 `Lint` 处就死了**（后面 12 个步骤全部 skipped），而 09:47 那次分支更新
**根本没有创建 workflow run**（GitHub 侧该次 synchronize 事件丢失）。两件事叠在一起，
才表现为「无 check runs」。

---

## 1. 事实采集（全部程序化，见 `runs/collect_evidence.py` → `runs/pr112_evidence.json`）

| 项目 | 值 | 判定 |
|---|---|---|
| `git ls-remote` `refs/heads/agent/m3256` | `9fcfe0df7cf7bc9883da95cf007a202803d5b06c` | 分支确已推上 GitHub |
| PR #112 `head.sha` | `9fcfe0d…`（同上） | 与分支头**一致** |
| PR #112 `base` | `main` @ `b77a718b…` | 目标分支正确 |
| 是否 fork | 否（`cxs503/TensorSPH`） | 非 fork，无需人工批准 |
| `merge-base(main, 分支)` | `b77a718b…`（= main 头） | 分支**不落后** |
| `rev-list --left-right --count main...分支` | `0 / 2` | 领先 2、落后 0 |
| check-runs for `9fcfe0d` | **0** | governor 看到的就是这个 |
| check-suites for `9fcfe0d` | **0** | 连 suite 都没建 |
| 该分支历史运行总数 | **1**（run `41782380278`，跑的是旧头 `57aed85`） | 见 §2 |

**删除一个常见误判**：任务书第 3 步设想的「分支落后/残损 → 三方合并推回」在这里
**不适用**——分支头就是 main 的快进后继（落后 0），没有残损历史，无需再合 main。

## 2. 唯一那次运行为什么是红的（阻塞性问题，可复现）

run [`41782380278`](https://github.com/cxs503/TensorSPH/actions/runs/41782380278)：
`event=pull_request`，`head_sha=57aed85`，`conclusion=failure`，创建于 `2026-11-04T05:57:18Z`
（= PR 开张那一刻）。

三个 matrix job（3.10 / 3.11 / 3.12）**全部在第 5 步 `Lint` 失败**，第 6–15 步（单测、
端到端算例、manifest 契约、neighbor_search、free-surface、静水、溃坝、fail-fast）**全部 skipped**。

在未改动的 PR 头上复现（`/tmp/pr112tree` @ `9fcfe0d`，ruff 0.15.0）：

```
benchmarks/conservation/make_doc.py:21:93: E501 Line too long
benchmarks/conservation/make_doc.py:30:78: E501 …
benchmarks/conservation/make_doc.py:31:86: E501 …
benchmarks/conservation/make_doc.py:34:83: E501 …
benchmarks/conservation/make_doc.py:37:82: E501 …
benchmarks/conservation/make_doc.py:273:29: W292 No newline at end of file
benchmarks/conservation/run.py:617:29: W292 No newline at end of file
tests/test_conservation.py:167:87: W292 No newline at end of file
Found 8 errors.
```

即：**即使 CI 重新触发，#112 也一定红**。这是本单真正要修的东西。

## 3. 09:47 那次分支更新为什么一条 run 都没建（观测性问题）

- `9fcfe0d`（`Merge branch 'main' into agent/m3256`）作者时间 `09:47:12Z`，PR `updated_at = 09:47:14Z`
  → 确实是一次 `synchronize`。但 `check-suites=0`、按分支过滤的运行总数仍是 1。
- **不是仓库 workflow 的问题**：`.github/workflows/tests.yml` 在 PR 头与 main **逐字节相同**；
  `on.pull_request.branches: [main]` 在；**没有** `paths` 路径过滤、没有 `concurrency`；
  仓库 `actions/permissions.enabled = true`；唯一 workflow 状态 `active`。
- **不是 Actions 故障**：`2026-11-04` 当天全仓共 **77 次运行**，紧邻的两个是 `09:43:41`
  （m3432，success）与 `09:59:37`（m3428），跨过了 09:47。
- **不是「服务端合并提交不触发」**：同仓近期 5 个 head 为 2-parent / committer=`GitHub`
  的 PR（m3289、m3276、m3258、m3251、m3197）**都吃到了 run**，其中 m3276 还是 **success**。
  该假设被自家历史证伪。

⇒ 故判定：**这一次 `synchronize` 的 workflow-run 创建在 GitHub 侧丢失**（单次/瞬时事件丢失），
仓库侧配置无缺陷。修复手段是「重新产生一次真实 push 或手动 dispatch」，而不是改 workflow。
（`workflow_dispatch` 本来就已声明，是这次可用的兜底触发器。）

## 4. 修复（commit `c8e27c9`，仅动 lint，不动结果）

```diff
--- a/benchmarks/conservation/make_doc.py
+++ b/benchmarks/conservation/make_doc.py
@@ -18,7 +18,8 @@ TEMPLATE = """# 守恒律、伽利略不变性与时间积分阶…
-**仓库**: TensorSPH ｜ **工件**: `benchmarks/conservation/` ｜ **测试**: `tests/test_conservation.py`
+**仓库**: TensorSPH ｜ **工件**: \
+`benchmarks/conservation/` ｜ **测试**: `tests/test_conservation.py`
@@ G1 / G1b / G2 / G4 表格行同理（4 处）
-| G1b 粘性项单对动量泄漏闭式 | `C ν m_i m_j …` | 实测/闭式 = {pair_ratio:.6f} | **机制确认**（…） |
+| G1b 粘性项单对动量泄漏闭式 | `C ν m_i m_j …` | \
+实测/闭式 = {pair_ratio:.6f} | **机制确认**（…） |
--- a/benchmarks/conservation/run.py        （末尾补换行）
--- a/tests/test_conservation.py           （末尾补换行）
```

**为什么用反斜杠续行而不是断行**：这 5 处超长行是 markdown 表格行/段落，位于模板字符串内，
硬断行会往生成的 `docs/verification_conservation.md` 里插入换行、改变产物。反斜杠续行由
Python 词法层吞掉，**产物逐字节不变**。

## 5. 验证（可复跑，日志见 `runs/ci_gate.log`）

| 检查 | 命令 | 结果 |
|---|---|---|
| 仓库 lint 门（CI 第 5 步原命令） | `python3 -m ruff check .` | **All checks passed!**（exit 0，修复前 8 errors） |
| 产物不变（关键回归判据） | `python3 benchmarks/conservation/make_doc.py` 后 `git status` | **`docs/verification_conservation.md` 未变**（逐字节复现） |
| 本单新增测试 | `pytest -q tests/test_conservation.py` | **9 passed** |
| 相关测试 | `pytest -q tests/test_conservation.py tests/test_neighbor_search.py tests/test_pair_based_rhs.py` | **45 passed**（8.65 s） |
| 端到端算例（CI 第 7 步） | `python3 examples/droplet_impact_2d/run.py --steps 3 --output artifacts/ci-droplet` | 跑通；`diagnostics.csv` / `run_manifest.json` 均非空 |
| manifest 契约（CI 第 8 步） | 按 workflow 断言逐条核对 | `software_execution=success`、`physical_validation=not_evaluated`、5 条 criteria 全 `pass`、`extrema_population=all_steps` 一致 |

## 6. 注意点与未完成（如实）

- **未完成**：本环境对 `git push` 是**机器级封锁**（连 `git push --dry-run origin HEAD:refs/heads/agent/m3256`
  都被安全策略拒绝），因此我**无法亲自**把修复推到 `agent/m3256`，也就**无法当场观测到 #112 转绿**。
  本单产出的是「已本地验证通过的修复提交 + 根因证据 + 一行可执行的收尾动作」。
- **收尾动作（需平台/owner 侧执行一次）**：把 `c8e27c9` 推到 `agent/m3256`（本分支 `agent/m3758`
  的头就是它，`9fcfe0d` 是其父，属**快进**，不会改写历史）。该次真实 push 会重新产生
  `pull_request: synchronize` 事件 → CI 起来 → 预期三 matrix 全绿。若只想先看现象，
  `workflow_dispatch`（ref=`agent/m3256`）可绕开丢失的 synchronize，但它跑的是**未修的** `9fcfe0d`，必红。
- **未跑全量单测**：本机 353 个用例的全量 `pytest` 被同机其它 agent 的重负载作业
  （并行 `pytest -n 8`）挤爆、进程被收割，只跑到 61%（无失败，仅 xfail 标记）。
  全量跑留给 CI（其 20 min 预算内、3 个 Python 版本各跑一遍）。本单改动只涉及换行与模板续行。
- **适用范围**：结论只对 `cxs503/TensorSPH` 的 `tests.yml` 有效；若将来把
  `agent/**` 加进 `on.push.branches`，这类「同步事件丢失」会多一层兜底，但也等于每次
  agent 推送都烧一遍 CI，属权衡取舍，本单**未改** workflow（无缺陷，不做无谓改动）。

## 7. 产物

| 文件 | 内容 |
|---|---|
| `c8e27c9` | lint 修复（本报告的核心交付） |
| `runs/collect_evidence.py` / `runs/pr112_evidence.json` | GitHub 事实采集与原始数据 |
| `runs/why_no_run.py` | 「服务端合并提交是否抑制运行」的证伪探针 |
| `runs/runs_20261104.json` | 当天 77 次运行的时间线原始数据 |
| `runs/make_figure.py` / `runs/pr112_figure.png` | 证据图（时间线 + lint 前后对比） |
| `runs/ci_gate.log` | 修复后 lint / 测试门复跑日志 |