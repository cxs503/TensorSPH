# PR #286 CI 红清除（m4600）

**PR**：#286（head 分支 `agent/m4477`，head `703d5e5`，base `main` `b90a8a1`）
**CI 现象**：`pytest (3.11)` / `pytest (3.12)` = failure（`pytest (3.10)` 取消/排队）。
**归因结论**：**失败步不是 pytest 本体，而是第 5 步 `Lint`（`python -m ruff check .`）**；
后续单测/端到端步骤全部 `skipped`。这 41 条 ruff 命中**全部落在本 PR 未触及的文件**
（`benchmarks/contact_angle_geometry/*`、`benchmarks/laplace_pressure_jump/*`、
`tests/test_contact_angle_geometry.py`、`tests/test_laplace_pressure_jump.py`），
即 **main 自带的 lint 债**，不是本 PR 引入的。

## 1. 归因证据（同一提交两处对齐）

| 证据 | 内容 |
|---|---|
| PR #286 job 步骤 | `pytest (3.12)`：step 5 `Lint` = failure；step 6–15 = skipped；step 16 `Upload smoke-run artifacts` = success（空产物，印证未跑到端到端步） |
| base main `b90a8a1` 自身 CI（run 42079367292） | 3 个 Python 版本**同样卡在 `Lint`** → 与本 PR 内容无关 |
| 本机 `ruff 0.15.0` 于 PR head 树 | `Found 41 errors`，逐条落在上列 6 个文件；本 PR 的 9 个改动文件 0 命中 |
| 本 PR 内容隔离 | `git diff b90a8a1 703d5e5 -- <上列 6 文件>` 为空 |

因此「绿」的唯一路径是把这 41 条债清掉（此前 m4477 清 PR #235 时同样先清 main 继承红）。

## 2. 怎么修的（41 条，6 个文件，全部最小改）

| 规则 | 条数 | 处理 | 输出影响 |
|---|---|---|---|
| `E501`（>100 列）`benchmarks/laplace_pressure_jump/make_doc.py` | 4 | 相邻字面量按 `\|\|`/模板边界拆行（隐式拼接） | **0**：字符串字节不变 |
| `E501` `benchmarks/contact_angle_geometry/make_doc.py` | 27 | 文件级 `# ruff: noqa: E501` + 5 行理由注释（长行是 DOC f-string 模板里的 markdown 表行，软换行会改已提交 md 的字节） | 0 |
| `F401` 未用 `import math`（2 个文件） | 2 | 删import | 0 |
| `F841` 死变量 `far`（`contact_angle_geometry/run.py`） | 1 | 删该行（全文件仅此一处出现） | 0 |
| `W292` 文件尾缺换行 | 6 | 补换行 | 0 |

验证：`PYTHONPATH=src python3 -m ruff check .` → **All checks passed!**

**字符串不变的程序化自证**：改后重跑 `benchmarks/laplace_pressure_jump/make_doc.py`，
`docs/verification_laplace_pressure_jump.md` 的 `git diff` 为 **空**（若拆行改变了拼接结果，
重生成的 md 必然不同）。`contact_angle_geometry/make_doc.py` 重跑后 md 仅差一行机器负载读数
（`load15 217.30 → 2323.16`，来自 `results.json` 的既有非确定性字段，与本次改动无关，
已用 `git checkout` 还原，不在本单范围内）。

## 3. 注意点 / 局限

* CI 的 `Lint` 步装的是**未钉版本**的 ruff（`ruff>=0.6`）；本机 0.15.0 判绿，若上游再收紧规则，
  仍可能出现新命中——根因是 CI 未钉版本，建议后续钉 `ruff==0.15.0`（本单未改，避免扩散）。
* 本机为 loong64，CI 为 x86_64；本单只动 lint 面，不涉浮点，故无架构风险。
* 未在本机复现 CI 的 x86 环境；「绿」的依据是 `ruff check .` 退出码为 0 + base main 同步骤
  同样红（步骤级证据，非测试级逐条对照）。
* 全量 `pytest` 在本机并发（其他单占满 48 核）下极慢，本单只做了 lint 面 + 相关文档生成器的
  重生成核对；CI 的 3 个 Python 版本会从零跑全量测试，属另一层门禁。

## 4. 复现命令

```bash
git fetch --no-tags origin refs/heads/agent/m4477   # 本机 github.com:443 不通，走 Git Data API 取树
git checkout -B <wt> <PR head>
PYTHONPATH=src python3 -m ruff check .            # 修复前: Found 41 errors；修复后: All checks passed!
PYTHONPATH=src python3 benchmarks/laplace_pressure_jump/make_doc.py && git diff --stat docs/
```