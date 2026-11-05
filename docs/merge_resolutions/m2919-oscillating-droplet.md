# PR #87（agent/m2919）→ main 冲突裁决记录

- 任务：mission m3820，治愈 TensorSPH#87（`mergeable=false`，main 已前进 29 个提交）
- 基线：main `a661054`（merge-governor 合入 #112 后的头），PR 分支 `agent/m2919` `ec91864`
- 三方合并基：`5800532`
- 合并提交：`f8b75f6`（`agent/m3820` 工作树，随后推回 `agent/m2919`）
- 机器证据：[`runs/m3820/merge_evidence.json`](../../runs/m3820/merge_evidence.json)（由 [`runs/m3820/audit_merge.py`](../../runs/m3820/audit_merge.py) 重新生成，不手抄）

## 1. 冲突全景：4 个 add/add，全部同源（同一基准两套独立实现）

`git merge --no-commit agent/m2919` 只报 4 个冲突，且都是 **add/add**：main 与 PR 分支在同一个
merge-base 之后各自新增了「振荡液滴 vs Rayleigh–Lamb 解析解」这一套基准。

| 冲突文件 | main 侧 | m2919 侧 | 性质 |
|---|---|---|---|
| `benchmarks/oscillating_droplet/run.py` | 567 行，`--mode frequency/ladder/damping/quiet`，第二矩+峰衰减+零交叉，另带 `trace_data.json`/`make_doc.py` | 475 行，`--case mode2/mode3/coarse/no_sigma`，两级网格阻尼余弦拟合+σ=0 负控 | 同基准、不同 API 的两套实现 |
| `benchmarks/oscillating_droplet/validation_report.json` | 136 行（`frequency`/`verdict` 两键） | 8935 行（`rows`/`runs`/`errors`，含逐算例历史曲线） | 两份互不兼容的 schema |
| `docs/verification_oscillating_droplet.md` | 131 行（由 `make_doc.py` 生成，数字程序注入） | 90 行（手写+json 注入，honest FAIL 记录） | 两份不同结论文档（−43.5 % vs −61.7 %） |
| `tests/test_oscillating_droplet.py` | 258 行，模块级测试 + `release_case` 夹具 + 2 个 strict-xfail 物理钉 | 153 行，4 个测试类（解析/估计器/观测量/求解器回归） | 两套测试，无函数名交集 |

**为什么不能机械 overlay**：两套模块都导出 `run_case` / `mode_amplitude` / `zero_crossing_period`，
但签名与语义不同（m2919 的 `mode_amplitude` 带 `mode` 参数并归一化到连续极限 1.5ε）。
任选一侧覆盖另一侧都会造成：① 静默丢失另一侧 475/567 行实现；② 另一侧测试 import 到错误实现而
「假绿」或直接崩；③ 报告/图被同名覆盖（两套 harness 都写 `validation_report.json` 与
`oscillating_droplet.png`）。

## 2. 逐文件解法（main 占名字面，m2919 侧原样并置）

原则：**main 保规范路径**（`make_doc.py`、`trace_data.json`、README、CI 都指向它），
**PR 侧内容逐字节保留**在 `*_lamb` 兄弟路径，只重写会互相覆盖的路径；两侧交叉引用。

| # | 冲突文件 | 解法 |
|---|---|---|
| 1 | `benchmarks/oscillating_droplet/run.py` | main 版**原样**保留在该路径 → m2919 版迁移为 `run_lamb.py`：正文不动，仅 ① 加 10 行「合并说明」note；② 输出名 `validation_report.json`→`validation_report_lamb.json`、`oscillating_droplet.png`→`oscillating_droplet_lamb.png`、`..._decay.png`→`..._lamb_decay.png`；③ 用法行路径改名；④ lint 修（E501 长 f-string 拆成 `err_txt`，语义等价；文件尾补换行）。 |
| 2 | `benchmarks/oscillating_droplet/validation_report.json` | main 版原样保留（`make_doc.py` 与 strict-xfail 钉读它）→ m2919 的 8935 行报告**逐字节**存为 `validation_report_lamb.json`，无字段改写、无抽样。 |
| 3 | `docs/verification_oscillating_droplet.md` | main 版（生成物）原样保留 → m2919 版存为 `docs/verification_oscillating_droplet_lamb.md`：仅改路径引用与图链接，正文/误差表/结论（含 FAIL 记录）不动；文件头加「合并说明」双链回 main 版与本文件。 |
| 4 | `tests/test_oscillating_droplet.py` | main 版原样保留 → m2919 版存为 `tests/test_oscillating_droplet_lamb.py`：仅 `from benchmarks.oscillating_droplet.run import` → `run_lamb`，文档字符串路径同步；测试体与断言一字未改。 |
| 5 | 二进制图（非冲突，来自 PR） | `oscillating_droplet.png` → `oscillating_droplet_lamb.png`，`oscillating_droplet_decay.png` → `oscillating_droplet_lamb_decay.png`（避免被 main 的 `plot()` 同名覆盖）。 |
| 6 | `benchmarks/capillary_wave/validation.py`（PR 新增，非冲突，自动合并） | 原样保留；仅 lint 修（删未用 import `stable_dt`、EOF 换行）。 |

合并后无删除：`main` 的 231 个路径全部在场且内容**逐字节不变**；PR 分支的 147 个路径全部在场
（87 个逐字节相同，其余为三方合并后含 main 后续修改的版本，另有 6 个按上表改名并置）。

## 3. 机器可复核的无损证明

`runs/m3820/audit_merge.py` 从 git 对象库重算（不依赖本文件文字）：

- `no_main_content_lost`：main 路径集合 − 合并树路径集合 = ∅，且 `main_paths_modified_by_merge` = ∅。
- `no_m2919_content_lost`：m2919 路径集合 − 合并树路径集合 = ∅（含 6 条改名映射）。
- `every_added_line_survived`：对两侧都改过的文本文件，逐行多重集校验「各自自 merge-base 起新增的
  每一行都出现在合并结果里」——**没有整行被静默丢弃**。
- `no_unexplained_edits_on_relocated_files`：迁移文件中相对 m2919 原件的每一处行级变化，都必须命中
  已声明的改名/说明/lint 类别，否则判失败。
- `no_conflict_markers`：全树 `<<<<<<<`/`>>>>>>>` 计数 = 0。

当前判定：`all_ok = true`（见 `merge_evidence.json` 的 `verdict` 与 `counts` 字段）。

## 4. 测试证据（本地，torch 2.7.1 / py3.11 / CPU）

见 `runs/m3820/test_evidence.md`：main 基线（`a661054` 干净树）与本合并树跑同一命令
`PYTHONPATH=src python -m pytest tests -q`，逐项对比通过数，**不低于基线**；两套振荡液滴测试
（main 版 + `_lamb` 版）在同一进程内同时收集通过，证明「两套实现共存且都被 CI 覆盖」。
`python -m ruff check .` 在基线树与本合并树均为 `All checks passed!`（CI lint 门）。

## 5. 后续（留档，不在本单范围）

两套实现并非长期理想形态：`_lamb` 与 `run.py` 度量同一物理量（模态 2 频率），却给出 −61.7 % 与
−43.5 % 两种 honest FAIL，说明「同一基准两种数值路径」本身是有价值的交叉证据，但也意味着
`_lamb` 一套应在后续单里择优并入主线（保留其 σ=0 负控与 n=3 分支），而不是永久并置。