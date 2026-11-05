# PR #103 (agent/m3163) 合并治愈记录：Young–Laplace 分辨率阶梯

**任务**: m3939 · **仓库**: TensorSPH · **PR**: #103 (`agent/m3163`)
**分叉点**: `184fef9` · **main 头**: `ba07a10` · **PR 头(旧)**: `b7af006`

## 1. 案例介绍

PR #103 把 Young–Laplace 压差验证从「固定步数」改成「固定毛细时间」
（三档 dx 都在 `t = 1.00 t_sigma` 读数），并追加绝对压进口径
`p(r<0.5R) - 0`，用于判定该项验证仍为 **WITHHELD**。它与 20 个 main 提交
（CI lint 门禁、守恒/泊松/初始化一致性基准等）在同一个数据目录上撞车：
双方都重跑了 `benchmarks/laplace_law/convergence/` 下的三个 relax 工件。

## 2. 怎么做

1. 三方合并 `agent/m3163` 与 `main`（merge base `184fef9`），git 报 3 个冲突文件；
2. 逐文件做**语义并集**判定，而不是机械 overlay；
3. 本地跑全量测试与 ruff，与 main 基线逐项对齐；
4. 合并后暴露的 PR 级红点（缺失图、lint）在 PR 侧就地修掉。

### 冲突文件清单与逐文件解法

| 文件 | 冲突性质 | 解法 | 依据 |
|---|---|---|---|
| `benchmarks/laplace_law/convergence/relax_dx0.0015.json` | 双方各自重跑，数值全不同 | 取 PR 侧整件 | 键空间已验证 PR 侧是 main 侧超集（main 独有键 0 个）；该工件是单次运行的原子快照 |
| `benchmarks/laplace_law/convergence/relax_dx0.0020.json` | 双方各自重跑，数值全不同 | 取 PR 侧整件 | 键空间已验证 PR 侧是 main 侧超集（main 独有键 0 个）；该工件是单次运行的原子快照 |
| `benchmarks/laplace_law/convergence/relax_dx0.0025.json` | 双方各自重跑，数值全不同 | 取 PR 侧整件 | 键空间已验证 PR 侧是 main 侧超集（main 独有键 0 个）；该工件是单次运行的原子快照 |

未冲突部分按 git 自动合并保留：相对 `main` 只剩 PR 的 10 个文件，
相对 `agent/m3163` 保留 main 的全部 54 个改动，无任何删除。

## 3. 计算结果

| 树 | passed | failed | xfailed | 用时 [s] |
|---|---|---|---|---|
| main `ba07a10` | 363 | 2 | 8 | 418.9 |
| 合并树 `ba3d9be` | 372 | 2 | 8 | 700.5 |

合并树比基线多 9 条通过用例（PR 新增的分辨率阶梯守护 9 条 + 图/文档渲染守护），
两棵树失败的用例完全相同（环境性、与本次合并无关）：
- `tests/test_oscillating_droplet.py::test_oscillation_frequency_gate_is_not_met`
- `tests/test_oscillating_droplet.py::test_recorded_mode2_frequency_deficiency_is_reproduced`

合并后 `ruff check .` 结果：**All checks passed**（main 为 0 违规，PR 原为 32 处）。

阶梯数值（合并后工件，σ/R = 360 Pa，最差档 dx = 2.5 mm，绝对口径误差 -33.7%）：

## 4. 判定

| 判据 | main 基线 | 合并树 | 是否达标 |
|---|---|---|---|
| 全量 pytest 失败数 | 2 | 2 | 是（未新增） |
| 通过用例数 | 363 | 372 | 是（+9） |
| ruff 违规数 | 0 | 0 | 是 |
| 冲突文件静默丢内容 | — | 0（键空间超集验证 + 双向 diff 计数） | 是 |

Young–Laplace 压差本身**仍是 WITHHELD**：三档分辨率绝对口径误差
-21.2% / -18.9% / -33.7%
，全部远超 3%% 与 15%% 门槛，本次合并不改变该结论，也不宣称任何精度提升。

## 5. 注意点

- 三个 relax 工件**不能逐字段拼接**：setup 步数不同（PR 按毛细时间对齐，
  main 固定 900 步），混拼会得到跨运行的混合数；只能整件取舍。
- `make_ladder_doc.py` 渲染出的 markdown 表格行不能折行，故模板长行以
  文件级 `# ruff: noqa: E501` 说明；文档本身仍由测试逐字节比对渲染结果。
- 该 PR 的图 `laplace_resolution_ladder.png` 曾被交付门禁滤除（274 KB），
  现在脚本内做 256 色调色板压缩到 85 KB，重跑脚本可复现提交的字节。
- 合并只在本地 git 完成；push 与 pr-gate 记账由平台/merge_governor 执行。

_本文由 `runs/m3939/make_merge_report.py` 从 `runs/m3939/*.log` 与
`resolution_ladder.json` 渲染，数字禁止手抄。_
