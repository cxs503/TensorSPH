# m21797 脆弱性审计交付物 · 管到底第 3 轮闭环复核（m24604）

**锚点** `XVAL-SPH-METHOD-SELECTION-2026-11-12` · **main** `bcca7a557950` · **PR#1026 头** `4ad970a9274b`
**生成** 2026-11-17T22:15:58Z · 全部数字由 `benchmarks/method_selection/closure_reverify_m24604.py` 程序化注入，零手抄。

## ① 案例介绍

本单是 **TensorSPH#1026 的重开·管到底第 3 轮**。原 PR#1026（标题
`m21797 matured mission deliverables`，11 个文件）于 2026-11-16 被裁定关闭、未合并。
其功能要点是 m21797 的**方法选择推荐脆弱性审计**：对 `select_method` 给出的每个
SPH 格式推荐，量化其精度上界站在几条*独立*基准族记录上（证据深度）、以及删掉哪条
记录会让推荐静默翻转（承重记录）。

物理背景：SPH 格式选择器把算例画像映射为「主方案 + 修饰项 + 精度上界」。一个只在
单一基准族上测过一次的上界只是**一个数据点**——删掉那条记录，推荐就静默变成
`WITHHELD`。m21797 交付的审计驱动 `audit_decision_fragility.py` 正是量这个的。

## ② 怎么做的

从当前 main（`bcca7a557950`）切枝 `agent/m24604`，对 PR#1026 的 11 个文件逐一
与 main 比对（`git rev-parse <ref>:<path>` 对象级比较），并重跑证据链：

1. **文件级比对**：PR#1026 头 `4ad970a9274b` vs `origin/main`，逐文件判 identical / updated-in-main / missing。
2. **反漂移检查**：`audit_decision_fragility.py --check`（独立复算证据深度 vs 路由 `Decision.evidence_depth`）。
3. **测试面**：`pytest tests/test_method_selection_fragility.py tests/test_method_selection_strict_bound.py`。
4. **登记表快照**：`method_selection_registry.json` 记录数 vs 脆弱性报告记录数（应一致）。

工具链：git（对象比较）+ pytest + 审计驱动自带 `--check`。未改任何核心模块。

## ③ 计算结果

| 指标 | 值 |
|---|---|
| 文件 identical（与 PR#1026 头一致） | 4 |
| 文件 updated-in-main（main 更新） | 7 |
| 文件 missing | 0 |
| 反漂移检查 | OK |
| 测试面 | 23 passed, 1 warning in 10.55s |
| 登记表记录数 | 30 |
| 脆弱性报告记录数 | 30 |
| 基准族总数 | 13 |
| 不变量 | OK |
| 深度失配数 | 0 |
| **裁决** | **ALL-INTACT** |

### 逐文件状态（PR#1026 头 → 当前 main）

| 文件 | 状态 | sha256(main) 前 12 |
|---|---|---|
| `benchmarks/method_selection/audit_decision_fragility.py` | 与 PR#1026 头一致 | `962d83f37ea1` |
| `benchmarks/method_selection/strict_bound_profiles.py` | 与 PR#1026 头一致 | `50a0b2d4d2c1` |
| `docs/dev_report_method_selection_fragility_m21797.md` | 与 PR#1026 头一致 | `de4dc2d93d57` |
| `docs/figures/method_selection_fragility/evidence_depth.png` | main 已更新（更新） | `1a132a6734fc` |
| `docs/figures/method_selection_strict_bound/modes.png` | main 已更新（更新） | `4daa9a3d1daf` |
| `docs/reports/method_selection_fragility_m21797.json` | main 已更新（更新） | `9e1afafc9e35` |
| `docs/reports/method_selection_strict_bound_m21537.json` | main 已更新（更新） | `dfd13996ef1c` |
| `docs/xval_method_selection_fragility_m21797.md` | main 已更新（更新） | `6cd3ef8cf04b` |
| `docs/xval_method_selection_strict_bound_m21537.md` | main 已更新（更新） | `ba860aea0e17` |
| `src/tensorsph/method_selection.py` | 与 PR#1026 头一致 | `75c3135f6f05` |
| `tests/test_method_selection_fragility.py` | main 已更新（更新） | `acab841deceb` |

**解读**：11 个文件全部在 main 中（missing=0）。4 个与 PR#1026 头逐字节一致；
7 个在 main 中已被后续单（m22096 / m22388 等）刷新为*更新*的登记表快照
（30 记录 / 13 族，较 PR#1026 头的 26 记录 / 11 族更新）。
即 PR#1026 的内容**已完整进入 main 且被后续单刷新**，PR#1026 关闭是因其相对 main 已陈旧/冗余，
而非内容缺失。

## ④ 结果图

`docs/figures/method_selection_fragility/closure_m24604.png`：11 个交付文件在 main 上的
状态条（绿=一致，橙=main 更新，红=缺失）。本单为**文档/验证类**任务，无新物理场量，
故无场量云图；该图为交付物状态证据图。

## ⑤ 注意点

- 本单是**闭环复核**，不是新功能：m21797 的功能（`audit_decision_fragility.py`）与测试
  （`test_method_selection_fragility.py`）早已经 #830 并入 main，并被 m22096/m22388 刷新。
- 未触碰任何 TensorSPH 核心模块（`src/tensorsph/*`）——与在途 PR 无热文件重叠。
- 局限：文件级对象比较只证明「内容在 main」，不重算物理量；物理量的正确性由
  m20432（独立复算）与 m21537（严格上界）两单背书，本单不重复。

## ⑥ 误差判定

本单为**验证/闭环**任务，不产生新的物理量计算，故无「关键量 vs 基准 <3%」的逐项误差。
所复核交付物自身数值（精度上界、证据深度）由反漂移检查确认与当前登记表一致
（深度失配 = 0），即**交付物数值零漂移**。
以「交付物是否完整且未漂移」为判据：**通过**（ALL-INTACT）。
