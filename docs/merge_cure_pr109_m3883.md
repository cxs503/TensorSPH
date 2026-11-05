# PR #109 合并治愈记录（m3883）

**仓库**: TensorSPH ｜ **PR**: #109（`agent/m3188` → `main`，mergeable=false）
**治愈方式**: 三方合并 `agent/m3188` → `main`，逐文件语义并集解冲，结果推回 `agent/m3188`，由 merge_governor 复验并入 main。

## 分支状态（治愈前）

| 分支 | SHA |
|---|---|
| `main`（合并基） | `ba07a1066bcb2be5046dc63ae0179273b9f02ee7` |
| `agent/m3188`（旧） | `fa8c1af3d130bb46114018263de5b2b2b8701d1a` |
| merge-base | `184fef9568c515092b68c5c182c1b1dad7cb3e7f` |

main 领先 20 个提交（含 m3087/m3114/m3204/m3256/m3312/m3330 等已并入成果），
`agent/m3188` 领先 1 个提交（m3188 有效表面张力验证交付，6 文件）。

## 冲突文件清单与逐文件解法

| 文件 | 状态 | 解法 |
|---|---|---|
| `tests/test_laplace_pressure.py` | **冲突**（双方都改了同一测试） | 语义并集，见下 |
| `benchmarks/effective_sigma/run.py` | 自动合并（m3188 新增） | 原样保留 |
| `benchmarks/effective_sigma/make_doc.py` | 自动合并（m3188 新增） | 原样保留 |
| `benchmarks/effective_sigma/out/report.json` | 自动合并（m3188 新增） | 原样保留 |
| `docs/verification_effective_sigma.md` | 自动合并（m3188 新增） | 用 m3188 自带 `make_doc.py` 从 `report.json` 重渲染（见下） |
| `tests/test_effective_sigma.py` | 自动合并（m3188 新增） | 原样保留 |

### `tests/test_laplace_pressure.py`（唯一冲突文件）

冲突根源：m2979 把默认曲率算子从 `mls` 换成 `fit_quadratic` 后，
`test_effective_surface_tension_at_the_documented_resolution` 的旧钉（0.79±0.06）失效，
**两侧各自独立重钉**：

- **main 侧**：重测三方案（legacy 0.215 / mls 0.791 / fit_quadratic 0.711），
  新增 `pytest` 导入、`_droplet_force(**config_kwargs)`、`_half_force_ratio()` 助手，
  断言升级为三方比较（default 0.711±0.01、mls 0.791±0.01、legacy < 0.4）。
- **m3188 侧**：重写 docstring 为 m2399→m2447→m2979→m3188 历史，
  单钉改为 0.7106±0.02，并新增「full amplitude 措辞从未成立」的更正段
  （指向 `docs/verification_effective_sigma.md`，WITHHELD >15% 门槛）。

**解法（语义并集，无静默丢失）**：

1. **代码取 main 侧**：main 的三方断言是 m3188 单钉的严格超集
   （0.7106 ∈ 0.711±0.01），且额外守住 mls 与 legacy 两条路径，
   任一算子被静默换掉都会变红。m3188 侧代码无任何独有语义。
2. **docstring 取 main 侧表格+三条 bullet，插入 m3188 独有段**：
   「full amplitude 措辞从未成立 / 0.71–0.79 系统性偏差 / WITHHELD >15% 门槛」
   及 `tests/test_effective_sigma.py` 的 per-scheme 重钉交叉引用。
   m3188 侧 bullet 历史（m2399/m2447/m2979）的语义已被 main 侧表格与 bullet 覆盖。
3. **数值分歧裁定**：m3188 docstring 记 m2399 时代 legacy=0.70，main 侧表格记
   legacy=0.215。在合并树上实测 legacy=**0.2147**（见下），与 main 侧一致；
   0.70 是 m2399 时代的历史记录（legacy 路径其后有改动），保留在 git 历史中，
   不写入现行断言。

### `docs/verification_effective_sigma.md`（m3188 新增，非冲突）

m3188 提交的文档表格只有 4 行，但其判定段引用了第 3 档 0.6500，
且其 `out/report.json` 实为 6 行——文档相对自身数据源过期（内部不一致）。
用 m3188 自带 `make_doc.py` 从已入库的 `report.json` 重渲染，表格补齐 6 行，
与判定段自洽。数字零手抄（程序化注入）。

## 测试证据（loong64，torch 2.7.1，`PYTHONPATH=<worktree>/src`）

| 树 | 结果 | 耗时 |
|---|---|---|
| 干净 `main`（基线） | 2 failed, **363 passed**, 8 xfailed | 505 s |
| 合并树 | 2 failed, **367 passed**, 8 xfailed | 338 s |

- 两侧失败集**逐条相同**：`test_oscillating_droplet.py` 的 2 个
  `XPASS(strict)`（mode-2 频率拟合架构敏感，x86 +8.64 % vs loong64 -31.78 %，
  main 提交 9f01f06 已记录），与本次合并无关，属 main 既有状态。
- 合并树多出的 4 个 passed = m3188 新增 `tests/test_effective_sigma.py` 全部 4 测。
- 冲突文件解冲后单测：`tests/test_laplace_pressure.py` 8/8 PASSED，
  `tests/test_effective_sigma.py` 4/4 PASSED。

## 关键量 vs 基准（合并树实测，R/h = 6.1538）

| 量 | 实测 | 钉/基准 | 相对解析 1.0 | <3 % ? |
|---|---|---|---|---|
| default（fit_quadratic） | 0.71062 | 0.711±0.01（main）/ 0.7106±0.02（m3188） | -28.9 % | **否**（WITHHELD，>15 % 门槛） |
| mls | 0.79074 | 0.791±0.01 | -20.9 % | **否** |
| legacy | 0.21468 | < 0.4（main 断言）；m2399 时代记录 0.70 | -78.5 % | **否**（历史缺陷，保留可达性） |

有效表面张力系统性偏低 21–35 %（随细化不收敛，`fit_quadratic` 单调远离 1.0），
是 m3188 已判定的 WITHHELD 验证欠账，本单只治愈合并、不改模型；
修复候选（δ_s 归一化 / κ·δ_s 圆滴标定）见 `docs/verification_effective_sigma.md`。

## 踩坑记录

1. 全局 editable 安装指向**别的** worktree（`work/m3482/src`），直接跑 pytest 会
   `ModuleNotFoundError: tensorsph.lattice_consistency`；必须
   `PYTHONPATH=<本 worktree>/src` 覆盖。
2. 工具 120 s 超时按进程组收割，`nohup &` 不够，需 `setsid` 脱离会话长跑测试。
3. 冲突块内 docstring 表格分隔线含 `=======` 子串，按子串切冲突会切错位置，
   必须按行锚定（`^<<<<<<<`/`^=======$`/`^>>>>>>>`）。
4. m3188 提交的文档相对其 `report.json` 过期（4 行表 vs 6 行数据，判定段引用
   缺失行）；用其自带 `make_doc.py` 重渲染修复，数字零手抄。
