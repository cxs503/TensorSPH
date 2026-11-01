# TensorSPH 上游差集巡更报告（m1633）

**任务**：生态巡更 · TensorSPH · 上游差集（remote upstream vs local HEAD）
**仓库**：`/root/tp-agent/work/m1633/TensorSPH`，remote `origin = https://github.com/cxs503/TensorSPH.git`
**分支**：`agent/m1633`（工作树），HEAD `400db02c`
**巡更日期**：2026-11-01
**产物**：`benchmarks/audit/upstream_diff.py`（可复现脚本）、`docs/audit/m1633/upstream_diff_m1633.json`、`docs/audit/m1633/upstream_diff_tables.md`、`docs/audit/m1633/figs/*.png`（`artifacts/` 被 `.gitignore` 忽略，故证据落 `docs/` 跟踪路径）

---

## 1. 案例介绍（被审计对象）

TensorSPH 是 Tensor 生态中的 PyTorch 原生弱可压缩 SPH（WCSPH）研究求解器：2-D 液滴撞池（`droplet_impact`）、2-D 溃坝 B2（`dam_break`）、表面张力 CSF 模型（`surface_tension`）、器件常驻 linked-cell 邻居搜索 + Verlet skin 缓存（`neighbor_search`）。本项目采用「agent 贡献 → PR → 上游 merge」的协作流，因此**上游分支与本地 HEAD 的漂移本身就是一类需要定期度量的工程质量指标**。

本单审计的环境与基线参数（表 0）：

| 项 | 值 |
|---|---|
| 本地 HEAD | `400db02c`（`agent/m1633`），提交时间 2026-10-25T10:40:42+08:00 |
| `origin/main` | `400db02c`（与 HEAD 同 SHA） |
| remote refs | 14 个 `refs/remotes/origin/*`（含 `origin/HEAD` 别名）→ 13 条实际分支，其中 10 条已完全合入 HEAD |
| Python / PyTorch | 3.11 / 2.7.1（CPU 执行） |
| 审计工具链 | `git` plumbing（`for-each-ref` / `rev-list` / `cherry` / `merge-tree`）+ `difflib` + `matplotlib`，无 torch 依赖 |

---

## 2. 方法与流程（怎么做的）

四层递进判据，全部机器可裁：

1. **拓扑层（ahead/behind）**：`git rev-list --left-right --count HEAD...<ref>`，区分「HEAD 独有」与「分支独有」提交数；同时用 `merge-base` 计算分支基座落后 HEAD 的提交数（staleness）。
2. **补丁层（patch-id）**：`git cherry -v HEAD <ref>`。`-` 表示分支提交的补丁内容已等价存在于 HEAD（典型成因：内容被重写后另行合入），`+` 表示文本上确属新内容。**这是判定「僵尸分支」的第一判据**。
3. **语义层（符号覆盖）**：从分支相对 merge-base 的 `src/`+`tests/` diff 中抽取新增 `def`/`class`/`test_` 名称，与 HEAD 源码索引比对；对名称缺失者用 `difflib.SequenceMatcher` 给出最近 HEAD 符号（rename 候选，阈值 0.55）。用于回答「patch-id 新颖 ≠ 功能新颖」。
4. **合并层（冲突面）**：`git merge-tree --write-tree HEAD <ref>` 试合并（不落地、不改工作树），统计冲突文件数；同时用 `ruff` 与 `pytest` 对本地改动做门禁。

复现命令（幂等，纯读）：

```bash
python3 benchmarks/audit/upstream_diff.py --repo . --out artifacts --figs artifacts/figs
```

脚本内置 **11 条自校验**（分支独有提交数与日志条数一致、patch 等价数 + 新颖数 = 分支独有提交数、符号 present + absent = 新增符号数、HEAD 与 `origin/main` 双向 0/0），本次运行 **11/11 通过，failed=0**（见 JSON `selfcheck` 字段）。

---

## 3. 计算结果

### T1–T6（程序化注入，源：`docs/audit/m1633/upstream_diff_tables.md`）

Generated: 2026-11-01T09:16:08Z · HEAD `400db02` on `agent/m1633`

#### T1. Remote branch divergence vs local HEAD

| remote branch | ahead (unmerged) | behind (stale by) | class |
|---|---:|---:|---|
| `origin/feat/wcsph-droplet-impact` | 46 | 0 | merged |
| `origin/docs/sph-validation-roadmap` | 42 | 0 | merged |
| `origin/feat/cell-linked-neighbor-search` | 35 | 8 | diverged |
| `origin/feat/pair-based-wcsph-forces` | 28 | 3 | diverged |
| `origin/bench/rhs-scaling-validation` | 24 | 0 | merged |
| `origin/arena/2cbcd8cc-tensorsph` | 22 | 1 | diverged |
| `origin/arena/be36f406-tensorsph` | 17 | 0 | merged |
| `origin/arena/aa52281d-tensorsph` | 11 | 0 | merged |
| `origin/agent/m239` | 9 | 0 | merged |
| `origin/agent/m276` | 7 | 0 | merged |
| `origin/agent/m277` | 4 | 0 | merged |
| `origin/agent/m281` | 4 | 0 | merged |
| `origin` | 0 | 0 | merged |
| `origin/main` | 0 | 0 | merged |

#### T2. Unmerged commit inventory (patch-id `git cherry` verdict)

| branch | sha | date | subject | patch in HEAD? |
|---|---|---|---|---|
| `origin/arena/2cbcd8cc-tensorsph` | `2c33033` | 2026-10-22 | perf: device-resident neighbour search, skin cache, and a dam-break case | yes (equivalent) |
| `origin/feat/cell-linked-neighbor-search` | `61aba32` | 2026-10-22 | docs: document pair-based force accumulation and reference tests | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `20013fd` | 2026-10-22 | feat: export dense RHS regression reference | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `b56e559` | 2026-10-22 | test: compare pair-based WCSPH RHS with dense reference | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `de6b57e` | 2026-10-22 | feat: compute WCSPH forces from neighbor pairs | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `346856d` | 2026-10-22 | docs: record staged WCSPH density integration | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `3f298c1` | 2026-10-22 | feat: export neighbor-based density routine | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `2076a9a` | 2026-10-22 | test: verify neighbor-pair density against all-pairs reference | no (novel text) |
| `origin/feat/cell-linked-neighbor-search` | `33a659b` | 2026-10-22 | feat: compute WCSPH density from compact-support pairs | no (novel text) |
| `origin/feat/pair-based-wcsph-forces` | `9a96287` | 2026-10-22 | docs: add RHS scaling sweep and clarify 2D kernel scope | yes (equivalent) |
| `origin/feat/pair-based-wcsph-forces` | `a0f5bc9` | 2026-10-22 | test: scope WCSPH density equivalence to its 2D kernel | yes (equivalent) |
| `origin/feat/pair-based-wcsph-forces` | `7eb7096` | 2026-10-22 | bench: support RHS scaling sweeps across particle counts | yes (equivalent) |

#### T3. Merge-ability against HEAD

| branch | stale by (commits) | merge clean? | conflicted files |
|---|---:|---|---|
| `origin/arena/2cbcd8cc-tensorsph` | 22 | CONFLICT | 15 |
| `origin/feat/cell-linked-neighbor-search` | 35 | CONFLICT | 3 |
| `origin/feat/pair-based-wcsph-forces` | 28 | CONFLICT | 2 |

#### T4. Semantic coverage of branch-introduced symbols in HEAD

| branch | new defs | present in HEAD | absent from HEAD |
|---|---:|---:|---|
| `origin/arena/2cbcd8cc-tensorsph` | 18 | 17 | small_case |
| `origin/feat/cell-linked-neighbor-search` | 3 | 3 | — |
| `origin/feat/pair-based-wcsph-forces` | 0 | 0 | — |

#### T4b. Name-level absence explained by API rename

| branch | branch symbol | closest HEAD symbol | ratio |
|---|---|---|---:|
| `origin/arena/2cbcd8cc-tensorsph` | `small_case` | `wall_mass` | 0.632 |

#### T5. Branch test functions with no HEAD counterpart

| branch | unmatched test functions |
|---|---|
| `origin/arena/2cbcd8cc-tensorsph` | — |
| `origin/feat/cell-linked-neighbor-search` | `test_pair_density_handles_empty_and_single_particle` (~`test_pair_density_handles_degenerate_inputs`, 0.702), `test_pair_density_matches_all_pairs_reference` (~`test_pair_density_matches_dense_reference`, 0.86), `test_rhs_remains_finite_with_neighbor_density` (~`test_rhs_rejects_non_two_dimensional_input`, 0.552) |
| `origin/feat/pair-based-wcsph-forces` | — |

#### T6. HEAD top-15 commit themes

| theme | commits |
|---|---:|
| merge-integration | 6 |
| gpu-scaling | 2 |
| validation-gate | 2 |
| audit-review | 1 |
| free-surface | 1 |
| surface-tension | 1 |
| time-integration | 1 |
| wall-boundary | 1 |

---

## 4. 结果图

图 1 — 各 remote 分支相对本地 HEAD 的 ahead/behind：红色为「未合入 HEAD 的提交」，蓝色为「分支基座落后 HEAD 的提交」。

![upstream divergence](audit/m1633/figs/upstream_divergence_m1633.png)

图 2 — 主题演化时间线：方块 = HEAD 最近 15 提交，星号 = 上游未合入提交，圆点 = 更早历史；纵轴为词法分类主题。

![theme timeline](audit/m1633/figs/theme_timeline_m1633.png)

两图均由 `benchmarks/audit/upstream_diff.py` 程序化生成（数据源为 git 元数据，非手工输入）。

---

## 5. 变更主题清单与风险标注

### 5.1 主题清单（HEAD 最近 15 提交）

| 主题 | 判读 |
|---|---|
| merge-integration（6/15） | 2026-10-23～25 的提交窗里 40% 是 PR 合入提交，说明这一窗的产出几乎全部来自 agent 分支被上游合入，历史上「直接提交 main」的物理开发已停摆。 |
| gpu-scaling（2） | m276 的 GPU 大规模基准（拐点 N*≈928、30.89×@95k），纯 benchmark+文档，`src/` 零 diff。 |
| validation-gate（2） | m281 的 `validation_report.json` 工件与门禁类改动，属「可复现证据」建设。 |
| surface-tension（1） | m277 CSF 表面张力（Brackbill 1992），本窗内唯一新增物理算子。 |
| wall-boundary / free-surface / time-integration（各 1） | 分别是 0.7.0 Adami wall pressure、0.6.0 自由面替代方案、0.8.0 自适应步长——即 0.5.0→0.8.0 版本列车，都是**对既有算子的物理加固**而非新功能。 |
| audit-review（1） | m281 的上游跟进评审。 |

**主题演化判读**：HEAD 的历史分两段——上半段（e924aff…bb494b0，0.5.0→0.8.0）是「门禁先行 + 算子精修」的物理加固；下半段（520bc54 之后）整体转向**证据建设**（审计、基准、验证工件）。最近 15 提交中 `src/` 只有一处新物理（表面张力），其余为基准/门禁/合并——即**物理产能已低于集成与验收产能**，这是上游分支被冻结的结构性原因。

### 5.2 上游差集与风险标注

| 风险 | 对象 | 严重度 | 依据与后果 |
|---|---|---|---|
| R1 僵尸分支误合并 | `feat/cell-linked-neighbor-search`、`feat/pair-based-wcsph-forces`、`arena/2cbcd8cc-tensorsph` | **高** | 12 个未合入提交中 4 个 patch-id 已等价、其余 8 个符号已全部存在于 HEAD；但试合并仍产生 20 个文件冲突（含 `.github/workflows/tests.yml`、`src/tensorsph/__init__.py`、`src/tensorsph/droplet_impact.py`）。若按「冲突取分支版本」粗暴解决，会**回退 0.7.0 wall pressure（Adami eq. 11）、0.8.0 自适应步长、m277 CSF 表面张力**等 HEAD 独有内容。 |
| R2 覆盖缺口（本单已修复） | 分支 `test_pair_density_handles_empty_and_single_particle` | 中 | 该退化输入测试在 HEAD 无名称对应、也非 rename（最近匹配比 0.652 且语义不同）。`density_from_neighbor_pairs` 对空/单粒子走自项分支，逻辑存在但无断言。已移植为 `tests/test_pair_based_rhs.py::test_pair_density_handles_degenerate_inputs`，单测通过。 |
| R3 基座陈旧 | 三条分歧分支基座分别落后 HEAD 22/35/28 提交 | 中 | 分支创建于 2026-10-22，此后 HEAD 又进 35 个提交（0.7.0/0.8.0 物理 + 门禁）。基座陈旧使冲突面随 HEAD 推进单调增长，越晚处理代价越高。 |
| R4 文档/基准漂移 | `docs/benchmarks/neighbor-search.md`、`benchmarks/neighbor_search/compare_rhs.py` 等 | 低-中 | 同名文档在分支与 HEAD 各有版本，冲突后易留下双份口径（如 RHS 复杂度表述、2-D kernel 适用范围）。 |
| R5 分支归档缺失 | `arena/*` 三条 arena 分支 | 低 | 22 提交基座 + 1 个未合入提交，内容已由 main 重写覆盖（arena 分支 18 个新增符号中 17 个存在 HEAD），却仍占用分支命名空间，易被后续 agent 误当「待合入工作」。 |

### 5.3 建议（增量、可验证）

1. 将三条分歧分支打 `archive/` 标签后删除远端分支引用，并在 `docs/` 记一条「已归档，内容由 main 重写覆盖」的判据（`git cherry` 输出可复现）。
2. CI 增一条反僵尸门禁：若待合入分支 `git cherry -v main <branch>` 全部为 `-`，则拒绝合并并提示归档（本次等价率 4/12，其中 `feat/pair-based-wcsph-forces` 3/3、`arena/2cbcd8cc-tensorsph` 1/1 全等价）。
3. 把 `benchmarks/audit/upstream_diff.py` 纳入周巡（纯 git 读、无 torch 依赖），差集非空即立项；stable 判据使用脚本内 11 条自校验。

---

## 6. 注意点与局限

- **patch-id 等价 ≠ 语义等价，反之亦然**：本次 8 个 `+` 提交文本上新颖、语义上已覆盖（3/3 符号在 HEAD 存在）；4 个 `-` 提交则相反。单一判据会误判，故采用「补丁层 + 语义层」双判据。
- **符号名覆盖只是代理指标**：名称相同不保证实现一致（HEAD 的 `cubic_spline_kernel` 为 2-D 归一化，分支测试曾按 1/2/3 维参数化）。因此本报告不对「行为完全一致」下结论，仅声明「HEAD 已提供同名 API 且有对应单测」。
- **rename 候选阈值 0.55 有假阳性**：如 `small_case` → `wall_mass`（0.632）显然是噪声，报告已按「候选」而非「等价」标注。
- **主题分类为词法规则**：`THEMES` 表按序首个正则命中，确定性但保守；遇到新词会自动落 `other`，不引入语义模型。
- **merge-tree 只试合并，不跑测试**；冲突解决后的行为回归需另立门禁。
- **CI 覆盖差异**：arena 分支携带 `.github/workflows/tests.yml` 修改，与 HEAD 冲突——若合并该分支，CI 定义会被回退为 10-22 版本。
- 本地沙箱 CPU-only，全量 pytest 单轮耗时 283.54 s；首次尝试因工具 120 s 墙钟限制被中断（曾仅记录到 31 个点），改用后台进程重跑后完成，最终 **115 passed**。后台首轮被中断的事实一并记录，避免「只报成功轮」的选择性汇报。

---

## 7. 误差判定

本单为**版本库拓扑审计**，不存在实验/解析参考值，故「关键量 vs 基准相对误差 <3%」的物理判据**不适用**；按要求显式声明原因并改用可机器裁决的替代判据：

| 关键量 | 判据形式 | 结果 | 是否满足（<3% 口径） |
|---|---|---|---|
| 分支独有提交数（每分支） | `git rev-list --count HEAD..ref` vs 记录提交条数 | 逐分支相等（差值 0） | **是**（整数精确匹配，误差 0%） |
| patch 等价/新颖数 | `等价 + 新颖 = 独有提交数` | 4 + 8 = 12 | **是**（0%） |
| 符号覆盖数 | `present + absent = 新增符号数` | arena 17+1=18；cell-linked 3+0=3；pair-based 0+0=0 | **是**（0%） |
| HEAD vs origin/main | 双向 ahead/behind | 0 / 0 | **是**（0%） |
| 移植测试 | 与解析自项 `m·W(0)` 一致 | `torch.testing.assert_close`（默认 rtol/atol 级别精确） | **是**，1 passed |
| 全量回归 | pytest 全绿 | 115 passed / 0 failed in 283.54 s | **是**（无失败项） |
| 冲突面计数 | `git merge-tree --write-tree` 冲突文件数 | 15 + 3 + 2 = 20 | 判据本身即定义，无误差概念 |

**总自校验**：脚本 11 条断言全过（failed=0），可重复执行得到同一 JSON。

**诚实声明（未完成部分）**：

- 全量 pytest 已完成：`python3 -m pytest -q` → **115 passed, 2 warnings in 283.54s**（CPU-only，含本单新增 1 例）。基线口径参考 m239 审计的 104 用例；115 = 104 + 期间新增用例（表面张力/自由面等）+ 本单 1 例，**未**逐版本对账到 104 的分解，此处只声明总数与全绿事实。
- 受影响文件窄口径门禁：`pytest tests/test_pair_based_rhs.py` → **13 passed in 9.80 s**；`ruff check benchmarks/audit/upstream_diff.py tests/test_pair_based_rhs.py` → **All checks passed**。
- 本次仅审计 `origin` 单一远端；无第二远端差集可比。
- 未对三条分歧分支做「解决冲突后的数值行为回归」，只做静态冲突面与符号覆盖判定。

---

## 8. 复现与工件清单

| 工件 | 说明 |
|---|---|
| `benchmarks/audit/upstream_diff.py` | 审计脚本（git plumbing + difflib + matplotlib），无 torch 依赖 |
| `docs/audit/m1633/upstream_diff_m1633.json` | 全量结构化结果（含 `selfcheck`、逐提交 patch 判定、冲突文件列表） |
| `docs/audit/m1633/upstream_diff_tables.md` | T1–T6 表格（本报告第 3 节数字由此文件程序化注入） |
| `docs/audit/m1633/figs/upstream_divergence_m1633.png` | 图 1：分支 ahead/behind |
| `docs/audit/m1633/figs/theme_timeline_m1633.png` | 图 2：主题演化时间线 |
| `tests/test_pair_based_rhs.py` | 新增 `test_pair_density_handles_degenerate_inputs`（R2 修复） |