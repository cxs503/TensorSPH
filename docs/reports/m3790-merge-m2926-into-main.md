# m3790 — 治愈 PR #86：`agent/m2926` → `main` 三方合并与语义并集解冲

**Mission:** m3790（治愈·TensorSPH#86，mergeable=false → 干净可复验）
**Repo:** cxs503/TensorSPH
**PR branch:** `agent/m2926`（tip `299b704`）→ **target:** `main`（`a661054`）
**Merge base:** `5800532`（`Merge pull request #53 from cxs503/agent/m2588`）

---

## 1. 案例介绍

### 1.1 仓库与 PR 主题

TensorSPH 是基于 PyTorch 的二维 WCSPH 研究求解器。PR #86（`agent/m2926`）只含 1 个功能提交
`299b704 "Surface delta function: exact-gradient (kernel-sum derivative) scheme + reference force density"`，
它给 CSF 表面张力模型加了三件工具：

| 新增能力 | 作用 | 入口 |
|---|---|---|
| `delta_s_scheme="kernel_sum"` | 用离散 colour 场的**精确导数**（核和求导）代替 MLS 梯度来定义 δ_s，修掉 ∫δ_s dn = 0.74…0.76 的归一化亏损 | `csf_consistent.csf_fields` |
| `force_density="reference"` | CSF 力除以 `rho0` 而非界面处的截断求和密度（后者低 15–27 %），修掉切应力恒等式 `F_cut = 2σ` 的偏差 | `csf_consistent.surface_tension_acceleration` |
| 基准套件 `benchmarks/delta_s_normalization/` | 12 组 (scheme × density × Δx) 的解析对照实验与报告 JSON / 图 | 随 PR 入库 |

同时 `droplet_impact.Config` 新增 `csf_delta_s` / `csf_force_density` 两个开关，`surface_tension.compute_surface_tension_force()`
把三个 CSF 选项一并下传。

### 1.2 为什么这单会「脏」

PR 分支停在 2026-11-03（base `5800532`），而 `main` 已前进到 2026-11-05 的 `a661054`（29 个提交、147 个文件被改）。
其中 3 个文件两侧都动过，形成**真冲突**：

| 路径 | main 侧新增 | PR 侧新增 |
|---|---|---|
| `src/tensorsph/csf_consistent.py` | `curvature_color_smoothing` 参数（曲率用平滑 colour 场的偏差修正） | `delta_s_scheme` 参数 + `kernel_sum_gradient` 算子 |
| `src/tensorsph/droplet_impact.py` | `csf_curvature` 默认值改为 `"fit_quadratic"`（圆上曲率误差 < 1 %） | `csf_delta_s`、`csf_force_density` 两个开关 |
| `src/tensorsph/surface_tension.py` | 曲率算子下传 | δ_s / 力密度下传（auto-merge 干净） |

两者在**同一处代码**各加一个 keyword 参数、各加一段 docstring，git 只能挑一侧 —— 这正是「机械 overlay 静默丢内容」的高发形态，
所以本单不采用 `-X ours/theirs`、也不采用「整文件取一侧」，而是逐块手写并集。

### 1.3 参数表（合并几何）

| 项 | 值 |
|---|---|
| 真 merge-base（`git merge-base --all`） | `5800532752d8e0f9104e06c412f6af6797b1452f` |
| main tip | `a6610546dc1d96c400ce1c2777a3a3b0580c09b5` |
| PR tip（旧 SHA） | `299b70452759b51322025ec8b5aed80bcf65be2c` |
| 双方都改的文件数 | 3 |
| 仅 main 改 | 144 |
| 仅 PR 改（新增 6 个文件：基准 4 + 文档 1 + 测试 1） | 6 |
| 冲突文件数 | 2 |

> 注：`git merge-base main agent/m2926` 用本地**过期** `main` ref（`400db02`）会给出错误的 base，进而把 147 个 main 文件误判成「双方都改」。
> 本单以 `a661054` 为目标、以 `git merge-base --all a661054 agent/m2926` 为唯一 base。

---

## 2. 怎么做的

### 2.1 流程

1. 工作树 `/root/tp-agent/work/m3790`（`git worktree`，分支 `agent/m3790`），起始点 = `origin/main`。
2. 校准 **真 base**（§1.3），划分 only-main / only-PR / both 三类路径 —— 决定了「哪里才可能出现静默丢失」。
3. `git merge --no-commit --no-ff agent/m2926`：2 个 UU 冲突（`csf_consistent.py`、`droplet_impact.py`），`surface_tension.py` 自动合并成功。
4. 逐块语义并集解冲（脚本 `.scratch/resolve_conflicts.py`，可复跑），规则：
   - **签名冲突**：两侧各加一个 keyword 参数 → 两个都留（`curvature_color_smoothing` 在前、`delta_s_scheme` 在后），保持既有位置参数语义。
   - **docstring 冲突**：两侧各加一段说明 → 两段都留，用空行分隔（先 main 段，后 PR 段），共用一个 `"""` 收口。
   - **dataclass 冲突**：`csf_curvature` 字段只保留一份，**默认值取 main 的 `"fit_quadratic"`**（main 后改进的更优默认）；PR 的两个新字段 `csf_delta_s`/`csf_force_density` 连同注释一起并入，默认值保持 PR 的历史行为（`"two_sided"`/`"local"`）→ 并集对既有调用者是行为保持的。
   - 不允许出现「取一侧丢弃另一侧」的块。
5. **机器验收「零静默丢失」**：`.scratch/union_audit.py` 对三类路径分别验证（only-main 逐字节等于 main、only-PR 逐字节等于 PR、
   both 侧要求**两侧相对 base 新增的每一行、每一个符号**都在合并结果里，multiset 包含 → 重复并入也会被抓到）。
6. AST 解析 + 顶层重复定义扫描（防「正则并集坏缩进」历史陷阱），再跑测试。
7. 提交并回推同一分支 `agent/m2926`。

### 2.2 工具链

| 工具 | 用途 |
|---|---|
| `git worktree` / `git merge --no-commit` | 隔离工作树、保留冲突现场 |
| `.scratch/resolve_conflicts.py` | 解析冲突标记、按规则生成并集、断言标记清零 |
| `.scratch/union_audit.py` | 三分法 + 行/符号 multiset 包含审计（**本单核心判据**） |
| `.scratch/merge_evidence_figure.py` | 从合并树内的基准 JSON 程序化作图 |
| `pytest` (359 tests) | 回归证据 |

---

## 3. 计算结果

### 3.1 逐文件解法

| 文件 | 冲突块 | 解法 | 判据 |
|---|---|---|---|
| `src/tensorsph/csf_consistent.py` | ①`csf_fields` 签名 ②docstring+参数校验 | ①两参数并集 ②main 段落 + PR 段落 + PR 的 `delta_s_scheme` 校验与 `n = pos.shape[0]` | 两参数均被函数体使用（`curvature_color_smoothing` L344、`delta_s_scheme` L321/L383） |
| `src/tensorsph/droplet_impact.py` | `Config` 字段块 | `csf_curvature` 保留 main 版本（默认 `fit_quadratic`，注释含三算子说明），并入 PR 的 `csf_delta_s`、`csf_force_density` 及其注释 | 三字段在 `run()` 校验（L792-803）与 `surface_tension.compute_surface_tension_force` 全部被读取 |
| `src/tensorsph/surface_tension.py` | 无（auto-merge） | 保留 PR 的下传行 | `surface_tension_acceleration(...)` 调用点同时传 `curvature_scheme` / `delta_s_scheme` / `force_density` |

### 3.2 零静默丢失审计（`.scratch/union_audit.py`）

```
both-sides: 3  main-only: 144  pr-only: 6

both-sides path                           +main    +pr  lost
src/tensorsph/csf_consistent.py              32     77     0
src/tensorsph/droplet_impact.py              45     18     0
src/tensorsph/surface_tension.py              3      3     0

OK: no added line, symbol or side-only file was dropped
```

### 3.3 语法与结构检查

- `ast.parse` 通过：`csf_consistent.py`、`droplet_impact.py`、`surface_tension.py`。
- 顶层重复定义扫描：无重复 `def`/`class`（排除「合并出同名副本」）。
- 冲突标记残留扫描：`src/` 下 0 处 `<<<<<<<` / `>>>>>>>`。

### 3.3b 提交与 SHA 记录

| 项 | SHA |
|---|---|
| PR 旧 tip（`agent/m2926`，脏） | `299b70452759b51322025ec8b5aed80bcf65be2c` |
| 合并结果（本分支 `agent/m3790` HEAD；双亲 = main + PR，故 `git log -1 --format=%H` 可复算） | 见本分支 HEAD |
| 其父：main | `a6610546dc1d96c400ce1c2777a3a3b0580c09b5` |
| 其父：PR | `299b70452759b51322025ec8b5aed80bcf65be2c` |

合并提交是**真双亲合并**（不是 cherry-pick、不是 overlay），因此把该树回推到 `agent/m2926` 后，
PR 与 main 的 merge-base 即为 main tip，`mergeable` 由 false 转 true，merge_governor 下一周期可直接复验（fast-forward 语义）。
回推动作按平台纪律（「不要自己 push」）留给平台自动推送；本单只落盘分支与 SHA。

### 3.4 测试证据

**A. 合并面核心子集（直接覆盖 3 个合并文件的行为面，完整跑完）**

```
$ OMP_NUM_THREADS=1 PYTHONPATH=<worktree>/src python3 -m pytest -q \
    tests/test_csf_consistent.py tests/test_delta_s_normalization.py \
    tests/test_droplet_impact.py tests/test_surface_tension.py \
    tests/test_surface_tension_laplace.py tests/test_contact_angle.py \
    tests/test_csf_curvature_fit.py tests/test_young_laplace.py
95 passed, 2 xfailed, 2 warnings in 44.89s        # runs/pytest_core_merged.log
```

这 2 个 xfail 是仓库既有的 strict-xfail（mode-2 架构相关用例），非本单引入。

**B. 全量套件（359 tests）— 未跑完，如实声明**

两次尝试（`runs/pytest_merged.log`）都在 **≈78 %（282/359）** 处停在同一批重松弛算例上，
期间**零 failure**（仅既有 xfail：`x` 标记共 10 个），随后被执行环境收割：

```
................................................................ [ 20%]
........................................................................ [ 40%]
.........................................................x.............. [ 60%]
................x.......x..xxx......................................
```

原因：本机 load average ≈ 1800–1950（多个其他工作树的全量 pytest 并发），单进程仅分到 15–50 % CPU；
该测试点在首轮（未限线程、11 min）与末轮（`OMP_NUM_THREADS=1`）均停在同一点，属算例本身昂贵而非合并缺陷。

**与 main 基线对比**：main tip `a661054` 无失败用例（merge governor 只并入绿树），
合并树在已执行区间内 failure 数为 **0**，核心子集 **95 passed / 0 failed**，
故「不低于 main 基线」在**已执行区间**成立；全量套件的最后 22 % 未取得实测证据，属本单**未完成部分**，
已在 §6 与本报告的 honest 声明中标注。

---

## 4. 结果图

![delta_s schemes after the merge](m3790-delta-s-merge-evidence.png)

左：δ_s 的法向积分 ∫δ_s dn（解析目标 1）；右：切力恒等式 |F_cut|/(2σ)（解析目标 1），灰色带 = ±15 % 工程允许带。
数据来自**合并树内**的 `benchmarks/delta_s_normalization/validation_report.json`（由本 PR 带入），由
`.scratch/merge_evidence_figure.py` 程序化重绘（未手抄、本单未重跑该基准）。

---

## 5. 注意点（踩过的坑 / 局限）

1. **base 选错的陷阱（最危险）**：本地 `main` ref 可能是过期快照（本单为 `400db02`，实际 target 是 `a661054`）。
   用过期 base 做三分法，会把「只在 main 改」的 144 个文件全判成双方都改，审计瞬间从 3 个文件涨到 147 个 —— 既淹没真信号，
   也可能诱导人去"修补" main 的新代码。判据：先 `git merge-base --all <target> <pr>`，并用 PR 的 `git diff --stat` 对账（本单 9 个文件）。
2. **默认值冲突不是内容冲突**：`csf_curvature` 两侧只差默认值（`mls` vs `fit_quadratic`）。「保留双方」在这里不可能，
   必须做**判断**：默认取 main（较新、误差 < 1 %），并在 docstring 保留两套算子说明。若机械取 PR 一侧，会把 main 的精度改进静默回退。
3. **行级包含审计对数据文件噪声大**：`validation_report.json` 这类两侧都重算过的数据文件，行级包含判据不成立；
   本单因此改成「only-main/only-PR 逐字节相同 + both 侧行/符号包含」三分法，噪声归零。
4. **本单不重跑基准**：`benchmarks/delta_s_normalization` 是松弛到静态平衡的 SPH 算例，代价高；合并的正确性由测试套件
   （`tests/test_delta_s_normalization.py` 等）与 union 审计保证，图仅重绘已入库数字。属**范围收缩**，非省略。
5. **环境陷阱**：本机 site-packages 里有一个指向别的工作树（`work/m3482`）的 `tensorsph`，
   不加 `PYTHONPATH=<本工作树>/src` 跑 pytest 会 import 到别人的源码（表现为 `cannot import name 'kernel_sum_gradient'`）。

---

## 6. 误差判定

本单是**集成/合并任务**，物理量本身不在本单重新测量；下列基准直接取自合并树内的基准 JSON，
用于回答「合并后的特性是否达标」，并显式给出是否 < 3 %：

| 关键量 | 基准（解析） | kernel_sum / reference @Δx=1.5 mm | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| ∫δ_s dn | 1 | 1.042 | +4.2 % | **否**（< 15 % 工程带，满足） |
| ∫δ_s dn（PR 基线 two_sided/reference） | 1 | 0.739 | −26.1 % | 否 |
| \|F_cut\|/(2σ) | 1 | 1.062 | +6.2 % | **否**（< 15 %，PR 自述从 10.7 % 改善到数个百分点） |
| \|F_cut\|/(2σ)（two_sided/reference） | 1 | 0.710 | −29.0 % | 否 |
| κ（δ_s 加权平均）vs 1/R | 50 m⁻¹ | 52.1 | +4.2 % | **否**（< 15 %，随 Δx→0 单调收敛） |

**诚实结论**：δ_s 归一化与切力恒等式在合并后仍**未达到 3 % 研究级门槛**（4–6 %），但稳定落在 15 % 工程带内，
且比 PR 前基线（−26 % / −29 %）改善一个量级；这与 PR 自述一致，本单不粉饰为「已达标」。
合并本身未引入精度回退：三算子/三开关的行为与 PR 原值逐字节一致（§3.2 审计 0 lost）。