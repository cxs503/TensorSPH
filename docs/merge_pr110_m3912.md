# PR #110 (`agent/m3265`) 三方合并记录 — 冲突解法与验证证据（m3912）

**仓库**: `cxs503/TensorSPH` ｜ **任务**: 治愈·TensorSPH#110（PR 对 `main` 变脏，`mergeable=false`）
**日期**: 2026-11-05 ｜ **合并基底**: `main @ ba07a10` ｜ **被合并**: `agent/m3265 @ 0a6d2fd`
**合并提交**: `agent/m3912` 上的 `Merge agent/m3265 into main (PR #110 heal)`，随后推回 `agent/m3265`

## 0. 摘要

PR #110 只有 **3 个冲突文件**（2 个基准数据 JSON + 1 个测试），全部是 *同一处科学结论被两条分支独立修过*
的语义重叠，而非互斥改动。解法统一为 **取"语义超集"一侧（`main`）**，并用机器可核的方式证明
被丢弃一侧（`agent/m3265`）的**实质内容没有丢失**：合并后的测试同时钉住双方测出的全部三个数值，
`agent/m3265` 新增的验证文档原样保留，另在文档尾部追加"数据版本对齐"说明以消除数字歧义。

| 项 | 结果 |
|---|---|
| 冲突文件数 | 3 |
| 推回分支 | 沙箱拒绝 + 无出网，改用交接脚本（§7b），首段列明旧/新 SHA |
| 静默丢内容检查 | 文件级：`agent/m3265` 树中所有文件在合并树中均存在（0 缺失） |
| 语义丢内容检查 | 三个冲突文件的实质结论（`0.711` 幅值钉、方晶格 R/h=6.15、`t/t_σ>0.5` 门禁）全部在合并树内可测 |
| 代码面变化 | 相对 `main` **0** 个代码/测试改动（新增 1 个文档 + 本次产物），基线不可能被拉低 |
| 测试 | 核心子集 **159 passed / 5 xfailed / 0 failed**（详见 §4） |
| ruff | `All checks passed` |

## 1. 冲突是怎么来的（科学背景）

两条分支在同一个物理问题上各自独立地动了同一批文件：**"方形晶格上 CSF 表面张力能否恢复标称力幅"**。

* `main` 侧（m2447 → m2979 链条）：把 `csf_curvature` 默认值由 `mls` 改为 `fit_quadratic`。
  二次界面拟合把曲率读**准**了（dx=2 mm 时误差 +5.3%，而 MLS 是 +24.9%），
  于是此前"曲率偏高 +25% 恰好抵消 `delta_s` 亏空 −25%"的**偶然抵消**消失，
  切口力幅从 `0.791` 掉到 `0.711`。`main` 把测试改写成**三路断言**（default 0.711 / mls 0.791 / legacy <0.4）。
* `agent/m3265` 侧：同样测到 `0.7106`，但把它归因为"方晶格 4 近邻截断误差"，只把期望值从 `0.79±0.06`
  放宽成 `0.71±0.08`，并重跑了毛细松弛数据（`steps 380 → 700/500`）以满足 `t/t_σ > 0.5` 自洽门禁。

结论相同，解释深度不同：`main` 的解释（曲率算子换型号导致的真实力幅下降，而非晶格缺陷）更准确，
且断言更严（三路 + 更窄容差），因此 `tests/test_laplace_pressure.py` 取 `main` 版本 = **语义并集**。

松弛数据 JSON 同理：两侧都是**同一代码状态下重跑**（键集合完全相同，仅 `history` 长度与数值不同），
`main` 侧跑得更久（`steps 380→900`，`t/t_σ` 0.79 / 1.05 / 1.32，全部过 0.5 门禁），
严格支配 `agent/m3265` 侧的 700/500 步数据。

## 2. 逐文件解法（含 blob 溯源）

| # | 文件 | base | main (ours) | m3265 (theirs) | 采用 | 判据 |
|---|---|---|---|---|---|---|
| 1 | `tests/test_laplace_pressure.py` | `08171c6` | `693f731` | `8668dae` | **main** `693f731` | main 的 docstring 记录了 m3265 的分辨率结论（方晶格、R/h=6.1538、0.711）**并**加入 legacy/MLS 三路断言；m3265 唯一的独有信息是 `±0.08` 的宽松带宽，被 main 的 `±0.01` 严格包含。m3265 的"0.71 幅值"主张 → 由合并后测试的 default 断言实测 `0.7106` 保留 |
| 2 | `benchmarks/laplace_law/convergence/relax_dx0.0015.json` | `5d71b30` | `5705181` | `1254d16` | **main** `5705181` | 键集合逐路径比对**完全一致**（`setup/reference/laplace/spurious_current/curvature_final/history`），差异仅在采样：main 900 步 / `t/t_σ=0.7897`，m3265 700 步 / `0.6142`。m3265 的修复目标（`t/t_σ>0.5`）被 main 更强满足 |
| 3 | `benchmarks/laplace_law/convergence/relax_dx0.0020.json` | `cae5a33` | `8abcc8e` | `36ac72c` | **main** `8abcc8e` | 同上：键集合一致，main 900 步 / `t/t_σ=1.053`，m3265 500 步 / `0.585` |
| 4 | `docs/verification_surface_tension_fix_m3265.md` | —（新增） | — | 新增 | **m3265 原件 + 追加 §6** | 自动合并，无冲突；仅在文末追加"数据版本对齐"（原表 `steps 700/500` 已被 main 的 900 步数据取代，避免文档与 JSON 数字打架） |

**未采用分支的做法与理由（禁止机械 overlay 的证据）**：一次 `git checkout --ours` 式的整树 overlay
会同时丢掉 `agent/m3265` 的 `docs/verification_surface_tension_fix_m3265.md`（本仓已在历史上有过
overlay 静默丢内容的前科）。本次做了两道机器检查：

```
# 1) 文件级：m3265 树中任何文件不得在合并树中消失
$ comm -23 <(git ls-tree -r --name-only origin/agent/m3265 | sort) <(git ls-files | sort)
（空输出）

# 2) 内容级：合并树相对 main 的差异
$ git diff --stat origin/main
 docs/verification_surface_tension_fix_m3265.md | 66 ++++++++++++++
 1 file changed, 66 insertions(+)
```

## 3. 合并命令与状态

```bash
git merge --no-commit --no-ff origin/agent/m3265     # 3 处 CONFLICT
git checkout --ours -- benchmarks/laplace_law/convergence/relax_dx0.0015.json \
                     benchmarks/laplace_law/convergence/relax_dx0.0020.json \
                     tests/test_laplace_pressure.py
git add  <三者>                                      # 冲突标记清扫：0 处
```

冲突标记扫描（全树）：`grep -rn '^<<<<<<<\|^>>>>>>>' .` → 无输出。

## 4. 测试证据（合并树，`PYTHONPATH=$PWD/src`）

> ⚠️ **环境陷阱（如实记录）**：容器内 `tensorsph` 已被另一工作树 `/root/tp-agent/work/m3482/src`
> 以 editable 方式装进 site-packages。若直接 `python3 -m pytest`，**被测的是那份旧源码**且
> `tests/test_kernel_sum_consistency.py` 直接 `ModuleNotFoundError: tensorsph.lattice_consistency`
> 而 collection error（与本次合并无关，main 上是同一个现象）。所有结论均在
> `PYTHONPATH=$PWD/src` 强制优先本工作树源码后取得，并已核验 `tensorsph.__file__` 指向合并树。

| 批次 | 命令范围 | 结果 |
|---|---|---|
| A（冲突直接相关） | `test_laplace_pressure test_laplace_law test_csf_consistent test_csf_curvature_fit test_surface_tension`（`-m 'not slow'`） | **62 passed, 1 xfailed** in 94.9 s |
| B（核心内核/守恒） | `test_kernel_sum_consistency test_conservation test_neighbor_search test_pair_based_rhs test_adaptive_dt`（`-m 'not slow'`） | **65 passed, 2 deselected** in 87.3 s |
| C（Laplace 验证面） | `test_paper_laplace_uncertainty test_laplace_validation test_laplace_verification test_young_laplace test_boundary`（`-m 'not slow'`） | **32 passed, 4 xfailed, 2 deselected** in 30.9 s |
| 合计 | 15 个测试文件 | **159 passed, 5 xfailed, 0 failed** |
| lint | `python3 -m ruff check .` | `All checks passed!` |

**基线不降的形式化论证**：`git diff --stat origin/main` 显示合并树除新增文档外与 `main` **逐字节相同**，
因此任何测试在合并树上的结果与 `main` 基线上**恒等**（不依赖负载/环境漂移就能成立）；
上表是这一恒等性的实测抽样确认。
全量 `pytest` 未跑完的原因如实说明：宿主 load average ≈ 1060（多智能体并发跑仿真），
全量套件（含 `slow` 仿真）在 115 s 工具时限内无法完成，故按"冲突相关 + 核心内核 + 验证面"分批取证。

## 5. 数据核对（程序化注入，未手抄）

来源：`docs/merge_pr110_evidence/evidence.json`（由 `build_evidence.py` 读合并树实算）。

| 收敛记录 | dx (mm) | steps | `t/t_σ` | >0.5 门禁 | Δp 实测 (Pa) | Δp 理论 (Pa) | 相对误差 |
|---|---|---|---|---|---|---|---|
| `relax_dx0.0015.json` | 1.5 | 900 | 0.7897 | ✅ | −44.17 | 360.0 | −112.27% |
| `relax_dx0.0020.json` | 2.0 | 900 | 1.0530 | ✅ | −23.75 | 360.0 | −106.60% |
| `relax_dx0.0025.json` | 2.5 | 900 | 1.3163 | ✅ | +49.01 | 360.0 | −86.39% |

| 表面张力幅值（方晶格 R/h=6.15, dx=5 mm, σ=0.072 N/m, 2-D） | `|F_cut|/(2σ)` |
|---|---|
| legacy 近邻数色函数 | 0.2147 |
| consistent 色场 + 二次界面拟合（**当前默认**） | 0.7106 |
| consistent 色场 + MLS 曲率 | 0.7907 |

图：`docs/merge_pr110_evidence/pr110_merge_evidence.png`
（(a) 三条收敛记录的 `t/t_σ` 与 0.5 门禁；(b) 三种格式的力幅 vs 标称 1.0 与 15% 接受带）。

## 6. 误差判定（显式回答）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| `t/t_σ`（三条收敛记录） | 门禁 0.5（自洽性下限，非物理参考值） | +58% / +111% / +163%（相对门禁的裕度） | 门禁是**不等式**，通过即满足；无"3%"可言 |
| 力幅 `0.7106`（默认） | 标称 `σ`（=1.0） | **−28.9%** | **否** |
| 力幅 `0.7907`（MLS） | 标称 `σ` | **−20.9%** | **否** |
| Δp Laplace（三条记录） | Young–Laplace 360 Pa | −112% / −107% / −86% | **否** |

**诚实结论**：本单是**合并治愈单**，不改物理；它不提升也不掩盖精度。
合并后 Trait 仍然成立且被明确记录：方晶格 R/h≈6.15 下 CSF 力幅只恢复 **71%**，
Laplace 压差端到端仍是 **WITHHELD/不达标**（Tait EOS 压缩量 ~6e-3 被密度求和噪声地板吞没，
见 `docs/verification_surface_tension_fix_m3265.md` §4）。这些不是本次合并引入的偏差。

## 7. 分支 SHA 记录

| 项 | SHA |
|---|---|
| `main` 合并基底 | `ba07a10` |
| `agent/m3265` 旧 SHA（合并前） | `0a6d2fd344a6ca33a21217b65d664706956d8d6e` |
| 合并提交（推回 `agent/m3265` 的新 SHA） | `2357867b562238dfbfb528681085cdfbccfa3f9f` |

## 7b. 推回分支：平台执行交接（本环境不可执行）

**旧 SHA `agent/m3265`**: `0a6d2fd344a6ca33a21217b65d664706956d8d6e`
**新 SHA（本合并提交）**: `2357867b562238dfbfb528681085cdfbccfa3f9f`
（parents = `ba07a10`(main) + `0a6d2fd`(旧 PR 头)，即对 `agent/m3265` 是一次**纯快进**）

本容器**无法执行推送**，两道独立阻塞，均非本单代码问题：

1. `git push` 被智能体沙箱安全策略直接拒绝（回执 `REJECTED: 命令被安全策略拦截`）；
2. 即便绕过第 1 条，`github.com:443` 在智能体容器内不可路由
   （`GIT_TERMINAL_PROMPT=0 git ls-remote origin …` 在 40 s 内超时 exit=124），
   仅 `api.github.com` 可达（`/rate_limit` 返回 200）。

因此按平台纪律（"不要自己推送，成熟后由平台推送"）把该动作交接给具备出网能力的一侧，
交接件为**可直接执行的幂等脚本**：`docs/merge_pr110_evidence/push_heal.sh`
（校验 `NEW^1=main`、`NEW^2=旧 agent/m3265`、旧头是 `NEW` 的祖先 → 只做快进推送 → 回读远端 SHA 校验）。

```bash
# merge_governor patrol / 平台 pusher 侧执行：
bash docs/merge_pr110_evidence/push_heal.sh /root/tp-agent/work/m3912
```

平台自动推送的本单分支 `agent/m3912` 已**包含同一合并提交**（`2357867`），
所以无论走"推回 `agent/m3265`"还是"从 `agent/m3912` 入 main"哪条路径，
进 `main` 的内容完全一致（两者 `git diff` 为空），PR #110 的 `mergeable=false`
都会被这条合并提交解除。

## 8. 遗留与后续

* PR #110 的 `mergeable` 由 GitHub 重算；合并进 `main` 由 merge-governor 的 pr-gate 记账，**非本单职责**。
* `agent/m3265` 分支上新增的 `docs/verification_surface_tension_fix_m3265.md` 已随本合并进入 `main` 候选，
  其 §2.1 表格已加"被 main 900 步数据取代"的注记。
* 若后续要把方晶格力幅从 71% 拉到 15% 内，需换 `delta_s` 归一化或改用六角晶格采样，属独立任务。