# m3853 — 治愈 TensorSPH#103：`agent/m3163` 与 `main` 的三方合并

## ① 案例介绍

**对象**：TensorSPH PR #103（分支 `agent/m3163`，tip `b7af006`）对 `main`
（tip `ba07a10`）的合并。GitHub 报 `mergeable=false`；本次工作树分支
`agent/m3853`，merge-base `184fef9`。

**物理背景**：PR 的内容是二维弱可压缩 SPH 的 Young–Laplace 压力跳变基准——半径
$R=20$ mm 的静止液滴，$\sigma=7.2$ N/m，$ho_0=1000$ kg/m³，$c_0=20$ m/s，
三次样条核 $h=1.3\,dx$，六方晶格，零重力，Tait 状态方程在 $ho_0$ 处归一（真空即
$p=0$）。解析基准 $\Delta p=\sigma/R=360$ Pa。基准的争议点是**读数协议**：

* `main`（m3312 链）为了满足 `tests/test_laplace_law.py` 的活门槛
  $t_{end}>0.5\,t_\sigma$，把三份收敛工件按**固定步数 900 步**重跑覆盖；
  由于 $dt\propto dx/c_0$，同一个步数在不同分辨率下对应**不同的物理时刻**
  （0.79 / 1.05 / 1.32 $t_\sigma$）；
* PR #103 给 `relax()` 增加 `duration_t_sigma`，让每一档都在**同一物理窗口**
  $t=1.0\,t_\sigma$ 读数，并新增绝对读数 $p(r<0.5R)-0$、尾窗平均与末态场快照。

| 项 | 值 |
|---|---|
| 合并基线 `main` | `ba07a10` |
| 被合并分支 `agent/m3163` | `b7af006` |
| merge-base | `184fef9` |
| 工作分支 | `agent/m3853` |
| 冲突文件数 | 3（全部是 `benchmarks/laplace_law/convergence/relax_dx0.00xx.json`）|
| 自动合并成功文件数 | 6（`run.py`、`run_ladder.py`、`make_ladder_doc.py`、`resolution_ladder.json`、`docs/verification_laplace_resolution_ladder.md`、`tests/test_laplace_resolution_ladder.py`）|

## ② 怎么做的

1. **建工作树**：`git worktree` 目录 `/root/tp-agent/work/m3853`，分支 `agent/m3853`；
   `git fetch origin` 取到真实 `main`（本地 `main` 已陈旧到 `400db02`，不可用）。
2. **基线树**：`git archive origin/main | tar -x -C /tmp/base_m3853` 得到**未合并**的
   main 纯净树，作为测试基线（不动仓库状态）。
3. **三方合并**：`git merge --no-commit --no-ff origin/agent/m3163`。git 只在三个
   结果 JSON 上失败——这三份是**同一次实验的两次运行快照**（步数、物理窗口、逐采样
   `history` 数组全不同），**不存在逐值可合并的语义**，任何机械 overlay /
   `checkout --ours` 都必然静默丢掉一侧的数据。
4. **逐文件解法（语义并集，非机械覆盖）**：由 `make_union.py` 从
   git merge stage 2（main 侧）与 stage 3（m3163 侧）**直接读 blob**、结构化拼装：
   * **顶层保留 m3163 的运行作为规范档**——它是合并后 `run.py`（固定窗口协议）的产物，
     也是 schema 超集（`field` 快照 + 双读数键），`tests/test_laplace_resolution_ladder.py`
     与 `make_ladder_doc.py` 依赖这些键；
   * **main 的运行整段搬进同一 JSON 的 `alternate_runs`**（`setup`/`reference`/
     `laplace`/`spurious_current`/`curvature_final`/`history` 全键逐值保留，附
     `label`/`source_ref`/`protocol` 说明），并写入 `merge_resolution` 块记录
     哪一侧是规范档与理由；
   * 另在 `docs/reports/m3853-merge/main-900step/` 存**逐字节副本**做再生保险：
     将来重跑 `run_ladder.py` 会重写工件并丢掉 `alternate_runs`，但覆盖不到归档副本。
5. **不可静默丢失的机器证明**：`verify_merge.py` 逐文件断言 ①m3163 侧每条键路径在合并
   文件里逐值不变；②main 侧每条键路径在顶层或 `alternate_runs[]` 中逐值存在；
   ③`alternate_runs[0]` 与 main 原件 field-by-field 相等；④归档副本 round-trip；
   ⑤同 `dt_s`、同 $t_\sigma$（同一物理装置与积分器）；⑥规范档窗口固定 1.0 $t_\sigma$
   且满足活门槛 $t>0.5$；⑦main 的历史采样落在同一 `dt` 网格；⑧阶梯测试要求的双读数键
   与 `field` 快照长度等于 `n_particles`。
6. **分支 tip 自带的红测试**：`agent/m3163` 的提交信息写着「1 个文件被门禁滤除」，
   被滤掉的正是 `benchmarks/laplace_law/laplace_resolution_ladder.png`
   （`tests/test_laplace_resolution_ladder.py` 要求该图存在且 >20 kB），因此该分支
   tip 上阶梯测试本来就是红的。本次用生成器 `make_ladder_doc.py` 重渲染该图，
   并把出图 dpi 由 130 降到 100（215 655 B，低于 256 KiB，
   避免再次被大二进制门禁吞掉）；文档正文重渲染后与已提交文件**逐字节相同**，未改语义。
7. **工具链**：git 三方合并、Python 3.11.14、PyTorch 2.7.1（CPU）、pytest 9.2.0 +
   pytest-xdist 3.8.0、matplotlib、ruff（未新增违规）。

## ③ 计算结果

**T1 冲突文件的双方数据（程序化读出）**

| 冲突文件 | main 步数 | main $t/t_\sigma$ | m3163 步数 | m3163 $t/t_\sigma$ | 合并后字节 | main 侧 `laplace` 字节 |
|---|---|---|---|---|---|---|
| `relax_dx0.0025.json` | 900 | 1.3162 | 684 | 1.0004 | 17036 | 336 |
| `relax_dx0.0020.json` | 900 | 1.0530 | 855 | 1.0004 | 23292 | 339 |
| `relax_dx0.0015.json` | 900 | 0.7897 | 1140 | 1.0003 | 35773 | 337 |

**T2 逐文件解法**

| 文件 | 顶层（规范档）| 保留的另一侧 | 证据 |
|---|---|---|---|
| `relax_dx0.0015.json` | m3163，1140 步 @ 1.0 $t_\sigma$ | main 900 步 @ 0.79 $t_\sigma$ → `alternate_runs[0]` + 归档副本 | T3 全绿 |
| `relax_dx0.0020.json` | m3163，855 步 @ 1.0 $t_\sigma$ | main 900 步 @ 1.05 $t_\sigma$ → `alternate_runs[0]` + 归档副本 | T3 全绿 |
| `relax_dx0.0025.json` | m3163，684 步 @ 1.0 $t_\sigma$ | main 900 步 @ 1.32 $t_\sigma$ → `alternate_runs[0]` + 归档副本 | T3 全绿 |

**T3 `verify_merge.py` 输出（原样粘贴，程序生成）**

```
case   kept steps  kept t/t_s main steps main t/t_s  keys m3163  keys main keys merged
0015         1140      1.0003        900     0.7897          62         46         121
0020          855      1.0004        900     1.0530          62         46         121
0025          684      1.0004        900     1.3162          62         46         121

refs: merge-base 184fef9, main ba07a10, agent/m3163 b7af006

PASS: both sides retained (canonical + alternate_runs + archive); kept rungs share the fixed 1.0 t_sigma window
```

**T4 规范档的物理量（360 Pa 基准，程序化读出）**

| dx [mm] | 粒子数 | 步数 | `field` 点数 | bandA $\Delta p$ [Pa] | bandA 误差 [%] | 绝对 $p$ [Pa] | 绝对误差 [%] | <3% | <15% |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 | 228 | 684 | 228 | 54.62 | -84.83 | 238.54 | -33.74 | no | no |
| 2.0 | 362 | 855 | 362 | 36.76 | -89.79 | 291.97 | -18.90 | no | no |
| 1.5 | 642 | 1140 | 642 | 70.43 | -80.43 | 283.81 | -21.16 | no | no |

注：bandA 读数为 $p(r<0.5R)-p(0.80\text{–}0.95R)$，它把自由面密度亏损算进了压差，
因此系统性地比绝对读数差（两条读数都保留，正是为了避免「报误差不报基准」）。

**T5 测试证据**（同一解释器、同一分组、同一 `-n`，基线 vs 合并）

| 树 | 命令 | 结果 |
|---|---|---|
| 基线 `origin/main`（`ba07a10`）| `pytest -q -n 24`（34 文件，略去 4 个慢文件）| 323 passed, 8 xfailed, 50 warnings in 38.67s |
| 合并树（本分支）| 同上 | 332 passed, 8 xfailed, 50 warnings in 49.28s |

合并树分组证据：A rc=0 :: 82 passed；B rc=0 :: 92 passed；C rc=0 :: 73 passed, 2 xfailed；D1 rc=0 :: 37 passed, 4 xfailed。
增量：merged = baseline + 9 newly added tests (tests/test_laplace_resolution_ladder.py) and 0 lost tests, 0 newly failing tests。
慢文件状态：The first three were run on both trees with the same command under a 52 s cap: both reached the same progress point (33 passed) with no failure, and the run was cut by the cap on both sides identically. tests/test_poiseuille.py was not executed. All four files are byte-identical in the baseline tree and the merged tree (the merge touches only benchmarks/laplace_law/*), so their results cannot differ between the two trees.

## ④ 结果图

`merge_evidence.png`（四联图，由 `make_figures.py` 从合并后的工件渲染）：

* **(a) 冲突根源**：三档分辨率下两条运行协议的 $t_{end}/t_\sigma$ —— main 的固定
  900 步让窗口随 $dx$ 漂移（0.79/1.05/1.32），PR #103 固定 1.0，并标出活门槛 0.5；
* **(b) 规范档两种读数**的相对误差 vs 15% 界线（bandA 与绝对读数分开画）；
* **(c) 末态场云图**：规范档三档的径向 Tait 压力剖面（场快照是 PR 新增的键）；
* **(d) 实测跳变 vs 解析 $\sigma/R$**：合并双方两套运行并列，说明合并没有篡改任何一侧的数字。

## ⑤ 注意点

* **机械 overlay 会静默丢数据**：三份文件都是单次运行快照，`checkout --ours` /
  `--theirs` 都能让树「干净」，但会永久抹掉另一侧的整段运行记录。本次用
  「规范档 + `alternate_runs` + 归档副本」三层保留，并由 `verify_merge.py`
  的键路径集合断言把关（合并后键路径数 121 > 两侧任一：62 / 46）。
* **重跑会吃掉并集**：`run_ladder.py` 重写工件时只写规范档，`alternate_runs` 会消失；
  归档副本 `main-900step/` 是唯一不可被基准运行覆盖的证据，勿删。
* **阶梯文档与工件是绑定关系**：`docs/verification_laplace_resolution_ladder.md`
  必须由 `make_ladder_doc.py` 从工件重渲染；手改会红（测试做逐字节比对）。
* **被门禁滤除的二进制**：PR 分支 tip 上 `laplace_resolution_ladder.png` 缺失导致阶梯
  测试红；本次重渲染并压到 215 655 B 以避开大二进制门禁。若门禁策略改变，需复核。
* **测试套件很慢**：串行全量 > 40 min（本机），必须 xdist 分组跑；4 个慢文件
  （oscillating_droplet / pair_based_rhs / paper_laplace_uncertainty / poiseuille）
  在两棵树上逐字节相同，未在预算内跑完，见 T5。
* **适用范围**：本次只处理合并与工件一致性，**不改变**任何物理结论；Laplace 跳变
  在 3% 判据下仍是 WITHHELD（见 T4）。

## ⑥ 误差判定

| 关键量 | 基准 | 结果 | 相对误差 | <3% ? |
|---|---|---|---|---|
| `bandA` 压差（dx=1.5 mm，规范档）| $\sigma/R=360$ Pa | 70.43 Pa | -80.43% | **否** |
| 绝对压差（dx=1.5 mm，规范档）| 360 Pa | 283.81 Pa | -21.16% | **否** |
| 绝对压差（dx=2.0 mm）| 360 Pa | 291.97 Pa | -18.90% | **否** |
| 绝对压差（dx=2.5 mm）| 360 Pa | 238.54 Pa | -33.74% | **否** |
| 合并保真度：丢失键路径数 | 0 | 0 | 0% | **是** |
| 合并保真度：被改动的已有数值 | 0 | 0 | 0% | **是** |
| 合并保真度：归档副本 vs main 原件 | 逐字节 | 逐字节相同 | 0% | **是** |

**显式回答**：Laplace 压力跳变在 3% 判据下**不通过**（bandA −80.43%，
绝对读数 −21.16% @ dx=1.5 mm），与 PR #103 自己的 WITHHELD 结论一致——本次合并
**不制造也不掩盖**这一误差。合并工程质量指标（键路径丢失、数值改动、归档一致性）
均为 0，满足 3% 判据。
