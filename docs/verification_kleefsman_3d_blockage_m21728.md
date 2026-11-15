# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — restoring the third dimension (blockage 40 % vs 100 %)

TensorSPH · branch `agent/m21728` · mission 21728 · anchor `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`

Status of this note: **the capability is delivered and tested; the production verdict is
`WITHHELD`**, because the two scored runs were submitted to the cluster and were still
queued when the shift ended (see §③ and §⑤.1). Nothing below is a completion claim for
the acceptance metric.

---

## ① 案例与本次要回答的问题

Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005), *A Volume-of-Fluid based
simulation method for wave impact problems*, **J. Comput. Phys. 206:363–393**,
DOI 10.1016/j.jcp.2004.12.007。水槽左端 1.228 × 0.550 m 水柱瞬时溃决，沿干底推进，
撞击 **x = 2.476 m** 处的方箱；H2 波高计在箱前，判据为**水位时程峰值与到达时刻 vs 实验，容差 < 10 %**。

| 量 | 实验值 | 源 |
|---|---|---|
| 槽长 / 高 / 宽 | 3.22 / 1.00 / **1.00** m | 原文 apparatus 图 |
| 水柱长 / 高 | 1.228 / 0.550 m（跨满宽度） | 同上 |
| 方箱 长×高×**宽** | 0.161 × 0.161 × **0.403** m | 同上 |
| 箱上游面 | x = 1.228 + 1.248 = 2.476 m（与 3.22 − 0.744 自洽，残差 0） | `benchmarks/kleefsman_obstacle/reference.json` |
| H2 峰值 / 时刻 | 0.370 m @ 1.7027 s | 原文 Fig. 19 右 panel 数字化（m20884） |
| H2 到达（阈值 0.02 m） | 0.3591 s | 同上 |

**本单要回答、而此前所有轮次都没有测过的一件事**：已合并的 2-D 算例把 0.161 m 方箱
当作全宽障碍（1.00 m 槽里 100 % 阻挡），实验只挡 **40 %** 宽度、**60 % 的水从侧面旁通**。
m21324 的 2.6 s 长窗复算已经把峰值判成 FAIL（+30.1 %，slab 算子），m21586 把这一残差
**归因**于二维全宽阻挡——但归因是断言，不是测量：

* m21324 / m21586 的每一份 2-D 判决里，峰值误差 15–52 %、到达误差 7.8–34.8 %；
* 这一残差从未与「旁通」这一几何差做过对照实验。

本单交付的正是这个对照实验的能力：**同一求解器、同一量测协议、同一基准，把水槽的第三个
维度还原，方箱给回真实的 0.403 m 宽**。

## ② 怎么做的

### 2.1 新增模块

`src/tensorsph/kleefsman_obstacle_3d.py`（复用 `dam_break_3d` 的 3-D 三次样条核、
`rhs_3d`、`apply_boundary_3d`、`wall_lattice_3d`、Verlet 邻居表）：

| 部件 | 做法 |
|---|---|
| 域 | x ∈ [0, 3.22]，y ∈ [0, 1.00]，z ∈ [0, 1.00] m（y 为竖直方向，与 `dam_break_3d` 一致） |
| 水柱 | x ∈ [0, 1.228]，y ∈ [0, 0.550]，**z 跨满 1.00 m**（与实验一致），静水压初始化 |
| 方箱 | x ∈ [2.476, 2.637]，y ∈ [0, 0.161]，z ∈ [0, 0.403]（默认贴一侧槽壁，见 §⑤.2）；**实心粒子块**，保证每个面上的压力镜像都有定义 |
| 非穿透 | 固体壁粒子之外再加一道硬约束 `clamp_out_of_obstacle`：落在箱体内的粒子沿最近面推出、法向速度归零，并**计数**（`obstacle_clamps`）。这道约束把「不穿透」变成不变量而不是期望——穿透的粒子会让箱前水位计读到箱体内部的粒子 |
| 量测 | 与 2-D 冻结协议**逐项同口径**：孔径 0.08 m、站位 0.05 m×箱面上游（x = 2.426）、到达阈值 0.02 m。3-D 下 slab 是体积计：`h = Σ(m/ρ0) / (孔径 × 阻挡宽度)`；2-D 下该式恰好退化为冻结协议的 `A/孔径` |
| 第二算子 | `height` = slab 内最高湿点（对应 2-D 的 `height` 算子），用于给出算子包络而不是挑一个好看的 |
| 旁通 | `bypass_count` = 越过箱体下游面、且 z 落在箱体跨度**之外**的流体粒子数。2-D 里这个数恒为 0——这就是几何差的直接测量 |

几何自检 `validate_case` 会拒绝任何不是该装置的量（槽长 3.22 / 高 1.00 / 宽 1.00、
柱 1.228 / 0.550 / 跨满宽度、箱 2.476 / 0.161 / 0.403），并拒绝箱体网格少于 3 个格子
（dx > 0.0537 m 时直接报错，避免「用不解析的箱子去比实验」）。

### 2.2 判据纪律（沿用 m21586，不放宽）

* 峰值与到达**两项都要 < 10 %** 才算 PASS，不挑算子、不挑站位；
* **censoring**：若记录的极大值落在最后两个采样点、且记录在实验峰时刻（1.703 s）之前结束，
  则该峰是截断峰，**即使数值恰好命中也不许判 PASS**（`score()` 里机器化，并有单元测试）；
* 判决文字由脚本给出，本文不手抄任何数字。

### 2.3 工具链

```
python benchmarks/kleefsman_obstacle_3d/run.py --dx 0.04 --t-end 2.6 \
       --wall-layers 3 --output-every 5 --tag dx0.04
python benchmarks/kleefsman_obstacle_3d/make_doc.py     # 图 + 表，程序化注入
```
`run.py` 写 `results/<tag>_time_series.csv`（t, slab, height, front_x, n_bypass）与
`results/<tag>.json`（run 摘要 + 逐项误差 + 判决）。`make_doc.py` 只读已提交产物，
叠加实验数字化曲线与 2-D 对照曲线出图。

## ③ 计算结果

### 3.1 已收回的实测（轻量自检，dx = 0.05 m，wall_layers = 3）

| 量 | 值 |
|---|---|
| 流体 / 壁面粒子 | 5 280 / 10 160 |
| 步长 | dt = 8.775e-4 s（Courant 限制 ×0.9） |
| 已积分到 | t = 0.351 s（受本机 120 s 单命令上限截断） |
| 该时刻箱前水位 | slab = 0.000 m → **到达时刻 > 0.351 s** |
| 质量漂移 | 0.0（显式积分不改变质量） |
| 边界逃逸 / 箱体夹持 | 0 / 0 |

这一条本身就是有信息量的：2-D 家族（m21324 的 2.6 s 长窗）在 dx = 0.04 给到达 0.484 s、
dx = 0.01 给 0.4378 s，而实验是 0.3591 s。3-D 自检在 0.351 s 时箱前仍是干的，
与 2-D 家族同一量级，说明**几何与量测没有接错**（若把箱前站位或孔径弄错，这里会提前出水）。

### 3.2 提交但未回收的生产算例

| 作业号 | 网格 | t_end | 提交时刻 | 状态 |
|---|---|---|---|---|
| cj3661 | dx = 0.04 m, 8 核 | 2.6 s | 10:27 | 收工前仍在排队（等待 21.9 min，队列 77 个在飞） |
| cj3662 | dx = 0.03 m, 8 核 | 2.6 s | 10:27 | 收工前仍在排队 |

**判决：`WITHHELD`。** 峰值/到达对实验的误差**尚未取得**，因此本单不声明该项达标，
也不声明残差可归因于旁通。回收作业后执行下一节命令即可得到机器给出的 PASS/FAIL。

### 3.3 结果图

本单**未附结果图**，原因是 3-D 场量云图与时程对比曲线都依赖 §3.2 尚未回收的时程 CSV；
`benchmarks/kleefsman_obstacle_3d/make_doc.py` 已就绪（三张图：H2 时程 2-D/3-D/实验叠加、
逐 run 误差柱状图含 10 % 门限线、旁通粒子数时程），回收算例后一条命令即可生成，
不手抄任何数。**不用占位图冒充结果**。

## ④ 验证（本单实际跑过的检查）

```
$ python -m pytest tests/test_kleefsman_obstacle_3d.py -q
26 passed
```

26 条断言里，关键几条是**不变量**而非「能跑」：

* `test_the_clamp_empties_the_box_and_reports_how_many_it_moved`——夹持后箱体内粒子数为 0，
  且被推出的粒子**只**丢掉离开那个面的法向速度分量（另两个分量保留）；
* `test_the_clamp_is_idempotent`——夹持是幂等的（第二次调用移动 0 个粒子）；
* `test_the_slab_gauge_reports_metres_of_water_over_the_blocked_width`——3×4×8 个 dx³ 的水
  必须读出 `3·4·8·dx³/(0.08·0.403)` m，与解析值相对误差 < 1e-12；
* `test_the_slab_gauge_ignores_water_that_bypassed_the_box`——同样高度但落在**开放跨度**里的水
  读数为 0（否则「箱前水位」会把旁通的水算进去，判据失去意义）；
* `test_scoring_uses_ten_percent_and_does_not_cherry_pick`——峰值恰中而到达晚 30 % 必须 FAIL；
* `test_a_censored_peak_cannot_pass`——截断峰即使数值命中也不许 PASS；
* `test_a_short_three_dimensional_run_is_conservative_and_keeps_the_box_solid`（slow）——
  质量零漂移、边界逃逸 0、系列有限、前锋离开水库。

同目录的既有算例测试：`pytest tests/ -k kleefsman -m "not slow"` →
**264 passed, 5 failed**。5 条失败全部在 `tests/test_kleefsman_text_landmarks.py`，
原因是缺少未提交的产物 `outputs/kleefsman2005_text/quotes.json`（`FileNotFoundError`），
**与本单改动无关**（本单只新增文件，未改任何既有模块）。

另注：`pytest tests/` 整体收集有 3 个**先于本单存在**的收集错误
（`benchmarks/xval_sphinx_tvc/*` 与 `benchmarks/case_poiseuille/run.py` 之间以裸名 `run`
互相导入的模块名冲突），需要 `--ignore` 这 3 个文件才能收集；本单未触碰这些路径。

## ⑤ 注意点 / 局限（如实）

1. **生产算例未回收**（§3.2）。集群队列在收工时仍有 77 个在飞作业，两个 8 核算例排队 22 min
   未起跑。本单因此只交付能力与验证，不交付判决。**这是本单最主要的未完成项。**
2. **方箱的横向位置是假设**：原文只说箱宽 0.403 m，装置图（Fig. 2/17）是位图，箱体在 1.00 m
   槽宽内的横向坐标不可机器读取。本模块把它做成参数 `obstacle_z`，默认 `0.0`（贴一侧槽壁，
   单侧 0.597 m 旁通通道，多数 SPH 复现采用）。居中放置（两侧各 0.2985 m）只需一个关键字，
   但**两种放置的旁通拓扑不同**，判决前应当两种都跑。
3. **箱体宽度的网格离散**：0.403 m 在 dx = 0.04 下是 10 格 = 0.40 m（偏窄 0.7 %），
   dx = 0.03 下 13 格 = 0.39 m（偏窄 3.2 %）。水位计的**分母仍取实验值 0.403 m**
   （我们比的是实验），而 `run.json` 同时记录 `obstacle_width_on_lattice_m`，
   让这项离散化误差可见而不是被藏起来。
4. **壁面自由滑移**（`wall_friction = 0`，求解器默认）。实验壁面有边界层，这会让 3-D 前锋
   比实际略快——方向与「3-D 应当比 2-D 慢」一致，但对峰值的净影响未标定。
5. **只要一项不过就 FAIL** 的纪律不变；本单不因为「3-D 更接近实验」就放宽 10 %。
6. dx = 0.05 自检里瞬时最大速度曾达 33.8 m/s（2-D dx = 0.04 为 5.98 m/s），出现在水库塌落
   初期的局部飞溅，压缩率 0.163 与 2-D 的 0.172 同量级；自适应步长会自动压缩 dt。
   这一数值**未被当作结果使用**，但它是粗网格飞溅的已知噪声，回收生产算例时要一并检查。

## ⑥ 误差判定：关键量 vs 基准

| 关键量 | 基准（Kleefsman Fig. 19） | 本单结果 | 相对误差 | 是否 < 3 % |
|---|---|---|---|---|
| 箱前水位峰值 | 0.370 m | **未取得** | **未取得** | **否——无结果，不可声明** |
| 箱前水位到达时刻（0.02 m 阈值） | 0.3591 s | **未取得**（已知 > 0.351 s） | **未取得** | **否——无结果，不可声明** |

**显式回答「是否 < 3 %」：不能回答。** 本单没有产生可比的 3-D 峰值与到达时刻（算例在排队），
因此既不能声称 < 3 %，也不能声称 < 10 %。相对可说的是：3-D 自检在 t = 0.351 s 时箱前仍为干，
说明到达时刻与 2-D 家族（0.438–0.484 s）同一量级，而不是出现几何/单位接错的量级错误——
这只是自洽性证据，不是精度证据。

（10 % 是本账本条目的验收容差；< 3 % 是平台对论文级精度的通用追问。二者本单都无法回答。）