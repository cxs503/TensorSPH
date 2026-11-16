# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 3-D 臂首个可判定判决（回收 cj3887/cj3888）

TensorSPH · 分支 `agent/m22606` · mission 22606 · 锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`

**结论（honest FAIL）：3-D 臂第一次拿到了覆盖实验峰时刻的完整记录，判决为 FAIL ——箱前水位峰值与到达时刻都远在 10 % 容差之外。本单不解锁「达标」，它解锁的是「这条判据终于可以被判」：此前 3-D 臂的最好状态是 `WITHHELD`（没有一条跑到峰后的记录）。**

本文全部数字由 `benchmarks/kleefsman_obstacle_3d/recovered_verdict.py` 从`benchmarks/kleefsman_obstacle_3d/results/recovered/*.csv` 与实验锚点程序化注入（零手抄），复现：`python benchmarks/kleefsman_obstacle_3d/recovered_verdict.py`。

## ① 案例与判据

Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation method for wave impact problems*, **J. Comput. Phys. 206:363-393**, DOI 10.1016/j.jcp.2004.12.007：水槽 3.22 x 1.00 x 1.00 m，左端 1.228 x 0.550 m 水柱瞬时溃决，沿干底推进撞击 x = 2.476 m 处 0.161 x 0.161 x 0.403 m 方箱；方箱只挡 40 % 宽度，60 % 的水从两侧旁通。H2 = 箱前水位探针。

实验锚点（Fig. 19 右 panel 数字化，m20884）：**峰值 0.370 m @ t = 1.7027 s；到达时刻（0.02 m 阈值）0.3591 s**。判据：峰值与到达时刻误差 < 10 %。

## ② 怎么做的

1. **回收**：m22082 递交的三条合法 rung（合法性由 `validate_case` 判：dx <= 0.0537，方箱每向至少 3 格）中，cj3887（dx = 0.05）与 cj3888（dx = 0.04）在队列里已是 `rc = 0`，但从未被取回。本单 fetch 两个作业，把 `results/<tag>_time_series.csv` 与 `_meta.json` 原样落到 `benchmarks/kleefsman_obstacle_3d/results/recovered/`（未改一个数字）。
2. **判定规则（本单新增的可判定性口径）**：一条记录是否「可判」不取决于是否跑到 `t_end = 2.6 s`，而取决于它是否覆盖所测事件 —— 峰值要求 `t_last >= 1.7027 s`（否则是截断窗口的极大值，标 `WITHHELD`，绝不当作峰值）；到达时刻要求穿越点在记录内。任何一条已判指标超出 10 % 即 FAIL，不允许靠「去掉不达标的那一项」翻盘。
3. **旁通计数**作为几何差别的直接测量（2-D 模型恒为 0）。

## ③ 结果

| rung | 作业/节点 | 样本数 | t_last [s] | 覆盖实验峰 | 峰值 [m] | 峰误差 | 到达 [s] | 到达误差 | 判决 |
|---|---|---|---|---|---|---|---|---|---|
| dx=0.05 | cj3887/n315 rc=0 | 468 | 2.049 | 是 | 0.1202 @ 1.588 s | 67.5 % | 0.5028 | 40.0 % | FAIL |
| dx=0.04 | cj3888/n315 rc=0 | 327 | 1.144 | 否 | 0.0933 @ 1.007 s | —（WITHHELD） | 0.4678 | 30.3 % | FAIL |

辅助读数（与判据无关，仅用于归因）：

| rung | 前锋越过箱面 x=2.476 m [s] | 最大旁通粒子数 | `height_operator` 最大值 [m] |
|---|---|---|---|
| dx=0.05 | 0.481 | 518 | 0.0000 |
| dx=0.04 | 0.443 | 628 | 0.0000 |

![H2 3-D vs experiment](../benchmarks/kleefsman_obstacle_3d/results/recovered/fig_h2_3d_vs_experiment.png)

![decided error bars](../benchmarks/kleefsman_obstacle_3d/results/recovered/fig_error_3d.png)

## ④ 误差判定（逐项，显式回答）

* **dx=0.05**：峰值 0.1202 m vs 0.370 m -> 67.5 %；到达 0.5028 s vs 0.3591 s -> 40.0 %。判决 **FAIL**。
* **dx=0.04**：峰值 0.0933 m vs 0.370 m -> —（记录止于 t=1.144 s，未覆盖 1.7027 s，故 WITHHELD）；到达 0.4678 s vs 0.3591 s -> 30.3 %。判决 **FAIL**。

* **是否 < 10 %？否。是否 < 3 %？否。** 总判决 `FAIL`（理由：at least one decided metric is outside the 10 % band）。这是 honest FAIL，不得写成为达标；本轮唯一的状态变化是 `WITHHELD -> FAIL 可判定`。

## ⑤ 注意点 / 局限（如实）

1. **两条记录都没跑到 `t_end = 2.6 s`**（dx=0.05 止于 2.049 s，dx=0.04 止于 1.144 s，`_meta.json` 的 `complete` 均为 `false`），仍受那个间歇性 glibc abort 影响。因此它们**不能**进 `ladder.py` 作为收敛阶梯的一级（该模块按 `t_last >= t_end` 判完整性）；本单的判决只用「覆盖实验峰即判」这一更弱但仍严格的口径，并把 dx=0.04 的峰值如实标 WITHHELD。
2. **第三条 rung（cj3889, dx=0.03）缺失**：收敛阶梯与 `f_inf` 外推仍然缺席，「3-D 解是否随网格收敛到实验值」本单**没有回答**。
3. **发现的记录缺陷（未修，如实入账）**：两条记录的 `height_operator_m` 列恒为 0.000000（468 + 327 条样本全零），而同一采样点 `slab_height_m` 非零；隔离单测 `peak_height()` 在同一形状张量上返回正确值（水库内站位 0.525 m）。即「运行中峰值算子落盘恒 0」尚未定位到行号，本单只把它机器化（`tests/test_kleefsman_3d_recovered_verdict.py` 的 xfail 护栏，修复后自动翻绿），不作猜测性结论。判据用 `slab` 算子（冻结协议主指标），判决不受它影响。
4. **站位仍是代码约定**：H2 的 x 在论文 Fig. 17 里是栅格照片、坐标不可机器读取，沿用「箱面 - 0.05 m」（2.426 m）并标为假设（m21586 已给 2.228 / 2.426 m 包络）。
5. **读数不是网格无关的**：dx=0.05 时孔径 0.08 m 只含 1.6 个粒子列，slab 按单粒子量子化（每粒子约 0.0039 m），所以 0.05 与 0.04 的差别里有相当一部分是量化噪声而非物理；这也是必须补 dx=0.03 的理由。

## ⑥ 产物

* `benchmarks/kleefsman_obstacle_3d/recovered_verdict.py`（判据器：可判定性口径 + 判决 + 图 + 本文）
* `benchmarks/kleefsman_obstacle_3d/results/recovered/verdict.json`、`benchmarks/kleefsman_obstacle_3d/results/recovered/fig_h2_3d_vs_experiment.png`、`benchmarks/kleefsman_obstacle_3d/results/recovered/fig_error_3d.png`
* `benchmarks/kleefsman_obstacle_3d/results/recovered/`：cj3887/cj3888 的原始时程与 `_meta.json`（作业回收证据，未改数字）
* `tests/test_kleefsman_3d_recovered_verdict.py`（可判定性规则、判决不得被提升、记录不得跨协议、已知缺陷的 xfail 护栏）
