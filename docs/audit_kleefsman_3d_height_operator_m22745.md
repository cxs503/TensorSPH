# 3-D Kleefsman height 算子记录可采性审计（m22745）

锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`（Kleefsman et al. 2005, J. Comput. Phys. 206:363, doi:10.1016/j.jcp.2004.12.007）。

## ① 案例与问题

m22606 回收的两条 3-D 生产记录（cj3887/dx=0.05、cj3888/dx=0.04）里 `height_operator_m` 列**全为零**，而同一批样本的 `slab_height_m` 列报告了成米级水体。m22606 把这一点记成「未修缺陷」。本单只做一件事：把这件事变成可判定命题。

关键事实：两个算子共用同一个筛子 —— `slab_height()` 与 `peak_height()` 都调用 `_station_mask()`，即**同一粒子集合、同一时刻**的两个函数。`slab` 是 `n·dx³/(aperture·width)`，`height` 是同一集合里 `y` 的最大值。因此 `slab` 为 `height` 提供下界证人：水柱有 n 个粒子，就不可能所有粒子都躺在 `y=0` 平面上——除非 n 不超过该底板面能容纳的柱位数。

## ② 方法

1. `Kleefsman3DRun.operator_consistency()`（本单新增于 `src/tensorsph/kleefsman_obstacle_3d.py`）：逐样本对比两个算子，把 `slab > 2·dx` 却 `height <= 0` 的样本记为 **impossible**。门槛 `2·dx` 的理由是宽容真实的退化：孔径 0.08 m 在 dx=0.10 时窄于一个粒子间距，band 里只剩一个贴底粒子，`slab <= dx` 且 `height = 0` 是真的（本单复现）。
2. 底板容量上界 `floor_columns(dx) = (aperture/dx + 1)·(width/dx + 1)`：同一 (x, z) 柱位不能叠两个粒子，所以它能同处 `y=0` 的粒子数有硬上界，换算成米就是「底板平面最多能解释多少水」。
3. 与 impossible 样本所需水量比较：只要前者小于后者，该列就不是这套算子在同一粒子集合上的输出。

## ③ 结果（程序化注入，零手抄）

| 记录 | dx [m] | 样本 | 末时刻 [s] | slab 峰值 [m] | height 峰值 [m] | 有湿样本 | impossible 样本 | impossible 样本水位 [m] | 底板最多解释 [m] | 清单带版本号 | 可采 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cj3887/dx0.05 | 0.05 | 468 | 2.049 | 0.1202 | 0.0000 | 354 | 101 | 0.1202 | 0.1047 | **否** | **否** |
| cj3888/dx0.04 | 0.04 | 327 | 1.144 | 0.0933 | 0.0000 | 194 | 55 | 0.0933 | 0.0655 | **否** | **否** |

![height audit](results/height_audit/fig_height_audit.png)

## ④ 判定（逐项，显式回答）

* cj3887：`height` 峰值 0.0000 m，`slab` 峰值 0.1202 m（≈31.0 个粒子）。impossible 样本 101 个，最坏样本水位 0.1202 m，需要 31.0 个粒子同时位于 `y=0`，而 band 内底板柱位只有 27 个（底板平面最多解释 0.1047 m 的水；即使按最宽松的底板排布 30 个柱位算，仍然不够）→ 该列为 **不可采**。
* cj3888：`height` 峰值 0.0000 m，`slab` 峰值 0.0933 m（≈47.0 个粒子）。impossible 样本 55 个，最坏样本水位 0.0933 m，需要 47.0 个粒子同时位于 `y=0`，而 band 内底板柱位只有 33 个（底板平面最多解释 0.0655 m 的水；即使按最宽松的底板排布 36 个柱位算，仍然不够）→ 该列为 **不可采**。

**整体读法（比单样本更强）：** 两条记录合计 548 个有湿样本，其中 548 个的 `height` 恰为 0.000000。只要 band 里有**任意一个**粒子离底，该列的 6 位小数就会非零；要 354+194 个样本全部保持恰零，只能是没有粒子被这个算子筛到、或者这列根本不是它算的。叠加下面的单样本几何不可能性，结论是**记录缺陷**，不是物理读数。

**与 Kleefsman2005 实验基准（峰值 0.370 m @ 1.7027 s、到达 0.3591 s）的关系：**本单不改判据、不改判决。m22606 的 `FAIL` 用的是 `slab` 算子（冻结协议主指标），不受本单影响；被否定的是「`height` 列可作为读数」这一前提。**本单不产生新的精度结论，也不存在 <3% 或 <10% 的达标声明。**

## ⑤ 注意点 / 局限（如实）

1. **没有定位到写坏那一列的代码行**。本单证明的是「已提交列不是现行算子的输出」（矛盾性证明），不是「哪一行写错了」。逆向尝试：现行 `run.py` + 现行算子在同几何下（本机 dx=0.10 诊断、集群 dx=0.05 复算）都**不会**输出全零列；而两条记录 `_meta.json` 里**没有任何代码版本号**，所以「当时跑的是哪一版」事后不可判定。这就是本单把版本号写进清单的理由。
2. **impossible 门槛是物理性的，不是数学恒等式**：理论上 n 个粒子全在 `y=0` 不违反 `slab>0`，只是这需要 n 超过底板柱位数（见上表比值）。门槛取 `2·dx` 已放过真实退化（单粒子贴底）。
3. 判决仍受 m22606 已列局限约束：两条记录都没跑到 `t_end = 2.6 s`；缺 dx=0.03 一级；站位仍是代码约定（箱面 - 0.05 m）。

## ⑥ 产物

* `src/tensorsph/kleefsman_obstacle_3d.py`：`Kleefsman3DRun.operator_consistency()`
* `benchmarks/kleefsman_obstacle_3d/run.py`：清单写入 `operator_consistency` 与 `code_revision`（git + 算子源码 sha1）
* `benchmarks/kleefsman_obstacle_3d/height_operator_audit.py`、`results/height_audit/audit.json`、`results/height_audit/fig_height_audit.png`
* `tests/test_kleefsman_3d_height_operator_audit.py`
