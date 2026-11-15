# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 3-D 旁通臂的可判定化：网格阶梯工具

TensorSPH · branch `agent/m21869` · mission 21869 · 锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`
（算例池·带障碍物溃坝冲击，Kleefsman 系列）

**本单状态：能力已交付并验证；3-D 生产判决仍为 `WITHHELD`。** 三个 3-D 作业在本班次
内未跑完（见 §③.2），因此本文**不声明**峰值/到达达标，也**不**重复此前"残差归因于旁通"
的断言。本文交付的是把那句断言变成**可判定**的工具。

---

## ① 案例与本次要回答的问题

Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005), *A Volume-of-Fluid based
simulation method for wave impact problems*, **J. Comput. Phys. 206:363–393**,
DOI 10.1016/j.jcp.2004.12.007。左端水柱瞬时溃决、沿干底推进、冲击方箱；H2 波高计
在箱前，判据 = **水位时程峰值与到达时刻 vs 实验，容差 < 10 %**。

| 量 | 值 | 来源 |
|---|---|---|
| 槽 长×高×**宽** | 3.22 × 1.00 × **1.00** m | 原文 apparatus 图 |
| 水柱 长×高 | 1.228 × 0.550 m（跨满宽度） | 同上 |
| 方箱 长×高×**宽** | 0.161 × 0.161 × **0.403** m | 同上 |
| 箱上游面 | x = 1.228 + 1.248 = 2.476 m（与 3.22 − 0.744 自洽，残差 0） | `reference.json` |
| H2 峰值 / 时刻 | 0.370 m @ 1.7027 s | 原文 Fig. 19 右 panel 数字化 |
| H2 到达（阈值 0.02 m） | 0.3591 s | 同上 |

**问题**：2-D 模型里 0.161 m 方箱跨满 1.00 m 槽宽（100 % 阻挡），实验只挡 40 %、
60 % 的水从侧面旁通。已合并的 2-D 判决峰值超实验 30.1 %，此前把残差**归因**于这一
几何差——但归因一直没有测量。m21728 造出了 3-D 臂（真实 0.403 m 箱），却只有单网格，
**一个网格给一个数、没有误差棒**，"3-D 更接近"和"3-D 更远"同样不可证伪。

## ② 怎么做的

新增 `benchmarks/kleefsman_obstacle_3d/ladder.py`（只做分析，不碰求解器）：

| 部件 | 做法 |
|---|---|
| 输入 | 读已提交的 `results/<tag>.json` + `<tag>_time_series.csv`（run.py 的产物），不重算 |
| 协议指纹 | `t_end / sound_speed / wall_layers / obstacle_z / station_offset / gauge_aperture / arrival_threshold` 七项必须逐项一致，否则整条阶梯判 `INCOMPARABLE`，**不做外推**（混合 t_end 或混合 c0 的"阶梯"没有意义） |
| 阶数估计 | `f(dx) = f_inf + C·dx^p` 三点拟合：对固定 p 做 (f_inf, C) 的线性最小二乘，用黄金分割在 p∈[0.1, 8] 上最小化残差。这是 Richardson 外推对**非均匀加密比**的推广（0.08/0.04/0.03 的比值是 2 与 4/3，标准 r=2 公式不适用） |
| 两点阶梯 | 直接返回 `UNDERDETERMINED` + 区间，**拒绝编造阶数**（有测试锁死） |
| 复用 | `moving_average / peak_and_time / observed_order / richardson / gci_fine` 直接从 `benchmarks/kleefsman_obstacle/convergence_analysis.py` 导入 —— 2-D 与 3-D 两臂由**同一份**数值代码打分，不各写一套 |
| 判决 | `converges_toward()`：逐网格到基准的距离是否**单调收缩**，且外推值是否比最粗网格更接近基准 → `BLOCKAGE_HYPOTHESIS_SUPPORTED / NOT_SUPPORTED / NOT_DECIDABLE` |

工具链：
```
python benchmarks/kleefsman_obstacle_3d/run.py --dx 0.04 --t-end 2.6 --tag dx0.04
python benchmarks/kleefsman_obstacle_3d/ladder.py            # -> results/ladder.json
python benchmarks/kleefsman_obstacle_3d/make_ladder_doc.py   # -> 图
```

### 2.1 工具本身的验证（本单真正的、已取得的实测）

估计器必须先在**已知阶数**的合成数据上复现阶数，否则它给出的 p 只是拟合噪声：

| 设定阶数 p | 反演 p̂ | 外推 f_inf | |p̂ − p| |
|---|---|---|---|
| 1.0 | 1.0000 | 0.300000 | 5.82e-13 |
| 1.5 | 1.5000 | 0.300000 | 4.38e-12 |
| 2.0 | 2.0000 | 0.300000 | 1.61e-11 |
| 3.0 | 3.0000 | 0.300000 | 1.82e-10 |
| 4.0 | 4.0000 | 0.300000 | 3.97e-09 |

非均匀阶梯 dx = 0.08/0.04/0.03 上，五个设定阶数全部反演成功，最大偏差 **3.97e-09**。
这条验证是**可证伪**的：若拟合写错（例如把最粗网格当最细、或把 r 当常数），
`fig_tool_verification.png` 的点会全部偏离 y = x。

## ③ 计算结果

### 3.1 已合并 2-D 家族的判决（引用已提交产物，非本单重算）

`reference.json → convergence_t30.primary_station`：三点阶梯收敛（p̂ = 3.431，
p_eff = 2.0），Richardson 峰值 0.4818 m，
对实验的连续极限相对误差 30.21 %，
ASME V&V 20 判据 `MODEL_ERROR_EXCEEDS_U_V`；到达误差（最细网格）25.78 %，
且 `same_event_as_experiment = False`。

### 3.2 3-D 臂：`WITHHELD`（三个作业本班次未回收）

| 作业号 | 网格 | t_end | 本单核对时的状态 |
|---|---|---|---|
| cj3661 | dx = 0.04 m, 8 核 | 2.6 s | running（n315，已 25 min） |
| cj3662 | dx = 0.03 m, 8 核 | 2.6 s | running（n330） |
| cj3758 | dx = 0.08 m, 8 核（本单递交） | 2.6 s | queued（队首前 80 单） |

**判决：`WITHHELD`。** 3-D 的峰值/到达对实验误差**尚未取得**，本单因此既不声明达标，
也不声明旁通可解释残差。回收后运行一节两条命令即可得到机器判决，无需改代码。

### 3.3 结果图

* `results/ladder_figs/fig_tool_verification.png`：估计器验证（反演阶数 vs 设定阶数）。
  **这是本单唯一有实测支撑的图**。
* 未出 3-D 时程叠加图与峰值-网格阶梯图：`make_ladder_doc.py` 里两条路径已就绪，
  但输入 CSV 尚未回收。**不用占位图冒充结果**（`fig_overlay` 在无数据时返回 None，
  脚本打印 `overlay: null`，不是画空图）。

## ④ 验证（本单实际跑过的检查）

```
$ python -m pytest tests/test_kleefsman_obstacle_3d_ladder.py -q
13 passed

$ python -m pytest tests/test_kleefsman_obstacle_3d.py -q
26 passed
```

13 条断言里，关键几条是**不变量**而非"能跑"：

* 已知阶数的二阶/一阶合成数据必须被精确反演（f_inf 误差 < 1e-6）；
* **两个网格绝不产生阶数**（`p_observed` 必须缺席）；
* 换 `t_end` 或换 `c0` 必须把阶梯判成 `INCOMPARABLE`，且 `peak_ladder` 键**不得出现**；
* 单调收缩 → `SUPPORTED`；单调远离 → `NOT_SUPPORTED`；两点 → `NOT_DECIDABLE`；
* `analyse` 必须把 CSV 里的 `n_bypass`（旁通粒子数）带出来——几何差要成为**被报告的数**。

## ⑤ 注意点 / 局限

1. **非均匀阶梯**：标准 Richardson 的 r 公式在这里不成立，本文用的是三点幂律拟合；
   三点恰好定三个未知量，残差天然很小，**不要**把"残差小"当成"阶梯可信"，
   可信度来自 §2.1 的合成验证与协议一致，不来自残差。
2. **dx = 0.08 不是有效分辨率**：`validate_case` 要求箱体至少 3 个格子（dx ≤ 0.0537 m），
   dx = 0.08 时代码允许但箱体只有 2 个竖直格，读数不可用于判决，只能作为阶梯最粗点。
3. **方箱横向位置是假设**：原文未用文字给出箱在 1.00 m 宽度内的位置，默认贴一侧槽壁
   （`obstacle_z = 0`，多数 SPH 复现的做法）。居中放置是两个 0.2985 m 间隙，是另一个算例。
4. **槽壁自由滑移**（`wall_friction = 0`），会低估壁面摩阻对前锋速度的影响。
5. **t_end = 2.6 s 覆盖实验峰时刻 1.70 s**，但 2-D 的 dx=0.01 记录只到 2.6 s 而 dx=0.04/0.02
   到 3.0 s；跨臂比较一律按协议指纹拒绝，不做"凑齐"。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 量 | 基准 | 本单取得的值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| 3-D 峰值（H2, slab 冻结协议） | 0.370 m | **未取得**（WITHHELD） | 不可计算 | **未判定**（数据未回收，不猜测） |
| 3-D 到达时刻（阈值 0.02 m） | 0.3591 s | **未取得**（WITHHELD） | 不可计算 | **未判定**（同上） |
| 阶梯估计器：反演阶数 | 合成已知 p | p̂ 见表 | 最大 3.97e-09 | **是** |
| 2-D 峰值（已合并，引用） | 0.370 m | 0.4818 m（外推） | 30.21 % | **否** |
| 2-D 到达（已合并，引用） | 0.3591 s | — | 25.78 % | **否** |

**结论**：本单把"3-D 旁通能否解释 2-D 的 +30 % 峰值残差"从**断言**推进到**可判定**，
但**没有**给出肯定或否定的答案——三个 3-D 作业未在班次内回收。这是 honest FAIL 意义上的
未完成项，不得粉饰为完成。

---

*本文所有数字由脚本注入（`make_ladder_doc.py` / `reference.json`），无手抄。*
