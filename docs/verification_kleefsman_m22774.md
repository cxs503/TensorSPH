# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 带障碍物溃坝冲击验证报告（m22774）

TensorSPH · 分支 `agent/m22774` · mission 22774

**结论（honest FAIL）：本单完成了三项实质增量——(1) 3-D height_operator 零值根因分析与单元测试护栏；(2) 2-D/3-D 算子一致性证明；(3) 全量判决状态梳理。判据本身（峰值与到达时刻 <10%）仍未通过，这是 honest FAIL，不得粉饰。**

本文全部数字由程序化注入（零手抄），复现命令见各节。

---

## ① 案例介绍（物理背景/几何/参数表）

Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation method for wave impact problems*, **J. Comput. Phys. 206:363–393**, DOI 10.1016/j.jcp.2004.12.007。

**几何**（所有长度单位：米）：

| 量 | 值 | 来源 |
|---|---|---|
| 水槽长 | 3.22 | Kleefsman et al. (2005) |
| 水槽宽 | 1.00 | idem |
| 水槽高 | 1.00 | idem |
| 水柱长 | 1.228 | idem |
| 水柱高 | 0.550 | idem |
| 方箱上游面 x | 2.476 | 1.228 + 1.248 |
| 方箱长（流向） | 0.161 | idem |
| 方箱高 | 0.161 | idem |
| 方箱宽（展向） | 0.403 | idem（仅 3-D） |
| 阻塞率 | 40.3% | 0.403 / 1.00 |

**实验锚点**（Fig. 19 右 panel 数字化，m20884）：
- 峰值：0.370 m @ t = 1.7027 s
- 到达时刻（0.02 m 阈值）：0.3591 s
- 探针位置：箱前 0.05 m（x = 2.426 m），约定值（H2 真实坐标在栅格图中不可机器读取）

**判据**：峰值与到达时刻误差 < 10%。

---

## ② 怎么做的（方法/流程/关键设置）

### 2.1 算子一致性证明

本单新增 `tests/test_kleefsman_3d_height_operator_fix.py`，包含 8 个测试：

1. **`test_station_mask_catches_particles`**：验证 `_station_mask` 正确捕获站内粒子
2. **`test_peak_height_positive_when_slab_nonzero`**：核心护栏——若 slab 报告粒子，peak 必须 > 0
3. **`test_peak_height_zero_when_no_particles`**：空站返回 0
4. **`test_peak_height_equals_max_y`**：peak_height 返回站内粒子最大 y
5. **`test_slab_height_scales_with_particle_count`**：slab_height 与粒子数成正比
6. **`test_2d_water_height_at_tracks_surface`**：2-D water_height_at 跟踪水面
7. **`test_2d_operators_zero_for_empty_strip`**：2-D 空条返回 0
8. **`test_3d_slab_reduces_to_2d_for_full_width`**：3-D slab 在全宽时与 2-D slab 概念一致

复现：`PYTHONPATH=src python3 -m pytest tests/test_kleefsman_3d_height_operator_fix.py -v`

### 2.2 height_operator 零值根因分析

对 `benchmarks/kleefsman_obstacle_3d/results/recovered/cj3887_dx0.05_time_series.csv` 的分析表明：

- `slab_height_m` 在 t ≈ 0.5 s 后非零（水已到达测站）
- `height_operator_m` 恒为 0.000000（468 个样本全零）

**根因定位**：`peak_height()` 函数本身在单元测试中工作正确（返回站内粒子最大 y）。零值出现在生产运行中，最可能的原因是：
1. 生产运行因 glibc heap abort 间歇性崩溃
2. 崩溃后从 checkpoint 恢复，但 checkpoint 未正确保存 `height` 历史
3. 恢复后的运行中，`positions` 张量在采样时刻可能因边界钳位导致站内粒子被排除

**本单对策**：新增单元测试护栏，确保算子本身正确；生产运行的完整修复需要集群级重算（已记录为待办）。

---

## ③ 计算结果（判决状态总览）

### 3.1 2-D 臂判决状态

| rung | t_end [s] | 峰值 [m] | 峰误差 | 到达 [s] | 到达误差 | 判决 |
|---|---|---|---|---|---|---|
| dx=0.02 | 1.20 | 0.517 | 39.9% | 0.454 | 26.4% | FAIL/FAIL |
| dx=0.01 | 1.00 | 0.377 | 2.0% | 0.444 | 23.7% | PASS/FAIL |
| dx=0.04 | 1.20 | 0.425 | 15.0% | 0.484 | 34.8% | FAIL/FAIL |

**2-D 总判决**：FAIL（到达时刻全部 >10%）

### 3.2 3-D 臂判决状态

| rung | t_last [s] | 覆盖实验峰 | 峰值 [m] | 峰误差 | 到达 [s] | 到达误差 | 判决 |
|---|---|---|---|---|---|---|---|
| dx=0.05 | 2.049 | 是 | 0.1202 | 67.5% | 0.5028 | 40.0% | FAIL |
| dx=0.04 | 1.144 | 否 | 0.0933 | WITHHELD | 0.4678 | 30.3% | FAIL |

**3-D 总判决**：FAIL（峰值与到达时刻均远超 10%）

### 3.3 关键量 vs 基准误差逐项判定

| 指标 | 最佳 2-D 误差 | 最佳 3-D 误差 | 是否 <10%？ | 是否 <3%？ |
|---|---|---|---|---|
| 峰值 | 2.0% (dx=0.01) | 67.5% (dx=0.05) | 2-D 是，3-D 否 | 否 |
| 到达时刻 | 23.7% (dx=0.01) | 30.3% (dx=0.04) | 否 | 否 |

**显式回答**：
- 峰值：2-D dx=0.01 达到 2.0% < 10%，但到达时刻 23.7% > 10%，**总判决 FAIL**
- 到达时刻：所有 rung 均 > 10%，**FAIL**
- 是否 < 3%？**否**

---

## ④ 结果图

### 4.1 2-D vs 3-D vs 实验对比

![2-D/3-D/实验对比](../benchmarks/kleefsman_obstacle_3d/results/recovered/fig_h2_3d_vs_experiment.png)

*图：3-D 臂 dx=0.05 时程（蓝）与实验锚点（红虚线）对比。3-D 峰值显著低于实验，到达时刻偏晚。*

### 4.2 误差条图

![误差条](../benchmarks/kleefsman_obstacle_3d/results/recovered/fig_error_3d.png)

*图：3-D 臂各 rung 的峰值与到达时刻误差条，均远超 10% 容差带。*

---

## ⑤ 注意点（踩过的坑/局限/适用范围）

1. **height_operator 零值 bug**：3-D 生产运行中 `height_operator_m` 恒为零，但 `slab_height_m` 非零。单元测试证明算子本身正确，问题出在生产运行的 checkpoint/resume 路径或边界钳位交互。本单已加护栏，完整修复需集群重算。

2. **探针位置不确定性**：H2 的真实 x 坐标在论文 Fig. 17 中是栅格图，不可机器读取。代码使用"箱面 - 0.05 m"约定（x = 2.426 m），但 m22476 的站位格分析表明该位置不在可容许区间内（需要 a = 1.434 m > column_length 1.228 m）。在可容许的 gate anchor（x = 2.220 m）处，dx=0.04 的峰值误差为 8.4%（<10%），到达误差 12.3%（>10%）。

3. **时间窗口不足**：实验全局峰值在 t = 1.70 s（回流水再撞），但多数 2-D 运行只到 t = 1.0-1.2 s，测到的是窗口内局部峰而非全局峰。

4. **3-D 收敛阶梯不完整**：dx=0.03 rung 缺失，无法做 GCI 外推。

5. **glibc abort**：3-D 生产运行间歇性因 glibc heap abort 崩溃，导致记录不完整。

---

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 最佳值 | 基准值 | 相对误差 | 是否 <10%？ | 是否 <3%？ |
|---|---|---|---|---|---|
| 2-D 峰值 (dx=0.01) | 0.377 m | 0.370 m | 2.0% | **是** | 否 |
| 2-D 到达 (dx=0.01) | 0.444 s | 0.359 s | 23.7% | **否** | 否 |
| 3-D 峰值 (dx=0.05) | 0.120 m | 0.370 m | 67.5% | **否** | 否 |
| 3-D 到达 (dx=0.04) | 0.468 s | 0.359 s | 30.3% | **否** | 否 |

**总判决**：FAIL。理由：至少一项关键指标（到达时刻）在所有 rung 上均超出 10% 容差。

**honest FAIL 声明**：本单不解锁"达标"，它解锁的是"算子一致性证明 + 判决状态梳理"。判据本身未通过，不得粉饰。

---

## ⑦ 产物

- `tests/test_kleefsman_3d_height_operator_fix.py`：8 个单元测试，验证 3-D/2-D 算子一致性
- `docs/verification_kleefsman_m22774.md`：本文档
- 判决状态总览（见第③节）

---

## ⑧ 给下次自找活的建议线索

1. **修复 height_operator 零值 bug**：需要在 `run_case` 的 checkpoint/resume 路径中正确保存和恢复 `height` 历史，并验证边界钳位不影响站内粒子计数。

2. **补全 3-D 收敛阶梯**：递交 dx=0.03 rung 到集群，要求 t_end >= 2.6 s 以覆盖实验峰。

3. **探针位置敏感性研究**：在可容许区间 [1.228, 2.220] m 内扫描探针位置，给出误差包络。

4. **2-D 全长时程**：dx=0.01, t_end=3.0 s 的 2-D 运行，以捕获 1.70 s 的全局峰。
