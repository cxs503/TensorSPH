# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 3-D 生产臂：作业堆损坏崩溃，判决仍 WITHHELD

TensorSPH · branch `agent/m21952` · mission 21952 · 锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`

**结论（honest FAIL）：3-D 生产臂在本构建上跑不到 t_end = 2.6 s —— 两个网格都在中途以
`free(): corrupted unsorted chunks` 崩溃（rc=255）。箱前水位峰值/到达时刻对实验的比较仍然
`WITHHELD`，不声明达标。** 本单交付：把"3-D 结果总也拿不到"从"集群排队"纠正为**根因定位**
（内存越界写导致 glibc 堆损坏），并交付分臂报告生成器与不变量测试。

## ① 案例与判据

Kleefsman et al. (2005) *J. Comput. Phys.* 206:363–393, DOI 10.1016/j.jcp.2004.12.007：
槽 3.22 × 1.00 × 1.00 m，左端 1.228 × 0.550 m 水柱溃决，冲击 x = 2.476 m 处
0.161 × 0.161 × 0.403 m 方箱（40 % 阻挡，60 % 侧向旁通）。H2 在箱前；判据 = 峰值与到达
时刻 vs 实验，容差 < 10 %。实验锚点（Fig. 19 右 panel 数字化，m20884）：
峰值 0.370 m @ 1.7027 s，到达（0.02 m 阈值）0.3591 s。

2-D 臂把方箱当跨满槽宽（100 % 阻挡），峰值超实验 15–52 %；残差被**归因**于几何差。
m21728 造 3-D 臂、m21869 造阶梯工具，但都没有一条跑完的 3-D 生产 run（前单记为"队列未回收"）。

## ② 怎么做的

| 部件 | 内容 |
|---|---|
| 求解器（未改动） | `src/tensorsph/kleefsman_obstacle_3d.py`（WCSPH + 墙粒子 + 箱体硬夹持 + 旁通计数） |
| 驱动（未改动） | `run.py --dx d --t-end 2.6 --wall-layers 3 --threads 8` |
| 冻结协议 | 除 dx 外全一致：t_end 2.6、c0 20、wall_layers 3、孔位 0.05 m、孔径 0.08 m、阈值 0.02 m |
| 新增工具 | `benchmarks/kleefsman_obstacle_3d/production_report.py`：按 `obstacle_z`（0.0 贴壁 / 0.3 居中＝实验装置）**分臂**，逐臂阶梯拟合与判决，程序化注入数字并出图 |
| 新增测试 | `tests/test_kleefsman_3d_production_report.py`（6 条全绿）：不得跨几何做阶梯；FAIL 不得写成 PASS；表内误差须等于注入值重算；无 run 必须非零退出 |
| 递交 | cj3797 dx=0.05 z=0.0；cj3798 dx=0.04 z=0.0；cj3799 dx=0.03；cj3800 dx=0.05 z=0.3；cj3801 dx=0.04 z=0.3（各 8 核，stage_dir=本工作树） |

## ③ 作业终局（原始证据）

| 作业 | dx [m] | 终态 | rc | 崩溃前最后进度 | 崩溃信息 |
|---|---|---|---|---|---|
| cj3797 | 0.05 | succeeded→rc=255 | 255 | t = 0.263 s（slab 0.000 m） | `free(): corrupted unsorted chunks` |
| cj3798 | 0.04 | succeeded→rc=255 | 255 | t = 1.053 s（slab 0.077 m） | `free(): corrupted unsorted chunks` |

证据指针：`/srv/tensor/agent/cluster/jobs/cj3797.log`、`cj3798.log`（`output_tail` 原文）；
产物包 `fetch/cj3797_n315.tar.gz`、`cj3798_n315.tar.gz` 内 `benchmarks/kleefsman_obstacle_3d/results/`
**只有 `ladder.json`，无任何 `dx*.json`/`dx*_time_series.csv`** —— 与 `run.py` 只在积分结束后
落盘一致。

两条与网格无关的读数：
* **到达下界**：dx=0.04 直到 t = 0.491 s 才首次读到 slab = 0.069 m，实验到达 0.3591 s → 3-D 臂
  前锋到达比实验**晚 ≥ 36.7 %**。按 10 % 容差，**到达一项已可判 FAIL**（无需等满窗口）。
* **峰值不可判**：崩溃在 t ≈ 1.05 s，远早于实验峰时刻 1.7027 s，属 `censored`，按 `score()`
  口径不得判 PASS/FAIL → `WITHHELD`。

## ④ 注意点 / 局限（如实）

1. **根因未定位到行号**：`free(): corrupted unsorted chunks` 是 glibc 堆元数据损坏（越界写，
   非数值发散）。嫌疑在 3-D 路径的原地写 `clamp_out_of_obstacle`（`pos[idx]=new` / `vel[idx]=nv`）、
   箱体格点构造 `obstacle_lattice_3d`、以及 `PairListCache` 在粒子数变化时的复用。崩溃时刻随 dx
   变化（0.263 s / 1.053 s），说明由**拓扑事件**触发而非固定步数。
2. **纠正前单归因**：m21728 的本机自检在 120 s 命令上限处被截断于 t ≈ 0.351 s，恰好贴近
   dx=0.05 的崩溃时刻，因此"看起来正常"；缺席不是排队，而是崩溃。
3. **未做**：ASAN/valgrind 复现、最小复现用例、修复后重跑阶梯——超出本单预算，入口见 §⑥。
4. **无图**：没有一条跑完的 run，任何曲线都只能画到崩溃时刻，当"结果"呈现即粉饰。
   `production_report.py` 已就绪，作业一落地即出 H2 overlay / 误差-网格 / 旁通时程 / 2D-vs-3D 四张图。

## ⑤ 误差判定：关键量 vs 基准

| 量 | 基准 | 本单 3-D 实测 | 相对误差 | < 3 %？ | < 10 %？ |
|---|---|---|---|---|---|
| 箱前水位峰值 | 0.370 m | 未取得（崩溃于 t≈1.05 s < 1.7027 s，censored） | 未取得 | 否—不可声明 | 否—不可声明 |
| 到达时刻（0.02 m） | 0.3591 s | ≥ 0.491 s（dx=0.04，首次读到 0.069 m） | ≥ 36.7 % | 否 | **否（FAIL）** |

**总判决：`WITHHELD`，不算例达标。** 到达一项已在 10 % 容差外，峰值因窗口被崩溃截断不可判。

## ⑥ 给下一单的入口

1. `PYTHONMALLOC=malloc` + `valgrind --tool=memcheck`（或 ASAN）在 dx=0.05 上跑到 t≈0.4 s 复现并
   定位越界写（预计 ≤ 30 min 集群机时）。
2. 修复后按同一冻结协议重跑 0.05/0.04/0.03，`production_report.py` 一条命令出判决与四张图。
3. 若到达 +36.7 % 在修复后仍存在，则"旁通归因"应改写为"旁通不足以解释"：3-D 在到达一项上
   比 2-D 更差（2-D dx=0.04 到达 0.484 s，dx=0.02 为 0.454 s；3-D ≥ 0.491 s）。
