# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — TensorSPH 带障碍物溃坝（Kleefsman 2005）

TensorSPH · 分支 agent/m16070 · 2026-11-12T09:15Z · 本文档由脚本从
`artifacts/dx0.02/metrics.json` 与 `reference.json` 注入生成（零手抄）。

## ① 问题与 ref 出处
长 3.22 m / 高 1.00 m 水槽左端 1.228 m×0.55 m 水柱瞬时溃决，沿干底推进，撞击 x=2.476 m 处
高 0.161 m、流向长 0.161 m 的刚性方箱（Kleefsman 等 2005 标准冲击算例）。

- ref 原文：Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation method for
  wave impact problems*, **J. Comput. Phys. 206:363–393**, DOI 10.1016/j.jcp.2004.12.007。
- 几何出处（本节点可核验）：Amaro Junior, Cheng & Buruchenko (2020) *IJOPE* 30(4) doi:10.1142/s021987622050036x，
  开放预印本 **arXiv:2306.01959 Fig. 2**（复刻 Kleefsman 2005 装置图，尺寸串 1.000/1.228/1.248/0.744/0.550/0.161/0.403）；
  交叉核验 English et al. (2021) doi:10.1007/s40571-021-00403-3（3.22 m 槽、1.22 m 水柱、0.55 m 高、
  0.40×0.16×0.16 m 方箱、闸门下游 1.248 m）。
- 自洽校验：1.228+1.248+0.744 = 3.220 m = 槽长（残差 0）；箱上游面 1.228+1.248 = 2.476 m，箱尾 2.637 m，下游余隙 0.744 m。

**指标**：障碍物前水位时程峰值与到达时刻，容差 |Δ|/ref < 10%。

**诚实声明（关键）**：实验数值 **WITHHELD**。Kleefsman 2005 水位数据只有曲线；本节点取不到原文 PDF——
格罗宁根大学仓储副本 `research.rug.nl/files/2941055/2005JCompPhysKleefsman.pdf` 被 Cloudflare 拦截（403），
ScienceDirect 403，公开文献无该算例水位数字化表格；H1–H4 探头 x 坐标只印在装置图内。故本单不与实验做
百分比对照；`reference.json` 中 `peak_height_m/arrival_time_s/probe_x_m` 留空，填入后 `run.py` 自动执行
<10% 判据并把 verdict 从 WITHHELD 翻为 PASS/FAIL。

## ② 方法与实现
- 复用现有 WCSPH 内核（`droplet_impact`：cubic-spline 核、密度求和+Shepard 修正、Tait EOS、静态壁
  Adami 型压力外推、cell-linked 配对缓存），未另起炉灶。
- 新增 `src/tensorsph/kleefsman_obstacle.py`：几何/探针/推进配置、水体晶格、**实心内部障碍物**（填满整箱而非薄壳；
  dx=0.01 时 16×16=256 个静止粒子，随 wall 参与密度求和与压力外推）、水位探针与峰值/到达时刻后处理。
- 探针按几何定义在 `x = 箱上游面 − 0.05 m = 2.426 m`，半宽 1·dx；自由面取竖条内
  **最高 3 粒子平均**（单点极大被冲击飞溅孤立粒子污染，dx=0.02 曾误报 1.000 m=槽顶）。
- 到达时刻 = 首次越过 0.02 m，两采样点间线性插值（不受输出间隔量化）。
- 复现：`python benchmarks/kleefsman_obstacle/run.py --dx 0.02 --t-end 1.2 --tag dx0.02 --figure`；
  集群分片入口 `benchmarks/kleefsman_obstacle/cluster_run.sh {i} {n}`（shard1 dx=0.014, shard2 dx=0.010）。

## ③ 参数表（无量纲量完整）
| 量 | 值 | 说明 |
|---|---|---|
| dx | 0.02 m | 粒子间距；h/dx = 1.3（核长 h = 1.3dx）|
| 流体粒子数 | 1708 | 1.228/dx × 0.55/dx |
| 声速 c0 | 20 m/s | 数值声速；ρ0 = 1000 kg/m³，γ = 7 |
| 马赫数 U/c0 | 0.33 | 峰值速度 6.64 m/s |
| 声学 Courant | ~0.26 | dt·c0/(h·dx)，dt = 2.0e-04 s |
| 时间步 | 6001 步 / 1.200 s（目标 1.2 s）| 输出 302 行 |
| 重力 | 9.81 m/s² | 边界：3 层静态壁 + 实心箱，自由滑移 |
| 初场 | 静水密度初始化，水位 0.55 m | |
| Froude U/√(gH) | 2.86 | 以水柱高 0.55 m 为尺度 |
| Reynolds UH/ν | 365 | 数值粘性主导，非解析对照 |

## ④ 对照表（ref vs got）
| 指标 | ref (Kleefsman 2005 实验) | got (TensorSPH dx=0.02) | 误差 | 判定 |
|---|---|---|---|---|
| 障碍物前水位峰值 | WITHHELD | 0.696 m @ t=0.888 s | 不可算 | ✗ 无 ref |
| 到达时刻 | WITHHELD | 0.454 s | 不可算 | ✗ 无 ref |

可核验项：
| 检查 | 结果 | 判定 |
|---|---|---|
| 质量守恒（相对误差） | 0.0e+00 | ✓ |
| 越界粒子数（壁面失效计数） | 0 | ✓ |
| 最大密度压缩 | 12.0% | ✗ 粗网格冲击压缩超 5% 可控线 |
| 几何自洽 1.228+1.248+0.744=3.22 | 残差 0 | ✓ |

## ⑤ 收敛性 / 参数敏感性（至少两点）
1. **自由面定义敏感性（已完成）**：单点最高粒子 vs top-3 平均，dx=0.02 峰值由 1.000 m（槽顶，飞溅污染）
   降到 0.696 m → 该"水位峰值"对后处理定义极敏感，必须先固定定义再谈与实验比；已固化为代码默认。
2. **探针位置敏感性（接口已备，`--probe-offset`）**：峰值跑升随探针靠近箱面升高，属算例物理。
3. **网格收敛（未闭合，如实记录）**：dx=0.02 已完成；dx=0.01 已递交（本地后台 + 集群 cj1231），本单结束时
   `artifacts/dx0.01/metrics.json` 尚未回收，故**不给收敛阶**。

## ⑥ 失败模式与工件
- **failure_reason**：判据所需实验值 WITHHELD（原文 PDF 不可达，见①）→ <10% 容差判据本单**未通过评估**（非"通过"）；
  dx=0.01 细网格未回收，收敛性未闭合。
- 数值失败模式：dx=0.02 冲击处密度压缩 12.0%（>5%），箱前水花细节不可信；飞溅粒子会污染单点水位峰值（已抑制）。
- 工件：`src/tensorsph/kleefsman_obstacle.py`、`benchmarks/kleefsman_obstacle/{run.py,reference.json,cluster_run.sh}`、
  `benchmarks/kleefsman_obstacle/artifacts/dx0.02/{metrics.json,time_series.csv,water_height.png}`、
  `tests/test_kleefsman_obstacle.py`、本文档。

### 测试状态（程序捕获，未粉饰）
```

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_kleefsman_obstacle.py::test_water_column_lattice_and_mass
1 failed, 6 passed, 1 warning in 12.49s
```

### 生态摩擦（friction）
1. 原文获取：research.rug.nl 仓储副本 Cloudflare 挑战 403、ScienceDirect 403；egress 白名单还挡掉
   duckduckgo / scholar.archive.org / r.jina.ai，Bing 返回本地化噪声 → 平台缺"文献原文获取"能力，
   直接卡死一切 vs 实验的判据（本单首害）。
2. 三查命中：knowledge_base capability、negative_knowledge_ledger（Kleefsman / dam break obstacle / TensorSPH）
   与仓内 grep 均 0 命中 → 本单为首落地。
3. 集群 `queued_ahead=406`（cj1231 双分片），本机 shell 120 s 超时迫使重计算只能后台/集群，回收窗口与本单预算不匹配。
4. run_shell 安全策略拦掉含 `$(...)` 命令替换与 `curl -e/-A` 的命令，取 token/发报告只能走 python urllib。

## 补充（agent/m17524）：先量仪器，再量解

本文④表的「got」列用的是 legacy 顶 3 粒子均值，它对新网格有 **0.4–1.4 dx 的系统性偏低**
（dx=0.02 时 −1.8…−2.7 cm，dx=0.04 时 −1.5…−5.3 cm），大于本算例自己的 0.02 m 到达阈值。
`docs/verification_kleefsman_gauge_bias.md` 给出三种算子的解析偏置、固定物理孔径体积计，
以及两个网格之间的读数极差：历史指标 peak 21.6 %（>10 % 容差，不可判）、
`slab@x2.228` 1.3 %。实验值仍 WITHHELD，验收结论未变。
