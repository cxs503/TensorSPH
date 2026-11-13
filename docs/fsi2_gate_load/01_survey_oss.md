# 调研①：开源横向 + 方法谱系（anchor `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`）

第 2 轮（m17584，2026-11-13）。前轮 m17519 在本锚点 honest-FAIL 且未留 `fail_stage`/`failure_reason`；
本单先做调研再动手，落点是"可计算的判据"，不是重跑上轮。

## 0. 本锚点"FSI2"的所指（先对表，避免白做）

| 候选含义 | 证据 | 本轮取舍 |
|---|---|---|
| 本仓 Elastic-gate FSI 的第 2 阶段交叉验证 | 本仓 `src/tensorsph/fsi_elastic_gate.py` + 三条前序锚点（m16214 交付 / m16257 审计 / m16352 第 2 轮） | **采用**：判据 = 流→固载荷 + 杆响应，且必须有 ref/got/误差 |
| SPHinXsys 的 `tests/2d_examples/test_2d_fsi2` | 前轮源码调研（`docs/fsi_elastic_gate/01_survey_oss.md` §2.1）：该测试是**圆柱后弹性梁的流致振动**（Zhang, Rezavand & Hu, *JCP* 404:109135, 2020），与弹性闸门无关 | **不采用**（本轮预算内无法引入不可压/低频长时积分求解器，且它的参数同样不是 Antoci 参数）；见 §3 迁移路线 |

## 1. 开源实现（读源码，不读二手描述）

来源：`docs/fsi_elastic_gate/01_survey_oss.md`（m16352，逐文件读 GitHub contents API，sha 记录在
`artifacts/fsi_elastic_gate/provenance.json`）。本单复核了该文的结论并只引用其中与本次改动直接相关的三条：

| 项目 | 源码事实 | 对本单的作用 |
|---|---|---|
| SPHinXsys `test_2d_elastic_gate/elastic_gate.cpp` | 固体用**总拉格朗日 SPH（超弹 Saint-Venant Kirchhoff）**，闸门是连续体粒子；`end_time=400` 无量纲；水块 100×140（**不是** Antoci 的 146×292） | 结构侧的两条可选路线：(a) TLSPH 连续体（重、需重写），(b) 离散杆 + 能量（本仓已有）。选 (b)，理由见 02 |
| SPHinXsys 流固耦合 | 流固之间用**成对压力相互作用力**（Adami 型壁面项） | 本仓 m16352 已证该成对和在壁面半空间核支撑下**系统性亏缺**（−20.4 % @ dx=10 mm），故保留"压力外推面元法"为生产路径，成对和只作 before/after 基线 |
| SPHinXsys `regression_test_tool/Observer_*.xml` | 只存它自己的回归输出，**不是实验数据** | 不可当 Antoci 基准；拒绝把自家回归当实验锚（上一轮"抄锚"风险的来源） |

**没有**任何开源实现提供 Antoci 2007 的实验数字表：其 `Observer_*.xml` 是自算回归，Chrome/Cloudflare
保护的期刊 PDF 不可取，arXiv/Zenodo 无数据集（前轮 5 条通道全部 403/拒连/超时）。本单因此**不再尝试**
把该曲线当判据，而是把判据建在**解析/独立求解**能给出的量上。

## 2. 方法谱系（教科书/文献标准做法，external anchor 逐条给出处）

| 环节 | 标准做法 | 出处 | 本仓落地 |
|---|---|---|---|
| 静水壁面载荷 | `p = p0 + ρg(h−y)` 积分，`F = ½ρg h_s²` | Landau & Lifshitz, *Fluid Mechanics* §1–§2 | `verify_hydrostatic_load`（面元外推路径） |
| 壁面压力外推 | 壁面一阶加权最小二乘拟合邻域压力平面，取壁面值乘面板长度 | 壁面边界压力外推（WCSPH 常用做法；一致性梯度修正见 Bonet & Lok, *CMAME* 177 (1999) 241） | `gate_load(..., load_transfer="panel")` |
| 小变形梁 | Euler-Bernoulli 悬臂 Green 函数 `δ=∫q(a)a²(3H−a)/(6EI)da` | 教科书梁理论 | `eb_tip_distributed` |
| 大变形梁 | Kirchhoff elastica，`EI dθ/ds = M(s)`，弧长打靶 | Bisshopp & Drucker, *Q. Appl. Math.* 3 (1945) 272 | `elastica_tip`（自行积分，**不抄表**） |
| 结构时间积分 | 显式辛欧拉 + **结构子循环**（固有不稳定频率决定子步数） | 交错分区 FSI 的常规做法（结构子循环，见 Farhat & Lesoinne 类 staggered 分区综述） | 本单新增 `rod_substeps` / `rod_substeps_effective()` |
| 静力平衡 | 能量最小化或其梯度残差求根（`dΠ/du = 0`），并**检查残差** | 非线性最小二乘/Newton 求根的标准纪律 | 本单重写 `rod_static_equilibrium` |

## 3. 迁移路线（未做部分，写清理由）

1. **TLSPH 连续体固体**（SPHinXsys 路线）：需要引入变形梯度的 SPH 离散与超弹本构，属独立大改，
   且与本仓现有离散杆校核链（EB/elastica）不共享代码；不在本轮范围。
2. **SPHinXsys `test_2d_fsi2`（圆柱+弹性梁流致振动）**：需要长时间低频积分与更小 CFL 的不可压/弱可压设置，
   并且其无量纲参数表要逐条从源码读出（`Ae = 7.8e3`、`c_f = 20√(140 g)` 等）；本轮预算不足，
   已在 `docs/fsi2_gate_load/03_report.md` 登记为下一阶段候选。
3. **Antoci 2007 实验曲线**：仍 WITHHELD（三次任务一致结论），本单不再把它写成可通过的判据。