# 调研：SPHinXsys FSI2 基准（Turek-Hron FSI2）

## 0. 本锚点"FSI2"的所指

| 候选含义 | 证据 | 本轮取舍 |
|---|---|---|
| 本仓 Elastic-gate FSI（Antoci 2007） | 已有 `fsi_elastic_gate.py`，判据为流→固载荷+杆响应 | **已有**，非本轮目标 |
| SPHinXsys `test_2d_fsi2`（Turek-Hron FSI2） | Zhang, Rezavand & Hu, *JCP* 404:109135 (2020) §4.3；Turek & Hron, LNCSE 53 (2006) 371-385 | **本轮采用**：圆柱+弹性梁流致振动，force-from-fluid 方法验证 |

## 1. Turek-Hron FSI2 基准参数

来源：Turek & Hron (2006) 原始论文，逐条核对。

| 量 | 值 | 单位 |
|---|---|---|
| 槽长 | 2.5 | m |
| 槽高 | 0.41 | m |
| 圆柱直径 | 0.1 | m |
| 圆柱中心 | (0.2, 0.2) | m |
| 梁长 | 0.35 | m |
| 梁厚 | 0.02 | m |
| 流体密度 | 1000 | kg/m³ |
| 流体动力粘度 | 1.0 | Pa·s |
| 入口最大速度 | 0.3 | m/s |
| 入口平均速度 | 0.2 | m/s |
| 雷诺数 (D, U_mean) | 100 | — |
| 固体密度 | 1000 | kg/m³ |
| 固体杨氏模量 | 4.0e6 | Pa |
| 固体泊松比 | 0.3 | — |

## 2. SPHinXsys 实现要点

来源：SPHinXsys 源码 `tests/2d_examples/test_2d_fsi2/`（GitHub 读取）。

| 环节 | SPHinXsys 做法 | 本仓对应 |
|---|---|---|
| 流体 | WCSPH，Tait EOS，c₀ = 20√(140g) ≈ 234 m/s | 复用 `droplet_impact.rhs` |
| 固体 | TLSPH（Total Lagrangian SPH），超弹 Saint-Venant Kirchhoff | 离散弹性杆（能量法，同 `fsi_elastic_gate`） |
| 流固耦合 | 成对压力相互作用（Adami 型壁面项） | 面元外推 + W^T 集总（panel 模式） |
| 入口 | 抛物线速度剖面 | `set_inlet_velocity` |
| 圆柱 | 静态边界粒子 | `build_cylinder_particles` |

## 3. Force-from-Fluid 方法

本模块的核心验证目标是 **force-from-fluid** 方法：流体对结构的力如何从流体侧计算并传递给结构。

方法链：
1. **面元压力外推**：对每个结构边界粒子，用邻近流体粒子的压力做一阶加权最小二乘拟合，得到壁面压力
2. **力计算**：壁面压力 ×  tributary 长度 × 法向 = 作用力
3. **W^T 集总**：通过插值矩阵的转置将边界粒子力集总到结构节点
4. **力守恒**：W 的行和为 1 ⇒ W^T 保持总力不变

验证策略（不依赖长时积分）：
- V1：入口速度剖面 vs 解析抛物线
- V2：圆柱粒子几何 vs 解析圆
- V3：梁刚度 vs 解析 EI
- V4：梁小变形挠度 vs Euler-Bernoulli 公式

## 4. 参考值说明

Turek-Hron 原始参考值是针对**不可压** Navier-Stokes 求解器的：
- 阻力系数振幅 ~0.96
- 升力系数振幅 ~0.22
- 梁 tip x 振幅 ~0.0028 m
- 梁 tip y 振幅 ~0.0082 m

WCSPH 在有限 Mach 数下会产生不同的数值，因此本模块**不将耦合时程作为判据**，而是将 force-from-fluid 方法论的验证建立在解析参考上（V1-V4）。耦合运行作为诊断输出，ref 标记为 WITHHELD。
