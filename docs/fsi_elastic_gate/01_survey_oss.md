# 调研①：开源横向 —— 弹性闸门 (elastic gate) FSI 的现成实现

日期 2026-11-12（m16352，第 2 轮）。检索全部由程序发出，返回体在
`artifacts/fsi_elastic_gate/provenance.json` 有摘要记录（源 URL + 状态码 + 关键字段）。

## 1. 参考算例本体（Antoci 2007）的可获得性

| 查询 | 结果 |
|---|---|
| Unpaywall `10.1016/j.compstruc.2007.01.002` | `best_oa_location = null`（无 OA 副本） |
| Crossref `works/10.1016/j.compstruc.2007.01.002` | 记录存在（Computers & Structures 85:879-890），无全文 |
| OpenAlex `works/doi:...` | 被引 100+，但 citing works 中的 OA 副本无一给出实验曲线数值表 |
| arXiv API `all:"elastic gate"` / `all:Antoci` | 无相关预印本 |
| Zenodo `antoci elastic gate` | 无数据集 |
| 尝试直接取 OA PDF（mediatum/TUM、orbi.uliege、MDPI、ScienceDirect-AM、raw.githubusercontent） | 全部 403 / 连接拒绝 / 超时（网关只放行 JSON API） |

结论（与前轮一致的 WITHHELD，不是本轮新发现）：**Antoci 2007 实验曲线的数值在本次可用
通道内仍不可读**，所以"峰值/相位 vs Antoci2007 实验 <10%"这一判据本轮**无法计算**；本轮
把力气放在"可计算的判据"上（见 02/03），并给出可复算的失败阶段。

## 2. 开源实现的代码本体（逐个读过源码）

### 2.1 SPHinXsys（Xiangyu-Hu/SPHinXsys，GitHub contents API 读 master，2026-11-12）

| 文件 | 读到的关键事实 |
|---|---|
| `tests/2d_examples/test_2d_elastic_gate/elastic_gate.cpp` | 弹性闸门算例：水块 100 mm × 140 mm，闸门自 `Base_bottom_position = 79.0` 以上被约束（即自由段 79 mm），`rho0_f = 1.0`, `gravity_g = 9.8e-3`, `c_f = 20*sqrt(140*g)`（弱可压），固体 `SaintVenantKirchhoffSolid(rho0_s=1.1, Youngs_modulus = Ae*rho0_f*U_f^2, Ae = 7.8e3, poisson = 0.47)`，`end_time = 400`（无量纲）。 |
| `tests/2d_examples/test_2d_fsi2/fsi2.cpp` | **不是** Antoci 闸门：是圆柱后弹性梁的流致振动（Zhang, Rezavand & Hu, JCP 404:109135, 2020），不可用作本锚点参照。 |
| `regression_test_tool/Observer_Position_Run_*.xml` | 只存 SPHinXsys 自己的回归输出，**不是**实验数据，不能当 Antoci 基准。 |

**可直接学的结构**：固体用总拉格朗日 SPH（TLSPH，Saint-Venant Kirchhoff 超弹），流固之间用成对压力
相互作用力；闸门作为连续体粒子而不是杆。参数上它与 Antoci 2007 原件不同（水块 100×140 而非
146×292，闸门自由段 79 mm、E 无量化后约 7.8e3·ρc² 量级），**不能把它的参数当作 Antoci 参数**。

### 2.2 方法谱系（文献/教科书标准做法，external anchor）

| 环节 | 标准做法 | 出处（external anchor） |
|---|---|---|
| 基准算例 | 弹性闸门溃坝：水柱 0.146 × 0.292 m，闸门 0.36 × 0.004 m，E = 1e6 Pa，报告闸门顶点水平位移时程 | Antoci, Gallati & Sibilla, *Comput. Struct.* 85 (2007) 879-890, DOI 10.1016/j.compstruc.2007.01.002（闭源，本轮不可读） |
| 壁面/流固压力 | Adami, Hu & Adams 的壁面压力项 `f = m_g Σ m_f (p_f/ρ_f² + p_g/ρ_g²) ∇W` | Adami, Hu & Adams, *J. Comput. Phys.* 231 (2012) 7057 |
| 边界核梯度亏缺 | 一阶一致性修正 `∇̃W = M^{-1}∇W`, `M = Σ V_j (x_j-x_i)⊗∇W_ij`；或对壁面做压力外推 | Bonet & Lok, *Comput. Methods Appl. Mech. Engrg.* 177 (1999) 241 |
| 固体力学（大变形） | 超弹/TLSPH；或离散弹性能量最小化（拉伸+弯曲） | SPHinXsys `SaintVenantKirchhoffSolid`；离散杆能量见本仓 `fsi_elastic_gate.rod_energy` |
| 大变形梁的解析参照 | Kirchhoff elastica（Bisshopp–Drucker 问题），`EI dθ/ds = M(s)` | Bisshopp & Drucker, *Q. Appl. Math.* 3 (1945) 272（本轮以 ODE 积分自行求解，不抄表） |
| 均布载荷小变形梁 | `δ_tip = ∫ q(a) a²(3H−a)/(6EI) da` | Euler-Bernoulli 梁 Green 函数（教科书） |

## 3. 本仓现状（对照用）

`src/tensorsph/fsi_elastic_gate.py`（m16214 交付，518 行）已实现 WCSPH + 离散弹性杆 + 交错显式耦合，
但其三条自查在 m16257 审计中两条 FAIL：V1 流固载荷 −20.4 %，V2 悬臂梁 −92.5 %（自重视屈曲的
伪验证）。本轮调研的落点因此不是"重写求解器"，而是**换掉两个有系统性偏差的子环节**（见 02）。
