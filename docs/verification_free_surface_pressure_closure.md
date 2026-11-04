# 自由面压强闭合（Adami 2012, eq. 11）：拿回 16 个百分点，剩下的缺口在力不在闭合

**仓库**: TensorSPH ｜ **代码**: `src/tensorsph/free_surface_pressure.py`（`Config.free_surface_pressure`）
｜ **工件**: `benchmarks/laplace_pressure_floor/`（`run.py` 的 `p0_closure` 块、`report.json`、`verification_free_surface_pressure_closure.png`）
｜ **测试**: `tests/test_free_surface_pressure.py`（9 passed）+ `tests/test_laplace_pressure_floor.py`

**结论（先给判定）**：把 `backlog_update_m2971.md` 的 P0 实现出来（截断壳层按 Adami 等 (2012)
式 (11) 外推压强），端到端 Young–Laplace 残差从基线均值 **35.9%** 降到
**20.0%**（`shell` 模式，逐点 19.9%–20.1%），**仍不过 15% 门槛**，
故 Laplace 压差维持 **WITHHELD**。三点可裁的结论：

1. **闭合有效但不够**：16 个百分点的改善在两端声速上一致
   （+17.7 pt / +14.1 pt），不是噪声。
2. **「照壁面处理」是错的**：`shell_density`（再按 Tait 反解壳层密度）反而只改善
   7 pt，均残差 29.0%——动量的 `p/rho^2` 在
   壳层被额外放大。这是负结果，按纪律保留。
3. **残差已不在闭合上**：同一批算例里，CSF 力的径向积分只有 **0.56–0.58**
   倍 `sigma/R`（压强读数 0.80–0.80 倍），
   而力积分与压强读数之间的差达 **0.24** 倍（图 d）——两条读数都不在 1 上，
   说明**表面张力力幅本身偏小**是新的门槛项，且我的力积分口径与
   `laplace_droplet` 在 R=0.1 上测到的 0.79 不一致（§⑤ 记为未结清的仪器欠账）。

## ① 案例介绍（物理背景/几何/参数）

算例与 `docs/verification_laplace_pressure_floor.md` 完全相同（同一 `run.py`，同一初始化），
本单只是在它上面打开了闭合开关：

半径 `R = 0.02 m` 的二维单相液滴，六方密排初始化，四周真空，无重力，
弱可压 Tait 方程（`p(rho0)=0`），解析基准 `Delta p = sigma/R = 360.0 Pa`。

| 参数 | 值 |
|---|---|
| 半径 R | 0.02 m |
| 表面张力 sigma | 7.2 N/m（10x 水） |
| 密度 rho0 / gamma | 1000 kg/m³ / 7 |
| 声速 c0 | 5, 20 m/s |
| 分辨率 dx | 0.002 m（N = 362, 362） |
| 毛细时间 | 0.0333 s，观测 1.5 t_sigma |
| 解析压差 | 360.0 Pa |

**被修的缺陷**：`tait_pressure` 钳 `p>=0`；截断核支撑使自由面壳层和密度亏缺（0.95–1.05R 约 −30%），
壳层粒子全落在 `p=0`，界面无法承载 `sigma/R`。上一单实测内部只压到解析所需的 0.57±0.06 倍。

## ② 怎么做的（方法/流程）

**闭合**（`free_surface_pressure.extrapolated_pressure`，仅作用于 `rho < 0.99 rho0` 的粒子）：

    p_i = max(0, sum_j (p_j + rho_j g·(r_i - r_j)) W_ij) / sum_j W_ij

= Adami, Ihmsen & Adams (2012) 式 (11) 在静粒子（`a_i = 0`）上的特化，与求解器**已有的静态壁面**
压强外推同一构造。body-force 项不可省：`tests/test_free_surface_pressure.py`
用 `p = rho G (y_s - y)` 验证该项下外推对线性压场**精确**（1e-10 相对），并同时证明去掉该项会引入
~1e-3 相对偏差（截断模板的一阶矩）。

| `Config.free_surface_pressure` | 内容 |
|---|---|
| `off`（默认） | 历史行为，保持既有数字与测试逐位不变 |
| `shell` | 只外推压强 |
| `shell_density` | 外推压强 + 壳层取 Tait 一致密度（**仅用于动量分母**，连续性方程仍用真实和密度） |

**判别器**：每个算例除四种压强口径外，还用 `implied_jump_from_force` 计算
`|rho0 * int a_r dr|`（CSF 加速度的径向分箱积分，向内为正）。这样能把「压强是否把力传出来」
与「力本身多大」分开。

**复现**：

    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --quick     # 8 个算例，~10 min
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py

## ③ 计算结果

| 模式 | dx [m] | c0 [m/s] | N | 基线残差 [%] | 闭合残差 [%] | 改善 [pt] | 压缩比 | 力积分/`sigma/R` | <15% |
|---|---|---|---|---|---|---|---|---|---|
| `shell` | 0.002 | 20 | 362 | -42.1 | -20.1 | +17.7 | 0.787 | 0.559 | FAIL |
| `shell` | 0.002 | 5 | 362 | -43.6 | -19.9 | +14.1 | 0.669 | 0.577 | FAIL |
| `shell_density` | 0.002 | 20 | 362 | -42.1 | -30.5 | +7.3 | 0.644 | 0.495 | FAIL |
| `shell_density` | 0.002 | 5 | 362 | -43.6 | -27.5 | +6.5 | 0.600 | 0.506 | FAIL |

基线（`off`）逐点 37.9%, 34.0%，与 `report.json` 的历史记录一致
（本单复现，未改默认路径）。力积分口径在基线上为
0.44, 0.43 倍 `sigma/R`，闭合后为
0.56, 0.58。

## ④ 结果图

![closure](verification_free_surface_pressure_closure.png)

（a）逐点残差：闭合把两根柱子都压到 15% 线上方一点；（b）内部压缩比 0.57→0.79/0.67，
仍未到 1；（c）芯部压强时程；（d）压强读数 vs 力积分读数：两者都偏离 1，且彼此不重合——
这就是「残差已不在闭合上」的直接证据。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

- **适用范围**：只修「截断核 → 密度亏缺 → 压强钳零」这一条链；`p>=0` 钳位保留，张力不稳定不处理。
- **阈值 0.99**：六方格子自身有 +0.59% 和密度偏置，阈值提到 1.0 会把整个液滴当自由面（等价于
  全局 Shepard 平均），本单没走这条路。
- **`shell_density` 是负结果**：壁面是静止的、自由面不是，照搬壁面的密度一致性会破坏静平衡。
  保留它而不是删掉，是为了让后来者不必重跑这条路。
- **力积分口径未标定**：本单测到 0.56–0.58 倍 `sigma/R`，
  而 `laplace_droplet` 在 R=0.1（10 倍半径、更厚的界面）上测到 0.79。液滴只有 2h/R ≈ 5% 的界面厚度，
  径向分箱（40 箱）可能欠分辨。**该口径不得单独作为精度结论**，只能作相对比较用。
- 只跑了 `dx=0.002`、`c0 ∈ {5, 20}`（预算约束下的 `--quick` 矩阵），
  `dx=0.0015` 与 `c0=10` 未纳入本单。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

基准：`Delta p = sigma/R = 360.0 Pa`（Young–Laplace 解析解，非实验）。

| 量 | 闭合前 | 闭合后（`shell`） | 相对误差（闭合后） | <3%? | <15%? |
|---|---|---|---|---|---|
| 芯部压差 `p_core`（tail 平均） | 37.9% / 34.0% | 20.1% / 19.9% | 20.1%（最差点） | **否** | 否 |
| 内部压缩比 | 0.57 / 0.57 | 0.79 / 0.67 | 21.3%（对 1） | 否 | 否 |
| CSF 力积分 `|int rho0 a_r dr|` | 0.44 / 0.43 | 0.56 / 0.58 | 44.1%–42.3% | 否 | 否 |

**显式回答**：关键量 `Delta p` **不是 <3%**（最优点 19.9%）。按精度标准（>15% 不可接受），
端到端 Young–Laplace 压差仍判 **WITHHELD**，本单不得被引用为「Laplace 已通过」。

**对上一单结论的修正（honest-FAIL 记录）**：`docs/verification_laplace_pressure_floor.md` §⑦ 把闭合
写成「唯一门槛项」并预测闭合后 4 点全部 <15%。本单实测：闭合只拿回 16 pt，
**该预测被证伪**。新的门槛项是表面张力力幅（力积分仅 0.56–0.58 倍
`sigma/R`），与 `tests/test_laplace_droplet.py` 记录的 `dp_ratio ≈ 0.79–0.84` 是同一缺口的两条线索。

## ⑦ 下一步（机器可裁的判据）

1. **P0'（新，最高优先）**：标定力积分口径（分箱数/界面厚度/R 依赖），先在 R=0.1 与 R=0.02 上
   复现同一个数（判据：两个半径上 `|int rho0 a_r dr| / (sigma/R)` 相差 <5%），再谈力幅修正。
2. **P1**：`grad c` 归一化（`N_delta`）与 `fit_quadratic` 曲率系统偏差二选一定量归因；
   判据：`tests/test_laplace_droplet.py::test_laplace_droplet` 的 `dp_ratio` 进入 `1±0.03`。
3. **P2**：闭合阈值/迭代扫描（1–3 次外推）；判据：`shell` 残差对阈值不敏感（漂移 <2 pt）。
4. 端到端 xfail 转正的前提不变：
   `tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct` 由 strict-xfail 转 xpass。
