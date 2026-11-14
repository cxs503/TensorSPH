# XVAL-SPH-PROJECTION-TAYLOR-GREEN — 周期投影 ISPH 对 Taylor-Green 涡的验证（m20500）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（residual debt 明列的 `XVAL-SPH-PROJECTION-TAYLOR-GREEN`）
**仓库/分支**: TensorSPH · `agent/m20500` ｜ **日期**: 2026-11-14
**判据（预声明）**: 速度场相对 L2 误差 ≤ **15%**（owner 精度标准），`t_end = 0.5`,
Re = 10, `dx = 0.05`、`Co = 0.1` 为主门配置。
**结论**: **honest FAIL——23.613% > 15%**。方法选型表因此**继续拒选**涡类剖面（不是"无记录"，
而是"有测量记录且超差"），证据洞由猜测升级为实测。

## 1. 为什么做这件事（缺口）

`src/tensorsph/method_selection.py` 只为"有记录且在 15% 内"的方案签发 `SELECTED`。登记表里
`isph_projection` 的唯一记录来自**壁面槽道**（CASE-ISPH-CHANNEL，0.87%），因此
`smooth_inviscid_vortex=True` 的剖面按设计返回 `WITHHELD`——路由表拒选涡类不是缺陷，而是
"没有涡类证据"的诚实表达。登记表 m20114 后已无"零记录"方案，剩下唯一未闭的口就是这个
**涡类-投影**记录。本单把它测出来。

## 2. 怎么做的

| 环节 | 选择 | 指针 |
|---|---|---|
| 参考解 | Taylor & Green (1937) *Proc. R. Soc. Lond. A* **158**:499 精确非定常 NS 解；能量按 `exp(-4νk²t)` 衰减 | `benchmarks/isph_taylor_green/run.py::exact_field` |
| 格式 | 周期 ISPH 投影（Cummins & Rudman 1999）: `D/G` 一阶一致差分算子 + M-矩阵（图）Laplacian 解压力泊松 | `src/tensorsph/isph_periodic.py` |
| 粘性 | 成对（图）Laplacian，`ν∇²u = -ν·L u`（符号由"正 ν 必耗散"测试钉住） | `viscous_laplacian` |
| 时间推进 | Heun（中点），每阶段一次投影 | `run.py::run_config` |
| 周期配对 | 最小镜像、无幽灵粒子、无壁面 | `periodic_pairs` / `pair_geometry` |
| 边界条件 | 环面：压力常数零模用**零均值投影**消去（不钉粒子，周期盒里钉粒子会特殊化一个粒子） | `poisson` |
| 判据 | 拉格朗日粒子位置上速度 vs 精确解相对 L2 误差 ≤ 15%（读数字前钉死） | `run.py: TOLERANCE_PCT` |

算子按各自核矩归一（`b_i = -(1/d)Σ_j V_j r_ij·∇W_ij`，`a_i = diag(A)`），使它们在**无序云**上
仍一阶一致；这是本单发现的第二个根因（见 §4）。

## 3. 结果（程序化产出 `outputs/isph_taylor_green/results.json`，零手抄）

| 配置 | 粒子 | 步数 | dt | 速度相对 L2 误差 | E/E_exact | 投影后/前离散散度 | vs 15% |
|---|---|---|---|---|---|---|---|
| **dx=0.05, Co=0.1（门）** | 400 | 77 | 6.50e-3 | **23.613 %** | 0.755 | 0.792 | **FAIL ✗** |
| dx=0.05, Co=0.05 | 400 | 154 | 3.25e-3 | 20.113 % | 0.797 | 0.832 | FAIL ✗ |
| dx=0.04, Co=0.1 | 625 | 96 | 5.20e-3 | 2.141e5 %（发散） | 4.59e6 | 2.267（散度被放大） | FAIL ✗（不稳定） |

算子本身的独立体检（`tests/test_isph_periodic.py`，20×20 晶格）：

| 量 | 测量 | 期望 |
|---|---|---|
| 精确 TG 场的离散散度 RMS | **3.75e-15** | 0（晶格上应到舍入） |
| 梯度算子对 `sin(kx)` 的相对误差 | 2.68 % | ≲5 %（一阶） |
| 图 Laplacian 对 `-∇²` 的相对误差 | 2.66 % | ≲5 % |
| 核矩 `b`（trace/dim） | 1.0105 | 2D 晶格一阶矩 |
| 正 ν 对能量的作用 | 单调下降 | 扩散 |

**误差判定**: 关键量（速度场相对 L2 误差）vs 解析基准 = **23.613 %**，**是否 < 3%：否**；
**是否 < 15%：否**。门配置与最细稳定配置都超差，故 **honest FAIL**。

## 4. 踩过的坑（三条根因，都是实测，不是推测）

1. **严格共轭对 `D Dᵀ` 在共位晶格上有棋盘零模。** 把 `projection_adjoint` 的算子
   （对称系数 `V_j p_i + V_i p_j`）上周期盒：锯齿场 `p=(-1)^{i+j}` 使每对相邻粒子 `p_i+p_j=0`，
   于是 `Dᵀp=0`、`D Dᵀ` 奇异，CG 沿零方向漂走——实测压力 **±4.36e19**、速度修正 **3.87e15**、
   `div_after/div_before = 1.9e16`。这就是本模块改用 M-矩阵（图）Laplacian 的原因
   （零空间只有常数）。
2. **核矩归一必须除维度。** `Σ_j -V_j r_ij·∇W_ij` 是一阶矩张量的**迹**（2D 晶格 = 2.021），
   直接拿它归一模化梯度会得到 **52 %** 误差；正确归一化因子是 trace/dim = **1.0105**，
   归一后梯度误差 2.68 %。同理，把 Laplacian 除以对角 `a_i` 会把二阶项（Laplacian 赖以存在的项）
   整体消掉，实测对光滑场返回 **ratio -1e-3（≈0）**——成对和本身已是 Cummins & Rudman 的一致离散。
3. **一阶 Euler + 大 CFL 的截断误差足以伪造"数值不稳定"。** `Co=0.2`、`dt=0.013` 时单步
   对流项误差 ≈ 4 %，第一步就 3.3 % 误差并逐级放大；换 Heun + `Co≤0.1` 后误差从 148 % 降到 23.6 %。

## 5. 局限与下一步（交给下一单的线索）

* 剩余 23.6 % 的两项来源：① `L` 与 `-D G` 是同一算子的**两个不同模板**，投影只能压掉约 20 %
  的离散散度（比值 0.79），残差每步再注入；② 能量被过度耗散（E/E_exact = 0.755），即
  成对 Laplacian 的有效粘性大于 ν。
* 可能的修法（按性价比）：(a) 交错/MAC 版周期投影（消除共位棋盘模态，`D` 与 `G` 天然共轭）；
  (b) 用 MLS 一阶一致算子对 + 一致性正则项（牺牲严格共轭换精度）；(c) 加粒子 shifting
  （仓内 `particle_shifting` 已有记录，8.83 %）后重测；(d) `dx=0.04` 的不稳定必须先在
  算子/时间积分层面解释清楚，否则分辨率阶梯不可信。
* 本单只跑到 `dx=0.04`（`dx=0.025` 需 4–5 min，被本机 120 s 短测门拦下，未递交集群）。

## 6. 选型后果（可执行）

`method_selection` 对涡类剖面的行为**不变**：仍 `WITHHELD`，且现在有"有记录但超差"的实测支撑。
`tests/test_isph_periodic.py::test_the_vortex_class_is_still_refused_by_the_selector` 把这条钉死：
任何一次"顺手修好"都不许在**没有新证据**的情况下把涡类剖面变成 `SELECTED`；
`test_committed_benchmark_record_is_an_honest_fail` 保证存储的工件不会被粉饰成 pass。

## 7. 复现

```bash
PYTHONPATH=src python benchmarks/isph_taylor_green/run.py \
    --out outputs/isph_taylor_green --configs "0.05:0.1,0.05:0.05,0.04:0.1"
PYTHONPATH=src python -m pytest -q tests/test_isph_periodic.py tests/test_method_selection.py
```
（首条 < 60 s、第二条 < 10 s，均在 <8 核 / <10 min 短测门内。）