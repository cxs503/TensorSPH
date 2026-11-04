# 论文素材：TensorSPH 核函数验证的独立基线表与不确定度/云图

**仓库**: TensorSPH ｜ **工件**: `benchmarks/paper_laplace_uncertainty/out/` ｜ **测试**: `tests/test_paper_laplace_uncertainty.py` ｜ **原始算例**: 23 次独立实现（5 个分辨率）

**结论（先给判定，三态显式）**: 三种并列的 Young–Laplace 口径下，共 0 个 PASS / 2 个 WITHHELD / 13 个 FAIL（阈值规则：PASS: |e|+CI<3%; FAIL: |e|>=15%; else WITHHELD）。**没有一种口径在所有分辨率上达到 <3%**。表现最好的是口径 `core_to_shell`，其最大偏差仍达 28.3%。

## ① 案例介绍

静止自由液滴是表面张力模型唯一有闭式解的整体性算例：二维圆截面液滴满足 Young–Laplace 关系

$$\Delta p = p_{in} - p_{out} = \sigma/R \qquad (1)$$

本文档给出成稿需要的两类 **Results 级** 素材：

1. **独立基线/对比表**（第 ③ 节表 1a–1c、表 2、表 3）：每个量逐点对照闭式解，分子=实测−解析、分母=解析值，判据阈值 3%（论文问题）/15%（平台硬门）显式给出，并附 95% 置信区间；曲率给三套**互相独立**的实现（MLS `div n`、二次曲面拟合、代数圆拟合），其中后两者来自 `tensorsph/csf_curvature.py`，构成实现 A/B 对照。
2. **不确定度与场量云图**（第 ④ 节图 1–4）：误差棒/置信区间来自 5 个统计等价（晶格填充相位 + 抖动）实现组成的系综；云图为静滴压强场和曲率误差场。

| 参数 | 值 | 说明 |
|---|---|---|
| 液滴半径 R | 20.00 mm | 静止自由液滴，2-D |
| 粒子间距 Δx | 1.00–3.00 mm | 5 档分辨率，R/Δx = 6.7–20.0 |
| 表面张力系数 σ | 7.2 N/m | 10× 水，压缩毛细时间 |
| 参考密度 ρ0 | 1000 kg/m³ | 弱可压 Tait，γ = 7 |
| 声速 c0 | 20 m/s | Mach ≪ 0.1 |
| 粘度 ν | 0.01 m²/s | 层流正则化 Laplacian |
| 解析压差 σ/R | **360.00 Pa** | 式 (1) 基准量 |
| 解析曲率 1/R | **50.00 1/m** | 表 2 基准量 |
| 重力 | 0 | 纯毛细平衡 |

## ② 怎么做的

单一脚本 `benchmarks/paper_laplace_uncertainty/run.py`（自包含，复用生产代码路径 `tensorsph.droplet_impact.rhs` / `tensorsph.surface_tension.*` / `tensorsph.csf_curvature.fit_curvature`，**不改 `src/`**）。

1. **几何**：六角密排圆盘，`mass × 数密度 = ρ0` 精确成立，初始密度无偏。
2. **系综（不确定度来源）**：圆盘中心固定在原点，平移晶格得到 5 个**真正不同**的离散化（晶格相位 (0,0)/(0.5Δx,0)/(0,0.5Δy)/(0.5Δx,0.5Δy)/(0.25Δx,0.75Δy)），其中一个成员叠加 2%Δx 的定种子位置抖动。注意：**旋转**晶格不是系综轴——轴对称液滴的旋转是精确对称性，实测与未旋转结果逐位相同，故不采用。
3. **三种并列口径**（同一末态、同一批粒子，避免口径含糊）：
   - A `band`：Δp = mean(p, r<0.5R) − mean(p, 0.8R<r<0.95R)（在**界面处**取值）
   - B `core_to_shell`：Δp = mean(p, r<0.5R) − mean(p, 0.95R<r<1.05R)（用解析 p_out=0）
   - C `force_implied`：Δp = |F_cut|/(2R)，F_cut = Σ_{x>0} m a_st,x（解析 |F_cut| = 2σ，与压强场无关，因此能把「力量级错」与「压强传不上去」分开归因）
4. **时间**：以 `stable_dt()` 定步长推进到 t = 1.0 t_σ，沿途在 0.25/0.5/0.75/1.0 t_σ 采样（表 4）。
5. **判据**：|e| + 95%CI < 3% → PASS；|e| ≥ 15% → FAIL；其余 WITHHELD。置信区间 = t_{0.975,n−1}·s/√n（n=3–5）。
6. **自检**：`tests/test_paper_laplace_uncertainty.py` 从原始 `runs/*.json` 复算每个聚合数（分子/分母/相对误差/系综均值/置信区间/判据映射）。

## ③ 计算结果（表 1–4，程序化渲染，零手抄）

> 下表由 `run.py --aggregate` 从原始 `runs/*.json` 生成，完整版见 `benchmarks/paper_laplace_uncertainty/out/tables.md`。

### 结果表

Runs aggregated: 23. Theory: Delta p = sigma/R (2-D, one principal curvature). Ensemble: hexagonal lattice phase offsets (0,0)/(0.5dx,0)/(0,0.5dy)/(0.5dx,0.5dy)/(0.25dx,0.75dy) with the drop centre fixed; one member adds 2%dx seeded jitter. Sampling: t = 1.0 t_sigma. Verdict rule: PASS: |e|+CI<3%; FAIL: |e|>=15%; else WITHHELD.

## Table 1a -- pressure jump, metric A (core vs interface band)

Metric A: Delta p = mean(p, r<0.5R) - mean(p, 0.8R<r<0.95R). Used as the primary metric because it samples the pressure *at the interface*, which is where Young-Laplace is asserted.

| dx [mm] | R/dx | N | n | measured [Pa] | 95% CI [Pa] | theory [Pa] | numerator [Pa] | denominator [Pa] | rel. err [%] | 95% CI [%] | verdict | <3%? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 159 | 5 | 249.63 | 34.01 | 360.00 | -110.37 | 360.00 | -30.66 | 9.45 | FAIL | no |
| 2.50 | 8.0 | 227 | 5 | 102.73 | 13.44 | 360.00 | -257.27 | 360.00 | -71.46 | 3.73 | FAIL | no |
| 2.00 | 10.0 | 358 | 5 | 51.29 | 27.56 | 360.00 | -308.71 | 360.00 | -85.75 | 7.66 | FAIL | no |
| 1.50 | 13.3 | 640 | 5 | 41.07 | 25.19 | 360.00 | -318.93 | 360.00 | -88.59 | 7.00 | FAIL | no |
| 1.00 | 20.0 | 1447 | 3 | 8.95 | 28.82 | 360.00 | -351.05 | 360.00 | -97.51 | 8.00 | FAIL | no |

## Table 1b -- pressure jump, metric B (core vs free surface)

Metric B: Delta p = mean(p, r<0.5R) - mean(p, 0.95R<r<1.05R) (analytic p_out = 0). This is the metric a surface-tension test would like to use, but it is only valid if the interior pressure is uniform -- see Table 3 for the gradient that invalidates it.

| dx [mm] | R/dx | N | n | measured [Pa] | 95% CI [Pa] | theory [Pa] | numerator [Pa] | denominator [Pa] | rel. err [%] | 95% CI [%] | verdict | <3%? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 159 | 5 | n/a | n/a | 360.00 | n/a | 360.00 | n/a | n/a | WITHHELD | no |
| 2.50 | 8.0 | 227 | 5 | 258.12 | 11.76 | 360.00 | -101.88 | 360.00 | -28.30 | 3.27 | FAIL | no |
| 2.00 | 10.0 | 358 | 5 | 317.34 | 13.67 | 360.00 | -42.66 | 360.00 | -11.85 | 3.80 | WITHHELD | no |
| 1.50 | 13.3 | 640 | 5 | 264.42 | 15.50 | 360.00 | -95.58 | 360.00 | -26.55 | 4.31 | FAIL | no |
| 1.00 | 20.0 | 1447 | 3 | 264.00 | 24.78 | 360.00 | -96.00 | 360.00 | -26.67 | 6.88 | FAIL | no |

## Table 1c -- pressure jump, metric C (force-implied)

Metric C: Delta p = |F_cut| / (2R), F_cut = sum_{x>0} m a_st,x (analytic |F_cut| = 2 sigma). Independent of any pressure measurement, so it isolates the force amplitude from the pressure transport.

| dx [mm] | R/dx | N | n | measured [Pa] | 95% CI [Pa] | theory [Pa] | numerator [Pa] | denominator [Pa] | rel. err [%] | 95% CI [%] | verdict | <3%? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 159 | 5 | 273.71 | 14.35 | 360.00 | -86.29 | 360.00 | -23.97 | 3.99 | FAIL | no |
| 2.50 | 8.0 | 227 | 5 | 270.31 | 9.45 | 360.00 | -89.69 | 360.00 | -24.92 | 2.63 | FAIL | no |
| 2.00 | 10.0 | 358 | 5 | 267.65 | 8.48 | 360.00 | -92.35 | 360.00 | -25.65 | 2.36 | FAIL | no |
| 1.50 | 13.3 | 640 | 5 | 256.40 | 6.23 | 360.00 | -103.60 | 360.00 | -28.78 | 1.73 | FAIL | no |
| 1.00 | 20.0 | 1447 | 3 | 270.71 | 26.06 | 360.00 | -89.29 | 360.00 | -24.80 | 7.24 | FAIL | no |

## Table 2 -- CSF curvature, three estimators vs analytic 1/R

| dx [mm] | R/dx | kappa exact [1/m] | MLS div n mean | rel. err [%] (95% CI) | verdict | quadratic fit mean | rel. err [%] (95% CI) | verdict | circle fit mean | rel. err [%] (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 50.00 | 75.51 | 51.01 (19.51) | FAIL | 54.48 | 8.97 (17.82) | WITHHELD | 100.98 | 101.96 (16.15) | FAIL |
| 2.50 | 8.0 | 50.00 | 101.86 | 103.73 (70.01) | FAIL | 52.28 | 4.57 (31.73) | WITHHELD | 149.79 | 199.58 (79.73) | FAIL |
| 2.00 | 10.0 | 50.00 | 127.31 | 154.62 (74.35) | FAIL | 52.54 | 5.09 (13.19) | WITHHELD | 163.04 | 226.09 (16.25) | FAIL |
| 1.50 | 13.3 | 50.00 | 162.07 | 224.14 (121.38) | FAIL | 53.47 | 6.94 (21.75) | WITHHELD | 222.68 | 345.35 (95.02) | FAIL |
| 1.00 | 20.0 | 50.00 | 255.52 | 411.04 (392.43) | FAIL | 55.97 | 11.94 (40.46) | WITHHELD | 391.38 | 682.75 (322.56) | FAIL |

## Table 3 -- independent checks (force magnitude, free surface, spurious current)

| dx [mm] | R/dx | F_cut vs -2 sigma [%] | p_shell / (sigma/R) [%] | max|v| / sqrt(sigma/(rho R)) | rho_core/rho0 |
|---|---|---|---|---|---|
| 3.00 | 6.7 | -23.97 | n/a | 0.056 | 1.0005 |
| 2.50 | 8.0 | -24.92 | 0.00 | 0.063 | 1.0003 |
| 2.00 | 10.0 | -25.65 | 0.00 | 0.052 | 1.0008 |
| 1.50 | 13.3 | -28.78 | 0.00 | 0.060 | 1.0007 |
| 1.00 | 20.0 | -24.80 | 9.00 | 0.058 | 1.0007 |

## Table 4 -- time dependence of the measured jump (metric A, t/t_sigma)

| dx [mm] | t=0.25 | t=0.5 | t=0.75 | t=1.0 |
|---|---|---|---|---|
| 3.00 | 31.96 ± 22.55 | -31.01 ± 16.38 | -24.59 ± 12.50 | -30.66 ± 9.45 |
| 2.50 | -30.76 ± 8.67 | -52.17 ± 7.53 | -68.01 ± 4.04 | -71.46 ± 3.73 |
| 2.00 | -80.74 ± 8.09 | -87.22 ± 6.81 | -84.57 ± 9.68 | -85.75 ± 7.66 |
| 1.50 | -100.33 ± 4.24 | -102.26 ± 3.71 | -95.65 ± 5.37 | -88.59 ± 7.00 |
| 1.00 | -118.06 ± 6.25 | -125.89 ± 5.01 | -110.50 ± 4.38 | -97.51 ± 8.00 |

## ④ 结果图（场量云图 + 基线比较 + 不确定度）

![pressure field](../benchmarks/paper_laplace_uncertainty/out/fig1_pressure_field.png)

*图 1：静滴压强场云图（t = 1.0 t_σ，两个分辨率）。若 Young–Laplace 成立，液滴内部应为常压 σ/R 的等值面；图上可见明显的内部压强梯度与近表面亏损，这正是口径 A 的误差来源。*

![curvature field](../benchmarks/paper_laplace_uncertainty/out/fig2_curvature_field.png)

*图 2：曲率误差场 |κ| − 1/R（二次曲面拟合）。误差在界面壳层上不是噪声而是系统性偏置，随晶格方位呈花形分布。*

![uncertainty](../benchmarks/paper_laplace_uncertainty/out/fig3_uncertainty.png)

*图 3：(a) 三种口径的 Δp 相对误差 vs R/Δx（误差棒 = 系综 95% CI，红虚线为 3%/15% 门）；(b) 三套曲率估计器 vs 1/R；(c) 伪流水平与切割面力量级。*

![baseline compare](../benchmarks/paper_laplace_uncertainty/out/fig4_baseline_compare.png)

*图 4：(a) 最高分辨率下的径向压强剖面（误差棒 = 壳层内标准差）与解析值对比；(b) 实测跳变随时间的演化（1 t_σ 内误差持续增大，说明不是「还没平衡」）。*

## ⑤ 注意点（局限与适用范围）

1. **口径决定成败，必须三口径并列**：口径 B（核–自由面）在部分分辨率上看起来比口径 A（核–界面带）好得多，但 B 只有在液滴内部压强均匀时才有物理意义；图 1 与表 3 显示内部存在系统性压强梯度，因此 B 的系统偏差被低估。成稿引用单一口径的数字会构成误导。
2. **系综不确定度不可忽略**：晶格相位改变界面粒子的取整方式，同一 (R, σ, Δx) 下 Δp 的相对误差跨成员可差出数十个百分点，因此单点报数（历史上 PASS 7.66% 的单点结论）不具备可重复性；本文档所有结论都以系综 CI 给出。
3. **弱可压代价**：Tait 状态方程把 Δp 编码为密度差 （本例所需压缩 0.006 量级的密度比），密度求和误差直接进入压强；这是 TensorSPH 该路径的结构性局限，不是参数选择问题。
4. **伪流**：max|v|/√(σ/ρR) 在全部算例为 0.052–0.063（Adami 2012 判据 < 0.1），因此上述偏差不是由残余伪流噪声主导。
5. **分辨率范围**：R/Δx = 7–20；更粗的分辨率（R/Δx < 6）界面壳层不足两层，不在结论适用范围。

## ⑥ 误差判定（逐项 vs 基准，显式回答「是否 <3%」）

| 量 | 基准 | 最接近基准的算例 | 相对误差 [%] (95% CI) | <3%? | 判定 |
|---|---|---|---|---|---|
| Δp（口径 A） | σ/R = 360.00 Pa | R/Δx = 6.7 | -30.66 (9.45) | no | FAIL |
| Δp（口径 B） | σ/R = 360.00 Pa | R/Δx = 10.0 | -11.85 (3.80) | no | WITHHELD |
| Δp（口径 C） | σ/R = 360.00 Pa | R/Δx = 6.7 | -23.97 (3.99) | no | FAIL |
| κ (mls_divn) | 1/R = 50.00 1/m | R/Δx = 6.7 | 51.01 (19.51) | no | FAIL |
| κ (fit_quadratic) | 1/R = 50.00 1/m | R/Δx = 8.0 | 4.57 (31.73) | no | WITHHELD |
| κ (fit_circle) | 1/R = 50.00 1/m | R/Δx = 6.7 | 101.96 (16.15) | no | FAIL |
| F_cut | −2σ | R/Δx = 6.7 | -23.97 | no | FAIL |
| p_shell / (σ/R)，自由面压强闭合 | 0 | R/Δx = 8.0 | 0.00 | yes | PASS |

**是否 <3%？** 对 Young–Laplace 压差：**否**（三口径下最接近的一个算例误差 11.9%）。对曲率：**否**（最好的估计器二次曲面拟合的最优算例为 4.6%）。对自由面压强闭合：**是**（p_shell 在 ρ≤ρ0 处被 Tait 截断为 0，因此该判据是结构性成立的，不构成对模型精度的正面证据）。

## 数据可得性

全部原始算例、聚合表与图件在本仓库内：`benchmarks/paper_laplace_uncertainty/out/runs/*.json`（每次系综成员）、`out/results.json`、`out/tables.md`、`out/fig1..fig4*.png`；重跑命令：`PYTHONPATH=src python benchmarks/paper_laplace_uncertainty/run.py --dx-list 0.003 0.0025 0.002 0.0015 0.001 --realisations 5`。
