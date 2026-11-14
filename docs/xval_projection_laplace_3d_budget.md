# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE 第5轮：三维误差预算（投影 SPH + Young-Laplace）

> 本页由 `benchmarks/hollow_sphere_3d/make_doc_knobs.py` 从 `benchmarks/hollow_sphere_3d/results_knobs/results_knobs.json` 生成，所有数字程序化注入，无手抄。

## ① 案例介绍

**物理背景**：表面张力在弯曲界面两侧产生压差，即 Young-Laplace 律；三维球形界面是$\Delta p = 2\sigma/R$（两个主曲率半径都等于 R），液膜（空心球）为 $\Delta p = 2\sigma(1/R_{in}+1/R_{out})$。二维圆是退化情形（第二主曲率为零），因此**只有三维球才能判别“把 $2\sigma/R$ 算成 $\sigma/R$”的因子 2 缺陷**。

**本案例的定位**：锚点 `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` 前四轮分别是①开源复现对拍（2D）、②二维误差预算、③well-balanced 投影、④动态平衡；第②轮的文档明确写着「this error-budget audit has not been run in 3-D」。本轮就把这份预算做在三维球上。

**几何与参数**（表内为无量纲晶格单位，长度按 $\Delta x$ 缩放，物理量按 $\sigma/R$ 标定）：

| 参数 | 值 |
|---|---|
| 表面张力 $\sigma$ | 0.072 N/m（水-空气 25 ℃） |
| 密度 $\rho_0$ | 1000.0 kg/m³（气相载体同密度填充，静力学不需要密度比） |
| 液相球半径 $R/\Delta x$ | 4 … 12（工作点 8） |
| 载体 / Dirichlet 球 | $R + pad$，pad = 2.5 … 8 $\Delta x$，外 1.6 $\Delta x$ 层 $p=0$ |
| 核函数 | 三次样条，$h/\Delta x$ = 1.2 … 1.5 |
| Bo / We | 0 / 0（静力学） |

## ② 方法

投影（不可压）SPH：压力不作为密度的函数，而是散度约束的 Lagrange 乘子：

$$a_{CSF} = -\frac{\sigma\kappa\delta_s n}{\rho},\quad \kappa=\nabla\cdot n,\quad \delta_s=|\nabla c|,\qquad -\nabla^2 p = -\rho\,\nabla\cdot a_{CSF},\quad p=0\ \text{(外载体层)}$$

算子全部是仓内已发布的 3D MLS 算子（`tensorsph.threed.hollow_sphere`：`_mls_gradient` / `_mls_divergence` / `pressure_poisson3`），本轮**没有新增离散格式**，只新增了（i）若干可测旋钮与（ii）一个缺陷修复。

**旋钮扫描设计**（工作点：$R/\Delta x=8$、pad=4$\Delta x$、$h/\Delta x=1.3$、cubic、界面位于原点；每次只动一个旋钮）：

| 旋钮 | 取值 | 性质 | 预期 |
|---|---|---|---|
| 界面在晶胞内的位置（translation） | 7 个偏移 | **物理不变量** | 跳变不应变化 |
| 晶格排布（cubic / fcc） | 2 种 | **物理不变量** | 跳变不应变化 |
| Dirichlet 标距（pad） | 2.5 / 3 / 4 / 6 / 8 $\Delta x$ | 数值旋钮 | 影响小、有界 |
| 核支撑（$h/\Delta x$） | 1.2 / 1.3 / 1.4 / 1.5 | 数值旋钮 | 影响小、有界 |
| 曲率路线 | 解析给定 / 自洽 $\nabla\cdot n$ | 方法学选择 | 自洽路线误差更大 |
| 分辨率 | $R/\Delta x$ = 4 … 12 | 收敛性 | 单调趋近解析值 |
| 空心球（两界面） | 4/10、6/14、8/18 | 物理 | 两界面叠加 |
| 负对照 | $\sigma=0$ | 判据有效性 | 必须无压差 |

**闸门在数字产生前声明**（owner 精度标准 15 %；3 % 为以往各轮自设的“可发表线”）：单构型 |err| < 15 %；平移/晶格/标距/核支撑 摆动 < 5 %；$R/\Delta x$=12 处 |err| < 3 %；空心球 < 5 %；负对照 |Δp| < 1e-06 Pa。

## ③ 计算结果

### 3.1 主线：球面 $\Delta p = 2\sigma/R$ 的分辨率阶梯

| $R_{out}/\Delta x$ | $R_{out}/h$ | lattice | curv. | pad/dx | $\Delta p$/Pa | $\Delta p_{theory}$/Pa | err % | flux err % | $\kappa$ err % | $\int\delta_s dn$ | $N_p$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 3.08 | cubic | analytic | 4 | 0.0348 | 0.0360 | -3.40 | -10.79 | -0.01 | 0.8970 | 2109 |
| 6 | 4.62 | cubic | analytic | 4 | 0.0237 | 0.0240 | -1.23 | -2.63 | -0.12 | 0.9761 | 4169 |
| 8 | 6.15 | cubic | analytic | 4 | 0.0179 | 0.0180 | -0.69 | -3.86 | -0.05 | 0.9623 | 7153 |
| 10 | 7.69 | cubic | analytic | 4 | 0.0142 | 0.0144 | -1.16 | -2.58 | -0.06 | 0.9754 | 11513 |
| 12 | 9.23 | cubic | analytic | 4 | 0.0119 | 0.0120 | -1.04 | -2.85 | -0.07 | 0.9729 | 17077 |
| 4 | 3.08 | cubic | mls | 4 | 0.0402 | 0.0360 | +11.68 | -10.27 | +0.57 | 0.8970 | 2109 |
| 6 | 4.62 | cubic | mls | 4 | 0.0250 | 0.0240 | +4.04 | -3.04 | -0.55 | 0.9761 | 4169 |
| 8 | 6.15 | cubic | mls | 4 | 0.0186 | 0.0180 | +3.38 | -3.04 | +0.81 | 0.9623 | 7153 |
| 10 | 7.69 | cubic | mls | 4 | 0.0146 | 0.0144 | +1.14 | -2.18 | +0.35 | 0.9754 | 11513 |
| 12 | 9.23 | cubic | mls | 4 | 0.0121 | 0.0120 | +0.63 | -2.21 | +0.58 | 0.9729 | 17077 |

### 3.2 物理不变量 A：界面在晶胞内的位置（translation）

| $R_{out}/\Delta x$ | offset/dx | lattice | curv. | pad/dx | $\Delta p$/Pa | $\Delta p_{theory}$/Pa | err % | flux err % | $\kappa$ err % | $\int\delta_s dn$ | $N_p$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | [0.0, 0.0, 0.0] | cubic | analytic | 4 | 0.0179 | 0.0180 | -0.69 | -3.86 | -0.05 | 0.9623 | 7153 |
| 8 | [0.5, 0.0, 0.0] | cubic | analytic | 4 | 0.0178 | 0.0180 | -0.83 | -4.11 | -0.07 | 0.9603 | 7224 |
| 8 | [0.0, 0.5, 0.0] | cubic | analytic | 4 | 0.0178 | 0.0180 | -0.83 | -4.11 | -0.07 | 0.9603 | 7224 |
| 8 | [0.0, 0.0, 0.5] | cubic | analytic | 4 | 0.0178 | 0.0180 | -0.83 | -4.11 | -0.07 | 0.9603 | 7224 |
| 8 | [0.5, 0.5, 0.0] | cubic | analytic | 4 | 0.0178 | 0.0180 | -1.14 | -2.40 | -0.09 | 0.9777 | 7248 |
| 8 | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | cubic | analytic | 4 | 0.0178 | 0.0180 | -0.90 | -2.49 | -0.04 | 0.9760 | 7250 |
| 8 | [0.5, 0.5, 0.5] | cubic | analytic | 4 | 0.0178 | 0.0180 | -1.24 | -1.74 | -0.04 | 0.9833 | 7208 |

摆动（max−min）= **0.55 %**（闸门 5 %），曲率误差摆动 0.05 %。

### 3.3 物理不变量 B：晶格排布 cubic vs fcc（含缺陷 D1 回归）

| $R_{out}/\Delta x$ | $R_{out}/h$ | lattice | curv. | pad/dx | $\Delta p$/Pa | $\Delta p_{theory}$/Pa | err % | flux err % | $\kappa$ err % | $\int\delta_s dn$ | $N_p$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 6.15 | cubic | analytic | 4 | 0.0179 | 0.0180 | -0.69 | -3.86 | -0.05 | 0.9623 | 7153 |
| 8 | 6.15 | fcc | analytic | 4 | 0.0178 | 0.0180 | -1.32 | -3.04 | -0.04 | 0.9704 | 14401 |

摆动 = **0.63 %**（闸门 5 %）。修复前 fcc 一列的误差是 **−50.66 %**（见 ⑤ 缺陷 D1）。

### 3.4 数值旋钮：Dirichlet 标距 与 核支撑

| $R_{out}/\Delta x$ | $R_{out}/h$ | lattice | curv. | pad/dx | $\Delta p$/Pa | $\Delta p_{theory}$/Pa | err % | flux err % | $\kappa$ err % | $\int\delta_s dn$ | $N_p$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 6.15 | cubic | analytic | 2.5 | 0.0169 | 0.0180 | -6.30 | +4.56 | -0.04 | 1.0464 | 4945 |
| 8 | 6.15 | cubic | analytic | 3 | 0.0176 | 0.0180 | -2.47 | +0.20 | -0.05 | 1.0030 | 5575 |
| 8 | 6.15 | cubic | analytic | 4 | 0.0179 | 0.0180 | -0.69 | -3.86 | -0.05 | 0.9623 | 7153 |
| 8 | 6.15 | cubic | analytic | 6 | 0.0180 | 0.0180 | +0.24 | -3.97 | -0.05 | 0.9613 | 11513 |
| 8 | 6.15 | cubic | analytic | 8 | 0.0181 | 0.0180 | +0.36 | -3.97 | -0.05 | 0.9613 | 17077 |

| $R_{out}/\Delta x$ | $R_{out}/h$ | lattice | curv. | pad/dx | $\Delta p$/Pa | $\Delta p_{theory}$/Pa | err % | flux err % | $\kappa$ err % | $\int\delta_s dn$ | $N_p$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 6.67 | cubic | analytic | 4 | 0.0182 | 0.0180 | +1.04 | -3.44 | -0.06 | 0.9667 | 7153 |
| 8 | 6.15 | cubic | analytic | 4 | 0.0179 | 0.0180 | -0.69 | -3.86 | -0.05 | 0.9623 | 7153 |
| 8 | 5.71 | cubic | analytic | 4 | 0.0174 | 0.0180 | -3.31 | -3.40 | -0.04 | 0.9667 | 7153 |
| 8 | 5.33 | cubic | analytic | 4 | 0.0174 | 0.0180 | -3.34 | -2.86 | -0.03 | 0.9720 | 7153 |

标距摆动（≥3 $\Delta x$）= **2.83 %**，核支撑摆动 = **4.38 %**。

### 3.5 曲率路线与空心球（两曲率半径）

| $R_{out}/\Delta x$ | $R_{out}/h$ | lattice | curv. | pad/dx | $\Delta p$/Pa | $\Delta p_{theory}$/Pa | err % | flux err % | $\kappa$ err % | $\int\delta_s dn$ | $N_p$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 6.15 | cubic | analytic | 4 | 0.0179 | 0.0180 | -0.69 | -3.86 | -0.05 | 0.9623 | 7153 |
| 8 | 6.15 | cubic | mls | 4 | 0.0186 | 0.0180 | +3.38 | -3.04 | +0.81 | 0.9623 | 7153 |

| $R_{out}/\Delta x$ | $R_{out}/h$ | lattice | curv. | pad/dx | $\Delta p$/Pa | $\Delta p_{theory}$/Pa | err % | flux err % | $\kappa$ err % | $\int\delta_s dn$ | $N_p$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 7.69 | cubic | analytic | 4 | 0.0492 | 0.0504 | -2.41 | -9.61 | -0.07 | 0.9624 | 11513 |
| 10 | 7.69 | cubic | mls | 4 | 0.0527 | 0.0504 | +4.63 | -18.71 | +4.40 | 0.9624 | 11513 |
| 14 | 10.77 | cubic | analytic | 4 | 0.0342 | 0.0343 | -0.31 | -3.99 | -0.09 | 0.9800 | 24405 |
| 14 | 10.77 | cubic | mls | 4 | 0.0357 | 0.0343 | +4.06 | -4.09 | +0.36 | 0.9800 | 24405 |
| 18 | 13.85 | cubic | analytic | 4 | 0.0260 | 0.0260 | -0.10 | -3.36 | -0.06 | 0.9820 | 44473 |
| 18 | 13.85 | cubic | mls | 4 | 0.0267 | 0.0260 | +2.87 | -2.57 | +0.16 | 0.9820 | 44473 |

### 3.6 负对照

$\sigma=0$：$\Delta p$ = **0.000e+00 Pa**（理论 0），$N_p$ = 7153，判据有效（压差不是求解器的偏置）。

## ④ 结果图

见 `benchmarks/hollow_sphere_3d/figures/knob_budget_3d.png`（本报告内嵌图 1）：
(a) 物理不变量（平移 + 晶格）；(b) 数值旋钮（标距、核支撑）；(c) 分辨率两条曲率路线 vs 解析 $2\sigma/R$；(d) 空心球两界面。

## ⑤ 注意点（踩坑与局限）

**缺陷 D1（本轮修复，属实质缺陷）**：`threed/hollow_sphere.py` 的 `lattice="fcc"` 选项把第二个（穿插）副本叠加到立方晶格上（最近邻间距 $\Delta x/\sqrt2$），**粒子数翻倍**，但每个粒子仍按 `rho0·dx³` 记账 → 载体携带的质量/体积翻倍，颜色壳的归一化 $\int\delta_s\,dn=1$ 被破坏，Young-Laplace 跳变掉到 **−50.66 %**，而 DVD 判据以外没人会察觉（CG 收敛、残差正常）。修复：每个格点只记 **一个格点的体积**（fcc 为 $dx^3/2$），fcc 误差回到 **-1.32 %**（闸门 3 %）。`tests/test_xval_projection_laplace_3d_budget.py` 用几何不变量（体积和、最近邻间距、$\int\delta_s dn$）钉住，而不是只钉数字。

**局限 1：仍是“填充载体”的静力学仪器**。气相用同密度粒子填满，所以本页度量的是 $\sigma/R$ 算子链，不是两相流动；密度比、真实自由面 BC、$\delta$-ISPH 时间推进都不在本文范围内。

**局限 2：$h/\Delta x$ 与 pad 都不是物理量**。它们能动 2–4 % 而两个物理不变量只动 0.1–1 %，说明残余误差是**离散化选择效应**，不是毛细力的各向异性——这与第②轮二维结论一致。

**局限 3：“解析曲率”一列把 $\kappa$ 当成已知量**，只隔离投影算子；真实预报能力要看 `mls`（$\nabla\cdot n$）那一列，误差明显更大，不能混为一谈。

**局限 4：只到 3 个方向轴上的平移**，未做一般旋转（立方晶格 + 通用旋转需要重新建格点，且会改变载体边界采样，会把“晶格旋转”与“边界采样”两个效应混在一起）。

**局限 5**：`pad = 2.5 Δx` 一列是刻意的操作性下限，用于展示标距效应会随距离衰减，不作为方法精度引用。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 实测 | 误差 | < 3 %? | < 15 %? |
|---|---|---|---|---|---|
| 球面跳变 $\Delta p$，$R/\Delta x=12$，解析曲率 | $2\sigma/R$ | 0.0119 Pa | -1.04 % | 是 | 是 |
| 球面跳变，$R/\Delta x=12$，自洽 $\nabla\cdot n$ | $2\sigma/R$ | 0.0121 Pa | +0.63 % | 是 | 是 |
| 物理不变量：平移摆动 | 0 | — | 0.55 % | 是 | 是 |
| 物理不变量：晶格摆动（cubic/fcc） | 0 | — | 0.63 % | 是 | 是 |
| 数值旋钮：标距摆动（≥3 $\Delta x$） | 0 | — | 2.83 % | 是 | 是 |
| 数值旋钮：核支撑摆动 | 0 | — | 4.38 % | 否 | 是 |
| 空心球 $\Delta p = 2\sigma(1/R_{in}+1/R_{out})$，最差构型 | 闭式解 | — | +4.63 % | 否 | 是 |
| 负对照 $\sigma=0$ | 0 Pa | 0.00e+00 Pa | 0（机器零） | 是 | 是 |

**全扫描最差单构型误差** = **+11.68 %**（owner 闸门 15 %）（构型：R/dx=4 mls，lattice=cubic，curvature=mls，$R_{out}/\Delta x$=4，pad/dx=4）。注意这个最差构型落在**粗网格端**（$R/\Delta x=4$ 的自洽曲率路线），不是标距或核支撑造成的，也不是工作点的精度；工作点一侧（$R/\Delta x\ge 8$ 且 pad$\ge$3$\Delta x$）的最差构型为 +4.63 %（shell 4/10 mls）。
**是否 < 3 %？** 关键量在 **$R/\Delta x\ge10$ 且解析曲率** 下为是（-1.04 %）；**自洽 $\nabla\cdot n$ 路线在 $R/\Delta x=12$ 处为 +0.63 %**，若超过 3 % 则如实写“否”，不得用解析曲率那一列代答。

## ⑦ 复现

```bash
PYTHONPATH=src python3 benchmarks/hollow_sphere_3d/audit_knobs.py          # 全量扫描 → JSON
PYTHONPATH=src python3 benchmarks/hollow_sphere_3d/make_doc_knobs.py      # JSON → 本页 + 图
PYTHONPATH=src python3 -m pytest tests/test_xval_projection_laplace_3d_budget.py -q
```

构件：`audit_knobs.py`（扫描器）、`results_knobs/results_knobs.json`（全部数字）、`figures/knob_budget_3d.png`（图）、`tests/test_xval_projection_laplace_3d_budget.py`（闸门 + 缺陷 D1 回归）、`src/tensorsph/threed/hollow_sphere.py`（体积记账修复 + `center` 旋钮）。
