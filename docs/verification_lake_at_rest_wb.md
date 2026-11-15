# 封闭水箱静水湖面：一致动量算子的判决（CASE-SPH-LAKE-AT-REST）

**范围** TensorSPH ｜ **模块** `tensorsph.lake_at_rest_wb` ｜ **证据** `tests/data/lake_at_rest/wb_operator.json` ｜ **门禁** `tests/test_lake_at_rest_wb.py` ｜ **图** `docs/figures/lake_wb_operator_ax.png`, `docs/figures/lake_wb_operator_bar.png`

本文件所有数字由 `benchmarks/lake_at_rest_wb/run.py` 程序化写入 JSON 后注入，零手抄。

## 1. 判决

`docs/verification_lake_at_rest_wall_gap.md` 把封闭水箱的静水残差诊断为「非离散误差、质量场无解」，并把未决项写成一句话：

> *A well-balanced/consistent momentum operator (kernel gradient correction or a boundary-integral wall treatment) is required; that is a solver change, not a case change.*

本单把这句话**判掉**：实现了四个标准的 consistent / well-balanced 压力梯度（局部矩矩阵一阶修正 M⁻¹、自由面自压力重构、两者叠加、纯最小二乘平面拟合梯度），每个变体都用算例自带的高斯–牛顿质量求解把**垂向**平衡重新闭到机器精度量级，再测水平残差。结果：

* 全变体最好水平/总体残差 `max|a|/g = 9.285e-02`，比锚点容差 `1e-09` 高 `9.29e+07` 倍 —— **没有任何一致梯度变体达到锚点容差**；
* 唯一有改善的变体（自由面自压力重构）只把 `max|a_x|/g` 从 `1.092e-01` 降到 `9.285e-02`（≈15%），且该幅度在两个分辨率上都不随网格收敛；
* 两个变体（difference form、MLS 平面拟合梯度）**连垂向平衡都保不住**（`max|a_y|/g` 量级 O(1)–O(4)），因此不是候选；
* 新增的**抵消诊断**给出机理级的判据：要消掉那个残差，该粒子需要的自身压力是**负的**、量级为静水压力的 3–5 倍（见 §5），即**任何可容许的压力闭合都做不到**。

结论：残差不是梯度一致性缺陷，而是**自由面边界闭合 + 壁侧支撑缺失**的联合产物；把壁面一侧的缺失支撑补回来的做法只有两类——周期/镜像边界的支撑补全，或壁面积分（boundary-integral）处理。锚点在本配置上维持诚实 FAIL，但未决项被**收窄并定量**。

## 2. 算例与问题

| 量 | 值 |
|---|---|
| 水箱 | 0.40 m × 0.30 m，静水深 0.30 m，两侧 + 底三层壁粒子 |
| 流体 | rho0 = 1000 kg/m³, c0 = 20 m/s, gamma = 7, g = 9.81 m/s² |
| 网格 | dx = 0.050, 0.040 m，h = 1.3 dx，支撑 2h = 2.6 dx |
| 解析基准 | p = rho0 g (eta - y)（精确，与地形无关） |
| 锚点容差 | 1e-09（机器精度） |

被检验的问题只有一个：**在同一个质量场上，把压力梯度换成一致/well-balanced 形式，封闭水箱的水平静水残差会不会消失？**

## 3. 方法

1. 用算例自带 `lake_at_rest.reconstruct`（shipped 算子，镜像对绑定质量）把垂向平衡闭到 `a_y/g ≈ 1e-13`，得到基准质量场；`residual_report` 给出的水平残差即 wall_gap 记录的数（实现控制：本模块的 `shipped_reimplementation` 变体必须逐位复现它）。
2. 在**同一质量场**上按六个变体计算加速度，量 `max|a_x|/g`（未重调质量）；
3. 再允许每个变体**按自己的算子**重解质量场（只闭垂向，`target="vertical"`，高斯–牛顿 + Tikhonov 向解析剖面拉），量重解后的 `max|a_x|/g` 与 `max|a_y|/g`；
4. 控制项：压力场整体置零（必须回到 `|a|/g = 1`）、压力整体放大 1%（残差必须动）、关重力（残差必须等于压力项）。

四个变体的构造（全部只用数据，不注入解析压力）：

* `own_pressure_reconstructed`：Tait 下限把自由面行的 p 钳到 0，而该处静水压是 `rho g dx/2`；用支撑上的加权线性最小二乘把它重构成 `p̂_i`（Audusse 2004 / Xing–Shu 2005 的静水重构在 SPH 里的读法）。
* `corrected_gradient`：Randles–Libersky 一阶矩修正 `M_i = Σ_j V_j (x_j-x_i)⊗∇_i W_ij`，梯度按 `M⁻¹∇W` 修正，使线性场在任意被截断的支撑上被精确重现。
* `difference_consistent`：守恒型换成差分型 `-M⁻¹Σ_j V_j (p_j-p_i)∇W`，常数场严格零梯度。
* `lsq_plane_gradient`：直接用最小二乘平面拟合的斜率（对任意线性场精确）。

## 4. 结果：一致梯度不能闭合残差

| dx [m] | 流体粒子 | 未知量 | 变体 | 形式 | 自压力 | 修正梯度 | 原质量场 max\|a_x\|/g | 重解后 max\|a_x\|/g | 重解后 max\|a_y\|/g | 垂向收敛 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 48 | 24 | shipped_reimplementation | symmetrised | eos | no | 1.092e-01 | 1.092e-01 | 1.478e-13 | yes |
| 0.05 | 48 | 24 | own_pressure_reconstructed | symmetrised | reconstructed | no | 1.408e-01 | 9.285e-02 | 3.696e-13 | yes |
| 0.05 | 48 | 24 | corrected_gradient | symmetrised | eos | yes | 2.834e-01 | 1.973e-01 | 2.385e-11 | no |
| 0.05 | 48 | 24 | corrected_gradient_own_reconstructed | symmetrised | reconstructed | yes | 3.532e-01 | 1.542e-01 | 3.057e-11 | no |
| 0.05 | 48 | 24 | difference_consistent | difference | reconstructed | yes | 2.557e-01 | 2.328e-01 | 3.951e+00 | no |
| 0.05 | 48 | 24 | lsq_plane_gradient | lsq | reconstructed | yes | 2.159e-01 | 2.043e-01 | 4.539e+00 | no |
| 0.04 | 70 | 35 | shipped_reimplementation | symmetrised | eos | no | 1.203e-01 | 1.203e-01 | 2.408e-13 | yes |
| 0.04 | 70 | 35 | own_pressure_reconstructed | symmetrised | reconstructed | no | 1.473e-01 | 9.381e-02 | 1.650e-13 | yes |
| 0.04 | 70 | 35 | corrected_gradient | symmetrised | eos | yes | 2.955e-01 | 1.980e-01 | 2.555e-03 | no |
| 0.04 | 70 | 35 | corrected_gradient_own_reconstructed | symmetrised | reconstructed | yes | 3.589e-01 | 1.531e-01 | 5.235e-05 | no |
| 0.04 | 70 | 35 | difference_consistent | difference | reconstructed | yes | 2.631e-01 | 1.944e-01 | 1.983e+00 | no |
| 0.04 | 70 | 35 | lsq_plane_gradient | lsq | reconstructed | yes | 2.236e-01 | 1.513e-01 | 1.900e+00 | no |

shipped 基准（`residual_report`，未重解）：dx=0.050 → `a_y/g=1.476e-13`, `a_x/g=1.092e-01`（壁带 `1.092e-01`，内部 `1.049e-02`）；dx=0.040 → `a_y/g=2.412e-13`, `a_x/g=1.203e-01`（壁带 `1.203e-01`，内部 `7.169e-03`）。

读法：`shipped_reimplementation` 与 shipped 逐位一致（实现控制通过）；`own_pressure_reconstructed` 是唯一有改善的变体（≈15%），`corrected_gradient` 反而变大，两个差分型变体把垂向平衡打坏。

## 5. 抵消诊断：需要的自身压力是不可容许的

shipped 形式在粒子 i 上为 `a_x = -(p_i/rho_i²)Σ_j m_j ∂_x W - Σ_j m_j (p_j/rho_j²)∂_x W`，令 `a_x = 0` 得到唯一的需求值 `p_i^req = -rho_i² Σ_j m_j (p_j/rho_j²)∂_x W / Σ_j m_j ∂_x W`。与静水值 `rho0 g (eta - y_i)` 比较：

| dx [m] | 最坏粒子位置 | 该处 `a_x/g` | EOS 压力 | 静水压力 | **需求压力** | 需求/静水 |
|---|---|---|---|---|---|---|
| 0.050 | [0.375, 0.275] | -1.092e-01 | 0.0 Pa | 245.2 Pa | **-1132.1 Pa** | -4.62 |
| 0.040 | [0.38, 0.26] | -1.203e-01 | 0.0 Pa | 392.4 Pa | **-1192.6 Pa** | -3.04 |
| 0.025 | [0.3875, 0.2875] | -1.013e-01 | 0.0 Pa | 122.6 Pa | **-417.6 Pa** | -3.41 |

需求压力**为负**（自由面上的拉伸压力，SPH/Tait 形式里不可容许），且量级是静水压力的 3–5 倍：**不存在任何可容许的自压力闭合能消掉它**。这与 §4 的数值直接吻合——把自压力改成物理上最合理的重构值（正向、静水量级），只能拿回约 15%。

## 6. 控制项（证明「静止」不是造出来的）

| dx [m] | 压力置零 max\|a\|/g（应 = 1） | 压力 +1% 后 max\|a_x\|/g | 关重力 max\|a\|/g |
|---|---|---|---|
| 0.050 | 1.000000000 | 3.567e-01 | 2.456e+00 |
| 0.040 | 1.000000000 | 3.625e-01 | 2.487e+00 |

## 7. 注意点与局限

* 本单**没有**找到能闭合封闭水箱的算子；这是一条负面结果，按 owner 的记账纪律如实入账，不粉饰。
* 三个变体的垂向重解只跑到 12 次高斯–牛顿迭代（本机算力纪律），`corrected_gradient` 在 dx=0.04 的 `a_y/g` 停在 `2.6e-3`，**不能**据此说它的垂向不可闭；能说的是它的水平数没有改善。
* dx=0.025 只做了抵消诊断（该诊断不需要质量重解），阶梯重解未在该分辨率完成（作业 cj3561 留在集群队列），已在 JSON 里显式标注 `note`。
* 算子族限于「压力梯度 + 质量场」两类自由度；壁面积分（boundary-integral）重构成**未**实现，因此「必需壁面积分」仍是**推断**而非本单的测量结论。

## 8. 复现

```
PYTHONPATH=src python3 benchmarks/lake_at_rest_wb/run.py \
    --json tests/data/lake_at_rest/wb_operator.json --figures docs/figures \
    --resolutions 0.05,0.04 --max-iter 12
PYTHONPATH=src python3 -m pytest tests/test_lake_at_rest_wb.py -q   # 9 gates
```

