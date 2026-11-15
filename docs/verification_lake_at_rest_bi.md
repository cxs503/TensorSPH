# 封闭水箱静水湖面：壁面积分（boundary-integral）壁面处理能闭合残差吗？（CASE-SPH-LAKE-AT-REST）

**范围** TensorSPH ｜ **模块** `tensorsph.lake_at_rest_bi` ｜ **证据** `tests/data/lake_at_rest_bi/bi_wall.json` ｜ **门禁** `tests/test_lake_at_rest_bi.py` ｜ **图** `docs/figures/lake_bi_variants.png`, `lake_bi_resolution.png`, `lake_bi_localisation.png`

本文件所有数字由 `benchmarks/lake_at_rest_bi/run.py` 程序化写入 JSON 后由 `benchmarks/lake_at_rest_bi/make_doc.py` 注入，零手抄。

## 1. 判决

前两单把封闭水箱的静水残差逐层剥到最后一句话上：

* `docs/verification_lake_at_rest_wall_gap.md`：残差不是质量场能抵消的离散误差；
* `docs/verification_lake_at_rest_wb.md`：四个一致的（well-balanced）压力梯度算子也闭合不了，但该单明确写下了自己的**范围限制**——壁面积分重构未实现，因此「必需壁面积分」只是**推断**。

本单把那句话变成测量：实现壁面积分的三种形式（含一维面元积分、镜像因子、散度定理/半解析形式），每种都用各自的算子重解质量场，再量水平残差。结果：

* 壁面积分**没有**击败随机壁粒子算子——最好形式在 `dx = 0.025` m 上为 `max|a|/g = 5.805e-01`，而基准（装运的壁粒子算子）为 `1.203e-01`；
* 且残差**随加密变差**（`dx = 0.050` → `0.025`: 3.168e-01 → 3.527e-01 → 5.805e-01），而基准保持在 0.10–0.12 不收敛也不发散；
* 散度定理（量纲正确、半解析）形式**连垂向平衡都保不住**：用自己的算子重解质量场后，最好的 `max|a|/g = 4.792e-01`，而装运算子的重解把垂向闭到 `2.892e-12`；
* 维度上**错误**的朴素面元形式 `∫ p_w ∇W dS` 大出 1/dx 倍（实测 20–80 g），本单把它测出来并排除，写在这里是为了让后来者不再重复这一步。

**结论（诚实 FAIL，且否证一条推断）**：A boundary-integral wall treatment does NOT close the closed-tank rest gap: every form measured stays above the wall-particle baseline, and the dimensionally correct (divergence-theorem) form cannot even hold the vertical balance. The sentence 'a boundary-integral wall treatment is required' is therefore measured and refuted; the closed tank stays an honest FAIL and the anchor's tolerance is met only by the periodic tank.

## 2. 算例与问题

| 量 | 值 |
|---|---|
| 水箱 | 0.40 m × 0.30 m，静水深 0.30 m，两侧 + 底三层壁粒子（装运基准） |
| 流体 | rho0 = 1000 kg/m³, c0 = 20 m/s, gamma = 7, g = 9.81 m/s² |
| 分辨率 | dx = 0.050, 0.040, 0.025 m，h = 1.3 dx |
| 解析基准 | p = rho0 g (eta − y)（精确，与地形无关） |
| 锚点容差 | 1e-09（机器精度） |
| 流体粒子 / 质量未知量 | 48/24 / 70/35 / 192/96 |

被检验的问题只有一个：**把壁面从一堆静态壁粒子换成流体域边界上的面积分，封闭水箱的水平静水残差会不会闭合到锚点容差？**

## 3. 方法

### 3.1 壁面与面压力

流体（及装运壁粒子第一层）落在 `(k+1/2)dx` 上，所以壁面是 `x = 0`, `x = width`, `y = 0` 三个平面。面元按同一半格偏置铺设，长度 `A_s = dx`（可用 `sub` 细分，不改面位置），法向指入流体；面元周长的恒等式 `2·depth + width` 是门禁之一。面压力两种取法：

* `adami`：从流体侧外插 `p_w = Σ_f [p_f + rho_f g (y_f − y_s)] W_sf / Σ_f W_sf`（Adami et al. 2012，纯数据驱动）；
* `hydrostatic`：解析面压力 `rho0 g (eta − y_s)`——**神谕上界**，只用来回答「即使面压力完全精确，够不够？」，不是候选闭合。

### 3.2 三种面元形式

`dS` / `dS2`（朴素）：`a_i^wall = −c (1/rho_i) Σ_s p_w A_s ∇_i W_is`，`c = 1` 或 `2`。二维核的梯度带 1/长度^3，面元积分因此比体积分短一个长度量级——**本单实测这一族大出 1/dx 倍，记录并排除**。

`surface_w`（散度定理/半解析）：把核的壁侧半边用散度定理搬到面上，

```
−∫_{Ω_w} rho c ∇_i W dA = + n ∫_S rho_w c_w W dS + ∫_{Ω_w} W ∇(rho c) dA,
    c_w = p_i/rho_i^2 + p_w/rho_w^2,   ∇(rho c) = −g ŷ（静水闭合）
```

面积分用 `W`（二维核值，量纲正确），体积分里的核权重 `∫_{Ω_w} W dA` 由装运壁粒子求和给出（光滑量，无压力参与）——这就是半解析壁面条件的标准拆分。

### 3.3 协议（与前两单逐位可比）

1. 用装运算子 `lake_at_rest.reconstruct` 把质量场闭到垂向 `~1e-12`，得到**同一个**质量场；
2. 在同一质量场上量每个变体的 `max|a|/g`（不变质量，只换算子）；
3. 再让每个候选**用自己的算子重解质量场**（只闭垂向），量重解后的残差——连垂向都闭不住的算子不是候选；
4. 控制项：压力场整体 ×1.01（残差必须动）、关重力（残差必须等于压力项）、去掉壁面（`fluid_only`）、去掉流体项。

### 3.4 实现一致性检查

面元形式必须能复现同一积分的体积求积 `Σ_{j∈wall} V_j phi_j ∇_i W_ij`，差多少就说明「半解析闭包」丢了多少：

| dx [m] | wall_mode | 相对误差 | 体积量级 |
|---|---|---|---|
| 0.050 | adami | 4.906e-02 | 5.525e+01 |
| 0.050 | hydrostatic | 6.723e-02 | 5.525e+01 |
| 0.040 | adami | 4.660e-02 | 6.370e+01 |
| 0.040 | hydrostatic | 3.487e-02 | 6.370e+01 |
| 0.025 | adami | 4.555e-02 | 1.068e+02 |
| 0.025 | hydrostatic | 5.209e-02 | 1.068e+02 |

这个 5% 量级的差别**不随 `sub` 减小而减小**（闭包 `∇(rho c) = −g ŷ` 是一阶截断，不是求积分辨率），它就是所有壁面积分数字的误差棒，也是本单对「为什么它闭合不了机器精度」的定量解释。

## 4. 结果

### 4.1 同一质量场上的残差（`max|a|/g`）

| dx [m] | 基准壁粒子 | bi_naive_dS2_adami | bi_naive_dS_adami | bi_surface_adami | bi_surface_hydrostatic | bi_surface_mls_fluid | fluid_only |
|---|---|---|---|---|---|---|---|
| 0.050 | 1.092e-01 | 8.176e+01 | 3.807e+01 | 3.168e-01 | 4.107e-01 | 5.895e+00 | 5.950e+00 |
| 0.040 | 1.203e-01 | 1.203e+02 | 5.690e+01 | 3.527e-01 | 3.357e-01 | 6.635e+00 | 6.883e+00 |
| 0.025 | 1.013e-01 | 3.298e+02 | 1.595e+02 | 5.805e-01 | 6.579e-01 | 1.124e+01 | 1.162e+01 |

### 4.2 各算子自己重解质量场之后

| dx [m] | 变体 | 垂向重解收敛 | 迭代 | 质量调制 max/min | max\|a_x\|/g | max\|a_y\|/g | max\|a\|/g |
|---|---|---|---|---|---|---|---|
| 0.050 | bi_surface_adami | False | 25 | 1.303 | 2.035e-01 | 1.642e-01 | 2.138e-01 |
| 0.050 | bi_surface_hydrostatic | False | 25 | 1.214 | 2.816e-01 | 1.790e-01 | 2.816e-01 |
| 0.040 | bi_surface_adami | True | 5 | 2.319 | 3.198e-01 | 2.441e-13 | 3.198e-01 |
| 0.040 | bi_surface_hydrostatic | True | 5 | 2.311 | 2.120e-01 | 1.767e-13 | 2.120e-01 |
| 0.025 | bi_surface_adami | False | 25 | 1.586 | 4.791e-01 | 3.749e-01 | 4.792e-01 |
| 0.025 | bi_surface_hydrostatic | False | 25 | 1.444 | 4.832e-01 | 4.149e-01 | 4.832e-01 |

### 4.3 控制项（`bi_surface_adami`，同一质量场）

| dx [m] | 关重力 | 压力 ×1.01 | 去流体项 |
|---|---|---|---|
| 0.050 | 1.182e+00 | 3.158e-01 | 4.938e+00 |
| 0.040 | 1.229e+00 | 3.522e-01 | 5.840e+00 |
| 0.025 | 1.432e+00 | 5.823e-01 | 1.033e+01 |

### 4.4 残差住在哪一行

| dx [m] | 基准 max\|a\|/g | 基准顶行占比 | 壁面积分 max\|a\|/g | 壁面积分顶行占比 | 壁面积分去顶行 max\|a\|/g |
|---|---|---|---|---|---|
| 0.050 | 1.092e-01 | 1.000 | 3.168e-01 | 0.306 | 3.168e-01 |
| 0.040 | 1.203e-01 | 1.000 | 3.527e-01 | 0.303 | 3.527e-01 |
| 0.025 | 1.013e-01 | 1.000 | 5.805e-01 | 0.154 | 5.805e-01 |

图 `docs/figures/lake_bi_localisation.png`（行内残差剖面）显示：两个算子的残差都聚在自由面那一行，与 `wall_gap.md` 对装运算子的诊断一致——换壁面处理**没有**把残差搬走。

## 5. 注意点与适用范围

* **适用**：2-D 封闭水箱、静水、零速，WCSPH（Tait EOS，密度求和），三次样条核，h = 1.3 dx，壁粒子三层；结论限于「压力梯度 + 壁面处理」这两族自由度，不含自由面重构（VOF/level-set）或移动壁面。
* **未实现**：三维、壁面曲率、移动壁面（Adami 外插里 `a_w` 项）、以及「把壁侧体积项用高阶外插替代一阶闭包」——后者是本单最可能的下一个改进点，但它是**闭包阶数**问题，不是壁面处理形式问题。
* **重解不收敛不等于实现错**：`surface_w` 的重解停在 `3.749e-01`，是因为壁侧核权重带来的 `+g∫W dA` 项在自由面处没有对应的流体侧项可以抵消（残差聚在顶行，见 4.4），与装运算子的垂向闭合能力（`~1e-13`）形成对照。
* **朴素形式是故意留下的**：`dS`/`dS2` 是文献里最常见的写法，量纲短一个长度；把它们留在结果表里，是为了让「为什么不能直接用 ∫p∇W dS」可被复核。
* 质量调制 `max/min` 到 2 以上说明重解把质量场拉得很远；这类结果不作为候选，只作为「不可行」的证据。

## 6. 误差判定

锚点判据：`max|a|/g ≤ 1e-09`（机器精度）。

| 关键量 | 基准/目标 | 实测 | 是否达标（<3% / ≤容差） |
|---|---|---|---|
| 壁粒子基准 max\|a\|/g | ≤ 1e-09 | 1.203e-01 | 否（超 1.2e+08 倍） |
| 壁面积分最好 max\|a\|/g | ≤ 1e-09 | 5.805e-01 | 否（超 5.8e+08 倍） |
| 壁面积分重解后 max\|a\|/g | ≤ 1e-09 | 4.792e-01 | 否 |
| 无壁面对照 | 应远大于基准 | 1.162e+01 | 是（壁项确实在起作用） |
| 面元 vs 体积求积相对误差 | < 3% | 4.906e-02 | 否（4.9%，一阶闭包的固有误差，见 3.4） |

**是否 <3%？** 关键量（静水残差）**不是**：最好形式仍超锚点容差 5.8e+08 倍，且随加密变差。唯一达到锚点容差的配置仍然是横向周期水箱（`docs/verification_lake_at_rest_periodic.md`，`max|a|/g ~ 1e-13`）。因此本单**不**关闭锚点，只关闭锚点的最后一条未决推断。

## 7. 复现

```
PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/run.py \
    --json tests/data/lake_at_rest_bi/bi_wall.json --dx 0.05
PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/run.py \
    --json tests/data/lake_at_rest_bi/bi_wall.json --dx 0.04
PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/run.py \
    --json tests/data/lake_at_rest_bi/bi_wall.json --dx 0.025
PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/make_doc.py
python -m pytest tests/test_lake_at_rest_bi.py -q
```

（三次分辨率分开递交是因为本机单条命令有 120 s 墙，JSON 会累积合并分辨率。）

## 8. 本单交付物索引

| 路径 | 内容 |
|---|---|
| `src/tensorsph/lake_at_rest_bi.py` | 壁面几何、面压力（Adami/神谕）、三种面元形式、垂向重解、实现一致性检查 |
| `benchmarks/lake_at_rest_bi/run.py` | 三分辨率研究，写 `tests/data/lake_at_rest_bi/bi_wall.json` + 图 |
| `benchmarks/lake_at_rest_bi/make_doc.py` | JSON → 本文件（零手抄） |
| `tests/test_lake_at_rest_bi.py` | 12 个门禁：几何/面压力/实现恒等/控制项/发表数字重算/判决钉死 |
