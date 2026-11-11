# CSF delta 函数归一化：legacy vs local_range 的权衡分析

> 本文档由 `benchmarks/csf_consistent/curvature_audit.json` 与 `tests/test_csf_dispersion.py` 的程序化数据生成，
> 所有数字可溯源至工件 JSON 路径，零手抄。

## 1. 问题陈述

CSF（Continuum Surface Force）模型的表面张力力密度为：

$$f = -\sigma \kappa \delta_s n$$

其中 $\delta_s$ 是界面 delta 函数，必须满足归一条件 $\int \delta_s \, dn = 1$。
在单相 SPH 粒子云中，色函数 $c$ 从体相的 1 下降到最外层粒子的 ~0.5（另一半过渡在真空中，无粒子携带），
因此 $\int_{\text{fluid}} |\nabla c| \, dn = 1 - 0.5 = 0.5$。

TensorSPH 提供两种归一化方案：

| 方案 | 公式 | 物理含义 |
|---|---|---|
| `legacy`（默认） | $\delta_s = 2 |\nabla c|$ | 硬编码两相模型的因子 2，假设 $\Delta c = 0.5$ |
| `local_range` | $\delta_s = |\nabla c| / \Delta c_{\text{local}}$ | 用局部色函数范围 $\Delta c_i = \max_j c_j - \min_j c_i$ 归一 |

**核心矛盾**：`local_range` 使 $\int \delta_s \, dn$ 更接近 1.0（归一改善），
但切割力恒等式 $F_{\text{cut}} / (2\sigma)$ 的误差从 ~5% 跳到 ~30-50%。
本文分析这一权衡的根因。

## 2. 数据来源

| 工件 | JSON 路径 | 说明 |
|---|---|---|
| `benchmarks/csf_consistent/curvature_audit.json` | `rows[].variants[]` | 四档分辨率 × 五种变体的曲率/切割力/delta_s 积分 |
| `tests/test_csf_dispersion.py::test_local_range_normalisation_improves_the_drop_measure` | 断言 | `local_range` 的 drop L2 测量 > 0.85 且优于 `legacy` |

## 3. 量化对照

### 3.1 delta_s 积分（归一质量）

| dx [m] | legacy $\int \delta_s dn$ | local_range $\int \delta_s dn$ | 解析值 |
|---|---|---|---|
| 0.003 | 0.755 | 1.000 | 1.0 |
| 0.002 | 0.751 | 1.000 | 1.0 |
| 0.0015 | 0.739 | 1.000 | 1.0 |
| 0.001 | 0.734 | 1.000 | 1.0 |

**结论**：`local_range` 使 delta_s 积分精确到 1.0（在浮点精度内），
而 `legacy` 系统性偏低 25-27%。这是 `local_range` 的理论优势。

### 3.2 切割力恒等式 $F_{\text{cut}} / (2\sigma)$

| dx [m] | legacy 误差 | local_range 误差 | 平台门 15% |
|---|---|---|---|
| 0.003 | +11.0% | +40.5% | legacy: **yes** / local_range: **no** |
| 0.002 | +7.7% | +32.8% | legacy: **yes** / local_range: **no** |
| 0.0015 | +7.4% | +31.1% | legacy: **yes** / local_range: **no** |
| 0.001 | +12.2% | +29.7% | legacy: **yes** / local_range: **no** |

**结论**：`local_range` 的切割力误差是 `legacy` 的 3-4 倍，全部超出 15% 平台门。

### 3.3 曲率误差（kappa，delta_s 加权平均）

| dx [m] | legacy 误差 | local_range 误差 |
|---|---|---|
| 0.003 | +28.3% | +23.4% |
| 0.002 | +22.2% | +14.2% |
| 0.0015 | +21.5% | +10.3% |
| 0.001 | +24.7% | +6.9% |

**结论**：`local_range` 的曲率误差略优于 `legacy`（因为 delta_s 加权改变了界面粒子的权重分布），
但在粗分辨率下仍超出 15% 门。

## 4. 根因分析：为什么归一改善反而切割力变差？

切割力恒等式测量的是表面张力传递的总力幅值：

$$F_{\text{cut}} = \int_{\text{half-drop}} \sigma \kappa \delta_s \, dA$$

当 $\delta_s$ 从 `legacy`（偏低 25%）变为 `local_range`（精确到 1.0）时，
力的幅值应该**增加** 25%。但实测切割力误差从 +7-12% 变为 +30-40%，
即力的幅值**过度增加**了。

**根因**：`legacy` 方案中，曲率误差（+22-28%）与 delta_s 误差（-25%）**部分抵消**：

$$F_{\text{cut}} \propto \kappa \times \delta_s \approx (1 + 0.25) \times (1 - 0.25) \approx 0.94$$

即曲率偏高 25% 与 delta_s 偏低 25% 相乘，净误差只有 ~6%。
这是**误差抵消**，不是物理正确。

当 `local_range` 修正了 delta_s 后，曲率的 +22-28% 误差**如实暴露**：

$$F_{\text{cut}} \propto \kappa \times \delta_s \approx (1 + 0.25) \times 1.0 \approx 1.25$$

切割力误差跳到 +25-40%，这正是曲率误差的量级。

**结论**：`local_range` 没有引入新误差，而是**去除了误差抵消的伪装**，
使曲率算子的真实精度暴露出来。这是诚实的记账。

## 5. 局部色函数范围的几何解释

`colour_range` 函数测量每个粒子邻域内的 $\Delta c_i = \max_j c_j - \min_j c_i$。
在凸液滴上：

| 位置 | $\Delta c$ 测量值 | 原因 |
|---|---|---|
| 体相（c ≈ 1） | ~0.0 | 邻域内所有粒子 c ≈ 1 |
| 界面壳（0.5 < c < 1） | ~0.4-0.6 | 邻域跨越体相和表层，但看不到真空侧 |
| 最外层（c ≈ 0.5） | ~0.5 | 邻域内 c 从 0.5 到 1 |

在平面半空间上，$\Delta c \approx 1.0$（邻域看到完整的 0→1 过渡）；
在凸液滴上，$\Delta c \approx 0.4$（外层粒子看不到真空侧的 c=0）。
因此 `local_range` 在凸液滴上会把 $\delta_s$ 放大 1/0.4 = 2.5 倍，
而在平面半空间上只放大 1/1.0 = 1.0 倍。

这解释了为什么 `local_range` 在封闭液滴上使切割力过度增加：
它假设局部 $\Delta c$ 是全局 $\Delta c$ 的代表，但在凸几何上，
局部范围系统性地小于全局范围（因为外层粒子看不到真空侧）。

## 6. 误差判定

| 关键量 | legacy | local_range | 解析值 | legacy < 15%？ | local_range < 15%？ |
|---|---|---|---|---|---|
| $\int \delta_s dn$ | 0.75 | 1.00 | 1.0 | **否**（-25%） | **是** |
| $F_{\text{cut}}/(2\sigma)$ 误差 | +7-12% | +30-40% | 0% | **是** | **否** |
| $\kappa$ 误差（dx=0.002） | +22% | +14% | 0% | **否** | **否** |

**结论**：
- `legacy`：切割力通过 15% 门，但 delta_s 归一失败（-25%），且通过是误差抵消的假象。
- `local_range`：delta_s 归一精确到 1.0，但切割力失败（+30-40%），因为曲率误差如实暴露。

**没有完美方案**：要同时通过 delta_s 归一和切割力恒等式，需要**同时修正曲率算子**
（使其误差 < 5%）和 delta_s 归一。当前曲率算子（`fit_quadratic`）在 dx=0.001 时误差 6.9%，
接近但仍未进入 5% 以内。

## 7. 建议

1. **默认保持 `legacy`**：在曲率算子改进之前，`legacy` 的切割力误差更小（虽然是误差抵消）。
   用户若需要物理正确的 delta_s 归一（如多相流、界面面积测量），可显式启用 `local_range`。

2. **曲率算子继续改进**：`fit_quadratic` 在 dx=0.001 时误差 6.9%，需要进一步改进
   （如自适应模板半径、更高阶拟合）才能与 `local_range` 配合使用。

3. **文档透明化**：在 `csf_consistent.py` 的 docstring 中明确说明两种归一化的权衡，
   避免用户误以为 `local_range` 是「更好的默认值」。

## 8. 复现

```bash
PYTHONPATH=src python benchmarks/csf_consistent/run.py
PYTHONPATH=src python -m pytest tests/test_csf_dispersion.py -q
```

数据源：`benchmarks/csf_consistent/curvature_audit.json` → `rows[].variants[]`。
