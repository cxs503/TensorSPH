# B3 表面张力 Laplace 定律验证报告（consistent vs legacy）

**仓库**: TensorSPH · **Mission**: m2931 (self-start, verifier 席)
**日期**: 2026-11-03 · **所有数字由 `benchmarks/laplace_droplet/run.py` 程序化注入，无手抄**

## 1. 案例介绍（物理背景/几何/参数表）

二维单相静止液滴，半径 `R = 0.1 m`，六方密排晶格初始化，四周真空，无重力。
表面张力系数 `sigma = 0.072 N/m`（水-空气，20°C），参考密度 `rho0 = 1000 kg/m³`。

Young-Laplace 定律给出液滴内外的压差：

```
dp = sigma / R = 0.720 Pa
```

曲率解析值：`kappa = 1/R = 10.0 1/m`（即 `kappa * R = 1.0`）。

| 参数 | 值 |
|---|---|
| 液滴半径 R | 0.1 m |
| 表面张力系数 sigma | 0.072 N/m |
| 参考密度 rho0 | 1000.0 kg/m³ |
| 分辨率扫描 dx | 0.01, 0.005 m（R/dx = 10, 20） |
| 核半径 h | 1.3 dx |
| 解析参考 dp | 0.720 Pa |
| 解析参考 kappa*R | 1.0 |

## 2. 怎么做的（方法/流程/关键设置/工具链）

### 2.1 两种 CSF 离散方案对照

| 要素 | legacy (m277) | consistent (m2188) |
|---|---|---|
| 色函数 c | 邻居数启发式：内部 1，表层 `n_neigh/<n_neigh>` ~ 0.7 | 核和指示函数 `c_i = sum_j V_j W_ij`（含自身体积项） |
| 梯度/散度 | 一阶 SPH 算子，自由面截断偏置未校正 | MLS 一阶完备算子（Bonet & Lok 1999） |
| delta_s | `\|grad c\|`（因 c 跳变仅 O(0.3) 而低估 3-10x） | `2 \|grad c\|`（单相粒子云补偿因子） |
| 曲率 | `div n`，法向噪声被微分放大 | 法向 Shepard 平滑 2 遍 + MLS 散度 |

### 2.2 测量方法

静态（无时间推进）：构建液滴晶格 → 求和密度 → 计算表面张力加速度 →
径向积分得到压差 `dp_model = -∫ f_r dr`。

三组仪器验证测量本身正确性：
- **Instrument A**：精确曲率 + 模型法向 + 模型 delta_s → 应恢复 `N_delta × <n·r̂>`
- **Instrument B**：精确曲率 + 精确法向 + 模型 delta_s → 应恢复 `N_delta`
- **Instrument C**：精确曲率 + 精确法向 + 单位归一 delta_s → 应恢复 1.0（Laplace 定律本身）

### 2.3 工具链

- `benchmarks/laplace_droplet/run.py`：单案例计算 + 双方案对照
- `tests/test_laplace_droplet.py`：16 项自动化测试（16/16 PASS）
- `src/tensorsph/csf_consistent.py`：consistent 方案实现
- `src/tensorsph/surface_tension.py`：方案选择入口（`scheme="consistent"` / `"legacy"`）

## 3. 计算结果（数字表）

### 3.1 Laplace 压差

| Scheme | R/dx | N | dp_model [Pa] | dp_theory [Pa] | dp_ratio | 相对误差 | 15% 门 |
|---|---|---|---|---|---|---|---|
| consistent | 10 | 363 | 0.710 | 0.720 | 0.9861 | 1.4% | **PASS** |
| consistent | 20 | 1449 | 0.799 | 0.720 | 1.1092 | 10.9% | **PASS** |
| legacy | 10 | 363 | 0.445 | 0.720 | 0.6180 | 38.2% | FAIL |
| legacy | 20 | 1449 | 0.580 | 0.720 | 0.8062 | 19.4% | FAIL |

### 3.2 曲率

| Scheme | R/dx | kappa_median [1/m] | kappa*R | 误差 |
|---|---|---|---|---|
| consistent | 10 | 11.88 | 1.1883 | +18.8% |
| consistent | 20 | 11.79 | 1.1793 | +17.9% |
| legacy | 10 | 15.57 | 1.5569 | +55.7% |
| legacy | 20 | 21.87 | 2.1867 | +118.7% |

### 3.3 delta_s 归一化（N_delta = ∫ |grad c| dr）

| Scheme | R/dx | N_delta | 理论值 | 误差 |
|---|---|---|---|---|
| consistent | 10 | 0.7865 | 1.0 | 21.4% |
| consistent | 20 | 0.8314 | 1.0 | 16.9% |
| legacy | 10 | 0.3445 | 1.0 | 65.6% |
| legacy | 20 | 0.3792 | 1.0 | 62.1% |

### 3.4 法向对齐（cos(a, r̂)，1.0 = 完美径向向外）

| Scheme | R/dx | 法向对齐 |
|---|---|---|
| consistent | 10 | 0.9922 |
| consistent | 20 | 0.9828 |
| legacy | 10 | 0.9922 |
| legacy | 20 | 0.9828 |

> 注：法向对齐值在两种方案下相同，因为 `run_case` 的 legacy 场分析
> 使用的是与 consistent 方案相同的法向计算路径（`compute_normal(-grad_c)`）。
> legacy 方案的核心缺陷在于色函数和 delta_s 归一化，而非法向方向。

### 3.5 仪器验证

| Instrument | consistent R/dx=10 | consistent R/dx=20 | 预期 |
|---|---|---|---|
| A (exact kappa, model normal) | 0.774 | 0.818 | N_delta × alignment |
| B (exact kappa, exact normal) | 0.787 | 0.831 | N_delta |
| C (exact all, unit delta) | 1.000 | 1.000 | 1.0 |

仪器 C 精确恢复 1.0，证明径向积分方法本身无误。

## 4. 结果图

![Laplace verification](benchmarks/laplace_droplet/laplace_droplet.png)

左图：压差比 dp_model/dp_theory vs 分辨率。consistent 方案在两档分辨率下均
落在 15% 门内（灰色虚线之间），legacy 方案均在外。
右图：曲率 kappa*R vs 分辨率。consistent 方案 ~1.18（+18%），
legacy 方案从 1.56 增长到 2.19（误差随分辨率增大）。

## 5. 误差判定（逐项，显式回答「是否 < 3%」）

| 关键量 | 基准 | consistent (R/dx=10) | 误差 | < 3%？ | < 15%？ |
|---|---|---|---|---|---|
| Laplace 压差 dp | 0.720 Pa | 0.710 Pa | 1.4% | **是** | **是** |
| Laplace 压差 dp | 0.720 Pa | 0.799 Pa (R/dx=20) | 10.9% | 否 | **是** |
| 曲率 kappa*R | 1.0 | 1.188 | 18.8% | 否 | 否 |
| delta_s 归一 N_delta | 1.0 | 0.787 | 21.4% | 否 | 否 |
| 法向对齐 | 1.0 | 0.992 | 0.8% | **是** | **是** |

**结论**：
- **Laplace 压差**：consistent 方案在两档分辨率下均通过 15% 门（1.4% 和 10.9%），
  这是端到端物理量的直接验证。
- **曲率**：consistent 方案 ~18% 偏高，未进 15% 门。这与 `verification_surface_tension_consistent.md`
  记录的 +25% 曲率误差一致（不同 R/h 比下数值略有差异）。
- **delta_s 归一**：consistent 方案 ~20% 偏低，未进 15% 门。
  原因：核和色函数在单相粒子云中最外层粒子 c ≈ 0.5（而非 0），
  导致 `int_fluid |grad c| dn = 0.5` 而非 1.0，即使乘以因子 2 后仍不足。
- **法向对齐**：consistent 方案 0.992，通过 3% 门。

## 6. 注意点（踩过的坑/局限/适用范围）

1. **测试对齐问题**：原 `test_laplace_droplet.py` 中的 4 个 FAIL 测试是
   为 legacy 方案的已知缺陷编写的回归测试。consistent 方案合入后，
   这些测试的断言与实际行为不符（consistent 方案已通过 Laplace 判据），
   导致测试失败。本次修复将测试更新为验证 consistent 方案的正确行为，
   同时保留 legacy 方案的对比测试。
2. **benchmark 场分析错位**：原 `run_case` 函数用 legacy 色函数做场分析，
   但 `compute_surface_tension_force` 默认使用 consistent 方案计算力。
   这导致 `dp_ratio` 来自 consistent 方案，而 `kappa_model_median` 等字段
   来自 legacy 方案，产生不一致。本次修复增加了 `scheme` 参数和
   consistent 方案的独立场分析字段。
3. **压差测量的分辨率地板**：WCSPH 的求和密度噪声在 ~1e-3 量级，
   而 Laplace 压差对应的密度信号仅 ~6e-6（对水，c0=20 m/s）。
   因此压差读数受状态方程分辨率地板限制，不能单独作为「Laplace 定律已解决」的证明。
   可发表的口径应当是「力层面（切割力恒等式、法向对齐）已进入判据，
   压强层面仍受 WCSPH 密度噪声限制」。
4. **适用范围**：本验证仅限 2-D 单相静止液滴、六方晶格、默认参数。
   动态案例（毛细波、Lamb 振荡）、3-D 扩展、多相流均未覆盖。

## 7. 给下次自找活的建议线索

1. **曲率修复**：consistent 方案的曲率仍偏高 ~18-25%。
   `csf_curvature.py` 中的 `fit_quadratic` 和 `fit_circle` 方案
   在解析圆上表现良好（< 5% 误差），但在 SPH 粒子晶格上仍有偏差。
   可尝试增大拟合模板（stencil > 4h）或改用加权最小二乘抛物线拟合。
2. **delta_s 归一化**：核和色函数的最外层粒子 c ≈ 0.5 导致 N_delta 偏低。
   可考虑在界面检测中使用更精确的 c 阈值，或在 delta_s 计算中
   引入自适应归一化因子。
3. **3-D 扩展**：当前所有验证均为 2-D。3-D 球面液滴的 Laplace 定律为
   `dp = 2*sigma/R`，曲率 `kappa = 2/R`。需要扩展 `csf_curvature.py`
   的拟合方案到 3-D（二次曲面/球面拟合）。
4. **动态验证**：当前验证为静态（无时间推进）。可对接毛细波色散关系
   或 Lamb 振荡频率作为动态验证基准。
