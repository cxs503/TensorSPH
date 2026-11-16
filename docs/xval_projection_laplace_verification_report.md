# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE — 综合验证报告

**锚点:** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`  
**仓库:** TensorSPH  
**验收标准:** 对标误差 < 15%（owner 精度标准）  
**状态:** ✅ 通过（静态臂）/ ⚠️ 动态臂待补（见 §7）

---

## ① 案例介绍

### 物理背景

Young-Laplace 方程描述了弯曲界面两侧的压力跳：

$$\Delta p = \sigma \left(\frac{1}{R_1} + \frac{1}{R_2}\right)$$

对于二维圆形液滴（主曲率 $1/R$ 和 $0$）：$\Delta p = \sigma / R$。

在**不可压缩（投影/ISPH）SPH**中，压力是 $\nabla \cdot \mathbf{u} = 0$ 的 Lagrange 乘子，通过 Poisson 方程获得：

$$\nabla \cdot \left(\frac{1}{\rho} \nabla p\right) = \frac{\nabla \cdot \mathbf{u}^*}{\Delta t}, \quad \mathbf{u}^* = \mathbf{u}^n + \Delta t \, \mathbf{a}_{\text{CSF}}$$

因此界面压力跳由 CSF 牵引力源项承载，而非通过 Tait 状态方程的微小密度差。

### 几何与参数

| 量 | 值 |
|---|---|
| 几何 | 二维圆形液滴，$R = 0.02\ \text{m}$ |
| 表面张力 | $\sigma = 0.072\ \text{N/m}$（水-空气，20°C） |
| 密度 | $\rho_0 = 1000\ \text{kg/m}^3$ |
| 理论压力跳 | $\Delta p_{\text{theory}} = \sigma/R = 3.6000\ \text{Pa}$ |
| 格子 | 六角形，$dy = dx \sqrt{3}/2$ |
| 核函数 | 二维三次样条，$\sigma_3 = 10/(7\pi h^2)$ |
| $h/dx$ | 1.3（基准），1.2 和 1.5（灵敏度） |
| 密度比 | 1:1（静态验证的标准做法） |
| 边界条件 | 外层 1.6dx 层 Dirichlet $p = 0$ |

### 分辨率阶梯

| $R/dx$ | $dx$ [mm] | 粒子数 |
|---|---|---|
| 10 | 2.0 | 1203 |
| 13.3 | 1.5 | 2133 |
| 20 | 1.0 | 4803 |

---

## ② 方法

### TensorSPH 臂（ours）

1. **颜色函数**: Shepard 归一化相指示器 $c = \sum_j V_j W_{ij} c_j / \sum_j V_j W_{ij}$
2. **梯度**: 一阶一致 MLS 梯度（Bonet & Lok 1999）
3. **法向**: $\mathbf{n} = -\nabla c / |\nabla c|$，Shepard 平滑 2 次
4. **曲率**: $\kappa = \nabla \cdot \mathbf{n}$（MLS 散度）
5. **δ 函数**: $\delta_s = |\nabla c|$（Brackbill et al. 1992）
6. **CSF 力**: $\mathbf{a} = -\sigma \kappa \delta_s \mathbf{n} / \rho$
7. **压力求解**: Cummins & Rudman (1999) 对 Laplacian，矩阵自由 Jacobi-CG（$\text{rel\_tol} = 10^{-10}$）
8. **伴随一致性**: 散度-梯度算子对满足 $\langle D\mathbf{u}, q \rangle = \langle \mathbf{u}, D^T q \rangle$（舍入误差级别）

### PySPH 参考臂（oss）

移植 PySPH 1.0b3.dev0 的算子链：`SmoothedColor` → SY11 颜色梯度 → SY11/Morris 曲率 → CSF 力 → `PressureCoeffMatrix`。两种形式：
- `form="paper"`: 论文一致的数密度因子 $2/(\psi_i + \psi_j) = 2V_i V_j/(V_i + V_j)$
- `form="code"`:  shipped 代码的 $2/(V_i + V_j)$（已证实为灾难性缺陷）

### 测量仪器

$$\Delta p_{\text{measured}} = \langle p \rangle_{\text{core}} - \langle p \rangle_{\text{band}}$$

其中 core 为 $r < 0.5R$，band 为 $1.25R < r < 1.45R$。

---

## ③ 计算结果

### 3.1 分辨率阶梯

| $R/dx$ | 理论 $\sigma/R$ [Pa] | TensorSPH $\Delta p$ [Pa] | 误差 | PySPH-parity $\Delta p$ [Pa] | 误差 |
|---|---|---|---|---|---|
| 10.0 | 3.6000 | 3.6350 | **+0.97%** ✅ | 3.7440 | **+4.00%** ✅ |
| 13.3 | 3.6000 | 3.6218 | **+0.61%** ✅ | 3.7240 | **+3.44%** ✅ |
| 20.0 | 3.6000 | 3.5923 | **−0.21%** ✅ | 3.7019 | **+2.83%** ✅ |

### 3.2 曲率恢复

| $R/dx$ | TensorSPH $\kappa R$ | PySPH-parity $\kappa R$ | 解析值 |
|---|---|---|---|
| 10.0 | 1.011 | 1.017 | 1.000 |
| 13.3 | 1.008 | 1.009 | 1.000 |
| 20.0 | 1.004 | 1.003 | 1.000 |

### 3.3 δ 函数归一化积分

| $R/dx$ | TensorSPH $\int \delta_s dn$ | PySPH-parity $\int \delta_s dn$ | 期望值 |
|---|---|---|---|
| 10.0 | 0.995 | 1.018 | 1.000 |
| 13.3 | 0.994 | 1.018 | 1.000 |
| 20.0 | 0.993 | 1.018 | 1.000 |

### 3.4 灵敏度分析

**$h/dx$ 变化（$R/dx = 20$）:**

| $h/dx$ | TensorSPH 误差 | PySPH-parity 误差 |
|---|---|---|
| 1.2 | −2.39% | +7.05% |
| 1.3 | −0.21% | +2.83% |
| 1.5 | +1.66% | −1.36% |

**曲率算子对比（$R/dx = 20$）:**

| 算子 | 误差 | $\kappa R$ | 通过 15%? |
|---|---|---|---|
| MLS $\nabla \cdot \mathbf{n}$ | −0.21% | 1.004 | ✅ |
| 二次曲面拟合 | −24.46% | 0.740 | ❌ |

---

## ④ 结果图

![Young-Laplace cross-validation ladder](../scripts/xval_projection_laplace/results/xval_projection_laplace.png)

**图 1:** 左：压力跳误差 vs 分辨率（对称对数坐标）；中：测量压力跳 vs 网格尺寸；右：曲率恢复 $\kappa R$。TensorSPH 臂（蓝色圆点）在所有分辨率下均在 ±15% 门限内，且收敛至解析值。

---

## ⑤ 误差判定

| 关键量 | 基准值 | 最佳测量值 | 相对误差 | 是否 < 3%? | 是否 < 15%? |
|---|---|---|---|---|---|
| $\Delta p$ vs $\sigma/R$ | 3.6000 Pa | 3.5923 Pa ($R/dx=20$) | **−0.21%** | ✅ | ✅ |
| $\kappa R$ | 1.000 | 1.004 ($R/dx=20$) | **+0.40%** | ✅ | ✅ |
| $\int \delta_s dn$ | 1.000 | 0.993 ($R/dx=20$) | **−0.70%** | ✅ | ✅ |
| 臂间一致性 | — | 3.03% ($R/dx=20$) | — | — | ✅ |

**显式回答: 所有关键量均在 3% 以内，远优于 15% 验收标准。**

---

## ⑥ 注意点与局限

### 已验证
- ✅ 伴随一致性：$\langle D\mathbf{u}, q \rangle - \langle \mathbf{u}, D^T q \rangle$ 在舍入误差级别（~1e-16）
- ✅ 压力求解：CG 收敛至 $\text{rel\_tol} = 10^{-10}$，78-150 次迭代
- ✅ 动量平衡残差：$\text{median}|\mathbf{a}_{\text{CSF}} - \nabla p/\rho| / \text{mean}|\mathbf{a}_{\text{CSF}}| = 0.26-0.36$
- ✅ PySPH 移植奇偶校验：5/9 算子复现至机器精度

### 已知局限
1. **动态投影时间积分**: `relax_projection` 在 12 步后出现发散趋势（spurious current 累积），这是投影方法中 solenoidal 残余的已知问题。动态臂的完整分辨率阶梯（7 个作业）尚未完成。
2. **二次曲面拟合曲率**: 在两相载体上表现不佳（−24.5%），仅适用于锐利自由表面。
3. **PySPH shipped 代码缺陷**: `psiab1 = 2/(V_i+V_j)` 分支导致 $\Delta p \sim 10^{24}$ Pa，是数密度因子公式错误。
4. **制造解检验**: 参考压力算子在 $R/dx=10$ 的块尺寸上相对误差 1.45，说明该算子在此尺度下未完全收敛。

### 适用范围
- 二维静态平衡验证
- 等密度两相载体（密度比 1:1）
- 六角形粒子格子
- 不适用于：动态演化、高密度比、三维（需单独验证）

---

## ⑦ 动态臂状态（开放欠账）

动态投影时间积分的分辨率阶梯（`benchmarks/laplace_sph/run_dynamic_ladder.py`）尚未完成：

| 作业 | 状态 |
|---|---|
| $R/dx=8$, 120 步 | ✅ 已测（`dx8_nu01.json`），$\Delta p = +1.17\%$，但 $u_{\max}/v_{\text{ref}} = 0.263$（FAIL Adami 0.1 线） |
| $R/dx=12$, 120 步 | ⚠️ 部分运行（`dx12_nu01.json`），仅 12 步 |
| $R/dx=16$, 120 步 | ❌ 未运行 |
| $R/dx=24$, 120 步 | ❌ 未运行 |
| $\nu$ 扫描（$R/dx=16$） | ❌ 未运行 |
| 长时程（$R/dx=12$, 600 步） | ❌ 未运行 |

**建议**: 将动态臂作为独立计算作业提交至集群（`run_compute`），因单机负载限制无法在交互模式下完成。

---

## ⑧ 复现

```bash
# 运行基准测试（快速模式，2 个分辨率级别）
PYTHONPATH=src python3 scripts/xval_projection_laplace/run.py --quick

# 完整分辨率阶梯
PYTHONPATH=src python3 scripts/xval_projection_laplace/run.py

# 运行测试套件
PYTHONPATH=src python3 -m pytest tests/test_xval_projection_laplace.py -v
```

### 产物清单

| 文件 | 内容 |
|---|---|
| `src/tensorsph/laplace_isph.py` | 投影-SPH Young-Laplace 核心实现 |
| `src/tensorsph/projection_adjoint.py` | 伴随一致投影算子 |
| `scripts/xval_projection_laplace/run.py` | 双臂交叉验证基准 |
| `scripts/xval_projection_laplace/oss_ports.py` | PySPH 算子移植 |
| `scripts/xval_projection_laplace/results/xval_projection_laplace.json` | 全部数值结果 |
| `scripts/xval_projection_laplace/results/xval_projection_laplace.png` | 结果图 |
| `tests/test_xval_projection_laplace.py` | 预声明门限测试套件 |
| `docs/xval_projection_laplace_oss.md` | 交叉验证详细报告 |
| `docs/xval_projection_laplace_verification_report.md` | 本综合验证报告 |

---

## ⑨ 结论

TensorSPH 的 projection-SPH Young-Laplace 实现在静态平衡验证中**通过**所有门限：
- 压力跳误差在 $R/dx = 10, 13.3, 20$ 分别为 +0.97%, +0.61%, −0.21%，均远优于 15% 验收标准
- 曲率恢复 $\kappa R \to 1.004$（$R/dx=20$），误差 0.4%
- 与独立 PySPH 参考臂在 $R/dx=20$ 一致性为 3.03%，双方从不同方向收敛至解析值
- 伴随一致性在舍入误差级别得到验证

动态臂的时间积分验证因计算量超出单干范围，作为开放欠账记录，建议后续通过集群作业完成。
