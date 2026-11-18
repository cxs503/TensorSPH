# CASE-SPH-OSC-DROP-FREQ · 振荡液滴频率验证报告（Round 1–9 综合）

> **锚点**: `CASE-SPH-OSC-DROP-FREQ`
> **参考**: Rayleigh (1879), *Proc. R. Soc. Lond.* 29:71; Lamb, *Hydrodynamics*, 6th ed., arts. 313–315
> **验收门限**: |频率误差| < 6%
> **综合轮次**: Rounds 1–9 全量证据（含 Round 9 动态臂根因分析）
> **工件**: `benchmarks/oscillating_drop_freq/out/round7.json`,
>   `benchmarks/oscillating_drop_freq/out/round8.json`,
>   `benchmarks/oscillating_drop_freq/out/round9_rootcause.json`,
>   `docs/figures/case_sph_osc_drop_freq_{convergence,error_bars,order,round8_synthesis,round9_rootcause}.png`

---

## ① 案例介绍

半径 $R$、密度 $\rho_0$、表面张力系数 $\sigma$ 的二维无粘液滴，
初始形状微扰 $r(\theta) = R[1 + \varepsilon \cos(m\theta)]$，
在表面张力作用下自由振荡。Rayleigh (1879) 给出精确解析频率：

$$\omega_m^2 = \frac{\sigma \cdot m(m^2 - 1)}{\rho_0 R^3}, \quad m = 2, 3, \ldots$$

其中 $m=1$ 为纯平移模态（$\omega=0$），$m=2$ 为四极振荡模态（面积一阶守恒）。

**物理参数表**：

| 参数 | 符号 | 值 | 单位 |
|---|---|---|---|
| 表面张力系数 | $\sigma$ | 0.072 | N/m |
| 密度 | $\rho_0$ | 1000.0 | kg/m³ |
| 液滴半径 | $R$ | 0.05 | m |
| 模态 | $m$ | 2, 3 | — |
| 微扰幅值 | $\varepsilon$ | 0.02, 0.05 | — |

**解析基准值**（$m=2$）：
- $\omega_2 = \sqrt{6\sigma/(\rho_0 R^3)} = 1.8590$ rad/s
- $T_2 = 2\pi/\omega_2 = 3.3798$ s

**验证量**：无量纲频率比 $Q = \omega_{\rm SPH} / \omega_{\rm Rayleigh}$，精确值 $Q = 1$。
门限：$|Q - 1| < 6\%$。

---

## ② 怎么做的

### 2.1 方法概述

TensorSPH 使用连续表面力（CSF, Brackbill et al. 1992）模型处理表面张力。
验证分两条路径：

1. **准静态线性响应**（Rounds 1–4, 7）：对微扰液滴施加 CSF 力场，
   通过色场（colour field）等值线提取界面形状，做模态投影得到等效刚度 $g$，
   进而得到 $Q = \sqrt{g}$。两条独立通道：
   - **通道 A**：模态投影（$c=1/2$ 等值线的 Legendre 投影）
   - **通道 B**：表面能（面积变化的二阶导数，无曲率/模态投影）
   - **通道 A'**：扣零点的差分通道 A（消除格点系统偏差）

2. **动态时间推进**（Rounds 5–6, 9）：直接积分运动方程，
   从零交叉点提取振荡周期。Round 9 对发散进行子系统力分解定位根因。

### 2.2 收敛分析（Round 7）

在分辨率阶梯 $R/dx = 12, 16, 24$ 上运行，用 ASME V&V 20 方法：
- **观测收敛阶** $p$：网格搜索拟合 $Q(h) = Q_\infty + C h^p$
- **GCI**（Grid Convergence Index）：$GCI_{\rm fine} = F_s |\varepsilon| / (r^p - 1)$，$F_s = 1.25$
- **验证不确定度**：$U_V = \sqrt{U_{\rm num}^2 + U_{\rm input}^2}$
- **V&V 20 判据**：$|Q_{\rm fine} - 1| \leq U_V$

### 2.3 动态臂根因分析（Round 9）

对 RHS 进行子系统力分解（压力 vs CSF），发现：
- Laplace 压缩的初始状态携带尖锐的表面压力梯度（$p_{\rm lap} \to 0$ 在 ~2h 内）
- SPH 压力梯度算子将其放大为声学脉冲
- $|a_{\rm press}|_{\max}$ 从 26.9 增长到 1792.5（66.5×），而 $|a_{\rm csf}|_{\max}$ 保持在 ~17–45
- **根因**：可压缩初始状态的声学启动不稳定性，不是 CSF 算子本身的缺陷

### 2.4 工具链

- **求解器**: TensorSPH (`tensorsph.capillary_oscillation`, `tensorsph.surface_tension`)
- **等值线提取**: `matplotlib.contour` (marching squares)
- **模态投影**: 梯形积分 $\int (r - \bar{r}) P_m(\cos\theta) \sin\theta \, d\theta$
- **收敛分析**: 自研网格搜索（无 scipy 依赖）
- **力分解**: `round9_dynamic_rootcause.py`（与生产代码完全镜像）
- **图表**: matplotlib 程序化生成，零手抄

---

## ③ 计算结果

### 3.1 准静态验证（Round 7，R/dx = 24）

| 通道 | 模态 | ε | Q(24) | 误差(%) | 门限通过 |
|---|---|---|---|---|---|
| A | 2 | 0.02 | 0.9689 | −3.11 | ✅ |
| A | 2 | 0.05 | 0.9622 | −3.78 | ✅ |
| A | 3 | 0.02 | 0.9573 | −4.27 | ✅ |
| A | 3 | 0.05 | 0.9573 | −4.27 | ✅ |
| A' | 2 | 0.02 | 0.9582 | −4.18 | ✅ |
| A' | 2 | 0.05 | 0.9579 | −4.21 | ✅ |
| A' | 3 | 0.02 | 0.9572 | −4.28 | ✅ |
| A' | 3 | 0.05 | 0.9572 | −4.28 | ✅ |
| B | 2 | 0.02 | 0.9691 | −3.09 | ✅ |
| B | 2 | 0.05 | 0.9625 | −3.75 | ✅ |

**门限通过率**: 10/10 = 100% ✅

### 3.2 渐近验证（Round 7，ASME V&V 20）

| 指标 | 值 |
|---|---|
| 总行数 | 10 |
| 通过行数 | **3** |
| 通过率 | **30%** |
| 方法 | GCI + 估计器控制偏置（$U_V = \sqrt{U_{\rm num}^2 + U_{\rm input}^2}$） |

### 3.3 收敛阶分布（Round 7）

| 统计量 | 值 |
|---|---|
| 中位数 p | **1.93** |
| 范围 | 0.20 – 4.00 |
| 单调行 | 8 / 10 |
| 非单调行 | 2 / 10（均为 l=2, channel A） |
| 一阶假设成立？ | **否**（R4 的 p=1 假定被证伪） |

### 3.4 系统性偏差（Round 8）

| 通道 | l | Q∞ 误差均值 | 标准差 | 行数 |
|---|---|---|---|---|
| A / A' / B | 3 | **−2.54%** | 0.02% | 4 |
| A_differential | 2 | **−0.10%** | 0.07% | 2 |

**关键发现**：l=3 模态在所有单调通道中一致地收敛到 −2.5%，不是 0。

### 3.5 动态臂根因（Round 9）

| 指标 | 值 | 说明 |
|---|---|---|
| |a_press|_max 增长 | 27.0 → 1792.5 (**66.5×**) | 声学脉冲放大 |
| |a_csf|_max 增长 | 17.1 → 45.3 (2.6×) | 次要增长 |
| 主导子系统 | **压力** | 声学启动脉冲 |
| CSF 角色 | 次要 | 径向平均小且向外 |
| σ→0 对照 | 静止 | 积分器+连续性干净 |

---

## ④ 结果图

### 图1: 收敛阶梯

![收敛阶梯](case_sph_osc_drop_freq_convergence.png)

> Q(R/dx) 收敛阶梯。所有通道在 R/dx=24 通过 6% 门限。

### 图2: 误差柱状图

![误差柱状图](case_sph_osc_drop_freq_error_bars.png)

> Q∞ 误差分布。l=3 一致偏离 −2.5%。

### 图3: 收敛阶分布

![收敛阶](case_sph_osc_drop_freq_order.png)

> 观测收敛阶 $p$ 的分布。中位数 $p = 1.93$，显著高于 R4 假定的 $p=1$。

### 图4: Round 8 综合合成

![Round 8 合成](case_sph_osc_drop_freq_round8_synthesis.png)

> 全量证据综合：门限通过率、V&V 20 通过率、收敛阶分布、系统性偏差。

### 图5: Round 9 根因分析

![Round 9 根因](case_sph_osc_drop_freq_round9_rootcause.png)

> 力分解轨迹：|a_press|_max 从 27.0 增长到 1792.5（66.5×），|a_csf|_max 保持在 ~17–45。
> 根因：Laplace 压缩初始状态的声学启动不稳定性。

---

## ⑤ 注意点 / 局限

1. **仅 3 个网格**：$R/dx = 12, 16, 24$。$R/dx = 32, 40$ 的结果（cj4101）从未取回，
   因此 $p$ 由 3 点定 3 参数，部分行退化（$p$ 顶到边界/区间宽）。
2. **l=3 的 −2.5% 系统偏差**：在所有单调通道中一致出现，来源未明。
   可能是曲率估计的系统偏差、界面重构的格点效应、或 CSF 方法的内在限制。
3. **动态臂 FAIL**：时间推进积分在所有配置下发散（Round 5），
   Round 9 定位根因为**声学启动不稳定性**——Laplace 压缩初始状态携带尖锐表面压力梯度，
   SPH 压力梯度算子将其放大为声学脉冲。
4. **本报告不跑新 SPH**：所有数字来自 `out/round7.json`、`out/round8.json`、
   `out/round9_rootcause.json`，零手抄。图表由 `gen_verification_figures.py` 和
   `round9_figure.py` 程序化生成。

---

## ⑥ 误差判定（逐项，明确回答"是否 <6 %"）

| 关键量 | 结果 | 基准 | 误差 | <6 %？ |
|---|---|---|---|---|
| Q(24)，全部 10 行 | 0.957–0.969 | 1（Rayleigh） | −4.3% ~ −3.1% | **是（10/10）✅** |
| Q∞，l=3（p≈1.93） | 0.9745 | 1 | **−2.55%**，U_V=2.3% | **否（V&V 20 FAIL）⚠️** |
| Q∞，l=2 A'（p≈1.05） | 0.9985 | 1 | −0.15%，U_V=5.3% | **是 ✅** |
| 动态臂 ω | **not_available** | 1 | — | **未测（发散）❌** |
| 估计器自身偏置 | ≤ 0.05% | 0 | — | **是 ✅** |

### 总账锚点结论

| 维度 | 状态 | 说明 |
|---|---|---|
| **6% 门限（R/dx=24）** | ✅ **通过** | 10/10 行通过 |
| **渐近验证（V&V 20）** | ⚠️ **部分通过** | 3/10 行通过，l=3 有 −2.5% 系统偏差 |
| **动态时间推进** | ❌ **不可用** | 声学启动不稳定性（Round 9 根因已定位） |
| **综合 verdict** | **PARTIAL** | 门限通过，但渐近性未完全建立，动态臂不可用 |

---

## ⑦ 复现

```bash
# 生成 Round 7 图表（不跑 SPH，<5s）
PYTHONPATH=src python benchmarks/oscillating_drop_freq/gen_verification_figures.py

# 生成 Round 9 图表（<5s）
PYTHONPATH=src python benchmarks/oscillating_drop_freq/round9_figure.py

# 准静态验证（轻量，~2s/点）
PYTHONPATH=src python benchmarks/oscillating_drop_freq/verify_rayleigh.py --quick

# 完整分辨率阶梯（~30s）
PYTHONPATH=src python benchmarks/oscillating_drop_freq/verify_rayleigh.py --merge

# Round 9 力分解（~85s）
PYTHONPATH=src python benchmarks/oscillating_drop_freq/round9_dynamic_rootcause.py
```

---

## ⑧ 下一步建议

| 优先级 | 任务 | 预期产出 | 估算成本 |
|---|---|---|---|
| P0 | 取回/重算 R/dx=32/40 | 5 网格收敛阶，确认 −2.5% 偏差是否持续 | 中（集群） |
| P0 | 修复动态臂稳定性 | 可信的 ω 从时间推进获得（需解决声学启动问题） | 高 |
| P1 | l=3 偏差溯源 | 定位是曲率/重构/CSF 的哪一环 | 中 |
| P1 | l=2 取向系综 | 确认非单调性是否为格点效应 | 中（集群） |
| P2 | 改进 CSF 曲率估计 | 降低系统偏差至 <1% | 高 |

---

**平台报告**：本报告由 `gen_verification_figures.py` 和 `round9_figure.py` 程序化生成图表，
所有数字来自 `out/round7.json`、`out/round8.json`、`out/round9_rootcause.json`，零手抄。
Mission: 25044, Tags: `oscillating-drop`, `rayleigh`, `verification`, `sph`, `capillary`, `round9`.
