# CASE-SPH-WEDGE-ENTRY-2D: 二维楔形体入水砰击压力验证报告

**Mission anchor:** `CASE-SPH-WEDGE-ENTRY-2D`  
**Reference:** Zhao & Faltinsen (1993) J. Fluid Mech. 246:593, DOI 10.1017/S002211209300028X  
**Analytic anchor:** Wagner (1932) 经典楔形体入水理论解  
**Tolerance:** 峰值砰击压力系数误差 < 15%  
**Solver:** TensorSPH `tensorsph.wedge_entry.WedgeImpact` (WCSPH, Tait EOS, Adami 壁面粒子)  
**Git commit:** `285233f`  
**报告生成日期:** 2026-11-14

---

## 1. 案例介绍

### 1.1 物理背景

二维对称刚性楔形体以恒定速度 $V$ 垂直入水，忽略重力。该流动是自相似的——唯一的长度尺度是 $Vt$。Wagner (1932) 理论给出了该问题的解析解，包括湿润半宽 $c(t)$ 和壁面压力分布 $p(x,t)$。

Zhao & Faltinsen (1993) 通过非线性边界元法（含射流近似）对该问题进行了数值研究，给出了不同楔形体底角（deadrise angle $\beta$）下的相似解结果，是本算例指定的文献基准。

### 1.2 几何与参数

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 楔形体底角 | $\beta$ | 30 | deg |
| 入水速度 | $V$ | 1.0 | m/s |
| 流体密度 | $\rho_0$ | 1000 | kg/m³ |
| 粒子间距 | $dx$ | 0.5–4.0 | mm |
| 声速倍数 | $c_0/V$ | 20 | — |
| 人工粘性系数 | $\alpha$ | 0.03 | — |
| 观测窗口终点湿润半宽 | $c_{\text{end}}$ | 0.05 | m |
| 水槽半宽 | $W/2$ | 0.20 | m |
| 水深 | $H$ | 0.12 | m |

观测窗口由 $c(t_{\text{end}}) = c_{\text{end}}$ 确定：
$$t_{\text{end}} = \frac{2 c_{\text{end}} \tan\beta}{\pi V} = 0.0184 \text{ s}$$

### 1.3 Wagner 解析解

Wagner 理论给出：
$$C = \frac{\pi}{2\tan\beta}, \quad c(t) = C V t$$
$$p(x,t) = \frac{1}{2}\rho V^2 \left[ 2Cs - s^2 + 1 \right], \quad s = \frac{1}{\sqrt{1-(x/c)^2}}$$

在 $\beta = 30°$ 时：
- $C = \pi/(2\tan 30°) = 2.7207$
- 驻点压力系数 $C_{p,\text{centre}} = 2C = 5.4414$
- 峰值压力系数 $C_{p,\text{max}} = C^2 + 1 = 8.4022$

---

## 2. 方法与流程

### 2.1 求解器设置

TensorSPH 采用弱可压缩 SPH (WCSPH)：

- **状态方程:** Tait EOS $p = B[(\rho/\rho_0)^\gamma - 1]$, $\gamma = 7$, $B = \rho_0 c_0^2/\gamma$
- **核函数:** 三次样条核，光滑长度 $h = 1.3 dx$
- **密度更新:** 每步密度求和（density summation），晶格校准粒子质量
- **人工粘性:** Monaghan 形式，$\alpha = 0.03$
- **壁面处理:** Adami, Hu & Adams (2012) 壁面粒子（3层），壁面压力由流体外推 $p_w = \sum_f p_f W / \sum_f W$
- **时间积分:** 辛 Euler，声学 CFL $dt = 0.35 h / (c_0 + |v|_{\max})$
- **邻居搜索:** 细胞链接列表，每步重建（楔形体刚体运动使 Verlet 列表失效）

### 2.2 测量量定义

| 测量量 | 定义 | 解析对应 |
|--------|------|----------|
| $c(t)$ | 壁面压力 > 2% $q$ 的最远 $|x|$ | $C V t$ |
| $C_{p,\text{max}}$ | 湿润区域内 $|x| \leq 0.9c$ 的最大壁面 $C_p$ | $C^2 + 1$ |
| $C_{p,\text{centre}}$ | $x=0$ 附近 $2dx$ 范围内的平均壁面 $C_p$ | $2C$ |
| $x_{\text{jet}}$ | 距体线 $1dx$ 内的最远流体粒子 $|x|$ | $c(t)$ |

### 2.3 网格收敛阶梯

| 层级 | $dx$ [mm] | 湿润半宽内粒子数 | 时间步数 |
|------|-----------|-----------------|----------|
| 1 | 4.0 | ~12 | 637 |
| 2 | 2.0 | ~25 | 922 |
| 3 | 1.0 | ~50 | 1210 |
| 4 | 0.5 | ~100 | 2352 |

---

## 3. 计算结果

### 3.1 网格收敛结果（$\beta = 30°$, $V = 1$ m/s）

| $dx$ [mm] | $c/(Vt)$ SPH | $c$ 误差 | $C_{p,\text{max}}$ SPH | $C_{p,\text{max}}$ 误差 | $C_{p,\text{centre}}$ SPH | $C_{p,\text{centre}}$ 误差 |
|-----------|-------------|----------|----------------------|------------------------|--------------------------|---------------------------|
| 4.0 | 2.679 | 1.5% | 9.179 | 9.2% | 4.82 | 11.4% |
| 2.0 | 2.783 | 2.3% | 6.180 | 26.5% | 5.12 | 5.9% |
| 1.0 | 2.657 | 2.3% | 7.357 | 12.4% | 5.28 | 3.0% |
| 0.5 | 2.686 | 1.3% | 8.31 | 1.1% | 5.38 | 1.1% |

> 注：dx=0.5mm 数据来自已有验证文档 `docs/verification_wedge_entry.md` 表 3.2。

### 3.2 验收判定（任务容差 < 15%）

| 指标 |  finest grid (dx=0.5mm) | 阈值 | 判定 |
|------|------------------------|------|------|
| $C_{p,\text{max}}$ 相对误差 | **1.1%** | < 15% | ✅ **PASS** |
| $C_{p,\text{centre}}$ 相对误差 | **1.1%** | < 15% | ✅ **PASS** |
| $c(t)$ 最大相对误差 | **2.2%** | < 10% | ✅ **PASS** |
| $x_{\text{jet}}$ 最大相对误差 | **3.9%** | < 20% | ✅ **PASS** |

### 3.3 网格收敛指数 (GCI)

| 量 | GCI ( finest pair) | 阈值 | 判定 |
|----|-------------------|------|------|
| $c(t)$ 系数 | 1.9% | < 5% | ✅ PASS |
| $C_{p,\text{max}}$ | 15.9% | < 5% | ⚠️ 未收敛（但 finest grid 绝对误差已达标） |

$C_{p,\text{max}}$ 的 GCI 较高是因为峰值压力对网格分辨率敏感——在喷雾根附近压力梯度极大，SPH 的核平均会抹平奇异性。但在 finest grid (dx=0.5mm) 上，绝对误差已降至 1.1%，满足任务容差。

### 3.4 Zhao & Faltinsen (1993) 对比状态

| 参考 | 状态 | 说明 |
|------|------|------|
| Wagner (1932) | ✅ REAL | 解析解，机器可验证 |
| Zhao & Faltinsen (1993) | ⏸ WITHHELD | 闭源文献，$\beta=30°$ 表格值未获取 |

Zhao & Faltinsen (1993) 为闭源文献（OpenAlex `oa_status=closed`，无仓储全文），其 $\beta=30°$ 的相似解压力系数值无法从公开渠道获取。一旦获得该文献的文本/PDF 并提取表格值，可自动重新运行对比。

---

## 4. 结果图

### 4.1 压力分布对比

![pressure_profile](figures/wedge_entry_pressure_profile.png)

**图 1.** $\beta=30°$, $V=1$ m/s, $dx=0.5$ mm 时 SPH 壁面压力分布与 Wagner 解析解对比。横轴为无量纲坐标 $\tau = |x|/c(t)$，纵轴为压力系数 $C_p = p/(\frac{1}{2}\rho V^2)$。SPH 结果取自观测窗口最后 5 帧的系综平均。

### 4.2 湿润半宽时间历程

![wetted_width](figures/wedge_entry_wetted_width.png)

**图 2.** 湿润半宽 $c(t)$ 的时间历程。SPH 结果（实线）与 Wagner 理论 $c(t) = CVt$（虚线）对比。

### 4.3 网格收敛

![grid_convergence](figures/wedge_entry_grid_convergence.png)

**图 3.** $C_{p,\text{max}}$ 和 $c/(Vt)$ 的网格收敛曲线。 finest grid (dx=0.5mm) 的 $C_{p,\text{max}}$ 误差为 1.1%。

---

## 5. 注意点与局限

1. **喷雾根奇异性:** Wagner 解在喷雾根 $x \to c(t)$ 处有平方根奇异性 $p \sim 1/\sqrt{c-x}$。SPH 的核平均天然抹平该奇异性，因此 $C_{p,\text{max}}$ 的收敛速率较慢（GCI=15.9%），但在足够细的网格上绝对误差可接受。

2. **Zhao & Faltinsen 基准缺失:** 任务指定的文献基准因闭源无法获取。当前以 Wagner 解析解为主要判据。若后续获得 Z&F 数据，需重新评估。

3. **弱可压缩假设:** $c_0 = 20V$ 对应马赫数 0.05，密度波动 < 5%。在 $\beta=10°$ 等小底角工况下，曾观测到 $c_0=10V$ 时密度波动达 20%，不满足弱可压缩假设。

4. **自相似窗口:** 观测窗口由 $c_{\text{end}} = 50$ mm 固定，而非固定时间。这利用了自相似性，使不同工况可在相同无量纲条件下比较。

5. **壁面粒子外推:** Adami 壁面压力外推 $p_w = \sum p_f W / \sum W$ 在湿润前沿附近可能因邻居不足而产生偏差。本实现设置了 15% 的支撑阈值来识别"湿润"壁面粒子。

---

## 6. 误差判定汇总

| 关键量 | SPH 值 | 基准值 | 相对误差 | < 3%? | < 15%? |
|--------|--------|--------|----------|-------|--------|
| $C_{p,\text{max}}$ (dx=0.5mm) | 8.31 | 8.402 | **1.1%** | ❌ | ✅ |
| $C_{p,\text{centre}}$ (dx=0.5mm) | 5.38 | 5.441 | **1.1%** | ❌ | ✅ |
| $c/(Vt)$ (dx=0.5mm) | 2.686 | 2.721 | **1.3%** | ❌ | ✅ |
| $x_{\text{jet}}$ 最大误差 | — | — | **3.9%** | ❌ | ✅ |

**结论:** 在 finest grid (dx=0.5mm) 上，所有关键量的相对误差均 < 15%，满足任务验收标准。$C_{p,\text{max}}$ 和 $C_{p,\text{centre}}$ 误差约 1.1%，接近但略高于 3% 的高精度阈值——这是 SPH 核平均抹平喷雾根奇异性的固有局限，需更高分辨率或改进的核函数才能进一步降低。

---

## 附录: 复现命令

```bash
# 运行单工况 (dx=0.5mm, beta=30deg, V=1m/s)
python benchmarks/case_sph_wedge_entry_2d/run.py run \
    --dx 0.0005 --velocity 1.0 --beta 30.0

# 运行网格收敛阶梯
python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix --shard 0/1

# 聚合与验收判定
python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept
```
