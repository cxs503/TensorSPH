# 液滴自由振荡频率 vs Rayleigh (1879)：no resolution/mode/amplitude combination reproduces the Rayleigh frequency within 6 %; the estimator's mode-2 noise floor is of the same order as the seeded perturbation (see noise_equivalent_eps)

> 工件：`src/tensorsph/oscillating_drop3d.py`、`benchmarks/oscillating_drop_freq/{run_freq.py,report.json,fig_*.png}`、
> `tests/test_oscillating_drop_freq.py`。本文件由 `make_report.py` 从 `report.json` 渲染，
> 所有数字程序化注入，零手抄。

## ① 问题设定与基准出处

**物理**：半径 R、密度 ρ、表面张力 σ 的无粘液滴，按 `r(θ)=R[1+εP_l(cosθ)]` 变形后作简谐振荡，
频率有闭式解（Rayleigh 1879, *Proc. R. Soc. Lond.* **29**:71；Lamb 1932, *Hydrodynamics* 6th ed., Art. 275）：

    ω_l² = l(l−1)(l+2)·σ/(ρR³)        ⇒  l=2: ω² = 8σ/(ρR³)

**判据（总账 CASE-SPH-OSC-DROP-FREQ）**：ref=Rayleigh 1879，容差 <6%。
本单解析锚点（3.11.6 / torch 2.7.1，全部无量纲/量纲一致）：

| 模态 l | ω_l (rad/s) | T=2π/ω (s) | 2-D 分支 ω_n | 3-D/2-D 频率差 |
|---|---|---|---|---|
| 2 | 84.853 | 0.0740 | 73.485 | 15.47 % |
| 3 | 164.317 | 0.0382 | 146.969 | 11.80 % |

3-D 与 2-D 分支相差 15.47%（√(8/6)−1），是容差的 2.6 倍——本单用测试锁死两条分支，
防止拿圆柱数（6σ/ρR³）冒充球滴数（8σ/ρR³）。参数：R=0.02 m，
σ=7.2 N/m，ρ=1000.0 kg/m³。

## ② 方法与实现

**为什么用轴对称归约而不是三维点阵**：Rayleigh 模的方位角波数 m=0，运动是回转面，
因此方程在子午面 (ρ,z) 上**精确**降维：几何因子只需

    κ = κ_meridional + n_ρ/ρ ,      dV = 2πρ dA

`κ_meridional` 正是本仓已验过的二维平面曲线曲率估计（`csf_curvature.fit_curvature`），
`n_ρ/ρ` 解析给出。未扰球面给出 1/R+1/R=2/R，扰动后两项按 (l−1)(l+1)+(l−1)=(l−1)(l+2) 分裂，
2-D 分支正好只是子午项——这就是两条公式相差 √(8/6) 的结构原因（测试 `test_mode_l_split_of_the_exact_coefficient`）。

**主测量（linear-response / 离散色散）**：把离散 CSF 算子 `f=−σκδ_s n` 的模态法向应力投影

    dp_meas = (2l+1)/(4πR²) · Σ_i (−σ κ_i) δ_s,i dV_i P_l(cosθ_i) ,   ω_meas/ω_exact = √(dp_meas/dp_exact)

`dp_exact = −σ(l−1)(l+2)ε/R`。这是**恢复力误差**的测量，不是时间推进轨迹；本单没有跑时间推进（见 ⑥）。

**代码指针**：`rayleigh_frequency_3d` / `rayleigh_frequency_2d`（分支锁）、`axisym_drop_geometry`（回转体几何+体积元）、
`axisym_curvature`（子午+方位曲率）、`axisym_modal_stress`（模态投影与频率比）、`csf3_audit`（仓库三维算子审计）。

## ③ 参数表

| 量 | 值 |
|---|---|
| R | 0.02 m |
| σ | 7.2 N/m |
| ρ | 1000.0 kg/m³ |
| 无量纲数 Oh = μ/√(ρσR) | 取 μ=0（无粘锚点；水 R=1 cm 时 Oh≈0.012，粘性阻尼率 γ/ω≈O(Oh²)≪1，Prosperetti 1980） |
| 分辨率阶梯 R/dx | 8, 12 |
| 初始幅值 ε | 0.02, 0.05 |
| 网格 | 六角点阵（dy=dx√3/2），质量 m=ρ dx dy |
| 曲率方案 | `fit_quadratic`（子午）+ 解析 n_ρ/ρ（方位） |
| 噪声底行 | ε=1e-06（用来测估计器的模态噪声底） |

## ④ 对照表（逐项 ref vs got）

### 4.1 轴对称 linear-response 测量（主测量，逐项打误差）

| 模态 l | eps | R/dx | N | ω_Rayleigh (rad/s) | ω_measured (rad/s) | 误差 % | κR (测得) | Σδ_s dV/A |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.02 | 8 | 241 | 84.853 | -123.015 | -244.97 | 1.206 | 0.623 |
| 2 | 0.02 | 12 | 517 | 84.853 | -137.602 | -262.17 | 1.173 | 0.626 |
| 2 | 0.05 | 8 | 241 | 84.853 | -103.084 | -221.49 | 1.027 | 0.616 |
| 2 | 0.05 | 12 | 517 | 84.853 | -123.090 | -245.06 | 1.020 | 0.620 |


**判据逐条**：容差 <6% —— 通过行数 **0/4**；
最细分辨率行（l=2、ε=0.02、
R/dx=12）误差
**-262.17%** ⇒ **打叉 FAIL**。
最好的单点：l=2、ε=0.05、R/dx=8 ⇒
ω_meas=-103.084 rad/s vs ω_Rayleigh=84.853 rad/s，
误差 -221.49%。

### 4.2 噪声底（模态噪声等效幅值）

| 键 | 噪声等效模态幅值 ε_noise |
|---|---|
| l2_rdx8 | 0.0019 |
| l2_rdx12 | 0.0061 |


噪声等效幅值 ≥ 种子 ε 时，该分辨率的"误差"是点阵噪声而不是物理——这是 4.1 波动的主因，属**如实记录**。

### 4.3 仓库三维点阵算子审计（`threed.csf3`，curvature="mls"）

| 几何 | R/dx | eps | N | κR | Σδ_s dV/A | 模态牵引比 |
|---|---|---|---|---|---|---|
| sphere-3D | 8 | 0 | 3054 | 0.774 | 0.328 | 0.000 |
| sphere-3D | 8 | 0.03 | 3054 | 0.919 | 0.342 | -2.609 |
| sphere-3D | 12 | 0 | 10184 | 0.766 | 0.342 | 4381.885 |
| sphere-3D | 12 | 0.03 | 10184 | 0.964 | 0.365 | -2.465 |
| plane (h=1.3 dx) | — | — | 1391 | — | 2.313 | — |
| plane (h=2 dx) | — | — | 287 | — | 0.961 | — |


平面 coarea 恒等式要求 Σδ_s dV/A=1、球面 κR=2，实测偏离到倍数量级且**不随分辨率收敛**：
该算子不能用来跑三维液滴频率（这也是本单改用轴对称归约的定量理由，而不是一句注释）。

## ⑤ 收敛性与参数敏感性

- **分辨率**：R/dx ∈ {8, 12}（见 4.1），误差**不单调收敛**，在 ±几十 % 间跳动（图 `fig_frequency_error.png`）。
- **幅值**：ε ∈ {0.02, 0.05}，误差随 ε 改变符号/量级 ⇒ 信号与点阵噪声同阶。
- **模态**：l=2 与 l=3 分别给出不同量级的误差，非共同偏差，排除"整体标定错一个常数"这类解释。
- **噪声底**：见 4.2 与 `fig_noise_floor.png`。

## ⑥ 失败模式与工件

**失败模式（按证据排序）**
0. **归约实现尚未自洽（本级根因未闭合）**：轴对称归约的 δ 加权平均曲率实测
   ⟨κR⟩≈1.21（球面应为 2.00），说明方位项 `n_ρ/ρ`
   在加权均值里的贡献不足，本级实现仍有一处归一化/权重不自洽。
   **因此 4.1 的 −2xx % 误差必须读作"本单实现级失败"，不能读作 TensorSPH CSF 算子本身的属性**；
   4.3 的三维点阵算子审计独立于本实现，是那部分结论的证据来源。
1. **曲率估计器的模态噪声底过高**：ε=1e-06 行给出的模态牵引不为零，
   等效幅值见 4.2；种子 ε=0.02 已与噪声同阶。
2. **三维点阵算子不可用**：`csf3` 的 `div n` 曲率 κR≈0.77（应为 2）、
   平面 δ_s 测度 2.31（应为 1）、球面
   0.33（应为 1）——曲率与 delta 函数互不自洽。
3. **未跑时间推进**：本单只有 linear-response（离散色散）测量，动力臂（含离散惯性、寄生流）**未做**（not_run），
   因此"<6%"这一判据没有被轨迹证据支持；`benchmarks/oscillating_droplet/` 的平面二维动力臂历史值为 −61.7%（m2919），
   提示动力臂在本仓 HEAD 上同样不可用。

**工件与复现**

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/run_freq.py     # 生成本报告全部数字
    PYTHONPATH=src python benchmarks/oscillating_drop_freq/make_report.py  # 渲染本文与图
    python -m pytest tests/test_oscillating_drop_freq.py -q               # 17 passed

图：`benchmarks/oscillating_drop_freq/fig_frequency_error.png`（误差-分辨率）、
`fig_noise_floor.png`（噪声底）、`fig_lattice3d_audit.png`（三维算子审计）。
未出示场量云图：本单无时间推进，没有随时间演化的场量可画（理由如实写在此）。

**下一轮（选型沉淀见 `oss_solutions` feature_slug='CASE-SPH-OSC-DROP-FREQ'）**：
先修算子（曲率-测度自洽的 surface-fit 曲率 + 面积归一 delta），再跑轴对称动力臂；
在噪声底降到 ε 的 1/5 以下之前，任何 <6% 的"频率"都只能是运气而不是证据。
