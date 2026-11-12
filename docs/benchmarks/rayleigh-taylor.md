# 多相 Rayleigh–Taylor 线性增长率（TensorSPH）

**算例**: 两层流体单模界面扰动，重流体在上，Atwood 数 A = 0.5；指标=早期界面扰动指数增长率 vs $\sigma=\sqrt{Agk}$。

**ref**: G. I. Taylor, *The instability of liquid surfaces when accelerated in a direction perpendicular to their planes. I*, Proc. R. Soc. Lond. A **201** (1950) 192–196, doi:10.1098/rspa.1950.0052（线性阶段增长率 $\sigma^2=A g k$，原文 §2 的扰动方程；A = (ρ₂−ρ₁)/(ρ₂+ρ₁)）。原文未给出 A=0.5 的具体数值表，故目标值由该解析式在给定参数下直接求值，并附两项自洽修正（有限深度、黏性）。

## ① 问题与参考值

- 目标值 $\sigma_{\rm ideal}=\sqrt{Agk}$ = **5.5515 s⁻¹**（A=0.5, g=9.81 m/s², k=2π/λ=2π m⁻¹, λ=1 m）。
- 有限深度修正（上下各 0.75λ，自由滑移刚性边界）$\sigma^2=kg(\rho_h-\rho_l)/(\rho_h\coth kd_h+\rho_l\coth kd_l)$ = **5.5510 s⁻¹**（修正因子 0.99992）。
- 黏性修正 $\sigma=\sqrt{\sigma_0^2+\nu^2k^4}-\nu k^2$ = **5.5117 s⁻¹**（ν=1e-3 m²/s，修正因子 0.99283）。

判据（**动手前**预设，见 experimental-design 技能）：以 $\sigma_{\rm num}$ 在振幅窗口 $2\eta_0\le\eta\le0.1\lambda$ 内对 $\ln\eta$ 做最小二乘斜率，与 $\sqrt{Agk}$ 相对误差 <10 % 为 PASS；窗口外、$\eta>0.1\lambda$ 的数据不得进入拟合。

## ② 方法与实现

TensorSPH 此前**没有多相时间推进器**（`benchmarks/two_phase_laplace` 只是静态 CSF 算子校验，且被登记为 L0：`negative_knowledge_ledger` negk:gate:93）。本算例新增 `src/tensorsph/rayleigh_taylor.py`，复用共享机器（cubic-spline 核、`pair_kernels`、`lattice_kernel_sum`、cell-linked 邻域、Morris 黏性 Laplacian 的标定因子），只加入密度间断必需的部件：

1. **逐粒子参考密度** $\rho_{0,i}$ 与各自的 Tait 刚度 $B_i=\rho_{0,i}c_0^2/\gamma$。
2. **数密度（体积）连续式**而非质量求和：$\delta_i=\sum_j W_{ij}$，$\rho_i=\rho_{0,i}\delta_i/\delta_{0}$。$\delta$ 计的是粒子体积，故被光滑的是**压缩率**（跨界面连续），而不是跳变 3 倍的密度（Solenthaler & Pajarola 2008, *Density contrast SPH*）。对照实验表明：改成质量求和 + 逐相 Tait 会在界面产生140×（轻相）／19×（重相）的压强误差，已由 `tests/test_rayleigh_taylor.py::test_mass_summation_is_unusable_across_the_jump` 钉住。
3. **密度反差对称压力力**（Grenier et al. 2009 哈密顿界面形式）$a_i=-\sum_j m_j (p_i+p_j)/(\rho_i\rho_j)\nabla_i W_{ij}$；$\rho_i=\rho_j$ 时与单相代码的 $m_j(p_i/\rho_i^2+p_j/\rho_j^2)$ **恒等**，有密度跳变时保持静水平衡的相透明性。
4. **静水解析格点初始化**：按 $s(y)=(1+\gamma p(y)/\rho_0c_0^2)^{1/\gamma}$ 反解竖向间距，并以初始格点标定 $\delta_{\rm ref}$（等价于单相求解器的 `calibrate_initial_mass`），使 t=0 压强**逐点等于**解析静水压强（残差 <1e-6 相对量级）；上下各 3 层静止壁面粒子以 Adami et al. (2012) 式(11) 外插壁压。
5. **线性本征函数激励**：$\xi_y=\eta_0\cos kx\,e^{-k|y-y_{\rm int}|}$、$\xi_x=\mp\eta_0\sin kx\,e^{-k|y-y_{\rm int}|}$（散度为零）以及对应的 $\sigma\eta_0$ 速度场，避免竖直位移带来的 $k\eta_0$ 压缩伪信号。
6. **两个独立振幅量计**：重相逐列质心 $\bar y(x)$（层厚均匀时 $y_{\rm int}=2\bar y-H$）与界面带 $v_y$ 的模态投影；前者按已知初始振幅标定，常数衰减因子不会伪装成增长率误差。

## ③ 参数表（无量纲量完整）

| 量 | 符号 | 值 | 无量纲化 |
|---|---|---|---|
| 波长 | λ | 1 m | 长度单位 |
| 界面位置 | y_int | 0.75 m | λ/4·3 |
| 层深 | d_l = d_h | 0.75 m | k d = 4.712 |
| 密度 | ρ_l / ρ_h | 500 / 1500 kg/m³ | ρ_h/ρ_l = 3 |
| Atwood 数 | A | 0.5 | — |
| 重力 | g | 9.81 m/s² | Froude 无关（纯浮力驱动） |
| 初始振幅 | η₀ | 0.01 m | η₀/λ = 0.01 |
| 声速 | c₀ | 40 m/s | M_max ≈ 0.009 |
| 运动黏性 | ν | 1e-3 m²/s | νk²/σ = 0.0071 |
| 静水压缩率 | 1-s | — | 底部轻相 1.74 %（p=14.7 kPa, B_l=114 kPa） |
| 粒子间距 | dx | 0.04 m（系列 0.04→0.00625） | λ/h = 19 |

## ④ 对照表（ref vs got，逐项打勾叉）

| dx [m] | 粒子数 | 步数 | σ_num [s⁻¹] | 相对 √(Agk) | 拟合 R² | <10% 判据 |
|---|---|---|---|---|---|---|
| 0.04 | 950 | 2334 | 2.125 | -61.7 % | 0.9993 | FAIL |
| 0.025 | 2440 | 1945 | 3.095 | -44.2 % | 0.9980 | FAIL |
| 0.02 | 3800 | 2891 | 3.971 | -28.5 % | 0.9994 | FAIL |
| 0.0125 | 9680 | 4627 | 4.192 | -24.5 % | 0.9999 | FAIL |

A = 0 对照（同几何、ρ_h=ρ_l，无 RT 驱动）：界面模态的**数值阻尼**，用于解释上表的系统性偏低：

| dx [m] | σ_num (A=0) [s⁻¹] |
|---|---|
| 0.05 | -4.430 |
| 0.025 | -2.037 |

## ⑤ 收敛性与参数敏感性

- dx=0.04: σ/√(Agk) = 0.383（-61.7 %）
- dx=0.025: σ/√(Agk) = 0.558（-44.2 %）
- dx=0.02: σ/√(Agk) = 0.715（-28.5 %）
- dx=0.0125: σ/√(Agk) = 0.755（-24.5 %）

敏感性（dx=0.025 固定）：

| 变化 | σ_num [s⁻¹] | σ_num/σ_th | 理论预期 |
|---|---|---|---|
| ν=0 | 3.238 | 0.583 | 1.000（$\nu k^2$ 阻尼） |
| ν=0.001 | 3.095 | 0.558 | — |
| ν=0.01 | 2.055 | 0.370 | 0.929（$\nu k^2$ 阻尼） |
| η₀=0.02 | 3.417 | 0.616 | 线性区应与 η₀ 无关 |

## ⑥ 失败模式与工件

- 唯一失败模式：**界面模态的数值阻尼**。在 A=0 对照中同一初始扰动按 -4.43 1/s（dx=0.05）衰減、-2.04 1/s（dx=0.025），近似 O(dx)，与实测增长率亏损（柯西式叠加 σ_meas ≈ σ_th − λ_num）一致。
- 未通过 <10% 判据的根因是**离散分辨率**而不是物理模型：σ/σ_th 随 dx 单调上升（dx=0.0125: -24.5%, dx=0.02: -28.5%, dx=0.025: -44.2%, dx=0.04: -61.7%），但 0.02→0.0125 只从 0.715 升到 0.755，收敛慢于一阶外推的乐观预期；按此趋势需 dx≲0.003 才可能进 10% 门，属**本轮 honest-FAIL**，接棒轮应改走界面锐化/质量求和+δ-SPH 或压力投影（ISPH）路线。
- 工件：`benchmarks/rayleigh_taylor/runs/*.json`（逐算例时序+拟合+场量快照）、`runs/table.json`、`runs/damping.json`、`figures/rt_growth.png`、`figures/rt_density_field.png`；复现：`PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.02`。

## 生态摩擦（如实记录）

1. TensorSPH 无多相推进器（本次补齐）；`oss_solutions` 中 SPHinXsys 的 `GAP-SPH-MULTIPHASE` 只有「逐粒子参考密度 + 界面力」的方法摘要可借，无数值基准。
2. `cell_linked_pairs` 只支持非周期盒子：x 周期由本模块的 `periodic_pairs`（加倍云 + 折叠 + 最小镜像）实现，未改动共享核（避免影响其它算例）。
3. 控制面 128 核被 ~300 负载占满，本地单算例墙钟波动 3–8 倍；0.0125/0.00625 细化档改走集群。
