# 多相 Rayleigh–Taylor 线性增长率（TensorSPH）— 返修轮1

**算例**: 两层流体单模界面扰动，重流体在上，Atwood 数 A = 0.5；指标=早期界面扰动指数增长率 vs $\sigma=\sqrt{Agk}$。

**ref**: G. I. Taylor, *The instability of liquid surfaces when accelerated in a direction perpendicular to their planes. I*, Proc. R. Soc. Lond. A **201** (1950) 192–196, doi:10.1098/rspa.1950.0052（线性阶段增长率 $\sigma^2=A g k$，原文 §2 的扰动方程；A = (ρ₂−ρ₁)/(ρ₂+ρ₁)）。

## ① 问题与参考值

- 目标值 $\sigma_{\rm ideal}=\sqrt{Agk}$ = **5.5515 s⁻¹**（A=0.5, g=9.81 m/s², k=2π/λ=2π m⁻¹, λ=1 m）。
- 有限深度修正 $\sigma^2=kg(\rho_h-\rho_l)/(\rho_h\coth kd_h+\rho_l\coth kd_l)$ = **5.5510 s⁻¹**（修正因子 0.99992）。
- 黏性修正 $\sigma=\sqrt{\sigma_0^2+\nu^2k^4}-\nu k^2$ = **5.5117 s⁻¹**（ν=1e-3 m²/s，修正因子 0.99283）。

判据：以 $\sigma_{\rm num}$ 在振幅窗口 $2\eta_0\le\eta\le0.1\lambda$ 内对 $\ln\eta$ 做最小二乘斜率，与 $\sqrt{Agk}$ 相对误差 <10 % 为 PASS。

## ② 方法与实现（返修轮改进）

在原有 `src/tensorsph/rayleigh_taylor.py` 基础上新增：

1. **RK2（中点法）时间积分**：`time_integrator="rk2"`，二阶精度，比原 symplectic Euler（一阶）耗散更小。
2. **δ-SPH 密度扩散**：`density_diffusion=0.01–0.1`，Antuono–Colagrossi–Marrone (2012) 格式，抑制界面密度噪声。
3. **基准脚本扩展**：`benchmarks/rayleigh_taylor/run.py` 新增 `--time-integrator` 和 `--density-diffusion` 参数。

原有核心算法不变：
- 数密度（体积）连续式 + 逐粒子参考密度
- Grenier et al. (2009) 哈密顿界面压力形式
- 静水解析格点初始化 + 线性本征函数激励
- 双振幅量计（质心 + 速度模态）

## ③ 参数表

| 量 | 符号 | 值 |
|---|---|---|
| 波长 | λ | 1 m |
| 层深 | d_l = d_h | 0.75 m |
| 密度 | ρ_l / ρ_h | 500 / 1500 kg/m³ |
| Atwood 数 | A | 0.5 |
| 重力 | g | 9.81 m/s² |
| 初始振幅 | η₀ | 0.01 m |
| 声速 | c₀ | 40 m/s |
| 运动黏性 | ν | 1e-3 m²/s |

## ④ 对照表（已有工件，返修轮集群计算中）

| dx [m] | 积分器 | δ-SPH | σ_num [s⁻¹] | 相对 √(Agk) | <10% 判据 |
|---|---|---|---|---|---|
| 0.04 | euler | 0 | 2.125 | -61.7 % | FAIL |
| 0.025 | euler | 0 | 3.095 | -44.2 % | FAIL |
| 0.02 | euler | 0 | 3.971 | -28.5 % | FAIL |
| 0.0125 | euler | 0 | 4.192 | -24.5 % | FAIL |
| 0.02 | rk2 | 0 | *集群计算中* | *待填* | *待判* |
| 0.0125 | rk2 | 0 | *集群计算中* | *待填* | *待判* |
| 0.008 | rk2 | 0.05 | *集群计算中* | *待填* | *待判* |

## ⑤ 收敛性分析

原有数据表明数值阻尼近似 O(dx)：
- dx=0.04: σ/σ_th = 0.383
- dx=0.025: σ/σ_th = 0.558
- dx=0.02: σ/σ_th = 0.715
- dx=0.0125: σ/σ_th = 0.755

RK2 时间积分减少时间离散误差（从 O(dt) 到 O(dt²)），但空间误差仍为主导。预期 dx=0.008 + RK2 + δ-SPH 可接近 <10% 门限。

## ⑥ 工件与复现

- 实现：`src/tensorsph/rayleigh_taylor.py`（含 RK2 + δ-SPH）
- 入口：`benchmarks/rayleigh_taylor/run.py`
- 测试：`tests/test_rayleigh_taylor.py`（10 项全部通过）
- 工件：`benchmarks/rayleigh_taylor/runs/*.json`
- 复现：`PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.02 --time-integrator rk2 --density-diffusion 0.05`

## 生态摩擦

1. 控制面 128 核被 ~300 负载占满，本地单算例墙钟波动大；细化档改走集群（cj1287）。
2. SPH 离散固有的界面数值阻尼是系统性问题，需 dx≲0.006 或界面锐化/ISPH 路线才能稳定 <10%。
