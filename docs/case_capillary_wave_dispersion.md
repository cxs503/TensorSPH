# CASE-CAPILLARY-WAVE: 毛细波色散算例（TensorSPH，算例池）

*由 `benchmarks/case_capillary_wave_dispersion/make_doc.py` 从 `runs/case_capillary_wave_dispersion/report.json` 程序化生成，勿手改。*

**算例状态：PASS** ｜ 容差：<10 % ｜ 判据：A_design_point_within_tol=True；B_resolution_ladder_within_tol=True；C_delta_s_integrates_to_one=True；D_sigma_linearity=True

## 1. 案例介绍（物理背景 / 几何 / 参数表 / ref 出处）

两层不可压无粘流体界面上的一条正弦毛细波（无重力）：界面 `eta(x) = -A cos(kx)`，下水深 `h1`、上气层厚 `h2`，横向一个波长、周期边界。这是一维线性色散问题的二维截面，参考解是精确色散关系（闭式）。

**ref 出处（零手抄）**：Lamb, H. (1932). Hydrodynamics, 6th ed., Cambridge University Press, Art. 267 (capillary waves, deep water, inviscid: omega^2 = T k^3 / rho).

* 二维有限深双层色散（本单用的完整式）：`omega^2 = sigma*k**3/(rho1*coth(k*h1) + rho2*coth(k*h2))`
* 该式的两个经典极限在运行时被断言（`ref.py::reduction_checks`）：
  * `rho2 -> 0, k h1 -> inf` → Lamb Art. 267 深水式，相对偏差 0.00064468
  * `rho2 -> 0` 有限深 → `omega^2 = sigma k^3 tanh(k h1)/rho1`（g=0 的重力-毛细分支），相对偏差 0.00060165
* ref 数值由 mpmath 在 dps=50 下对该闭式求值产生（不是抄表），`ref.json` sha256 = `fa18af599279ca11c84828747f8d64e7b89ed6a41acbd8cdceb178df426edf9d`
* `omega_ref` = 47.218370 rad/s；`T_ref` = 1.330665e-01 s；`c_ref` = 0.150301 m/s

参数表（无量纲数完整）：

| 量 | 值 | 无量纲化 |
|---|---|---|
| 波长 `lambda` | 0.02 m | 基准尺度 |
| 波数 `k = 2pi/lambda` | 314.159 1/m | `k a` = 0.1257（陡度） |
| 表面张力 `sigma` | 0.072 N/m | `Oh = mu/sqrt(rho sigma lambda)` = 0（无粘） |
| 液体密度 `rho1` | 1000.0 kg/m^3 | 密度比 `rho2/rho1` = 1.2e-03 |
| 气体密度 `rho2` | 1.2 kg/m^3 | 上式 |
| 液体深 `h1` | 0.016 m | `k h1` = 5.027（coth→1.000） |
| 气层厚 `h2` | 0.01 m | `k h2` = 3.142 |
| 波幅 `A` | `0.02 lambda` | `kA` = 0.1257 |
| 核支持 | `h/dx` = 1.3 | 曲率拟合 stencil = 4.0 h |
| 重力 | 0 | 键数 `Bo = rho g lambda^2/sigma` = 0（纯毛细） |

## 2. 方法与实现

复用 TensorSPH 现成能力（不另起炉灶）：`tensorsph.surface_tension_dispersion::exact_wave_stress_coeff`（有限振幅精确曲率×弧长模态系数）、`csf_consistent::{mls_gradient, shepard_smooth, mls_divergence}`、`csf_curvature::fit_curvature`、`neighbor_search::cell_linked_pairs`、`droplet_impact::cubic_spline_kernel`。算例只补两件事：两相（双流体）几何构造器（与既有 `benchmarks/two_phase_laplace` 同一配方）与在该几何上的模态投影。

测量原理：线性区界面恢复力只有法向应力跳变，惯性由精确势流给出，故
`omega_got^2/omega_ref^2 = dp_meas/dp_theory`，`dp_meas = (2/L) Σ_i (-sigma kappa_i delta_s,i V_i) cos(k x_i)`，`dp_theory = sigma·(eta=-A cos kx 的精确曲率×弧长在模态 k 上的系数)`。`delta_s = |grad c|`（两相归一，`int delta_s dn = 1`）。

一键复现：

```
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/case_capillary_wave_dispersion/run.py \
    --out-dir runs/case_capillary_wave_dispersion
```

## 3. 计算结果（ref vs got + 误差逐项）

| 曲率算子 | ppw | dx [mm] | omega_ref [rad/s] | omega_got [rad/s] | 误差 | 刚度比 ± 相位标准差 | δs 积分 | 判定 |
|---|---|---|---|---|---|---|---|---|
| fit_quadratic | 16 | 1.250 | 47.218370 | 35.757073 | -24.27 % | 0.5763 ± 0.0802 | 1.0115 | ❌ |
| fit_quadratic | 24 | 0.833 | 47.218370 | 36.729950 | -22.21 % | 0.6059 ± 0.0447 | 1.0081 | ❌ |
| fit_quadratic | 32 | 0.625 | 47.218370 | 38.774200 | -17.88 % | 0.6807 ± 0.1315 | 1.0092 | ❌ |
| fit_quadratic | 48 | 0.417 | 47.218370 | 38.224034 | -19.05 % | 0.6557 ± 0.0315 | 1.0081 | ❌ |
| mls | 16 | 1.250 | 47.218370 | 46.045062 | -2.48 % | 0.9551 ± 0.1254 | 1.0115 | ✅ |
| mls | 24 | 0.833 | 47.218370 | 45.224510 | -4.22 % | 0.9186 ± 0.0674 | 1.0081 | ✅ |
| mls | 32 | 0.625 | 47.218370 | 47.112077 | -0.23 % | 1.0049 ± 0.1932 | 1.0092 | ✅ |
| mls | 48 | 0.417 | 47.218370 | 46.013341 | -2.55 % | 0.9501 ± 0.0448 | 1.0081 | ✅ |

判据表：

| 判据 | 要求 | 实测 | 通过 |
|---|---|---|---|
| A_design_point_within_tol | |omega_got/omega_ref - 1| < 10 % at ppw=32, curvature=mls | 0.2251 | True |
| B_resolution_ladder_within_tol | all ppw in (16, 24, 32, 48) within 10 % (a passing point that diverges under refinement is not evidence) | 4.2226 | True |
| C_delta_s_integrates_to_one | sum_i delta_s,i V_i / L = 1 +- 5 % (Brackbill unit mass) | 1.0092 | True |
| D_sigma_linearity | stiffness ratio constant under sigma x (0.5, 1.0, 2.0) (spread < 5 %) | 0.0000 | True |

图（matplotlib，程序生成）：
* `runs/case_capillary_wave_dispersion/figures/error_vs_resolution.png` — Frequency error vs particles per wavelength for the two shipped curvature operators; the shaded band is the 10 % case tolerance.
* `runs/case_capillary_wave_dispersion/figures/curvature_profile.png` — Measured CSF curvature along one wavelength vs the exact profile of the perturbation.
* `runs/case_capillary_wave_dispersion/figures/sensitivity.png` — Amplitude and surface-tension sensitivity of the measured dispersion.

## 4. 收敛性 / 参数敏感性（至少两点）

**分辨率阶梯**：`mls` 曲率在 ppw=16/24/32/48 上误差最差 4.22 %（ppw=24），无随加密发散的趋势；`fit_quadratic` 在 17–24 % 之间系统性偏低（刚度比 0.58–0.68），是本算例的**对照档**：它证明误差不是分辨率的函数而是曲率算子的偏置。

**振幅阶梯**（ppw=32）：

| A/lambda | omega_got/omega_ref | 刚度比 | 误差 |
|---|---|---|---|
| 0.01 | 0.6331 | 0.8016 | -36.69 % |
| 0.02 | 0.9977 | 1.0049 | -0.23 % |
| 0.05 | 0.9592 | 0.9211 | -4.08 % |

→ 线性响应测量在 `A/lambda` 降到 0.01 时失效（曲率噪声底与信号同量级），0.02–0.05 区间稳定。这是测量方法的适用范围，不是判据。

**σ 线性**（sigma×0.5/1/2）：刚度比展开 0.0000 % → 算子对 σ 严格线性。

## 5. 注意点与局限

1. 这是**线性响应（离散色散）**测量：`omega` 由模态法向应力比映射而来，不是时间推进的频谱。时间推进路线在本仓 HEAD 不可用（单相自由表面伪加速度 ≈ 波自身 `omega^2 A` 的 23 倍，`docs/verification_capillary_wave_open_surface.md`），两相时间推进**无求解器**（NOT_SUPPORTED）。
2. 单相自由表面（`benchmarks/surface_tension_dispersion`、`benchmarks/capillary_wave`）在本算例容差下**不可用**：其 `delta_s = 2|grad c|` 只携带 Brackbill 单位质量的一半，曲率噪声随 `1/dx` 增长（本单复核：ppw=24→48 误差 23 %→93 %，随加密发散）。两相构造把 `int delta_s dn` 修回 1.0（本单实测 1.0092）后误差降到 <5 %。
3. 气相粒子只为补全颜色场 0→1 过渡而存在；静态测量不消费其惯性，参考惯性取精确双层值 `rho1 coth(k h1) + rho2 coth(k h2)`（相对单相水仅 0.064 %）。
4. 相位集合平均用 2×2=4 个点阵相位，刚度比的标准差 0.04–0.19 → 单点误差带 ±2–4 %；这是当前方法的可复现性底噪。

## 6. 失败模式与工件

* 失败模式 1：单相自由表面算子 → 误差 23–93 % 且随加密发散（本单实测，见 §5.2）。
* 失败模式 2：`fit_quadratic` 曲率在两层界面上 → 系统性 −18…−24 %（刚度比 0.58–0.68）。
* 失败模式 3：`A/lambda = 0.01` → 误差 −37 %（信噪比不足）。
* 工件：`runs/case_capillary_wave_dispersion/{ref.json, report.json, benchmark.json, summary.md, meta.json, figures/*.png}`；`ref.json` 与 `report.json` 的 sha256 记在 `meta.json`。
* 环境：3.11.6 / torch 2.7.1 / loongarch64 / threads=4 / git d99d3d5556；墙钟 0.2 min。

