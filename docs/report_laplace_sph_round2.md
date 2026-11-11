# CASE-LAPLACE-SPH 第2轮：Δp=σ/R 在投影（不可压）SPH 液滴上的验证

## 1. 案例介绍
二维静止圆形界面，Young-Laplace 律 Δp=σ/R。σ=0.072 N/m（水-空气 20℃），R=0.02 m，ρ0=1000.0 kg/m³，**ref（解析）：Δp=σ/R=3.6000 Pa**。维度数：R/dx=[10.0, 13.3, 20.0]，R/h=[7.69, 10.26, 15.38]，Bo=0（无重力），We=0（静力学无惯性）。

**前几轮为何必然失败（本轮核心调研结论）**：本仓液滴的压力来自弱可压缩 Tait p=B((ρ/ρ0)^γ−1)，B=ρ0c0²/γ=57142.9 Pa。承载 Δp=3.6 Pa 需要密度比 Δp/B=6.30e-05（R=0.1 m 几何更低至 1.8e-6，见 benchmarks/laplace_droplet/resolvability_audit），而 WCSPH 密度场噪声约 1e-3 —— 信号比噪声低一个数量级以上，核/曲率修补都无解。既有证据吻合：一致化 CSF 的松弛 WCSPH 误差 -78.6%（benchmarks/csf_consistent/validation_report.json），最优自由面闭合变体 -20.1%（docs/verification_laplace_pressure_closure.md）。这是 EOS 路线问题，不是调参问题。

## 2. 方法与实现（含出处）
开源横向调研：oss_inventory / oss_solutions 中唯一 SPH 仓 Xiangyu-Hu/SPHinXsys（GAP-SPH-MULTIPHASE，src/shared/particle_dynamics/fluid_dynamics/surface_tension.h）用 CSF surface-tension 动力学 + 密度梯度动力学，多相做成 material 层；其 Laplace 压差同样取自压力场而非 EOS 读数。方法谱系：Brackbill/Kothe/Zemach 1992（CSF，δs=|∇c|）；Cummins & Rudman 1999（投影/不可压 SPH 的泊松压力）。选型 = 把压力从 EOS 读数换成拉格朗日乘子：
```
u* = u + dt a_CSF ;   -∇²p = -ρ ∇·a ;  p=0 于外层载流子边界 ;  u = u* - dt ∇p/ρ
```
算子全部复用本仓：Shepard 平滑相指示色函数（∫|∇c|dn=1）→ MLS 一阶一致梯度 → 法向二次 Shepard 平滑 → κ=∇·n → δs=|∇c| → CSF 加速度；压力用 Cummins-Rudman 配对 Laplacian + 无矩阵 CG。代码指针：src/tensorsph/laplace_isph.py（laplace_isph_case / pressure_poisson / relax_projection）。

## 3. 参数表
| 参数 | 值 |
|---|---|
| σ | 0.072 N/m |
| R | 0.02 m |
| ρ0 | 1000.0 kg/m³ |
| 气相加载体厚度 | 0.012 m（同密度，仅补全 1→0 色函数，**不是多相流模型**）|
| h/dx | 1.3 |
| 容差 | 5%（CASE-LAPLACE-SPH）|
| Δp/B（WCSPH c0=20 m/s）| 6.30e-05 |

## 4. 对照表（逐项 ref vs got）
| R/dx | R/h | N | Δp 实测 [Pa] | Δp 理论 [Pa] | 比值 | 误差 |
|---|---|---|---|---|---|---|
| 10.0 | 7.69 | 1203 | 3.6350 | 3.6000 | 1.0097 | +0.97 % |
| 13.3 | 10.26 | 2150 | 3.6218 | 3.6000 | 1.0061 | +0.61 % |
| 20.0 | 15.38 | 4773 | 3.5923 | 3.6000 | 0.9979 | -0.21 % |

判据逐条：laplace_error_finest ✅；laplace_error_all_levels ✅；gauge_self_test ✅；cross_check_flux ✅；bulk_balance ✅；negative_control_sigma0 ✅；spurious_currents_adami_0p1 ❌。

| 判据 | 实测 | 限值 | 结论 |
|---|---|---|---|
| `laplace_error_finest` | 0.0021326281493778687 | 0.05 | PASS |
| `laplace_error_all_levels` | 0.009714066814682676 | 0.05 | PASS |
| `gauge_self_test` | 0.0028563007471908985 | 0.03 | PASS |
| `cross_check_flux` | 0.00597318851754669 | 0.03 | PASS |
| `bulk_balance` | 0.00582913613495103 | 0.05 | PASS |
| `negative_control_sigma0` | 0.0 | 1e-10 | PASS |
| `spurious_currents_adami_0p1` | nan | 0.1 | FAIL |

消融（最细网格）：model 0.9979；κ_exact 0.9948；δ_unit 1.0047；gauge 1.0029。二次曲面拟合曲率在此载流子上反而更差（Δp 误差 -24.5%，κ 误差 -29.8%），与单向自由面载流子上的 +25% 结论相反 —— 原因已定位：单向面时 ∇·n 丢掉壳体外半层，完整载流子 + 二次平滑法向下 ∇·n 更优。

## 5. 收敛性与参数敏感性
最细两级观测阶 2.58（误差已在仪器地板，非干净截断阶）。
气相采样带 r/R ∈ {(1.15,1.35),(1.25,1.45),(1.35,1.55)} → 比值 [0.998, 0.9979, 0.9976]（散布 ≲1%，说明不是采样伪迹）；h/dx=1.2/1.3/1.4 → [0.9761, 0.9979, 1.0202]；拟合 stencil 2/3/4/5h → κ 误差 [-107.6, -53.0, -29.8, -18.4]%（单调收敛 ⇒ 小 stencil 欠分辨）。

**诚实 FAIL 项**：投影时间推进不稳定 —— 0.5 个毛细时间时 max|u|/v_ref=0.889（Adami 限值 0.1），第 13 步发散。原因（实测非猜测）：投影只移除 CSF 加速度的**有散部分**，残差主体无散（界面壳中位残差 0.362，壳外仅 0.0058），故静态压力场确实平衡（壳外 <1%），而时间推进发散。下一步：伴随一致的散度/梯度配对 + ghost 粒子边界，或半隐式毛细项。

## 6. 误差判定：是否 <3%？
**静力学 Δp 关键量：是** —— 最细 0.21%，全三级 ≤0.97%，仪器自检 0.29%，独立通量路线 0.60%，全部 <3% 且远优于 5% 验收线。**动力学附流指标：否**（FAIL），已如实登记，未粉饰。

## 注意点与工件
- 踩坑1：不能从点式加速度或 Tait 密度差测 Laplace 跳（前者量纲吃 |∇c|，后者信号 < 噪声）。
- 踩坑2：气相「同密度」只是色函数补全装置，不可当多相流能力宣称。
- 踩坑3：二次曲面拟合在本载流子需更宽 stencil 才收敛，跨载流子曲率结论不可直接搬运。
- repro：`PYTHONPATH=src python benchmarks/laplace_sph/run.py --figure`；`PYTHONPATH=src python -m pytest tests/test_laplace_sph_isph.py -q`（11 passed）。
- 工件：src/tensorsph/laplace_isph.py、benchmarks/laplace_sph/{run.py,results.json,make_doc.py,publish_report.py,laplace_sph_isph.png}、docs/verification_laplace_sph_isph.md。
- 方法选型已沉淀 oss_solutions（feature_slug='CASE-LAPLACE-SPH'，id 399）。
