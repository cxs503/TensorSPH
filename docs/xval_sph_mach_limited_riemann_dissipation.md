# XVAL-SPH-MACH-LIMITED-RIEMANN-DISSIPATION — 第2轮：可运行 + 对标合格

> 锚点：`XVAL-SPH-MACH-LIMITED-RIEMANN-DISSIPATION`（Xiangyu-Hu/SPHinXsys 的
> mach-limited riemann dissipation 做法）。验收：对标 Sod 精确解误差 <15%（owner 标准），
> 目标 <3%。本文件由 `benchmarks/xval_mach_limited_riemann/make_doc.py` 从
> `results/results.json` **程序化生成**，表中每个数字都可溯源到该 JSON。

**结论一句话**：第1轮 honest-FAIL（t≈0.014 崩、0 快照）的两个成因都已定位并修复——
(i) 粒子位置未随流动对流（拉格朗日 SPH 积分成了欧拉冻结网格）；(ii) Mach 限制用单粒子
`|v|/c`，在静止气体里恒为 0，恰好在激波要形成处把粘性关掉。修好后 Sod 全程跑通，
星区压力/密度/速度对文献值误差 ≤+0.60%，
判据 <15% 通过、<3% 目标亦通过。低 Mach 光滑流动的虚假耗散：限 Mach 后 ν 从
1.33e-02 降到 9.37e-04（14.2× 降低），
已接近离散格式自身的本底 3.34e-04。

## ① 案例介绍（物理背景 / 几何 / 参数表）

**算例 A（主验收）Sod (1978) 激波管**：一维理想气体 Riemann 问题，隔板位于 x=0.5，
左右态压力比 10:1、密度比 8:1。左行稀疏波、接触间断、右行激波三波结构有精确解，
是检验可压缩 SPH 耗散算子（激波捕捉 vs 光滑区过度耗散）的标准算例。

| 项目 | 取值 |
|---|---|
| 计算域 | x ∈ [0, 1]，反射壁（镜像虚粒子） |
| 左态 (ρ, u, p)_L | (1.0, 0.0, 1.0) |
| 右态 (ρ, u, p)_R | (0.125, 0.0, 0.1) |
| 比热比 γ | 1.4 |
| 终止时刻 | t_end = 0.2 |
| 粒子数 N | 200 / 400 / 800 / 1600 |
| 光滑长度 | h = 2.5 dx（三次样条核，1D，支撑 2h） |
| 初始间断 | tanh 抹平 1.5 dx |
| CFL | 0.15（另加粘性步长限制） |

文献参考值（Sod 1978 / Toro 1999 Table 4.1，星区）：

| 量 | 参考值 |
|---|---|
| p* | 0.30313 |
| u* | 0.92745 |
| ρ*_L | 0.42632 |
| ρ*_R | 0.26557 |

**算例 B（区分性诊断）封闭盒内线性驻声波**：均匀气体 ρ0=p0=1（c0=√(γp0/ρ0)），
初始取驻波本征函数 u = A c0 cos(kx)、ρ=ρ0（k=π/L，A=0.02 即 M=0.02，6 个周期）。
模态声能 E_ac = Σ m(½u² + ½ p'²/(ρ0c0²)) 对无损本征模态严格守恒，因此其对数斜率给出
有效粘性 ν = -slope(E_ac)/k²。Sod 无法区分各耗散模型（全是压缩主导），此算例可以。

## ② 怎么做的（方法 / 流程 / 关键设置）

拉格朗日可压缩 SPH（理想气体 p=(γ-1)ρe，内能 e 独立求解）：

```
dρ_i/dt = Σ_j m_j (v_i - v_j)·∇_i W_ij
dv_i/dt = -Σ_j m_j Q_ij ∇_i W_ij
de_i/dt = ½ Σ_j m_j Q_ij (v_i - v_j)·∇_i W_ij
Q_ij = (p_i + p_j)/(ρ_i ρ_j) + Π_ij            （算术平均压力形式，静态接触连续）
```

四种耗散模型（同一框架下只换 Π_ij）：

| 模型 | Π_ij |
|---|---|
| av-full | Monaghan 线性+二次：(-α c̄ μ + β μ²)/ρ̄，α=1，β=2，μ=h v_ij·r_ij/(r²+η²h²) |
| av-mach-abs | 同 av-full，但 α_eff = α·max(min(|v_i|/c_i, |v_j|/c_j, 1), 0.01)（**单粒子绝对 Mach，第1轮**） |
| av-mach-rel | 同 av-full，但 α_eff = α·max(min(M_ij, 1), 0.01)，M_ij = max(-v_ij·e_ij, 0)/c̄（**成对相对法向 Mach，本单**） |
| riemann-rel | 声学(线性) Riemann 界面压力 Π = -½·M_ij'·c̄·u_rel/ρ̄，p* = p̄ - ½ρ̄c̄u_rel（Godunov-SPH/Parshikov-Inutsuka） |

关键设置：RK2 中点法；`x` 每步随流动对流（**拉格朗日**，第1轮漏掉的就是这一步）；
两侧镜像虚粒子（压力/密度/内能镜像、速度反号）保证壁面对称；质量 m=ρ0dx 冻结；
每步重算邻居对（KDTree）。

工具链：`src/tensorsph/compressible/riemann_dissipation.py`（实现 + `run_sod_riemann` /
`run_acoustic_pulse`）、`benchmarks/xval_mach_limited_riemann/run.py`（Study driver，
输出 results.json + 4 图）、`tests/test_riemann_dissipation.py`（22 项测试）。

流程：先复现第1轮崩点 → 定位两个成因 → 修补 → 四模型 Sod 矩阵 + 分辨率梯次 + 声学诊断 →
判据判定 → 报告。

## ③ 计算结果

### 3.1 Sod 激波管（t=0.2）：四模型对照

| 模型 | N | 步数 | 完成 | L1(ρ) | L1(p) | L1(u) | 稀疏波区 L1(ρ) | 激波厚(10-90%)/h | dt_min | 总能量相对变化 |
|---|---|---|---|---|---|---|---|---|---|---|
| av-full | 400 | 464 | True | 0.01137 | 0.01426 | 0.02865 | 0.02344 | 11.14 | 3.84e-04 | -2.88e-06 |
| av-full | 800 | 934 | True | 0.00869 | 0.01073 | 0.01990 | 0.02546 | 25.27 | 5.51e-05 | -1.47e-06 |
| av-mach-abs | 400 | 465 | True | 0.01054 | 0.01324 | 0.02354 | 0.02342 | 11.14 | 1.72e-05 | -2.54e-06 |
| av-mach-abs | 800 | 934 | True | 0.00830 | 0.01025 | 0.01746 | 0.02546 | 25.27 | 1.58e-04 | -1.30e-06 |
| av-mach-rel | 400 | 474 | True | 0.01127 | 0.01447 | 0.03486 | 0.02343 | 11.14 | 2.39e-04 | -2.60e-06 |
| av-mach-rel | 800 | 954 | True | 0.00885 | 0.01117 | 0.02519 | 0.02551 | 25.27 | 2.07e-04 | -1.38e-06 |
| riemann-rel | 400 | 533 | True | 0.01394 | 0.01913 | 0.06856 | 0.02343 | 10.27 | 2.26e-04 | -1.89e-06 |
| riemann-rel | 800 | 1081 | True | 0.01064 | 0.01468 | 0.05416 | 0.02553 | 25.26 | 1.45e-04 | -1.18e-06 |

### 3.2 星区量 vs 文献值（N=800）

| 量 | 文献值 | 计算值 | 相对误差 | <3%？ |
|---|---|---|---|---|
| p_star (av-full) | 0.30313 | 0.30130 | -0.60% | 是 |
| u_star (av-full) | 0.92745 | 0.92783 | +0.04% | 是 |
| rho_star_L (av-full) | 0.42632 | 0.42416 | -0.51% | 是 |
| rho_star_R (av-full) | 0.26557 | 0.26547 | -0.04% | 是 |
| p_star (av-mach-abs) | 0.30313 | 0.30130 | -0.61% | 是 |
| u_star (av-mach-abs) | 0.92745 | 0.92785 | +0.04% | 是 |
| rho_star_L (av-mach-abs) | 0.42632 | 0.42424 | -0.49% | 是 |
| rho_star_R (av-mach-abs) | 0.26557 | 0.26543 | -0.05% | 是 |
| p_star (av-mach-rel) | 0.30313 | 0.30130 | -0.60% | 是 |
| u_star (av-mach-rel) | 0.92745 | 0.92786 | +0.04% | 是 |
| rho_star_L (av-mach-rel) | 0.42632 | 0.42426 | -0.48% | 是 |
| rho_star_R (av-mach-rel) | 0.26557 | 0.26537 | -0.07% | 是 |
| p_star (riemann-rel) | 0.30313 | 0.30117 | -0.65% | 是 |
| u_star (riemann-rel) | 0.92745 | 0.92825 | +0.09% | 是 |
| rho_star_L (riemann-rel) | 0.42632 | 0.42456 | -0.41% | 是 |
| rho_star_R (riemann-rel) | 0.26557 | 0.26579 | +0.08% | 是 |

接触间断两侧压力跳跃（应力求 ~0）：

| 模型 | N=400 | N=800 |
|---|---|---|
| av-full | 0.0024 | 0.0070 |
| av-mach-abs | 0.0015 | 0.0067 |
| av-mach-rel | 0.0013 | 0.0061 |
| riemann-rel | 0.0051 | 0.0081 |

### 3.3 分辨率梯次

| 模型 | N=200 | N=400 | N=800 | N=1600 | 拟合阶数 |
|---|---|---|---|---|---|
| av-mach-rel | 0.01743 | 0.01127 | 0.00885 | 0.00788 | 0.38 |
| av-full | 0.01887 | 0.01137 | 0.00869 | 0.00775 | 0.42 |

### 3.4 低 Mach 光滑流动的虚假耗散（区分性诊断）

| 模型 | ν_measured | 相对本底倍数 | E_ac(end)/E_ac(0) | 相对 av-full |
|---|---|---|---|---|
| inviscid-floor | 3.34e-04 | 1.0× | 0.9733 | 39.9× 更小 |
| av-full | 1.33e-02 | 39.9× | 0.2718 | 1.0× 更小 |
| av-mach-abs | 9.16e-04 | 2.7× | 0.9230 | 14.6× 更小 |
| av-mach-rel | 9.37e-04 | 2.8× | 0.9222 | 14.2× 更小 |
| riemann-rel | 1.47e-03 | 4.4× | 0.8852 | 9.1× 更小 |

解读：av-full 的一阶人工粘性把 6 个周期的声波能量打掉 73%；相对 Mach 限制只打掉 8%，与本底（α=0，3%）同量级。

### 3.5 图

- `benchmarks/xval_mach_limited_riemann/results/fig1_sod_profiles.png` — Sod t=0.2 profiles, N=800
- `benchmarks/xval_mach_limited_riemann/results/fig2_error_profiles.png` — density error vs exact solution
- `benchmarks/xval_mach_limited_riemann/results/fig3_convergence.png` — L1 density error vs dx
- `benchmarks/xval_mach_limited_riemann/results/fig4_acoustic_damping.png` — acoustic modal energy decay

## ④ 注意点（坑 / 局限 / 适用范围）

1. **第1轮 honest-FAIL 的两个真实成因**（都是缺陷，不是「SPH 的已知挑战」）：
   (a) `run_sod_riemann` 的主循环只积分 ρ/u/e，**不更新粒子位置**，连续性方程在冻结网格上
       累积误差，16 步内 dt 塌到 <1e-14（复现：`run_sod_riemann(RiemannDissipationConfig(
       n_particles=400, mach_limit_mode='absolute'))` 在第1轮代码上返回 0 快照）；
   (b) Mach 限制用单粒子 `|v|/c`：Sod 初始 v≡0 → α_eff=α·floor=0.01，激波要形成的地方没有耗散。
2. **能量不是好的判据**：成对对称的能量方程使总能量在耗散存在时仍守恒到 ~1e-6 量级
   （表中总能量相对变化列），所以不要把「能量守恒」当作耗散是否正确的证据；用 L1/星区量/声学 ν。
3. **L1 收敛阶 0.38–0.42（次线性）**：h=2.5dx 固定时误差由被抹平的接触/激波区主导，
   星区量收敛远快于此。不要宣传「约一阶」；正式阶数需要 h 与 dx 同时收敛的梯次（未做，属欠账）。
4. **dt_min 的瞬态**：av-full 在 N=800 出现 dt_min=5.5e-5（远低于 CFL 的 5.5e-4），是初始压力
   间断处的瞬态振荡；相对 Mach 限制型 dt_min 稳定在 2e-4 量级，这是「过度耗散型方案会自激」的
   一个侧面证据。
5. **不是 SPHinXsys 源码移植**：本仓 OSS 快照 `docs/xval_sphinx_tvc/oss_source/` 里没有
   SPHinXsys 的 dissipative/riemann solver 头文件，因此算子是按公开做法（Godunov-SPH /
   Ji et al. 2019 的 Mach 缩放思想）自行实现的，命名与 SPHinXsys 一致只表示「同类做法」，
   不是逐行对齐。若后续拿到上游源码，应做逐项对照（欠账）。
6. **适用范围**：1D、理想气体、反射壁；未验证 2D/3D、真实气体、真空出现（ρ→0）的强膨胀。
   riemann-rel 在速度场 L1 上明显更差（0.054 vs 0.025），故选默认方案为 av-mach-rel。

## ⑤ 误差判定（逐项回答「是否 <3%」）

默认方案（av-mach-rel，N=800）：

| 关键量 | 相对误差 | <3%？ | <15%（owner 标准）？ |
|---|---|---|---|
| ρ*_L vs 0.42632 | -0.48% | 是 | 是 |
| ρ*_R vs 0.26557 | -0.07% | 是 | 是 |
| u* vs 0.92745 | +0.04% | 是 | 是 |
| p* vs 0.30313 | -0.60% | 是 | 是 |
| L1(ρ) 全场的相对 L1 | +0.89% | 是 | 是 |
| L1(ρ) 稀疏波（低 Mach 光滑区） | +2.55% | 是 | 是 |
| L1(p) 全场 | +1.12% | 是 | 是 |
| evaluate_metrics 的最大星区相对误差 | +0.72% | 是 | 是 |

结论：**判据通过**。owner 精度标准（<15%）在 N=400 与 N=800 全部四模型上均满足；
目标 <3% 在 N=800 的默认方案上满足（最大项为 L1(ρ) 稀疏波区）。
不满足项如实列出：L1 收敛阶 0.38–0.42（无「阶数」验收要求，但不应宣称一阶）；
接触间断压力跳跃 0.61%（N=800，非零但小）；上游源码逐行对齐未做。

## ⑥ 第1轮记录（honest-FAIL，保留不覆盖）

第1轮（mission m17732）在 t≈0.014 崩、`steps=15`、0 快照，报告结论为
「强间断导致密度趋零、加速度爆炸＝SPH 已知挑战」。本单复现后确认该结论**归因错误**：
真实成因是 ② 中的 (a)(b) 两个实现缺陷，与强间断本身无关（修好后 N=200…1600 全部稳定，
ρ_min 始终 >0.1）。第1轮已验证的部分（TRRS 星区压力 ~1.2%、Roe 平均对称性、均匀流静止）
仍然成立，已并入本单测试集。

## 复现命令

```bash
cd <worktree>
PYTHONPATH=src python3 benchmarks/xval_mach_limited_riemann/run.py   # ~4 min, 单核
python3 benchmarks/xval_mach_limited_riemann/make_doc.py              # 重新生成本文件
PYTHONPATH=src python3 -m pytest tests/test_riemann_dissipation.py -q  # 22 passed
```

