# CASE-SPH-OSC-DROP-FREQ · Round 5 — 时间推进臂（dynamic arm）：实现 + honest FAIL

锚点：`CASE-SPH-OSC-DROP-FREQ`（ref: Rayleigh 1879；容差 <6%）。本轮的增量不是"再算一遍准静态"，而是
把第 4 轮明确留白的**时间推进臂**做出来并如实报告结果：**未通过——不是"算出来超门限"，而是"没有产出可信 ω"**。

## ① 案例介绍

半径 $R$、密度 $\rho_0$、表面张力 $\sigma$ 的无粘液滴，被形变为 $r(\theta)=R[1+\varepsilon P_l(\cos\theta)]$
后的自由振荡：$\omega_l^2=l(l-1)(l+2)\sigma/(\rho_0R^3)$（Rayleigh 1879；Lamb 1932 Art. 275），
$l=2$ 时 $\omega=\sqrt{8\sigma/(\rho_0R^3)}$。无量纲化 $R=\rho_0=\sigma=1$ ⇒ $\omega_{ref}=2.8284$。

| 参数 | 值 | 说明 |
|---|---|---|
| $R/dx$ | 7 / 9（动态臂）；24（第 4 轮准静态臂） | 动态臂的成本限制在这里 |
| $\varepsilon$ | 0.05 | 线性区 |
| $c_0$ | 20 | $\rho_0c_0^2=400$，Laplace 压 $\delta\rho/\rho\approx0.5\%$ |
| $\gamma$ | 7（Tait，$p\ge0$ 截断） | Monaghan 1994 |
| $\alpha_{av}$ | 0.02 | Monaghan & Gingold 1983 |
| $\delta_{SPH}$ | 0.1（也有 0 对照） | Antuono et al. 2012 |
| $h$ | $1.3\,dx$，quintic（支持 3h） | 仓库既有核 |

## ② 怎么做的

**精确的 $m=0$ 轴对称归约**（新模块 `src/tensorsph/oscillating_drop_dynamic.py`）。Rayleigh 模态是旋转面，
$v_\phi=0$，于是子午半平面 $(\rho,z)$ 上：

$$\frac{D\mathbf v}{Dt}=-\frac{1}{\rho}\nabla_2 p+\frac{1}{\rho}\mathbf f_{csf},\quad
\mathbf f_{csf}=\sigma\kappa\delta_s\mathbf n,\qquad
\frac{D\rho}{Dt}=-\rho\nabla_2\!\cdot\!\mathbf v-\rho\frac{v_\rho}{\rho_{axis}}$$

即**动量方程完全保持平面形式**，连续性方程多一个柱坐标源项；曲率用 $\kappa=\kappa_{meridional}+n_\rho/\rho$
（与准静态臂 `axisym_curvature` 同一拆分）。对称轴用**镜像粒子**（Colagrossi & Landrini 2003）处理：
每次力评估都在 $(\rho,z)\cup(-\rho,z)$ 的加倍云上做，镜像速度径向分量取反，因而轴上无截断、$v_\rho(0)=0$ 自动成立。

其余零件全部取仓库已验证的那一套：Monaghan (1992) 连续性、δ-SPH 密度扩散、Tait EOS + $p\ge0$ 截断、
对称压力形式动量、Monaghan & Gingold 人工粘性、Brackbill et al. (1992) CSF（`csf_fields` + `fit_quadratic`
曲率 + `legacy` 表面 delta）、kick-drift-kick leapfrog，$\Delta t=0.25\min\{h/(c_0+|v|),\ (\pi/h)^{-3/2}\sqrt{\rho_0/\sigma}\}$。

初始化：六角格子映射到解析形状（比例径向映射 $r\to r[1+\varepsilon P_l]$，避免把原点附近的点推到表面），
质量 2 次不动点标定，密度取"平滑求和密度 × Laplace 压缩因子"
$\rho_0(1+2\sigma/(\rho_0c_0^2R))^{1/\gamma}$——**这一步很关键**：不施加 Laplace 压缩，内部从 $p=0$ 起跳，
表面 CSF 牵引无补偿，启动声学脉冲比模态信号大 1–2 个量级。测量量是面积加权四极投影
$A_l(t)=\sum_i dA_i r_i^2P_l(\cos\theta_i)/\sum_i dA_i$ 归一化到 $\varepsilon$，$\omega,\gamma$ 由
`fit_damped_cosine` 拟合。

工具链：pytest（9 passed / 1 skipped），matplotlib 出图，全部提交经调度系统（本机只跑 <10 min 短测）。

## ③ 计算结果（程序化，见 `benchmarks/oscillating_drop_freq/round5_dynamic.py` 与 `tests/`）

| 臂 | 设置 | 结果 |
|---|---|---|
| A：$\sigma\to0$ 对照（$R/dx=9$，50 步） | 只留压力+连续性+轴源项 | $\lvert v\rvert_{max}$ 从 0 涨到 **214**（$R/dx=7$ 时 **281**）——**不稳定** |
| B：δ-SPH 对照（$\sigma\to0,\varepsilon=10^{-3}$，25 步） | 静止云 + 只开扩散 | $\lvert\Delta\rho\rvert$ 达 **1.7 %**（$R/dx=7$）/0.5 %/步（$R/dx=9$）——**自由面被人为"充气"** |
| C：毛细臂（$\sigma=1,\varepsilon=0.05$，$R/dx=9$，400 步 ≈ 0.05 周期） | 测量臂 | $\lvert v\rvert_{max}$ 0.013→**619**，振幅 $\varepsilon$→**0.66**，$\rho_{max}$→**6.4ρ₀**，$\Delta t$ 1.8e-3→5.6e-5 |
| D：毛细臂关扩散 | δ-SPH=0 | 同样发散（减缓但未愈） |
| 短短线（90 步）拟合 | | $\omega_{meas}=37.2$ ⇒ 误差 **+1216 %**（无意义，说明拟合不应被引用） |

## ④ 结果图

`docs/figures/case_sph_osc_drop_freq_round5.png`
左：$\varepsilon=0.05$ 毛细臂的模态振幅轨迹（虚线=线性模态应有值；实测迅速离开线性区，不是振荡）。
右：三组配置 $\sim0.1$ 时间单位后的 $\lvert v\rvert_{max}$（对数坐标）与模态速度尺度
$\omega\varepsilon R=0.141$ 的对比——爆掉 2–3 个量级。

## ⑤ 注意点 / 局限（踩过的坑）

1. **一条被撤回的观察**：手写诊断循环里每步重设密度后 $\lvert v\rvert\sim10^{-11}$，看起来"$\sigma\to0$ 稳定"。
   那不是积分，是重初始化。按真正的驱动逻辑跑，**无表面张力也发散**，因此第 5 轮的结论必须写成
   "压力/连续性子系统本身未验证"，不能把责任全推给 CSF 曲率算子。
2. **δ-SPH 在这里是反向作用**：自由面的 $\sim6\%$ 密度亏空是**截断亏空**，不是物理梯度；扩散把它当梯度去"抹平"，
   等于在自由面注质量。已作为可复现测试固定（`test_delta_sph_inflates_the_free_surface`）。
3. **CSF 带内曲率噪声仍在**：界面带上仍能取到 $\kappa R\approx-16.6$ 而 $\delta_s=0.42$ 的粒子（表面平均 $\kappa R=1.95$），
   单粒子伪牵引 $\sigma\kappa\delta_s/\rho\approx7$ vs 模态加速度 $\omega^2\varepsilon R=0.4$。
4. **时间步不是元凶**：全程 $\Delta t$ 由声学 CFL 自动收紧并始终被遵守，爆掉发生在物理空间。
5. **未做**：系综/多取向、$R/dx\ge12$、$c_0$ 敏感性、能量守恒审计（能量法可以定位注入通道，是下一步首选）。

## ⑥ 误差判定（逐项，明确回答"是否 <6%"）

| 关键量 | 结果 | 基准 | 误差 | <6%？ |
|---|---|---|---|---|
| 动态 $\omega$（$l=2$，$R/dx=9$） | 无可信值（发散） | Rayleigh 2.8284 | — | **无（FAIL-to-instrument）** |
| 四极估计器线性度（$\varepsilon=0.01/02/04$） | 比值 1.00±2 % | 线性 | ≤2 % | 是（估计器本身） |
| 初始态 Laplace 压缩（内部中值） | 0.9985 vs 1.0007 | 解析 | 0.2 % | 是 |
| 自由面初始密度 | <0.99ρ₀ | 自由面 p=0 | 定性 | — |
| 3D/2D Rayleigh 分支比 | $\sqrt{8/6}=1.1547$ | 闭式 | 0（被 test 钉住） | 是 |

**honest 结论**：`CASE-SPH-OSC-DROP-FREQ` 在动态侧**仍未闭合**，本轮产出的是
(a) 精确轴对称归约与可复用基础设施，(b) 一组可复现的失效测试（把"爆掉"钉住，修好必须翻转它们），
(c) 定位信息：失效在压力/连续性时间推进，而非仅仅 CSF 曲率算子；δ-SPH 在毛细自由面上有害。
不建议把锚点置 done。

## ⑦ 复现

```bash
python -m pytest tests/test_oscillating_drop_freq_round5.py -q          # 约 25 s
PYTHONPATH=src python benchmarks/oscillating_drop_freq/round5_dynamic.py --quick
PYTHONPATH=src python benchmarks/oscillating_drop_freq/round5_dynamic.py  # 递交调度系统
SPH_ROUND5_FULL=1 python -m pytest -k gate_target  # 目标门测试（当前 xfail）
```
