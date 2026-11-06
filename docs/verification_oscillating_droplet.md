# 验证欠账复核：2-D 振荡液滴（自由振荡频率）vs Rayleigh–Lamb 解析解

**仓库**: TensorSPH ｜ **工件**: `benchmarks/oscillating_droplet/` ｜ **测试**: `tests/test_oscillating_droplet.py`

**结论（先给判定）**：`docs/algorithm-gap-analysis.md` 的 B3 行此前只结清了静态 Young–Laplace 压差，**振荡液滴这一半始终是 open**。本轮补上它：以面积守恒椭圆释放 2-D 液滴，用二阶矩测模态 2 振幅，与 Rayleigh–Lamb 二维频率 $\omega_2=\sqrt{6\sigma/(\rho R^3)}=73.48$ rad/s 比对。

全部 1 个算例的最大频率偏差为 **43.5%**，平台 15% 门限判定：**不通过**；3% 严格判定：**未通过**。

## ① 案例介绍

静止 2-D 液滴（无限长柱截面）受表面张力驱动做自由振荡。对表面做 $r(\theta)=R(1+\varepsilon\cos m\theta)$ 的扰动，界面曲率的一阶展开为

$$\kappa(\varepsilon)=\frac{1}{R}\left[1+(m^2-1)\varepsilon\right],\qquad p'=\sigma\kappa'\varepsilon=\frac{\sigma(m^2-1)}{R}\varepsilon$$

与内部无旋场 $\phi=A r^m\cos m\theta$ 的 Bernoulli 平衡、自由面运动学条件联立，得 Rayleigh–Lamb 二维频率

$$\omega_m^2=\frac{\sigma\,m(m^2-1)}{\rho R^3}\qquad\Longrightarrow\qquad m=2:\;\omega_2=\sqrt{\frac{6\sigma}{\rho R^3}}$$

参考文献：Rayleigh (1879) Proc. London Math. Soc. 10:4；Lamb (1932) *Hydrodynamics* 6th ed. art. 267；Chandrasekhar (1961) *Hydrodynamic and Hydromagnetic Stability* ch. XII。曲率展开式 (1) 本身不用手抄：`test_laplace_curvature_expansion_gives_the_rayleigh_lamb_coefficient` 用极坐标曲率公式对 $m=2,3,4$ 数值求导复核。

| 参数 | 值 | 说明 |
|---|---|---|
| 液滴半径 R | 0.020 m | 面积等效半径 |
| 粒子间距 dx | 2.00 mm | R/dx = 10，h = 2.60 mm |
| 表面张力系数 σ | 7.2 N/m | 10× 水，压缩毛细时间 |
| 参考密度 ρ0 | 1000 kg/m³ | Tait γ = 7 |
| 声速 c0 | 20 m/s | Mach ≪ 0.1 |
| 运动粘度 ν | 5.0e-05 m²/s | 基准算例；粘度序列另见 ③ |
| 初始扰动 ε | 0.10 | 椭圆 a = R(1+ε)，b = R/(1+ε)（面积守恒）|
| 粒子数 N | 357 | 六角密排，被椭圆裁剪 |
| 重力 | 0 | 纯毛细振荡 |
| 解析角频率 ω₂ | **73.485 rad/s** | 待比对基准量 |
| 解析周期 T | **0.08550 s** | = 2.565 t_σ |
| 毛细时间 t_σ | 0.03333 s | √(ρR³/σ) |
| 时间步 Δt | 3.900e-05 s | 声学 Courant 限 |
| 模拟时长 | 0.1539 s = 1.80 T | 3947 步 |

## ② 怎么做的

新增 `benchmarks/oscillating_droplet/run.py`，复用生产代码路径 `tensorsph.droplet_impact.rhs`（含 `csf_scheme="consistent"` 的 CSF 力、`csf_curvature="fit_quadratic"` 的曲率算子），**未改任何 `src/` 文件**（本单是验证单）。

1. **初始条件**：六角密排格点被椭圆 $a=R(1+\varepsilon)$, $b=R/(1+\varepsilon)$ 裁剪，故 $ab=R^2$ 面积零偏差（用 $b=R(1-\varepsilon)$ 会引入 $O(\varepsilon^2)$ 平均半径漂移）。速度场为零，密度由求和给出。
2. **观测量**：质量加权二阶矩张量的特征值 $\lambda_{1,2}$ 给出 $A=2\sqrt{\lambda_1}$, $B=2\sqrt{\lambda_2}$，$s(t)=(A-B)/(A+B)$。均匀椭圆 $\langle x^2\rangle=a^2/4$，故 $s=\varepsilon$（一阶）。
3. **频率估计**：对 $s(t)$ 做阻尼余弦最小二乘 $s=c+e^{-\lambda t}(P\cos\omega t+Q\sin\omega t)$，$(\omega,\lambda)$ 网格搜索 + 全迹精修，$(c,P,Q)$ 闭式解；另给零交叉法（逐半周期）作独立估计。
4. **独立交叉检查**：总动能也是纯运动学量，$KE\propto s^2$ 故以 $2\omega$ 振荡，拟合其频率得 $\omega_{KE}/2$，与几何估计对照。
5. **变参**：分辨率阶梯（R/dx 固定、dx 变化）、粘度序列（检验 $\lambda\propto\nu$）、零 g 静滴 + 扩散密度项开/关（gap 分析 P1 行点名要的「quiet free-surface case」）。
6. **工具链**：PyTorch 2-D 弱可压缩 SPH（CPU）、pytest、matplotlib。本文所有数字由 `make_doc.py` 从 `validation_report.json` 与 `convergence/*.json` 注入。

## ③ 计算结果

### 3.1 频率 vs 解析解（逐算例）

| 算例 | ω 实测 [rad/s] | 误差 | < 3%？ | < 15%？ | 零交叉法误差 | 动能法误差 |
|---|---|---|---|---|---|---|
| 基准算例 dx = 2.0 mm | 41.50 | -43.5% | 否 | 否 | +391.9% | -20.6% |
| 解析值 | 73.48 | — | — | — | — | — |

零交叉法误差按 $\omega$ 本身计；动能法误差按 $2\omega$ 基准计（`fit_kinetic_energy_rel_error_pct` 即相对 $2\omega_2$）。

### 3.2 阻尼（$\lambda$ 与粘度的关系）

| 算例 | ν [m²/s] | λ 实测 [1/s] | λ 峰拟合 [1/s] | 量级估计 2νm²/R² [1/s] | 比值 |
|---|---|---|---|---|---|
| 基准算例 dx = 2.0 mm | 5.0e-05 | 0.000 | -6.308 | 1.000 | 0.00 |

量级估计 $\lambda=2\nu k^2$（$k=m/R$）来自无旋平面毛细波的耗散近似，**不是**二维粘性液滴的闭式解：二维粘性柱的色散关系只有 Bessel 行列式形式（Chandrasekhar 1961 ch. XII），没有可逐项比对的标量公式。因此阻尼一项的证据等级为 **WITHHELD**（有量级对照，无可验收基准）；可验收的只有「$\lambda$ 随 ν 单调、近似线性」这一结构性判据。

### 3.3 零 g 静滴：噪声地板与扩散密度项（gap 分析 P1 行）

本轮未跑该分支。

### 3.4 图

![oscillating droplet](oscillating_droplet.png)

左：各算例的模态 2 振幅 $s(t)$（可见衰减振荡）；右：实测 $\omega$ 与 $\sqrt{\sigma m(m^2-1)/\rho R^3}$ 的比值，绿带为 ±15% 平台门限。

## ④ 注意点（坑与适用范围）

1. **二维/三维公式不能混用**：三维球谐公式 $\omega_m^2=\sigma m(m-1)(m+2)/(\rho R^3)$ 在 $m=2$ 给出 84.85 rad/s，比二维值高 **15.5%**（正好等于平台门限）。用在 2-D 液滴上等于把判据变成抛硬币；`test_two_dimensional_and_three_dimensional_laws_differ_by_the_gate` 钉住这一点。
2. **界面半径的 Fourier 分解在 R/dx = 10 时不可用**：把 $r-\langle r\rangle$ 直接投影到 $\cos2\theta$ 会因椭圆面积加权而在**一阶抵消**（$\langle\cos2\theta\rangle=\varepsilon$ 恰好抵消 $\langle r\cos2\theta\rangle=\varepsilon R$）；改用外圈半径凸包后，格点锯齿 $O(dx)$ 与几何信号 $2\varepsilon R$ 同量级，同一条 $\varepsilon=0.10$ 的初始椭圆上测得 0.055–0.13（差 45%–30%）。故本基准的主估计量取二阶矩（整体积分量），界面几何只作定性参考。
3. **动能法只测到 2ω**：$KE\propto s^2$ 是平方量，能给出独立频率但分辨率低，用作交叉检查而非独立验收。
4. **代价 ∝ c0√(ρR/σ)**：一个振荡周期所需步数 $T/\Delta t$ 随 R 增大而增大，故 `tests/` 里的 slow 用例取 R = 1 cm 的同 R/dx 算例，判据不变。
5. **适用范围**：单相弱可压 SPH + CSF 表面张力 + 零重力；不含壁面、接触角、多相。振荡振幅 $\varepsilon=0.10$ 已接近线性和二阶频率漂移区间，更大振幅需要二阶理论修正后再比对。

## ⑤ 误差判定（逐项，显式回答「是否 < 3%」）

| 关键量 | 基准 | 实测（基准算例） | 误差 | < 3%？ | < 15%？ |
|---|---|---|---|---|---|
| 模态 2 频率 ω | 73.48 rad/s | 41.50 rad/s | -43.5% | 否 | 否 |
| 周期 T（零交叉） | 0.08550 s | 0.01738 s | +391.9% | 否 | 否 |
| 初始模态振幅 ε | 0.10 | 0.0970 | -3.0% | 否 | 是 |
| 阻尼 λ | 无闭式解（量级 2νm²/R²） | 0.000 1/s | 不可判定 | WITHHELD | WITHHELD |
| 质量守恒 | 0 | 0.00e+00 | — | 是 | 是 |

**逐项回答**：频率一项**未达 3%**，全算例最大绝对偏差 43.5%；阻尼一项**无二维闭式基准**，按 honest 口径记 WITHHELD，不给假达标。

## ⑥ 复现

```bash
PYTHONPATH=src python benchmarks/oscillating_droplet/run.py --mode all --plot
PYTHONPATH=src python benchmarks/oscillating_droplet/make_doc.py
PYTHONPATH=src python -m pytest tests/test_oscillating_droplet.py -q
```


## ⑦ 诚实记录：本轮运行的数值状态（honest FAIL，不粉饰）

上表数字来自一次**数值已经发散**的运行，必须把证据摆出来，否则「−43.5% 频率偏差」会被误读成
表面张力动力学结论：

| 诊断量 | 实测 | 判据 | 判定 |
|---|---|---|---|
| 最大速度 max\|v\| | 1.82e5 m/s（= 3.0e5 × 毛细速度） | ≪ √(σ/ρR) = 0.600 m/s | FAIL（发散） |
| 终态模态振幅 s | 0.936（初值 0.097） | ≤ ε = 0.10（守恒量） | FAIL（能量注入） |
| 阻尼拟合 λ | 0.0（峰拟合 −6.3，r² = 0.07） | λ ≥ 0 | FAIL（拟合无意义） |
| 质量守恒 | 0（构造上严格） | 0 | 通过（但质量守恒不能救发散） |

**结论**：频率读数 ω = 41.5 rad/s 是发散轨迹上的伪拟合，**不构成对 Rayleigh–Lamb 律的物理判定**，
本单的证据等级只能记 **WITHHELD / honest-FAIL（运行无效）**，不能记「表面张力频率已验证」也不能记
「已证伪」。已定位的可疑因素（供下一单）：

1. 粘度取 ν = 5e-5 m²/s（比 m2174/m2588 的表面张力算例 ν = 0.01 低 200 倍），低粘度下自由面
   配对/张拉不稳定没有任何耗散压制；先前所有通过的 CSF 算例都用 ν = 0.01。
2. 初始时刻 p ≡ 0 而 CSF 力立刻非零，密度为求和值（自由面亏缺），声学瞬态可能直接触发爆点；
   缺一个「零速 + 压强偏置」的自洽静平衡初始化（正是 m2174 backlog P0 第 1 条未做）。
3. 未见 dt 的毛细（表面张力）稳定性判据参与 `stable_dt`，只有声学/粘性/Courant 三项。

**未完成的部分（如实列出）**：分辨率阶梯（R/dx 变化）、粘度序列（λ ∝ ν）、零 g 静滴 + 扩散密度项
开/关对比（gap P1 行点名）本轮**未跑**——预算与机时限制下只完成了基准算例，且该算例发散。
`make_doc.py` 对缺失分支会写「本轮未跑该分支」，不会伪造数字。

## 附录 A：合并保留的 m2919 独立谱系（PR #87）

`agent/m2919` 分支送来的是同一物理量的**独立实现**（`--case` CLI、任意方位模态几何 `lobed_droplet`、网格搜索阻尼余弦拟合器），与 main 线**互斥地占据同一路径**，因此三方合并不能把两份代码塞进一个文件。处置：main 线保留为规范实现（功能超集：4 个 study + 图 + 本文档的注入器），m2919 线的**测量、文档、通用模态几何与负控测试**逐项并入，未丢一个数字：

| m2919 物证 | 合并后位置 | 处置 |
|---|---|---|
| 逐字测量报告 | `benchmarks/oscillating_droplet/validation_report_m2919.json` | 原样保留 |
| 逐字验证文档 | `docs/verification_oscillating_droplet_m2919_lineage.md` | 原样保留 |
| `run.py` 实现（被 main 线超集取代） | `artifacts/merge_pr87/m2919/` | 原样保留 |
| `tests/test_oscillating_droplet.py` | 本仓测试第 5 节 + 上述 `artifacts/` 存档 | 性质并入 |
| `lobed_droplet`（任意模态几何） | `run.py::lobed_droplet` | 并入主线 |
| n=1 无回复力守卫、σ=0 负控 | `run.py::analytic_omega` / `run_case(steps=)` | 并入主线 |

m2919 线自己的测量（与其文档一致，同为 honest FAIL）：

| 算例 | n | dx [mm] | ω 解析 [rad/s] | ω 实测 [rad/s] | 误差 | < 15%？ | 稳定性 |
|---|---|---|---|---|---|---|---|
| mode2 | 2 | 1.5 | 73.48 | 28.15 | -61.7% | 否 | particle left the box at step 778 (0.44 T) |
| mode3 | 3 | 1.5 | 146.97 | 56.31 | -61.7% | 否 | particle left the box at step 1280 (1.46 T) |
| coarse | 2 | 2.0 | 73.48 | 62.11 | -15.5% | 否 | particle left the box at step 1200 (0.91 T) |
| no_sigma | 2 | 1.5 | 0.00 | 38.31 | — | — | stable |

**两线对照（同一物理构型：R = 20 mm、dx = 2 mm、ε = 0.10）**：main 线基准算例 -43.5%、m2919 线 `coarse` 算例 -15.5%。两条独立实现都判 FAIL，但量值相差 2.8 倍——这不是「互相印证」，而是提示该量在当前格式下**不可复现**：两线各自的运行都在中途发散（main 线见本文「诚实记录」一节，m2919 线见上表的 `stability` 列），发散时刻不同就给出不同的伪频率。因此合并后的口径是**不把任何一方的频率当作结论**，只把「偏差随分辨率/发散程度剧烈漂移」这一事实连同两线原始数据一起锁进仓库。漂移由 `test_oscillating_droplet.py` 的数值钉子守在 CI 里（m4786 更正：该钉子曾写成 strict-xfail，语义自相矛盾——xfail 要求断言**失败**，于是复现记录值的运行变成 XPASS 把 main 判红；现改为普通断言 + loong64 架构限定，复现记录值即通过，漂移才失败）。

`no_sigma` 行是 σ = 0 负控（合并后由 `test_without_surface_tension_the_mode_does_not_oscillate` 守住）：无表面张力时二阶矩仍会因压力噪声给出一个**伪频率**，所以负控的判据只能是「离 ω₂ 足够远」，不能写「拟合失败」。


## 附录 B：m4786 —— 数值钉子的语义更正与架构界定

main 在 `799310b0` 变红，红的两条是 `test_recorded_mode2_frequency_deficiency_is_reproduced`
与 `test_oscillation_frequency_gate_is_not_met`，**失败形态是 XPASS(strict) 而不是断言失败**：

| 项 | 事实 |
|---|---|
| 钉子记录的 CI 算例误差 | −31.78 %（R = 1 cm、dx = 1 mm、c0 = 20 m/s、ν = 5e-5、ε = 0.10） |
| m4786 复测（loong64、单线程、seed 0） | −31.78 %，落在 `abs=2.0` pp 容差内 → 断言**通过** |
| `xfail(strict)` 的语义 | 断言**必须失败**才算 XFAIL；断言通过即 XPASS，strict 下 XPASS = CI 红 |
| 结论 | 钉子本身是「记录值复现器」，写 strict-xfail 等于要求它**永远不成立**，语义自相矛盾 |

处置：去掉 strict-xfail（改为普通断言），并把钉子**架构限定**在记录它的 loong64
（`skipif(platform.machine() not in {"loongarch64","loong64"})`）。理由是同代码在 x86_64 上量到
+8.64 %、在 loong64 上量到 −31.78 %（m3312 记录）：本算例的液滴是极欠阻尼振荡器（ν 低、
自由面张拉不稳定无耗散），离开平衡态的时刻由机器的浮点累加顺序决定，因此拟合出的频率
**只在单一架构内可复现**。把钉子限定在记录它的架构上，既保住了漂移检测，也不宣称该数字可移植。

**这一节不能读作「物理已被接受」**：平台判据仍是 FAIL（|−31.78 %| ≥ 15 %），
`test_oscillation_frequency_gate_is_not_met` 用**反向断言**把 FAIL 钉住——CI 绿只表示
「已记录的缺陷被复现」，一旦哪天真修好了、误差进了 15 %，该测试会先红，提醒重渲染本文档。
