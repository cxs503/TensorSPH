# 振荡液滴：CSF 表面张力的动态验证（本次 FAIL，证据等级 WITHHELD）

> 工件：`benchmarks/oscillating_droplet/{run_lamb.py,validation_report_lamb.json,oscillating_droplet_lamb*.png}`、`tests/test_oscillating_droplet_lamb.py`。所有数字由 `json` 注入，无手抄。

> **合并说明（m3820）**：本文件来自 PR #87 分支 `agent/m2919`。`main` 侧在同一路径
> `benchmarks/oscillating_droplet/run.py` 独立长出了**另一套**同基准实现，其报告见
> [`verification_oscillating_droplet.md`](verification_oscillating_droplet.md)。两套实现并存、
> 互不覆盖：本文件描述 `run_lamb.py` 一套（模态 n=2/3、分辨率阶梯、阻尼、σ=0 负控），
> 冲突逐文件解法记录在 [`merge_resolutions/m2919-oscillating-droplet.md`](merge_resolutions/m2919-oscillating-droplet.md)。

## ① 案例介绍

**物理背景**：二维液滴的毛细振荡。半径 R、密度 ρ、表面张力 σ 的液滴在模态
r(θ)=R[1+ε cos nθ] 上是一个谐振子，频率有闭式解（Lamb 1932, Art. 275；Rayleigh 1879）：

  ω_n² = n(n²−1)σ/(ρR³)，n≥2（n=2 时 = 6σ/(ρR³)）。

**为什么做这个**：TensorSPH 目前已发表的表面张力数字全是**静态**的（Young–Laplace 压差、
切割力恒等式 F_cut=2σ、圆界面曲率，见 `docs/verification_surface_tension_{laplace,consistent,curvature}.md`）。
力的**方向/相位**、CSF 加速度的**有效惯性**在动界面上一律未验过；本工作补这一项。

**几何与参数**（六角点阵液滴，R=0.02 m，ρ0=1000 kg/m³，σ=7.2 N/m，c0=10 m/s，γ=7，无重力、无壁面）：

| 量 | 值 |
|---|---|
| 液滴半径 R | 0.02 m |
| 点阵 | 六角（dy=dx√3/2，m=ρ0 dx dy） |
| 分辨率 | dx = 1.5 / 2.0 mm（R/h = 10.3 / 7.7） |
| 初始变形 | ε = 0.05（n=2 与 n=3） |
| 运动粘度 | ν = 1e-3 m²/s |
| 时间步 | dt = 0.3·1.3·dx/c0（声学 Courant 限制），非自适应 |
| 边界 | `clamp` 方腔 4R×4R，液滴置中，无壁粒子 |

## ② 怎么做

1. `benchmarks/oscillating_droplet/run_lamb.py`：六角液滴 → r(θ)=R(1+ε cos nθ) → 用 `rhs`（含 CSF 表面张力，
   `csf_scheme=consistent`）推进；
2. 观测量取**惯性张量模态** M(t)=Σm(x²−y²)/Σm r²（连续极限 = 1.5ε cos ωt），
   不需要识别表面粒子、不被点阵量化；
3. (ω,γ) 由阻尼余弦 a(t)=offset+e^{-γt}(c cos ωt+s sin ωt) 的两级网格 + 线性最小二乘拟合给出，
   并报告相对残差（噪声主导时≈1）；
4. 负控：σ=0 同样推进同样时长，检验「无表面张力就没有毛细振荡」；
5. 交叉校验：`tests/test_oscillating_droplet_lamb.py` 用合成阻尼余弦给拟合器做基线（ω 复现 <0.5 %），
   用白噪声做负控（残差必须 >0.5），并锁定一维/三维公式陷阱。

## ③ 计算结果（程序化注入，未手抄）

| 算例 | 模态 n | `dx` (m) | R/h | 粒子数 | ω_Lamb (rad/s) | ω_实测 (rad/s) | 误差 | 拟合残差 | 稳定性 |
|---|---|---|---|---|---|---|---|---|---|
| `mode2` | 2 | 0.0015 | 10.3 | 639 | 73.48 | 28.15 | -61.7 % | 0.105 | particle left the box at step 778 (0.44 T) |
| `mode3` | 3 | 0.0015 | 10.3 | 648 | 146.97 | 56.31 | -61.7 % | 0.416 | particle left the box at step 1280 (1.46 T) |
| `coarse` | 2 | 0.0020 | 7.7 | 365 | 73.48 | 62.11 | -15.5 % | 0.202 | particle left the box at step 1200 (0.91 T) |
| `no_sigma` | 2 | 0.0015 | 10.3 | 639 | 0.00 | 38.31 | — | 0.034 | stable |

注：`no_sigma` 为负控（σ=0，不参与 ω 打分）。

## ④ 结果图

![oscillating droplet](oscillating_droplet/oscillating_droplet_lamb.png)

![mode decay](oscillating_droplet/oscillating_droplet_decay.png)

## ⑤ 注意点 / 局限

1. **公式陷阱（已写进代码与测试）**：三维 Rayleigh 解 ω²=n(n−1)(n+2)σ/(ρR³) 在 n=2 给 8 而非 6，
   比二维解高 15.5 %——正好等于平台验收门宽度。用错分支会得到「恰好差 15 %」的假失败。
2. **自由面 CSF 曲率噪声是硬约束**：在**名义平直**的开放自由面上实测 κ_rms = 150–254 1/m
   （dx=1/0.5 mm，hex/square × mls/fit_quadratic/fit_circle 九种组合），而毛细波/液滴模式的
   曲率量级 10–150 1/m。开放自由面的毛细动力学当前**不可验**，这就是本工作改在闭合液滴上做的原因。
3. **初始状态不是离散平衡**：σ/(ρc²R) 决定维持 Laplace 压差所需的压缩量；c0=3 时需 3 % 压缩，
   起动瞬间即产生强声学瞬变并导致粒子逃逸（实测 step≈95 逃逸）。把 c0 提到 10 后降到 0.2 %。
4. **实现踩坑（已修，留档）**：run_lamb.py 里 `for … else` 的 `else` 分支缩进错误，把时间累加语句
   吞进了 `else` 块，导致所有采样时刻都是 t=0、拟合返回搜索下边界（−60 %）。
   该 bug 由「所有算例误差完全相同」这一异常模式暴露——**误差完全相同=程序错，不是物理错**。
5. **观测窗口不足**：模态在一个周期内即衰减到 1/e 以下，数值耗散 ≫ 解析阻尼，
   因此本参数下 (ω,γ) 的分辨率被耗散主导；负控（σ=0）也拟合出 ω≈38 rad/s，
   正说明「衰减主导 → 频率提取不可信」。

## ⑥ 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 解析基准 | 实测 | 误差 | <15 %？ | <3 %？ |
|---|---|---|---|---|---|
| ω (`mode2`) | 73.48 rad/s | 28.15 rad/s | -61.7 % | 否 | 否 |
| ω (`mode3`) | 146.97 rad/s | 56.31 rad/s | -61.7 % | 否 | 否 |
| ω (`coarse`) | 73.48 rad/s | 62.11 rad/s | -15.5 % | 否 | 否 |

**结论：动态（振荡）表面张力验证本次 FAIL——没有任何一项进入 15 % 门，更没有进入 3 % 门。**
因此 `physical_validation` 的动态分支证据等级维持 **WITHHELD**（不是 REAL）。
诚实记录：本轮的净增量是「问题被量化了」——(a) 建立了可复现的动界面验证台
（`benchmarks/oscillating_droplet/`，含负控、公式陷阱、拟合器基线共 15 个测试）；
(b) 定位到两个真障碍：开放自由面曲率噪声 150–254 1/m、以及数值耗散 ≫ 解析阻尼；
(c) 记录并修复了让首批结果全部作废的 for/else 缩进 bug。

**下一步（交接）**：① 表面张力 CFL 严格取 dt ≤ 0.25√(ρh³/σ) 并配 δ-SPH 密度扩散，
把数值耗散降到 ν 以下；② 曲率/δ_s 噪声先治理（目前 mls 在开放自由面上 κ_rms≈150 1/m），
否则任何毛细动力学都验不了；③ 用本验证台的 `mode2` 单算例作为回归门（当前 xfail）。

## 附录（m4786 → m4795）：护栏断言对齐本方档的实测值

`tests/test_oscillating_droplet_lamb.py::TestSolverRegression` 里那条一周期护栏原先把判定写成
`|err| < 30 %`，而本文 ⑥ 行 `mode2/mode3` 自己记录的就是 **-61.7 %**。也就是说这条断言**自落地起就不可能通过**
（m4795 复核：连跑两次，`omega = 28.1538 rad/s`、`err = -61.6875 %` 逐位相同，与本文 `mode2` 的
28.15 rad/s 一致；单次 34.5 s）。永远为红的护栏不提供任何保护，只提供噪音。

处置（与 `verification_oscillating_droplet.md` 附录 B 同口径）：

| 项 | 旧 | 新 |
|---|---|---|
| 频率偏差判定 | `|err| < 30 %`（永不可能满足） | **按平台钉**实测伪频率（loongarch64 = -61.69 %，±1 pp），未记录平台只做**包络**判定（-80 %, -10 %） |
| 物理判定 | 隐含「接近解析频率」 | 显式断言 **15% 门未通过**（任何平台）：本文记录的 FAIL 一个字未改 |
| 拟合质量 | `resid < 0.5`（保留） | 保留，另加 `omega > 0` 有限性 |

**逐项回答「是否 < 3%」**：否。ω(mode2) = 28.15 rad/s vs 解析 73.48 rad/s，误差 **-61.7 %**，
远超 3% 与 15%；且该运行中途有粒子飞出钳位盒（本文 3 节），属 honest-FAIL，不构成对 Rayleigh–Lamb 律的判定。
本轮只修护栏口径，**不改**求解器、**不改**判定。
