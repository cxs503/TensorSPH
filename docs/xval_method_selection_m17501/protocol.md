# XVAL-SPH-METHOD-SELECTION-2026-11-12 · 预注册实验协议（先于看数据）

锚点 `XVAL-SPH-METHOD-SELECTION-2026-11-12`（mission 17501）。
本页在**任何波槽实测数字落地之前**写定判据、指标、变量、重复与停止规则；
结果表在 `docs/reports/xval_method_selection_m17501.json`（程序化生成，零手抄），
正文报告见 `docs/report_xval_method_selection_m17501.md`。

---

## 0. 决策（Decision）

在 TensorSPH 的规则波波槽 `CASE-WAVE-FLUME` 上，决定 **2026-11-12 记录的方法选型
（`oss_solutions` id=408，`repo=TensorSPH / feature_slug=CASE-WAVE-FLUME / tag=method-selection-2026-11-12`）
是否可被本仓代码独立复现，以及是否应把成熟开源实现（SPHinXsys，源码级做法见
`oss_solutions` id=355/356/358）的两条替代做法移植进来**。

## 1. 备选方案（alternatives）

| 代号 | 方案 | 出处 |
|---|---|---|
| **A（ref，被对标项）** | WCSPH + 活塞一阶传递造波 + 幂次海绵 + 启停窗强阻尼（`viscosity_settle=0.02`, `xsph_settle=0.1`）+ 传播期 δ-SPH 密度扩散 `δ=0.1`（Antuono 2010）+ 谐波最小二乘读数 | `oss_solutions` id=408（TensorSPH, 2026-11-12）；`docs/wave_flume_method_research.md` |
| **A0（对照，去掉启停窗）** | 同 A，但 `settle=0`、`viscosity_settle=xsph_settle=0`（=前轮 honest-FAIL 的配置） | `docs/verification_wave_flume.md` §3 |
| **B1（OSS 未移植项①）** | 不做 δ-SPH，用**压缩量闭环密度正则化** ρ ← compression·ρ₀（SPHinXsys DensityRegularization） | `oss_solutions` id=356（源码级：`CompressionSummation`/`AverageCompression`/`DensityRegularization`；src 无 DeltaSPH） |
| **B2（OSS 未移植项②）** | 声学 Riemann 耗散 + Mach 限幅器（u_diss ∝ limited_mach²），压缩→0 时耗散自动消失 | `oss_solutions` id=355（源码级：`p_star = p_ave + 0.5 ρ₀ c₀ u_jump · limited_mach`） |
| **C（参数分支）** | A 的 δ 取值扫描 0.0 / 0.1 / 0.2（B1 的"等价极值"就是 δ→0） | 本单设计 |

本单实际计算 A、A0、δ 扫描（A0/δ=0 是 B1 的可执行代理：B1 与 δ-SPH 的差别在
"整体压缩量" vs "局部密度梯度"，δ=0 给出"无任何密度正则化"的下界）；
B2 的取舍用**量纲/量级**判据定量裁（见 §7），不实现。

## 2. 假设

* H1（可复现性）：A 的实测波高对活塞传递函数目标的误差 **≤15%**（owner 精度标准），
  即 2026-11-12 的方法选型在实测层面成立。
* H2（启停窗必要性）：A 的最坏探头误差显著小于 A0；若 A0 出现 `max|v| > c0` 或压缩 >10%，
  则 A0 判为不稳定（数值爆炸），启停窗被证必要性。
* H3（δ-SPH 必要性）：δ∈{0, 0.1, 0.2} 之间波高误差差异 **<5 个百分点**（即 δ-SPH 对波高
  不敏感）→ 移植 B1（压缩量闭环）对波高判据无收益；若差异 >5 pp，则 δ 是需要选型的真参数。

## 3. 指标（metrics）与判据（predeclared thresholds）

| 指标 | 定义 | 判据 |
|---|---|---|
| G1 波高误差 | `(H_meas − H_target)/H_target`，逐探头，`H_meas`=已知频率谐波最小二乘幅值 | `max_gauge |·| ≤ 15%`（owner 精度标准）；本仓原门 10% 同时报告 |
| H_target | 活塞一阶传递 `H/S = 2(cosh 2kh − 1)/(sinh 2kh + 2kh)`（Dean & Dalrymple 1991 eq. 6.38） | 解析，无自由参数 |
| G2 衰减归因 | `μ_meas`（ln(H/2)~x 最小二乘）vs `μ_theory = σ²√(νσ)/(4 g c_g sinh²kh)` | 报告超量 `μ_meas − μ_theory`，归因数值；不作达标门 |
| 色散 | 双探头波长 vs `ω²=gk tanh kh` | 报告误差，门 10% |
| 稳定性 | `max|v|/c0`、`max_density_compression`、逃逸粒子数 | `max|v| < c0` 且 compression <10% |
| 估计器稳健性 | 分析窗起点 `start=4.0 s` vs `5.0 s`（同记录） | 两窗波高差 <3 pp（否则判"估计器敏感"） |

停止规则：单算例若 `failed=true`（数值发散/几何校验拒收），不重跑调参掩盖，
按 FAIL 记录；阶梯点少一档不阻塞报告，如实标注缺口。

## 4. 变量与混淆（controlled / changed / confounders）

* 控制：几何 h=0.25 m、L=3.2 m、海绵 L_s=0.9 m（强度 4.0、指数 3.0）、T=1.0 s、
  `c0=30 m/s`、`ν=1e-6 m²/s`、`sponge_strength=4.0`、探头位置（几何决定）、
  时间步 = 声学 Courant 步、`dt` 上限 4e-4。
* 变化：分辨率（`depth_cells` 13/16/20 → H/dx = 2.08/2.56/3.2）、
  δ（0/0.1/0.2）、启停窗（开/关）、波陡（H/h = 0.16/0.10）。
* 混淆：①分辨率与步长耦合（dx 变→dt 变→步数变），因此**只比无量纲量**（H/dx、M=√(gh)/c0）；
  ②分析窗含 ramp 后瞬态 → 用两窗交叉检查；③本机 load 高（共享 128 核）→ 只影响墙钟不影响结果。
* 不做的变量：海绵强度/指数（另有账本条目）、主动吸收（不在本锚判据）。

## 5. 重复与样本（repetition rationale）

确定性求解器（无随机种子、无并行归约不确定性），单次运行即一次样本；**重复以"分辨率阶梯 +
参数点"代替随机重复**（3 档分辨率 × {A, A0, δ 扫描, 波陡点} 共 7 个算例）。
估计器不确定度用两个分析窗 + 4 个探头（空间离散度）刻画，不用重复跑堆样本。

## 6. 环境与记录

* 代码：本仓 `TensorSPH`（分支 `agent/m17501`），`src/tensorsph/wave_flume.py`（求解器不改）、
  `src/tensorsph/wave_flume_grade.py`（评级）。
* 命令（repro）：见报告 §2 与 `docs/reports/xval_method_selection_m17501.json` 的 `commands` 字段。
* 平台：本机（128 核，load ≈ 95–130，`--threads 8`）；集群 `run_cluster` 已递交
  cj1555–cj1559（排队 396–400 项未派发，见报告 §6 摩擦）。
* 原始数据：`runs/local/wf_*.json`（每算例一个 JSON，含探头时序、压力/压缩历史、剖面）。

## 7. 预注册的分析与判据（不做结果后调参）

1. 逐算例用 `grade_regular_record(record, start=4.0, tolerance=0.10)` 判级；同一记录再用
   `start=5.0` 复判一遍（窗敏感度）。
2. 误差表 = 逐探头 G1 + 最坏值 + 衰减/色散/反射 + 稳定性；打勾 = `|err| ≤ 15%`。
3. 阶梯图：`max|err|` vs `H/dx`，画 15% 与 10% 两条门线。
4. B2（Mach 限幅 Riemann 耗散）取舍的定量判据：限幅器给出的无量纲耗散 ∝ Mach²，
   本算例 M = √(g h)/c0（波致质点速度/声速）~0.02–0.04 → 耗散 ≤ 1.6e-3，
   而 δ-SPH 项在本算例的量级由 δ=0.1 直接给出；若 B2 的耗散量级 <1% 且无激波/破碎，
   则**不移植**（无收益，且引入一个额外限幅器参数）。
5. 结论只覆盖：顺直渠、规则波、H/h≤0.16、kh=1.20（过渡水深）、2-D、无破碎。
   不外推到破碎波/三维/强非线性。