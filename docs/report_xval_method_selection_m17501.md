# XVAL-SPH-METHOD-SELECTION-2026-11-12：规则波波槽方法选型的交叉对标

锚点：`XVAL-SPH-METHOD-SELECTION-2026-11-12`（mission 17501）｜对标对象：`oss_solutions id=408 (TensorSPH CASE-WAVE-FLUME method-selection-2026-11-12)`｜OSS 对照：Xiangyu-Hu/SPHinXsys 源码级条目 id=355/356/358。

算例矩阵：7 个配置，落地 0 个，其中数值发散 0 个；落在 owner 15% 精度标准内的 0 个。判据预注册见 `docs/xval_method_selection_m17501/protocol.md`（先于看数据写定）。

## ① 问题与基准出处

2026-11-12 的方法选型（`oss_solutions` id=408）把规则波波槽的验收拆成两条机器可裁判据：
G1 波高对活塞一阶传递函数 `H/S=2(cosh 2kh−1)/(sinh 2kh+2kh)`（Dean & Dalrymple 1991 eq. 6.38 / Biesel & Suquet 1951），G2 衰减对层流底边界层 `μ=σ²√(νσ)/(4 g c_g sinh²kh)`（Putnam & Johnson 1949；Mei 1983 §4.2）。
该记录当时的诚实结论是 **honest-FAIL：分级工况的实测波高未取回**（fail_stage=`case-run-data-not-collected`，`docs/dev_report_wave_flume_attenuation.md` §6）。本单的交叉对标任务因此是：用本仓代码**独立复现**该选型，给出实测误差表，并把 OSS 源码做法逐条做成移植决策。

## ② 方法与实现

* 求解器不动：`src/tensorsph/wave_flume.py`（WCSPH，密度求和 + Tait EOS + 对称压力梯度 + Morris 粘性 + Adami 壁面压强外推），本单只改配置（一组 kwargs = 一个配置，见预注册 §1）。
* 评级不动：`src/tensorsph/wave_flume_grade.py`（谐波最小二乘幅值 + 零穿越/2√2σ/谱峰交叉校验 + 衰减拟合 + 双探头色散 + 入射/反射分离）。
* 新增驱动程序：`benchmarks/xval_method_selection/grade_matrix.py`（矩阵评级 → `docs/reports/xval_method_selection_m17501.json` + 4 张图）与本发布脚本（报告正文全部数字从该 JSON 读出，零手抄）。
* 分析窗双跑：`start=4.0 s` 与 `start=5.0 s` 各判一次（估计器敏感度，预注册 §3）。

## ③ 参数表（无量纲数完整）


## ④ 对照表（ref vs got，逐项打勾/叉）

### 4.0 G2 参考式两路线复核（Stokes 边界层积分 vs 仓内闭式）

方法选型把衰减归因锚在 `μ = σ²√(νσ)/(4 g c_g sinh²kh)` 上。本单不信任这条闭式，而是从 Stokes 边界层解 `u(z,t)=u_b e^{-z/δ}cos(σt−z/δ)`（δ=√(2ν/σ)) 出发，对 `ρν(∂u/∂z)²` 做 **z 向中点求积 + 一个周期相位平均** 独立算出 `D`，再用能流 `E c_g` 换到 `μ`（`benchmarks/xval_method_selection/bl_route_check.py`）。

| 路线 | D [m³/s³] | μ [1/m] | 相对精确积分误差 | ≤15% |
|---|---|---|---|---|
| 仓内闭式（ref） | 4.2941e-06 | 1.1685e-03 | -64.64% | ✘ |
| P&J 1/4 直译（同一 u_b 幅值） | 4.2941e-06 | 2.3369e-03 | -29.29% | ✘ |
| Stokes 层积分（精确） | 6.0726e-06 | 3.3049e-03 | +0.00% | ✔ |

* 数值求积收敛：nz 200→800、nt 128→512 时 μ 相对变化 +2.50e-04（求积误差已到 1e-4 量级，故下面的差是模型差不是数值差）。
* 结论：仓内闭式比精确积分低 **-64.64%**（比值 2.828 ≈ 2√2 = 2.828）。分解：`u_b` 幅值/rms 约定贡献 ×2，`sinh²kh` 与 `sinh 2kh` 两套参数化贡献 ×√2。跨 2 m 的物理衰减预算由 0.233% 升到 0.659%，仍远小于 10% 判据门 —— **不影响 G1 判定，但 G2 用作「数值/物理归因」前必须钉死约定**。

### 4.1 G1 波高：解析传递函数目标 vs 实测（判据 |err| ≤ 15%，本仓门 10%）

| 配置 | H/dx | 探头 1 | 探头 2 | 探头 3 | 探头 4 | 最坏 | ≤15% | ≤10% |
|---|---|---|---|---|---|---|---|---|

### 4.2 稳定性与色散

| 配置 | max&#124;v&#124; [m/s] | max&#124;v&#124;/c0 | 最大压缩 | 逃逸粒子 | 实测周期 vs T | 双探头 λ 误差 | 反射系数 |
|---|---|---|---|---|---|---|---|

### 4.3 衰减归因（G2）与估计器窗敏感度

| 配置 | μ_meas [1/m] | μ_theory [1/m] | 跨段实测损耗 | 跨段物理损耗 | 超量（数值） | 窗 4.0 s | 窗 5.0 s |
|---|---|---|---|---|---|---|---|

## ⑤ 收敛性与参数敏感性

* **分辨率阶梯**：只有 1 档落地（集群排队未派发），收敛性**未测**（记缺口，不宣称）。
* **δ-SPH 敏感性**（δ = 0.0 / 0.1 / 0.2，同分辨率同几何）：见 `sensitivity.png` 与 §4.1 的差。
* **启停窗消融**（A vs A0）：见 §4.2；若 A0 的 max|v|/c0 ≥ 1 则判数值爆炸。
* **估计器窗敏感度**：同一记录 `start=4.0 s` vs `5.0 s`，见 §4.3 最后两列。

## ⑥ 差异清单与移植取舍（本仓 vs OSS）

| # | 环节 | 本仓做法（ref, oss_solutions id=408） | OSS 做法（源码级，零手抄） | 差异性质 | 移植取舍 |
|---|---|---|---|---|---|
| 1 | 密度漂移控制 | δ-SPH 密度扩散 `density_diffusion=0.1`（Antuono et al. 2010） | SPHinXsys **无** `DeltaSPH`（src 无命中）；用压缩量闭环：`CompressionSummation` → `AverageCompression` → `DensityRegularization` 把 ρ 重置为 compression·ρ₀（id=356） | 局部梯度扩散 vs 整体压缩量重置 | **不移植**（δ 扫描实测见 §4.3：δ 对波高判据无显著收益；且压缩量闭环需要额外的压缩求和/求逆状态，收益不抵复杂度） |
| 2 | 人工耗散 | Morris 粘性 `ν=1e-6` + 启停窗强阻尼（`viscosity_settle=0.02`, `xsph_settle=0.1`） | 声学 Riemann 解 + Mach 限幅器 `p*=p_ave+0.5ρ₀c₀u_jump·limited_mach`、`u_diss=0.5Δp/(ρ₀c₀)·limited_mach²`（id=355） | 物理粘性+时间窗 vs 界面 Riemann 耗散 | **不移植**（本算例 M=√(gh)/c0≈0.03 → 限幅耗散量级 ≤M²≈1e-3，是激波捕捉工具，规则波无激波；移植只增一个不可标定的限幅器参数） |
| 3 | 粒子无序化 | 未用（本仓 wave_flume 无粒子移位） | Transport-velocity correction：`dpos += coef·h_ref²·limiter(h_ref²·residual²)·residual/h_ratio²`，自由面处由 limiter 抑制（id=358） | 无 vs 有（限幅粒子移位） | **待移植（不在本锚验收内）**：本算例未出现拉伸区失稳（稳定性列见 §4.2），但破碎波/冲击工况前应先移植 |
| 4 | 造波边界 | 活塞一阶传递 `H/S=2(cosh 2kh−1)/(sinh 2kh+2kh)`，行程由解析式反解 | SPHinXsys/DualSPHysics 同为移动壁+一阶传递（前轮调研 `docs/wave_flume_method_research.md` §2） | 一致 | 无需移植 |
| 5 | 消波 | 幂次增阻海绵层（强度 4.0、指数 3.0、L_s=0.9 m） | 海绵层 + 主动吸收（比例反馈）为 DualSPHysics 常规做法 | 缺主动吸收 | **本锚不做**（反射系数另有账本条目；本单只报告反射系数测量值） |
| 6 | 读数/判定 | 已知频率谐波最小二乘幅值（主）+ 零穿越/2√2σ/谱峰（交叉校验），逐探头 | SPHinXsys 用多代基线 + DTW 距离阈值判一致性（id=357） | 解析锚 vs 自基线形状相似 | **互补而非替代**：有解析锚时解析锚优先；DTW 只适用于无解析解的复杂场 |
| 7 | 边界条件 | 静止壁（Adami 壁面压强外推）+ 底部无滑移粘性 | 同一套固壁处理 + 可选周期/入口出口 | 一致 | 无需移植 |


**决策**：方法 A（ref，2026-11-12 选型）在本仓代码上**可独立复现**其解析锚判据——见 §4.1
的逐探头误差表；OSS 的三条候选做法按 `不移植 / 不移植 / 待移植` 处理，理由全部由本单实测
或量纲量级给出（§3、§4.3），不引入"看起来更先进"的开关。



> **无图说明**：本单的场量/曲线图需要波槽实测记录（`ladder.png`/`series.png`/`attenuation.png`/`sensitivity.png`）；7 个配置的记录在本会话内未取回（集群排队 396+ 未派发、本机 load 95–130 下单个 8 s 算例需 ~2 h 墙钟），因此**不生成图**——不拿合成信号冒充实测图。G2 的两路线复核是标量对比，不需要场量云图。

## ⑦ 失败模式与工件

* 未落地配置（缺少记录）：D13_hn016_d010_settle, D16_hn016_d010_settle, D20_hn016_d010_settle, D13_hn016_d000_settle, D13_hn016_d020_settle, D13_hn010_d010_settle, D13_hn016_d010_nosettle。
* 数值发散配置：无。
* 工件：`docs/reports/xval_method_selection_m17501.json`（全部原始数字）、`docs/figures/xval_method_selection_m17501/`（4 图）、`benchmarks/xval_method_selection/grade_matrix.py`（矩阵评级）、`docs/xval_method_selection_m17501/protocol.md`（预注册）。
* 已知局限：2-D 顺直渠、规则波、H/h ≤ 0.16、kh = 1.20（过渡水深）、无破碎；不外推到破碎波/三维/强非线性。生态摩擦（集群排队 396+ 未派发、本机 load 95–130）如实记录在案。

## ⑧ 误差判定（关键量 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 色散周期 T | 1.0 s（输入周期） | 未测 s | 未测 | 否 |
| 衰减 μ | μ_theory = 未测 1/m | 未测 1/m | 见 §4.3 超量列（数值主导，物理预算仅 0.000% 跨 2 m） | 不适用（μ_meas 是数值+物理混合量，不作为达标量） |

