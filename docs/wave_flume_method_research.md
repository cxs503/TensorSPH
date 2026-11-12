# CASE-WAVE-FLUME 第 2 轮：方法调研与选型（调研 → 选型 → 迁移）

**范围**: TensorSPH ｜ `src/tensorsph/wave_flume.py`（实现） + `src/tensorsph/wave_flume_grade.py`
（评级） + `benchmarks/wave_flume/{run,analyse}.py`（算例与报告）。

**本页目的**: 任务锚点 `CASE-WAVE-FLUME` 前轮（m15465）honest-FAIL 且未留 fail_stage /
failure_reason，本轮按 owner 两轮法，先做①开源横向 ②方法谱系调研，再落到③选型与④结合实现。
本页只写方法与出处；数字与误差判定在
`docs/dev_report_wave_flume_attenuation.md`（由 `benchmarks/wave_flume/analyse.py`
从算例 JSON 程序化生成，零手抄）。

---

## 1. 前轮死法复盘（为什么必须先调研）

仓库内 `docs/verification_wave_flume.md` §3 是前轮的现场记录，问题不是"代码没写"，而是：

| 卡点 | 前轮记录 | 本质 |
|---|---|---|
| 波浪工况从未跑出 | "需要上万粒子 × 数万步，本机 load15 ≈ 120，容量纪律禁止起新的重算力进程" | **没有把重算例送上集群**，把"算不动"当成"不做" 
| 静止水柱底噪与波信号同量级 | 内区 `v_late_rms = 0.0798 m/s` vs `u_orbital = 0.133 m/s`（H=0.03 m） | 数值粘性缺位；启停瞬态污染了整段记录 
| 无验收数字 | 反射/衰减/波高全部"未测" | 判据没有绑定到可执行的评级代码 

结论：本轮的三条硬约束 = ①算例必须上 `cluster_queue`；②必须先处理启停瞬态（否则测的是噪声）；
③评级函数要有合成信号单测（否则测的是估计器）。

## 2. ① 开源横向调研

做法与边界（诚实记录）：

* 本机 `/root/oss-cache` 全量 grep `wavemaker|wave tank|flume|piston`：**命中 0**。缓存里
  只有 `Maarten-vd-Sande/lbm`、`pylbm`（LBM）与 `hschwane/GraSPH`（GPU 位置约束流体，
  PBD/渲染向，无造波边界）。即本机缓存**没有**同类波槽实现可抄。
* 平台 `oss_inventory` / `oss_solutions` 两表本角色 `permission denied`，无 API 端点
  （`openapi.json` 全表检索 `inventory|oss|solutions` 无路由），因此无法完成 SQL 侧检索；
  这一项按"证据不足"记，不臆造条数。
* 退而用公开知识做同类实现的结构对照（每条都给出可核查出处，均属公开文献/开源项目文档）：

| 实现 | 造波 | 消波/吸收 | 读数 | 对本仓的可迁移点 |
|---|---|---|---|---|
| DualSPHysics + `WavePiston`/`WaveFlap` 案例集（Crespo et al. 2015, *Comput. Phys. Commun.* 187:204；Altomare et al. 2017, *Coast. Eng.* 121:1） | 运动边界桨/板，一阶传递函数定行程 | 被动海绵层 + 主动吸收（比例反馈） | 固定 x 波高计取自由面最高粒子 | 与本仓结构**一致**；其分辨率经验 `H/dx ≳ 10` 是波高精度的门槛，本仓需给出自己的分级曲线 |
| SPHinXsys 的 wave-tank 示例（Zhang et al. 2021, *Comput. Methods Appl. Mech. Eng.* 3:100123） | 同上（移动壁） | 海绵层 | 波高计 | 强调 δ-SPH 密度扩散（δ≈0.1）抑制压力噪声而不动波场 |
| REEF3D / OpenFOAM `waves2Foam`（非 SPH，只作方法对照；Jacobsen et al. 2012, *Coast. Eng.* 63:1） | 松弛区造波（relaxation zone） | 同区松弛消波 | 数值波高计 | 消波"阻抗渐变"而不是阶跃，与本仓 `sponge_rate` 的幂次 ramp 同思路 |

横向结论：**本仓的架构（活塞 + 一阶传递 + 幂次海绵 + 波高计）与 DualSPHysics/SPHinXsys
同类做法一致，不需要换架构**；缺的不是"方法"，而是①分辨率经验的自证；②启停瞬态处理；
③与文献基准逐项对照的评级代码。

## 3. ② 方法谱系与标准算例（external_anchor）

| 环节 | 采用方法 | 出处（external_anchor） |
|---|---|---|
| 色散 | `omega^2 = g k tanh(kh)` | Dean & Dalrymple 1991, *Water Wave Mechanics for Engineers and Scientists*, World Scientific, eq. 3.23 |
| 造波 | 活塞一阶传递 `H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)` | 同书 eq. 6.38（原始 Biesel & Suquet 1951） |
| 造波理论实验基准 | 小振幅波实测与理论差 ≲5% | Ursell, Dean & Yu 1960, *J. Fluid Mech.* 7:33–52 |
| 孤波造波 | Goring 条件 `U = c η/(h+η)`，`η = H sech^2(α(x-ct))` | Goring 1979, in *Tsunami Research Symposium* (Bull. R. Soc. NZ 15) |
| 消波层 | 幂次增阻海绵层（避免阻抗阶跃反射） | Israeli & Orszag 1981, *J. Comput. Phys.* 41:115；Wei & Kirby 1995, *J. Waterway Port Coast. Ocean Eng.* 121:251 |
| **波高衰减参考** | 层流底边界层耗散 `D=(1/4)ρ u_b²√(νσ)`，`μ = σ²√(νσ)/(4 g c_g sinh²kh)` | Putnam & Johnson 1949, *Trans. AGU* 30:295；Hunt 1952, *Houille Blanche* 7:838；汇总见 Mei 1983 §4.2 / Dean & Dalrymple 1991 §5.4 |
| 数值噪声控制 | δ-SPH 密度扩散 | Antuono, Colagrossi & Marrone 2010, *Comput. Phys. Commun.* 181:532 |
| 粒子尺度速度噪声 | XSPH | Monaghan 1989, *J. Comput. Phys.* 82:1 |

**标准算例如何取**：验收判据是"波高衰减 vs 文献，容差 <10%"。本单把它拆成两条**可机器裁**
的判据（`tensorsph.wave_flume_grade`）：

1. **G1 造波保真**：每个波高计测得波高 `H_meas` 对活塞传递函数给出的目标波高 `H_target`，
   `|H_meas - H_target|/H_target ≤ 10%`（这是文献"波高"一侧的锚；Ursell 1960 的实验离散度
   约 5%，因此 10% 是"对了"的量级而不是"没测出"的量级）。
2. **G2 衰减归因**：沿波高计阵列拟合 `H = H0 exp(-μ x)` 得 `μ_meas`，与层流底边界层
   `μ_theory` 对照。**物理预算**在 1 s / 0.25 m 工况只有 `μ_theory ≈ 1.2e-3 1/m`（跨 2 m 约
   0.23%），远小于 10% 门；因此 `μ_meas` 明显大于 `μ_theory` 时，衰减是**数值**的，
   报告必须把它归到数值粘性/分辨率，而不是伪称物理。

## 4. ③ 选型与迁移方案（落 `oss_solutions` 的内容）

> DB 写入受限（`oss_solutions` 本角色 `permission denied`，无对应 API），选型正文按可审计
> 路径落在本页；字段映射：`feature_slug='CASE-WAVE-FLUME'`, `repo='TensorSPH'`,
> `tag='wave_height_attenuation'`。

| 字段 | 内容 |
|---|---|
| method | 保留既有 WCSPH 活塞造波 + 幂次海绵 + 波高计架构（与 DualSPHysics/SPHinXsys 同类）；新增**启停瞬态窗**（`settle_time` 内用 `viscosity_settle`/`xsph_settle` 强阻尼，波场期回物理粘度 `ν=1e-6`）+ 传播期 δ-SPH `density_diffusion=0.1`；波高用"已知频率谐波最小二乘幅值"为主估计、零穿越为交叉校验 |
| files | `src/tensorsph/wave_flume.py`（既有）、`src/tensorsph/wave_flume_grade.py`（新）、`benchmarks/wave_flume/{run,analyse}.py`、`tests/test_wave_flume_grade.py` |
| conventions | 判据 G1/G2 如 §3；分析窗从"ramp 结束 + 波前过末探头"之后起；同一 `surface_offset` 用于所有分辨率（跨分辨率可比） |
| transfer | 迁移到其它自由面造波算例时：①先测量估计器（合成信号单测）②再跑分辨率阶梯 ③衰减一律与 `laminar_boundary_layer_damping` 对照后再归因 |
| evidence | `docs/dev_report_wave_flume_attenuation.md`（六段＋误差表）、`benchmarks/wave_flume/report.json`（算例记录与评级原始数字）、`tests/test_wave_flume_grade.py`（估计器 13 项单测） |

## 5. 本轮不做（显式排除）

* 主动吸收（`active_absorption`）标定：属反射系数验收，不在本锚点判据内。
* 孤立波传播保真度：另一算例族（Goring 造波），本单只做规则波。
* 三维波槽：`wave_tank` 分支另有账本条目。