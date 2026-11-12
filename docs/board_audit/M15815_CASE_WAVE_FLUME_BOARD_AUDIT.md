# M15815 · CASE-WAVE-FLUME 交差核验（board-audit）

**被审对象**：mission **m15493**（锚点 `CASE-WAVE-FLUME`，仓 TensorSPH，status=succeeded，
PR #505 merged，HEAD `b7de27543919`）
**审计单**：m15815 ｜ 类型 = 交差核验（非重做）｜ verdict_by = `board-audit`
**唯一裁判**：`tensor_benchmark_cases WHERE anchor_slug='CASE-WAVE-FLUME'` → **0 行**
（全表 23 行、repo 含 SPH 的 0 行）→ **登记基准缺席**，本文按被审单 goal 自带的判据
（ref: 文献波高衰减；容差 <10%）扮演裁判，并逐项标注该判据无平台登记。

---

## ① 案例介绍（物理背景与判据）

2-D 数值波槽（`src/tensorsph/wave_flume.py`，1208 行）：WCSPH（密度求和 + Tait 状态方程 +
对称压力梯度 + Morris 粘性 + Adami 壁面压强外推）+ 活塞造波 + 幂次海绵层 + 波高计阵列。
被审单声明的判据两条（机器可裁）：

* **G1 造波保真**：`|H_meas − H_target| / H_target ≤ 10%`，`H_target` 由活塞一阶传递函数
  `H/S = 2(cosh 2kh − 1)/(sinh 2kh + 2kh)` 给出 —— Biesel & Suquet 1951；Dean & Dalrymple 1991
  《Water Wave Mechanics for Engineers and Scientists》eq. 6.38。
* **G2 衰减归因**：沿波高计拟合 `H = H0 exp(−μx)`，与层流底边界层 `μ = σ²√(νσ)/(4 g c_g sinh²kh)`
  对照 —— Putnam & Johnson 1949；Hunt 1952；Mei 1983 §4.2；Dean & Dalrymple 1991 §5.4。

**审计独立复算的参考值**（`scripts/audit/m15815_step2_references.py`，Newton 解色散 +
闭式传递函数 + 闭式衰减，**不 import 被审仓库任何函数**）：

| 工况 | k [1/m] | λ [m] | c [m/s] | c_g [m/s] | H_target [m] | 冲程 S [m] | μ [1/m] | 2 m 物理衰减 |
|---|---|---|---|---|---|---|---|---|
| **分级工况 T=1.0 s, h=0.25 m** | 4.818973 | 1.303843 | 1.303843 | 0.936528 | 0.04000 | 0.03440 | 1.168472e-03 | 0.2334% |

与被审仓库 `wave_flume_grade.laminar_boundary_layer_damping` 交叉核对：三个工况
（T 1.0/h 0.25、T 1.0/h 0.20、T 0.8/h 0.15）相对差分别 1.9e-16 / 4.8e-16 / -1.5e-16
→ **尺子本身是对的**（这一点被审单没虚标）。

## ② 怎么做的（审计流程）

1. **取证**：读 `tensor_reports` id=14606（m15493 报告全文）、`tensor_dev_missions` id=15493 的
   `result/checks/changed_files`、`git archive b7de27543919` 干净检出（含 `runs/` 两条实测记录）。
2. **三件套核验**：工件是否存在并入库 → `git show --stat b7de275`；ref vs got 数值 → 用**被审
   仓库自己的判级链**（`wave_flume_grade.grade_regular_record`）跑已提交的两条 record；repro_cmd
   → 原样执行（run + analyse 两段）。
3. **独立复算**：另写一条代码路径（Newton 解色散 + LS 谐波拟合）复算参考值与波高，避免
   "用被审的尺子量被审的东西"自证。
4. **单测复核**：干净检出里 `pytest tests/test_wave_flume_grade.py tests/test_wave_flume.py -q`
   → **54 passed**（其中 grade 文件单项 **13 passed**，与被审单 `checks` 声明一致）。
5. **判级**：按功能实质给 L0–L5，不因报告完整度加分。

## ③ 计算结果（数字表，程序化生成）

### 3.1 已提交工件的实际工况 ≠ 被审单参数表

| 项 | 被审单 §3 参数表（分级工况） | 已提交 `runs/wf_local_D20.json` | 已提交 `runs/wf_quick.json` |
|---|---|---|---|
| 水深 h | 0.25 m | **0.20 m** | **0.15 m** |
| 波高 H | 0.04 m（H/h=0.16） | **0.030 m** | **0.0225 m** |
| 周期 T | 1.0 s | 1.0 s | **0.8 s** |
| dx / 粒子数 | 0.0125 m（D20） | 0.0100 m / 6400 | 0.0107 m / 4186 |
| 模拟时长 | 8.0 s（+1.0 s settle） | **3.701 s** | **3.035 s** |
| 判级窗 `--start 4.0` | 报告 repro_cmd 指定 | **无此窗（记录到 3.70 s 就结束）** | **无此窗（3.04 s）** |

→ 两条已提交记录**都不是**分级工况；被审单 §4 逐项写"未测"在语义上没错，但把"跑出来了
但是坏的"降格说成"没跑出来"。

### 3.2 已提交记录的数值健康度（发散筛查）

| 记录 | max\|v\| [m/s] | 声速 c0 | 最大密度压缩 | min ρ/ρ0 | 逃逸粒子 | 步数 |
|---|---|---|---|---|---|---|
| wf_local_D20 | **109.28** | 30 | **0.449** | 0.269 | 144 | 38770 |
| wf_quick | **42.62** | 30 | 0.101 | 0.269 | 0 | 27570 |

max|v| = 3.6×c0（Mach 3.6）、密度压缩 44.9%、min ρ/ρ0 = 0.269 —— WCSPH 在弱可压假设
（|ρ/ρ0−1| ≲ 1%）下**已经失效**，这不是规则波记录。时间线（probe 1 的 max|v|）：

| t [s] | 1.04 | 1.73 | 2.43 | 3.12 | 3.35 | 3.70 |
|---|---|---|---|---|---|---|
| max\|v\| [m/s] | 0.105 | 1.071 | 2.972 | 3.578 | **63.73** | 38.55 |

造波开始（settle 0.8 s 后）约 2 s 内速度由 0.1 m/s 涨到 3 m/s（≈32× 活塞峰值速度
Sω/2≈0.093 m/s），随后爆掉。

### 3.3 按被审仓库自己的判级链判这两条记录（ref vs got）

`wf_local_D20`（目标 0.030 m，10% 门）：

| 探头 | x [m] | H_meas（被审链） | 误差 | H（审计独立 LS） | 误差 |
|---|---|---|---|---|---|
| 探头 1 | 1.031 | 0.03916 | +30.5% | 0.05972 | +99.1% |
| 探头 2 | 1.182 | 0.03520 | +17.3% | 0.05668 | +88.9% |
| 探头 3 | 1.334 | 0.02457 | -18.1% | 0.04098 | +36.6% |
| 探头 4 | 1.485 | 0.03535 | +17.8% | 0.05689 | +89.6% |

`wf_quick`（目标 0.0225 m）：

| 探头 | x [m] | H_meas（被审链） | 误差 | H（审计独立 LS） | 误差 |
|---|---|---|---|---|---|
| 探头 1 | 1.037 | 0.02114 | -6.1% | 0.02553 | +13.4% |
| 探头 2 | 1.139 | 0.01489 | -33.8% | 0.02053 | -8.7% |
| 探头 3 | 1.241 | 0.02666 | +18.5% | 0.03210 | +42.7% |
| 探头 4 | 1.343 | 0.02314 | +2.9% | 0.02859 | +27.1% |

| 判据 | wf_local_D20 | wf_quick |
|---|---|---|
| G1 worst 误差（10% 门） | **30.54% → FAIL** | **33.82% → FAIL** |
| G2 μ_meas | **0.4398 1/m**（理论 1.790476e-03，比值 **246×**） | **−0.8358 1/m（负衰减）**（理论 2.898554e-03） |
| 色散 | 窗内上升过零不足 → 无法判 | λ_meas/λ_theory−1 = **+470%**、周期误差 **+168%** |
| 干点比例 | 4 探头均 0 | 4 探头均 0 |

**这就是本锚点现存唯一的实测证据：判负**。被审单 §6 把这两条记录写成"未写出 JSON"，
`git show --stat b7de275` 却显示四个文件都在 01:39 入库 → 报告结论与工件正面矛盾。


## ④ 结果图

* `docs/board_audit/m15815/fig1_committed_series.png` —— 已提交两记录的自由面时程 vs 目标
  ±H/2 包络：幅值不连贯、相位漂移，同时给出 max|v|/压缩率，肉眼可辨"发散而非规则波"。
* `docs/board_audit/m15815/fig2_height_error.png` —— 逐探头波高误差（被审链柱 vs 审计独立 LS
  点线）叠 10% 门：两记录最差 30.5%/33.8%，且两估计器最大差 52%，说明记录不构成单频波。
* `docs/board_audit/m15815/fig3_reference_ruler.png` —— 参考尺子自证：H/S 与 μ(T,h) 闭式
  复算曲面 + 分级工况点，证明判据参考值可独立复现（不是死数字）。

## ⑤ 注意点（坑与局限）

1. **估计器不是瓶颈，格式才是**：波高链在合成信号上 13 项单测通过（H 恢复误差 <0.1%、
   μ 恢复 <1e-9），但作用在发散记录上给出 +30.5% 而不是 +99%（审计独立估计）——
   两条路都判 FAIL，**结论对估计器选择稳健**，但**误差量级不稳定**：先修格式再谈百分数。
2. **窗口必须声明**：报告 repro_cmd 用 `--start 4.0`，而提交记录只到 3.0–3.7 s；
   原样执行 → `FAILED: fewer than two usable gauges`（实测）。判据窗应写进 config 并写入 record。
3. **分辨率**：分级工况 H/dx = 3.2（D20）/ 5.12（D32），远低于 SPH 波槽常规 H/dx ≳ 10；
   λ/dx = 121 尚可，但波高只有 3 个粒子 → 10% 波高门在此分辨率下缺乏收敛性依据。
   "两档梯度"不是收敛性证据（两档都在同一粗区间）。
4. **物理衰减被数值耗散淹没**：分级工况物理 μ = 1.168e-3 1/m，2 m 只衰 0.233%；
   实测 0.44 1/m（≈246×）→ G2 在当前数值耗散下不可判。
5. 审计未做：D32 复跑（成本 4×）、GPU 路径、与 OpenFOAM/DualSPHysics 的交叉验证
   （本机无此软件）、m15493 之后是否已有新单覆盖本锚点。

## ⑥ 误差判定（关键量 vs 基准，显式回答"是否 <3%"）

| 关键量 | 基准 | 实测 | 误差 | <3%？ |
|---|---|---|---|---|
| 分级工况波高 G1 | H_target = 0.0400 m（Biesel 传递函数） | **未产出有效记录**（本级唯一记录判负 30.5%/33.8%） | 不可判 | **否（无证据）** |
| 记录 A 波高（h=0.20 工况） | 0.0300 m | 0.0392 / 0.0246 m（探头 1/3） | +30.5% / −18.1% | **否** |
| 记录 B 波高（h=0.15 工况） | 0.0225 m | 0.0211 / 0.0149 m | −6.1% / −33.8% | **否** |
| 衰减 μ（记录 A） | 1.7905e-3 1/m | 0.4398 1/m | 246× | **否** |
| 参考尺子自证（审计 vs 仓库闭式） | 闭式解 | 3 工况相对差 ≤5e-16 | ~0 | **是** |
| 波高估计器（合成信号） | 解析 H=0.05 m | 见单测 | <0.1% | **是** |

**结论**：判据参考（尺子）达标；**被测量（波槽真实工况）无一条达标记录**，
本锚点算例池验收 **不成立**。

## 判级

| 级别 | 判定 | 依据 |
|---|---|---|
| L0 文档 | 达 | 方法研究 + 六段报告齐全（`docs/wave_flume_method_research.md`、report id=14606） |
| L1 数学/单元验证 | 达 | `tests/test_wave_flume_grade.py` 13 passed（干净检出自跑复核）；解析锚自洽 |
| **L2 独立功能实现** | **达（判级=2）** | `wave_flume.py`(1208 行真求解器) + `wave_flume_grade.py`(441 行) 输入→计算→输出，`analyse.py` 端到端出表出图 |
| L3 系统集成 | 部分 | 包内模块 + 被 `freesurface`/`startup_quiescence`/`couette` 复用；但**不在** `tensorsph/verification/` 梯级、无平台 API 入口 |
| L4 基准验证 | **不达** | 分级工况无有效记录；唯一记录 30.5%/33.8% 判负；衰减 246× 理论；repro 的 `--start 4.0` 在原样工件上不可执行；`tensor_benchmark_cases` 无本锚点行 → 连裁判都没登记 |
| L5 科研工程成果 | 不达 | — |

**裁定：level = 2**（真代码 + 单元验证成立；L4 基准验证在功能实质上不成立 →
`reopened`，返修项见下）。

### 返修清单（reopened 的施工单）

1. **R1 修格式稳定性**：造波启动 2 s 内 max|v| 涨到 32× 活塞速度后爆掉 → 复核活塞冲程/速度
   限幅、`viscosity_settle→physical` 切换步、δ-SPH 与 XSPH 的耦合；判据先加机器门
   `max|v| < 0.3 c0` 与 `max compression < 0.02`（现在两条记录分别 3.64×c0 / 0.449）。
2. **R2 提交"真"分级工况**：h=0.25/H=0.04/T=1.0，simulate 8 s，H/dx ≥ 10（D32 起）；
   并把 `analysis_window` 写进 `config`，`analyse.py` 从记录里读默认窗，消掉 `--start 4.0`
   对 3.7 s 记录的不可执行。
3. **R3 登记裁判**：`INSERT INTO tensor_benchmark_cases(anchor_slug, case_name, ref_source,
   metric, tolerance, repo) VALUES ('CASE-WAVE-FLUME','wave-height G1','Dean & Dalrymple 1991
   eq. 6.38','|H-H_target|/H_target','0.10','TensorSPH')` 及 G2 行 —— 否则下一单仍无唯一裁判。
4. **R4 报告与工件一致性门**：报告写"未写出"而工件已入库，属自述与证据矛盾；建议 gate 增
   `report_claims.artifacts_consistent` 检查（工件存在则必须给出 got 判级）。

### 审计产物

* `runs/m15815/grade_committed_records.json`、`runs/m15815/independent_references.json`、
  `runs/m15815/figures.json`、`runs/m15815/verdict_evidence.json`、`runs/m15815/audit.json`
* `runs/m15815/repro_wt/`（`git archive b7de27543919` 干净检出 + 审计复跑工件）
* `docs/board_audit/m15815/fig*.png`、本文
* 脚本：`scripts/audit/m15815_step1_grade_records.py`、`m15815_step2_references.py`、
  `m15815_step3_figures.py`、`m15815_step4_doc_verdict.py`、`m15815_registry_probe.py`
