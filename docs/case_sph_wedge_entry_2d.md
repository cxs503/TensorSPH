# CASE-SPH-WEDGE-ENTRY-2D — 二维楔形体入水砰击压力 (TensorSPH)
**算例池锚点**：`CASE-SPH-WEDGE-ENTRY-2D` · **仓库**：TensorSPH · **参考**：Zhao & Faltinsen (1993) J. Fluid Mech. 246:593 (DOI 10.1017/S002211209300028X) · **容差**：峰值砰击压力系数 <15 %
**本文档由 `benchmarks/case_sph_wedge_entry_2d/make_doc.py` 从工件目录程序化生成**：文中每个数字都从 `summary.json` / `refs_derived.json` / `refs.json` 读出，人工不写数。

## ① 问题与参考出处
二维对称刚性楔形体（deadrise angle β = 30°，即每个面与未扰动自由面的夹角）以恒定垂向速度 V 入静水，忽略重力与表面张力。恒定速度入水在忽略重力时是**自相似**问题：唯一长度尺度是 V·t，Wagner 湿半宽 c(t) = C·V·t，C = π/(2 tan β)。判据量 = 楔形体湿表面峰值砰击压力系数 Cp_max = p_max / (½ρV²)。
**Zhao & Faltinsen (1993)**（任务指定参考）：Zhao, R. & Faltinsen, O. (1993) Water entry of two-dimensional bodies. Journal of Fluid Mechanics 246, 593-612. DOI 10.1017/S002211209300028X
* 书目核验：GET https://api.crossref.org/works/10.1017/s002211209300028x -> HTTP 200, title='Water entry of two-dimensional bodies', volume=246, page=593-612, authors=[Zhao R. (first), Faltinsen O.]
* 开放获取核验：GET https://api.openalex.org/works/doi:10.1017/s002211209300028x -> open_access.is_oa=false, oa_status='closed', any_repository_has_fulltext=false
* 已取得摘要（OpenAlex record W2051783622, abstract_inverted_index (publisher-deposited abstract), retrieved 2026-11-12）：“A numerical method for studying water entry of a two-dimensional body of arbitrary cross-section is presented. It is a nonlinear boundary element method with a jet flow approximation. The method has been verified by comparisons with new similarity solution results for wedges with deadrise angles varying from 4 to 81 degrees. A simple asymptotic solution for small deadrise angles alpha based on Wagner (1932) agrees with the similarity solution for small alpha.”
* **β = 30° 的具体表/图数值：WITHHELD** —— The beta = 30 deg number lives in the paper's tables/figures; the article is closed access (OpenAlex oa_status=closed, no repository full text) and the ecosystem has no document-delivery channel and no registered copy (checked 2026-11-12: knowledge_base LIKE '%zhao|faltinsen|wedge|deadrise%' -> 0 numeric rows; oss_solutions -> 0 rows). Reporting a remembered or secondary-source number as the original table value would be fabrication, so the cell stays empty and the comparison is WITHHELD (not PASS, not FAIL).
* 闭合条件：The paper delivered to the repo as a text-extractable file (PDF/text) with page/table coordinates for the wedge similarity-solution pressure coefficient at beta = 30 deg; then the digitised value is entered here with page/table/figure coordinates and the comparison re-run with --stage accept (no solver change needed).
**Wagner 解析锚（本算例的主判据，可机器裁决）**：C = π/(2 tan β) = **2.7207**，Cp_apex = C = **2.7207**，Cp_max = C² + 1 = **8.4022**（峰值位于 τ_peak = |x|/c = √(1−1/C²) = 0.93）。公式来源 `src/tensorsph/wedge_entry.py`（模块 docstring 式 (1)-(3)，`wagner_pmax`/`wagner_p_centre`），由 `run.py: refs_derived()` 现算，**不手抄**。该解与 Z&F 论文属同一理想化（无粘不可压势流、忽略射流），且 Z&F 摘要自述“小 deadrise 下 Wagner 渐近解与其非线性相似解一致”，故 30° 处 SPH-vs-Wagner 误差**不能替代** SPH-vs-Z&F，但它是本算例唯一可即时机器裁决的参考。

### 预声明设计（先于任何一次运行写死，见 `run.py` 常量）
* 算例矩阵：网格阶梯 β=30°, V=1 m/s, dx ∈ {4, 2, 1, 0.5} mm（相邻比 2，Richardson/GCI 要求）；自相似控制点 β=30°, V=5 m/s, dx=1 mm。
* 判据（`THRESHOLDS`）：最细网格 |Cp_band_peak − Cp_max,Wagner| ≤ 15 %；最粗网格 ≤ 30 %（趋势必须收敛）；顶点 Cp ≤ 15 %；最细对 GCI ≤ 15 %；V 自相似偏差 ≤ 5 %；弱可压质量守恒 max ρ/ρ₀ − 1 ≤ 5 %。
* 停止规则：网格阶梯跑满 4 级即停；若 0.5 mm 级在资源上限内未完成，报告以 1 mm 级为最细级并显式声明缺失（不插值、不外推冒充）。
* 指标定义（互不等价，逐个报）：`cp_band_peak` = τ≤1 内壁面 Cp 的窗口系综平均（后 5 帧按 τ 分箱平均）最大值；`cp_at_peak_tau` = 同一剖面在 Wagner 自身峰值横坐标处读数；`cp_centre_mean` = 顶点 Cp 后 5 帧平均；`cp_interior_max`/`cp_footprint_max` = SPH |x|≤0.9c 窗 / 全足迹的时程最大值（后者含自由面喷溅根部，Wagner 在该处发散，故只报不判）。
* 指标集修订声明：粗网格试点（dx=4/2 mm）暴露 WCSPH 壁面压力逐帧抖动（顶点 Cp 在 1.4–8.2 之间跳），因此**在开始细网格计算前**把判据的主量从“全足迹时程最大”改为“τ 分箱系综峰值”，阈值**未放宽**（15 % 不变），改动仅为指标定义与噪声抑制方式，记录于此以保可审计。

## ② 方法与实现
复用 TensorSPH 现成能力（未另起炉灶）：`src/tensorsph/wedge_entry.py` 的 `WedgeImpact` —— 弱可压 SPH（Tait 状态方程 p = B[(ρ/ρ₀)^γ − 1]，γ=7，B = ρ₀c₀²/γ）、Monaghan 人工粘性、Adami et al. (2012) 静止/运动壁面粒子压力外推 p_w = Σ p_f W / Σ W、每步密度求和、每步重建邻域（刚体运动使 Verlet 表失效）、格点标定质量使满支撑粒子恰好 ρ=ρ₀。
新增代码只做**驱动 + 判据 + 聚合**，不改求解器：
* `benchmarks/case_sph_wedge_entry_2d/run.py` —— stages `run` / `matrix`（可 `--shard i/n`）/ `accept`；每例一个 JSON 工件（配置、环境、逐帧时程、τ 分箱壁面压力剖面），`accept` 聚合出 `summary.json` / `table_cp.csv` / `refs_derived.json`。
* `benchmarks/case_sph_wedge_entry_2d/refs.json` —— 参考值**出处**唯一真源（含 Z&F 的书目核验、摘要获取方式、数值 WITHHELD 理由）。
* `benchmarks/case_sph_wedge_entry_2d/make_figures.py` / `make_doc.py` —— 图与本文档，均只读工件 JSON。
复现：
```bash
python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix --shard 0/5 --out-dir artifacts/case_sph_wedge_entry_2d
python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept
python benchmarks/case_sph_wedge_entry_2d/make_figures.py
python benchmarks/case_sph_wedge_entry_2d/make_doc.py
```

## ③ 参数表（无量纲数完整）
| 参数 | 值 | 备注 |
|---|---|---|
| deadrise β | 30° (0.5236 rad) | 任务指定 |
| 入水速度 V | 1 m/s（阶梯）/ 5 m/s（自相似控制） | 恒速 |
| 密度 ρ₀ | 1000 kg/m³ | 水 |
| 声速比 c₀/V | 20 | 人工声速，Mach = V/c₀ = 0.05（弱可压） |
| 状态方程 | Tait, γ = 7 | B = ρ₀c₀²/γ |
| 人工粘性 α | 0.03 | Monaghan |
| 核函数支撑 h/dx | 1.3 | 三次样条（solver 内部） |
| Courant | 0.35（声速） | dt ≈ 0.35 dx/c₀ |
| 重力 | **关闭** | 自相似前提；Fr = V/√(gL) → ∞ |
| 表面张力 / Weber | 不建模 / We → ∞ | 惯性主导的砰击阶段 |
| Reynolds | 不建模（无粘 + 人工粘性） | 势流对照下的标准设置 |
| 楔形半宽 / 槽宽 | 0.20 m / 0.40 m | 观测窗止于 c_end = 0.05 m |
| 槽深 | 0.12 m | 有限深误差折入算例误差 |
| 帧数 | 13 | t ∈ [0, t_end]，t_end = 2 c_end tanβ/(πV) |
| C = π/(2 tanβ) | 2.7207 | Wagner 湿宽系数 |
| τ_peak | 0.93 | Wagner 峰值横坐标 |

| case | dx [m] | 流体粒子数 | 壁面粒子数 | dt [s] | t_end [s] | steps | τ_peak 处每 dx 覆盖 |
|---|---|---|---|---|---|---|---|
| `beta30_V1_dx40e-4` | 0.004 | — | — | — | — | 314 | — |
| `beta30_V1_dx20e-4` | 0.002 | — | — | — | — | 556 | — |

（粒子数/dt/t_end 见各 case 工件的 `derived` 字段；本表 steps 直接来自工件。）

## ④ 对照表（ref vs got + 误差 + 判据）
### 4.1 参考值
| 参考 | Cp_max | 状态 | 来源 |
|---|---|---|---|
| Wagner (1932) 解析 | 8.4022 | REAL | 仓内现算 `wagner_peak_over_q` |
| Zhao & Faltinsen (1993) β=30° | — | **WITHHELD** | 闭源 + 仓内无登记副本，见 refs.json |

### 4.2 SPH 结果（全部工件：`artifacts/case_sph_wedge_entry_2d/`）
| case | kind | dx [mm] | V [m/s] | steps | Cp band peak | Cp@τ_peak | apex Cp | interior max Cp | footprint max Cp |
|---|---|---|---|---|---|---|---|---|---|
| `beta30_V1_dx40e-4` | ladder | 4 | 1 | 314 | 32.01 | 3.971 | 4.527 | 32.01 | 32.01 |
| `beta30_V1_dx20e-4` | ladder | 2 | 1 | 556 | 28.05 | 4.286 | 4.873 | 19.83 | 28.05 |

### 4.3 误差与判决（对 Wagner 解析锚）
| case | dx [mm] | 阈值 | Δ Cp_band_peak | 判决 | Δ Cp@τ_peak | 判决 | Δ apex Cp | 判决 | Δ 全足迹峰值（仅报） | 质量守恒 |
|---|---|---|---|---|---|---|---|---|---|---|
| `beta30_V1_dx40e-4` | 4 | 30 % | +280.94 % | ❌ FAIL | +52.74 % | ❌ FAIL | +66.37 % | ❌ FAIL | +280.9 % | max ρ/ρ₀−1 = +4.71 % ✅ PASS |
| `beta30_V1_dx20e-4` | 2 | 15 % | +233.81 % | ❌ FAIL | +48.99 % | ❌ FAIL | +79.10 % | ❌ FAIL | +233.8 % | max ρ/ρ₀−1 = +4.61 % ✅ PASS |

**对 Z&F (1993) 的判决：WITHHELD**（β=30° 原文表/图数值不可得；不以记忆值或二手值代替）。可机器裁决的部分见上表。

## ⑤ 收敛性与参数敏感性
| 量 | 细网格 | 粗网格 | dx 比 | GCI | 观测阶 p | 判决 |
|---|---|---|---|---|---|---|
| cp_band_peak | 28.05 (dx=2 mm) | 32.01 (dx=4 mm) | 2 | 0.1765 | n/a | ❌ FAIL |
| cp_at_peak_tau | 4.286 (dx=2 mm) | 3.971 (dx=4 mm) | 2 | 0.09182 | n/a | — |
| cp_centre_mean | 4.873 (dx=2 mm) | 4.527 (dx=4 mm) | 2 | 0.08884 | n/a | — |
| cp_footprint_max | 28.05 (dx=2 mm) | 32.01 (dx=4 mm) | 2 | 0.1765 | n/a | — |

### 5.3 压缩性误差预算（失败根因的定量化）
弱可压 SPH 的压力由密度误差直接生成：Δp = B[(ρ/ρ₀)^γ − 1]，B = ρ₀c₀²/γ。把各例的最大密度偏移换算成压力系数，得到**与解是否正确无关的噪声地板**：Cp_budget = 2(c₀/V)²/γ·[(ρ/ρ₀)^γ − 1]。
| case | max ρ/ρ₀ | c₀/V | Cp 预算（密度噪声地板） | Cp 实测 band peak | 参考 Cp_max |
|---|---|---|---|---|---|
| `beta30_V1_dx40e-4` | 1.0471 | 20 | 43.48 | 32.01 | 8.402 |
| `beta30_V1_dx20e-4` | 1.0461 | 20 | 42.38 | 28.05 | 8.402 |

敏感性两点：(i) 网格阶梯 dx = 4 → 0.5 mm；(ii) 速度 V = 1 → 5 m/s （同 dx，检验恒速自相似性）。

## ⑥ 失败模式与工件
见 `docs/case_sph_wedge_entry_2d.md` 的失败模式小节与 `artifacts/case_sph_wedge_entry_2d/summary.json` 的 `missing` 字段。

## ⑦ 参考值审计后的状态（m17079，2026-11-13）

参考侧审计（`docs/audit_wagner_pressure_reference_m17079.md`）发现两处分母问题，本文件 ④.3 / ⑤ 的误差百分比**已过期**：

1. **单位错配（已修）**：实测 `Cp = p/q`（`q = ½ρV²`），但 `refs_derived()['cp_centre']` 曾写成 `C = p/(ρV²)`，
   比实测小一半。修为 `2C`（= 剖面 τ=0 值 = `wagner_p_centre/q`）。按 §4.2 的实测 `cp_centre_mean = 4.873`
   重算：旧误差 +79.11 %（FAIL）→ **新误差 −10.45 %（<15 %，PASS）**。
2. **剖面 +1 q（未改，机器钉住）**：仓内 `wagner_p_over_q = 2Cs−s²+1`；由其自身势做数值欧拉 Bernoulli 得
   `2Cs−s²`，全足迹差恒为 1 个动压 `q`（0≤τ≤0.95：1.000000 ± 1.2e-7）。故峰值参考可写 `C²+1 = 8.4022`（仓内）
   或 `C² = 7.4022`（欧拉）。**两种参考下 `cp_band_peak` 均 FAIL**（+233.8 % / +278.9 %），算例峰值结论不变。

重判需重跑（原始工件未入 git）：`python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept && python benchmarks/case_sph_wedge_entry_2d/make_doc.py`。
Zhao & Faltinsen (1993) 仍为 **WITHHELD**，本次审计未产生任何替代它的数值。

## ⑧ 测量链复核（m17548，2026-11-13）

上轮 FAIL 判决（`cp_band_peak` +233.8 % @ dx=2 mm）经同一轨迹重算后确认**主要由分箱估计量产生**：
`profile_bins=120` 大于每帧湿面节点数（dx=2 mm 时 62 个），bin 内采样数 0–2，120 bin 峰值 28.05
与该算例逐点最大压力 28.047 相同，即"一个 bin 就是一个粒子"。同一轨迹换分箱数，峰值在
9.22 … 28.05 间变化（极差 3.04×）。噪声鲁棒量在 dx=2 mm 均 <15 %：c/(C·V·t) −0.54 %、
顶点 Cp −10.45 %、峰值位置 τ=0.92–0.94 vs 解析 0.930。峰值 Cp 本身当前**不可判定**
（估计量不确定度 >30 % > 容差），Z&F (1993) 仍 **WITHHELD**。
全文与数据：`docs/case_sph_wedge_entry_2d_metric_resolution_m17548.md`；
工件：`artifacts/case_sph_wedge_entry_2d_metric/`；复现脚本 `benchmarks/case_sph_wedge_entry_2d/metric_resolution.py`。

## ⑨ 同算子复核（m18382，2026-11-13）

历轮把**离散化后的测量值**直接比 Wagner 的**点值**。本单把 Wagner 场推过**同一测量算子**（壁面节点采样 + Adami 核外推 h=1.3dx + 分箱峰值 + 样本门），在同算子上重做比较（未改求解器/算例/阈值，未重跑 SPH；输入仅已入仓的 `docs/artifacts/case_sph_wedge_entry_2d_ensemble/`）：

| 量（61 帧池化） | dx = 2 mm | dx = 4 mm |
|---|---|---|
| 实测峰值 Cp | 5.850 | 10.360 |
| 实测 vs 点值 8.4022（历轮口径） | −30.4 % | +23.3 % |
| 算子真值（Wagner 经同一算子） | 7.494 | 6.870 |
| **同算子残差** | **−21.9 %** | **+50.8 %** |

结论：①测量链的固定偏置（核外推把峰值附近 ~2h 场平均）在 dx = 2 mm 吃掉 **−10.8 %**、dx = 4 mm **−18.2 %**，是负数常数项而非解的误差；②扣掉后最细网格仍 **−21.9 %** 且跨网格**变号**，故**峰值 Cp 仍不可裁决**（预声明的网格一致性判据未过），验收结论不变；③前向模型经两次对表（逐帧湿节点数偏差 ~1–2 个节点、顶点 τ→0 处算子无偏到 0.04 %）后才允许使用；④**更正 m17899 ⑤.2**：“湿面节点随入水加深下降/接触判据失效”被其自身工件否定（`n_wet` 逐帧单调增）；
⑤Zhao & Faltinsen (1993) 仍 **WITHHELD**；残差为负的方向与“线性理论在 β = 30° 高估峰值”一致，但这是**假说**，不作为修正因子。

全文：`docs/case_sph_wedge_entry_2d_operator_bias_m18382.md`；工件：`docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json`；脚本：`benchmarks/case_sph_wedge_entry_2d/{estimator_bias,make_operator_bias_figure,make_operator_bias_doc}.py`；测试：`tests/test_wedge_estimator_bias.py`。
