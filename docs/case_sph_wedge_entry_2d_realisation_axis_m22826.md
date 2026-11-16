# CASE-SPH-WEDGE-ENTRY-2D — 独立实现轴：把「同一条轨迹换估计量」换成「换离散相位」

**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力；ref: Zhao & Faltinsen 1993 JFM 246:593；容差 `< 15 %`）
**本单**: m22826（generalist 席自找活）　**仓库**: TensorSPH　**日期**: 2026-11-16

本单**没有**新判据、新参考值、也没有放宽门槛。它做的是上一轮登记的**欠账**（`docs/case_sph_wedge_entry_2d_noise_floor_m22798.md` ⑦-3）：

> 「要真判开这个锚点，最便宜的路是**独立实现**而不是更细网格…… 需要的改动很小但属于求解器：
> `WedgeConfig` 增加预声明初值扰动…… 本单按『只改工作树、最小改动』没有动求解器。」

本单就动了这个求解器开关，并在 `dx = 4 mm` 上把整条链路（扰动 → 跑 → 聚合 → 判决规则）跑通。

## ① 案例与问题

β = 30° 对称刚体楔以恒速 `V = 1 m/s` 入静水，重力/表面张力关，自相似（唯一长度尺度 `Vt`）。
受理指标是湿面**峰值砰击压力系数** `Cp_max = p_max / q`，`q = ρV²/2`。

历史四轮（pooled 120-bin m17548、pooled 门控 m17899、分辨率带 m22695、cell footprint m21677）
的读数在 `6.8 … 13.4 q` 之间摆动近 2 倍；m22798 把残差标度律量化成 `dx^{+0.64}`，
判为**极值偏置**（噪声）而非格式收敛误差，并指出：61 帧取自**同一条**轨迹，`N_eff ≈ 47`
只能把标准误压到 +16.6 %，落在 15 % 门槛之外；真正缺的独立维度是**多实现**。

## ② 怎么做的

### 2.1 求解器改动（`src/tensorsph/wedge_entry.py`，默认关闭）

新增两个**亚网格离散相位**偏移，由种子确定性抽取（`realisation_phases(seed, scale)`）：

| 字段 | 含义 | 界 | 为什么是这个而不是随机抖动 |
|---|---|---|---|
| `phase_x_dx` | 流体格点整体**横向**平移，单位 `dx` | `≤ 0.25` | 刚性平移不改任何粒子间距，因此**初始求和密度与 `mass_unit` 标定保持精确**；随机位置抖动会污染初始密度并激发出与待测量同量级的声学瞬变 |
| `gap_jitter_dx` | 叠加到 `initial_gap_dx` 的**入水相位**偏移 | `≤ 0.25` | 时钟仍把 `t = 0` 重定为首次接触，观测窗与自相似横坐标不变 |

竖向相位（平移自由面）**刻意不提供**：那会移动自由面、从而改变入水时刻，是改物理而不是改相位。
`scale = 0` 时两偏移恒为 0 —— 已提交的全部历史工件仍**逐位复现**（负控测试）。

### 2.2 驱动与判据（新增 `benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py`）

一次调用跑一个实现；`--analyse` 聚合目录内全部实现。预声明规则（在任何实现工件存在之前写定）：

```
obs_i   = by_frames["61"].resolution_band.peak_median     （ladder_closure 判的那个统计量）
truth_g = bias_report.json 的 synth_smoothed.peak_median   （同算子 Wagner，与种子无关）
r_i     = obs_i / truth_g - 1
U_real  = max( t95(N-1) * sd(r_i) / sqrt(N) ,  median_i(band spread_i) )
judgeable_g  <=>  U_real <= 0.15
verdict      = ladder_closure.verdict(rows)               （原规则，未改）
```

另两条**同样预声明**的辅助问题：**A1 实现是否真的相互独立**（`cp_centre` 逐帧序列的两两
Pearson 相关；相关 → 1 说明扰动没去相关，此时实现集给出的不确定度是**下界**，方向危险，必须报）；
**A2 答案是否依赖扰动幅度**（`scale ∈ {0, 0.25, 1}`，若随幅度单调变化则旋钮不是在采样相位）。

### 2.3 工具链

`src/tensorsph/wedge_entry.py`（相位开关）· `benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py`
（`collect(..., seed, scale)` 透传，默认路径不变）· `seed_ensemble.py`（跑 + 聚合）·
`tests/test_wedge_realisation.py`（12 项）。

## ③ 计算结果（`dx = 4 mm`，本单实跑）

主表（`docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json`）：

| 实现 | 相位 `phase_x_dx` | 入水相位 `gap_jitter_dx` | 步数 | `obs`（61 帧分辨率带中位） | 带宽 spread | `r_i` vs 同算子 Wagner |
|---|---|---|---|---|---|---|
| 无扰动（`scale=0`，= 已提交算例） | 0.0000 | 0.0000 | 329 | **10.041** | 23.5 % | +34.0 % |
| seed 1 | −0.1828 | +0.1737 | 347 | 8.680 | 24.4 % | +15.8 % |
| seed 2 | +0.2280 | +0.2239 | 352 | 8.676 | 11.9 % | +15.8 % |
| seed 3 | −0.1310 | +0.0221 | 334 | 7.509 | 12.3 % | +0.2 % |

聚合（N = 3，尺度 1.0 者入集）：残差中位 **+26.3 %**，全距 `+9.3 % … +26.3 %`，
`U_ci = 24.4 %`，`U_band = 12.3 %`，**`U_real = 24.4 % > 15 %` → 该网格不可判**。

**A1（独立性）**：`cp_centre` 逐帧序列两两 Pearson 相关中位 **0.797**（不是 1）——
四种相位确实把轨迹分开了；但 0.797 也说明它们并非完全不相关（同一物理过程 + 同一算子），
所以 `U_ci` 应视为**该实现规模下的估计**，不是极限精度。

**A2（幅度依赖）**：`scale = 0.25`（seed 1/2）读 8.249 / 8.093，`scale = 1.0`（seed 1/2）
读 8.680 / 8.676 —— 两者与无扰动的 10.041 的差距**同量级、不随幅度单调**，
与「旋钮在采样离散相位」相容，与「旋钮在改物理」不相容（N 太小，只能说相容）。

**最值得记下来的一条**：无扰动的对称格点读到 `10.041 q`，而三个相位化实现读到 `7.51…8.68 q`，
**无扰动值落在实现分布的高端之外**。也就是说历史单轨迹读数（10.04 / 6.07 / 6.80 / 8.98）
不宜再被当作"该网格的读数"，而应被当作**该网格实现分布中的一个成员**——这正好解释了它们
摆动近 2 倍：摆动的是实现的抽样，不是流动。N = 3 太小，不足以给出分布本身（见 ⑤）。

## ④ 结果图

`docs/figures/wedge_realisation_m22826.png`（`scripts/` 内脚本从工件渲染，数字零手抄）：
左panel = 每个实现的 `cp_centre(t)` 逐帧序列（可视化 A1：曲线形状相同但逐帧错开）；
右panel = 每实现的 `obs` 与同算子 Wagner 参考（7.495 q）的水平线，及实现残差区间。

## ⑤ 注意点 / 局限（诚实声明）

1. **本单只跑到 `dx = 4 mm`**。锚点判开需要的 `dx = 2 mm / 1 mm` 实现集已递交集群
   **cj4427**（4 mm seed 4–8、4 mm `scale 0.25` seed 3–5、2 mm seed 1–3，8 核，timeout 3600 s），
   递交时排队位次 63，**本单收尾时仍在 queued**，工件未取回 → 细网格的 `U_real` **本单没有读数**，
   状态如实标 `not_run`，不得写成通过。
2. `dx = 4 mm` 本来就是「不可判」的粗网格（历史 `U = 23.5 %`）；本单的价值是**方法链**
   （相位 → 实现 → `U_real` → 判决）跑通并可复现，不是这个网格的判决。
3. N = 3 的实现集太小：`t95(2) = 4.303` 使 `U_ci` 保守；`sd` 本身的不确定度约 `1/sqrt(2N)`。
   真正的实现规模应 ≥ 6（`t95(5) = 2.571`），这正是 cj4427 的规模。
4. 相位界 `0.25 dx` 只保证流体格点离侧壁 `≥ dx/4`（推导见 `PHASE_X_MAX_DX` 注释）；
   若未来改 `half_width/dx` 使 `nx` 变奇，该推导仍成立（余量 `≥ dx/2`）。
5. 实现集**不能**回答「哪个读数是对的」：它只把**仪器**的不确定度量化。物理残差（是否 < 15 %）
   仍要等细网格实现集，且 `Zhao & Faltinsen 1993` 原文表值仍为 **WITHHELD**
   （本单未获取原文，未把二手 COM 值当原文值使用）。

## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）

| 量 | 基准 | 结果 | 偏差 | < 3 % ? |
|---|---|---|---|---|
| `scale = 0` 的 61 帧分辨率带中位 vs 已提交 `ensemble_dx4mm_c0v20.json` | 10.041483876237576 | 10.041483876237576 | **0（逐位）** | **是** |
| 相位平移一致性：`Δx` 全粒子是否同一常数 | 解析 `phase_x_dx·dx` | 极差 1.4e-17 m | 0 | **是** |
| 侧壁/楔体格点是否被移动 | 位移应为 0 | 逐位相同 | 0 | **是** |
| `t_contact` 偏移 vs `gap_jitter_dx·dx/V` | 解析 | 一致到 1e-12 相对 | 0 | **是** |
| `U_real`（4 mm）vs 15 % 判据 | 0.15 | 0.244 | 高于门槛 63 % | **否**（故 NOT_DECIDABLE） |
| 实现残差中位 vs 同算子 Wagner | 0 % | +26.3 % | +26.3 % | **否** |
| 细网格（2 mm / 1 mm）实现集 | — | **not_run**（cj4427 未出队） | — | **未判** |

**回答「是否 < 3 %」：可复现性层面是（逐位 0 %）；判定量层面不是。**
`honest-FAIL`：本单**没有**把锚点判开，锚点维持 `open` / `NOT_DECIDABLE`。

## ⑦ 结论与给下一单的可证伪登记

1. **方法轴已通**：`WedgeConfig` 有了预声明、默认关闭、逐位可复现的实现开关，
   实现集 → `U_real` → 既有 `ladder_closure.verdict` 全链路可一行复现。
2. **登记预测（下一单可直接证伪）**：按 `U ∝ N^{-1/2}`，若 `dx = 1 mm` 的单实现标准误确为 +16.6 %
   （m22798），则 **N = 6 个实现给出 `U_ci ≈ 6.8 % < 15 %`**，该网格转为可判；
   若 cj4427/后续在 N = 6 时仍 `U_ci > 15 %`，则「实现维度足够」这一路线**被推翻**，
   该锚点只能靠更细网格或换指标（如换用空间平均/换能器 footprint 口径）推进。
3. **注意一个方向性风险**：4 mm 上「无扰动 10.04 vs 相位化 7.5–8.7」提示**对称格点可能系统性偏高**。
   若细网格复现该模式，则历史 PASS 读数（`literature_regrade` 的 6.805，`U = 6.5 %`）应重新
   标注为其实现分布的一个**高值成员**，而不是该网格的代表值。这是下一单必须正面回答的问题。

## ⑧ 复现

```bash
# 一个实现（本地 ≤ 20 s @ dx=4mm）
python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py --dx 0.004 --seed 1 \
    --threads 2 --out docs/artifacts/case_sph_wedge_entry_2d_realisations
# 聚合 + 判决
python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py --analyse \
    --out docs/artifacts/case_sph_wedge_entry_2d_realisations
# 测试（12 项，含两条负控）
python -m pytest tests/test_wedge_realisation.py -q
```

## ⑨ 与既有红测的关系（不掩盖）

工作树里 `tests/test_case_sph_wedge_entry_2d.py::test_refs_derived_is_computed_not_typed`
**在本单之前就已红**：它断言 `zhao_faltinsen_1993.status == "WITHHELD"`，而后续轮次已把
`refs.json` 该条目升级为 `REAL`（二手开放获取表 COM 列，带出处与量纲换算记录）。
另：该文件在不设 `PYTHONPATH=benchmarks/case_sph_wedge_entry_2d` 时 collection 即失败
（`literature_refs` 不在搜索路径）。两者均为**既有缺陷**，不属于本单改动，
本单未修改它们以避免与该轮证据口径冲突，登记在此供下一单处理。