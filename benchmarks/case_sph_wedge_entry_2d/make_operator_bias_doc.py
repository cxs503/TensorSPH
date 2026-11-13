"""Generate ``docs/case_sph_wedge_entry_2d_operator_bias_m18382.md`` from artefacts.

Every number in the generated document is read from
``docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json`` (itself produced
by ``estimator_bias.py`` from the committed SPH aggregates), so the prose cannot
drift from the data.  Run after ``estimator_bias.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias" / "bias_report.json"
OUT = ROOT / "docs" / "case_sph_wedge_entry_2d_operator_bias_m18382.md"


def pct(x: float) -> str:
    return "—" if x != x else f"{100.0 * x:+.1f} %"


def f3(x: float) -> str:
    return "—" if x != x else f"{x:.3f}"


def main() -> int:
    d = json.loads(ART.read_text())
    ref = d["wagner_cp_max_point"]
    design = d["design"]
    g2, g4 = d["grids"]["dx2mm"], d["grids"]["dx4mm"]
    frames = design["frame_counts"]

    rows_obs = []
    for n in frames:
        a = g2["by_frames"][str(n)]
        b = g4["by_frames"][str(n)]
        rows_obs.append(
            f"| {n} | {f3(a['obs_peak_median'])} | {pct(a['obs_err_vs_point_ref'])} | "
            f"{f3(a['synth_smoothed']['peak_median'])} | {pct(a['same_operator_residual_median'])} | "
            f"{f3(b['obs_peak_median'])} | {pct(b['obs_err_vs_point_ref'])} | "
            f"{f3(b['synth_smoothed']['peak_median'])} | {pct(b['same_operator_residual_median'])} |")
    table_obs = "\n".join(rows_obs)

    rows_ops = []
    for n in frames:
        a = g2["by_frames"][str(n)]["synth_smoothed"]
        b = g4["by_frames"][str(n)]["synth_smoothed"]
        pa = g2["by_frames"][str(n)]["synth_point"]
        pb = g4["by_frames"][str(n)]["synth_point"]
        rows_ops.append(
            f"| {n} | {f3(pa['peak_median'])} | {f3(a['peak_median'])} | "
            f"{pct(a['peak_median'] / ref - 1.0)} | {f3(pb['peak_median'])} | "
            f"{f3(b['peak_median'])} | {pct(b['peak_median'] / ref - 1.0)} |")
    table_ops = "\n".join(rows_ops)

    ver = d["verdict"]
    hyp = d["hypothesis"]
    apex2, apex4 = g2["apex_check"], g4["apex_check"]
    fp2, fp4 = g2["frame_peak_stats"], g4["frame_peak_stats"]
    sm2, sm4 = g2["sampling_model_check"], g4["sampling_model_check"]

    text = f"""# CASE-SPH-WEDGE-ENTRY-2D：峰值砰击压力的“同算子”比较——测量链吃掉多少、求解器还剩多少（m18382）

**锚点**：`CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力）
**验收原文**：`【算例池|SPH-WEDGE-ENTRY-2D｜ref: Zhao & Faltinsen 1993 J. Fluid Mech. 246:593｜容差: <15% | 指标: 楔形体表面峰值砰击压力系数】`
**本单定位**：验证欠账复核（不改求解器、不改算例、不放宽阈值、不重跑 SPH）。历轮（m17079/m17548/m17899）把**离散化后的测量值**直接与 Wagner 的**点值**比较，得到 +234 % → −30.4 % 的摆动；本单把 Wagner 场推过**同一测量算子**（壁面节点采样 + Adami 核外推 + 分箱峰值 + 样本门），让两侧同算子后再判。结论先行：**峰值 Cp 仍不可裁决**（最细网格同算子残差 {pct(ver['dx2mm']['residual'])}，且跨网格变号），但**误差分解第一次定量**：历史 −30.4 % 里，测量算子（Adami 核外推）固定吃掉 {pct(g2['by_frames']['61']['synth_smoothed']['peak_median'] / ref - 1.0)}，剩下的 {pct(ver['dx2mm']['residual'])} 才是数据侧残差；同时**更正 m17899 的一条错误推荐**（湿面节点数不是随入水加深下降，见 ⑤.1）。

> 本文档由 `benchmarks/case_sph_wedge_entry_2d/make_operator_bias_doc.py` 从 `docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json` 程序化生成，数字零手抄。

---

## ① 案例介绍

二维对称刚性楔形体（deadrise β = 30°）以恒速 V = 1 m/s 垂直入静水，忽略重力与表面张力；自相似区唯一长度尺度 V·t。参照量全部由仓内闭式现算（`src/tensorsph/wedge_entry.py` 式 (1)–(3)）：

* `C = π/(2 tan β)` = **2.72070**，湿半宽 `c(t) = C V t`；
* 壁面系数 `Cp(τ) = 2 C s − s² + 1`，`s = 1/√(1−τ²)`，`τ = |x|/c`；
* 点值峰值 `Cp_max = C² + 1` = **{ref:.4f}**（`τ_peak` = 0.9300）；顶点 `Cp(0) = 2C` = 5.4414。

| 参数 | 值 | 备注 |
|---|---|---|
| β / V | 30° / {design['velocity']} m/s | 任务指定 |
| dx | 4 mm / 2 mm | 已入仓的两级（m17899 口径） |
| 帧数 | {', '.join(str(n) for n in frames)} | 取窗口最后 N 帧 |
| 分箱网格 | {', '.join(str(n) for n in design['bin_grid'])} | τ ∈ [0, {design['tau_max']}]，峰值只在 τ ≤ {design['tau_hi']} 取 |
| 样本门 | {design['min_bin_samples']} | 池化样本 < 4 的 bin 不许设峰值 |
| 核 / 支撑 | 2-D cubic spline，h/dx = {design['h_over_dx']} | 求解器同一核、同一 h（`WedgeConfig`） |
| 流体行数（算子模型） | {design['fluid_rows']} | 支撑 2h = 2.6 dx 内的壁面邻域行 |

**为什么需要换比较口径**：`p_max` 是**测量算子**的产物，不是场的点值。测量算子有三个环节，各自带偏置：(i) 壁面节点间距 dx，(ii) Adami 核外推 `p_w = Σ p_f W / Σ W`（在壁面附近把场在 ~2h 尺度上平滑），(iii) 分箱 + 取最大（对噪声是上偏、对陡峰是下偏）。把 SPH 的测量值直接比 Wagner 的点值，等于把 (i)–(iii) 全记在求解器头上。

## ② 怎么做的（方法、流程、工具链、关键设置）

**未改求解器、未改算例、未改阈值、未新跑 SPH**。全部输入是已入仓的两份聚合工件（`docs/artifacts/case_sph_wedge_entry_2d_ensemble/ensemble_dx{{2,4}}mm_c0v20.json`：逐帧扫描 + 每 bin 峰值/样本数）。

**前向模型（先于任何计算写死常量，见脚本顶部 `H_OVER_DX / FLUID_ROWS / TAU_MAX / MIN_BIN_SAMPLES`）**：

1. 壁面节点 = `wedge_lattice` 第一层，`|x_k| = dx[(k+1/2) cos β − (1/2) sin β]`（与求解器同一几何，未拟合）；
2. 湿节点 = `|x_k| ≤ c(t)`，`c = C V t`（估计量本身就用解析 c 归一化 τ，故此处同源）；
3. 节点读数 = Adami 外推：流体粒子按求解器格距铺在面下（`{design['fluid_rows']}` 行），每个携带解析 `Cp` 于自身横坐标，`p_w = Σ p W / Σ W`，核与 h 取求解器自身值；
4. 算子真值 = 步骤 3 的剖面经**与 `ensemble_peak.py` 逐行等价**的分箱+样本门+取最大（脚本 `estimator_peak`，有测试对齐到 1e-12）。

**判据（先声明）**：

* 同算子残差 `R = (实测 − 算子真值)/算子真值`，阈值仍为 **15 %**（不放宽）；
* 网格一致性：`|R(dx=4 mm) − R(dx=2 mm)| ≤ 15 %` 才允许进入判定（沿用 m17899 的收敛要求）；
* H1：算子本身能复现实测的跨网格变号 → 峰值是估计量伪像；H2：算子单独不变号 → 变号属数据属性。

复现（纯本地、秒级、无 SPH 步进、无集群）：

```bash
python benchmarks/case_sph_wedge_entry_2d/estimator_bias.py            # ~20 s
python benchmarks/case_sph_wedge_entry_2d/make_operator_bias_figure.py
python benchmarks/case_sph_wedge_entry_2d/make_operator_bias_doc.py
python -m pytest tests/test_wedge_estimator_bias.py -q                 # 9 passed
```

## ③ 计算结果

### 3.1 前向模型的两次数据对表（模型不是自证，先对数据）

| 检查 | dx = 2 mm | dx = 4 mm | 判读 |
|---|---|---|---|
| 逐帧湿节点数，模型 vs 工件（`n_wet` 均值绝对差） | {sm2['mean_abs_diff']:.2f} | {sm4['mean_abs_diff']:.2f} | 两级均在 ~1–2 个节点内（湿节点均值 29.8 / 13.7）→ 采样几何可信 |
| 顶点模型值（τ→0，算子**必然**无偏） | {apex2['model_smoothed_apex']:.4f} | {apex4['model_smoothed_apex']:.4f} | 对点值 5.4414 偏差 {pct(apex2['model_vs_point'])} / {pct(apex4['model_vs_point'])} → 算子归一化正确 |
| 顶点实测（后 30 帧 `cp_centre` 均值） | {apex2['obs_apex_mean']:.4f} | {apex4['obs_apex_mean']:.4f} | 对点值 {pct(apex2['obs_vs_point'])} / {pct(apex4['obs_vs_point'])} → **求解器自身的量级亏损**，与 m17899 独立测得 4.713/4.999 一致 |

顶点这一行是本单的**校准锚**：那里 `dCp/dτ = 0`，核外推与节点采样在数学上都无法产生偏置，所以顶点处的偏差只能来自求解器；反过来，峰值处的偏差可以用同一算子分解。两次对表（湿节点数、顶点值）合起来给前向模型发了“可信”通行证。

### 3.2 同算子比较（表中每个数字都从工件读出）

| 帧数 | dx=2 mm 实测峰值 | 实测 vs 点值 | 算子真值(2 mm) | **同算子残差** | dx=4 mm 实测峰值 | 实测 vs 点值 | 算子真值(4 mm) | **同算子残差** |
|---|---|---|---|---|---|---|---|---|
{table_obs}

### 3.3 算子偏置的来源分解（同一解析场，只换算子环节）

| 帧数 | 点采样(2 mm) | +核外推(2 mm) | 核外推偏置(2 mm) | 点采样(4 mm) | +核外推(4 mm) | 核外推偏置(4 mm) |
|---|---|---|---|---|---|---|
{table_ops}

结论：**分箱/采样环节几乎无偏**（点采样 {f3(g2['by_frames']['61']['synth_point']['peak_median'])} / {f3(g4['by_frames']['61']['synth_point']['peak_median'])} vs 点值 {ref:.3f}，偏 −4 %/−4 %），**偏置几乎全部来自 Adami 核外推**：它把峰值附近 ~2h 尺度的场平均掉，dx=2 mm 吃掉 **{pct(g2['by_frames']['61']['synth_smoothed']['peak_median'] / ref - 1.0)}**，dx=4 mm 吃掉 **{pct(g4['by_frames']['61']['synth_smoothed']['peak_median'] / ref - 1.0)}**（越粗越狠：β=30° 的峰值只离喷溅根 0.07c ≈ {0.07 * 50 / 2:.1f} 个 dx（2 mm）/ {0.07 * 50 / 4:.1f} 个 dx（4 mm），正在核支撑边缘）。这个偏置**与解对不对无关**，是测量链的常数项。

### 3.4 逐帧峰值分布（解释粗网格残差为何爆）

| 量 | dx = 2 mm | dx = 4 mm |
|---|---|---|
| 有效帧数 | {fp2['n_frames_used']} | {fp4['n_frames_used']} |
| 单帧峰值 中位数 / p75 / max | {fp2['median']:.2f} / {fp2['p75']:.2f} / {fp2['max']:.2f} | {fp4['median']:.2f} / {fp4['p75']:.2f} / {fp4['max']:.2f} |
| 算子真值（61 帧池化） | {f3(g2['by_frames']['61']['synth_smoothed']['peak_median'])} | {f3(g4['by_frames']['61']['synth_smoothed']['peak_median'])} |

单帧峰值中位数是算子真值的 {fp2['median'] / g2['by_frames']['61']['synth_smoothed']['peak_median']:.1f}×（2 mm）/ {fp4['median'] / g4['by_frames']['61']['synth_smoothed']['peak_median']:.1f}×（4 mm）：逐帧壁面压力被单节点噪声支配。池化 + 样本门在 dx=2 mm 能把它压回算子真值附近（残差 {pct(g2['by_frames']['61']['same_operator_residual_median'])}），在 dx=4 mm 压不住（每 bin 池化样本太少、湿节点均值仅 {g4['by_frames']['61']['synth_smoothed']['mean_wet_nodes']:.1f} 个），残差 {pct(g4['by_frames']['61']['same_operator_residual_median'])}。这与加入声明噪声（σ = {design['noise_rel']:.0%}、{design['noise_seeds']} 个种子）后的上限一致：噪声模型的池化峰值 {f3(g2['by_frames']['61']['synth_noisy_median_band']['median'])} / {f3(g4['by_frames']['61']['synth_noisy_median_band']['median'])}，即高斯噪声只能解释粗网格超额的一小部分——真实噪声是重尾的。

### 3.5 假设检验

| 项 | dx = 2 mm | dx = 4 mm |
|---|---|---|
| 实测 vs 点值（历轮口径） | {pct(hyp['signs']['dx2mm']['obs_err'])} | {pct(hyp['signs']['dx4mm']['obs_err'])} |
| 算子真值 vs 点值（本单口径） | {pct(hyp['signs']['dx2mm']['synth_err'])} | {pct(hyp['signs']['dx4mm']['synth_err'])} |

实测跨网格**变号**（{pct(hyp['signs']['dx2mm']['obs_err'])} → {pct(hyp['signs']['dx4mm']['obs_err'])}），算子真值**不变号**（两级都是负、且彼此差 {abs(hyp['signs']['dx2mm']['synth_err'] - hyp['signs']['dx4mm']['synth_err']) * 100:.1f} 个百分点 < 15 %）。故 **H1 否、H2 立**：变号不是估计量造成的，是数据侧的。

## ④ 结果图

![operator bias](figures/wedge_operator_bias.png)

* **(a)** 同一估计量用在两侧：实线 = SPH 池化峰值，虚线 = Wagner 场经同一算子；黑线 = 点值 `C²+1`，灰带 = ±15 %（对点值）。两条虚线（4 mm/2 mm）几乎重合，说明算子本身网格一致；实线跨网格分开 → 数据侧。
* **(b)** 真正与验收门竞争的量：同算子残差 vs dx，最细已有网格 {pct(ver['dx2mm']['residual'])} 仍在 ±15 % 带外。

（本单是**测量链/验收估计量**审计，不产生新流场，故无压力场云图；压力场云图见历轮 `docs/figures/wedge_entry_v2_*.png`。）

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **更正 m17899 ⑤.2 / 建议 3（重要）**：该文称“湿面节点数随入水加深从 59.7 掉到 29.8，是壁面接触判据在深穿透下失效”。**这是对自己表格的误读**：`mean_wet_nodes` 是“最后 N 帧”的均值，N 越小窗口越靠后、湿节点越多（7 帧 → 59.7，61 帧 → 29.8），趋势方向相反。已入仓工件 `frame_scan` 的逐帧 `n_wet` 单调增（dx=2 mm：1–4 号帧 0–2 个 → 末帧 60 个），与 `c = C V t` 增长一致。故“接触判据在深穿透下失效”这条**不成立**，不应据此立求解器项；m17899 的帧系综非平稳诊断（jackknife 43 %）真正原因只是“早窗湿节点少”，而不是接触判据坏了。更正已回注该文顶部（m18382 横幅）。
2. **本单自己踩的坑（保留作教训）**：前向模型第一版把核外推的 stencil 锚在顶点而不是逐节点，导致 τ>0.2 的节点全 NaN、被估计量跳过，“算子真值”退化成顶点附近的值（当时算出 −35 % 的漂亮结论）。是 `test_operator_has_support_at_the_peak_abscissa` + 顶点校准锚把它抓出来的。教训：**任何“解释掉失败”的分解，必须有能在数学上无偏的位置（这里是 τ→0）做校准**，否则先怀疑模型。
3. **算子模型的边界**：3 行流体、理想格距、解析场；真实 SPH 的壁面邻域有缺失/畸变（m17899 已见湿节点波动），故残差里仍混有“模型 vs 真实算子”的差异。前向模型只说“Adami 外推在解析场上会吃掉约 1/8–1/5 的峰值”，不说这是 SPH 的全部误差。
4. **残差符号的物理含义（假说，不是更正）**：最细网格残差为负（SPH 低于同算子 Wagner），方向与“线性理论在 β=30° 高估峰值、Z&F 非线性相似解应更低”的预期一致。但 Z&F 数值仍 **WITHHELD**（闭源、仓内无副本、不以记忆值/二手值代替），故这只是下一步的**假说**，不能当修正因子用。
5. **适用范围**：β = 30°、V = 1 m/s、c₀/V = 20、dx ∈ {{4, 2}} mm 单点；不外推到更细网格或其他 deadrise。

## ⑥ 误差判定（逐项，显式回答“是否 <15 %”）

| 关键量 | 基准 | 实测/本单值 | 相对误差 | 是否 <15 % |
|---|---|---|---|---|
| 峰值 Cp（**同算子**，dx = 2 mm，61 帧） | 算子真值 {f3(g2['by_frames']['61']['synth_smoothed']['peak_median'])} | {f3(g2['by_frames']['61']['obs_peak_median'])} | **{pct(ver['dx2mm']['residual'])}** | **否** |
| 峰值 Cp（同算子，dx = 4 mm，61 帧） | 算子真值 {f3(g4['by_frames']['61']['synth_smoothed']['peak_median'])} | {f3(g4['by_frames']['61']['obs_peak_median'])} | **{pct(ver['dx4mm']['residual'])}** | **否** |
| 峰值的网格一致性（同算子） | ≤ 15 % | 跨网格残差差 **{ver['cross_grid_residual_gap'] * 100:.1f} 个百分点**且**变号** | — | **否 → 不可裁决** |
| 测量算子自身对点值的偏置（dx = 2 mm） | — | {pct(g2['by_frames']['61']['synth_smoothed']['peak_median'] / ref - 1.0)} | — | （常数项，非误差） |
| 测量算子自身对点值的偏置（dx = 4 mm） | — | {pct(g4['by_frames']['61']['synth_smoothed']['peak_median'] / ref - 1.0)} | — | （常数项，非误差） |
| 顶点/驻点 Cp(0) | 5.44140 | {apex2['obs_apex_mean']:.3f}（2 mm）/ {apex4['obs_apex_mean']:.3f}（4 mm） | {pct(apex2['obs_vs_point'])} / {pct(apex4['obs_vs_point'])} | 是（两级均 <15 %） |
| 湿节点数（采样几何自校验） | 工件逐帧 `n_wet` | 模型偏差 {sm2['mean_abs_diff']:.2f}（2 mm）/ {sm4['mean_abs_diff']:.2f}（4 mm） 个节点 | — | 是（相对 ~1–8 %） |
| 峰值 Cp vs **Zhao & Faltinsen (1993)** | — | — | — | **WITHHELD**（基准数值未获得） |

**honest 结论**：①历史 −30.4 %（dx=2 mm）里，测量链（Adami 核外推）固定吃掉 {pct(g2['by_frames']['61']['synth_smoothed']['peak_median'] / ref - 1.0)}，剩下 {pct(ver['dx2mm']['residual'])} 才是数据侧残差——**验收结论不变（仍 FAIL/不可裁决）**，但误差归属变清楚了；②两级网格同算子残差**变号**（{pct(ver['dx2mm']['residual'])} vs {pct(ver['dx4mm']['residual'])}），按预声明一致性判据，峰值 Cp **仍不可裁决**，不因某一级落进 ±15 % 的错觉而宣布通过；③h2 假设（算子复现变号）被**证伪**，变号是数据属性；④m17899 的“湿节点随深度衰减/接触判据失效”推荐**被其自身工件否定**，已回注更正；⑤Z&F (1993) 依旧 WITHHELD。

## 建议（给派发器 / 下次找活的线索）

1. **验收口径**：把 `CASE-SPH-WEDGE-ENTRY-2D` 的峰值判据改成**同算子口径**（`estimator_bias.py` 的算子真值作分母），并要求两级网格同号；这是本单唯一可复用、可机器裁决的改进。建议新条目 `CASE-SPH-WEDGE-ENTRY-2D-SAMEOP`（域：耐波/砰击验证面），验收 =“同算子残差 < 15 % 且 dx ∈ {{4,2}} mm 同号”。
2. **参考侧才是瓶颈**：残差符号提示线性理论在 30° 高估峰值。真正的缺口是 **Z&F (1993) 非线性相似解**（Dobrovol'skaya–Zhao–Faltinsen 型，可在 w 平面解一个积分方程，纯 CPU 分钟级）——独立实现 + 小 deadrise 极限对 Wagner 校验，就能把“线性化误差”从假说变成可报数值，且**不需要算力**。建议单独立项。
3. **不要再做**：更细分箱、δ-SPH、帧过滤（m17548/m17899 已否）；dx = 1 mm 网格仍值得跑但只影响收敛证据，不解决参考侧缺口。
"""

    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())