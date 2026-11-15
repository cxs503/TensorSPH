# CASE-SPH-WEDGE-ENTRY-2D — 参考侧升级与再判定（文献锚，m21430）

**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力，`feature_ledger` 状态 `open`）
**本单类型**: 验证欠账（把已合并算例的**分母**对开放文献复核）＋ 前轮审计结论的推翻
**日期**: 2026-11-15　**工件**: `literature_refs.py` / `literature_regrade.py` /
`docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json` / 3 图 / 17 项测试
**本轮无新 SPH 计算**（本机 load15 ≈ 63–71，按容量纪律只做轻量分析）。

## ① 为什么这是增量而不是重做

前六轮（m17548/m17899/m18382/m20161/m21005/m21151/m17079）把 SPH 侧做到了
「粗网格不可裁决、`dx=1 mm` 可裁决」，判决停在 `NOT_DECIDABLE`，
并且唯一非 `WITHHELD` 的锚只有一个 Wagner 渐近点值 `C²+1 = 8.402`（q 单位）。

本轮不动求解器，动的是**分母**：该算例的验收对象是「峰值砰击压力系数 vs
Zhao & Faltinsen 1993（30° deadrise）」，而 Z&F 一直是 `WITHHELD`（闭源）。
在开放文献里找到了一张同行评议表（Jain, Novaković, Bogaert & van der Meer 2022,
*JFM* **934** A26, DOI 10.1017/jfm.2021.1129，开放版 arXiv:2011.10378v3, **Table 1**），
它同时列出 β=30° 的 **Z&F 复合解（COM）**、**Dobrovol'skaya 精确相似解（EIM）**、
GWM/MLM/OLM/ASM 六个模型的 $C_{p,max}$。于是：

* 判据里的「Z&F 1993 数值」从 `WITHHELD` 变成**带坐标的可核验值**（仍标注为二手，
  见 `refs.json` 的 `beta30_via_open_source` 字段与 `withheld_reason` 不变）；
* 顺带发现：仓内唯一的锚 `C²+1` 属于**奇异族**（Wagner 型剖面在喷溅根部发散），
  它在 30° 上比 Z&F 复合解高 **+21.42 %** ——
  单这一项就吃掉了 15 % 容差的一倍半。

## ② 参考集（β = 30°，本表全部由 `refs.json` 解析，单位换算单点负责）

原文定义 $C_{p,max} \equiv P_{max}/(\rho V^2)$，本算例定义 $C_p = p/q$（$q=\rho V^2/2$），
故**表值 ×2** 才是本算例的 $C_p$（`literature_refs.cp_q_units`；乘错一次就是 100 % 误差，
m17079 审计的 A2 缺陷正是这一类）。

| 模型 | 出处 | $C_p$（q 单位） | $P_{max}/\rho V^2$ | 本算例设计判据 |
|---|---|---|---|---|
| OLM | Logvinovich 1969 (ordinary) | 3.700 | 1.850 |  |
| MLM | modified Logvinovich (Korobkin & Malenica 2005) | 9.620 | 4.810 |  |
| GWM | generalised Wagner (Korobkin 2004) | 8.480 | 4.240 |  |
| EIM | exact similarity, Dobrovol'skaya 1969 / Wang & Faltinsen 2017 | 6.900 | 3.450 |  |
| ASM | asymptotic, Wang & Faltinsen 2017 | 7.400 | 3.700 |  |
| COM | composite, Zhao & Faltinsen 1993 (mission-designated) | 6.920 | 3.460 | ** |

模型形散度（相对 COM）±42.8 %，
即**参考族自身的不确定度已超过 15 % 容差**：任何残差落在族内都不构成求解器证据。
EIM 与 COM 相差仅 0.3 %，二者互为旁证；
GWM/MLM 与仓内锚同族（仓内锚 vs GWM 差 0.9 %）。

## ③ 再判定（预声明规则，见 `literature_regrade.py` 模块文档）

门槛 $G=0.15$；网格「可裁决」= 估计量不确定度 ≤ G；参考 = **Z&F 1993 复合解 6.92（q 单位）**。

| grid | $C_p$ measured | estimator uncertainty | residual vs reference | judgeable | verdict |
|---|---|---|---|---|---|
| dx4mm | 10.041 | 23.5 % | +45.11 % | no | UNSCORED |
| dx2mm | 6.071 | 43.0 % | -12.27 % | no | UNSCORED |
| dx1mm | 6.805 | 6.5 % | -1.66 % | yes | PASS |

判决 **PARTIAL_SINGLE_GRID** —— 1 judgeable grid(s) inside the gate; closure rule asks for >= 2 with the same sign。

同一批数据、同一规则，换成精确相似解（EIM = 6.90）作为参考：

| grid | $C_p$ measured | estimator uncertainty | residual vs reference | judgeable | verdict |
|---|---|---|---|---|---|
| dx4mm | 10.041 | 23.5 % | +45.53 % | no | UNSCORED |
| dx2mm | 6.071 | 43.0 % | -12.02 % | no | UNSCORED |
| dx1mm | 6.805 | 6.5 % | -1.38 % | yes | PASS |

判决 **PARTIAL_SINGLE_GRID**（差别 < 0.3 %，两参考互为旁证）。

**旧记录的诚实并列**（`ladder_closure.json` 原样保留在 `regrade.json` 的 `ladder_reported_by_previous_rule`）：

| grid | $C_p$ | 对仓内 Wagner 渐近 8.402 | 不确定度 | 可裁决 |
|---|---|---|---|---|
| dx4mm | 10.041 | +23.30 % | 23.5 % | no |
| dx2mm | 6.071 | -30.37 % | 43.0 % | no |
| dx1mm | 6.805 | -19.76 % | 6.5 % | yes |

旧判决 `NOT_DECIDABLE`（原因：`only 1 judgeable grid(s) (estimator spread > 15%)`）。
**结论不是「旧判决错了」，而是它的分母不是判据指定的那个**：换成指定参考后，
最细网格 `dx=1 mm` 的残差从 ≈−19.8 % 变到 -1.66 %（|·| < 15 %）。

## ④ 前轮审计 A1 结论被推翻（m17079）

`docs/audit_wagner_pressure_reference_m17079.md` 断言仓内剖面
`wagner_p_over_q = 2Cs − s² + 1` 比「欧拉 Bernoulli」高一个动压，
并据此给出「参考侧本身偏 +13.5 %」。本轮用原文 **eq. (10)** 复核：

$$p(x,0)=\rho V^2\left[\frac{\pi}{2\tan\beta}\frac{c}{\sqrt{c^2-x^2}}-\frac12\frac{x^2}{c^2-x^2}\right]
\;\Longleftrightarrow\; \frac{p}{q}=2Cs-s^2+1$$

程序化对比（`regrade.json: reference_side_audit`，τ ∈ [0, 0.97] 共 4001 点）：

| 量 | 值 |
|---|---|
| max abs(仓内剖面 − eq.10) | **7.105e-15**（q 单位） |
| max 相对偏差 | 1.739e-15 |
| mean(仓内 − eq.10) | +1.954e-17 |
| mean(m17079「欧拉」变体 − eq.10) | -1.000（恰好 −1 q） |

即：**仓内剖面就是文献剖面**；A1 的「欧拉」列多扣了 $\tfrac12\rho V^2$ 这一项
（等价于把均匀流项在 Bernoulli 常数与 $|\nabla\phi|^2$ 里重复计一次），
是**审计侧的规范双计**，不是被审对象的偏差。A2（`cp_centre` 的单位 ×2 修正）**成立且应保留**。
A1 的连带结论「SPH 峰值 +233 % 里含 13.5 % 的参考偏差」因此不成立——
偏差是 0，SPH 的峰值残差全部来自数据侧。

## ⑤ 注意点与局限（不粉饰）

1. **参考仍非原文**：Z&F 1993 全文在生态内不可读，`table_or_figure_value_beta_30` 保持 `null`，
   `status` 保持 `WITHHELD`；本轮用的是同行评议论文中**复述**的 Z&F 复合解值 + 精确相似解值。
   二者相差 0.3 % 才使其可信，但这不等于读过原文。
2. **判决仍是单网格**：`dx=2 mm` 的点残差虽在门槛内（-12.27 %），
   但其 jackknife 不确定度 43 % 使该网格不可裁决——按预声明规则**不记分**。
   闭合仍需 `dx=0.5 mm`（算力）或把估计量带宽压到 15 % 以下。
3. **估计量向下偏**：本估计量（`resolution_band.peak_median`）对奇异峰有平滑偏差
   （同算子作用在解析场上 `dx=1 mm` 时低估 7.6 %），所以「对点值 −1.7 %」里含偏差抵消，
   不能读成求解器精度。<15 % 的两个独立口径（对指定参考 −1.7 %；同算子残差 −12.4 %）一致地通过。
4. **模型形散度 42.8 %**（相对 COM）：
   族内换参考即可翻判决，说明该算例的 15 % 容差在物理上是**紧的**；
   再判定时应同时报参考族，不得只挑一个模型（两个可相互旁证的参考 COM/EIM 差 0.3 %，可作首选）。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 残差 / 偏差 | 是否 < 3 % |
|---|---|---|
| 仓内 Wagner 剖面 vs 文献 eq. (10) | 7.1e-15 q（≈1e-15 相对） | **是**（机器精度，< 3 % 且 < 1e-12） |
| Z&F 复合解 vs 精确相似解 (EIM) | 0.29 % | **是** |
| 仓内渐近锚 vs GWM | 0.92 % | **是** |
| 仓内渐近锚 vs Z&F 复合解（族差） | +21.42 % | **否**（这是族差，不是误差） |
| SPH `dx=1 mm` vs Z&F 复合解 | -1.66 % | 否（但 < 15 % 门槛，PASS） |

## ⑦ 复现

```bash
PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/literature_regrade.py   # 写 regrade.json + 3 图
python benchmarks/case_sph_wedge_entry_2d/make_literature_regrade_doc.py         # 写本文档（数字注入）
python -m pytest tests/test_case_sph_wedge_entry_2d_literature_regrade.py -q     # 17 passed
```

图：`docs/figures/case_sph_wedge_entry_2d_regrade_ladder.png`（阶梯 vs 文献带）、
`..._boundary.png`（$C_{p,max}$ vs β 六模型 + 仓内锚）、
`..._profile.png`（剖面恒等 + m17079 变体的 −1 q 偏移）。
