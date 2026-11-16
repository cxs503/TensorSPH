# CASE-SPH-WEDGE-ENTRY-2D — 实现轴真的独立吗？设计效应把「N=6 ⇒ 可判」的登记预测推翻

**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力；ref: Zhao & Faltinsen 1993 JFM 246:593, DOI 10.1017/s002211209300028x；容差 `< 15 %`）
**本单**: m23165（generalist 席自找活）　**仓库**: TensorSPH　**类型**: 验证欠账审计

本文件由 `benchmarks/case_sph_wedge_entry_2d/make_design_effect_doc.py` 从
`docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json` 渲染；工件由 `seed_ensemble.py` 生成（只读已提交的 17 份实现工件，**本轮无新 SPH 计算**）。

## ① 案例与问题

β = 30° 对称刚体楔以恒速 `V = 1 m/s` 入静水，重力/表面张力关，自相似（唯一长度尺度 `Vt`）。受理指标是湿面峰值砰击压力系数 `Cp_max = p_max/q`。

m22826 给本算例装上了**独立实现轴**（`WedgeConfig.phase_x_dx` / `gap_jitter_dx`，亚网格离散相位），并登记了一条可直接证伪的预测：

> 「按 `U ∝ N^{-1/2}`，N = 6 个实现给出 `U_ci ≈ 6.8 % < 15 %`，该网格转为可判。」

这条预测有一个**未检验的前提**：6 个实现是 6 个**独立**样本。m22826 自己把检验写成了辅助问题 A1（`cp_centre` 逐帧序列的两两 Pearson 相关），但只报了 ρ，没有把它接回 `U`。本单做这件事：把测到的 ρ 变成**有效样本数**，再用它重算不确定度。这是纯分析，没有新增判据、没有放宽门槛、没有动求解器。

## ② 怎么做的

1. **A1 的输出接回规则**。同一网格 N 个实现的逐帧 `cp_centre` 序列给出两两 Pearson ρ；取中位数 ρ̄，按整群抽样的标准设计效应：

   $$\mathrm{deff}=1+(N-1)\,\bar\rho,\qquad N_{\rm eff}=N/\mathrm{deff},$$

   再用 `U_ci = t_{0.95}(N_eff-1)\,s/\sqrt{N_eff}` 重算，最后与带宽散布取大：
   `U_corrected = max(U_ci(N_eff), median band spread)`。ρ̄ ≤ 0 时 deff = 1，即退回 m22826 的原始口径（这一点写成负控测试）。
2. **预声明口径原样保留**。`U_realisation_ci` / `U_band` / `judgeable` 三个字段一个字没改，新的量只加在 `design_effect` 子字典里，历史工件与旧判决仍逐位可复现；`ladder_closure.verdict` 也仍然吃原始的 `rule_inputs`。
3. **两个分母并列**。闭环规则判的是「相对**同算子** Wagner 的残差」，而总账受理判的是「相对 **Z&F 1993 复合解（COM）** 的误差」——本算例上两者相差 21.4 %。因此每个网格同时报 `residual_vs_com`（受理口径）与 `residual_vs_wagner`（诊断口径），**不合并成一个数**，也不拿一个口径的 PASS 冒充另一个。
4. **dx = 1 mm / 2 mm 的实现集**已递交集群（`cj4654` 6 个实现 × dx=1 mm、`cj4655` 补 dx=2 mm 的 seed 4–6）。写作时二者仍在排队，故细网格**如实记为 `WITHHELD`**，本文件只对已取回的网格下结论。

## ③ 计算结果（程序化注入）

### 3.1 设计效应主表

| grid | N | ρ̄ (A1) | deff | N_eff | sd(r) | U_naive (独立假设) | U_corr (A: √deff 加宽) | U_corr (B: N_eff 重算) | judgeable naive → A → B |
|---|---|---|---|---|---|---|---|---|---|
| dx4mm | 8 | 0.815 | 6.70 | 1.19 | 0.0628 | 5.3 % | 13.6 % | 73.1 % | yes → yes → no |
| dx2mm | 3 | 0.651 | 2.30 | 1.30 | 0.1515 | 37.6 % | 57.1 % | 168.5 % | no → no → no |

`dx = 4 mm`、N = 8：ρ̄ = 0.815，deff = 6.70，N_eff = 1.19。也就是说 **8 个实现的实际信息量约等于 1 个独立样本**；`U_naive` = 5.3 % 与 `U_corrected` = 73.1 % 相差 13.9 倍。

### 3.2 两个分母（受理口径 vs 诊断口径）

| grid | obs (q units) | vs COM (ledger) | vs Wagner (same operator) | in ±15 % vs COM? |
|---|---|---|---|---|
| dx4mm | 8.113 | +17.2 % | −3.4 % | no |
| dx2mm | 8.543 | +23.5 % | +1.7 % | no |

### 3.3 逐实现读数（nominal scale = 1.0）

| grid | seed | phase_x/dx | gap_jitter/dx | obs peak | band spread |
|---|---|---|---|---|---|
| dx4mm | 1 | -0.1828 | +0.1737 | 8.680 | 24.4 % |
| dx4mm | 2 | +0.2280 | +0.2239 | 8.676 | 11.9 % |
| dx4mm | 3 | -0.1310 | +0.0221 | 7.509 | 12.3 % |
| dx4mm | 4 | -0.1320 | -0.1984 | 8.107 | 8.3 % |
| dx4mm | 5 | +0.0615 | +0.1209 | 8.207 | 21.4 % |
| dx4mm | 6 | +0.1467 | +0.1610 | 8.120 | 8.3 % |
| dx4mm | 7 | -0.0881 | -0.1746 | 7.664 | 10.3 % |
| dx4mm | 8 | -0.1366 | +0.2311 | 7.808 | 7.9 % |
| dx2mm | 1 | -0.1828 | +0.1737 | 7.796 | 14.5 % |
| dx2mm | 2 | +0.2280 | +0.2239 | 8.543 | 12.6 % |
| dx2mm | 3 | -0.1310 | +0.0221 | 10.026 | 33.3 % |

### 3.4 扰动幅度对照（A2，`scale ∈ {0, 0.25, 1}`）

* `dx4mm`：scale=0，n=1，obs 中位 10.041，残差中位 +46.2 %；scale=0.25，n=5，obs 中位 8.093，残差中位 +17.8 %；scale=1，n=8，obs 中位 8.113，残差中位 +18.1 %。
* `dx2mm`：scale=1，n=3，obs 中位 8.543，残差中位 +14.0 %。

`scale = 0` 与已提交工件逐位一致（m22826 的负控），sanity 通过。

## ④ 图

![design effect](docs/figures/case_sph_wedge_entry_2d_design_effect_m23165.png)

(a) 逐实现读数对两个分母：实现的散点宽度就是「换离散相位引起的变化」，它与两个分母之间的距离（21.4 %）同量级 —— 这是本算例容差紧的根源；
(b) 实测 ρ → `N_eff`：点落在 N_eff ≈ 1 处，虚线是「N = 6 时靠加实现能买到多少」，在 ρ̄ ≈ 0.8 时几乎买不到东西。

## ⑤ 注意点与局限（不粉饰）

1. **本单不产生新的 SPH 读写数**，也不改变任何网格的 obs。它审的是「不确定度」这一层。
2. **ρ̄ 是每帧序列的相关，不是残差的相关**。用逐帧 `cp_centre` 是因为它对整条轨迹的共同信号敏感，但它给的是 deff 的**上界侧**估计（比残差相关的估计更保守）。本单据此加宽 U，方向是安全的（不会把不可判判成可判）。
3. **N_eff ≈ 1.2 时 t 分布自由度只有 1**，`U_corrected` 因此到 70 % 量级。这不是数值病态，而是「几乎没有独立样本」的如实表达；把它读成「机器坏了」是误读。
4. **细网格仍是 `WITHHELD`**：`cj4654`（dx=1 mm × 6）与 `cj4655`（dx=2 mm seed 4–6）写作时在队列里（提交时刻 11-16 08:01，位置 ≈117）。若它们出队，同一脚本 `--analyse` 一行即可重算，无需改代码。
5. **分母歧义没有消解**：COM 仍是二手引用（原文闭源），这在 `docs/case_sph_wedge_entry_2d_literature_regrade.md` ⑤-1 已登记，本单不重复。

## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）

| 量 | 基准 | 结果 | 偏差 | < 3 % ? |
|---|---|---|---|---|
| 旧判决复现（`U_realisation_ci` / `U_band` / `judgeable` 三个字段） | m22826 工件 | 三字段逐位未变 | 0 % | **是** |
| `U_corrected`（4 mm） | `U_naive` 5.3 % | 73.1 % | ×13.9 | **否**（这是本单的结论，不是误差） |
| `U_naive` 独立假设下的 N=6 外推 | 0.166/√6 = 6.8 % | 实测 ρ̄ 下 73.1 % | 推翻 | **否** |
| `residual_vs_com` 两个分母的比值 | refs.json COM/Wagner | 见 3.2 | 0 % | **是** |

**回答「是否 < 3 %」：可复现性层面是（旧字段 0 % 改动）；判定量层面不是** —— 本单的关键量 `U_corrected` 与登记预测相差一个数量级。`honest-FAIL`：锚点维持 `open`，判决维持 `NOT_DECIDABLE`，本单**没有**把任何网格推进到可判。

## ⑦ 结论与给下一单的可证伪登记

1. **登记预测被推翻（前提，不是结论）**：m22826 §⑦-2 的「N = 6 ⇒ `U_ci ≈ 6.8 % < 15 %`」以「实现相互独立」为前提；实测 ρ̄ = 0.815（dx = 4 mm，N = 8）使 `N_eff ≈ 1.19`，因此**加实现数不能把 U 降到门槛内**。这条路要成立，必须先让实现去相关。
2. **可证伪的替代路线**：把扰动从「亚网格相位」（振幅 ≤ 0.25 dx、只挪格点原点）换成会真正改初始场的量（例如 `alpha_visc`/`c0`/入水角的**预声明小扰动**，或两相 seeding 的随机位置抖动），并测同一 A1 ρ̄。预测：ρ̄ 应从 ≈0.8 降到 ≤0.3，此时 N = 6 才对得上 `U ≈ 6.8 %`。若 ρ̄ 仍 > 0.7，则「实现轴」整条路线被推翻，该锚点只能靠更细网格（`dx ≈ 0.35–0.38 mm`）或换指标。
3. **`dx = 4 mm` 的 `judgeable=True` 应降级**：它在独立性假设下才成立；按本单口径 `judgeable_corrected = no`。历史 `literature_regrade` 的单网格 PASS 建立在**无扰动对称格点**的一个成员上，而 4 mm 上「无扰动 10.04 vs 相位化 8.11（中位）」差 −19 %，说明对称格点系统性偏高 —— 细网格是否复现同一模式，正是 `cj4654` 出队后第一个要回答的问题。

## ⑧ 复现

```bash
# 聚合（只读已有实现工件，秒级）
python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py --analyse \
    --out docs/artifacts/case_sph_wedge_entry_2d_realisations \
    --bias docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json
# 本文档与图
python benchmarks/case_sph_wedge_entry_2d/make_design_effect_doc.py
# 契约测试
python -m pytest tests/test_wedge_design_effect.py -q
```
