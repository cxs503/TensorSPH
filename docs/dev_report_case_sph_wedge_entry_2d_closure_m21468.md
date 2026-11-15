# CASE-SPH-WEDGE-ENTRY-2D — dx = 0.5 mm 闭合轮（m21468）

**算例锚点**：`CASE-SPH-WEDGE-ENTRY-2D`（β = 30° deadrise，V = 1 m/s 恒速入水，自相似窗口，重力/表面张力关闭）
**指标**：楔形体表面峰值砰击压力系数，基准 = Zhao & Faltinsen (1993) 30° deadrise（**仍 WITHHELD**，见 §5）；
本仓可判定的同算子基准 = Wagner (1932) 解析锚（同一理想化，`tensorsph.wedge_entry` 现算）
**容差**：< 15 %（能量化到 `U_g ≤ 0.15` 与 `|r_g| ≤ 0.15` 两条）

## 1. 本单唯一欠账与结论

m21151 结束时唯一未闭合项是**阶梯最细一级 `dx = 0.5 mm`**（作业 cj3219 未取回终态），
因此峰值 Cp 判决一直是 `NOT_DECIDABLE`（只有 `dx = 1 mm` 一个可裁决网格）。本单递交
`cj3461`（8 核 / 2090 步 / 2362.2 s / 61 帧 / 7044 池化样本）取回该级并重算整条链。

**结论（honest, 未粉饰）**：加密到 0.5 mm **没有**让峰值变得可裁决，判决仍为 `NOT_DECIDABLE`，
并且现在有两个独立原因，二者都落在已入库工件里：

1. `U(0.5 mm) = 20.9 % > 15 %` → 不可裁决网格；
2. 即使可裁决，最细两级残差**变号**（`r(1 mm) = −12.4 %`、`r(0.5 mm) = +15.2 %`），
   按预声明规则仍禁判 PASS。

## 2. 阶梯全表（数字全部读自 `docs/artifacts/.../ladder_closure.json`）

| 网格 | obs（池化门控峰值 Cp） | 同算子真值 synth | **U = max(spread, jackknife)** | 残差 r | 点值残差 vs Wagner Cp_max | 可裁决 |
|---|---|---|---|---|---|---|
| dx = 4 mm | 10.041 | 6.870 | 23.5 %（spread 23.5 / jk 0.1） | +46.2 % | +23.3 % | 否 |
| dx = 2 mm | 6.071 | 7.494 | 43.0 %（spread 11.8 / jk **43.0**） | −19.0 % | −30.4 % | 否 |
| dx = 1 mm | 6.805 | 7.765 | 6.5 %（spread 3.2 / jk 6.5） | −12.4 % | −19.8 % | **是** |
| **dx = 0.5 mm** | **8.984** | 7.801 | **20.9 %**（spread **0.0** / jk **20.9**） | **+15.2 %** | −9.6 % | 否 |

`U` 随网格**非单调**（23.5 → 43.0 → 6.5 → 20.9 %），残差符号交替（+ − − +）。

## 3. 两个新发现（本单的实质增量）

### 3.1 「加密直到可裁决」路线被实测否定（m17548 route b）

`resolution_band.spread_rel` 在 0.5 mm 上恰好为 **0.0**，但这**不是**估计量变安静了，而是
**退化**：分辨率一致带 `n_bins_range = [29, 119]`，而候选 bin 数网格
`{6, 8, 10, 12, 16, 20, 30}` 只有一个（30）落在带内，所以"跨 bin 数的离散度"变成**单点离散度**。
真正的不确定度来自 jackknife（前后半帧 8.856 vs 7.325，20.9 %），即 61 帧系综里**峰值随帧不平稳**。

若判决器只用 bin-count spread，就会把 0.5 mm 判成"可裁决，spread = 0"，把 21 % 的帧间漂移
伪装成 PASS。现规则 `U = max(spread, jackknife)` 从构造上挡住了这一点，本单新增测试
`test_degenerate_spread_is_not_evidence_at_the_fine_rung` 把它钉住（并对 `n_bins_used` 长度做守卫）。

推论（写入建议）：`dx ≤ 0.5 mm` 时 bin 候选网格必须加密（例如 30…120 步进 2），
否则 spread 统计量在细网格上恒为 0、失效。这是**仪器**缺陷，不是求解器缺陷。

### 3.2 点值巧合：0.5 mm 的表观好一致性是误差抵消

`obs(0.5 mm) = 8.984` vs Wagner 点值 8.4022 只差 −9.6 %（落在 15 % 内），
但同一算子真值是 7.801（估计量在 0.5 mm 上系统性低估 7.2 %），同算子残差因此是 +15.2 %。
**报"vs Wagner 点值 −9.6 %"会是选择性指标**；本单按预声明只报同算子残差。

### 3.3 已发表表格行的去留（`table_erratum.py` 重算）

`docs/verification_wedge_entry.md` Table 3.2 的 `dx = 0.5 mm` 行（table Cp = 8.49）
此前因"无该网格工件"而不可判，现在**可复现（gap 5.8 %，reproduced）**；
但 `dx = 2 mm`（9.57?→ 实际 closest 28.9 %）与 `dx = 1 mm`（28.9 %）两行仍 **unreproducible**，
故 `summary.verdict.usable_as_acceptance = false` 不变——Table 3.2 依旧不能当验收证据。

## 4. 误差判定（逐项，显式回答「是否 < 3 %」）

| 关键量 | 值 | 基准 | 相对误差 | < 3 % | < 15 % |
|---|---|---|---|---|---|
| Wagner `Cp_max`（模块 vs 独立闭式 `C²+1`） | 8.402203300817021 | 闭式解析 | ≈1e-16 | ✅ | ✅ |
| 峰值 Cp，同算子残差 @ dx = 0.5 mm | +15.2 % | Wagner 场过同一算子 | 15.2 % | ❌ | ❌ |
| 峰值 Cp，同算子残差 @ dx = 1 mm | −12.4 % | 同上 | 12.4 % | ❌ | ✅ |
| 峰值 Cp 判决 | **NOT_DECIDABLE** | 需 ≥2 可裁决同号网格 | — | ❌ | 未裁决 |
| Z&F 1993 表值 @ β = 30° | 空 | Zhao & Faltinsen 1993 | 无法判定 | — | **WITHHELD** |

**是否 < 3 %**：解析锚达到机器精度；**求解器峰值 Cp 不满足 3 %，也不满足 15 %（0.5 mm 上 15.2 %，越界）**；
Z&F 基准无法判定（闭源、生态内无正本，本单不填任何记忆值/二手值）。

## 5. 为什么 Z&F 一格仍然是空的（本单的探针尝试也失败）

本单额外尝试用开放全文通道去取 30° 相似解数值（OpenAlex `cites:W2051783622, is_oa:true`
返回 242 条引用文献；arXiv 全文抓取探针），**在 120 s 预算内无任何文本抽取结果返回**，
故未能取得带页码坐标的二手数值。按 `refs.json` 的纪律，`table_or_figure_value_beta_30`
保持 `null`，状态继续 `WITHHELD`。闭环动作照旧：把正本投递进仓库后填数，无需改求解器。

## 6. 复现命令（全部本地、无重型进程）

```bash
# 1) 最细一级（本单已跑，集群 cj3461，8 核 2362 s）
python benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py --dx 0.0005 --frames 61 \
    --threads 8 --out docs/artifacts/case_sph_wedge_entry_2d_ensemble
# 2) 偏置真值 + 阶梯判决 + 表格 erratum（本地各 <30 s）
python benchmarks/case_sph_wedge_entry_2d/estimator_bias.py
python benchmarks/case_sph_wedge_entry_2d/ladder_closure.py
python benchmarks/case_sph_wedge_entry_2d/table_erratum.py
# 3) 文档与图（数字自动注入）
python benchmarks/case_sph_wedge_entry_2d/make_ladder_figure.py \
    --out docs/figures/wedge_ladder_closure_m21468.png
python benchmarks/case_sph_wedge_entry_2d/make_ladder_doc.py \
    --out docs/case_sph_wedge_entry_2d_closure_m21468.md
# 4) 验收
python -m pytest tests/test_case_sph_wedge_entry_2d_closure_m21468.py \
    tests/test_case_sph_wedge_entry_2d_acceptance.py \
    tests/test_case_sph_wedge_entry_2d.py -q
```

## 7. 注意点 / 局限

1. 本单**没有**把算例判成 PASS；判决仍是 `NOT_DECIDABLE`，且现在有两处独立证据支持该判决。
2. `judgeable` 的定义（`U ≤ 15 %`）与 `≥2 个同号可裁决网格` 都是**预声明**的，本单未放宽。
3. 0.5 mm 单例 2362 s 是在 8 线程、61 帧、c0 = 20V 下测的；再细一级（0.25 mm）按 8× 外推约 5.2 h，
   且 bin 网格必须先加密（否则 spread 恒 0），因此不建议在修好仪器前继续加密。
4. 帧间不平稳（jackknife 20.9 %）的物理解释**未**在本单定量归因（候选：早期声学暂态、
   湿节点数随帧增长导致的 bin 样本量差异）。这是下一步该做的定量工作，不是可以顺手断言的事。

## 8. 给下次自找活的建议线索

* 建议新条目 `slug = CASE-SPH-WEDGE-ENTRY-2D-PEAK-ESTIMATOR`，
  标题「算例池·二维楔形体入模砰击峰值压力的估计量替换（固定 τ 剖面拟合 vs 分箱峰值）」，
  量化验收：「在 dx ∈ {2, 1, 0.5} mm 三级上，新估计量给出的帧间 jackknife 相对离散 < 15 %
  且三级残差同号；否则如实记为 FAIL」，所属域 `TensorSPH/slamming`。
  理由：本单已证明限制项是**测量算子**而非网格（U 非单调、0.5 mm 上 bin 带退化），
  继续加密的性价比已经被实测否掉。
* 次要项：`ladder_closure.py` 的 `bin_grid` 应从设计文件读入而非内置，
  并支持在 `dx ≤ 0.5 mm` 时自动细化候选网格（本单以测试记录了退化现象，未改脚本）。