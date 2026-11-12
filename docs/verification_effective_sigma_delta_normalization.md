# 有效表面张力的缺失因子定位：δ_s 归一化，不是曲率（TensorSPH）

**仓库**: TensorSPH ｜ **工件**: `benchmarks/effective_sigma/` ｜ **测试**: `tests/test_effective_sigma_delta_normalization.py`
**接的单**: `docs/verification_effective_sigma.md` 末尾的「后续（P0 候选）：CSF 力装配需要一次积分一致性校准……属 `src/` 改动，需单独派单」。
**复现**（静态力评估，无时间积分，总计 < 2 s / 例）：

```bash
python benchmarks/effective_sigma/decompose.py            # 表 1 的全部数字
python benchmarks/effective_sigma/make_figure_normalization.py
python -m pytest tests/test_effective_sigma_delta_normalization.py -q
```

## 1. 欠账是什么

圆滴上 CSF 面力 `f = -σ κ δ_s n`（κ = 1/R，n 外法向），右半部积分（单宽）

    F_x = σ ∫ (1/R) n_x dA = σ ∫_{-π/2}^{π/2} cos t dt = 2σ ,

故 `|F_right|/(2σ) → 1`（h→0 一致离散）。m3188/m2979 的实测是 0.71（`fit_quadratic`）/ 0.79（`mls`），且**不随细化收敛**——`fit_quadratic` 反而远离（0.7106 → 0.6818 → 0.6500）。前一份记录只给出了「系统性偏差」的判定并把候选修法列为两条：δ_s 归一化，或 κ·δ_s 乘子标定。本单把它**定量分解**到三项上，判定是哪一项。

## 2. 三项分解（全部是可闭式核对的积分）

| 量 | 解析目标 | 含义 |
|---|---|---|
| `∫δ_s dV` | 界面长度 `2πR`（单宽） | δ_s 是否把界面「量」够了 |
| `∫κ δ_s dV` | `κ·A_s = 2π`（κ=1/R） | 上者 × 曲率 |
| `\|∫κ δ_s n dV\|` | `2` | 方向/矢量和（=力比 × 2） |

`delta_s` 的 `kappa_iface_mean` 在 `fit_quadratic` 下是 36.46 vs `1/R = 25`（**+46 %**），即曲率是**偏高**的；而力偏低 29 %。两个偏差方向相反，说明缺失因子在 δ_s 一侧，不在曲率一侧。

## 3. 结果（`decompose.py` 程序化输出，零手抄）

表 1 `legacy`（现默认，δ_s = 2|∇c|）与 `local_range`（δ_s = |∇c|/(c_bulk−c_void)）对比：

| 曲率算子 | dx [mm] | ∫δ_s dV / 2πR | ∫κδ_s dV / 2π | \|F_right\|/(2σ) |
|---|---|---|---|---|
| fit_quadratic | 5.00 | 0.6677 | 0.7012 | **0.7106** |
| fit_quadratic | 2.50 | 0.6844 | 0.6734 | **0.6818** |
| mls | 5.00 | 0.6677 | 0.7861 | **0.7907** |
| mls | 2.50 | 0.6844 | 0.7647 | **0.7678** |
| fit_quadratic | 5.00 | 0.9136 | 0.9594 | **0.9723** |
| fit_quadratic | 2.50 | 0.9363 | 0.9213 | **0.9328** |
| mls | 5.00 | 0.9208 | 1.0902 | **1.0959** |
| mls | 2.50 | 0.9948 | 1.1202 | **1.1170** |

（后四行 = `local_range`。δ_s 均值：`legacy` 45.36 / 153.85 h⁻¹ = 0.295/h；`local_range` 62.06 = 0.403/h；dx 减半时两者同比例变化，说明这是**归一化常数**问题而非分辨率问题。）

![delta normalization](../benchmarks/effective_sigma/effective_sigma_delta_normalization.png)

## 4. 判定

1. **主导缺陷 = δ_s 归一化的单相特例，不是曲率、不是分辨率。** `legacy` 的 `∫δ_s dV` 只到界面长度的 **66.8 %**，力比 0.711 与它同量级；`local_range` 把该积分抬到 **91.4 %**，力比随之到 **0.972**（同一曲率算子、同一网格、同一邻域）。换归一化使力 +37 %，换曲率算子只改 0.71↔0.79。
2. **`2|∇c|` 的推导少了一半。** 模块注释用「流体内 `∫|∇c| dn = 1−1/2 = 1/2`」论证乘 2 即够；实测 `c_iface_mean = 0.907`（界面带上粒子的颜色均值），即真正的 c 落差只有 ~0.09–0.5 的一部分被颗粒承载，`Δc` 实际取多少要**按支撑域实测**（`colour_range` 已在 m2188 实现），而不是把 1/2 写死。写成 hard-code 的 2 恰好给不出正确值——这正是 0.67 vs 1 的来源。
3. **`local_range` 的残差与曲率偏差同向可解释。** `fit_quadratic` + `local_range`：力的残差 −2.8 %/−6.7 %，`∫κδ_s dV` 为 0.96/0.92（曲率偏高 +46 % 与 δ_s 形状共同决定），量级一致；`mls` + `local_range` 则过冲到 +9.6 %/+11.7 %（`mls` 曲率本身 +60…+96 %）。两者都在 **15 %** 平台门槛内，`fit_quadratic+local_range` **< 3 %（5 mm 档 −2.8 %）**，2.5 mm 档 −6.7 % 未达 3 %。
4. **不改变默认值。** 默认仍是 `legacy`：`Config.csf_delta_normalization = "legacy"`，所有已发布的 Laplace/Young–Laplace 数字（0.71 钉、closure ladder 269 Pa 等）**保持原样**，避免「修一个门、崩一串门」。改进以**可选择项**落地（`Config.csf_delta_normalization = "local_range"`），由各算例显式切换并各自重跑验收——这是本单的边界，跨算例重跑未做。

## 5. 注意点 / 局限

- 本单是**静态力评估**（无时间积分），验证的是「CSF 算子把解析力算对了没有」，不是「Laplace 压差读数对了没有」。后者还要过自由面压强闭合那道账（`laplace_report` 仍 FAIL：+27 %）。
- `∫κδ_s dV` 的目标 `2π` 与 `docs/verification_effective_sigma.md` 里 `kappa_delta_s_target = πR` 的量纲不一致（`πR` 是长度、该积分无量纲）；本单取 `κA_s = 2π`。原 diagnostic 的 target 属笔误，未改其历史报告。
- `local_range` 的 `colour_range` 在**单相**云上用支撑域实测 c 的极差，数值上依赖邻域半径与 h/dx；2.5 mm 档 0.9363 < 5 mm 档 0.9136… 之间非单调，趋势未在更细网格上封口（3.2k 粒子档 mls 已达 0.995）。
- 未做的事：把 `local_range` 设为默认并重跑 laplace/Young–Laplace/表面张力阶梯全套（需单独派单，成本是那几档算例的重跑）；`fit_quadratic` 下 2.5 mm 档 −6.7 % 的残差来源（δ_s 形状 vs 曲率 +46 % 的相对贡献）未拆分。

## 6. 误差判定

| 关键量 | 基准 | legacy | local_range | 是否 <3 % |
|---|---|---|---|---|
| `\|F_right\|/(2σ)`，dx=5 mm，`fit_quadratic` | 1.0 | 0.7106（−28.9 %） | **0.9723（−2.8 %）** | legacy 否 / local_range **是** |
| 同上，dx=2.5 mm | 1.0 | 0.6818（−31.8 %） | 0.9328（−6.7 %） | 两者皆否 |
| `∫δ_s dV/(2πR)`，dx=5 mm | 1.0 | 0.6677（−33.2 %） | 0.9136（−8.6 %） | 两者皆否 |
| `\|F_right\|/(2σ)`，dx=5 mm，`mls` | 1.0 | 0.7907（−20.9 %） | 1.0959（+9.6 %） | 两者皆否 |

结论：**归一化是主导缺陷已确证**（同一曲率、同一网格下 0.71 → 0.97）；`local_range` 在 5 mm 档进入 3 % 内，2.5 mm 档 6.7 % 未达 3 %，如实记录未达标，不改判据放宽。