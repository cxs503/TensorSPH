# CASE-SPH-WEDGE-ENTRY-2D — resolution band re-measured on the fixed candidate grid（m23280）

**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力）
**本单**: m23280（tp-infra-1 自找活）　**日期**: 2026-11-16
**工件**: `benchmarks/case_sph_wedge_entry_2d/rebuild_from_frames.py`（新）、
`docs/artifacts/case_sph_wedge_entry_2d_ensemble_bandfix/`（重测的 4/2 mm 读数 + 逐帧样本）、
`..._rebin/`（离线重分箱报告）、`..._ladder_bandfix/ladder_closure.json`、
`docs/figures/case_sph_wedge_entry_2d_band_remeasure_m23280.png`、`tests/test_case_sph_wedge_entry_2d_band_remeasure.py`

## ① 为什么这是欠账而不是重做

`docs/case_sph_wedge_entry_2d_band_coverage_m22695.md` §5 把 m22695 的修复停在半路，原文：
> *"re-measuring the band at 40..240 bins requires re-running the ensembles (a solver run,
> not a re-binning of committed data -- the per-bin peaks are not stored)"*

即：仪器修好了，但**没有一张网格用它重新测过**；判决仍然建立在「`dx=0.5 mm` 的带只有 1 个成员、
spread 恒等于 0」的那批读数上。本轮做两件事：
1. 把**逐帧壁面样本**（`tau, cp`）落盘，并写 `rebuild_from_frames.py` 用同一套估计量离线复算，
   从此重分箱不再需要求解器；
2. 用修复后的 `BAND_GRID` 重跑**同一算例、同一设置、同一估计量**（只改候选分箱表），
   并把新读数送进**同一份** `ladder_closure.py` / `literature_regrade.py` 判据。

## ② 方法（可复现的最小命令）

```bash
PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py \
    --dx 0.002 --threads 4 --frames 61 \
    --out artifacts/case_sph_wedge_entry_2d_ensemble \
    --dump-frames artifacts/case_sph_wedge_entry_2d_ensemble/frames_dx2mm_c0v20.json
python benchmarks/case_sph_wedge_entry_2d/rebuild_from_frames.py \
    --ensemble .../ensemble_dx2mm_c0v20.json --frames .../frames_dx2mm_c0v20.json
PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/ladder_closure.py \
    --artifacts docs/artifacts/case_sph_wedge_entry_2d_ensemble_bandfix \
    --out       docs/artifacts/case_sph_wedge_entry_2d_ladder_bandfix
```

求解器、几何、`beta=30°`、`V=1 m/s`、`c0/V=20`、Courant 0.35、`alpha_visc=0.03`、
61 帧自相似窗口、样本门 `MIN_BIN_SAMPLES=4` 全部未动；容差 15 % 未动。

## ③ 结果（程序化注入，零手抄）

| grid | dx [mm] | 物理带 | 候选（pre-fix） | 候选（重测） | $C_p$（pre-fix） | $C_p$（重测） | 变化 | spread pre-fix → 重测 | jackknife | $U$（重测） | 对 COM 残差 | 离线复算偏差 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dx4mm | 4 | [4, 16] | 6, 8, 10, 12, 16 | 6, 8, 10, 12, 16 | 10.041 | **10.041** | +0.00 % | 23.53 % → 23.53 % | 0.12 % | 23.53 % | +45.11 % | 0.00e+00 |
| dx2mm | 2 | [7, 30] | 8, 10, 12, 16, 20, 30 | 8, 10, 12, 16, 20, 24, 30 | 6.071 | **6.291** | +3.63 % | 11.83 % → 11.41 % | 42.95 % | 42.95 % | -9.08 % | 0.00e+00 |
| dx1mm | 1 | [15, 64] | 16, 20, 30 | **未重测** | 6.805 | — | — | 3.16 % | 6.48 % | — | — | — |
| dx0.5mm | 0.5 | [29, 119] | 30 | **未重测** | 8.984 | — | — | 0.00 % | 20.90 % | — | — | — |

参考（`refs.json`，q 单位）：Z&F 1993 复合解 COM = **6.920**，Dobrovol'skaya 精确相似解 EIM = **6.900**；两者差 0.29 %，互为旁证。门 `G = 0.15`。

![band remeasure](docs/figures/case_sph_wedge_entry_2d_band_remeasure_m23280.png)

### 阶梯闭合（重测集，`ladder_closure.py` 原样判据）

| grid | obs $C_p$ | $U$ | 同算子真值 | 同算子残差 | 可裁决 |
|---|---|---|---|---|---|
| dx4mm | 10.041 | 23.53 % | 6.870 | +46.16 % | False |
| dx2mm | 6.291 | 42.95 % | 7.494 | -16.05 % | False |

判决：**NOT_DECIDABLE** — only 0 judgeable grid(s) (estimator spread > 15%)（可裁决集 `[]`）。

`missing_grids`（未进入本次重测）：`['dx1mm', 'dx0.5mm']`。

## ④ 注意点与局限（不粉饰）

* **两张细网格没有交付**：`dx=1 mm`（集群作业 `cj4733`）与 `dx=0.5 mm`（`cj4732`）在本单
  截止时仍处于排队态（队列 136+ 作业在前，09:36 递交，28.9 min 时仍 `queued`）。
  因此本单**不改判**：算例判决仍以已提交的 pre-fix 读数为准（`literature_regrade.json`：
  `PARTIAL_SINGLE_GRID`，仅 `dx=1 mm` 可裁决）。任何「全网格 <15 %」的说法都不成立。
* `dx=2 mm` 重测后 `U` 仍由 **jackknife（帧二分）** 撑住，不是带 spread；
  带修复把该网格的 spread 从 11.83 % 抬到上表值，但 `U = max(spread, jackknife)` 由 jackknife 决定，
  所以**带修复不能**让 `dx=2 mm` 变成可裁决——这一点与 m22695 §5 的预判一致，本轮首次用数据证实。
* 重跑的可复现性：`dx=4 mm` 的带读数与已提交工件**逐位一致**（见上表「变化」列为 0.00 %），
  说明本机 re-run 与造工件的那次是同一离散化路径，重测差异可全部归因于候选分箱表。
* 离线复算（`rebuild_from_frames.py`）在 4 mm/2 mm 上对每个带成员的偏差见上表——
  它是「工件 == 样本的函数」这一点的直接证据，但**不能**替代求解器：样本本身仍需一次 run。

## ⑤ 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 值 | 基准 | 误差 | < 3 % ? |
|---|---|---|---|---|
| 离线复算 vs 工件带峰值（dx4mm） | 见工件 | 同一估计量 | 0.0e+00（相对） | **是**（机器精度） |
| 离线复算 vs 工件带峰值（dx2mm） | 见工件 | 同一估计量 | 0.0e+00（相对） | **是**（机器精度） |
| `dx=4 mm` 重测 vs 已提交读数 | 10.0415 | 10.0415 | +0.00e+00 % | **是**（0，确定性复现） |
| `dx=2 mm` 重测 vs pre-fix 读数（带修复的影响） | 6.2914 | 6.0708 | +3.63 % | 否（这是仪器差，不是求解器误差） |
| `dx=2 mm` 重测 vs COM | 6.2914 | 6.9200 | -9.08 % | 否（但该网格 `U`=43.0 % 不可裁决，**不记分**） |
| 算例判决 | 见 §③ 阶梯表 | 判据要求 ≥2 个同号可裁决网格 | 本单只有 ≤2 个粗网格且均不可裁决 | 判决**不变**（NOT_DECIDABLE / PARTIAL_SINGLE_GRID） |

