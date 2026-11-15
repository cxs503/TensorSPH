# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE，第 7 轮：气相压强台阶的机理判定与读数对称性

锚点 `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`。第 6 轮把气相的压强台阶（`<p>(s=+6..8 dx)` 随分辨率*增大*）归因于“Poisson 源残余单极子经 Dirichlet 规范成像填满气相”，并把“强制源净电荷为零”留作后续实验。本轮把该归因当作可判定的假设来检验，同时检验第二个候选机理（净牵引力偶极子）与读数仪器本身。

## ① 案例与方法

静止椭球液滴（与第 5、6 轮同一载流子）：`pi a b = pi R_eq^2`，`sigma` = 0.072 N/m，`rho0` = 1000 kg/m³（两相同密度静态载流子），owner 精度门限 15 %。构型：a:b=3.0 `R_eq/dx`=20、a:b=3.0 `R_eq/dx`=40、a:b=2.0 `R_eq/dx`=20、a:b=2.0 `R_eq/dx`=40；第 5 轮预先声明的可分辨性前提为 `R_curv,min/dx >= 5`（本轮的 `r_curv_min_over_dx` 见 JSON）。

四个臂，共用一个格点、一套 colour/curvature/force 算子、一个 Poisson 求解器：

| 臂 | 做法 | 检验的假设 |
|---|---|---|
| `shipped` | `rhs = -div(f)`（第 4–6 轮源码） | 基线（缺陷） |
| `src_neutral` | `neutralize_source()` 去掉自由集均值 | **H1 单极子** |
| `traction_balanced` | `f <- f - F_net/A_shell`（闭合界面约束） | **H2 偶极子** |

读数仪器研究：出厂探针是**非对称**的（内侧 2–4 dx、外侧 3–6 dx），另测四组对称探针 `sym_2_4 / sym_3_6 / sym_4_7 / sym_5_8`（内外同偏移）。第 6 轮的 “moved probes” 臂是全局壳层、24 个站位读到同一个数，因此从未真正测过这一项。

## ② 结果（数字由 JSON 注入）

### 机理判定

| 构型 | 源相对电荷 | 净牵引力比 | 气相台阶 `s=6..8 dx` (Pa) shipped / src_neutral / traction_balanced |
|---|---|---|---|
| a:b=3.0 `R_eq/dx`=20 | -2.63e-16 | 2.16e-03 | +0.476 / +0.476 / +0.476 |
| a:b=3.0 `R_eq/dx`=40 | 2.65e-16 | 1.95e-03 | +0.754 / +0.754 / +0.754 |
| a:b=2.0 `R_eq/dx`=20 | -1.18e-16 | 1.18e-03 | +0.250 / +0.250 / +0.250 |
| a:b=2.0 `R_eq/dx`=40 | -3.59e-17 | 7.91e-04 | +0.422 / +0.422 / +0.422 |

**H1 被否定**：出厂源的相对电荷在所有构型上为 2.7e-16（判据 `h1_source_charge_rel_shipped` >= 1e-03），即该载流子的源本来就是中性的；`src_neutral` 是空操作，气相台阶一字未动。

**H1 的措辞也不成立**：`pressure_poisson` 把 Dirichlet 行替换为恒等行，组装后的矩阵**非奇异**，右侧带电并不使 CG 停滞——单元测试 `test_dirichlet_gauge_makes_a_charged_rhs_solvable` 用 10 % 带电右侧测得相对残差仍达求解容差。也就是说“常数模不可表示、CG 停在有限残差”这一说法在第 6 轮文本中是错的；带电的真正后果是在吸收边界下注入净源、产生不衰减的台阶。

**H2 也没有解释力**：净牵引力比 shipped = 2.16e-03（第 5 轮 1e-2 门限内），精确平衡后 = 3.0e-17，而气相台阶变化 < 5 %（`test_h2_traction_dipole_cure_does_not_move_the_tail`）。判据 `vapour_tail_reduction_traction_cure_min` = 4× → 失败。

**结论**：台阶既不是源单极子也不是净牵引力偶极子的像，而是**核函数抹平后的界面过渡带本身**（H3）：在 `s = 3..8 dx`（≈ 2.3–6 h）处并不存在平坦的气相平台，出厂外侧探针读到的正是过渡带的外翼。

### 读数仪器：探针非对称

同一压力场、同一站位、只改探针偏移（`shipped` 臂，偏差均值为 24 个站位平均）：

| 构型 | shipped (2–4 / 3–6) | sym_2_4 | sym_3_6 | sym_4_7 | sym_5_8 |
|---|---|---|---|---|---|
| a:b=3.0 `R_eq/dx`=20 | +38.57 % (rms 52.01) | +29.49 % (rms 44.43) | +41.72 % (rms 54.84) | +44.66 % (rms 59.72) | +47.11 % (rms 64.02) |
| a:b=3.0 `R_eq/dx`=40 | +15.94 % (rms 30.06) | +10.32 % (rms 24.52) | +17.47 % (rms 31.90) | +20.73 % (rms 36.51) | +23.28 % (rms 40.56) |
| a:b=2.0 `R_eq/dx`=20 | +10.74 % (rms 28.01) | +7.70 % (rms 24.55) | +13.19 % (rms 31.50) | +14.95 % (rms 36.03) | +16.44 % (rms 40.18) |
| a:b=2.0 `R_eq/dx`=40 | +6.80 % (rms 19.24) | +3.92 % (rms 16.82) | +8.06 % (rms 20.26) | +9.51 % (rms 23.19) | +10.66 % (rms 25.81) |

最佳对称探针组合：`sym_2_4`。全 4 个构型上最大 |偏差| = 29.49 %（含 `R_curv,min/dx` < 5 的不可分辨水平）；**可分辨子集**（2 个构型）上最大 |偏差| = 7.70 %。

对称化把出厂仪器在每一档上的偏差都降低了：38.57→29.49 %、15.94→10.32 %、10.74→7.70 %、6.80→3.92 %（非对称→对称，逐构型）。其中 a:b=3:1 `R_eq/dx`=40 正是第 6 轮报 +15.94 %、判 owner 门限失败的那一档。

判据 `i1_symmetric_probes_beat_shipped`：**通过**（对称 +29.49 % vs 非对称 +38.57 %）。

## ③ 误差判定（逐项，含 <3 % 与否）

| 量 | 基准 | 实测（最佳对称探针） | < 3 % ? |
|---|---|---|---|
| a:b=3.0 `R_eq/dx`=20 的压差读数均值 | `sigma kappa_window` | +29.49 % | 否 |
| a:b=3.0 `R_eq/dx`=40 的压差读数均值 | `sigma kappa_window` | +10.32 % | 否 |
| a:b=2.0 `R_eq/dx`=20 的压差读数均值 | `sigma kappa_window` | +7.70 % | 否 |
| a:b=2.0 `R_eq/dx`=40 的压差读数均值 | `sigma kappa_window` | +3.92 % | 否 |
| a:b=3.0 `R_eq/dx`=20 的牵引线积分 | `sigma kappa_window` | +4.52 % | 否 |
| a:b=3.0 `R_eq/dx`=40 的牵引线积分 | `sigma kappa_window` | -0.05 % | 是 |
| a:b=2.0 `R_eq/dx`=20 的牵引线积分 | `sigma kappa_window` | +0.97 % | 是 |
| a:b=2.0 `R_eq/dx`=40 的牵引线积分 | `sigma kappa_window` | +0.93 % | 是 |

owner 15 % 门限，可分辨子集：7.70 % → 通过；全构型（含不可分辨档）：29.49 % → **失败（honest FAIL）**。

3 % 声明线，可分辨子集：7.70 % → **失败（honest FAIL）**——最好的可分辨档为 a:b=2:1 `R_eq/dx`=40 的 3.92 %，仍高于 3 %。

**没有任何门限被放宽**：不可分辨档（a:b=3:1、`R_eq/dx`=20，tip 处 `R_curv,min/dx` = 3.31 < 5）单列在表里，既没有从均值里剔除，也没有改判。

## ④ 图

![vapour tail and probe symmetry](results/neutral_source.png)

左：气相侧压强沿法向距离的剖面（三臂），右：同一压力场下不同探针偏移的读数偏差（出厂臂）。

## ⑤ 注意点与局限

* 载流子的气相填满液体密度粒子（第 1–6 轮同一静态载流子）；Young-Laplace 定律只含`sigma` 与 `R`，密度比不入式——这不是多相流模型，也不声称是。
* 本轮**没有**把压差读数带进 3 %（可分辨子集最好 3.92 %，见上）；它给出的是**因果排除 + 一项仪器修正**：源单极子与净牵引力偶极子都不是台阶的来源，因此“强制源净电荷为零”这条被第 6 轮推荐的后续实验是无效投资，不应再派单；有效的是把内外探针对称化（每档都降 3–6 个百分点）。
* 探针非对称是**仪器偏置**，不是物理改进：对称化只改变了读数，没有改变力模型，因此牵引线积分（不受仪器改动影响的独立仪器）数值不变——两条路线的偏差差值是判断“读数是否被修好”的独立证据。
* 探针外移到 5–8 dx 后 rms 上升（见上表），说明**站位级**散射仍是核尺度抹平的固有极限，宽探针只是把偏差降低、把方差抬高；这一权衡写进了表，未挑选有利量。
* 第 6 轮的 `moved_probes`/`wide_probes` 臂是全局壳层（24 站位同值），在那里被标为“not run”是正确的；本轮的探针集是**逐站位**掩码，否则又会得到同一个数。

## ⑥ 复现

```
python3 scripts/xval_projection_laplace/neutral_source.py
python3 scripts/xval_projection_laplace/make_neutral_source_doc.py
python3 -m pytest tests/test_xval_projection_laplace_source_charge.py -q
```

工件：`src/tensorsph/laplace_isph.py`（`neutralize_source`、`pressure_poisson(neutralize=, return_diag=)`）、`scripts/xval_projection_laplace/neutral_source.py`、`results/neutral_source.json`、`results/neutral_source.png`、`tests/test_xval_projection_laplace_source_charge.py`。
