# 表面张力 delta 函数归一：`2|∇c|` → `|∇c|/(1−c_s)`

**范围**: `src/tensorsph/csf_consistent.py`（`surface_color_level` /
`delta_s_from_gradient`，新增；`csf_fields`、`surface_tension_acceleration` 增参）、
`src/tensorsph/droplet_impact.py`（`Config.csf_delta_s`）、`src/tensorsph/surface_tension.py`（接线）
**基准**: 2-D 静止液滴的**三条互不依赖的解析判据**（Young-Laplace、切割力恒等式、
delta 函数归一）；本文件所有数字由 `benchmarks/csf_delta_s/{run.py,make_doc.py}` 从
`validation_report.json` 程序化注入，无手抄。

## 1. 案例介绍（物理背景/几何/参数）

CSF 表面张力模型把界面力写成

```
f = -sigma * kappa * delta_s * n ,     delta_s = 界面 delta 函数
```

`delta_s` 的唯一判据是**归一**（Brackbill et al. 1992）：

```
int_{interface} delta_s dn = 1
```

它决定了「一个粒子代表多少界面面积」。若 `delta_s` 偏小 x 倍，CSF 力（以及 Laplace 压差）
就按同一倍数偏小，与曲率算子无关。上一轮 `docs/verification_surface_tension_curvature.md`
把曲率修进 15% 门后，暴露出这条欠账：`int delta_s dn` 在三档分辨率上只有
0.755 / 0.751 / 0.739，而切割力恒等式顺便从 7.4% 掉到 10.7%。

| 参数 | 值 |
|---|---|
| 液滴半径 R | 0.02 m |
| 表面张力系数 sigma | 7.2 N/m |
| 参考密度 rho0 | 1000 kg/m^3 |
| 声速 c0 | 20 m/s（Mach ≈ 0.03） |
| 状态方程 | Tait, gamma = 7 |
| 光滑长度 | h = 1.3 dx（自身体积项 V·W(0) = 0.233） |
| 分辨率扫描 | dx = 0.003 / 0.002 / 0.0015 m（N = 161 / 362 / 642） |
| 曲率算子 | `fit_quadratic`（上一轮的结论，本轮固定不动，用于隔离变量） |
| 解析参考 | `kappa = 1/R = 50 1/m`；`F_cut = 2 sigma = 14.4 N/m`；`int delta_s dn = 1` |

## 2. 怎么做的（方法/流程）

### 2.1 先测「到底缺多少」，而不是先调常数

单相 SPH 粒子云里，色函数只存在于流体侧：核和指示函数 `c_i = Σ_j V_j W_ij` 在体相饱和到
1，在最外层粒子处只保留「半个核」。历史上取 `delta_s = 2 |∇c|`，其隐含假设是

> 表层粒子的核权重恰好一半 ⇒ 流体侧色函数分支为 `[1/2, 1]`。

本轮把这个假设**直接测了**：在模型自己的六方半格子上取最外层一行的 `c`，
得 `c_s = 0.731`（含自身体积项 0.233）——不是 1/2。也就是说
流体侧分支实际是 `[0.73, 1]`，`2|∇c|` 显然不足。

### 2.2 参数无关的归一：把流体侧分支重标到 [0,1]

`int_fluid |∇c| dn` 就是流体侧分支的跨度；把色函数重标为
`c̃ = (c − c_s)/(1 − c_s)` 后

```
int_fluid |∇c̃| dn = 1     ⇒     delta_s = |∇c| / (1 − c_s)
```

`c_s` 用**闭式估计**而非拟合：平面自由面上半个核落在真空中，自身项两边都在，故

```
c_s = V W(0) + (1 − V W(0)) / 2 = (1 + V W(0)) / 2 = 0.6165   (h = 1.3 dx)
```

该闭式只用到 `V`（= m/rho0）与核函数，无自由参数。实测流体侧梯度积分为 0.377，
与闭式跨度 `1 − c_s = 0.3835` 相差 1.7% —— 这就是 `1/(1−c_s) = 2.607` 的可信依据。

### 2.3 实现（改动最小、默认行为不变）

| 位置 | 改动 |
|---|---|
| `csf_consistent.surface_color_level(vol, h)` | 新增：`(1 + V W(0))/2` |
| `csf_consistent.delta_s_from_gradient(grad_c, vol, h, scheme)` | 新增：`"factor2"`（历史）与 `"influid"`（本轮） |
| `csf_fields(..., delta_s_scheme="factor2")` | 新增形参，默认沿用历史归一 |
| `surface_tension_acceleration(..., delta_s_scheme=...)` | 透传 |
| `Config.csf_delta_s = "factor2"` | 新增字段 + 校验（非法值报错） |
| `surface_tension.compute_surface_tension_force` | 从 `cfg` 透传 |

**默认保持 `factor2`**：换默认会改动所有既有基准/回归数字，本轮只交付「可验证的修复 +
A/B 证据」，切换默认属于下一单（见 §7）。

### 2.4 判定方式（三身份 + 两条负控）

1. `int delta_s dn = 1`（本轮的**直接判据**）；
2. `F_cut = 2 sigma`，但**在力密度 `f = -sigma kappa delta_s n` 上测量**——见 §5 第 1 条：
   `a = f/ρ` 会被自由面密度亏损放大，不能当精度判据；
3. `F_cut/(2σ)` 的加速度版（只作对照，不作判据）；
4. 负控 A：曲率 `kappa` 与力的方向**必须逐位不变**（这个旋钮只该改幅值）；
5. 负控 B：`factor2` 必须**仍然**给出 0.74–0.76（回归钉死）。

工具链：`benchmarks/csf_delta_s/run.py`（instant + relax + JSON）、
`benchmarks/csf_delta_s/make_doc.py`（出图 + 注入本文件的表）、
`tests/test_csf_delta_s.py`（11 个测试）。

## 3. 计算结果（数字表，程序化注入）

<!-- TABLES -->

#### delta_s scheme = `factor2`

| dx [m] | N | ∫δs dn [-] | err | <3 %? | <15 %? | F_cut on f [-] | err | F_cut on a [-] | err | ρ_shell/ρ0 (δs-weighted) | kappa [1/m] | κ err | cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0015 | 642 | 0.7391 | -26.09 % | no | no | 0.7102 | 28.98 % | 0.8934 | 10.66 % | 0.8076 | 50.43 | +0.9 % | +0.9997 |
| 0.0020 | 362 | 0.7510 | -24.90 % | no | no | 0.7132 | 28.68 % | 0.8992 | 10.08 % | 0.8016 | 52.66 | +5.3 % | +0.9997 |
| 0.0030 | 161 | 0.7545 | -24.55 % | no | no | 0.7864 | 21.36 % | 1.0123 | 1.23 % | 0.7972 | 58.89 | +17.8 % | +0.9993 |

#### delta_s scheme = `influid`

| dx [m] | N | ∫δs dn [-] | err | <3 %? | <15 %? | F_cut on f [-] | err | F_cut on a [-] | err | ρ_shell/ρ0 (δs-weighted) | kappa [1/m] | κ err | cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0015 | 642 | 0.9637 | -3.63 % | no | yes | 0.9260 | 7.40 % | 1.1648 | 16.48 % | 0.8076 | 50.43 | +0.9 % | +0.9997 |
| 0.0020 | 362 | 0.9791 | -2.09 % | yes | yes | 0.9299 | 7.01 % | 1.1724 | 17.24 % | 0.8016 | 52.66 | +5.3 % | +0.9997 |
| 0.0030 | 161 | 0.9838 | -1.62 % | yes | yes | 1.0253 | 2.53 % | 1.3199 | 31.99 % | 0.7972 | 58.89 | +17.8 % | +0.9993 |

#### 时间推进（relax，dx = 0.002 m）

| scheme | Δp [Pa] | err vs σ/R | spurious max\|v\|/√(σ/ρR) | <0.1? |
|---|---|---|---|---|
| `factor2` | -24.25 | -106.7 % | 0.0275 | yes |
| `influid` | 38.45 | -89.3 % | 0.0364 | yes |
