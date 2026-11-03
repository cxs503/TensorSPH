"""Publish the m2574 report; numbers injected from benchmarks/laplace_scaling/results.json."""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m2574")
d = json.loads((ROOT / "benchmarks/laplace_scaling/results.json").read_text())
v, t = d["verdict"], d["verdict"]["bias_trend"]
grid, ladder, fix = d["grid"], d["resolution_ladder"], d["fixed_dx_sweep"]
sup = v["support_sensitivity"]
bt = t["bias_at_R_over_dx"]

rows = "\n".join(
    f"| {c['radius_m']*1000:.0f} | {c['resolution_R_over_dx']:.0f} | {c['sigma_N_per_m']:g} | {c['n_particles']} | "
    f"{c['cut_force']['ratio_measured_over_theory']:.4f} | {c['cut_force']['rel_error_pct']:+.2f} % | "
    f"{'pass' if c['cut_force']['within_15pct'] else 'FAIL'} |" for c in grid)
rows_l = "\n".join(
    f"| {c['resolution_R_over_dx']:.0f} | {c['n_particles']} | {c['cut_force']['ratio_measured_over_theory']:.4f} | "
    f"{c['cut_force']['rel_error_pct']:+.2f} % | {c['cut_force_geometric']['ratio_measured_over_theory']:.4f} | "
    f"{c['angle_average']['ratio_momentum_mean']:.4f} ± {0.5*c['angle_average']['ratio_momentum_spread']:.3f} | "
    f"{c['curvature']['delta_s_weighted_signed']*c['radius_m']:.3f} | {c['delta_s']['integral_dn']:.4f} | "
    f"{c['cut_force_geometric']['rho_over_rho0_force_weighted']:.4f} |" for c in ladder)
rows_f = "\n".join(
    f"| {c['radius_m']*1000:.0f} | {c['resolution_R_over_dx']:.0f} | "
    f"{c['cut_force']['ratio_measured_over_theory']:.4f} | {c['cut_force']['rel_error_pct']:+.2f} % |" for c in fix)
ref = next(c for c in ladder if c["resolution_R_over_dx"] == 10.0)
k_in = ref["curvature"]["inner_half_mean"] * ref["radius_m"]
k_out = ref["curvature"]["outer_half_mean"] * ref["radius_m"]
png = base64.b64encode(Path("/tmp/laplace_scaling_small.png").read_bytes()).decode()

markdown = f"""## ① 案例介绍

TensorSPH 于 m2447 合入一致化 CSF 表面张力格式（`src/tensorsph/csf_consistent.py`：kernel-sum 颜色场 +
MLS 梯度/散度 + Shepard 平滑法向），取代 legacy 格式（曲率为负、法向不连贯、δ_s 弱 3×）。
其验收证据只有**单一参数点** `R = 20 mm, σ = 7.2 N/m`（`benchmarks/csf_consistent`，`R/Δx = 8…20`）。
单点无法区分「复现了 Laplace **定律**」与「在某一半径某一张力下碰巧对数」。

待答：Young-Laplace 2-D 圆柱形式 `Δp = σ/R ⟺ F_cut = 2σ` 在 (R, σ, R/Δx) 空间成立到什么范围？

| 量 | 取值 |
|---|---|
| 几何 | 2-D 圆盘，hex 点阵，`m = ρ0·dx·dy` |
| 物性 | `ρ0 = 1000 kg/m³`，`c0 = 20 m/s`，`γ = 7`，`ν = 0.01` |
| 核函数 | cubic spline，`h = 1.3Δx`（另扫 1.2 / 1.5） |
| 表面张力 | `σ = 3.6 / 7.2 / 14.4 N/m`，`csf_scheme = "consistent"` |
| 半径 | 网格轴 `R = 15/20/30 mm`（固定 `R/Δx = 10`）；固定 `Δx = 2 mm` 轴 `R = 12/20/30 mm` |
| 分辨率阶梯 | `R/Δx = 8, 10, 16, 20, 28, 40`（`R = 20 mm, σ = 7.2`） |
| 判据 | 每点 `|F_cut/2σ − 1| ≤ 15 %`（平台标准） |

**为何不测压强**：该点真实 `Δp = 360 Pa` 需相对密度变化 `Δp/K ≈ 1.3e-5`，低于 SPH 噪声
（`benchmarks/laplace_droplet` 可分辨性审计：`signal_to_noise < 0.002`，判 `not resolvable`）。
可绕开刚性 Tait 状态方程的 CSF 矩是切割面单位长度受力 `F_cut = Σ_{{x>0}} m_i a_i`。

## ② 怎么做的

`benchmarks/laplace_scaling/run.py`（纯静态诊断，无时间推进，float64，2 线程）：hex 圆盘点阵 →
`density_from_neighbor_pairs` → `csf_fields` → **生产入口** `compute_surface_tension_force` 取加速度；
同一受力的三种独立读数：动量估计量 `Σ m a`、几何估计量 `Σ A_cell ρ a`、切面角平均（0/15/30/45°，
hex 点阵仅 6 次对称）；再给曲率的内/外半区拆分与 `∫δ_s dn`；外加分辨率阶梯到 `R/Δx = 40`
与核支撑扫描。工件：`results.json`（数字源）→ `make_doc.py` → `docs/verification_laplace_scaling.md`
（零手抄）+ 三联图。

## ③ 计算结果

### 3.1 (R, σ) 网格振幅 —— {v['amplitude_status']}

| R [mm] | R/Δx | σ [N/m] | N | F_cut/2σ | 误差 | ≤15 % |
|---|---|---|---|---|---|---|
{rows}

网格最大偏差 **{v['max_abs_ratio_error_pct']:.2f} %**（{v['n_within_tolerance']}/{v['n_points']} 点达标）。

### 3.2 标度指数 —— {v['scaling_status']}

`F_cut ~ σ^p R^q`：`p = {v['sigma_exponent']:.6f}`（理论 1），`q = {v['radius_exponent']:.2e}`（理论 0）。
**这是自洽性检验而非独立验证**：受力对 σ 线性是构造使然，网格轴固定 `R/Δx` 使三半径几何自相似，
`p=1,q=0` 只反映离散算子的量纲齐次性。能证伪 R 无关性的是 3.4 的固定 Δx 一轴。

### 3.3 分辨率阶梯（R = 20 mm，σ = 7.2 N/m）

| R/Δx | N | F_cut/2σ（动量） | 误差 | 几何估计量 | 角平均 ± 半幅 | κR | ∫δ_s dn | 壳层 ρ/ρ0 |
|---|---|---|---|---|---|---|---|---|
{rows_l}

- 角平均与单向 x 切割差 **≤ 0.8 %** → 残差为系统性偏置，非点阵求积噪声；
- 偏置**随加密增大**：每单位 `R/Δx` 增 `{t['d_bias_per_R_over_dx']:.4f}`；线性外推门限在
  `R/Δx ≈ {t['extrapolated_R_over_dx_where_15pct_is_breached']:.1f}`，实测夹逼：`R/Δx = 20` 为 `{bt['20']*100:+.1f} %`（过），
  `R/Δx = 28` 为 `{bt['28']*100:+.1f} %`（FAIL）。

### 3.4 固定 Δx = 2 mm 半径扫描

| R [mm] | R/Δx | F_cut/2σ | 误差 |
|---|---|---|---|
{rows_f}

三半径散布 **{v['fixed_dx_radii']['spread']*100:.2f} %** → 点阵非自相似时受力近似 R 无关：**定律形式保住，偏置属于分辨率**。

### 3.5 核支撑敏感度（R/Δx = 10）

| h/Δx | F_cut/2σ | 误差 |
|---|---|---|
| 1.2 | {sup['1.2']:.4f} | {(sup['1.2']-1)*100:+.2f} % |
| 1.3（默认） | {sup['1.3']:.4f} | {(sup['1.3']-1)*100:+.2f} % |
| 1.5 | {sup['1.5']:.4f} | {(sup['1.5']-1)*100:+.2f} % |

残差**不是**收敛的物理量：`1.2Δx` 仅 0.9 %（远在门限内），默认 `1.3Δx` 7.7 %，`1.5Δx` 7.9 %。

### 3.6 曲率与 δ_s：同一受力的两半

`R/Δx = 10` 处 δ_s 加权曲率内半区 `κR = {k_in:.3f}`、外半区 `κR = {k_out:.3f}`（精确 1）；
`∫δ_s dn = {ref['delta_s']['integral_dn']:.4f}`（理想 1）。CSF 力是乘积 `σκδ_s`：两因子各偏 ~±25 %，
乘积落在门限内。signed 与 `|κ|` 加权值打印位全同（`{ref['curvature']['delta_s_weighted_signed']*ref['radius_m']:.4f}` vs
`{ref['curvature']['delta_s_weighted_abs']*ref['radius_m']:.4f}`）→ `benchmarks/csf_consistent` 的 `|κ|` 未掩盖符号错误，
是真实算子偏差（内半区 {(k_in-1)*100:+.1f} %、外半区 {(k_out-1)*100:+.1f} %）。

## ④ 结果图

三联图：(a) `F_cut/2σ` 随分辨率，灰带 = 15 % 容差，点线 = 趋势击穿门限处，空心方块 = 固定 Δx 三半径；
(b) 同比值随 σ，三半径塌成平线，即定律的 `Δp ∝ σ`；(c) 受力的两半——内外半区曲率拆分 + 核支撑扫描对 `κ = 1/R` 的评分。

## ⑤ 注意点

1. **验收量自身带 15 % 级不确定度**：动量估计量与几何估计量在自由面差 `1/ρ̄ ≈ 1.25`
   （`R/Δx = 10` 壳层 `ρ/ρ0 = {ref['cut_force_geometric']['rho_over_rho0_force_weighted']:.3f}`）；`R/Δx ≥ 28` 时两者分歧已超容差本身，
   该判据在该分辨率下无法分辨模型好坏。
2. **标度指数不是证据**（见 3.2）。
3. **仅静态诊断**：虚假流、压强场、真实振荡液滴阻尼不在本工件内。
4. **仅 2-D**：`Δp = σ/R`；3-D 球为 `2σ/R`，本仓无 3-D 格式，不作外推。
5. **过程坑**：`benchmarks/csf_consistent/run.py --quick` 会就地覆写已提交的 `validation_report.json`
   （只写 instant 段、丢掉 relax 段）；本次已 `git checkout --` 还原，建议后续给 `--quick` 加 `--out` 默认落临时文件。

## ⑥ 误差判定

| 判据 | 实测 | 门限 | 判定 |
|---|---|---|---|
| 振幅 `|F_cut/2σ − 1|`（网格） | {v['max_abs_ratio_error_pct']:.2f} % | 15 % | **{v['amplitude_status']}** |
| 隐含 `Δp` 相对误差（网格） | {v['max_abs_dp_error_pct']:.2f} % | 15 % | **{v['amplitude_status']}** |
| σ 指数 `p` | {v['sigma_exponent']:.6f} | 1 ± 5 % | **{v['scaling_status']}** |
| R 指数 `q` | {v['radius_exponent']:.2e} | 0 ± 0.1 | **{v['scaling_status']}** |
| 偏置随加密变化 | {bt['8']*100:+.1f} % → {bt['40']*100:+.1f} % | 必须下降 | **FAIL** |
| 15 % 成立的 `R/Δx` 上限 | ≤ {t['extrapolated_R_over_dx_where_15pct_is_breached']:.0f}（20 过 / 28 挂） | 全分辨率 | **有界** |

**是否 < 3 %？——否。** 最好情形（`h = 1.2Δx`, `R/Δx = 10`）0.9 %，默认设置 7.7 %。
**是否 < 15 %？——在 `R/Δx ≲ 20`、`h/Δx ≈ 1.2–1.5` 内：是**（网格最大 7.66 %）；**渐近：否**
（`R/Δx = 28` 为 `{bt['28']*100:+.1f} %`，`R/Δx = 40` 为 `{bt['40']*100:+.1f} %`）。
故诚实表述为「`R/Δx ≲ 20` 内 15 % 以内、偏置随加密增大」，而非无保留的「复现 Young-Laplace 跳变」。

## 交付物与后续线索

- `benchmarks/laplace_scaling/{{run.py, make_doc.py, results.json, laplace_scaling.png}}`；
  `docs/verification_laplace_scaling.md`（make_doc 注入）；`tests/test_laplace_scaling.py`（11 项，含仪器自检与包络钉住，
  「偏置随加密增大」被故意外写成断言）；`tests/test_laplace_droplet.py` 4 条 legacy 断言已重钉到当前模型。
- 线索：(a) 换 κ 算子（MLS 二阶张量 / 抛物线拟合）使偏置随加密下降——届时 `test_bias_grows_with_refinement` 会红，
  强制更新验证记录；(b) 给 `--quick` 加 `--out`；(c) 余下 4 条与表面张力无关的基线红测试单独收口。
"""

report = {
    "title": "TensorSPH 表面张力验证欠账：Laplace 定律 (R,σ,R/Δx) 包络实测——单点 PASS(7.66%)，"
             f"但偏置随加密增大，R/Δx≈{t['extrapolated_R_over_dx_where_15pct_is_breached']:.0f} 击穿 15% 门限",
    "summary": ("m2447 合入的一致化 CSF 此前只有 R=20 mm/σ=7.2 N/m 单点验收。本单补 (R,σ) 网格 + 固定 Δx 半径轴 + "
                f"R/Δx=8…40 分辨率阶梯 + h/Δx 扫描 + 切面角平均：振幅在 R/Δx≤20 内最大误差 {v['max_abs_ratio_error_pct']:.2f}%"
                f"（9/9 点达标），标度指数 p={v['sigma_exponent']:.6f}、q={v['radius_exponent']:.1e}（属量纲齐次性，非独立验证）；"
                f"关键负面结论：偏置随加密单调增大 {bt['8']*100:+.1f}%→{bt['40']*100:+.1f}%，"
                f"15% 门限外推在 R/Δx≈{t['extrapolated_R_over_dx_where_15pct_is_breached']:.1f}（20 过、28 挂 +20.0%），"
                f"并随 h/Δx 摆动（1.2Δx 仅 0.9%）。仪器自检发现动量/几何两种受力估计量在自由面差 1/ρ̄≈1.25，"
                "即该判据在 R/Δx≥28 时自身不确定度已超容差。另修复 4 条因默认格式变更（m2447）而一直红的 legacy 断言。"),
    "scenario_key": "tensorsph-laplace-law-scaling-envelope",
    "mission_id": 2574,
    "tags": ["TensorSPH", "surface-tension", "Young-Laplace", "validation-debt", "CSF", "honest-FAIL",
             "resolution-envelope"],
    "markdown": markdown,
    "figures": [{
        "name": "laplace_scaling.png",
        "caption": "TensorSPH 一致化 CSF 的 Laplace 定律包络：(a) F_cut/2σ 随 R/Δx（灰带=15% 容差，点线=线性趋势击穿门限处 "
                   "R/Δx≈20.8，空心方块=固定 Δx 三半径）；(b) 同比值随 σ，三半径塌成平线，即 Δp∝σ；(c) 曲率内/外半区分裂与 "
                   "核支撑 h/Δx 敏感度对 κ=1/R 的评分。",
        "png_base64": png,
    }],
}

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = urllib.request.Request(base + "/auth/login",
                              data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                               "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
tok = json.loads(urllib.request.urlopen(login, timeout=30).read())
token = tok.get("access_token") or tok.get("token")
print("payload KB:", len(json.dumps(report)) // 1024)
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(report).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + token}, method="POST")
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    print("REPORT ID:", resp.get("id"), "| keys:", list(resp)[:8])
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read()[:600])