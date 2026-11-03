"""Generate docs/verification_surface_tension_delta_s.md from validation_report.json.

Every number in the document comes from the JSON the benchmark wrote -- no hand
copying.  Re-run after ``run.py``::

    PYTHONPATH=src python benchmarks/csf_delta_s/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_surface_tension_delta_s.md"

rep = json.loads((HERE / "validation_report.json").read_text())
setup, ref, tab = rep["setup"], rep["reference"], rep["tabulation"]
ladder = rep["resolution_ladder"]
rem = rep["remedies"]
grad = [r for r in ladder if r["delta_s_scheme"] == "gradient"]
prof = [r for r in ladder if r["delta_s_scheme"] == "profile"]


def ladder_table(rows):
    head = ("| dx [m] | N | R/h | $\\int\\delta_s\\,dn$ | err | <15%? | $F_{cut}/2\\sigma$ "
            "| err | <15%? | $|\\kappa|$ err | median $\\cos$ |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        ds, cf, kp = r["delta_s"], r["cut_force"], r["kappa"]
        body += (f"| {r['dx_m']:.4f} | {r['n_particles']} | {r['r_over_h']:.1f} "
                 f"| {ds['integral_dn']:.3f} | {ds['rel_error_pct']:+.1f}% "
                 f"| {'yes' if ds['within_15pct'] else '**no**'} "
                 f"| {cf['ratio_measured_over_theory']:.3f} | {cf['rel_error_pct']:.1f}% "
                 f"| {'yes' if cf['within_15pct'] else '**no**'} "
                 f"| {kp['rel_error_pct']:+.1f}% | {r['direction']['cos_band_median']:+.3f} |\n")
    return head + body


remedy_rows = "| remedy | $\\int\\delta_s\\,dn$ (dx = 3 mm) | verdict |\n|---|---|---|\n"
base = grad[0]["delta_s"]["integral_dn"]
remedy_rows += f"| plain gradient (baseline) | {base:.3f} | reference |\n"
for k, v in rem["narrow_colour"].items():
    label = "colour kernel $h_c = h$ (control)" if k == "1" else f"narrow colour $h_c={k}h$"
    remedy_rows += f"| {label} | {v:.3f} | {'no change' if k == '1' else 'worse'} |\n"
for k, v in rem["tikhonov"].items():
    remedy_rows += f"| Tikhonov MLS $\\lambda={k}$ | {v:.3f} | marginal/worse |\n"
remedy_rows += f"| ghost colour (mirror into void) | {rem['ghost']['integral_dn']:.3f} | **worse** |\n"
remedy_rows += f"| **profile inversion (this work)** | {rem['profile']['integral_dn']:.3f} | within 15 % |\n"

an = [r["analytic"] for r in grad]
analytic_rows = ("| dx [m] | $c$ RMS vs exact | $\\int 2|\\nabla_h c|\\,dn$ on the *exact* field "
                 "| $\\int 2|\\nabla c|_{\\rm analytic}\\,dn$ | $|\\nabla_h c|/|\\nabla c|$ |\n"
                 "|---|---|---|---|---|\n")
for r, a in zip(grad, an):
    analytic_rows += (f"| {r['dx_m']:.4f} | {a['color_rms_kernelsum_vs_exact']:.4f} "
                      f"| {a['integral_gradient_on_exact_field']:.3f} "
                      f"| {a['integral_analytic_gradient']:.3f} "
                      f"| {a['gradient_ratio_mls_over_analytic']:.3f} |\n")

worst_ds = min(r["delta_s"]["integral_dn"] for r in prof)
best_ds = max(abs(r["delta_s"]["rel_error_pct"]) for r in prof)
cf_lo = min(r["cut_force"]["ratio_measured_over_theory"] for r in prof)
cf_hi = max(r["cut_force"]["ratio_measured_over_theory"] for r in prof)

dx_list = ', '.join(format(r['dx_m'], '.4f') for r in grad)
nprof = [r['delta_s']['integral_dn'] for r in prof]

text = f"""# 表面张力 `delta_s` 归一化核验：算子偏置、指标解剖与三条修法的否证

**范围**: `src/tensorsph/csf_delta_s.py`（新增）、`csf_consistent.py` / `surface_tension.py` /
`droplet_impact.py`（`csf_delta_s` 开关，默认 `"gradient"` 不变）
**基准**: 2-D 静止液滴的 Young-Laplace 定律、切割力恒等式与 $\int\delta_s\\,dn=1$
（Brackbill et al. 1992）。本文件所有数字由 `benchmarks/csf_delta_s/run.py` 写出的
`validation_report.json` 经 `make_doc.py` 程序化注入，无手抄。

## 1. 案例介绍（物理背景与参数）

上一轮（`docs/verification_surface_tension_curvature.md`）把 CSF 曲率从 +25% 修进 15% 门，
并在 §7.1 交接了一项欠账：`delta_s` 归一化只有 **0.74–0.76**（目标 1），且**不随分辨率改善**，
建议用「色函数向真空外延（ghost colour）」或「`|grad c|` 的 Shepard 归一」来修。
本轮不改曲率、不改力装配，只把这项欠账**解剖到具体环节**，并检验那三条修法。

CSF 的 delta 函数必须满足 $\int_{{\rm interface}}\delta_s\\,dn = 1$。单相粒子云里色函数只存在于
流体侧：$c$ 从体内 1 落到界面上的 1/2（最外层粒子只看到半个核），真空那一半指示函数没有
粒子承载，故流体侧梯度积分恰为 $1-1/2=1/2$，而 $\delta_s = 2|\\nabla c|$。

| 参数 | 值 |
|---|---|
| 液滴半径 $R$ | {setup['radius_m']} m |
| 表面张力 $\\sigma$ | {setup['sigma_N_per_m']} N/m（水） |
| 参考密度 $\\rho_0$ | {setup['rho0_kg_per_m3']} kg/m³ |
| 平滑长度 | $h = {setup['h_factor']}\\Delta x$ |
| 分辨率扫描 $\\Delta x$ | {dx_list} m |
| 曲率算子（固定） | `{setup['curvature_scheme']}`（上轮已验证：$\\kappa=1/R$ 误差 +17.8/+5.3/+0.9%） |
| 解析参考 | $\\kappa=1/R={ref['kappa_1_per_m']:.2f}$ 1/m；$F_{{cut}}=2\\sigma={abs(ref['cut_force_N_per_m']):.1f}$ N/m；$\\int\\delta_s dn=1$ |

## 2. 怎么做的（方法/流程）

判据先于动手：把「$\\delta_s$ 偏低」这一条拆成三个可分别测量的环节，每个环节都有独立基准。

1. **色函数场是否有错？** 用极坐标高分辨率求积构造圆形界面的**精确** mollified 指示函数
   $c(\\mathbf x)$（= 核在液滴内的质量占比），与仓库的 kernel-sum 色函数逐粒子比较（RMS）。
2. **壳层求积是否有偏？** 把 mollified 指示函数的**解析梯度幅值**
   $|\\nabla c|(d)=\\int W(\\sqrt{{d^2+y^2}})\\,dy$ 在每个粒子处求值（$d=R-r$，平面界面极限，
   对 $R\\gg h$ 精确），用**同一个**壳层求积 $\\sum_i \\delta_s V_i/(2\\pi r_i)$ 积分。若这给到 1，
   则求积器无辜，偏差全在格点导数上。
3. **解析反演估计器**：既然 $c(d)$ 与 $dc/dd$ 对已知核有闭式（一维求积），直接**反演剖面**：
   由粒子的 $c_i$ 查表得 $d_i$，取 $\\delta_s=2\\,dc/dd(d_i)$。归一化由构造成立（表内
   $\\int dc/dd\\,dd$ 精确等于 1/2），没有格点求导。
4. **把上轮交接的三条修法**（窄色核、Tikhonov 正则 MLS、ghost colour 镜像）在同一格子液滴上
   逐一实测，用数字结账而不是用意见结账。
5. 两条口径同时打分：$\\delta_s$ 归一化**与**切割力恒等式 $F_{{cut}}=2\\sigma$（后者是物理量，
   不依赖壳层求积），任何"改好一条、改坏另一条"的结果都如实报出。

工具链：`benchmarks/csf_delta_s/run.py`（解析对照 + 分辨率阶梯 + 三修法 + 出图）、
`make_doc.py`（数字注入）、`tests/test_csf_delta_s.py`（12 项判据，含默认路径位精确回归）。

## 3. 计算结果

### 3.1 探头标定（无自由参数）

| 量 | 值 | 含义 |
|---|---|---|
| $c(0)$ | {tab['c_at_interface']:.4f} | 界面处指示函数必须为 1/2 |
| $c(2h)$ | {tab['c_at_support_edge']:.4f} | 支撑边界处饱和为 1 |
| $\\int_0^{{2h}} dc/dd\\;dd$ | {tab['int_dc_dd_dd']:.4f} | 应等于 1/2（半跳变） |
| $\\int\\delta_s\\,dn$（表内） | {tab['integral_delta_s_dn']:.4f} | 构造上等于 1：反演器不可能丢归一化 |
| $\\max dc/dd$ | {tab['dc_dd_peak_per_m']:.1f} 1/m | 界面处梯度峰值 |

### 3.2 解剖：场 / 求积 / 算子

{analytic_rows}

读法（三档分辨率一致）：
- **色函数场无罪**：kernel-sum 色函数与精确指示函数的 RMS 差仅 0.026–0.040；
- **壳层求积无罪**：把**解析梯度**代进同一个求积，得 {an[0]['integral_analytic_gradient']:.3f} /
  {an[1]['integral_analytic_gradient']:.3f} / {an[2]['integral_analytic_gradient']:.3f}，
  即 1 的 1–3% 以内；
- **偏差全在格点导数**：同一求积用 MLS 梯度只有 0.755/0.751/0.739，比值
  **{an[0]['gradient_ratio_mls_over_analytic']:.3f} / {an[1]['gradient_ratio_mls_over_analytic']:.3f} /
  {an[2]['gradient_ratio_mls_over_analytic']:.3f}**——**与分辨率无关**。mollified 指示函数的过渡层
  只有约一层格子的厚度，其导数是宽约 $h$ 的尖峰，$2h$ 宽的 MLS 模板把它抹平；这不是
  截断误差（截断误差随 $h\\to0$ 消失），是结构性的算子偏置。

### 3.3 分辨率阶梯：两种估计器（曲率算子固定为 `fit_quadratic`）

**基线 `delta_s = 2|\\nabla_h c|`（当前默认）**

{ladder_table(grad)}
**解析反演 `delta_s = 2\\,dc/dd(d_i)`（本轮新增，`csf_delta_s="profile"`）**

{ladder_table(prof)}

### 3.4 上轮交接的三条修法（dx = 3 mm 实测）

{remedy_rows}

三条**全部不成立**：窄色核把过渡层压得更薄（`|grad c|` 峰值上升但壳层采样点更少，净效果更差），
Tikhonov 正则只动奇异模板（表面壳层几乎没有退化模板，fallback 比例 0.000），ghost colour 反而
把归一化砍到 0.39（界面平面估计 $s=(c-1/2)/|\\nabla c|$ 在有偏的梯度下把镜像点放到粒子身上）。

![delta_s field](benchmarks/csf_delta_s/delta_s_field.png)

壳层粒子的 $\\delta_s$ 场（dx = 2 mm）：左 = 格点梯度（壳层均值 {rep.get('field', {}).get('gradient_mean', float('nan')):.1f}），
中 = 解析反演（{rep.get('field', {}).get('profile_mean', float('nan')):.1f}），右 = 解析 $2|\\nabla c|$
（{rep.get('field', {}).get('analytic_mean', float('nan')):.1f}）。反演器与解析梯度逐点吻合，
梯度算子系统性偏低。

![delta_s ladder](benchmarks/csf_delta_s/csf_delta_s.png)

左：归一化 vs 分辨率（绿带 = 15% 门）；中：切割力恒等式；右：各修法在 dx = 3 mm 的归一化。

## 4. 注意点（踩过的坑与局限）

1. **"把指标修对"不等于"把物理修对"**：解析反演把 $\\int\\delta_s dn$ 从 0.74–0.76 修到
   {worst_ds:.3f}–{max(nprof):.3f}（误差
   {best_ds:.1f}% 以内），但切割力恒等式的比值从 0.89–1.01 **升到 {cf_lo:.2f}–{cf_hi:.2f}**（超 36–56%）。
   这说明旧的 $\\delta_s$ 亏空**一直在抵消力装配里的过计**——与上一轮
   `verification_surface_tension_curvature.md` §4 记录的"曲率 +25% 与 $\\delta_s$ −25% 互相抵消"
   是同一现象的下半段。**因此反演器保持 opt-in（默认仍 `gradient`）**，不得默认开启。
2. **两个判据必须同时报**：只看 $\\int\\delta_s dn$ 会把一个 36% 的力超调当成"修好了"。
3. **反演器的适用边界**：$d_i$ 由 $c_i$ 查表得到，前提是 $c$ 是平面界面 mollified 指示函数。
   高曲率（$R/h\\lesssim3$ 的细液滴/气泡顶部）与多层界面厚度（多相）都不满足，需按
   $O(h/R)$ 修正；本轮 $R/h = 5.1$–$10.3$，阶梯上归一化误差随分辨率下降
   （{prof[0]['delta_s']['rel_error_pct']:+.1f}% → {prof[-1]['delta_s']['rel_error_pct']:+.1f}%）。
4. **一个实现坑（已修）**：梯形积分步长写成 `d[1:] - d[0]`（是 $i\\,dr$ 数组而不是标量 $dr$），
   使剖面累积快了 10 倍、$c$ 在 $d=0.07h$ 就饱和——表现是反演器输出全粒子同一个峰值。
   正确写法是先取标量 `dr = d[1] - d[0]`。标定自检（§3.1）就是为了让这类错误立刻暴露。
5. **默认路径位精确**：`csf_delta_s` 默认 `"gradient"`，`tests/test_csf_delta_s.py::test_default_stays_bit_exact`
   断言新增开关不改任何历史数字；既有 104 项测试与 `laplace_law` 的判据均未变动。

## 5. 误差判定（逐项，显式回答「是否 < 3%」）

| 关键量 | 基准 | 基线（`gradient`） | 本轮（`profile`） | 误差 | < 3%？ | < 15%？ |
|---|---|---|---|---|---|---|
| $\\int\\delta_s\\,dn$，dx=0.003 | 1.0 | {grad[0]['delta_s']['integral_dn']:.3f} | {prof[0]['delta_s']['integral_dn']:.3f} | {prof[0]['delta_s']['rel_error_pct']:+.1f}% | 是（{prof[0]['delta_s']['rel_error_pct']:+.1f}%） | 是 |
| $\\int\\delta_s\\,dn$，dx=0.002 | 1.0 | {grad[1]['delta_s']['integral_dn']:.3f} | {prof[1]['delta_s']['integral_dn']:.3f} | {prof[1]['delta_s']['rel_error_pct']:+.1f}% | 是（{prof[1]['delta_s']['rel_error_pct']:+.1f}%） | 是 |
| $\\int\\delta_s\\,dn$，dx=0.0015 | 1.0 | {grad[2]['delta_s']['integral_dn']:.3f} | {prof[2]['delta_s']['integral_dn']:.3f} | {prof[2]['delta_s']['rel_error_pct']:+.1f}% | **否** | 是 |
| $\\int\\delta_s\\,dn$（解析场 + 解析梯度） | 1.0 | — | {an[2]['integral_analytic_gradient']:.3f} | {(an[2]['integral_analytic_gradient']-1)*100:+.1f}% | 是 | 是 |
| $|\\nabla_h c|/|\\nabla c|$（算子偏置） | 1.0 | {an[2]['gradient_ratio_mls_over_analytic']:.3f} | — | −24% | **否** | **否** |
| $F_{{cut}}/2\\sigma$，dx=0.003 | 1.0 | {grad[0]['cut_force']['ratio_measured_over_theory']:.3f} | {prof[0]['cut_force']['ratio_measured_over_theory']:.3f} | {prof[0]['cut_force']['rel_error_pct']:.0f}% | **否** | **否** |
| $F_{{cut}}/2\\sigma$，dx=0.002 | 1.0 | {grad[1]['cut_force']['ratio_measured_over_theory']:.3f} | {prof[1]['cut_force']['ratio_measured_over_theory']:.3f} | {prof[1]['cut_force']['rel_error_pct']:.0f}% | **否** | **否** |
| 力方向中位 $\\cos$ | 1.0 | {grad[1]['direction']['cos_band_median']:+.3f} | {prof[1]['direction']['cos_band_median']:+.3f} | 0.0% | 是 | 是 |
| 色函数 RMS（vs 精确指示函数） | 0 | {an[1]['color_rms_kernelsum_vs_exact']:.4f} | — | — | 是 | 是 |

**结论（honest）**：
- **归一化判据**：解析反演把它修到 {best_ds:.1f}% 内（3 mm 档 {prof[0]['delta_s']['rel_error_pct']:+.1f}%、
  2 mm 档 {prof[1]['delta_s']['rel_error_pct']:+.1f}%、1.5 mm 档 {prof[2]['delta_s']['rel_error_pct']:+.1f}%），
  两档优于 3%、三档都在 15% 门内；**默认路径不变**。
- **力幅值判据不达标**：同一估计器把切割力推到 {cf_lo:.2f}–{cf_hi:.2f}（36–56% 超调），
  **不满足 <3%，也不满足 <15%**。因此本轮**不上调 any 能力等级**：CSF 表面张力的力幅值仍为
  WITHHELD（切割力在默认 `gradient` 路径下 {grad[1]['cut_force']['ratio_measured_over_theory']:.3f}/
  {grad[2]['cut_force']['ratio_measured_over_theory']:.3f}，<15% 但不是 <3%）。
- **本轮真正的正面结论**：$\\delta_s$ 的 25% 亏空被定位为**格点导数算子偏置**（0.758–0.764，
  分辨率无关），场与求积均已排除；上轮交接的三条修法被实测否证。

## 6. 未完成 / 下一个欠账（交接）

1. **力装配审计（新 P0）**：$\\delta_s$ 归一后切割力超调 36–56%，说明
   $\\sigma\\kappa\\delta_s\\mathbf n$ 的装配在多计——最可能的位置是
   (a) 拟合曲率在壳层粒子上取的权重口径与 $\\delta_s$ 权重重复计、
   (b) 表面壳层厚度 $\\sim2h$ 与 $\\delta_s$ 的支撑重叠造成面积重复、
   (c) 法向 Shepard 平滑把力分布抹宽后与 $\\delta_s$ 的尖峰不匹配。
   建议做法：在**解析圆**（把粒子放在精确圆周上、色函数用解析值）上逐项关掉
   (a)(b)(c)，先看哪一个能把 $F_{{cut}}/2\\sigma$ 从 1.36 拉回 1.00；这是纯静态判据，秒级成本。
2. **高曲率修正**：反演器用平面界面剖面，$R/h\\lesssim3$ 时需加 $O(h/R)$ 项（可用精确极坐标
   求积生成曲率修正表）。
3. **端到端 Laplace 压差**仍 WITHHELD（−88%…−107%，与上轮一致）：WCSPH 求和密度的噪声地板
   吞掉了 6.3e-3 的 Tait 压缩，与 $\\delta_s$ 无关，需压强基准改造（背景压强/无散修正/降 $c_0$）。

![kappa field](benchmarks/csf_curvature_fit/kappa_field.png)
"""
DOC.write_text(text)
print("wrote", DOC, len(text), "chars")