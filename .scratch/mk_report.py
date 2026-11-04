import base64, io, json, os, subprocess
from PIL import Image
src='benchmarks/laplace_pressure_closure/laplace_pressure_closure.png'
im=Image.open(src).convert('RGB')
im.thumbnail((1100,900))
buf=io.BytesIO(); im.save(buf,format='JPEG',quality=72)
b64=base64.b64encode(buf.getvalue()).decode()
print('figure bytes', len(buf.getvalue()), 'b64', len(b64))
rep=json.load(open('benchmarks/laplace_pressure_closure/report.json'))
runs=rep['runs']; ctrls=rep['controls']; lat=rep['lattice_audit']
rows="\n".join(f"| {r['dx_m']:g} | {r['sound_speed']:g} | {r['steps']} | {r['absolute_time_average_tail']['delta_p_Pa']:.1f} | {r['absolute_time_average_tail']['rel_error_pct']:+.1f} % | {r['force_audit']['delta_s_unit_integral']:.3f} | {r['force_audit']['virial_ratio']:.3f} | {r['force_audit']['kappa_ratio_to_1_over_R']:.2f} |" for r in runs)
crows="\n".join(f"| {c['dx_m']:g} | {c['sound_speed']:g} | {c['absolute_time_average_tail']['delta_p_Pa']:.1f} | {c['absolute_time_average_tail']['rel_error_pct']:+.1f} % | {c['force_audit']['delta_s_unit_integral']:.3f} |" for c in ctrls)
lrows="\n".join(f"| {l['dx_m']:g} | {l['n_particles']} | {l['scales']['1']['delta_s_unit_integral']:.3f} → {l['scales']['2']['delta_s_unit_integral']:.3f} | {l['scales']['1']['virial_ratio']:.3f} → {l['scales']['2']['virial_ratio']:.3f} |" for l in lat)
md=f"""# 自由面 CSF 表面 delta 归一化：Young–Laplace 压差的真正门槛项（TensorSPH m3298）

**结论**: 上一单（m2971）把端到端 Young–Laplace 压差 −34…−38 % 归因给「自由面压强闭合」并开出
shell-pressure 外推 P0。本单先测力预算再动手，实测表明 **P0 定位偏了一格**：稳态液滴径向力预算
`∫(ρa_csf)_r dr` 只有 `σ/R` 的 0.46–0.76 倍，**驱动力本身短一半**，压强梯度只是平衡到这个偏小的力上。
短掉的因子在 CSF 表面 delta 函数：`δ_s=|∇c|` 的单位积分实测 **0.889（格子上）**、**0.53（弛豫界面）**，
因为弥散界面的色函数剖面横跨自由面两侧，外侧一半（c<0.5）在真空中**没有粒子可采样**。
`Config.csf_delta_scale=2.0` 补齐外半剖面后，四个验收点全部进入 15 % 判据（最差 −11.0 %，
最好 −4.9 %），端到端压差自 WITHHELD 转为 **REAL（15 % 判据内；3 % 判据未达，原因已量化）**。

## ① 案例与几何
R=0.02 m 二维单相液滴（单主曲率 → Δp=σ/R=360 Pa），σ=7.2 N/m，ρ0=1000 kg/m³，γ=7，
h=1.3Δx，无重力，粘度 0.01 m²/s，六方格子 + 质量校准，观测 1.5 t_σ（t_σ=0.0333 s），
Δx∈{{0.002, 0.0015}} m × c₀∈{{20, 5}} m/s 四验收点 + 2 个 legacy 对照。

## ② 方法与判据
稳态径向平衡 `dp/dr=ρa_r, p(R)=0 ⇒ p_core=∫(ρa)_r dr`；把右端拆成 CSF 力与压强梯度两项分别积分。
两个**与压强场无关的精确恒等式**做独立基准：① `(1/2πR)ΣVδ_s=1`（delta 单位归一）；
② 半圆面 virial `−(1/2σ)Σ_{{x>0}}VF_x=1`（2σ，单位深度）。验收判据：4 点 |误差|<15 %，
`tests/test_laplace_pressure_closure.py`（17 passed，含 floor 的 legacy 钉住测试）。

## ③ 结果
**格子上的精确恒等式**
| Δx [m] | N | δ_s 单位积分（×1 → ×2） | virial 比（×1 → ×2） |
|---|---|---|---|
{lrows}

**闭合后（δ_s×2）**
| Δx [m] | c₀ [m/s] | 步数 | p_core(t-avg) [Pa] | 误差 | δ_s 积分 | virial 比 | κR |
|---|---|---|---|---|---|---|---|
{rows}

**legacy 对照（δ_s×1）**
| Δx [m] | c₀ [m/s] | p_core [Pa] | 误差 | δ_s 积分 |
|---|---|---|---|---|
{crows}

## ④ 图
laplace_pressure_closure.png：(a) 压强径向剖面 legacy vs 闭合 + σ/R 参考线；(b) 端到端误差柱状图 + ±15 % 带；
(c) 两个精确恒等式实测值；(d) 内压弛豫历史。

## ⑤ 注意点
1. 压强钳 p≥0 是**结果不是原因**；shell-pressure 外推会抹掉界面压强梯度，对端到端无益。
2. 2.0 来自 `c(n)+c(−n)=1` 的对称性（平面剖面精确补齐），不是调参：scale 阶梯 1.0/1.6/1.8/2.0 在
   Δx=0.002 上单调收敛 −37.9→−17.5→−12.6→−8.5 %。
3. 默认 `csf_delta_scale=1.0`，已记录结果逐位不变（毛细波/振荡液滴/撞池/静水壁/floor）；
   其余自由面基准的 legacy 口径是明确欠账（P1）。
4. 单相自由面前提：有壁/多相时界面两侧都有粒子，δ_s 不应放大。
5. 凸界面理想补齐因子 (1+h/R)/(1−h/R)≈2.30 > 2.0，所以 2.0 欠补齐、残差偏负——可检验预言。

## ⑥ 误差判定（是否 <3 %）
| 关键量 | 基准 | 实测 | 误差 | <3 % | <15 % |
|---|---|---|---|---|---|
| 压差（最差点 Δx=0.0015, c₀=5） | 360.0 Pa | 320.6 Pa | −11.0 % | 否 | 是 |
| 压差（最优点 Δx=0.0015, c₀=20） | 360.0 Pa | 342.3 Pa | −4.9 % | 否 | 是 |
| 表面张力力（virial，弛豫面） | 1.0 | 0.67–0.75 | −25…−33 % | 否 | 否 |
| δ_s 单位积分（×2，弛豫面） | 1.0 | 0.64–0.83 | −17…−36 % | 否 | 部分 |
| 曲率 κR | 1.0 | 0.92–1.72 | −8…+72 % | 否 | 是（均值 <1.4） |
| legacy 对照 | 360.0 Pa | 223.6/237.6 Pa | −37.9/−34.0 % | 否 | 否 |

**显式回答：端到端压差未达 3 %**（最好 −4.9 %，最差 −11.0 %），但已达 15 % 平台标准。
残差来源已量化：常数 2.0 相对凸界面理想因子 2.30 系统性欠补齐 13 %，与实测残差同量级；
下一步 P0' = 按局部界面几何的积分补齐（或 Akinci 2013 面积式），目标 <3 %。virial/δ_s 归一
在弛豫面上仍未进 15 %，说明界面弥散后剖面补齐比平面近似更缺，也是 P0' 的判据之一。未粉饰。

## ⑦ 工件
- `benchmarks/laplace_pressure_closure/{{run.py, make_doc.py, report.json, laplace_pressure_closure.png}}`
- `docs/verification_laplace_pressure_closure.md`（数字由 make_doc 注入）
- `tests/test_laplace_pressure_closure.py`（9 条）；`tests/test_laplace_pressure_floor.py` xfail 按设计改写为 legacy 钉住
- src 改动 3 处：`Config.csf_delta_scale`（默认 1.0）、`csf_consistent.surface_tension_acceleration(delta_scale)`、`surface_tension.compute_surface_tension_force` 透传
- 已知既有红：`tests/test_laplace_law.py::test_committed_verdicts_are_self_consistent`（committed 工件 `relax_dx0.0015.json` t_over_t_sigma=0.333<0.5，本单未触碰该工件，属既有欠账）
"""
body={"title":"Free-surface CSF surface-delta closure: the real gate on the Young-Laplace jump (TensorSPH m3298)",
 "summary":"Measured the radial momentum budget before implementing m2971's P0: the CSF surface-tension force itself is only 0.46-0.76 of sigma/R because |grad c| integrates to ~0.5 of unit at a free surface (the outer half of the diffuse colour profile lies in vacuum). Added Config.csf_delta_scale (default 1.0, legacy-preserving); with 2.0 all four acceptance points land inside the 15% criterion (-4.9%..-11.0%) vs -34%/-38% legacy. 3% not met; residual quantified as the planar-vs-convex completion factor 2.0 vs 2.30.",
 "scenario_key":"tensorsph_laplace_pressure_closure","mission_id":3298,
 "tags":["TensorSPH","surface_tension","Young-Laplace","verification","CSF"],
 "markdown":md,"figures":[{"name":"laplace_pressure_closure.png","caption":"(a) pressure profile legacy vs closed; (b) jump error vs +-15% band; (c) exact delta normalisation and half-disk virial; (d) relaxation history","png_base64":b64}]}
json.dump(body,open('/tmp/report_m3298.json','w'))
print('report json bytes',os.path.getsize('/tmp/report_m3298.json'))
