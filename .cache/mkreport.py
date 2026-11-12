import json,base64,math,os,urllib.request
from pathlib import Path
art=Path('artifacts/case_sph_wedge_entry_2d')
s=json.loads((art/'summary.json').read_text()); r=json.loads((art/'refs_derived.json').read_text())
w=r['wagner']; rows=s['rows']; c2=[x for x in rows if abs(x['dx']-2e-3)<1e-12][0]
def f(v,n=4): return 'n/a' if not isinstance(v,float) or not math.isfinite(v) else f'{v:.{n}g}'
def pc(v,n=1): return 'n/a' if not isinstance(v,float) or not math.isfinite(v) else f'{100*v:+.{n}f}%'
tab="| case | dx [mm] | Cp band peak | Δ | Cp@τ_peak | Δ | apex Cp | Δ | max ρ/ρ₀−1 | 判决 |\n|---|---|---|---|---|---|---|---|---|---|\n"
bud="| case | max ρ/ρ₀ | c₀/V | Cp 密度噪声地板 | Cp 实测 | 参考 |\n|---|---|---|---|---|---|\n"
for x in rows:
    tab+=f"| `{x['case_id']}` | {x['dx']*1e3:g} | {f(x['cp_band_peak'])} | {pc(x['err_band_peak'])} | {f(x['cp_at_peak_tau'])} | {pc(x['err_at_peak_tau'])} | {f(x['cp_centre_mean'])} | {pc(x['err_centre_mean'])} | {pc(x['max_rho_ratio']-1,2)} | {x['verdict_band_peak']} |\n"
    bud+=f"| `{x['case_id']}` | {f(x['max_rho_ratio'],5)} | {x['c0_over_V']:g} | {f(x['cp_density_budget'])} | {f(x['cp_band_peak'])} | {f(x['cp_max_reference'])} |\n"
md=f"""## ① 案例介绍
二维对称刚性楔形体入水砰击：deadrise β=30°，恒速 V=1 m/s，静水，忽略重力（自相似，唯一长度尺度 V·t）。判据量=Cp=p/(0.5ρV²) 的湿表面峰值。参考=Zhao & Faltinsen (1993) JFM 246:593（DOI 10.1017/S002211209300028X），容差 <15%。
**参考可得性**：Crossref 核验 246:593-612 / Zhao R. 第一作者 + Faltinsen O.；OpenAlex 核验 open_access=false、oa_status=closed、无仓储全文；已取出版商摘要（非线性 BEM+射流近似；4°–81° 楔形相似解；小 deadrise 下 Wagner 渐近解与相似解一致）。**β=30° 原文表/图数值不可得 → WITHHELD**，不以记忆/二手值冒充。
主判据锚=同一理想化 Wagner 闭式（仓内现算）：C={f(w['C'],6)}，Cp_apex=C={f(w['cp_centre'],6)}，Cp_max=C²+1={f(w['cp_max'],6)}，τ_peak={f(s['tau_peak'],5)}。

## ② 方法与实现
复用 `src/tensorsph/wedge_entry.py::WedgeImpact`（弱可压 SPH，Tait γ=7，B=ρ₀c₀²/γ；Monaghan 人工粘性；Adami 2012 壁面压力外推；每步密度求和与邻域重建；格点标定质量）。新增仅为驱动+判据+工件：`benchmarks/case_sph_wedge_entry_2d/{run.py,refs.json,make_figures.py,make_doc.py}`（`--stage matrix --shard i/n` / `accept`）；文档与图只读工件 JSON，零手抄。
复现：`python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix --shard 0/5 --out-dir artifacts/case_sph_wedge_entry_2d && python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept && python benchmarks/case_sph_wedge_entry_2d/make_figures.py && python benchmarks/case_sph_wedge_entry_2d/make_doc.py`

## ③ 参数表（无量纲完整）
β=30°；V=1 m/s；ρ₀=1000；c₀/V=20（Mach=0.05）；Tait γ=7；h/dx=1.3；声速 Courant 0.35；重力关闭（Fr→∞）；表面张力不建模（We→∞）；无粘+人工粘性 α=0.03（Re 不建模）；槽 0.40×0.12 m，观测窗 c_end=0.05 m；t_end=2c_end tanβ/(πV)=0.01838 s；帧数 13；C={f(w['C'],6)}；τ_peak={f(s['tau_peak'],5)}；每例 steps 见工件。

## ④ 计算结果（ref vs got，误差逐项）
{tab}
Wagner 参考：Cp_max={f(w['cp_max'],6)}，Cp_apex={f(w['cp_centre'],6)}。
**判决：对 Wagner 锚 ❌ FAIL（+281%/+234%）；对 Z&F(1993) ⏸ WITHHELD（原文表值不可得，不臆造）。**

## ⑤ 注意点（失败根因/局限）
{bud}
1) 弱可压 SPH 的压力由密度误差直接生成：Cp=2(c₀/V)²/γ·[(ρ/ρ₀)^γ−1]，是**与解正确性无关的噪声地板**；实测 Cp 与该地板同量级 → 误差主因是 c₀/V=20 下密度误差对压力的放大（灵敏度≈800·δρ/ρ），达 15% 需 δρ/ρ≲0.16%，实测 4.6%。
2) 指标互不等价：band peak 由 τ≈0.90 喷溅根部尖峰主导，Cp@τ_peak 反而偏低 ~49%，apex Cp 偏高 66–79%——只报一个数会掩盖这一点。
3) 收敛：dx 4→2 mm，band peak 32.01→28.05（方向正确），GCI(2/4mm)=0.177>15%；1 mm、0.5 mm 级与 V=5 m/s 控制点**未跑完**（控制面负载 128+；集群排队未派发），按停止规则如实记 missing，不插值外推。
4) Z&F 数值闭合条件写入 `refs.json: what_would_close_it`：取得论文文本后填页码/表号，`--stage accept` 自动裁决，无需改求解器。

## ⑥ 误差判定（显式回答是否 <3%）
| 量 | 参考 | got(dx=2mm) | 误差 | <3%? |
|---|---|---|---|---|
| Cp_max（τ≤1 系综峰值） | {f(w['cp_max'])} | {f(c2['cp_band_peak'])} | {pc(c2['err_band_peak'])} | 否 |
| Cp@τ_peak | {f(w['cp_max'])} | {f(c2['cp_at_peak_tau'])} | {pc(c2['err_at_peak_tau'])} | 否 |
| Cp_apex | {f(w['cp_centre'])} | {f(c2['cp_centre_mean'])} | {pc(c2['err_centre_mean'])} | 否 |
| Z&F β=30° | WITHHELD | — | — | 不可判（参考不可得，如实记） |

工件：`artifacts/case_sph_wedge_entry_2d/` 下 summary.json、table_cp.csv、refs_derived.json、两例工件 JSON、fig1–fig4.png；`docs/case_sph_wedge_entry_2d.md`（程序化生成）；`tests/test_case_sph_wedge_entry_2d.py` 13 passed（含判据器负控 +50% 必 FAIL）。
"""
figs=[]
for name,cap in [('fig1_profile_vs_wagner.png','壁面 Cp(τ) 剖面 vs Wagner（dx=4/2 mm）'),('fig3_grid_convergence.png','峰值 Cp 网格收敛与相对误差')]:
    figs.append({'name':name,'caption':cap,'png_base64':base64.b64encode((art/name).read_bytes()).decode()})
rep={'title':'CASE-SPH-WEDGE-ENTRY-2D：二维楔形体入水砰击压力（TensorSPH，第 1 轮）',
 'summary':'β=30° 恒速入水 WCSPH：dx=4/2 mm 峰值 Cp 32.01/28.05 vs Wagner 8.40（+281%/+234%，FAIL），顶点 Cp 偏高 66-79%；误差与 c₀/V=20 的密度噪声地板同量级（ρ 偏移 4.6%）；Z&F(1993) 30° 表值闭源不可得，登记 WITHHELD 不臆造。',
 'scenario_key':'CASE-SPH-WEDGE-ENTRY-2D','mission_id':16210,
 'tags':['TensorSPH','楔形体入水','砰击压力','Zhao-Faltinsen-1993','Wagner','算例池','honest-FAIL'],
 'markdown':md,'figures':figs}
base=os.environ.get('TENSOR_API_BASE','http://127.0.0.1:8010/api/v1')
def post(url,payload,token=None):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json',**({'Authorization':'Bearer '+token} if token else {})})
    with urllib.request.urlopen(req,timeout=30) as r: return r.status, r.read()[:300]
st,b=post(base+'/auth/login',{'username':os.environ['TENSOR_AGENT_USER'],'password':os.environ['TENSOR_AGENT_PASS']})
tok=json.loads(b.decode() if isinstance(b,bytes) else b)
tok=tok.get('access_token') or tok.get('token')
print('login',st,bool(tok))
st2,b2=post(base+'/tensor/reports',rep,tok)
print('report',st2,b2[:200])
