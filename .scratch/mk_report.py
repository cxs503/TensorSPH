# ruff: noqa: E501
# Markdown payload strings are emitted verbatim; wrapping them would change
# the layout of the generated tables and figure captions.
import base64
import io
import json
import os
import urllib.request
from pathlib import Path

HERE = Path("benchmarks/laplace_pressure_closure")
force = json.loads((HERE / "report_force_balance.json").read_text())
ladder = json.loads((HERE / "raw" / "relax_ladder.json").read_text())
dp = force["reference"]["analytic_dp_Pa"]

def png_b64(path, max_bytes=55000):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    for width in (1500, 1200, 1000, 900, 800, 700):
        im2 = im.resize((width, int(im.height * width / im.width)))
        buf = io.BytesIO()
        im2.save(buf, "PNG", optimize=True)
        if len(buf.getvalue()) <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode()
    return base64.b64encode(buf.getvalue()).decode()

LBL = {"closure_off": "closure off (Tait clamp)",
       "closure_on_all_donors": "closure on, donors=all",
       "closure_on_interior_layers2": "closure on, donors=interior, 2 layers",
       "closure_on_interior_layers4": "closure on, donors=interior, 4 layers"}

md = []
A = md.append
A("## 1. 案例介绍")
A("")
A(f"二维液滴在 CSF 表面张力模型下必须承载 Young–Laplace 压差 $\\Delta p=\\sigma/R={dp:.0f}$ Pa"
  f"（$\\sigma={force['reference']['sigma_N_per_m']:g}$ N/m，$R={force['reference']['radius_m']*1e3:.1f}$ mm，"
  f"$\\rho_0={force['reference']['rho0']:g}$ kg/m³，$c_0$=20 m/s，$h=1.3\\Delta x$）。")
A("")
A("历史欠账：`docs/verification_surface_tension_{laplace,consistent,curvature}.md` 的 −93.6 %/−75.6 %/−87.6 %，"
  "m2971 拆解出四个误差源并把**自由面压强闭合（C3）**列为唯一门槛项（残留 −34 %…−38 %），留下 P0："
  "实现 Adami/Ihmsen/Adams (2012, JCP 231:7057, §3.3) 的压强外推再测。本单完成该 P0，并证伪了一个竞争假设。")
A("")
A("## 2. 怎么做的")
A("")
A("* **实现**（`src/tensorsph/droplet_impact.py`）：新增 `Config.surface_pressure_closure`（默认 **off**，旧数字逐位不变）、"
  "`surface_pressure_threshold=0.99`、`surface_pressure_donors ∈ {interior, all}`、`surface_pressure_layers`。"
  "`rhs` 内以 `free_surface_pressure()` 替换自由面粒子的 Tait 压强："
  "$p_i=\\max(0,\\sum_j(p_j+\\rho_j\\mathbf g\\cdot\\mathbf r_{ji})W_{ij}/\\sum_j W_{ij})$。"
  "`interior` = 单侧外推（仅已知压强粒子当供体，按层迭代，线性场上精确）；`all` = 字面 Adami 公式（含自由面邻居的 p=0）。"
  "`rhs_all_pairs` 稠密路径同步接线，密度、Tait 律、CSF 力均未改动。")
A("* **静态力平衡诊断**（`analysis.py`，秒级，不时间推进）：由 $dp/dr=\\rho a_r$ 得 "
  "$\\Delta p=-\\int_0^R\\rho a_r\\,dr$，用**通量式**估计器（每个环带 $\\sum_p f_pV_p/(2\\pi r_c)$）避免环带平均的积分误差。")
A("* **松弛阶梯**（`relax_ladder.py`）：1.5 个毛细时间推进，读**绝对压强 + 时间平均**尾段（C1 修正口径）。")
A("* **测试**：`tests/test_free_surface_pressure_closure.py`（11 条闭式门）+ `tests/test_laplace_pressure_closure.py`（7 条证据锚定）。")
A("")
A("## 3. 计算结果")
A("")
A("### 3.1 表面张力力的积分（静态，两种分辨率）")
A("")
A("| $\\Delta x$ [mm] | $-\\int\\rho a_r dr$ [Pa] | /$(\\sigma/R)$ | 偏差 | 环带平均（错误）估计器 [Pa] |")
A("|---|---|---|---|---|")
for c in force["cases"]:
    fs = c["force_side"]
    A(f"| {c['dx_m']*1e3:.1f} | {fs['delta_p_force_Pa']:.1f} | {fs['delta_p_force_over_analytic']:.3f} | "
      f"{fs['deficit_pct']:+.1f} % | {fs['naive_shell_mean_estimator_Pa']:.1f} |")
A("")
sens = list(force["bin_sensitivity"].values())[0]
A("通量式估计器的分箱敏感性：" + "，".join(f"{k} 箱 → {v:.2f} Pa" for k, v in sens.items()) + "（<0.05 %）。")
A("")
A("### 3.2 松弛阶梯（$\\Delta x$=2.0 mm，$c_0$=20 m/s）")
A("")
A("| 变体 | 尾段 $\\langle p_{core}\\rangle$ [Pa] | 尾段误差 | 终态误差 | $v_{max}/v_{ref}$ | $\\rho_{core}/\\rho_0-1$ |")
A("|---|---|---|---|---|---|")
for c in ladder["cases"]:
    A(f"| {LBL.get(c['case'], c['case'])} | {c['tail_mean_Pa']:.1f} | {c['tail_rel_error_pct']:+.1f} % | "
      f"{c['final_abs_rel_error_pct']:+.1f} % | {c['vmax_over_vref']:.3f} | {c['rho_core_over_rho0']-1:+.2e} |")
A("")
A("## 4. 结果图")
A("")
A("图内三面板：(a) 累积通量积分 `-∫ρa_r` 在两种分辨率下都逼近 σ/R；(b) 各闭合变体的松弛读数与 ±15 % 门限；"
  "(c) 松弛后压强径向剖面 vs σ/R。")
A("")
A("## 5. 注意点")
A("")
A("1. **估计器坑（差点得出相反结论）**：首版把被积函数当作环带平均值乘带宽，得到 185–272 Pa（0.51–0.76 σ/R），"
   "且随分箱 100→800 漂移一倍——看起来像“力幅缺 25–50 %”。改用通量式后该集成在 0.05 % 内稳定在 0.997/0.925。"
   "本仓任何径向积分都应使用通量式并报告分箱宽度。")
A("2. **两种供体集是不同算子**：`all` 用自由面邻居的 p=0 稀释（稳定，v=0.020，优于基线 0.030）；"
   "`interior` 是真正的单侧外推（线性场精确，松弛尾段 +8.8 % 进入 15 % 带），但外层过压导致液滴“呼吸”，"
   "$v_{max}/v_{ref}=0.388$ ≫ Adami 0.1 判据，**不予采纳**。")
A("3. 本单松弛阶梯只跑了一种分辨率/一种声速（机器 load15 达 200–350，容量纪律下未加压）；"
   "静态诊断覆盖两分辨率。闭合本身的 $\\Delta x/c_0$ 阶梯是一条命令的下一步。")
A("4. 默认关闭：这是建模选择而非 bug 修复，默认打开会静默改变所有已发布算例数字。")
A("")
A("## 6. 误差判定")
A("")
A("| 量 | 结果 | 基准 | 误差 | 是否 <3 % |")
A("|---|---|---|---|---|")
for c in force["cases"]:
    fs = c["force_side"]
    A(f"| $-\\int\\rho a_r dr$（$\\Delta x$={c['dx_m']*1e3:.1f} mm） | {fs['delta_p_force_Pa']:.1f} Pa | σ/R={dp:.0f} Pa | "
      f"{fs['deficit_pct']:+.1f} % | {'是' if abs(fs['deficit_pct'])<3 else '否'} |")
for c in ladder["cases"]:
    A(f"| 松弛 $p_{{core}}$（{LBL.get(c['case'])}） | {c['tail_mean_Pa']:.1f} Pa | σ/R={dp:.0f} Pa | "
      f"{c['tail_rel_error_pct']:+.1f} % | {'是' if abs(c['tail_rel_error_pct'])<3 else '否'} |")
A("")
A("**结论（honest FAIL，不粉饰）**：Young–Laplace 能力保持 **WITHHELD**。最佳*稳定*变体为 −20.1 %（闭合 off 为 −37.9 %），"
  "改善 17.8 个百分点但仍在 15 % 门之外；+8.8 % 的变体因 $v_{max}/v_{ref}=0.388$ 判为不可采纳。"
  "**已确证**：残留 20 % 不在表面张力力幅上（力积分 0.997 σ/R，<3 % 达标），"
  "而在壳体压强力的离散化（$p/\\rho^2$ 配残缺 $\\rho$ 与界面壳厚度）——这是下一张单的 P0。")

fig = png_b64(HERE / "laplace_pressure_closure.png")
report = {
    "title": "TensorSPH 自由面压强闭合（Adami 2012 §3.3）：Laplace 欠账推进与残留 20% 归因",
    "summary": "实现并验证 wcSPH 自由面压强闭合（默认 off）：松弛读数由 -37.9% 改善到 -20.1%（稳定变体）；"
               "单侧变体 +8.8% 但虚假流超判据被否。静态通量积分证明离散 CSF 力可承载 0.997σ/R，"
               "残留欠账定位到壳体压力离散化而非力幅。11+7 条测试绿。Young-Laplace 仍 WITHHELD。",
    "scenario_key": "sph.surface_tension.laplace_pressure_closure",
    "mission_id": 3151,
    "tags": ["TensorSPH", "surface_tension", "Young-Laplace", "free_surface", "WCSPH", "verification", "WITHHELD"],
    "markdown": "\n".join(md),
    "figures": [{"name": "laplace_pressure_closure.png",
                 "caption": "(a) 通量积分 -∫ρa_r 逼近 σ/R；(b) 闭合变体松弛读数 vs ±15% 门；(c) 压强径向剖面",
                 "png_base64": fig}],
}
payload = json.dumps(report).encode()
print("payload KB", len(payload) / 1024)

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER")
pw = os.environ.get("TENSOR_AGENT_PASS")
tok = os.environ.get("TOKEN")
if not tok and user:
    req = urllib.request.Request(base + "/auth/login",
        data=json.dumps({"username": user, "password": pw}).encode(),
        headers={"Content-Type": "application/json"})
    tok = json.loads(urllib.request.urlopen(req, timeout=30).read()).get("access_token")
req = urllib.request.Request(base + "/tensor/reports", data=payload,
    headers={"Content-Type": "application/json", "Authorization": "Bearer %s" % tok})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=90).read())
    print("posted", json.dumps(resp)[:400])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:500])
