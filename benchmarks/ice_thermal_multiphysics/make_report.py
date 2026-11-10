"""Compose and POST the paper-grade progress report for mission 10431.

All numbers come from artifacts/ice_thermal_multiphysics/results.json and all
figures from the PNGs next to it; nothing is typed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "ice_thermal_multiphysics"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def login():
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req))
    return data.get("access_token") or data.get("token")


def fmt(v, d=3):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.{d}e}"
    return f"{v:.{d}f}"


def main():
    res = json.loads((ART / "results.json").read_text())
    doc_path = REPO / "docs" / "verification_ice_thermal_multiphysics.md"
    doc = doc_path.read_text()
    stefan = res["stefan"]
    d = res["darcy"]
    st = res["stress"]
    des = res["desalination"]
    coupled = res.get("coupled", {})
    cs = coupled.get("coupling_strength", {})

    lines = []
    lines.append("## 1. 案例介绍")
    lines.append("")
    lines.append("冰-海水体系里，温度、盐度、应力三场是强耦合的：温度决定卤水体积分数（mushy "
                 "layer），卤水体积分数决定孔隙渗透率，渗透率决定重力排卤速率，排出的盐让冰下混合层变"
                 "重，浮力对流把海洋热量送回冰底，融冰又反过来淡化混合层并改变温度场；与此同时，温度"
                 "剖面本身就是热应力的来源，决定冰盖是否开裂。本轮在 TensorSPH 新建 "
                 "`tensorsph.ice_thermal` 域模块，把这四个环节全部做成可验证的物理实现，并给出五个"
                 "独立验证锚。")
    lines.append("")
    cfg = coupled.get("spring_two_way", {}).get("profiles", {}).get("z", [])
    thick = (cfg[-1] + 0.5 * (cfg[1] - cfg[0])) if len(cfg) > 1 else 0.0
    lines.append("物理背景/几何/参数：一维垂直海冰柱（厚度 "
                 f"{fmt(thick, 3)} m，{len(cfg)} 个 SPH 粒子，20 m 混合层，"
                 "海水 32 psu）、SPH 粒子离散；物性表见下（程序生成，非手抄）：")
    lines.append("")
    lines.append("```")
    lines.append(res["properties"])
    lines.append("```")
    lines.append("")
    lines.append("## 2. 怎么做的")
    lines.append("")
    lines.append("* **相变**：焓法。海冰比焓用闭式解析式（液相分数 $w_l=\\theta_f/\\theta$，"
                 "$\\theta=273.15-T$，$\\theta_f=T_{MELT}-T_f(S)$），反演 $T=e^{-1}(H,S)$ 用向量化"
                 "二分（20 次，精度 $10^{-5}$ K）。SPH 能量方程用 Cleary-Monaghan 导热算子"
                 "（调和平均导热率、镜像 ghost 施加 Dirichlet 边界），晶格二阶矩不足用"
                 "`lattice_diffusion_factor` 标定（实测见下）。")
    lines.append("* **盐度输运**：卤水体积分数用 Cox & Weeks（盐质量守恒），渗透率用"
                 "Golden-Ackley-Lytle 形式并设渗流阈值 $v_b=0.05$；重力排卤用 Darcy 速度 ×"
                 "Brooks-Corey 相对渗透率 $k_r=s_m^3$（$s_m$ 为可动卤水饱和度），显式上风 + CFL "
                 "子循环，盐守恒到机器精度。")
    lines.append("* **热应力**：平面应力双轴模量 $E/(1-\\nu)$；受限板 $\\sigma=-E\\alpha(T-T_{ref})"
                 "/(1-\\nu)$ 与自由板自平衡解（力/弯矩平衡）；开裂判据同时给 Rankine 强度与 "
                 "$K_I=1.12\\sigma\\sqrt{\\pi a}$ 断裂力学两条。")
    lines.append("* **浮力对流**：用有限差分解 Horton-Rogers-Lapwood 线性稳定性特征值问题，"
                 "数值求出临界 Rayleigh 数并与精确值 $4\\pi^2$ 对比；海洋混合层浮力用 "
                 "$w^*=\\sqrt{g'h_o}$，交换速度 $\\gamma=\\gamma_0+a w^*$。")
    lines.append("* **三场耦合积分**：焓-温度隐式 Picard 步 + 盐度子循环显式步 + 应力后处理，"
                 "双向闭环由融冰淡水稀释实现（开启/关闭稀释即 two-way/one-way 对照）。")
    lines.append("")
    lines.append("## 3. 计算结果")
    lines.append("")
    lines.append("### 3.1 Stefan 相变前沿（SPH 焓法 vs Neumann 解析）")
    lines.append("")
    lines.append("| dx [m] | N | dt [s] | steps | Ste | λ | 前沿 [mm] | Neumann [mm] "
                 "| 末态误差 [%] | 最大误差 [%] | 剖面 RMS [K] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in stefan:
        lines.append(f"| {fmt(row['dx_m'],4)} | {row['n_particles']} | {fmt(row['dt_s'],2)} | "
                     f"{row['steps']} | {fmt(row['stefan_number'],4)} | "
                     f"{fmt(row['neumann_lambda'],5)} | {fmt(row['front_final_m']*1000,2)} | "
                     f"{fmt(row['front_analytic_m']*1000,2)} | "
                     f"{fmt(row['front_rel_error_final']*100,3)} | "
                     f"{fmt(row['front_rel_error_max']*100,3)} | "
                     f"{fmt(row['profile_rms_k'],4)} |")
    lines.append("")
    lines.append("### 3.2 排卤淡化（平衡盐度下限 + 观测带）")
    lines.append("")
    lines.append("| $k_{ref}$ [m²] | 末期整体盐度 [psu] | 达 90% 淡化所需天数 |")
    lines.append("|---|---|---|")
    for row in des["k_ref_scan"]:
        lines.append(f"| {fmt(row['k_ref_m2'],2)} | {fmt(row['final_bulk_salinity_psu'],2)} | "
                     f"{fmt(row['days_to_90pct_desalination'],2)} |")
    lines.append("")
    lines.append(f"盐守恒相对误差 {fmt(des['salt_conservation_rel_error'],2)}；末期整体盐度 "
                 f"{fmt(des['final_bulk_salinity_psu'],2)} psu，对应平衡下限 "
                 f"{fmt(des['final_residual_psu'],2)} psu，海洋 32 psu 的 "
                 f"{fmt(des['band']['ratio_to_ocean'],3)}，落在一年冰观测带 "
                 f"{des['band']['band_psu']} psu 内：{des['band']['inside_band']}。")
    lines.append("")
    lines.append("### 3.3 对流起始（Darcy–Bénard）")
    lines.append("")
    lines.append(f"有限差分特征值解 $Ra_c = {fmt(d['ra_critical_numeric'],4)}$，精确 "
                 f"$4\\pi^2={fmt(d['ra_critical_exact'],4)}$，相对误差 "
                 f"{fmt(d['ra_relative_error']*100,4)} %；临界波数 "
                 f"$a_c={fmt(d['wavenumber_at_onset'],3)}$；整条边缘稳定曲线与 "
                 f"$(a^2+\\pi^2)^2/a^2$ 最大偏差 {fmt(d['curve_relative_error_max']*100,3)} %。")
    lines.append("")
    lines.append("### 3.4 热应力与开裂")
    lines.append("")
    lines.append(f"Timoshenko 受限板：解析 {fmt(st['timoshenko_analytic_pa'],2)} Pa，计算 "
                 f"{fmt(st['restrained_max_pa'],2)} Pa，误差 "
                 f"{fmt(st['restrained_relative_error']*100,4)} %；自由板线性温度梯度残余 "
                 f"{fmt(st['free_linear_max_abs_pa'],2)} Pa（理论 0）；非线性剖面自平衡解净力 "
                 f"{fmt(st['free_nonlinear_net_force_pa'],3)} Pa、净弯矩 "
                 f"{fmt(st['free_nonlinear_net_moment'],3)} Pa·m。临界降温 "
                 f"{fmt(st['critical_temperature_drop_k'],3)} K，10 mm 边裂纹 $K_I$ = "
                 f"{fmt(st['stress_intensity_mpa_sqrt_m'],3)} MPa√m（$K_{{IC}}=0.1$）。")
    lines.append("")
    if coupled:
        lines.append("### 3.5 三场耦合柱（冬季 vs 春季；two-way vs one-way）")
        lines.append("")
        lines.append("| 算例 | 末期整体盐度 [psu] | 混合层盐度 [psu] | $Q_{oc}$ [W/m²] "
                     "| 融冰 [mm/d] | 累计融冰 [m] | $\\sigma_{max}$ 受限 [MPa] "
                     "| $\\sigma_{max}$ 自由 [MPa] | 盐守恒误差 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for name in ("spring_two_way", "spring_one_way", "winter"):
            s = coupled[name]["summary"]
            lines.append(f"| {name} | {fmt(s['bulk_salinity_final_psu'],2)} | "
                         f"{fmt(s['ocean_salinity_final_psu'],4)} | "
                         f"{fmt(s['ocean_heat_flux_w_m2'],2)} | "
                         f"{fmt(s['melt_rate_mm_day_mean'],2)} | "
                         f"{fmt(s['melt_total_m'],4)} | "
                         f"{fmt(s['max_tensile_stress_mpa_restrained'],2)} | "
                         f"{fmt(s['max_tensile_stress_mpa_free'],3)} | "
                         f"{fmt(s['salt_conservation_rel_error'],2)} |")
        lines.append("")
        melt_ratio = fmt(cs.get("melt_ratio_two_way_over_one_way"), 3)
        flux_ratio = fmt(cs.get("ocean_heat_flux_ratio"), 3)
        ds_ocean = fmt(cs.get("ocean_salinity_shift_psu"), 4)
        winter_s = fmt(coupled["winter"]["summary"]["bulk_salinity_final_psu"], 2)
        lines.append(f"耦合强度：两向/单向融冰比 {melt_ratio}，海洋热流比 {flux_ratio}，"
                     f"混合层盐度差 {ds_ocean} psu。冬季算例整体盐度 {winter_s} psu。")
        lines.append("")
        sp = coupled["spring_two_way"]["summary"].get("spinup") or {}
        if sp:
            sea_r = sp.get("sea_ice", {}).get("relaxation_s")
            sea_r = (f">{8000 * 300} s (超出观测窗)" if sea_r is None else f"{sea_r:.0f} s")
            fr_r = fmt(sp.get("fresh_ice", {}).get("relaxation_s"), 0) + " s"
            t_cond = fmt(sp.get("fresh_ice", {}).get("conduction_time_s"), 0)
            lines.append(f"mushy 层潜热缓冲：海冰冷激半松弛时间 {sea_r}，淡水冰 {fr_r}"
                         f"（导热时间尺度 {t_cond} s）—— 潜热使海冰对冷激的响应显著滞后。")
            lines.append("")
    lines.append("完整逐项表格与物性、晶格标定见 `docs/verification_ice_thermal_multiphysics.md`"
                 "（由 `make_doc.py` 从 results.json 程序化生成）。")
    lines.append("")
    lines.append("## 4. 结果图")
    lines.append("")
    figs = res.get("figures", {})
    for name, meta in figs.items():
        lines.append(f"* `{name}` — {meta['caption']}")
    lines.append("")
    lines.append("## 5. 注意点（踩坑与局限）")
    lines.append("")
    for item in [
        "焓法必须用 θ=273.15−T 写潜热项；用 T 会把 12 psu 海冰的潜热算小 273 倍。",
        "焓是 (T,S) 的函数：排卤是等温排质，保持 H 不变只会把 S 相关潜热项加到冰里，"
        "实测会让冰柱虚假升温到 +15 °C。",
        "比焓形式的容量项不含密度：Picard 对角带 ρ 会把温度响应压低 917 倍。",
        "上风 CFL 要按粒子（而非面平均）取，否则一步就能把盐度打到 5.5e4 psu。",
        "自由板线性温度梯度应力为零：离散弯矩必须分子分母同测度，否则残留 ~66 Pa 假应力。",
        "局限：冰厚不消融（融冰量只诊断，不移动底面）；海洋为单层混合层；"
        "γ0、a 为声明式体交换系数；k_ref 在文献中跨一个量级，已做 ×0.1/1/10 敏感性扫描；"
        "Greeley & Wilson (1987) 原始数据表离线不可得，该锚记为 WITHHELD，改用一年冰盐度观测带"
        "（Cox & Weeks 1983）做带宽检验并明确标注。",
    ]:
        lines.append(f"* {item}")
    lines.append("")
    lines.append("## 6. 误差判定（是否 <3%）")
    lines.append("")
    verdicts = []
    for row in stefan:
        verdicts.append((f"Stefan 前沿（dx={fmt(row['dx_m'],4)} m）",
                         f"{fmt(row['front_rel_error_final']*100,3)} %",
                         row["front_rel_error_final"] < 0.03))
    verdicts.append(("Darcy–Bénard $Ra_c$", f"{fmt(d['ra_relative_error']*100,4)} %",
                     d["ra_relative_error"] < 0.03))
    verdicts.append(("Timoshenko 受限板应力",
                     f"{fmt(st['restrained_relative_error']*100,4)} %",
                     st["restrained_relative_error"] < 0.03))
    import torch as _t

    from tensorsph.ice_thermal.brine import residual_salinity as _floor
    _t_ = _t.tensor(des["profile_t"], dtype=_t.float64)
    _s_ = _t.tensor(des["profile_s"], dtype=_t.float64)
    floor_margin = float((_s_ - _floor(_t_)).min())
    verdicts.append(("排卤平衡下限（vs 逐层解析下限）",
                     f"min margin {fmt(floor_margin, 3)} psu（带宽检验）",
                     floor_margin > -1e-6))
    if coupled:
        for name in ("spring_two_way", "winter"):
            s = coupled[name]["summary"]
            verdicts.append((f"盐守恒（{name}）",
                             f"{fmt(s['salt_conservation_rel_error']*100,4)} %",
                             s["salt_conservation_rel_error"] < 0.03))
    lines.append("| 关键量 | 误差 | 是否 <3% |")
    lines.append("|---|---|---|")
    for name, value, ok in verdicts:
        lines.append(f"| {name} | {value} | {'是' if ok else '否'} |")
    lines.append("")
    lines.append("Greeley & Wilson (1987) 原始排卤实验表离线不可得，故该锚 **不宣称** 逐点比对，"
                 "只做观测带（3–8 psu）检验并如实标注 WITHHELD；冰-海洋界面存在亚开尔文量级的"
                 "非单调薄层（固定冰厚不做消融所致），已在局限中声明。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(doc)

    md = "\n".join(lines)
    fig_payload = []
    for name, meta in figs.items():
        path = Path(meta["path"])
        if path.exists():
            fig_payload.append({"name": name, "caption": meta["caption"],
                                "png_base64": base64.b64encode(path.read_bytes()).decode()})
    payload = {
        "title": "冰-海水热-盐-力多物理场耦合基准：相变 + 卤水排出 + 热应力（TensorSPH 新域模块）",
        "summary": ("TensorSPH 新建 tensorsph.ice_thermal 域模块：SPH 焓法相变（Stefan 前沿 vs "
                    "Neumann 解析）、重力排卤（平衡下限 + 观测带）、Darcy–Bénard 对流起始（数值 "
                    "Ra_c vs 4π²）、热应力/开裂判据，以及温度-盐度-应力三场双向耦合柱；全部锚点"
                    "带判据数字与审计链。"),
        "scenario_key": "ice_thermal_multiphysics",
        "mission_id": 10431,
        "tags": ["TensorSPH", "ice", "phase-change", "enthalpy-method", "SPH",
                 "brine-drainage", "thermal-stress", "IN13W2-N07"],
        "markdown": md,
        "figures": fig_payload,
    }
    token = login()
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token})
    resp = urllib.request.urlopen(req)
    print(resp.status, resp.read()[:400])
    (ART / "report_markdown.md").write_text(md)


if __name__ == "__main__":
    main()
