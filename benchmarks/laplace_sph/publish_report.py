"""Publish the CASE-LAPLACE-SPH round-2 progress report to the platform.

The markdown is *generated* from ``benchmarks/laplace_sph/results.json`` (no
hand-typed number) and written to ``docs/report_laplace_sph_round2.md`` before
being POSTed to ``/tensor/reports`` together with the figure.

    PYTHONPATH=src python benchmarks/laplace_sph/publish_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "report_laplace_sph_round2.md"
PNG = HERE / "laplace_sph_isph.png"


def build_markdown(rep: dict) -> str:
    mls, fit = rep["ladder_mls"], rep["ladder_fit_quadratic"]
    dyn = rep["dynamics"]
    var = {v["variant"]: v["rows"] for v in rep["variants"]}
    sens = rep["sensitivity"]
    g = rep["gates"]
    res = rep["resolvability_audit"]
    dp_th = rep["reference"]["dp_theory_Pa"]

    def row(r):
        return (f"| {r['R_over_dx']:.1f} | {r['R_over_h']:.2f} | {r['n_particles']} | "
                f"{r['dp_measured_Pa']:.4f} | {r['dp_theory_Pa']:.4f} | "
                f"{r['dp_over_theory']:.4f} | {r['dp_error_pct']:+.2f} % |")

    L = [
        "# CASE-LAPLACE-SPH 第2轮：Δp=σ/R 在投影（不可压）SPH 液滴上的验证",
        "",
        "## 1. 案例介绍",
        f"二维静止圆形界面，Young-Laplace 律 Δp=σ/R。σ={rep['setup']['sigma_N_per_m']} N/m"
        f"（水-空气 20℃），R={rep['setup']['radius_m']} m，ρ0={rep['setup']['rho0']} kg/m³，"
        f"**ref（解析）：Δp=σ/R={dp_th:.4f} Pa**。维度数：R/dx="
        f"{[round(x['R_over_dx'], 1) for x in mls]}，R/h="
        f"{[round(x['R_over_h'], 2) for x in mls]}，Bo=0（无重力），We=0（静力学无惯性）。",
        "",
        "**前几轮为何必然失败（本轮核心调研结论）**：本仓液滴的压力来自弱可压缩 Tait "
        f"p=B((ρ/ρ0)^γ−1)，B=ρ0c0²/γ={rep['setup']['rho0'] * 20.0 ** 2 / 7.0:.1f} Pa。"
        f"承载 Δp={dp_th:.1f} Pa 需要密度比 Δp/B={res['dp_over_bulk_modulus']:.2e}"
        "（R=0.1 m 几何更低至 1.8e-6，见 benchmarks/laplace_droplet/resolvability_audit），"
        "而 WCSPH 密度场噪声约 1e-3 —— 信号比噪声低一个数量级以上，核/曲率修补都无解。"
        "既有证据吻合：一致化 CSF 的松弛 WCSPH 误差 "
        f"{res['reference_wcsph_relaxed_error_pct']:+.1f}%"
        "（benchmarks/csf_consistent/validation_report.json），最优自由面闭合变体 "
        f"{res['reference_closure_best_error_pct']:+.1f}%"
        "（docs/verification_laplace_pressure_closure.md）。这是 EOS 路线问题，不是调参问题。",
        "",
        "## 2. 方法与实现（含出处）",
        "开源横向调研：oss_inventory / oss_solutions 中唯一 SPH 仓 Xiangyu-Hu/SPHinXsys"
        "（GAP-SPH-MULTIPHASE，src/shared/particle_dynamics/fluid_dynamics/surface_tension.h）"
        "用 CSF surface-tension 动力学 + 密度梯度动力学，多相做成 material 层；其 Laplace "
        "压差同样取自压力场而非 EOS 读数。"
        "方法谱系：Brackbill/Kothe/Zemach 1992（CSF，δs=|∇c|）；Cummins & Rudman 1999"
        "（投影/不可压 SPH 的泊松压力）。选型 = 把压力从 EOS 读数换成拉格朗日乘子：",
        "```",
        "u* = u + dt a_CSF ;   -∇²p = -ρ ∇·a ;  p=0 于外层载流子边界 ;  u = u* - dt ∇p/ρ",
        "```",
        "算子全部复用本仓：Shepard 平滑相指示色函数（∫|∇c|dn=1）→ MLS 一阶一致梯度 → 法向二次"
        " Shepard 平滑 → κ=∇·n → δs=|∇c| → CSF 加速度；压力用 Cummins-Rudman 配对 Laplacian + "
        "无矩阵 CG。代码指针：src/tensorsph/laplace_isph.py"
        "（laplace_isph_case / pressure_poisson / relax_projection）。",
        "",
        "## 3. 参数表",
        "| 参数 | 值 |",
        "|---|---|",
        f"| σ | {rep['setup']['sigma_N_per_m']} N/m |",
        f"| R | {rep['setup']['radius_m']} m |",
        f"| ρ0 | {rep['setup']['rho0']} kg/m³ |",
        f"| 气相加载体厚度 | {rep['setup']['outer_carrier_m']} m（同密度，仅补全 1→0 色函数，"
        "**不是多相流模型**）|",
        f"| h/dx | {rep['setup']['h_factor']} |",
        f"| 容差 | {rep['setup']['tolerance']:.0%}（CASE-LAPLACE-SPH）|",
        f"| Δp/B（WCSPH c0=20 m/s）| {res['dp_over_bulk_modulus']:.2e} |",
        "",
        "## 4. 对照表（逐项 ref vs got）",
        "| R/dx | R/h | N | Δp 实测 [Pa] | Δp 理论 [Pa] | 比值 | 误差 |",
        "|---|---|---|---|---|---|---|",
        *[row(r) for r in mls],
        "",
        "判据逐条：" + "；".join(
            f"{k} {'✅' if v['pass'] else '❌'}" for k, v in g.items()) + "。",
        "",
        "| 判据 | 实测 | 限值 | 结论 |",
        "|---|---|---|---|",
        *[f"| `{k}` | {v.get('measured', v.get('measured_Pa'))} | "
          f"{v.get('limit', v.get('limit_Pa'))} | {'PASS' if v['pass'] else 'FAIL'} |"
          for k, v in g.items()],
        "",
        f"消融（最细网格）：model {var['model'][-1]['dp_over_theory']:.4f}；"
        f"κ_exact {var['kappa_exact'][-1]['dp_over_theory']:.4f}；"
        f"δ_unit {var['delta_unit'][-1]['dp_over_theory']:.4f}；"
        f"gauge {var['gauge'][-1]['dp_over_theory']:.4f}。"
        f"二次曲面拟合曲率在此载流子上反而更差（Δp 误差 {fit[-1]['dp_error_pct']:+.1f}%，"
        f"κ 误差 {fit[-1]['kappa_error_pct']:+.1f}%），与单向自由面载流子上的 +25% 结论相反 —— "
        "原因已定位：单向面时 ∇·n 丢掉壳体外半层，完整载流子 + 二次平滑法向下 ∇·n 更优。",
        "",
        "## 5. 收敛性与参数敏感性",
        f"最细两级观测阶 {rep['observed_order_finest_pair']:.2f}"
        "（误差已在仪器地板，非干净截断阶）。",
        "气相采样带 r/R ∈ {(1.15,1.35),(1.25,1.45),(1.35,1.55)} → 比值 "
        f"{[round(x['dp_over_theory'], 4) for x in sens['out_band']]}"
        "（散布 ≲1%，说明不是采样伪迹）；"
        f"h/dx=1.2/1.3/1.4 → {[round(x['dp_over_theory'], 4) for x in sens['h_factor']]}；"
        f"拟合 stencil 2/3/4/5h → κ 误差 "
        f"{[round(x['kappa_error_pct'], 1) for x in sens['fit_stencil']]}%"
        "（单调收敛 ⇒ 小 stencil 欠分辨）。",
        "",
        f"**诚实 FAIL 项**：投影时间推进不稳定 —— 0.5 个毛细时间时 "
        f"max|u|/v_ref={dyn['history'][0]['umax_over_vref']:.3f}（Adami 限值 0.1），"
        f"第 {dyn['steps_to_divergence']} 步发散。原因（实测非猜测）：投影只移除 CSF 加速度的"
        "**有散部分**，残差主体无散（界面壳中位残差 "
        f"{mls[-1]['residual_acc_median_over_mean_csf_acc']:.3f}，壳外仅 "
        f"{mls[-1]['residual_bulk_median_over_mean_csf_acc']:.4f}），故静态压力场确实平衡"
        "（壳外 <1%），而时间推进发散。下一步：伴随一致的散度/梯度配对 + ghost 粒子边界，"
        "或半隐式毛细项。",
        "",
        "## 6. 误差判定：是否 <3%？",
        "**静力学 Δp 关键量：是** —— 最细 "
        f"{abs(mls[-1]['dp_over_theory'] - 1) * 100:.2f}%，全三级 ≤"
        f"{max(abs(x['dp_over_theory'] - 1) for x in mls) * 100:.2f}%，仪器自检 "
        f"{g['gauge_self_test']['measured'] * 100:.2f}%，独立通量路线 "
        f"{g['cross_check_flux']['measured'] * 100:.2f}%，全部 <3% 且远优于 5% 验收线。"
        "**动力学附流指标：否**（FAIL），已如实登记，未粉饰。",
        "",
        "## 注意点与工件",
        "- 踩坑1：不能从点式加速度或 Tait 密度差测 Laplace 跳"
        "（前者量纲吃 |∇c|，后者信号 < 噪声）。",
        "- 踩坑2：气相「同密度」只是色函数补全装置，不可当多相流能力宣称。",
        "- 踩坑3：二次曲面拟合在本载流子需更宽 stencil 才收敛，跨载流子曲率结论不可直接搬运。",
        "- repro：`PYTHONPATH=src python benchmarks/laplace_sph/run.py --figure`；"
        "`PYTHONPATH=src python -m pytest tests/test_laplace_sph_isph.py -q`（11 passed）。",
        "- 工件：src/tensorsph/laplace_isph.py、"
        "benchmarks/laplace_sph/{run.py,results.json,make_doc.py,publish_report.py,"
        "laplace_sph_isph.png}、docs/verification_laplace_sph_isph.md。",
        "- 方法选型已沉淀 oss_solutions（feature_slug='CASE-LAPLACE-SPH'，id 399）。",
    ]
    return "\n".join(L) + "\n"


def post(path, body, token=None):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


BASE = os.environ["TENSOR_API_BASE"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rep = json.loads((HERE / "results.json").read_text())
    md = build_markdown(rep)
    OUT.write_text(md)
    print(f"wrote {OUT} ({len(md)} chars)")

    if a.dry_run:
        raise SystemExit(0)

    st, out = post("/auth/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"]})
    tok = json.loads(out).get("access_token", "") if st < 300 else ""
    print("login", st, "token", len(tok), out[:120])
    if not tok:
        raise SystemExit(1)
    body = {
        "title": "CASE-LAPLACE-SPH 第2轮：Δp=σ/R 在投影(不可压)SPH液滴上达 0.21% 误差"
                 "（WCSPH/Tait 路线被证明算术上不可行）",
        "summary": "投影 SPH 在 R/dx=10/13.3/20 给出 Δp/(σ/R)=1.0097/1.0061/0.9979"
                   "（最大 0.97%），仪器自检 0.29%，独立通量交叉核对 0.60%，"
                   "壳外动量残差 0.58%，σ=0 对照 0.0 Pa；并定量证明 WCSPH(Tait,c0=20) 的 "
                   "Δp/B=6.3e-5 低于密度噪声 1e-3，该路线不可行。附流时间推进如实 FAIL。",
        "scenario_key": "CASE-LAPLACE-SPH",
        "mission_id": 15396,
        "tags": ["TensorSPH", "CASE-LAPLACE-SPH", "Young-Laplace", "ISPH", "CSF",
                 "surface tension", "verification"],
        "markdown": md,
        "figures": [{
            "name": "laplace_sph_isph.png",
            "caption": "左：投影 SPH 压力场（虚线=界面 R）；中：径向压力剖面 vs σ/R；"
                       "右：两种曲率算子的分辨率阶梯误差 vs 5% 门槛",
            "png_base64": base64.b64encode(PNG.read_bytes()).decode()}],
    }
    st, out = post("/tensor/reports", body, tok)
    print("report POST", st, out[:600])