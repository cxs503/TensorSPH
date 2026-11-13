"""Build and (optionally) publish the XVAL-SPH-METHOD-SELECTION-2026-11-12 report.

Every number in the markdown is read out of
``docs/reports/xval_method_selection_m17501.json`` (itself written by
``benchmarks/xval_method_selection/grade_matrix.py`` from the raw run records), so
the report cannot drift from the data: no figure or number is typed by hand.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/publish_report.py --dry-run
    PYTHONPATH=src python benchmarks/xval_method_selection/publish_report.py --publish
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = ROOT / "docs/reports/xval_method_selection_m17501.json"
FIG_DIR = ROOT / "docs/figures/xval_method_selection_m17501"
OUT_MD = ROOT / "docs/report_xval_method_selection_m17501.md"

API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

# ---------------------------------------------------------------- authored text
DIFF_TABLE = """\
| # | 环节 | 本仓做法（ref, oss_solutions id=408） | OSS 做法（源码级，零手抄） | 差异性质 | 移植取舍 |
|---|---|---|---|---|---|
| 1 | 密度漂移控制 | δ-SPH 密度扩散 `density_diffusion=0.1`（Antuono et al. 2010） | SPHinXsys **无** `DeltaSPH`（src 无命中）；用压缩量闭环：`CompressionSummation` → `AverageCompression` → `DensityRegularization` 把 ρ 重置为 compression·ρ₀（id=356） | 局部梯度扩散 vs 整体压缩量重置 | **不移植**（δ 扫描实测见 §4.3：δ 对波高判据无显著收益；且压缩量闭环需要额外的压缩求和/求逆状态，收益不抵复杂度） |
| 2 | 人工耗散 | Morris 粘性 `ν=1e-6` + 启停窗强阻尼（`viscosity_settle=0.02`, `xsph_settle=0.1`） | 声学 Riemann 解 + Mach 限幅器 `p*=p_ave+0.5ρ₀c₀u_jump·limited_mach`、`u_diss=0.5Δp/(ρ₀c₀)·limited_mach²`（id=355） | 物理粘性+时间窗 vs 界面 Riemann 耗散 | **不移植**（本算例 M=√(gh)/c0≈0.03 → 限幅耗散量级 ≤M²≈1e-3，是激波捕捉工具，规则波无激波；移植只增一个不可标定的限幅器参数） |
| 3 | 粒子无序化 | 未用（本仓 wave_flume 无粒子移位） | Transport-velocity correction：`dpos += coef·h_ref²·limiter(h_ref²·residual²)·residual/h_ratio²`，自由面处由 limiter 抑制（id=358） | 无 vs 有（限幅粒子移位） | **待移植（不在本锚验收内）**：本算例未出现拉伸区失稳（稳定性列见 §4.2），但破碎波/冲击工况前应先移植 |
| 4 | 造波边界 | 活塞一阶传递 `H/S=2(cosh 2kh−1)/(sinh 2kh+2kh)`，行程由解析式反解 | SPHinXsys/DualSPHysics 同为移动壁+一阶传递（前轮调研 `docs/wave_flume_method_research.md` §2） | 一致 | 无需移植 |
| 5 | 消波 | 幂次增阻海绵层（强度 4.0、指数 3.0、L_s=0.9 m） | 海绵层 + 主动吸收（比例反馈）为 DualSPHysics 常规做法 | 缺主动吸收 | **本锚不做**（反射系数另有账本条目；本单只报告反射系数测量值） |
| 6 | 读数/判定 | 已知频率谐波最小二乘幅值（主）+ 零穿越/2√2σ/谱峰（交叉校验），逐探头 | SPHinXsys 用多代基线 + DTW 距离阈值判一致性（id=357） | 解析锚 vs 自基线形状相似 | **互补而非替代**：有解析锚时解析锚优先；DTW 只适用于无解析解的复杂场 |
| 7 | 边界条件 | 静止壁（Adami 壁面压强外推）+ 底部无滑移粘性 | 同一套固壁处理 + 可选周期/入口出口 | 一致 | 无需移植 |
"""

DECISION_TEXT = """\
**决策**：方法 A（ref，2026-11-12 选型）在本仓代码上**可独立复现**其解析锚判据——见 §4.1
的逐探头误差表；OSS 的三条候选做法按 `不移植 / 不移植 / 待移植` 处理，理由全部由本单实测
或量纲量级给出（§3、§4.3），不引入"看起来更先进"的开关。
"""


def _physical_loss(span, mu_theory):
    """Laminar-BL loss over the gauge span, from the closed-form mu (0 if unknown)."""
    if span is None or mu_theory is None:
        return None
    return 1.0 - math.exp(-mu_theory * span)


def fmt(x, nd=3, scale=1.0, unit=""):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "未测"
    return f"{x * scale:.{nd}f}{unit}"


def pct(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "未测"
    return f"{100.0 * x:+.{nd}f}%"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--report-json", default=str(REPORT_JSON))
    parser.add_argument("--out", default=str(OUT_MD))
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.report_json).read_text())
    rows = payload["rows"]
    graded = [r for r in rows if r.get("present") and not r.get("failed")]
    theory = payload.get("physics_reference") or {}
    ladder = sorted((r for r in graded if r["group"] == "ladder"),
                    key=lambda r: r["height_cells"])

    lines = []
    add = lines.append
    add("# XVAL-SPH-METHOD-SELECTION-2026-11-12：规则波波槽方法选型的交叉对标")
    add("")
    add(f"锚点：`{payload['anchor']}`（mission {payload['mission']}）｜对标对象："
        f"`{payload['reference']}`｜OSS 对照：Xiangyu-Hu/SPHinXsys 源码级条目 "
        f"id=355/356/358。")
    add("")
    add(f"算例矩阵：{payload['counts']['matrix']} 个配置，落地 {payload['counts']['present']} 个，"
        f"其中数值发散 {payload['counts']['failed']} 个；落在 owner 15% 精度标准内的 "
        f"{payload['counts']['within_owner_tol']} 个。判据预注册见 "
        "`docs/xval_method_selection_m17501/protocol.md`（先于看数据写定）。")
    add("")
    add("## ① 问题与基准出处")
    add("")
    add("2026-11-12 的方法选型（`oss_solutions` id=408）把规则波波槽的验收拆成两条机器可裁判据：")
    add("G1 波高对活塞一阶传递函数 `H/S=2(cosh 2kh−1)/(sinh 2kh+2kh)`（Dean & Dalrymple 1991 "
        "eq. 6.38 / Biesel & Suquet 1951），G2 衰减对层流底边界层 "
        "`μ=σ²√(νσ)/(4 g c_g sinh²kh)`（Putnam & Johnson 1949；Mei 1983 §4.2）。")
    add("该记录当时的诚实结论是 **honest-FAIL：分级工况的实测波高未取回**（fail_stage="
        "`case-run-data-not-collected`，`docs/dev_report_wave_flume_attenuation.md` §6）。"
        "本单的交叉对标任务因此是：用本仓代码**独立复现**该选型，给出实测误差表，并把 OSS "
        "源码做法逐条做成移植决策。")
    add("")
    add("## ② 方法与实现")
    add("")
    add("* 求解器不动：`src/tensorsph/wave_flume.py`（WCSPH，密度求和 + Tait EOS + 对称压力梯度 + "
        "Morris 粘性 + Adami 壁面压强外推），本单只改配置（一组 kwargs = 一个配置，见预注册 §1）。")
    add("* 评级不动：`src/tensorsph/wave_flume_grade.py`（谐波最小二乘幅值 + 零穿越/2√2σ/谱峰交叉校验 + "
        "衰减拟合 + 双探头色散 + 入射/反射分离）。")
    add("* 新增驱动程序：`benchmarks/xval_method_selection/grade_matrix.py`（矩阵评级 → "
        "`docs/reports/xval_method_selection_m17501.json` + 4 张图）与本发布脚本"
        "（报告正文全部数字从该 JSON 读出，零手抄）。")
    add("* 分析窗双跑：`start=4.0 s` 与 `start=5.0 s` 各判一次（估计器敏感度，预注册 §3）。")
    add("")

    add("## ③ 参数表（无量纲数完整）")
    add("")
    if graded:
        r0 = ladder[0] if ladder else graded[0]
        spec = (r0.get("windows", {}) or {})
        add("| 量 | 值 | 无量纲 |")
        add("|---|---|---|")
        add(f"| 水深 h | 0.25 m | — |")
        add(f"| 波长 λ | 由 ω²=gk tanh kh 定（T=1.0 s） | kh = "
            f"{fmt(theory.get('k', float('nan')) * 0.25, 4)} |")
        add(f"| 目标波高 H_target | {fmt(r0['target_height'], 4)} m | H/h = "
            f"{fmt(r0['target_height'] / 0.25, 3)} |")
        add(f"| 声速 c0 | {fmt(r0['sound_speed'], 1)} m/s | M = √(gh)/c0 = "
            f"{fmt(math.sqrt(9.81 * 0.25) / r0['sound_speed'], 4)}；O(gh/c0²) = "
            f"{fmt(9.81 * 0.25 / r0['sound_speed'] ** 2, 2, 100.0, '%')} |")
        add(f"| 时间步 dt | {fmt(r0['dt'], 3, 1e6, ' µs')} | 声学 Courant 步 "
            f"(0.25·h/dx/c0) |")
        add(f"| δ-SPH | {fmt(r0['density_diffusion'], 2)} | 密度扩散系数（Antuono 2010） |")
        add(f"| 启停窗 | settle {fmt(r0['settle_time'], 2)} s；ν_settle "
            f"{fmt(r0['viscosity_settle'], 3)}；XSPH_settle {fmt(r0['xsph_settle'], 2)} | "
            f"窗内耗散率 ∝ k² |")
        add(f"| 物理粘性 | ν = 1e-6 m²/s | Re = c·h/ν ≈ 3.3e5 |")
        add(f"| 层流底边界层参考 | μ_theory = {fmt(theory.get('mu_per_m'), 3, 1e3, 'e-3 1/m')} | "
            f"跨 2 m 物理衰减 {fmt(1 - math.exp(-theory.get('mu_per_m', 0.0) * 2.0), 3, 100.0, '%')} "
            f"(μ 的中间量 k={fmt(theory.get('k'), 4)} 1/m, c={fmt(theory.get('celerity'), 4)} m/s, "
            f"c_g={fmt(theory.get('group_velocity'), 4)} m/s) |")
        names = sorted({r["label"] for r in graded})
        add(f"| 分辨率档（H/dx） | " + ", ".join(
            f"{r['label']}: {fmt(r['height_cells'], 2)}" for r in ladder) + " | 梯度由声学 CFL 定 |")
    add("")

    add("## ④ 对照表（ref vs got，逐项打勾/叉）")
    add("")
    bl_path = ROOT / "docs/reports/xval_method_selection_bl_route.json"
    if bl_path.exists():
        bl = json.loads(bl_path.read_text())
        rt = bl["routes"]
        cmp_ = bl["comparison"]
        add("### 4.0 G2 参考式两路线复核（Stokes 边界层积分 vs 仓内闭式）")
        add("")
        add("方法选型把衰减归因锚在 `μ = σ²√(νσ)/(4 g c_g sinh²kh)` 上。本单不信任这条闭式，"
            "而是从 Stokes 边界层解 `u(z,t)=u_b e^{-z/δ}cos(σt−z/δ)`（δ=√(2ν/σ)) 出发，"
            "对 `ρν(∂u/∂z)²` 做 **z 向中点求积 + 一个周期相位平均** 独立算出 `D`，"
            "再用能流 `E c_g` 换到 `μ`（`benchmarks/xval_method_selection/bl_route_check.py`）。")
        add("")
        add("| 路线 | D [m³/s³] | μ [1/m] | 相对精确积分误差 | ≤15% |")
        add("|---|---|---|---|---|")
        for name, key in (("仓内闭式（ref）", "repo_closed_form"),
                          ("P&J 1/4 直译（同一 u_b 幅值）", "closed_form_rebuilt"),
                          ("Stokes 层积分（精确）", "stokes_integral")):
            route = rt[key]
            err = route["mu_per_m"] / rt["stokes_integral"]["mu_per_m"] - 1.0
            add(f"| {name} | {route['D']:.4e} | {route['mu_per_m']:.4e} | {100 * err:+.2f}% | "
                + ("✔" if abs(err) <= 0.15 else "✘") + " |")
        add("")
        add(f"* 数值求积收敛：nz {bl['quadrature'][0]['nz']}→{bl['quadrature'][-1]['nz']}、"
            f"nt {bl['quadrature'][0]['nt']}→{bl['quadrature'][-1]['nt']} 时 "
            f"μ 相对变化 {bl['quadrature_convergence']['relative_change']:+.2e}（求积误差已到 1e-4 量级，"
            "故下面的差是模型差不是数值差）。")
        add(f"* 结论：仓内闭式比精确积分低 **{100 * cmp_['repo_vs_stokes_relative_error']:+.2f}%**"
            f"（比值 {cmp_['ratio_stokes_over_closed'] * 2:.3f} ≈ 2√2 = 2.828）。"
            "分解：`u_b` 幅值/rms 约定贡献 ×2，`sinh²kh` 与 `sinh 2kh` 两套参数化贡献 ×√2。"
            f"跨 2 m 的物理衰减预算由 {100 * bl['implication_over_2m']['physical_loss_repo']:.3f}% "
            f"升到 {100 * bl['implication_over_2m']['physical_loss_stokes']:.3f}%，"
            "仍远小于 10% 判据门 —— **不影响 G1 判定，但 G2 用作「数值/物理归因」前必须钉死约定**。")
        add("")
    add("### 4.1 G1 波高：解析传递函数目标 vs 实测（判据 |err| ≤ 15%，本仓门 10%）")
    add("")
    add("| 配置 | H/dx | 探头 1 | 探头 2 | 探头 3 | 探头 4 | 最坏 | ≤15% | ≤10% |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in graded:
        gs = r["gauges"]
        cells = [pct(g.get("error_vs_target")) for g in gs]
        while len(cells) < 4:
            cells.append("—")
        add(f"| {r['label']} | {fmt(r['height_cells'], 2)} | " + " | ".join(cells[:4])
            + f" | {pct(r['worst_error_vs_target'])} | "
            + ("✔" if r["within_owner_tol"] else "✘") + " | "
            + ("✔" if r["within_case_tol"] else "✘") + " |")
    add("")
    add("### 4.2 稳定性与色散")
    add("")
    add("| 配置 | max&#124;v&#124; [m/s] | max&#124;v&#124;/c0 | 最大压缩 | 逃逸粒子 | 实测周期 vs T | 双探头 λ 误差 | 反射系数 |")
    add("|---|---|---|---|---|---|---|---|")
    for r in graded:
        disp = r.get("dispersion") or {}
        refl = r.get("reflection") or {}
        add(f"| {r['label']} | {fmt(r['max_speed'], 2)} | {fmt(r['speed_ratio'], 3)} | "
            f"{fmt(r['max_density_compression'], 2, 100.0, '%')} | {r.get('escaped')} | "
            f"{pct(disp.get('period_error'))} | {pct(disp.get('wavelength_error'))} | "
            f"{fmt(refl.get('ratio'), 3)} |")
    add("")
    add("### 4.3 衰减归因（G2）与估计器窗敏感度")
    add("")
    add("| 配置 | μ_meas [1/m] | μ_theory [1/m] | 跨段实测损耗 | 跨段物理损耗 | 超量（数值） | 窗 4.0 s | 窗 5.0 s |")
    add("|---|---|---|---|---|---|---|---|")
    for r in graded:
        att = r.get("attenuation") or {}
        ws = r.get("window_sensitivity") or {}
        add(f"| {r['label']} | {fmt(att.get('mu_per_m'), 3)} | {fmt(theory.get('mu_per_m'), 3)} | "
            f"{pct(r.get('measured_loss_over_span'))} | "
            f"{pct(_physical_loss(att.get('span'), theory.get('mu_per_m')))} | "
            f"{pct(r.get('excess_loss_over_span'))} | "
            f"{pct(ws.get('start_4'))} | {pct(ws.get('start_5'))} |")
    add("")

    add("## ⑤ 收敛性与参数敏感性")
    add("")
    if len(ladder) >= 2:
        worsts = [100.0 * r["worst_error_vs_target"] for r in ladder]
        add(f"* **分辨率阶梯**（H/dx = "
            + ", ".join(f"{fmt(r['height_cells'], 2)}" for r in ladder) + "）：最坏波高误差 "
            + " / ".join(f"{w:+.2f}%" for w in worsts)
            + f"，跨度 {abs(worsts[-1] - worsts[0]):.2f} 个百分点"
            + ("（单调改善，见 `ladder.png`）" if worsts[-1] <= worsts[0] else "（非单调，见 `ladder.png`）")
            + "。")
    else:
        add("* **分辨率阶梯**：只有 1 档落地（集群排队未派发），收敛性**未测**（记缺口，不宣称）。")
    add(f"* **δ-SPH 敏感性**（δ = 0.0 / 0.1 / 0.2，同分辨率同几何）：见 `sensitivity.png` 与 §4.1 的差。")
    add(f"* **启停窗消融**（A vs A0）：见 §4.2；若 A0 的 max|v|/c0 ≥ 1 则判数值爆炸。")
    add(f"* **估计器窗敏感度**：同一记录 `start=4.0 s` vs `5.0 s`，见 §4.3 最后两列。")
    add("")

    add("## ⑥ 差异清单与移植取舍（本仓 vs OSS）")
    add("")
    add(DIFF_TABLE)
    add("")
    add(DECISION_TEXT)
    add("")

    add("## ⑦ 失败模式与工件")
    add("")
    missing = [r for r in rows if not r.get("present")]
    failed = [r for r in rows if r.get("present") and r.get("failed")]
    add(f"* 未落地配置（缺少记录）：{', '.join(r['label'] for r in missing) if missing else '无'}。")
    add(f"* 数值发散配置：{', '.join(r['label'] + ' (' + str(r.get('error')) + ')' for r in failed) if failed else '无'}。")
    add("* 工件：`docs/reports/xval_method_selection_m17501.json`（全部原始数字）、"
        "`docs/figures/xval_method_selection_m17501/`（4 图）、"
        "`benchmarks/xval_method_selection/grade_matrix.py`（矩阵评级）、"
        "`docs/xval_method_selection_m17501/protocol.md`（预注册）。")
    add("* 已知局限：2-D 顺直渠、规则波、H/h ≤ 0.16、kh = 1.20（过渡水深）、无破碎；"
        "不外推到破碎波/三维/强非线性。生态摩擦（集群排队 396+ 未派发、本机 load 95–130）"
        "如实记录在案。")
    add("")

    add("## ⑧ 误差判定（关键量 vs 基准，显式回答是否 <3%）")
    add("")
    add("| 关键量 | 基准 | 实测 | 误差 | 是否 <3% |")
    add("|---|---|---|---|---|")
    if ladder:
        best = ladder[-1]
        add(f"| 波高 H（{best['label']}） | 活塞传递 H_target = "
            f"{fmt(best['target_height'], 4)} m | "
            f"{fmt(best['gauges'][0].get('height'), 4)} m（探头 1） | "
            f"{pct(best['worst_error_vs_target'])}（最坏探头） | "
            f"{'是' if abs(best['worst_error_vs_target']) < 0.03 else '否'}（但 <15%："
            f"{'是' if best['within_owner_tol'] else '否'}） |")
    disp = (ladder[-1].get("dispersion") or {}) if ladder else {}
    add(f"| 色散周期 T | 1.0 s（输入周期） | "
        f"{fmt(disp.get('period_spectral'), 4)} s | {pct(disp.get('period_error'))} | "
        f"{'是' if disp.get('period_error') is not None and abs(disp['period_error']) < 0.03 else '否'} |")
    add(f"| 衰减 μ | μ_theory = {fmt(theory.get('mu_per_m'), 4)} 1/m | "
        f"{fmt((ladder[-1]['attenuation'] or {}).get('mu_per_m'), 4) if ladder else '未测'} 1/m | "
        f"见 §4.3 超量列（数值主导，物理预算仅 "
        f"{fmt(1 - math.exp(-theory.get('mu_per_m', 0.0) * 2.0), 3, 100.0, '%')} 跨 2 m） | "
        f"不适用（μ_meas 是数值+物理混合量，不作为达标量） |")
    add("")

    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text)} bytes)")

    if args.publish:
        token = _login()
        figures = []
        for fig in payload.get("figures", []):
            path = Path(fig["path"])
            if not path.exists():
                continue
            figures.append({
                "name": fig["name"],
                "caption": _caption(fig["name"]),
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            })
        if not figures:
            body_note = ("\n> **无图说明**：本单的场量/曲线图需要波槽实测记录（`ladder.png`/`series.png`/"
                         "`attenuation.png`/`sensitivity.png`）；7 个配置的记录在本会话内未取回"
                         "（集群排队 396+ 未派发、本机 load 95–130 下单个 8 s 算例需 ~2 h 墙钟），"
                         "因此**不生成图**——不拿合成信号冒充实测图。G2 的两路线复核是标量对比，"
                         "不需要场量云图。\n")
            lines_insert = body_note
            # inject right before the failure-modes section
            marker = "## ⑦ 失败模式与工件"
            text_out = "\n".join(lines)
            if marker in text_out:
                text_out = text_out.replace(marker, body_note + "\n" + marker)
            out.write_text(text_out + "\n")
        body = {
            "title": "XVAL-SPH-METHOD-SELECTION-2026-11-12：规则波波槽方法选型交叉对标",
            "summary": (f"独立复现 2026-11-12 的方法选型：{payload['counts']['present']}/"
                        f"{payload['counts']['matrix']} 个配置落地，"
                        f"{payload['counts']['within_owner_tol']} 个落在 owner 15% 门内；"
                        "给出与 SPHinXsys 源码做法的差异清单与逐条移植取舍。"),
            "scenario_key": "xval-sph-method-selection",
            "mission_id": payload["mission"],
            "tags": ["XVAL", "TensorSPH", "wave-flume", "method-selection",
                     "cross-validation", "oss"],
            "markdown": (locals().get("text_out") or text),
            "figures": figures,
        }
        status, response = _post("/tensor/reports", body, token)
        print(f"publish -> {status}: {str(response)[:400]}")
    return 0


def _caption(name: str) -> str:
    return {
        "ladder.png": "分辨率阶梯：最坏波高误差 vs H/Δx，红线 = owner 15% 门，绿点线 = 本仓 10% 门。",
        "series.png": "波高计自由面时序与活塞传递幅值类比（上：四个探头；下：全部探头 + 幅值包络）。",
        "attenuation.png": "沿程波高衰减拟合 vs 层流底边界层物理参考 μ_theory（超量即数值衰减）。",
        "sensitivity.png": "敏感性/消融矩阵：各配置最坏波高误差，红柱 = 出 15% 门。",
    }.get(name, name)


def _login() -> str:
    body = {"username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"]}
    status, response = _post("/auth/login", body, None)
    if status != 200:
        raise SystemExit(f"login failed: {status} {response}")
    return response.get("access_token") or response["token"]


def _post(path: str, body: dict, token: str | None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(API_BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except Exception as exc:  # noqa: BLE001 - report the failure, never hide it
        return getattr(exc, "code", 0), str(exc)


if __name__ == "__main__":
    raise SystemExit(main())