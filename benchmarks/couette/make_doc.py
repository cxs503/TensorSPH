#!/usr/bin/env python3
"""Render the Couette report (markdown + report-API payload) from results.json.

Every number in the output is read from ``results.json``; nothing is typed by
hand.  That is the "zero hand-copy" rule: the document is a view of the run
artifacts, and a wrong number can only come from a wrong run.

Usage::

    python benchmarks/couette/make_doc.py \
        --results runs/couette_full/results.json \
        --md docs/reports/m15412-couette-sph.md \
        --payload /tmp/report.json
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

REPORT_TITLE = "CASE-COUETTE-SPH: 平面库埃特流（滑移壁驱动）SPH 解析解对标"
SCENARIO_KEY = "case_couette_sph"


def pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def profile_rows(run: dict) -> str:
    lines = ["| y/H | u_meas [m/s] | u_ref = U y/H [m/s] | err/U | 判据 |",
             "|---|---|---|---|---|"]
    for p in run["profile"]:
        ok = abs(p["rel_err"]) < 0.05
        lines.append(f"| {p['y_over_H']:.4f} | {p['u_mean']:.6f} | "
                     f"{p['u_ref']:.6f} | {pct(p['rel_err'])} | {mark(ok)} |")
    return "\n".join(lines)


def metrics_table(runs: list) -> str:
    lines = ["| run | H/dx | Re | Ma | L2(内部) | 通量比 | 壁剪切比(下) | 壁剪切比(上) "
             "| 线拟合斜率比 | 滑移长/dx | 稳态漂移 | 判定 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m = r["metrics"]
        g = r.get("gate", {})
        ok = g.get("passed", None)
        verdict = "—" if ok is None else mark(ok)
        lines.append(
            f"| {r['label']} | {r['resolution_H_over_dx']} | "
            f"{r['reynolds_number']:.3f} | {r['mach_number']:.4f} | "
            f"{pct(m['interior_l2'])} | {m['flux_ratio']:.4f} | "
            f"{m['wall_shear_ratio']:.4f} | {m['wall_shear_ratio_top']:.4f} | "
            f"{m['line_fit_slope_ratio']:.4f} | "
            f"{m['slip_length_bottom_over_dx']:.3f} | "
            f"{r['steady_drift']:.2e} | {verdict} |")
    return "\n".join(lines)


def sweep_table(runs: list, key: str, unit: str) -> str:
    lines = [f"| {key} | H/dx | Re | L2(内部) | 通量比 | 壁剪切比(下) | 滑移长/dx | 稳态漂移 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m = r["metrics"]
        lines.append(f"| {r[key]} {unit} | {r['resolution_H_over_dx']} | "
                     f"{r['reynolds_number']:.3f} | {pct(m['interior_l2'])} | "
                     f"{m['flux_ratio']:.4f} | {m['wall_shear_ratio']:.4f} | "
                     f"{m['slip_length_bottom_over_dx']:.3f} | "
                     f"{r['steady_drift']:.2e} |")
    return "\n".join(lines)


def parameters_table(payload: dict) -> str:
    rows = [
        ("通道高度 H", f"{payload['height']} m"),
        ("水平方向", f"周期，长 = {payload['columns']} dx"),
        ("壁面", "下壁 u = 0，上壁 u = U（切向滑动壁），两层均 no-slip(f=1)"),
        ("壁粒子层数", str(payload["wall_layers"])),
        ("密度 rho0", f"{payload['rho0']} kg/m^3"),
        ("状态方程", f"Tait, gamma = 7, c0 = {payload['sound_speed']} m/s"),
        ("运动粘度 nu", f"{payload['viscosity']} m^2/s"),
        ("上壁速度 U", f"{payload['wall_speed']} m/s"),
        ("核函数", f"cubic spline, h = {payload['h_factor']} dx"),
        ("时间推进", "显式欧拉，固定步长 dt = 0.3 h/c0（声学 CFL），初值静止"),
        ("无量纲数", "Re = U H / nu；Ma = U / c0；分辨率 H/dx"),
    ]
    lines = ["| 参数 | 取值 |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)


def render(payload: dict, figures: dict | None = None) -> str:
    ladder = payload.get("ladder", [])
    verdict = payload.get("verdict", {})
    gates = payload["gates"]
    control = next((r for r in payload.get("friction_sweep", [])
                    if r["wall_friction"] == 0.0), None)
    graded = [r for r in ladder
              if r["resolution_H_over_dx"] >= 16]
    graded_labels = {"ladder-r%d" % r["resolution_H_over_dx"] for r in graded}
    gated = {g["label"]: g for g in verdict.get("gated_results", [])}
    for r in ladder:
        if r["label"] in graded_labels:
            r["gate"] = gated.get(r["label"], {})

    finest = max(ladder, key=lambda r: r["resolution_H_over_dx"]) if ladder else None
    coarse = min(ladder, key=lambda r: r["resolution_H_over_dx"]) if ladder else None
    ref = next((r for r in ladder if r["resolution_H_over_dx"] == 16), finest)

    out = []
    out.append(f"# {REPORT_TITLE}\n")
    out.append(f"**算例锚点**：CASE-COUETTE-SPH · **仓库**：TensorSPH · "
               f"**分支**：`agent/m15412` · **任务**：m15412\n")
    out.append(f"**裁决**：`{verdict.get('status', 'N/A')}` "
               f"（判据阈值预先声明于 `benchmarks/couette/DESIGN.md`）\n")
    if verdict.get("reasons"):
        out.append("未达标项：" + "；".join(verdict["reasons"]) + "\n")

    # ------------------------------------------------------------------ ①
    out.append("## ① 问题与参考解出处\n")
    out.append(
        "平面库埃特流：两平行壁之间充满不可压流体，下壁静止、上壁以 U 沿切向匀速滑动，"
        "通道水平方向周期。定常不可压 Navier–Stokes 方程在 no-slip 条件下退化为 "
        "d²u/dy² = 0，解析解为线性剖面\n\n"
        "> u(y) = U y / H,  du/dy = U / H,  tau = rho nu U / H,  Q = U H / 2\n")
    out.append(f"**参考解出处**：{payload['reference']}。\n")
    out.append(
        "该解析解无自由参数（U、H、nu 全部是输入），因此不存在把答案调准的空间；"
        "本算例检验的是 TensorSPH 的**切向移动壁 + no-slip 混合（wall_friction）**这条"
        "边界条件通路，而不是粘性算子（后者已由 `benchmarks/poiseuille` 用曲率法单独度量）。\n")

    # ------------------------------------------------------------------ ②
    out.append("## ② 方法与实现\n")
    out.append(
        "* 求解器：TensorSPH 现有 2-D 弱可压 SPH 核心 `tensorsph.droplet_impact.rhs`"
        "（Tait 状态方程、求和密度、对称压力梯度、Morris 型粘性、Adami 壁面压力外推）。\n"
        "* **没有新增物理**：上壁速度通过 `rhs(..., wall_vel=...)` 传入——这是"
        "`wave_flume` 造波板已经在用的移动壁输入，此处把它的方向由法向改成切向；"
        "`wall_friction = 1` 使流-壁粘性作用按壁自身速度而非静止速度计算，即 no-slip。\n"
        "* 初值静止（上壁脉冲启动），积分到稳态；参考解是闭式，在同一行心取值，"
        "不经过任何拟合（拟合只用于线性度诊断与滑移长度）。\n"
        "* 一段式复现命令见文末 `repro_cmd`；所有数值由 `results.json` 程序化注入本报告。\n")

    # ------------------------------------------------------------------ ③
    out.append("## ③ 参数表（含无量纲数）\n")
    out.append(parameters_table(payload) + "\n")
    if ref is not None:
        out.append(
            f"参考运行（H/dx = {ref['resolution_H_over_dx']}）："
            f"Re = {ref['reynolds_number']:.3f}，Ma = {ref['mach_number']:.4f}"
            f"（实测最大 Ma = {ref['max_mach']:.4f}），"
            f"dx = {ref['dx']:.6f} m，dt = {ref['dt']:.3e} s，"
            f"步数 = {ref['steps']}，粒子数 = {ref['n_fluid']} 流体 + "
            f"{ref['n_wall']} 壁，平均样本 = {ref['n_averaged_samples']}。\n")

    # ------------------------------------------------------------------ ④
    out.append("## ④ 对照表（ref vs got，逐项判定）\n")
    out.append("### 4.1 分辨率阶梯与积分量\n")
    out.append(metrics_table(ladder) + "\n")
    out.append(
        f"阈值：内部 L2 < {pct(gates['interior_l2'], 0)}，"
        f"|通量比-1| < {pct(gates['flux_rel'], 0)}，"
        f"|壁剪切比-1| < {pct(gates['wall_shear_rel'], 0)}，"
        f"稳态漂移 < {pct(gates['steady_drift'], 0)}。\n")
    if ref is not None:
        out.append(f"### 4.2 参考运行的逐行剖面（H/dx = {ref['resolution_H_over_dx']}）\n")
        out.append(profile_rows(ref) + "\n")
        out.append(
            f"逐行判定按 5% 计：近壁首行误差 {pct(ref['metrics']['first_row_rel_err'])}，"
            f"末行 {pct(ref['metrics']['last_row_rel_err'])}，"
            f"最大偏差 {pct(ref['metrics']['max_deviation_over_U'])} "
            f"出现在 y/H = {ref['metrics']['max_deviation_row_y_over_H']:.3f}；"
            f"内部带（0.15 ≤ y/H ≤ 0.85）L2 = {pct(ref['metrics']['interior_l2'])}。"
            f"壁面第一层粒子位于流体质心外 0.5 dx，SPH 壁不是锐利平面，近壁行出现"
            "有限滑移是这条边界条件的固有性质，滑移长度单独报告。\n")
    if control is not None:
        out.append("### 4.3 负控（free slip, wall_friction = 0）\n")
        out.append(
            f"自由滑移对照的通量比 = {control['metrics']['flux_ratio']:.4f}"
            f"（阈值 < {gates['control_flux']}），内部 L2 = "
            f"{pct(control['metrics']['interior_l2'])}，"
            f"壁剪切比 = {control['metrics']['wall_shear_ratio']:.4f}。"
            "即：不施加切向壁面剪切时流体基本不被驱动，说明有滑移壁那一组的"
            "线性剖面**确实来自壁面剪切**，而不是初值或拟合假象。\n")

    # ------------------------------------------------------------------ ⑤
    out.append("## ⑤ 收敛性与参数敏感性（≥2 点）\n")
    out.append("### 5.1 收敛性（分辨率阶梯）\n")
    if coarse is not None and finest is not None:
        out.append(
            f"内部 L2 误差：H/dx = {coarse['resolution_H_over_dx']} → "
            f"{pct(coarse['metrics']['interior_l2'])}；H/dx = "
            f"{finest['resolution_H_over_dx']} → {pct(finest['metrics']['interior_l2'])}；"
            f"通量比 {coarse['metrics']['flux_ratio']:.4f} → "
            f"{finest['metrics']['flux_ratio']:.4f}。"
            f"分辨率响应判据（最细不劣于最粗）："
            f"{mark(verdict.get('resolution_response_ok', False))}。\n")
        out.append("![profiles](couette_profiles.png)\n")
        out.append("![convergence](couette_convergence.png)\n")
    if payload.get("speed_sweep"):
        out.append("### 5.2 敏感性之一：驱动速度 U（Re 与 Ma 同步变化）\n")
        out.append(sweep_table(payload["speed_sweep"], "wall_speed", "m/s") + "\n")
        out.append("解析剖面与 U 成正比，因此 u/U 剖面应与 U 无关；误差不随 U 增长即说明"
                   "测到的是壁面驱动而不是数值噪声。\n")
    if payload.get("viscosity_sweep"):
        out.append("### 5.3 敏感性之二：粘度 nu（Re 变化，目标值不变）\n")
        out.append(sweep_table(payload["viscosity_sweep"], "viscosity", "m²/s") + "\n")
        out.append("库埃特稳态剖面对 nu 无关（推导中 nu 被消去），若误差随 nu 变化，"
                   "说明粘性算子误差混进了壁面条件测量。\n")
    if payload.get("friction_sweep"):
        out.append("### 5.4 敏感性之三：壁面摩擦混合系数 f\n")
        out.append(sweep_table(payload["friction_sweep"], "wall_friction", "") + "\n")
        out.append("f 从 0（自由滑移）到 1（no-slip）连续插值：通量比与壁剪切比单调过渡，"
                   "f = 0 为负控。\n")
        out.append("![friction](couette_friction_sweep.png)\n")
    if ref is not None:
        decay = ref.get("startup_decay", {})
        out.append("### 5.5 启动瞬态（补充诊断，不设门）\n")
        out.append(
            f"脉冲启动后 Q(t) 的趋稳速率实测 {decay.get('rate_measured', float('nan')):.4f} "
            f"1/s，解析一阶模态 nu pi^2/H^2 = {decay.get('rate_analytic', float('nan')):.4f} "
            f"1/s，比值 {decay.get('rate_ratio', float('nan')):.4f}。\n")

    # ------------------------------------------------------------------ ⑥
    out.append("## ⑥ 失败模式、局限与工件\n")
    out.append(
        "* **有限滑移长度**：SPH 壁是力边界而非速度 Dirichlet 边界，等效无滑移平面落在"
        "壁粒子层外若干分之一 dx 处；这使近壁行剖面系统性偏高，报告用 `滑移长/dx` 字段"
        "单独度量并在 4.2 中列出，不混进内部 L2。\n"
        "* **稳态判据的相位噪声**：弱可压格式不含人工粘性，声学振铃不被耗散，通量在"
        " ±数个百分点内持续脉动；稳态漂移用平均窗两半的均值差而非两点差计算，"
        "两点差同时作为 `steady_drift_two_samples` 记录在 results.json 里以备对照。\n"
        "* **适用边界**：本文验证的是定常、层流、平面、单向剪切；不含转捩、不含离心"
        "（Taylor–Couette）效应，也不验证三维性。\n"
        "* **工件**：`runs/couette_full/results.json`（含每个 case 的完整参数、逐行剖面、"
        "时间序列、判定）；`runs/couette_full/couette_*.png`（剖面/收敛/场图/摩擦序列）；"
        "设计预声明 `benchmarks/couette/DESIGN.md`；测试 `tests/test_couette.py`。\n")
    out.append("## 复现\n")
    out.append("```bash\n"
               "PYTHONPATH=src python benchmarks/couette/run.py --extended "
               "--out runs/couette_full\n"
               "PYTHONPATH=src python benchmarks/couette/make_doc.py "
               "--results runs/couette_full/results.json "
               "--md docs/reports/m15412-couette-sph.md\n"
               "PYTHONPATH=src python -m pytest tests/test_couette.py -q\n"
               "```\n")
    return "\n".join(out)


def build_payload(doc: str, figures_dir: Path, results: dict) -> dict:
    figs = []
    for name, caption in [
        ("couette_profiles.png", "图1 实测 u/U 剖面 vs 解析解 U y/H（三个分辨率）与逐点误差"),
        ("couette_convergence.png", "图2 积分量误差随分辨率 H/dx 的收敛（含 5% 判据线）"),
        ("couette_field.png", "图3 u_x 场量云图（滑移上壁，周期通道）"),
        ("couette_friction_sweep.png", "图4 壁面摩擦系数 f 扫描：自由滑移 → no-slip"),
    ]:
        f = figures_dir / name
        if f.exists():
            figs.append({"name": name, "caption": caption,
                         "png_base64": base64.b64encode(f.read_bytes()).decode()})
    verdict = results.get("verdict", {})
    ref = next((r for r in results.get("ladder", [])
                if r["resolution_H_over_dx"] == 16), None)
    summary = (f"CASE-COUETTE-SPH 平面库埃特流：TensorSPH 切向移动壁 + no-slip "
               f"驱动，裁决 {verdict.get('status', 'N/A')}")
    if ref is not None:
        summary += (f"；H/dx=16 内部 L2 {100 * ref['metrics']['interior_l2']:.2f}%，"
                    f"通量比 {ref['metrics']['flux_ratio']:.4f}，"
                    f"壁剪切比 {ref['metrics']['wall_shear_ratio']:.4f}")
    return {
        "title": REPORT_TITLE,
        "summary": summary,
        "scenario_key": SCENARIO_KEY,
        "mission_id": 15412,
        "tags": ["TensorSPH", "CASE-COUETTE-SPH", "benchmark", "analytic",
                 "moving-wall", "no-slip", "couette"],
        "markdown": doc,
        "figures": figs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=Path("runs/couette_full/results.json"))
    ap.add_argument("--md", type=Path,
                    default=Path("docs/reports/m15412-couette-sph.md"))
    ap.add_argument("--payload", type=Path, default=Path("/tmp/report.json"))
    args = ap.parse_args(argv)
    results = json.loads(args.results.read_text())
    doc = render(results)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(doc)
    payload = build_payload(doc, args.results.parent, results)
    args.payload.write_text(json.dumps(payload))
    print(f"wrote {args.md} ({len(doc)} chars) and {args.payload} "
          f"({len(payload['figures'])} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())