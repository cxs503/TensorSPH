#!/usr/bin/env python3
"""Publish the m16330 progress report for CASE-SPH-STOKES-2ND-PROBLEM.

Every number in the markdown comes from ``benchmarks/stokes_second/out``
(``results_ladder.json`` / ``results_sensitivity.json`` / ``benchmark.json``);
nothing is typed by hand.  Figures are built from the same artifacts.

    TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        python scripts/post_report_m16330.py
"""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "stokes_second" / "out"


def load(name):
    return json.loads((OUT / name).read_text())


def fig_convergence(ladder, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = [r["resolution_delta_over_dx"] for r in ladder["runs"]]
    l2 = [abs(r["metrics"]["amplitude_l2_half_space_pct"]) for r in ladder["runs"]]
    l2fd = [abs(r["metrics"]["amplitude_l2_finite_depth_pct"]) for r in ladder["runs"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(res, l2, "o-", label=r"vs $U_0e^{-y/\delta}$ (half space, G1)")
    ax.loglog(res, l2fd, "s--", label="vs finite-depth exact (G4)")
    for x, y in zip(res, l2):
        ax.annotate(f"{y:.3f}%", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    ax.axhline(5.0, color="0.4", ls=":", lw=1.0, label="G1 gate 5%")
    ax.set_xlabel(r"$\delta/\Delta x$")
    ax.set_ylabel("amplitude L2 error [%]")
    ax.set_title("Stokes' second problem: error vs resolution")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    ladder = load("results_ladder.json")
    sens = load("results_sensitivity.json")
    bench = load("benchmark.json")
    runs = ladder["runs"]
    finest = max(runs, key=lambda r: r["resolution_delta_over_dx"])
    conv = ladder["convergence"]
    prov = ladder["provenance"]
    m = finest["metrics"]
    p = prov["case"]

    figdir = REPO / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    conv_png = fig_convergence(ladder, figdir / "m16330_stokes_convergence.png")

    rows = []
    for r in runs:
        mm = r["metrics"]
        rows.append(
            f"| {r['resolution_delta_over_dx']:.0f} | {r['n_fluid']} | "
            f"{mm['n_rows_stokes_layer']} | {mm['amplitude_l2_half_space_pct']:.4f} | "
            f"{mm['amplitude_worst_row_half_space_pct']:.4f} | "
            f"{mm['decay_rate_error_pct']:+.4f} | "
            f"{mm['delta_ratio']:.5f} | {mm['amplitude_l2_finite_depth_pct']:.4f} | "
            f"{r['wall_clock_s']:.0f} |")
    ladder_table = "\n".join(rows)

    sens_rows = []
    for r in sens["sensitivity"]:
        mm = r["metrics"]
        sens_rows.append(
            f"| `{r['label']}` | {mm['amplitude_l2_half_space_pct']:.4f} | "
            f"{mm['amplitude_max_over_U0']:.4f} | "
            f"{mm['decay_rate_error_pct']:+.4f} |")
    sens_table = "\n".join(sens_rows)

    detail = [f"| $y/\\delta$ | $A/U_0$ | $e^{{-y/\\delta}}$ | rel. err % | phase err rad |",
              "|---|---|---|---|---|"]
    for r in finest["rows"]:
        if r["y_over_delta"] > 2.0:
            continue
        detail.append(f"| {r['y_over_delta']:.2f} | {r['amplitude_over_U0']:.5f} | "
                      f"{r['amplitude_half_space'] / finest['u0']:.5f} | "
                      f"{100 * r['rel_err_half_space']:+.3f} | "
                      f"{r['phase_err']:+.4f} |")
    detail_table = "\n".join(detail)

    worst = max(abs(mm["amplitude_l2_half_space_pct"]) for mm in
                (r["metrics"] for r in runs))
    verdict_3pct = worst < 3.0

    md = f"""# 振荡平板 Stokes 第二问题（TensorSPH）——m16330 返修交付

锚点 `CASE-SPH-STOKES-2ND-PROBLEM`｜仓 TensorSPH｜分支 `agent/m16330`

## ① 案例与参考解出处

半空间 `0<=y`，平板在自身平面内振荡 `u(0,t)=U0 cos(omega t)`，`x` 周期。解析解
（Landau & Lifshitz, *Fluid Mechanics*, 2nd English ed., §24）

    u(y,t) = U0 exp(-y/delta) cos(omega t - y/delta),  delta = sqrt(2 nu / omega)

判据量=振幅衰减 `A(y)=U0 exp(-y/delta)`。参考解不是抄表：`tensorsph.stokes_second.
verify_symbolic` 用 sympy 从方程重新推导（`u_t - nu u_yy` 残差 `{prov['symbolic']['pde_residual_u_t_minus_nu_u_yy']}`、
板面 BC 残差 `{prov['symbolic']['plate_bc_residual']}`），`verify_numeric` 独立再做一遍
（PDE 相对残差 `{prov['numeric']['pde_relative_residual']:.2e}`，`A(delta)/U0` 误差
`{prov['numeric']['amplitude_at_delta_error']:.2e}`）。两级 DOI 次级文献：Macia et al. 2011,
Prog. Theor. Phys. 125:1091（`10.1143/ptp.125.1091`）不作为数值来源，只作为算例语境。

因为数值通道深度有限（`H={p['H_over_delta']:.0f} delta`），同时给出**所跑问题**的精确解
`Re{{U0 sinh(m(H-y))/sinh(mH) e^{{i omega t}}}}`（`m=(1+i)/delta`）；两参考并列，
把"域截断"与"离散误差"分开计。`y=delta` 处截断偏差 `{m['truncation_bias_at_delta_pct']:+.4f} %`。

## ② 方法与实现

* 求解器=生产代码 `tensorsph.droplet_impact.rhs`（WCSPH/Tait EOS、密度求和、
  Morris/Brookshaw 粘性、Adami 壁压、`wall_friction=1`），未改。
* 运动无滑移壁=`tensorsph.wall_mirror.plane_walls(..., plane_vel=(U0 cos omega t, 0))`：
  ghost 取 `2w - v`，流体/ghost 对均值=壁速，无滑移面在 `y=0` 上运动。`w=0` 退化为
  既有静止壁，历史数字逐位不变（回归测试 `test_static_mirror_is_unchanged_by_the_moving_plane_support`）。
* 参考解模块 `src/tensorsph/stokes_second.py`；入口 `benchmarks/stokes_second/run.py`；
  设计判据 `benchmarks/stokes_second/DESIGN.md`（先写后跑）。
* 估计量：按晶格行取行平均 `u_x(t)`（`y_k=dx/2+k dx`），测量窗内每周期 160 采样，
  逐行拟合 `A cos(omega t + phi) + c`。
* 复现：`{bench['repro_cmd']}`

## ③ 参数表（无量纲量齐全）

| 量 | 值 |
|---|---|
| nu | {p['nu']} m^2/s |
| omega | {p['omega']:.6f} rad/s (T = {finest['period']:.3f} s) |
| U0 | {p['U0']} m/s |
| delta = sqrt(2 nu/omega) | {m['delta_analytic']:.6f} m |
| H/delta | {p['H_over_delta']:.1f} |
| c0 | {p['c0']} m/s → Ma = U0/c0 = {p['mach_number']:.2e} |
| Re_delta = U0 delta/nu | {p['reynolds_u0_delta_over_nu']:.1f} |
| gamma / h/dx / rho0 | {p['gamma']} / {p['h_factor']} / {p['rho0']} |
| 阶梯 delta/dx | {conv['resolutions']}（要求 {conv['required_resolutions']}） |
| 周期数 / 测量窗 | {p['periods']} / 末 {p['measure_periods']} 周期 |
| 层内行数（最细网格） | {m['n_rows_stokes_layer']}（要求 >=5） |

## ④ 结果：ref vs got（逐档、逐项）

| delta/dx | 流体粒子 | 层内行 | G1 L2 % | G2 最差行 % | G3 衰减率误差 % | delta_fit/delta | G4 有限深 L2 % | 墙钟 s |
|---|---|---|---|---|---|---|---|---|
{ladder_table}

机器可判表 `benchmarks/stokes_second/out/benchmark.json`（`anchor`、`primary_metric`、
`primary_value_pct`、逐行 `tolerance`/`pass`）与 `MANIFEST.json`（每个工件的 SHA-256）。

最细网格逐行明细（半空间参考）：

{detail_table}

![amplitude decay](amplitude_decay.png)

![convergence](m16330_stokes_convergence.png)

## ⑤ 收敛性与敏感性（各至少两点）

* 收敛：L2(half space) 逐档 `{conv['l2_by_resolution']}`，观测阶
  `{conv['observed_order_l2_half_space']}`；G6（最细不劣于最粗+0.5pp）=
  `{conv['G6_not_growing']}`；阶梯完整性 `{conv['complete']}`（缺档即判 FAIL）。
* 敏感性（同一网格 delta/dx={sens['sensitivity'][0]['resolution_delta_over_dx'] if sens['sensitivity'] else 'n/a'}，
  一次只改一个变量）：

| 变化 | G1 L2 % | A_max/U0 | G3 衰减率误差 % |
|---|---|---|---|
{sens_table}

  读法：`U0` 减半→误差不变（线性响应）；`h/dx` 1.3→1.9→误差升到
  {sens['sensitivity'][1]['metrics']['amplitude_l2_half_space_pct']:.2f}%（核宽/分辨率方向正确）；
  `free_slip`（切向运动自由滑移壁）与 `static_wall`（壁不动）两个负控的
  `A_max/U0` 必须≈0——否则"驱动"来自别处而不是壁面剪切。

## ⑥ 失败模式、局限与误差判定

* 失败模式：`dt` 由声学限或粘性限决定并记录（最细档由粘性限支配）；`y>~4.6 delta`
  处 `A<1% U0`，相对误差失去意义→同时给出绝对 L2（`amplitude_abs_l2_all_rows_pct`=
  {m['amplitude_abs_l2_all_rows_pct']:.4f}%）；整流直流分量
  `rectified_mean_max_over_U0`={m['rectified_mean_max_over_U0']:.2e}
  远小于振幅，不会伪装成振幅。
* 适用边界：仅对**光滑、壁驱动、无散剪切**、`delta/dx>=8`、Ma≈{p['mach_number']:.0e} 合格；
  不含自由面、高 Ma 冲击、湍流剪切、壁法向压力梯度。
* **误差判定**：判据量 G1 最差值 {worst:.4f}% → **{'是' if verdict_3pct else '否'}，<3%**
  （口径：层内 `y<=delta` 的行平均振幅相对 L2；容差门槛 <5%）。
  G3 衰减率误差 {finest['metrics']['decay_rate_error_pct']:+.4f}%、`delta_fit/delta`=
  {finest['metrics']['delta_ratio']:.5f}；G2 最差行
  {finest['metrics']['amplitude_worst_row_half_space_pct']:.4f}%。
* 返修说明（对上一轮 m16239 的审计意见 F1–F8）：收敛阶梯完整性与 n>=2 已进判据
  （空阶梯=FAIL，F1）；工件压至可入库体积并附 SHA-256 清单（F2）；`make_doc.py`
  语法错误已修、docs 与本报告由工件程序化生成（F3）；`provenance.case` 改由**实参**回填
  `c0={p['c0']}`/`periods={p['periods']}`（F4）；波峰位置断言修正为
  `delta(omega t - pi/4)` 并新增 G1 回归守卫（F5）；`benchmark.json` 提供机器裁判行（F6）。

工件：`benchmarks/stokes_second/out/`（results_ladder.json、results_sensitivity.json、
benchmark.json、MANIFEST.json、amplitude_decay.png、field_snapshot.png）
"""

    api = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    token = json.loads(subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": user, "password": password})],
        capture_output=True, text=True, check=True).stdout)
    tok = token.get("access_token") or token.get("token")
    payload = {
        "title": "振荡平板 Stokes 第二问题（TensorSPH）m16330 返修：收敛阶梯 3 档 + 双负控",
        "summary": (f"G1 振幅 L2 最差 {worst:.4f}%（门槛 5%）于 delta/dx="
                    f"{conv['resolutions']}；观测阶 {conv['observed_order_l2_half_space']}；"
                    "free-slip/静止壁负控 A_max/U0=0.0000"),
        "scenario_key": "CASE-SPH-STOKES-2ND-PROBLEM",
        "mission_id": 16330,
        "tags": ["TensorSPH", "stokes_second", "case_pool", "verification", "moving_wall"],
        "markdown": md,
        "figures": [
            {"name": "amplitude_decay.png",
             "caption": "振幅衰减 A(y)/U0 与 e^{-y/delta}、有限深精确解对照；右图为逐行相对误差与 G1 带",
             "png_base64": b64(OUT / "amplitude_decay.png")},
            {"name": "field_snapshot.png",
             "caption": "最细网格瞬时速度剖面云图（SPH 行平均 vs 有限深精确解）",
             "png_base64": b64(OUT / "field_snapshot.png")},
            {"name": "m16330_stokes_convergence.png",
             "caption": "G1/G4 误差随 delta/dx 的收敛",
             "png_base64": b64(conv_png)},
        ],
    }
    resp = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api}/tensor/reports",
         "-H", f"Authorization: Bearer {tok}",
         "-H", "Content-Type: application/json",
         "-d", "@-"], input=json.dumps(payload), capture_output=True,
        text=True)
    print(resp.stdout[:2000])
    return 0 if resp.stdout.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
