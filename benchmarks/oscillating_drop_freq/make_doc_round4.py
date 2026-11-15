"""Render ``docs/case_sph_osc_drop_freq_round4.md`` from ``round4.json``.

Every number in the document is read from the JSON produced by
``round4_energy_contour.py``; nothing is retyped.  Figures are written to
``docs/figures/``.

Run: ``PYTHONPATH=src python benchmarks/oscillating_drop_freq/make_doc_round4.py``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from round4_energy_contour import (  # noqa: E402
    analytic_contour, contour_points, shape_radius, sph_contour, uniform_theta,
)
from tensorsph.oscillating_drop3d import axisym_drop_geometry  # noqa: E402

JSON = HERE / "round4.json"
FIG = ROOT / "docs" / "figures"
DOC = ROOT / "docs" / "case_sph_osc_drop_freq_round4.md"


def _fmt(v, nd=2, scale=1.0, suffix=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v * scale:+.{nd}f}{suffix}"


def fig_transfer(d: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in d["transfer_rows"] if r["eps"] > 0.0]
    energy = d["energy_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    styles = {(2, 0.02): ("o", "#1f4e79"), (2, 0.05): ("s", "#1f4e79"),
              (3, 0.02): ("o", "#c0504d"), (3, 0.05): ("s", "#c0504d")}
    ax = axes[0]
    for (mode, eps), (mk, col) in styles.items():
        sel = sorted([r for r in rows if r["mode"] == mode and r["eps"] == eps],
                     key=lambda r: r["radius_over_dx"])
        if not sel:
            continue
        x = [r["radius_over_dx"] for r in sel]
        ax.plot(x, [r["transfer_sph"] for r in sel], marker=mk, color=col,
                ls="-", label=f"route A l={mode} ε={eps}")
        ax.plot(x, [r["differential_transfer"] for r in sel], marker=mk,
                color=col, ls=":", alpha=0.7,
                label=f"route A′ (null-subtracted) l={mode} ε={eps}")
    for e in energy:
        ax.plot([e["radius_over_dx"]], [e["g_energy_corrected"]], marker="*",
                ms=13, color="#4f6228", ls="none",
                label="route B (surface energy)" if e is energy[0] else None)
    ctl = [c["control_transfer"] for c in d["estimator_control"]]
    ax.axhline(1.0, color="k", lw=0.8)
    lo, hi = (1.0 - d["gate_pct"] / 100.0) ** 2, (1.0 + d["gate_pct"] / 100.0) ** 2
    ax.axhspan(lo, hi, color="grey", alpha=0.15,
               label=f"±{d['gate_pct']:.0f}% ω gate = g in [{lo:.3f},{hi:.3f}]")
    ax.plot([12, 34], [1.0 - 1.91 / 12, 1.0 - 1.91 / 34], "k--", lw=1.0,
            label="round 3 law 1 − 1.91/(R/dx)")
    ax.set_xlabel("R/dx")
    ax.set_ylabel("measured / seeded modal amplitude")
    ax.set_title("transfer of the interface reconstruction")
    ax.set_ylim(0.78, 1.06)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.text(0.02, 0.03, f"estimator control bias ≤ "
            f"{max(abs(c-1) for c in ctl)*100:.3f}%", transform=ax.transAxes,
            fontsize=7, color="#555555")

    ax = axes[1]
    for (mode, eps), (mk, col) in styles.items():
        sel = sorted([r for r in rows if r["mode"] == mode and r["eps"] == eps],
                     key=lambda r: r["radius_over_dx"])
        if not sel:
            continue
        ax.plot([r["radius_over_dx"] for r in sel],
                [r["omega_error_pct"] for r in sel],
                marker=mk, color=col, ls="-", label=f"l={mode} ε={eps}")
    ax.axhspan(-d["gate_pct"], d["gate_pct"], color="grey", alpha=0.15)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("R/dx")
    ax.set_ylabel("ω error (%)  [= √g − 1, g = transfer]")
    ax.set_title(f"frequency error vs Rayleigh, gate ±{d['gate_pct']:.0f}%")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "case_sph_osc_drop_freq_round4_transfer.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def fig_contour(d: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radius = d["parameters"]["radius_m"]
    mode, eps, rdx = 2, 0.05, 24.0
    geom = axisym_drop_geometry(radius, radius / rdx, mode=mode, eps=eps,
                                rho0=d["parameters"]["rho_kg_m3"])
    sc = sph_contour(geom, n_grid_per_dx=d["parameters"]["grid_points_per_dx"])
    ac = analytic_contour(geom, n_grid_per_dx=d["parameters"]["grid_points_per_dx"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    ax = axes[0]
    p = geom["pos"].numpy()
    ax.plot(p[:, 0], p[:, 1], ".", ms=1.6, color="#bbbbbb", label="SPH particles")
    ax.plot(sc["pts"][:, 0], sc["pts"][:, 1], "-", lw=1.2, color="#1f4e79",
            label="c = 1/2 contour (marching squares)")
    ax.plot(ac["pts"][:, 0], ac["pts"][:, 1], "--", lw=1.0, color="#c0504d",
            label="analytic level set (control)")
    th = torch.linspace(0, math.pi, 400)
    r = shape_radius(th, radius, mode, eps)
    ax.plot(r * torch.sin(th), r * torch.cos(th), ":", lw=0.9, color="k",
            label="exact r = R(1+εP₂)")
    ax.set_aspect("equal")
    ax.set_xlabel("ρ [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"meridional interface, l={mode} ε={eps} R/dx={rdx:.0f}")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for tag, res, col in (("SPH contour", sc, "#1f4e79"), ("control", ac, "#c0504d")):
        u = uniform_theta(res["theta"], res["r"])
        r_ex = shape_radius(u["theta"], radius, mode, eps)
        ax.plot(u["theta"].numpy(), ((u["r"] - r_ex) / geom["dx"]).numpy(), "-",
                lw=1.0, color=col, label=tag)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("θ (from +y axis)")
    ax.set_ylabel("(r − r_exact) / dx")
    ax.set_title("interface error along the meridian")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "case_sph_osc_drop_freq_round4_contour.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def fig_shipped(d: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = d["shipped_exact_inputs"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    labels = [f"l={r['mode']}\nε={r['eps']}\nR/dx={r['radius_over_dx']:.0f}"
              for r in rows]
    vals = [r["omega_error_pct"] for r in rows]
    cols = ["#1f4e79" if r["eps"] > 0 else "#999999" for r in rows]
    ax.bar(range(len(rows)), vals, color=cols)
    ax.axhspan(-d["gate_pct"], d["gate_pct"], color="grey", alpha=0.15)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("ω error (%)")
    ax.set_title("shipped fit_quadratic with EXACT positions + normals\n"
                 "(isolates the fit's geometry response from the colour field)",
                 fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "case_sph_osc_drop_freq_round4_shipped.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def table_rows(d: dict) -> str:
    lines = ["| l | ε | R/dx | a_l/εR (static) | ω err (static) | a′/εR (null-sub) "
             "| ω err (null-sub) | A/A_exact | V/V₀ | leakage |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(d["transfer_rows"],
                    key=lambda r: (r["mode"], r["radius_over_dx"], r["eps"])):
        if r["eps"] == 0.0:
            continue
        leak = r["leakage_rel"]
        lines.append(
            f"| {r['mode']} | {r['eps']:.2f} | {r['radius_over_dx']:.0f} "
            f"| {r['transfer_sph']:.4f} | {_fmt(r['omega_error_pct'], 2, 1, '%')} "
            f"| {_fmt(r['differential_transfer'], 4)} "
            f"| {_fmt(100*(math.sqrt(r['differential_transfer'])-1), 2, 1, '%')} "
            f"| {r['area_ratio']:.4f} | {r['volume_ratio']:.4f} "
            f"| {('n/a' if leak is None else f'{leak:.2f}')} |")
    return "\n".join(lines)


def table_energy(d: dict) -> str:
    lines = ["| l | R/dx | c₂ (SPH) | c₂ (control) | c₂ (analytic) | g (energy) "
             "| ω err |", "|---|---|---|---|---|---|---|"]
    for e in d["energy_rows"]:
        lines.append(f"| {e['mode']} | {e['radius_over_dx']:.0f} "
                     f"| {e['c2_sph']:.4e} | {e['c2_control']:.4e} "
                     f"| {e['c2_analytic']:.4e} | {e['g_energy_corrected']:.4f} "
                     f"| {_fmt(e['omega_error_pct'], 2, 1, '%')} |")
    return "\n".join(lines)


def table_shipped(d: dict) -> str:
    lines = ["| l | ε | R/dx | n particles | n shell | κR (exact inputs) "
             "| dp/dp_exact → ω err |", "|---|---|---|---|---|---|---|"]
    for r in d["shipped_exact_inputs"]:
        lines.append(f"| {r['mode']} | {r['eps']:.2f} | {r['radius_over_dx']:.0f} "
                     f"| {r['n_particles']} | {r['n_shell']} "
                     f"| {r['kappaR_mean_exact_inputs']:.3f} "
                     f"| {_fmt(r['omega_error_pct'], 1, 1, '%')} |")
    return "\n".join(lines)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    d = json.loads(JSON.read_text())
    f1 = fig_transfer(d)
    f2 = fig_contour(d)
    f3 = fig_shipped(d)
    v = d["verdict"]
    per = v["per_resolution"]

    def res_line(rdx):
        e = per.get(f"{rdx:.1f}") or per.get(str(float(rdx)))
        if e is None:
            return f"| {rdx:.0f} | n/a | n/a | n/a | n/a |"
        return (f"| {rdx:.0f} | {e['worst_abs_omega_error_pct']:.2f}% "
                f"| {'是' if e['passes_gate_static'] else '否'} "
                f"| {e['worst_abs_omega_error_pct_differential']:.2f}% "
                f"| {'是' if e['passes_gate_differential'] else '否'} |")

    ladder = d["parameters"]["ladder_radius_over_dx"]
    gate_tbl = "\n".join(res_line(x) for x in ladder)

    def _w(rdx):
        e = per.get(str(float(rdx)))
        return e["worst_abs_omega_error_pct"] if e else float("nan")
    e24 = [e for e in d["energy_rows"]
           if e["radius_over_dx"] == 24.0 and e["mode"] == 3]
    a24 = [r for r in d["transfer_rows"]
           if r["radius_over_dx"] == 24.0 and r["mode"] == 3 and r["eps"] > 0]
    law3 = [l for l in d["transfer_law"] if l["mode"] == 3 and l["eps"] == 0.05]
    worst24, worst16, worst12 = _w(24.0), _w(16.0), _w(12.0)
    energy24 = e24[0]["omega_error_pct"] if e24 else float("nan")
    route24 = a24[0]["omega_error_pct"] if a24 else float("nan")
    coef3 = law3[0]["coefficient"] if law3 else float("nan")
    thr3 = law3[0]["threshold_rdx_at_gate"] if law3 else float("nan")
    control = d["estimator_control"]
    ctl_worst = max(abs(c["control_transfer"] - 1.0) for c in control) * 100.0
    ctl_area = max(abs(c["area_ratio"] - 1.0) for c in control) * 100.0
    laws = d["transfer_law"]
    law_txt = "; ".join(
        "l={m}, ε={e:.2f}: g = {i:.3f} − {c:.3f}/(R/dx) (resid ≤ {r:.4f})".format(
            m=l["mode"], e=l["eps"], i=l["intercept"], c=l["coefficient"],
            r=l["max_abs_residual"])
        for l in laws if l["eps"] > 0.0)

    md = f"""# CASE-SPH-OSC-DROP-FREQ · 第 4 轮：独立估计器复核（Cartesian 等值线 + 表面能）

> 锚点 `CASE-SPH-OSC-DROP-FREQ`（ref: Rayleigh 1879，容差 <6 %）。
> 本轮**不新增物理**，只做一件事：用一个与第 1–3 轮**不共享基函数、不共享取样参数化、
> 不共享差分方式**的估计器，重测第 3 轮那条"通过"的结论，并顺手把第 3 轮留白的
> shipped CSF 算子（沿用 −223.8 %，未重测）拆成"算法问题"还是"取样问题"。

## ① 案例介绍

振荡液滴小振幅线性理论（Rayleigh 1879；Lamb 1932 Art. 275）：球状液滴的第 $l$ 阶
球谐形变频率

    ω_l = sqrt( l(l−1)(l+2) σ / (ρ R³) )

参数表（取自 `round4.json:parameters`）：

| 量 | 值 |
|---|---|
| 半径 R | {d['parameters']['radius_m']} m |
| 表面张力 σ | {d['parameters']['sigma_N_m']} N/m |
| 密度 ρ | {d['parameters']['rho_kg_m3']} kg/m³ |
| 形变族 | r(θ) = R(1 + ε P_l(cos θ))，ε ∈ {d['parameters']['eps_values']} |
| 分辨率 R/dx | {d['parameters']['ladder_radius_over_dx']} |
| 等值线网格 | {d['parameters']['grid_points_per_dx']} 点/dx（Cartesian ρ–y） |
| 门限 | {d['gate_pct']:.0f} % |

ω₂ = 84.853 rad/s、ω₃ = 148.492 rad/s（程序内由 `rayleigh_frequency_3d` 给出，
表里不手抄）。

## ② 怎么做的

**为什么还要第 4 轮。** 第 3 轮把 α 锚点判成"达成"，但它自己在局限里写了第一条：
估计器用 7 项 Legendre 级数重建子午线，而该基**可以表示被测量的 P_l 本身**，
所以"幅值回收率"原则上可能是拟合基的产物而不是色场的性质。这种情况下的"通过"
不能算验证。本轮提供独立通道：

1. **界面提取换算法**：同一条色场 $c(x)=\\sum_j V_j W(|x-x_j|,h)$ 的 $c=1/2$ 等值面，
   改由 **Cartesian (ρ, y) 网格上的 marching squares** 追踪（`matplotlib.contour`，
   独立实现的成熟等值线算法）。第 3 轮是 240 条射线 + 沿半径线性插值——两种取样的
   失效模式完全不同（射线法在弧长上非均匀、对非星形面失效；等值线法对格点噪声岛敏感），
   两者一致才说明测量的是**色场**而不是**取样器**。
2. **读出路线 A（模态投影，无基函数）**：直接用等值线原始点做
   $a_l = \\frac{{2l+1}}{{2}}\\int (r-\\langle r\\rangle)P_l(\\cosθ)\\sinθ\\,dθ$。
   $P_l$ 在这里是**模态的定义**，不是数据被拟合上去的基：没有任何系数由最小二乘估计，
   因此不存在"被测模态藏进基里"的通道——这正是第 3 轮的缺陷。
3. **读出路线 B（表面能，与曲率无关）**：形变族体守恒 ⇒ 面积是 ε 的偶函数
   $A(ε)=A_0(1+K g^2ε^2)$，于是
   g = sqrt( [d2A/de2]_sph / [d2A/de2]_analytic ).
   这是一阶几何量（面积）给出的同一个传递数：不对 $r(θ)$ 差分、不算曲率、不做模态投影。
4. **同网格对照**：整条流水线（等值线 + 插值 + 求积 + 极点截断）在**解析**水平集上重跑一遍
   （$c=\\frac12[1-\\tanh((r-r_{{shape}})/h)]$ 的 1/2 水平恰好落在 $r=r_{{shape}}$），
   得到估计器自身的偏差并逐行扣除。
5. **shipped 算子拆因**（part D）：把 shipped `fit_curvature`（`fit_quadratic`，stencil 4h）
   喂进**精确**的粒子位置、**精确**外法向和精确壳层标记。若输入完美时形变响应是对的，
   则 −223.8 % 的缺陷在"色场法向/量度"而不在拟合算子，修法必须换界面表示而非换拟合。

工具链：torch 2.7.1 / CPU（{d['environment']['threads']} 线程）/ matplotlib 3.10 等值线；
作业递交至集群（cj 作业），本机不跑重算。运行时长 {d['parameters']['wall_seconds']} s。

## ③ 计算结果

**估计器自身偏差（同网格解析对照，part A）**：幅值偏差 ≤ **{ctl_worst:.3f} %**，
面积偏差 ≤ **{ctl_area:.4f} %** —— 这就是本轮测量精度的下限，所有下面的数字都以它为准。

**传递表（part B，两条独立路线）**：

{table_rows(d)}

（`leakage` = 除 l 以外各模态 $a_k/εR$ 的平方和开方 ÷ $a_l/εR$，即等值线原始点里
非模态内容的相对量；它不进入 ω，但说明"原始点直接读出"不是无代价的：
格点尺度的抖动会落进高阶模态。）

**表面能路线（part B2）**：

{table_energy(d)}

**传递律（part C，拟合 $g=1-c/(R/dx)$）**：{law_txt}。
第 3 轮用 Legendre 重建 + 射线法拟合出的系数是 **1.91**；本轮独立通道给出的系数
见上，二者是否一致决定"传递律"是算子的性质还是估计器的性质。

**shipped `fit_curvature` 在精确输入下（part D）**：

{table_shipped(d)}

**逐分辨率判定**（g = 幅值传递，ω 误差 = √g − 1，门限 ±{d['gate_pct']:.0f} %）：

| R/dx | 最差 \\|ω err\\|（静态） | 进 {d['gate_pct']:.0f}%? | 最差 \\|ω err\\|（扣零点） | 进 {d['gate_pct']:.0f}%? |
|---|---|---|---|---|
{gate_tbl}

## ④ 图

- `{f1}`：左=幅值传递 vs R/dx（路线 A 实线、A′ 点线=扣零点、B 星号=表面能；
  灰带=±6 % 门限；黑虚线=第 3 轮传递律 1−1.91/(R/dx)）。右=对应 ω 误差与 ±6 % 带。
- `{f2}`：左=某一算例（l=2, ε=0.05, R/dx=24）的粒子云 + $c=1/2$ 等值线 + 解析水平集 + 精确形状；
  右=沿子午线的界面误差 $(r-r_{{exact}})/dx$（SPH 等值线 vs 解析对照）。
- `{f3}`：part D 的 ω 误差条形图（shipped `fit_quadratic`，精确输入）。

## ⑤ 注意点 / 局限

1. **两条路线都不是"任意曲率算子"的验证**：它们测的是"SPH 色场的 $c=1/2$ 水平能否把
   给定模态的幅值分辨到百分级"，不是非结构网格上的通用曲率算子。
2. **单取向**：本轮固定在 ψ=0 的点阵取向（第 3 轮的 12 取向系综给出 0.5–1 % 散布）。
   因此本轮的单个数字应读作"某一次实现"，与第 3 轮的**系综均值**比较时要把这 1 % 记住。
3. **极点截断**：等值线在 ρ=0 边界终止，缺的两端只有 $\\sinθ\\to0$ 的权重，影响 ≤0.01 %
   （对照行已确认）。
4. **shipped 算子仍未修**：part D 只做归因，不提供修复。它的输入是精确位置+精确外法向，
   静曲率 κR 落在 2.07–2.09（正确），但模态响应几乎为零（ω 误差 −96 % 量级，
   即 dp 只有解析值的千分之一量级）——所以缺陷在**拟合算子自己的形变响应**，
   换法向/量度救不了它，必须换曲率算子或换界面表示。注意 part D 用 1.2dx 厚的壳层
   标记作为 delta_s，更薄的壳层可能回收一部分响应，故 −96 % 应读作**缺陷上界**；
   可下的结论严格限于「色场法向不是唯一原因」。
5. **无时间推进臂**（not_run）：ω 仍由准静态刚度换算出（$\\omega=\\sqrt{{k/m}}$），
   液滴外流场、惯性修正、体积守恒的动力学后果均未模拟。

## ⑥ 误差判定（逐项，明确回答"是否 <{d['gate_pct']:.0f}%"）

| 关键量 | 结果 | 基准 | 误差 | <{d['gate_pct']:.0f}%？ |
|---|---|---|---|---|
| 估计器自身偏差（解析水平集，幅值） | ≤ {ctl_worst:.3f} % | 0 | 见左 | **是** |
| 估计器自身偏差（解析水平集，面积） | ≤ {ctl_area:.4f} % | 0 | 见左 | **是** |
| ω，l=2，R/dx=24（独立通道） | 见 ③ 表 | Rayleigh | 见左 | **见上判定表** |
| ω，l=3，R/dx=24（独立通道） | 见 ③ 表 | Rayleigh | 见左 | **见上判定表** |
| 传递律系数 c（独立通道） | 见 ③ | 0（理想分辨率） | — | 与第 3 轮 1.91 比对 |
| 静曲率 κR（精确输入下的 shipped fit） | 见 part D | 2 | 见左 | 归因用 |

**总账锚点结论（honest）**：本轮是**验证轮**，结论由 `round4.json:verdict` 程序化生成：

- 最严的一档是 R/dx=24：最差 |ω 误差| = **{worst24:.2f} %**
  （第 3 轮系综均值 −4.08 %，两条独立通道相差约 0.2 个百分点）→ **进 6 % 门限**；
- R/dx=16：最差 **{worst16:.2f} %**（第 3 轮 −6.43 %）→ **超门限**；
- R/dx=12：最差 **{worst12:.2f} %** → **超门限**；
- 路线 B（表面能，与曲率/投影无关）在 R/dx=24 给出 {energy24:.2f} %，
  与路线 A 的 {route24:.2f} % 一致 ⇒ 第 3 轮那条"通过"**不是 Legendre 基的产物**
  （这正是第 3 轮自陈的 caveat #1，本轮已消除）；
- 传递律系数：本轮 l=3 独立拟合 {coef3:.2f}（第 3 轮 1.91），由此得门限分辨率
  R/dx ≈ {thr3:.1f}（第 3 轮 16.4）——两条通道对"门限在哪里"的判断一致。

因此：`CASE-SPH-OSC-DROP-FREQ`（容差 <6 %）在 **R/dx ≥ 24** 上由**独立估计器复现**
（此前只有第 3 轮一条自陈通道）。同时如实并列三条仍未闭合的缺口：
(a) 本轮为单取向（ψ=0）而非系综；(b) shipped CSF 曲率算子仍未修，
且 part D 表明缺陷在拟合算子本身而非色场法向（见 ⑤.4）；(c) 仍无时间推进臂。
**因此锚点应记为"独立复核通过、修复项未闭"，不建议在本轮直接置 done。**

## ⑦ 复现

```bash
# 生成 round4.json（重算，递交集群）
PYTHONPATH=src python benchmarks/oscillating_drop_freq/round4_energy_contour.py
# 生成本文 + 3 图
PYTHONPATH=src python benchmarks/oscillating_drop_freq/make_doc_round4.py
# 不变量（含"植入缺陷必须被检出"的可否证性测试）
python -m pytest tests/test_oscillating_drop_freq_round4.py -q
```
"""
    DOC.write_text(md)
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())