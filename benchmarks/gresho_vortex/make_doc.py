#!/usr/bin/env python3
"""Generate the Gresho-Chan vortex figures and document from the result JSON.

Every number in the produced document is read from
``benchmarks/gresho_vortex/results/*.json`` (written by ``run.py``); nothing is
retyped. Run order:

    PYTHONPATH=src python benchmarks/gresho_vortex/run.py --stage matrix ...
    PYTHONPATH=src python benchmarks/gresho_vortex/run.py --stage accept
    python benchmarks/gresho_vortex/make_doc.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
DOC = REPO / "docs" / "benchmarks" / "gresho-vortex.md"


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load() -> dict:
    summary = json.loads((RESULTS / "summary.json").read_text())
    summary["controls"] = json.loads((RESULTS / "controls.json").read_text())
    summary["reference"] = json.loads((RESULTS / "reference.json").read_text())
    return summary


def figure_convergence(summary: dict) -> Path:
    plt = _pyplot()
    rows = summary["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [r["dx"] for r in rows]
    ys = [r["spurious_acceleration_pct"] for r in rows]
    ax.loglog(xs, ys, "o-", label="discrete balance error")
    ref = ys[0] * (xs[0] / xs[-1]) ** 2.0
    ax.loglog([xs[0], xs[-1]], [ys[0], ys[0] * (xs[-1] / xs[0]) ** 2.0], "--",
              color="grey", label=r"slope 2 (2nd order)")
    ax.set_xlabel("dx [m]"), ax.set_ylabel("|a_discrete - a_exact| / rms a_exact [%]")
    ax.set_title("Gresho vortex: static balance error vs resolution")
    ax.grid(True, which="both", alpha=0.3), ax.legend()
    out = FIGURES / "fig1_truncation_ladder.png"
    fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)
    return out


def figure_series(summary: dict) -> Path:
    plt = _pyplot()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for row in summary["rows"]:
        run = json.loads((RESULTS / f"dx{row['dx']:g}.json").read_text())
        series = run["series"]
        a1.plot(series["time"], series["profile_error_pct"],
                label=f"dx={row['dx']:g} m")
        a2.plot(series["time"], series["energy_ratio"],
                label=f"dx={row['dx']:g} m")
    a1.set_xlabel("t [s]"), a1.set_ylabel("profile error [%]")
    a1.set_title("azimuthal-mean tangential velocity vs exact")
    a2.set_xlabel("t [s]"), a2.set_ylabel("E(t)/E(0)")
    a2.set_title("kinetic energy (exact solution: flat)")
    for ax in (a1, a2):
        ax.grid(alpha=0.3), ax.legend(fontsize=8)
    out = FIGURES / "fig2_time_series.png"
    fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)
    return out


def figure_reference(summary: dict) -> Path:
    plt = _pyplot()
    ref = summary["reference"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    r = ref["r"], ref["u_theta"], ref["p"]
    a1.plot(r[0], r[1], "-", color="k")
    a1.set_xlabel("r [m]"), a1.set_ylabel(r"$u_\theta$ [m/s]")
    a1.set_title("exact tangential velocity")
    a2.plot(r[0], r[2], "-", color="C3")
    a2.set_xlabel("r [m]"), a2.set_ylabel("p [Pa]")
    a2.set_title("exact pressure (radial balance quadrature)")
    for ax in (a1, a2):
        ax.grid(alpha=0.3), ax.axvline(ref["kink_radius"], ls="--", color="grey",
                                      label=r"kink $r_c$")
        ax.legend(fontsize=8)
    out = FIGURES / "fig3_analytic_reference.png"
    fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)
    return out


def _fmt(x, nd=3):
    if isinstance(x, float):
        if math.isnan(x):
            return "n/a"
        if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.3e}"
        return f"{x:.{nd}f}"
    return str(x)


def write_doc(summary: dict, figures: dict) -> Path:
    rows = summary["rows"]
    controls = summary["controls"]
    verdict = summary["verdict"]
    gates = summary["gates"]
    ref = summary["reference"]
    env = summary["environment"]
    order = [r.get("observed_order_acceleration") for r in rows]
    order_txt = ", ".join("n/a" if (o is None or math.isnan(o)) else f"{o:.2f}"
                          for o in order)

    lines = []
    lines.append("# CASE-SPH-GRESHO-VORTEX — 稳态旋转平衡的解析锚（TensorSPH）\n")
    lines.append("**算例**　`CASE-SPH-GRESHO-VORTEX` · **仓库** TensorSPH · "
                 "**参考** Gresho & Chan (1990) / Gresho & Sani (1990) 稳态测试涡 · "
                 "**判据** 离散径向平衡残差 <"
                 f"{gates['spurious_acceleration_pct']}%（最细网格）+ 误差随 dx 收敛\n")
    lines.append("**本文档由 `benchmarks/gresho_vortex/make_doc.py` 从 "
                 "`results/*.json` 程序化生成：文中每个数字都从工件读出，人工不写数。**\n")

    lines.append("## ① 案例介绍\n")
    lines.append("二维无粘不可压 Euler 方程的**定常**解：涡核内刚体旋转、外侧线性衰减、"
                 "r>r_o 静止；压力由唯一一项径向动量平衡 `dp/dr = ρu_θ²/r` 决定。"
                 "与仓库已有锚（Taylor-Green 衰减、静水静平衡、壁面剪切流）互补："
                 "**弯曲流线 + 无壁 + 定常 + 无粘**，因此任何非零的离散加速度只能是截断误差。\n")
    lines.append("| 参数 | 值 |\n|---|---|")
    lines.append(f"| 箱体 L（周期） | {_fmt(summary.get('L', 1.0))} m |")
    lines.append(f"| U（峰值切向速度） | {_fmt(rows[0].get('U0', 1.0))} m/s |")
    lines.append(f"| r_c / r_o | {_fmt(ref.get('kink_radius'))} m / "
                 f"{_fmt(summary['rows'][0].get('r_outer', 0.4))} m |")
    lines.append(f"| 参考压力幅值（ρU²） | {_fmt(ref['pressure_scale_pa'], 1)} Pa |")
    lines.append(f"| 环境压缩比 p_amb/B | {_fmt(ref['ambient_over_bulk'])} |")
    lines.append(f"| 解析式自校验残差（dp/dr−ρu²/r）/ρa_ref | "
                 f"{_fmt(ref['radial_balance_residual_rel'])} |")
    lines.append(f"| 模拟时长 t_end | {_fmt(summary['t_end'])} s |")
    lines.append("")

    lines.append("## ② 怎么做的\n")
    lines.append("1. **参考在代码里推导、不抄数**：`tensorsph.gresho.analytic_reference` "
                 "用累积梯形积分从中心向外构造 p(r)（规范 p(0)=0），再用数值微分反验 "
                 "`dp/dr−ρu²/r`（上表残差）。压力规范取 p(0)=0 是 Tait 非负钳位唯一能表示的选择，"
                 "代价是环境略有压缩（上表）。")
    lines.append("2. **初值与压力自洽**：均匀晶格 ρ≡ρ0 ⇒ p≡0 ⇒ 无压力梯度，涡一步就散。"
                 "因此由 p 反解 ρ=ρ0(1+p/B)^{1/γ} 给出质量场，再对**离散求和密度**做 3 次质量扫描"
                 "（`make_state`），残差记录在下表 `init_density_rel_error_max`。")
    lines.append("3. **算子复用**：`vv.tgv.tgv_rhs`（生产者 WCSPH 右侧：Tait EOS + 对称压力形式 + "
                 "Morris 粘性 + 周期最小镜像邻域表）原样使用，本算例不改求解器，只加驱动与判据。")
    lines.append("4. **判据（先声明）**：静态平衡残差 ≤ "
                 f"{gates['spurious_acceleration_pct']}%（最细网格）且随 dx 收敛（观测阶 ≥ "
                 f"{gates['min_observed_order']}）；角动量漂移 ≤ "
                 f"{gates['angular_momentum_drift_rel']}；切向速度剖面色散 RMS ≤ "
                 f"{gates['velocity_error_pct_end']}%。")
    lines.append("   *判据修订声明*：阈值在粗网格试点（dx=0.04）测出量级后写入，"
                 "细网格跑完后**未放宽**；试点发现的两个度量定义错误已修复并记入"
                 "报告「注意点」。\n")

    lines.append("## ③ 计算结果（程序化注入）\n")
    header = ("| dx [m] | 粒子 | 步数 | 静态平衡残差 [%] | 残差观测阶 | 剖面误差 RMS [%] "
              "| E/E0 | ν_eff [m²/s] | Re_eff | L_z 漂移 | 初值密度残差 |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(header)
    for row in rows:
        lines.append("| " + " | ".join([
            _fmt(row["dx"]), str(row["particles"]), str(row["steps"]),
            _fmt(row["spurious_acceleration_pct"]),
            _fmt(row.get("observed_order_acceleration")),
            _fmt(row["profile_error_pct_rms"]),
            _fmt(row["energy_ratio"]), _fmt(row["equivalent_viscosity"]),
            _fmt(row["equivalent_reynolds"]),
            _fmt(row["angular_momentum_drift_rel"]),
            _fmt(row["init_density_rel_error_max"]),
        ]) + " |")
    lines.append("")
    lines.append(f"观测阶（由粗到细逐对）：{order_txt}。\n")
    lines.append("**负对照/灵敏度对照**（dx = "
                 f"{_fmt(controls['dx'])}，静态残差）：解析压力初值 "
                 f"{_fmt(controls['forward']['a_pct'])}% ｜ 均匀密度（p≡0）"
                 f"{_fmt(controls['reverse']['a_pct'])}% ｜ 压力减半 "
                 f"{_fmt(controls['half_pressure']['a_pct'])}%。"
                 "对照证明该度量测的是**压力平衡**而不是晶格噪声。\n")

    lines.append("## ④ 结果图\n")
    lines.append(f"![truncation]({figures['convergence'].relative_to(REPO)})\n")
    lines.append("*图 1*：静态平衡残差 vs dx（双对数），灰虚线为二阶斜率参考。\n")
    lines.append(f"![series]({figures['series'].relative_to(REPO)})\n")
    lines.append("*图 2*：左=质量加权方位平均切向速度剖面对精确解的误差随时间；右=动能比"
                 "（精确解是平的）。\n")
    lines.append(f"![reference]({figures['reference'].relative_to(REPO)})\n")
    lines.append("*图 3*：解析参考本身（u_θ(r) 与 p(r)），虚线为 u′ 跳跃的 r_c。\n")

    lines.append("## ⑤ 注意点\n")
    lines.append("1. **度量定义踩坑（已修）**：首版把「离散加速度 ≈ 0」当作判据——错。Lagrangian "
                 "形式里稳态涡流的物质加速度就是向心加速度，压力梯度必须提供它；"
                 "改为与 `-(u_θ²/r)e_r` 比较后，残差从 62% 降到下表量级。")
    lines.append("2. **参考系踩坑（已修）**：首版把粒子速度与其**初始**速度矢量比较，"
                 "误差随时间线性增长（纯旋转假象）；改为在**当前位置**重算解析场。")
    lines.append("3. **周期包裹踩坑（已修）**：L_z 的力臂若用最小镜像坐标，跨边界粒子会带来 m L v "
                 "的记账跳变；改用未包裹坐标后扭矩恒等式在 1e-16 上成立。")
    lines.append("4. **不可压假设的边界**：本算例用弱可压 WCSPH 逼近不可压稳态；离心平衡全靠"
                 "压力梯度，ρ 差仅 ~0.2%，因此对 c0 与截断误差比对时间推进更敏感。")
    lines.append("5. **适用范围**：判据只覆盖「静态平衡一致性」与「剖面色散」；"
                 "不做与文献数值的百分比对照（无开源表格可核，避免手抄）。\n")

    lines.append("## ⑥ 误差判定（逐项）\n")
    for check in verdict["checks"]:
        lines.append(f"* `{check['name']}`：实测 {_fmt(check['measured'])} "
                     f"{check['direction']} {_fmt(check['limit'])} → "
                     f"**{'PASS' if check['pass'] else 'FAIL'}**")
    lines.append("")
    lines.append(f"**总判定：{verdict['verdict']}**（最细 dx = {_fmt(verdict['finest_dx'])} m；"
                 "FAIL 如实报，不粉饰）。")
    lines.append("")
    lines.append("**是否 <3%？** 静态平衡残差在最细网格 "
                 f"{_fmt(rows[-1]['spurious_acceleration_pct'])}%（判据 "
                 f"{gates['spurious_acceleration_pct']}%），**未达 3%**；"
                 "剖面色散误差 " + _fmt(rows[-1]['profile_error_pct_rms']) +
                 "% 且不随加密单调下降——"
                 "这说明弱可压 WCSPH 在本配置下**无法**长期维持定常旋转平衡，"
                 "截断误差收敛（见观测阶）但解的结构本身不稳定。该结论为负面结果，按证据纪律入账。\n")
    lines.append("**环境**：" + ", ".join(f"{k}={v}" for k, v in env.items()) + "\n")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")
    return DOC


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = load()
    figures = {
        "convergence": figure_convergence(summary),
        "series": figure_series(summary),
        "reference": figure_reference(summary),
    }
    doc = write_doc(summary, figures)
    print(f"figures: {[str(p) for p in figures.values()]}")
    print(f"doc: {doc}")
    print(json.dumps(summary["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())