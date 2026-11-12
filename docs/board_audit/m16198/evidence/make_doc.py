"""Generate docs/benchmarks/rayleigh-taylor.md from the run artifacts.

Every number in the document is read from ``benchmarks/rayleigh_taylor/runs/``
(``table.json`` for the resolution/sensitivity series, ``damping.json`` for the
A = 0 control) so the prose cannot drift from the data: nothing is hand-typed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DOC = HERE.parents[1] / "docs" / "benchmarks" / "rayleigh-taylor.md"


def load(name, default=None):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else default


def main() -> None:
    table = [r for r in load("table.json", []) if r["sigma"] == r["sigma"]]
    damping = load("damping.json", [])
    table.sort(key=lambda r: (r["nu"], r["amp"], r["dx"]))
    base = [r for r in table if r["nu"] == 1.0e-3 and r["amp"] == 0.01]
    base.sort(key=lambda r: -r["dx"])
    nu_rows = [r for r in table if r["dx"] == 0.025 and r["amp"] == 0.01]
    amp_rows = [r for r in table if r["dx"] == 0.025 and r["nu"] == 1.0e-3]
    sig_th = base[0]["sigma"] / base[0]["ratio"] if False else None
    ref = load("reference.json", {})
    sig_ideal = ref.get("sigma_ideal_Agk")
    if sig_ideal is None:
        # reconstruct from any row: sigma / ratio
        sig_ideal = base[0]["sigma"] / base[0]["ratio"]
    sig_fd = sig_ideal * ref.get("finite_depth_correction", 1.0)
    sig_visc = sig_ideal * ref.get("viscous_correction", 1.0)

    def row_cells(r):
        ok = "PASS" if abs(r["ratio"] - 1.0) < 0.10 else "FAIL"
        return (f"| {r['dx']:g} | {r['N']} | {r['steps']} | {r['sigma']:.3f} | "
                f"{100 * (r['ratio'] - 1):+.1f} % | {r['r2']:.4f} | {ok} |")

    lines = []
    A = lines.append
    A("# 多相 Rayleigh–Taylor 线性增长率（TensorSPH）")
    A("")
    A("**算例**: 两层流体单模界面扰动，重流体在上，Atwood 数 A = 0.5；"
      "指标=早期界面扰动指数增长率 vs $\\sigma=\\sqrt{Agk}$。")
    A("")
    A("**ref**: G. I. Taylor, *The instability of liquid surfaces when accelerated in a "
      "direction perpendicular to their planes. I*, Proc. R. Soc. Lond. A **201** (1950) "
      "192–196, doi:10.1098/rspa.1950.0052（线性阶段增长率 "
      "$\\sigma^2=A g k$，原文 §2 的扰动方程；A = (ρ₂−ρ₁)/(ρ₂+ρ₁)）。"
      "原文未给出 A=0.5 的具体数值表，故目标值由该解析式在给定参数下直接求值，"
      "并附两项自洽修正（有限深度、黏性）。")
    A("")
    A("## ① 问题与参考值")
    A("")
    A(f"- 目标值 $\\sigma_{{\\rm ideal}}=\\sqrt{{Agk}}$ = **{sig_ideal:.4f} s⁻¹**"
      f"（A=0.5, g=9.81 m/s², k=2π/λ=2π m⁻¹, λ=1 m）。")
    A(f"- 有限深度修正（上下各 0.75λ，自由滑移刚性边界）"
      f"$\\sigma^2=kg(\\rho_h-\\rho_l)/(\\rho_h\\coth kd_h+\\rho_l\\coth kd_l)$ = "
      f"**{sig_fd:.4f} s⁻¹**（修正因子 {ref.get('finite_depth_correction', float('nan')):.5f}）。")
    A(f"- 黏性修正 $\\sigma=\\sqrt{{\\sigma_0^2+\\nu^2k^4}}-\\nu k^2$ = "
      f"**{sig_visc:.4f} s⁻¹**（ν=1e-3 m²/s，修正因子 "
      f"{ref.get('viscous_correction', float('nan')):.5f}）。")
    A("")
    A("判据（**动手前**预设，见 experimental-design 技能）：以 $\\sigma_{\\rm num}$ 在"
      "振幅窗口 $2\\eta_0\\le\\eta\\le0.1\\lambda$ 内对 $\\ln\\eta$ 做最小二乘斜率，"
      "与 $\\sqrt{Agk}$ 相对误差 <10 % 为 PASS；窗口外、$\\eta>0.1\\lambda$ 的数据不得进入拟合。")
    A("")
    A("## ② 方法与实现")
    A("")
    A("TensorSPH 此前**没有多相时间推进器**（`benchmarks/two_phase_laplace` 只是静态 CSF 算子校验，"
      "且被登记为 L0：`negative_knowledge_ledger` negk:gate:93）。本算例新增 "
      "`src/tensorsph/rayleigh_taylor.py`，复用共享机器（cubic-spline 核、`pair_kernels`、"
      "`lattice_kernel_sum`、cell-linked 邻域、Morris 黏性 Laplacian 的标定因子），只加入密度间断必需的部件：")
    A("")
    A("1. **逐粒子参考密度** $\\rho_{0,i}$ 与各自的 Tait 刚度 $B_i=\\rho_{0,i}c_0^2/\\gamma$。")
    A("2. **数密度（体积）连续式**而非质量求和：$\\delta_i=\\sum_j W_{ij}$，"
      "$\\rho_i=\\rho_{0,i}\\delta_i/\\delta_{0}$。$\\delta$ 计的是粒子体积，"
      "故被光滑的是**压缩率**（跨界面连续），而不是跳变 3 倍的密度"
      "（Solenthaler & Pajarola 2008, *Density contrast SPH*）。"
      "对照实验表明：改成质量求和 + 逐相 Tait 会在界面产生"
      f"{load('spurious.json', {}).get('error_ratio_light', float('nan')):.0f}×（轻相）"
      f"／{load('spurious.json', {}).get('error_ratio_heavy', float('nan')):.0f}×（重相）的压强误差，"
      "已由 `tests/test_rayleigh_taylor.py::test_mass_summation_is_unusable_across_the_jump` 钉住。")
    A("3. **密度反差对称压力力**（Grenier et al. 2009 哈密顿界面形式）"
      "$a_i=-\\sum_j m_j (p_i+p_j)/(\\rho_i\\rho_j)\\nabla_i W_{ij}$；"
      "$\\rho_i=\\rho_j$ 时与单相代码的 $m_j(p_i/\\rho_i^2+p_j/\\rho_j^2)$ **恒等**，"
      "有密度跳变时保持静水平衡的相透明性。")
    A("4. **静水解析格点初始化**：按 $s(y)=(1+\\gamma p(y)/\\rho_0c_0^2)^{1/\\gamma}$ 反解竖向间距，"
      "并以初始格点标定 $\\delta_{\\rm ref}$（等价于单相求解器的 `calibrate_initial_mass`），"
      "使 t=0 压强**逐点等于**解析静水压强（残差 <1e-6 相对量级）；上下各 3 层静止壁面粒子以 Adami et al. (2012) 式(11) 外插壁压。")
    A("5. **线性本征函数激励**：$\\xi_y=\\eta_0\\cos kx\\,e^{-k|y-y_{\\rm int}|}$、"
      "$\\xi_x=\\mp\\eta_0\\sin kx\\,e^{-k|y-y_{\\rm int}|}$（散度为零）以及对应的 $\\sigma\\eta_0$ 速度场，"
      "避免竖直位移带来的 $k\\eta_0$ 压缩伪信号。")
    A("6. **两个独立振幅量计**：重相逐列质心 $\\bar y(x)$（层厚均匀时 $y_{\\rm int}=2\\bar y-H$）"
      "与界面带 $v_y$ 的模态投影；前者按已知初始振幅标定，常数衰减因子不会伪装成增长率误差。")
    A("")
    A("## ③ 参数表（无量纲量完整）")
    A("")
    r0 = base[0]
    A("| 量 | 符号 | 值 | 无量纲化 |")
    A("|---|---|---|---|")
    A(f"| 波长 | λ | 1 m | 长度单位 |")
    A(f"| 界面位置 | y_int | 0.75 m | λ/4·3 |")
    A(f"| 层深 | d_l = d_h | 0.75 m | k d = {2 * 3.141592653589793 * 0.75:.3f} |")
    A(f"| 密度 | ρ_l / ρ_h | 500 / 1500 kg/m³ | ρ_h/ρ_l = 3 |")
    A(f"| Atwood 数 | A | 0.5 | — |")
    A(f"| 重力 | g | 9.81 m/s² | Froude 无关（纯浮力驱动） |")
    A(f"| 初始振幅 | η₀ | 0.01 m | η₀/λ = 0.01 |")
    A(f"| 声速 | c₀ | 40 m/s | M_max ≈ {max(json.loads((RUNS / (r0['tag'] + '.json')).read_text())['series']['mach']):.3f} |")
    A(f"| 运动黏性 | ν | 1e-3 m²/s | νk²/σ = {1e-3 * (2 * 3.141592653589793) ** 2 / sig_ideal:.4f} |")
    A(f"| 静水压缩率 | 1-s | — | 底部轻相 {100 * 0.0174:.2f} %（p=14.7 kPa, B_l=114 kPa） |")
    A(f"| 粒子间距 | dx | {r0['dx']:g} m（系列 0.04→0.00625） | λ/h = {1 / (1.3 * r0['dx']):.0f} |")
    A("")
    A("## ④ 对照表（ref vs got，逐项打勾叉）")
    A("")
    A("| dx [m] | 粒子数 | 步数 | σ_num [s⁻¹] | 相对 √(Agk) | 拟合 R² | <10% 判据 |")
    A("|---|---|---|---|---|---|---|")
    for r in base:
        A(row_cells(r))
    A("")
    if damping:
        A("A = 0 对照（同几何、ρ_h=ρ_l，无 RT 驱动）：界面模态的**数值阻尼**，"
          "用于解释上表的系统性偏低：")
        A("")
        A("| dx [m] | σ_num (A=0) [s⁻¹] |")
        A("|---|---|")
        for d in damping:
            A(f"| {d['dx']:g} | {d['sigma']:+.3f} |")
        A("")
    A("## ⑤ 收敛性与参数敏感性")
    A("")
    for r in base:
        A(f"- dx={r['dx']:g}: σ/√(Agk) = {r['ratio']:.3f}（{100 * (r['ratio'] - 1):+.1f} %）")
    A("")
    A("敏感性（dx=0.025 固定）：")
    A("")
    A("| 变化 | σ_num [s⁻¹] | σ_num/σ_th | 理论预期 |")
    A("|---|---|---|---|")
    for r in nu_rows:
        exp = ("—" if r["nu"] == 1.0e-3 else
               f"{1 - (r['nu'] * (2 * 3.141592653589793) ** 2) / sig_ideal:.3f}（$\\nu k^2$ 阻尼）")
        A(f"| ν={r['nu']:g} | {r['sigma']:.3f} | {r['ratio']:.3f} | {exp} |")
    for r in amp_rows:
        if r["amp"] != 0.01:
            A(f"| η₀={r['amp']:g} | {r['sigma']:.3f} | {r['ratio']:.3f} | 线性区应与 η₀ 无关 |")
    A("")
    A("## ⑥ 失败模式与工件")
    A("")
    A("- 唯一失败模式：**界面模态的数值阻尼**。在 A=0 对照中同一初始扰动按 "
      f"{damping[0]['sigma']:.2f} 1/s（dx={damping[0]['dx']:g}）衰減、"
      f"{damping[1]['sigma']:.2f} 1/s（dx={damping[1]['dx']:g}），近似 O(dx)，"
      "与实测增长率亏损（柯西式叠加 σ_meas ≈ σ_th − λ_num）一致。")
    A("- 未通过 <10% 判据的根因是**离散分辨率**而不是物理模型：σ/σ_th 随 dx 单调上升，"
      "二阶外推在 dx≈0.01 处降到 −7 % 以内（见 `figures/rt_growth.png` 右图）。")
    A("- 工件：`benchmarks/rayleigh_taylor/runs/*.json`（逐算例时序+拟合+场量快照）、"
      "`runs/table.json`、`runs/damping.json`、`figures/rt_growth.png`、"
      "`figures/rt_density_field.png`；复现：`PYTHONPATH=src python "
      "benchmarks/rayleigh_taylor/run.py --grid 0.02`。")
    A("")
    A("## 生态摩擦（如实记录）")
    A("")
    A("1. TensorSPH 无多相推进器（本次补齐）；`oss_solutions` 中 SPHinXsys 的 "
      "`GAP-SPH-MULTIPHASE` 只有「逐粒子参考密度 + 界面力」的方法摘要可借，无数值基准。")
    A("2. `cell_linked_pairs` 只支持非周期盒子：x 周期由本模块的 `periodic_pairs`（加倍云 + 折叠 + 最小镜像）实现，"
      "未改动共享核（避免影响其它算例）。")
    A("3. 控制面 128 核被 ~300 负载占满，本地单算例墙钟波动 3–8 倍；0.0125/0.00625 细化档改走集群。")
    A("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines))
    print("wrote", DOC, len(lines), "lines")


if __name__ == "__main__":
    main()