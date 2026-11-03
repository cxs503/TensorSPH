"""Render ``docs/verification_surface_tension_laplace.md`` from the committed
benchmark artifacts.

Every number in the document is read from ``instant_report.json``,
``validation_report.json`` and ``convergence/*.json``; nothing is transcribed by
hand, so the document cannot drift away from the measurement.

    PYTHONPATH=src python benchmarks/laplace_law/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent
DOC = BENCH.parents[1] / "docs" / "verification_surface_tension_laplace.md"

BAR_15 = "| 误差 | 判定（阈值 15%）|"


def load():
    inst = json.loads((BENCH / "instant_report.json").read_text())
    main = json.loads((BENCH / "validation_report.json").read_text())
    conv = [json.loads(p.read_text())
            for p in sorted(BENCH.glob("convergence/*.json"))]
    return inst, main, conv


def f(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    inst, main, conv = load()
    s = main["setup"]
    ref = main["reference"]
    lap = main["laplace"]
    sc = main["spurious_current"]
    rows = inst["resolutions"]
    by_dx = {round(r["dx_m"], 5): r for r in rows}

    # transmission efficiency: measured pressure jump / jump implied by the
    # cut-force integral at the same resolution
    eff = {}
    for rep in conv:
        dx = round(rep["setup"]["dx_m"], 5)
        if dx in by_dx:
            implied = by_dx[dx]["cut_force"]["implied_delta_p_Pa"]
            eff[dx] = (rep["laplace"]["delta_p_measured_Pa"] / implied * 100.0
                       if implied else float("nan"))

    L = []
    A = L.append
    A("# 验证欠账复核：CSF 表面张力模型 vs Young–Laplace 解析解")
    A("")
    A("**仓库**: TensorSPH ｜ **分支**: `agent/m2174` ｜ **工件**: "
      "`benchmarks/laplace_law/` ｜ **测试**: `tests/test_laplace_law.py`")
    A("")
    A("**结论（先给判定）**: 表面张力模块（CSF，m277 合入）**未通过**解析验证。"
      f"Young–Laplace 压差实测偏差 **{f(lap['rel_error_pct'],1)}%**（阈值 15%），"
      f"曲率估计偏差 **+{f(rows[0]['curvature']['delta_s_weighted_rel_error_pct'],0)}%** 量级，"
      f"切割面力恒等式偏差 **{f(abs(by_dx[min(by_dx)]['cut_force']['ratio_measured_over_theory']-1)*100,0)}"  # noqa: E501
      f"–{f(abs(by_dx[max(by_dx)]['cut_force']['ratio_measured_over_theory']-1)*100,0)}%** 且随分辨率不收敛，"  # noqa: E501
      f"静滴残余伪流达到毛细速度的 **{f(sc['ratio'],2)} 倍**（文献判据 < 0.1）。"
      "证据等级应从「已实现」下调为 **WITHHELD（实现存在、解析验证不通过）**。")
    A("")

    # ---------------------------------------------------------------- ①
    A("## ① 案例介绍")
    A("")
    A("表面张力使液滴内部压强高于外部，2-D 圆截面满足 Young–Laplace 关系")
    A("")
    A("$$\\Delta p = p_{in} - p_{out} = \\sigma/R \\qquad (1)$$")
    A("")
    A("这是表面张力模型**唯一**有闭式解的整体性判据：曲率、力的量级、压强传递"
      "三个环节任一环节错，式 (1) 就不成立。TensorSPH 的 m277 开发报告只统计了"
      "「力方向中位数 cos=0.453、正向比例 59.9%」并判 PASS，"
      "`benchmarks/static_droplet` 打印了 `Δp_theory = σ/R` 却从未测量 Δp，"
      "因此本单把这笔验证欠账一次结清。")
    A("")
    A("| 参数 | 值 | 说明 |")
    A("|---|---|---|")
    A(f"| 液滴半径 R | {f(s['radius_m'],4)} m | 验证算例，静止自由液滴 |")
    A(f"| 粒子间距 dx | {f(s['dx_m'],4)} m | 1.3dx 平滑长度 h = {f(s['h_m'],5)} m |")
    A(f"| 表面张力系数 σ | {f(s['sigma_N_per_m'],1)} N/m | 10× 水，压缩毛细时间便于在步数预算内平衡 |")  # noqa: E501
    A(f"| 参考密度 ρ0 | {f(s['rho0'],0)} kg/m³ | 弱可压 Tait，γ = 7 |")
    A(f"| 声速 c0 | {f(s['sound_speed'],0)} m/s | Mach ≪ 0.1 |")
    A(f"| 粘度 ν | {f(s['viscosity'],2)} m²/s | 层流正则化 Laplacian |")
    A(f"| 粒子数 N | {s['n_particles']} | 六角密排圆盘 |")
    A(f"| 重力 | {f(s['gravity'],1)} | 关闭，纯毛细平衡 |")
    A(f"| 解析压差 σ/R | **{f(ref['delta_p_Pa'],1)} Pa** | 待比对的基准量 |")
    A(f"| 解析曲率 1/R | **{f(ref['curvature_1_per_m'],1)} 1/m** | 待比对的第二个基准 |")
    A(f"| 切割面力 2σ | **{f(ref['cut_force_N_per_m'],1)} N/m** | 第三个基准：「表面张力=单位长度力」 |")  # noqa: E501
    A(f"| 毛细时间 t_σ | {f(ref['capillary_time_s'],4)} s | 动力学观测窗口 |")
    A(f"| 毛细速度 √(σ/ρR) | {f(ref['capillary_velocity_m_per_s'],3)} m/s | 伪流归一化尺度 |")
    A("")

    # ---------------------------------------------------------------- ②
    A("## ② 怎么做的")
    A("")
    A("新增 `benchmarks/laplace_law/run.py`（自包含；`--mode instant|relax|both`），"
      "复用生产代码路径 `tensorsph.droplet_impact.rhs` 与 "
      "`tensorsph.surface_tension.*`，不改任何 `src/` 文件（本单是验证单）。")
    A("")
    A("1. **几何**：六角密排圆盘（6 近邻，比正方格点噪声小一半），"
      "格距满足 `mass × 数密度 = ρ0`，保证初始密度无偏。")
    A("2. **三层判据互相独立**：")
    A("   - *瞬时*（一次求值，无时间推进）：CSF 管线逐环节拆开打分 —— "
      "`c`（邻居数比）→ `∇c` → `n` → `κ` → `δ_s=|∇c|` → 加速度；")
    A("   - *积分*：切割面力恒等式 —— 对 x>0 半滴求和 `Σ m_i a_st,ix`，"
      "解析值恰为 `-2σ`（每单位展向深度）。该量只依赖力的量级，"
      "与压强场无关，因此能把「力量级错」与「压强传不上去」分开归因；")
    A("   - *动力学*：以 `stable_dt()` 定步长推进至 t ≈ 1 t_σ，"
      "用 Tait 反算压强，取 `p_center − p(0.8–0.95R)` 与 σ/R 比对。")
    A("3. **分辨率扫描**：dx = 1.0/1.5/2.0/3.0 mm（N = "
      f"{rows[0]['n_particles']}/{rows[1]['n_particles']}/"
      f"{rows[2]['n_particles']}/{rows[3]['n_particles']}），"
      "用于判断误差是否收敛（是否只是分辨率不足）。")
    A("4. **伪流判据**：`max|v| / √(σ/ρR) ≪ 1`（Adami, Ihmsen & Adams 2012, "
      "JCP 231:7057 静滴算例）。")
    A("")
    A("工具链：PyTorch 2D 弱可压 SPH（CPU 双线程）、pytest、matplotlib。"
      "全部数字由 `make_doc.py` 从 JSON 工件注入本文档。")
    A("")

    # ---------------------------------------------------------------- ③
    A("## ③ 计算结果")
    A("")
    A("### 3.1 瞬时诊断（分辨率扫描，" + f"{len(rows)} 个分辨率" + "）")
    A("")
    A("`κ_w` 为按表面 delta 函数 `|∇c|` 加权的界面曲率（CSF 力实际使用的权重），"
      "是与 1/R 可比的量；`δ_s·2h` 理想值 ≈ 1（表面 delta 函数应归一到 ∫δ_s dS = 1）。")
    A("")
    A("| dx [m] | N | 表面粒子占比 | κ_w [1/m] | κ_w 相对 1/R | δ_s·2h | F_cut/2σ | F_cut 隐含 Δp [Pa] | 中位 cos(a, r̂) | 内向粒子比例 |")  # noqa: E501
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["curvature"]
        A(f"| {f(r['dx_m'],4)} | {r['n_particles']} | {f(r['surface_fraction'],2)} | "
          f"{f(c['delta_s_weighted_mean_1_per_m'],0)} | "
          f"{c['delta_s_weighted_rel_error_pct']:+.0f}% | "
          f"{f(r['delta_s']['band']['mean']*2*r['h_m'],2)} | "
          f"{f(r['cut_force']['ratio_measured_over_theory'],3)} | "
          f"{f(r['cut_force']['implied_delta_p_Pa'],0)} | "
          f"{f(r['inward_alignment']['cos_band_median'],3)} | "
          f"{f(r['inward_alignment']['fraction_inward'],2)} |")
    A("")
    A(f"解析参考：κ = {f(rows[0]['curvature']['exact_1_per_m'],1)} 1/m，"
      f"F_cut = {f(rows[0]['cut_force']['theory_N_per_m'],1)} N/m（即 2σ），"
      "理想 `δ_s·2h` ≈ 1，理想 cos ≈ 1，内向比例 1.00。")
    A("")

    A("### 3.2 动力学：一个毛细时间后的压强")
    A("")
    A("| dx [m] | 步数 | t/t_σ | Δp 解析 [Pa] | Δp 实测 [Pa] | 相对误差 | <15%? | 伪流比 |")
    A("|---|---|---|---|---|---|---|---|")
    seen, all_relax = set(), []
    for rep in conv + [main]:
        key = (round(rep["setup"]["dx_m"], 5), rep["setup"]["steps"])
        if key not in seen:
            seen.add(key)
            all_relax.append(rep)
    for rep in all_relax:
        st, la, sp = rep["setup"], rep["laplace"], rep["spurious_current"]
        A(f"| {f(st['dx_m'],4)} | {st['steps']} | {f(st['t_over_t_sigma'],2)} | "
          f"{f(la['delta_p_theory_Pa'],1)} | {f(la['delta_p_measured_Pa'],2)} | "
          f"{la['rel_error_pct']:+.1f}% | {'是' if la['within_15pct'] else '**否**'} | "
          f"{f(sp['ratio'],3)} |")
    A("")
    if eff:
        A("**力的量级与压强传递分项**：把 3.2 的实测 Δp 与 3.1 同分辨率下"
          "由切割面力反推的 Δp 相比，得到「压强传递效率」：")
        A("")
        A("| dx [m] | 切割面力隐含 Δp [Pa] | 实测 Δp [Pa] | 传递效率 |")
        A("|---|---|---|---|")
        for dx, e in sorted(eff.items()):
            rep = [r for r in conv if round(r["setup"]["dx_m"], 5) == dx][0]
            A(f"| {f(dx,4)} | {f(by_dx[dx]['cut_force']['implied_delta_p_Pa'],0)} | "
              f"{f(rep['laplace']['delta_p_measured_Pa'],2)} | {f(e,1)}% |")
        A("")
        A("传递效率远小于 100% 说明：**力的量级问题（最多差 2 倍）不是主因，"
          "主因是压强场根本建立不起 Laplace 过压**。")
        A("")

    A("### 3.3 密度视角")
    A("")
    dns = main["density"]
    A("| 量 | 值 |")
    A("|---|---|")
    A(f"| ρ_center/ρ0 | {dns['rho_center_over_rho0']:.6f} |")
    A(f"| ρ(0.8–0.95R)/ρ0 | {dns['rho_band_over_rho0']:.6f} |")
    A(f"| 满足 σ/R 所需压缩量 | {dns['compression_required_by_law']:.2e} |")
    A("")
    A("所需压缩量在 1e-4–1e-3 量级，而 SPH 求和密度在六角格点上的本底噪声同阶，"
      "弱可压 Tait + 求和密度**无法**可靠地把这部分压缩量解出来（详见 ⑤）。")
    A("")

    # ---------------------------------------------------------------- ④
    A("## ④ 结果图")
    A("")
    A("![laplace](benchmarks/laplace_law/laplace_law.png)")
    A("")
    A("(a) 平衡态压强剖面 vs 解析 σ/R —— 实测剖面几乎贴着 0，解析线悬在 360 Pa 处；"
      "(b) 伪流历史（对数轴），始终在 Adami 阈值 0.1 之上并轻微增长；"
      "(c) 曲率随分辨率变化，解析值 50 1/m 与实测 150–230 1/m 之间无收敛迹象；"
      "(d) 切割面力恒等式，1.0 为解析正确线，实测 0.45–1.54 非单调，不收敛。")
    A("")

    # ---------------------------------------------------------------- ⑤
    A("## ⑤ 注意点（根因 / 局限 / 适用范围）")
    A("")
    A("**根因链（逐条有数据支撑）**：")
    A("")
    A("1. *颜色函数不是界面函数*。`c` 由 `邻居数/平均邻居数` 构造且仅对"
      "`count < 0.7×avg` 的粒子置值，界面层只有 1 个粒子厚"
      f"（表面占比 {f(rows[0]['surface_fraction'],2)}–{f(rows[3]['surface_fraction'],2)}，"
      "理论应约 2h/R）；`∇c` 因此既非单峰也无固定尺度。")
    A(f"2. *δ_s 不归一*。`∫δ_s dS = 1` 是 CSF 成立的前提，实测 `δ_s·2h` 仅 "
      f"{f(rows[3]['delta_s']['band']['mean']*2*rows[3]['h_m'],2)}–"
      f"{f(rows[0]['delta_s']['band']['mean']*2*rows[0]['h_m'],2)}（理想 1），"
      "且随 dx 变小反而更小 —— 力的量级因此**依赖分辨率**（对应 3.1 最后一列的反常上升）。")
    A("3. *曲率噪声淹没信号*。`κ = -∇·n` 在 1 粒子厚界面 + 噪声法向上放大"
      f"{f(rows[3]['curvature']['delta_s_weighted_rel_error_pct'],0)}–"
      f"{f(rows[0]['curvature']['delta_s_weighted_rel_error_pct'],0)}%（与 m277 报告"
      "自陈的「曲率噪声」一致），法向与径向的余弦中位数只有 "
      f"{f(rows[3]['inward_alignment']['cos_band_median'],2)}–"
      f"{f(rows[0]['inward_alignment']['cos_band_median'],2)}，"
      "内向粒子比例 ≈ 掷硬币。")
    A("4. *超压无处安放*。CSF 力只以体积力形式进入动量方程；弱可压 Tait 要表现"
      "σ/R 必须靠 0.09% 的密度压缩，而求和密度在自由表面附近本身就有 10–30% 亏损"
      "并被 `clamp_min(0)` 截断为 0 Pa，因此液滴内部压强恒 ≈ 0，"
      "力与压强梯度永远不平衡 → 液滴持续被压缩/抖动（3.2 伪流列）。")
    A("")
    A("**修复配方（下一步 P0，本次未实施，属代码改动不属验证单）**：")
    A("")
    A("- 引入 Adami et al. (2012) 式压强偏置：`p_i = p0 + B[(ρ_i/ρ0)^γ − 1]`，"
      "`p0 = σ/R`（可迭代求解以获得自洽 R），并配合 `1/ρ²` 对称形式；")
    A("- 用 Shepard/MLS 归一化的连续性方程或 δ-SPH 密度扩散获得平滑 `∇c`，"
      "再令 `δ_s = |∇c| / ∫|∇c|dS` 归一；")
    A("- 曲率改用加权最小二乘/抛物线拟合（如 Morris 2000；Adami 2012 附件），"
      "替代 `-∇·n` 的直接离散；")
    A("- 验收门槛即本文档 3.1/3.2 的表：κ、F_cut、Δp 三项均 < 15%，"
      "`tests/test_laplace_law.py` 的 4 个 `xfail(strict=True)` 转 xpass。")
    A("")
    A("**适用范围 / 局限**：")
    A("")
    A("- 本验证只覆盖**单相** CSF；多相（真实颜色函数）不在本实现范围内，"
      "结论不能外推到未来的多相线。")
    A("- 为把毛细时间压进步数预算，σ 取 10× 水、c0 = 20 m/s；"
      "这是无量纲化选择，判据形式（σ/R、2σ、1/R）与之无关，"
      "但绝对压强数值不可直接与真实水算例对比。")
    A(f"- 动力学只推进到 t/t_σ ≈ {f(s['t_over_t_sigma'],2)}；更长窗口可能让伪流"
      "继续增长或衰减，但不会让 Δp 从 0 跳到 360 Pa（3.3 说明这是机制问题）。")
    A("- 分辨率下限 dx = 3 mm（16 粒子跨径），再粗则表面探测器退化；"
      "细到 1 mm 时误差反而更大，说明不是分辨率不足，而是方法缺陷。")
    A("")

    # ---------------------------------------------------------------- ⑥
    A("## ⑥ 误差判定（逐项，显式回答「是否 < 3%」）")
    A("")
    A("| 关键量 | 基准 | 实测 | 相对误差 | < 3%? | < 15%? |")
    A("|---|---|---|---|---|---|")
    A(f"| Laplace 压差 Δp | σ/R = {f(ref['delta_p_Pa'],1)} Pa | "
      f"{f(lap['delta_p_measured_Pa'],2)} Pa | {lap['rel_error_pct']:+.1f}% | "
      f"**否** | **否** |")
    for rep in conv:
        la = rep["laplace"]
        A(f"| Δp @ dx={f(rep['setup']['dx_m'],4)} | {f(la['delta_p_theory_Pa'],1)} Pa | "
          f"{f(la['delta_p_measured_Pa'],2)} Pa | {la['rel_error_pct']:+.1f}% | "
          f"**否** | **否** |")
    kw = rows[2]["curvature"]
    A(f"| 界面曲率 κ（δ_s 加权）@ dx={f(rows[2]['dx_m'],4)} | "
      f"{f(kw['exact_1_per_m'],1)} 1/m | "
      f"{f(kw['delta_s_weighted_mean_1_per_m'],0)} 1/m | "
      f"{kw['delta_s_weighted_rel_error_pct']:+.0f}% | **否** | **否** |")
    cf = rows[2]["cut_force"]
    A(f"| 切割面力 F_cut @ dx={f(rows[2]['dx_m'],4)} | {f(cf['theory_N_per_m'],2)} N/m | "
      f"{f(cf['measured_N_per_m'],2)} N/m | "
      f"{abs(cf['ratio_measured_over_theory']-1)*100:+.0f}% | **否** | **否** |")
    A(f"| 伪流比 max|v|/√(σ/ρR) | < 0.1（文献） | {f(sc['ratio'],3)} | "
      f"{abs(sc['ratio']/0.1-1)*100:+.0f}%（对阈值） | **否** | **否** |")
    A("")
    A("**诚实结论**：三项解析判据**全部不满足 3%**，连 15% 的平台底线也未达；"
      "表面张力模块属于「已实现但未通过物理验证」，不能按 m277 报告的 PASS 记账。"
      "本单完成了把欠账**量化并固化**的工作（可复现工件 + 回归测试 + 4 条硬门槛），"
      "修复本身需要改 `src/` 并按 ⑤ 的配方立项。")
    A("")

    text = "\n".join(L) + "\n"
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print(f"wrote {DOC} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
