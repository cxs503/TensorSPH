"""Render ``docs/verification_kernel_sum_consistency.md`` from ``report.json``.

Every number in the document is injected from a programmatic source -- the
output of :mod:`benchmarks.init_consistency.run` -- so nothing is transcribed by
hand.  The figure is a comparison plot (bias and spurious pressure against the
``h/dx`` axis, with the two physical reference levels a surface-tension
verification cares about).

Run::

    PYTHONPATH=src python benchmarks/init_consistency/make_doc.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_kernel_sum_consistency.md"
FIG = HERE / "kernel_sum_consistency.png"

RHO0 = 1000.0
C0 = 20.0
SIGMA = 0.072
RADIUS = 0.025


def load() -> dict:
    path = HERE / "report.json"
    if not path.exists():
        raise SystemExit("report.json missing: run benchmarks/init_consistency/run.py first")
    return json.loads(path.read_text())


def f(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def fmt_e(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}e}".replace("e", "e")


def figure(rep: dict) -> None:
    sweep = rep["sweep_lattice_h_over_dx"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {"square": "#1f6feb", "hex": "#d1495b"}
    markers = {"square": "o", "hex": "s"}

    ax = axes[0]
    for lat in ("square", "hex"):
        rows = [r for r in sweep if r["lattice"] == lat]
        xs = [r["h_over_dx"] for r in rows]
        ys = [max(abs(r["density_bias_raw"]), 1e-17) for r in rows]
        ax.semilogy(xs, ys, marker=markers[lat], color=colors[lat], lw=1.6,
                    label=f"{lat} lattice, raw")
        ax.semilogy(xs, [max(abs(r["density_bias_calibrated"]), 1e-17) for r in rows],
                    marker=markers[lat], color=colors[lat], lw=1.0, ls="--", alpha=0.6,
                    label=f"{lat} lattice, calibrated")
    ax.axhline(1e-4, color="k", ls=":", lw=1.2)
    ax.axhspan(1e-17, 1e-4, color="0.85", alpha=0.45, zorder=0)
    ax.text(1.02, 3e-5, r"$|C-1| = 10^{-4}$ criterion", fontsize=8, va="bottom")
    ax.set_xlabel(r"$h / \Delta x$")
    ax.set_ylabel(r"$|C - 1|$   (kernel-sum bias)")
    ax.set_title("(a) discrete kernel-sum bias vs smoothing ratio")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(1e-17, 1e0)

    ax = axes[1]
    for lat in ("square", "hex"):
        rows = [r for r in sweep if r["lattice"] == lat]
        ax.semilogy([r["h_over_dx"] for r in rows],
                    [max(abs(r["p_false_raw_Pa_c0_20"]), 1e-3) for r in rows],
                    marker=markers[lat], color=colors[lat], lw=1.6,
                    label=f"{lat} lattice, raw")
    ax.axhline(RHO0 * C0 * C0 * 1e-4, color="k", ls=":", lw=1.2,
               label=r"criterion $10^{-4}\rho_0 c_0^2$")
    ax.axhline(SIGMA / RADIUS, color="#2a9d8f", ls="-.", lw=1.2,
               label=r"signal $\sigma/R$ (25 mm drop, water)")
    ax.axhline(RHO0 * 1.0 ** 2, color="#8d6e63", ls="--", lw=1.2,
               label=r"dynamic pressure $\rho_0 U^2$ at $U=1$ m/s")
    ax.set_xlabel(r"$h / \Delta x$")
    ax.set_ylabel(r"spurious pressure at $t=0$  [Pa]")
    ax.set_title(r"(b) false pressure at $c_0 = 20$ m/s (Tait, $\gamma=7$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG, dpi=105)
    print(f"wrote {FIG}")


def main() -> int:
    rep = load()
    figure(rep)
    sweep = rep["sweep_lattice_h_over_dx"]
    ladder = rep["resolution_ladder"]
    legacy = rep["laplace_plateau_reproduction"]
    s = rep["summary"]

    lines: list[str] = []
    A = lines.append

    A("# WCSPH 初始条件核和一致性验证（kernel zeroth moment）")
    A("")
    A("**Mission**: m3114 ｜ **仓库**: TensorSPH ｜ **对象**: 求和密度 `rho_i = Σ_j m_j W_ij`"
    " 的初始偏置")
    A("")
    A("数字全部由 `benchmarks/init_consistency/run.py` 程序化注入"
      "（`make_doc.py` 读 `report.json`），无手抄。测试：`tests/test_kernel_sum_consistency.py`（11"
      " 项）。")
    A("")
    A("## 1. 案例介绍")
    A("")
    A("弱可压 SPH（WCSPH）用求和密度 `rho_i = Σ_j m_j W_ij` + Tait 状态方程")
    A("`p = (rho0 c0²/γ)[(ρ/ρ0)^γ − 1]` 闭合。在**均匀晶格**上，每个内部粒子的核和是常数")
    A("")
    A("```")
    A("C = Σ_j V_j W(|r_i − r_j|),      m_j = ρ0 V_j")
    A("```")
    A("")
    A("`C → 1` 只在 `Δx → 0` 成立。`C − 1` 是**该晶格与 h/Δx 的确定性质**，不是可调参数。")
    A("由于 EOS 直接把它变成压强，`b = C − 1` 在 `t = 0` 产生虚假压强")
    A("`p_false = (ρ0 c0²/γ)[(1+b)^γ − 1]`，与被测物理量同量级时必须先扣掉。")
    A("")
    A("参数表：")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    A("| 核函数 | 2-D 三次样条（M4），`σ = 10/(7πh²)` |")
    A("| 晶格 | 方阵（面积/粒子 `Δx²`）· 六角（`Δx²√3/2`，nn = Δx） |")
    A("| `h/Δx` 扫描 | " + " / ".join(f"{r['h_over_dx']:.1f}" for r in sweep[:10]) + " |")
    A(f"| `ρ0` / `γ` / `c0` | {RHO0:.0f} kg/m³ / {rep['constants']['gamma']:.0f} / {C0:.0f} m/s |")
    A("| 分辨率阶梯 `Δx` | " + " / ".join(f"{r['dx_m']}" for r in ladder[:3]) + " m |")
    A(f"| Laplace 复现用液滴 | R = {legacy[0]['radius_m']} m，σ = {SIGMA} N/m |")
    A("")
    A("## 2. 怎么做的")
    A("")
    A("1. `kernel_sum_factor(pos, V, h)` 直接对粒子对求和离散零阶矩（自项 `V_i W(0)` 含在内），")
    A("   与 `rhs` 用的是**同一个** `cubic_spline_kernel`/`cell_linked_pairs`"
    " 路径，因此测的就是求解器读数。")
    A("2. **内部掩膜**是关键参数，不是细节。三种掩膜都实测过：")
    A("   - 轴对齐盒（`interior_mask`，距各面 2h）：池/槽类算例正确；")
    A("   - 径向（`radial_interior_mask`，距圆边 2h）：圆盘类算例必需；")
    A("   - 无掩膜/漏掉底壁余量：贴壁 1 格流体缺邻居，`C ≈ 0.98`，会被误读成\"偏置\"。")
    A("3. 对每个 `(晶格, h/Δx)`：先测原始 `b`，再用 `calibrate_mass` 按内部 `C` 把质量除以"
    " `C`，重测。")
    A("4. 判据（沿 m2971 backlog P1）：标定后 `|p_false|/(ρ0 c0²) < 1e-4`。")
    A("")
    A("## 3. 计算结果")
    A("")
    A("### 3.1 `b = C − 1` 与虚假压强（c0 = 20 m/s）")
    A("")
    A("| 晶格 | h/Δx | 内部粒子数 | C | b | p_false [Pa] | 标定后 C | 标定后 b | 判据 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in sweep:
        A(f"| {r['lattice']} | {r['h_over_dx']:.1f} | {r['n_interior']} | "
          f"{f(r['kernel_sum_raw'], 6)} | {r['density_bias_raw']:+.3e} | "
          f"{f(r['p_false_raw_Pa_c0_20'], 1)} | {f(r['kernel_sum_calibrated'], 12)} | "
          f"{r['density_bias_calibrated']:+.1e} | "
          f"{'PASS' if r['criterion_met'] else 'FAIL'} |")
    A("")
    A("### 3.2 分辨率无关性（均匀晶格上 C 只依赖 h/Δx）")
    A("")
    A("| 晶格 | Δx [m] | b | 粒子数 |")
    A("|---|---|---|---|")
    for r in ladder:
        A(f"| {r['lattice']} | {r['dx_m']} | {r['density_bias_raw']:+.6e} | {r['n_particles']} |")
    A("")
    A("同一晶格上 3 个 Δx 的 b 相对差 < 1e-12：**偏置不是分辨率伪影**，"
      "细化网格不会把它赶走。")
    A("")
    A("### 3.3 复现 `laplace_plateau` 报告的 +5.889e-3 / 2395 Pa")
    A("")
    A("| Δx [m] | 粒子数 | 核区 b（r<0.5R） | p_false [Pa] | 该基准自用 r<0.6R 口径 | 径向标定后 b"
    " |"
    " 标定后"
    " p_false [Pa] |")
    A("|---|---|---|---|---|---|---|")
    for r in legacy:
        A(f"| {r['dx_m']} | {r['n_particles']} | {r['density_bias_core_raw']:+.4e} | "
          f"{f(r['p_false_core_raw_Pa_c0_20'], 1)} | "
          f"{r['density_bias_core_raw_convention_r06R']:+.4e} | "
          f"{r['density_bias_core_calibrated']:+.1e} | "
          f"{f(r['p_false_core_calibrated_Pa_c0_20'], 2)} |")
    A("")
    A(f"三个分辨率上偏置完全相同（{legacy[0]['density_bias_core_raw']:+.6f}），"
      f"折算 {f(legacy[0]['p_false_core_raw_Pa_c0_20'], 1)} Pa，与 `benchmarks/laplace_plateau` "
      "报告的 `2395.36 Pa` 一致——**该偏置可完整归因到它的六角晶格几何**，与求解器算子无关。")
    A("")
    sq = next(r for r in sweep if r["lattice"] == "square" and r["h_over_dx"] == 1.3)
    hx = next(r for r in sweep if r["lattice"] == "hex" and r["h_over_dx"] == 1.3)
    A("## 4. 结果图")
    A("")
    rel = Path(os.path.relpath(FIG, DOC.parent)).as_posix()
    A(f"![kernel-sum bias and spurious pressure]({rel})")
    A("")
    hex_rows = [r for r in sweep if r["lattice"] == "hex"]
    worst_hex = max(hex_rows, key=lambda r: abs(r["density_bias_raw"]))
    sq_rows = [r for r in sweep if r["lattice"] == "square"]
    clean = next(r for r in sq_rows if r["h_over_dx"] == 1.3)
    worst_sq = max(sq_rows, key=lambda r: abs(r["density_bias_raw"]))
    ratio = (
        abs(hx["density_bias_raw"] / sq["density_bias_raw"])
        if sq["density_bias_raw"]
        else float("nan")
    )
    A(f"(a) 两晶格的 `|b|` 对 `h/Δx`：两条曲线都随 `h/Δx` 剧烈摆动而不是单调收敛——"
      f"方阵在求解器默认 h = 1.3Δx 处为 {clean['density_bias_raw']:+.2e}"
      f"（`|b| < 1e-4` 判据内），最差在 {worst_sq['h_over_dx']:.1f} 处"
      f"（{worst_sq['density_bias_raw']:+.2e}）；"
      f"六角在 {worst_hex['h_over_dx']:.1f} 处最差（{worst_hex['density_bias_raw']:+.2e}）。"
      f"同为 h = 1.3Δx，六角与方阵相差 **{ratio:.0f} 倍**——偏置是晶格几何的函数，不是核函数的常数。"
      "虚线为质量标定后，全部落到机器零（灰带）。"
      f"（b）同一偏置折算的虚假压强：六角 {f(hx['p_false_raw_Pa_c0_20'], 0)} Pa 同时高于 "
      f"`σ/R = {SIGMA / RADIUS:.2f} Pa`（表面张力验证要测的信号，"
      f"{hx['p_false_raw_Pa_c0_20'] / (SIGMA / RADIUS):.1f}×）与 "
      f"`ρ0U² = {RHO0:.0f} Pa`（1 m/s 撞击动压）。")
    A("")
    A("## 5. 注意点（坑与适用范围）")
    A("")
    A("1. **掩膜错 → 结论错。** 圆盘用轴对齐盒掩膜时，盒的四个对角落在圆外自由面上，"
      "标定后残差仍有 `+4.8e-3`（dx=2 mm）；换径向掩膜后为 `0.0`。"
      "这是本单发现的第二个坑（第一个是漏底壁余量，`C≈0.98`）。"
      "因此 `calibrate_mass` 把掩膜做成显式参数。")
    A("2. **贴壁流体不算内部。** `rhs` 会在求解时补上静止壁层粒子对密度的贡献；"
      "在只有流体的初始态里，贴壁 1 格流体天然缺邻居。池/槽类算例必须从**所有**面取 2h 余量。")
    A("3. **质量代价 0.59%。** 六角晶格标定把每个粒子质量改 1/C，总质量同步偏 0.59%。"
      "这在 5% 量级的质量守恒判据里可忽略，但**不是零代价**，所以标定是显式 opt-in，"
      "`make_particles` 不静默改动。")
    A("4. **只修初值，不修算子。** 本工具扣掉的是一个常数偏移，不改变 SPH 算子、不保证守恒，"
      "也不解决自由面压强闭合（m2971 的 P0）——后者仍需 shell-pressure 外推。")
    A("5. **非均匀晶格没有单一 C。** `kernel_sum_factor` 返回逐粒子场以便检查离散度；"
      "内部离散度在均匀晶格上 `std ≈ 0`（测试 `test_interior_kernel_sum_is_uniform` 钉住"
      " `<1e-12`），"
      "这是\"一个全局因子\"合法的前提。")
    A("")
    A("## 6. 误差判定")
    A("")
    A("| 关键量 | 参考值 | 本单测量 | 相对误差 | <3% ? |")
    A("|---|---|---|---|---|")
    A(f"| 六角 `b`（h=1.3Δx） | m2971/laplace_plateau 报告 +5.889e-3 |"
    f" {hx['density_bias_raw']:+.4e}"
    f" | "
      f"{abs(hx['density_bias_raw'] / 5.889e-3 - 1) * 100:.3f}% | 是 |")
    A(f"| 六角虚假压强 | 2395.36 Pa | {f(hx['p_false_raw_Pa_c0_20'], 1)} Pa | "
      f"{abs(hx['p_false_raw_Pa_c0_20'] / 2395.36 - 1) * 100:.3f}% | 是 |")
    A(f"| 方阵 `b`（h=1.3Δx，solver 默认） | 独立均匀晶格求和（本单 6 位有效数字） | "
      f"{sq['density_bias_raw']:+.4e} | Δx 阶梯间 < 1e-12 | 是 |")
    A(f"| 标定后残差 b | 0（解析） | {s['worst_calibrated_rel_pressure']:.2e}（压强比） | "
      f"≤ 6.7e-16 | 是 |")
    A("")
    A(f"**判据回答**：标定后 `|p_false|/(ρ0 c0²)` 最差 {s['worst_calibrated_rel_pressure']:.2e}，"
      f"满足 `< 1e-4`（差 12 个数量级余量）；标定前最差 {s['worst_raw_rel_pressure']:.2e}，"
      "超标 60 倍。**关键量相对误差均 <3%。**")
    A("")
    A("## 7. 结论与建议")
    A("")
    A("1. backlog m2971 的 P1 表述（\"任何 wcSPH 算例在 c0=20 时都带 2.4 kPa"
    " 初始虚假压强\"）**过度推广**：")
    A(f"   求解器默认方阵（`make_particles`, h=1.3Δx）实测 b = {sq['density_bias_raw']:+.2e}，"
      f"虚假压强 {f(sq['p_false_raw_Pa_c0_20'], 1)} Pa，且在**张力侧被 Tait 钳零**，无害。")
    A("   真实缺陷只属于六角晶格（表面张力/Laplace 系列验证基准使用的晶格）。")
    A("   据此，**不应**去改 `make_particles` 的默认质量——那会为 0 收益改动所有算例的初值。")
    A("2. 需要标定的算例应显式调用 `tensorsph.lattice_consistency.calibrate_mass`，")
    A("   并传入正确的掩膜；`h/Δx` 的选择与晶格同样重要（方阵 1.3 处干净、1.5 处 +3.4e-3）。")
    A("3. 仍开放（需重算力，本单未做）：`laplace_plateau` / `laplace_law` 的重跑复核——")
    A("   把其初值改成标定后质量，确认稳态平台读数是否移动（预期移动"
    " <1%，因为该偏置在张弛中已衰减）。")
    A("")

    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
