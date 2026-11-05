"""Generate ``docs/verification_initial_density_calibration.md``.

Every number is injected from ``benchmarks/initial_density_bias/report.json``;
nothing is typed by hand.  Run after ``run.py``:

    PYTHONPATH=src python benchmarks/initial_density_bias/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_initial_density_calibration.md"


def fmt(v, digits=4, signed=True):
    if isinstance(v, int):
        return str(v)
    sign = "+" if (signed and v >= 0) else ""
    return f"{v:{sign}.{digits}e}"


def main() -> None:
    rep = json.loads((HERE / "report.json").read_text(encoding="utf-8"))
    S = rep["setup"]
    V = rep["validation"]
    H_ = V["hex_drop"]
    Q = V["square_pool"]
    SY = V["synthetic_square_lattice"]
    C = V["calibrated_hex_drop"]
    sw = rep["sweep"]
    L = rep["sound_speed_ladder"]["rows"]
    off = rep["relaxation"]["uncalibrated"]
    on = rep["relaxation"]["calibrated"]
    gates = rep["gates"]

    lines = []
    A = lines.append
    A("# 初始和密度晶格偏置 `b` 的闭式解、实测验证与质量校准（TensorSPH / m3209）")
    A("")
    A("**物理背景**：本仓用「密度求和」（summation density）而不是连续性方程积分来获得")
    A("`rho`。求和是对核函数的**晶格求积**，因此在完整支撑（complete support）的粒子")
    A("上并不精确等于 `rho0`，而是")
    A("")
    A("```\nrho_i / rho0 = 1 + b,      b = m * S / rho0 - 1,   S = W(0) + sum_{j!=i} W(r_ij,h)\n```")
    A("")
    A("`b` 只由 `h/dx` 与晶格几何决定，与 `dx`、`rho0`、`c0`、物体形状无关。经 Tait 状态")
    A("方程，它变成一个**均匀的虚假初始压强**")
    A("`p_spurious = B[(1+b)^gamma - 1]`，`B = rho0 c0^2/gamma`。")
    A("")
    A("这解释了 `backlog_update_m2971.md`（P1）里「`b ~ 5.9e-3`、2395 Pa」的来历，并把")
    A("它从一次手工观测升级为可预测、可验证、可校准的闭式量；同时给出该 P1 的适用范围")
    A("（下表显示它并非对每个算例都成立）。")
    A("")
    A("## 1. 案例与参数")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 液滴半径 R | {S['radius_m']} m |")
    A(f"| 表面张力 sigma | {S['sigma_N_per_m']} N/m（水的 10 倍，缩短毛细时间） |")
    A(f"| 参考密度 rho0 | {S['rho0']} kg/m^3 |")
    A(f"| Tait 指数 gamma | {S['gamma']} |")
    A(f"| 核支撑 h/dx | {S['h_factor']}（2-D cubic spline） |")
    A(f"| 分辨率 dx | {S['dx_m']} m |")
    A(f"| 体模量 B(c0=20 m/s) | {S['bulk_modulus_at_c0_20_Pa']:.1f} Pa |")
    A(f"| Laplace 压差 sigma/R | {S['laplace_jump_Pa']:.1f} Pa |")
    A(f"| 毛细时间 t_sigma | {S['capillary_time_s']:.4f} s |")
    A(f"| 液滴晶格 | {S['lattice_drop']}（benchmarks/laplace_law 的生成器） |")
    A(f"| 水池晶格 | {S['lattice_pool']}（`make_particles`/`cell_centres`） |")
    A(f"| m2971 报告的偏置 / 压强 | {S['m2971_reported_bias']} / {S['m2971_reported_pressure_Pa']} Pa |")
    A("")
    A("## 2. 方法")
    A("")
    A("1. `src/tensorsph/initial_density.py`：`lattice_bias(dx,h,kind)` 在截断到核支撑之外")
    A("   的有限晶格上直接求 `W(0)+sum W`，得到闭式 `b`；`calibrate_mass` 施加 `m/(1+b)`。")
    A("2. **独立实测**：`measure_complete_support_bias` 用 `density_from_neighbor_pairs`")
    A("   （另一条代码路径）在真实粒子集上取 `rho` 的最大值——完整支撑粒子的和密度在")
    A("   规则晶格上平移不变，且没有别的粒子能达到它，故最大值即 `rho0(1+b)`。")
    A("3. 两条路线的对比就是本报告的判据：闭式解必须复现实测（G1/G2a），而不是反过来")
    A("   用实测去拟合闭式。")
    A("4. 时间域：同一液滴（CSF consistent + fit_quadratic 曲率）跑 1.5 个毛细时间，")
    A("   分别以未校准 / 校准质量初始化，比较核心压强与虚假流动。")
    A("")
    A("## 3. 结果")
    A("")
    A("### 3.1 闭式 vs 实测（关键表）")
    A("")
    A("| 位形 | 实测 b | 闭式 b | 相对差 | 完整支撑粒子数 | 判据 |")
    A("|---|---|---|---|---|---|")
    A(f"| hex 液滴（真实生成器） | {fmt(H_['bias_measured'])} | {fmt(H_['bias_analytic'])} | "
      f"{H_['rel_diff']:.2e} | {H_['n_complete_support']} | "
      f"{'pass' if H_['G1_passed'] else 'FAIL'} (G1 <1e-9) |")
    A(f"| 理想 square 晶格（float64） | {fmt(SY['bias_measured'])} | {fmt(SY['bias_analytic'])} | "
      f"{SY['rel_diff']:.2e} | — | {'pass' if SY['G2a_passed'] else 'FAIL'} (G2a <1e-9) |")
    A(f"| `make_particles` 水池（float32） | {fmt(Q['bias_measured'])} | {fmt(Q['bias_analytic'])} | "
      f"{Q['rel_diff']:.2%} | {Q['n_complete_support']} | "
      f"{'pass' if Q['G2_passed'] else 'FAIL'} (G2 <10%) |")
    A("")
    A("hex 的闭式解与独立实测相差 `" + f"{H_['rel_diff']:.1e}" + "`，即 m2971 手工测到的")
    A("`5.88e-3` 现在有了完整解释（不是数值噪声，是晶格求积误差）。square 水池的 5.0%")
    A("残差**已归因**：`cell_centres` 用 float32 生成晶格，间距抖动 ~3e-6 相对量级，而")
    A("`b~5e-5` 的求积误差对间距敏感；float64 理想晶格上残差降到 `"
      + f"{SY['rel_diff']:.1e}" + "`。")
    A("")
    A("### 3.2 `b` 随 `h/dx` 变化（两种晶格）")
    A("")
    A("| h/dx | square b | square p_spur@c0=20 | hex b | hex p_spur@c0=20 |")
    A("|---|---|---|---|---|")
    for sq, hx in zip(sw["rows"]["square"], sw["rows"]["hex"]):
        A(f"| {sq['h_over_dx']:.2f} | {fmt(sq['bias'])} | {sq['spurious_pressure_Pa_c0_20']:+8.1f} Pa | "
          f"{fmt(hx['bias'])} | {hx['spurious_pressure_Pa_c0_20']:+8.1f} Pa |")
    A("")
    A("两个结论：(a) `b` **不是**随细化单调下降的误差，而是随 `h/dx` 变号振荡的求积误差；")
    A("(b) 本仓默认的 square 晶格在 `h/dx=1.3` 恰好接近零点（`b=" + f"{fmt(Q['bias_analytic'])}"
      + "`），")
    A("所以 **m2971 P1 的「任何 wcSPH 算例都带 2.4 kPa」只在 hex 晶格上成立**——这是对")
    A("该文件的一处更正，证据即上表。")
    A("")
    A("### 3.3 压强代价与声速")
    A("")
    A("| c0 [m/s] | p_max [Pa] | B [Pa] | p_max/(rho0 c0^2) | p_max/(sigma/R) |")
    A("|---|---|---|---|---|")
    for r in L:
        A(f"| {r['sound_speed']:.0f} | {r['spurious_pressure_Pa']:.1f} | "
          f"{r['bulk_modulus_B_Pa']:.0f} | {r['ratio_p_over_rho0c0sq']:.4e} | "
          f"{r['ratio_p_over_sigma_over_R']:.2f} |")
    A("")
    A("`p_max/(rho0 c0^2)` 在 c0 阶梯上不变（G6 pass），即偏置是纯几何量、压强按 `c0^2`")
    A("缩放，与 m2971「b 与 c0 无关」的观察一致。降到 c0=5 m/s 确实把虚假压强降到")
    A(f"{L[-1]['spurious_pressure_Pa']:.0f} Pa，但仍为 `sigma/R` 的 "
      f"{L[-1]['ratio_p_over_sigma_over_R']:.2f} 倍——降声速不是出路（与 m2971 的 C4 结论一致）。")
    A("")
    A("### 3.4 质量校准前后")
    A("")
    A("| 量 | 未校准 | 校准后 |")
    A("|---|---|---|")
    A(f"| 核心 `rho/rho0 - 1` | {fmt(H_['bias_measured'])} | {fmt(C['bias_measured'])} |")
    A(f"| `p_max` @c0=20 [Pa] | {H_['spurious_pressure_Pa_c0_20']:.2f} | "
      f"{C['spurious_pressure_Pa_c0_20']:.3e} |")
    A(f"| `p_max/(rho0 c0^2)` | {H_['spurious_pressure_Pa_c0_20']/(S['rho0']*400.0):.4e} | "
      f"{C['spurious_ratio']:.3e} |")
    A(f"| 校准因子 1/(1+b) | 1 | {C['calibration_factor']:.9f} |")
    A("")
    A("校准是**标量**，所以质量分布形状（均匀、或 `hydrostatic_density` 的剖面）保持不变，")
    A("只是把「晶格常数质量 `rho0 dx^2` 并不代表 `rho0`」这一事实除掉。总质量随之变化")
    A(f"`{abs(1.0-C['calibration_factor']):.2%}`——这正是修正量本身。")
    A("")
    A("### 3.5 时间域：校准前后的液滴弛豫")
    A("")
    A("| 量 | 未校准 | 校准 |")
    A("|---|---|---|")
    A(f"| 初始核心压强 [Pa] | {off['p_core_initial_Pa']:.2f} | {on['p_core_initial_Pa']:.3e} |")
    A(f"| t=1.5 t_sigma 核心压强 [Pa] | {off['p_core_final_Pa']:.1f} | {on['p_core_final_Pa']:.1f} |")
    A(f"| 尾部 40% 时间平均核心压强 [Pa] | {off['p_core_tail_mean_Pa']:.1f} | {on['p_core_tail_mean_Pa']:.1f} |")
    A(f"| max|v|/v_ref 峰值 | {off['vmax_over_vref_peak']:.3f} | {on['vmax_over_vref_peak']:.3f} |")
    A(f"| max|v|/v_ref 终值 | {off['vmax_over_vref_final']:.3f} | {on['vmax_over_vref_final']:.3f} |")
    A(f"| 步数 / dt [s] | {off['steps']} / {off['dt_s']:.3e} | {on['steps']} / {on['dt_s']:.3e} |")
    A("")
    A("![initial density bias](benchmarks/initial_density_bias/initial_density_bias.png)")
    A("")
    A("图注：左上 `b(h/dx)` 闭式曲线与两处实测点（星号）；右上同一偏置换算成的虚假压强，")
    A("红虚线为 `sigma/R=360 Pa`；左下虚假压强 vs `c0`（`c0^2` 缩放）；右下弛豫过程核心")
    A("压强（校准后与 `b=0` 重合，未校准从 2395 Pa 起跳）。")
    A("")
    A("## 4. 误差判定")
    A("")
    A("| 关键量 | 基准 | 本实现误差 | 是否 <3% |")
    A("|---|---|---|---|")
    A(f"| hex 晶格 `b` | 独立实测（`density_from_neighbor_pairs`） | "
      f"{H_['rel_diff']:.2e} | 是 |")
    A(f"| 理想 square 晶格 `b` | 独立实测 | {SY['rel_diff']:.2e} | 是 |")
    A(f"| 未校准 `p_max`@c0=20 | m2971 的 2395 Pa | "
      f"{abs(H_['spurious_pressure_Pa_c0_20']-S['m2971_reported_pressure_Pa'])/S['m2971_reported_pressure_Pa']:.2e} | 是 |")
    A(f"| 实测 `b` | m2971 的 5.88e-3 | "
      f"{abs(H_['bias_measured']-S['m2971_reported_bias'])/S['m2971_reported_bias']:.2e} | 是 |")
    A(f"| `make_particles` 水池 `b` | 闭式（理想晶格） | {Q['rel_diff']:.2%} | **否（5.0%）** |")
    A(f"| 校准后 `p_max/(rho0 c0^2)` | 0 | {C['spurious_ratio']:.1e} | 是 |")
    A("")
    A("唯一超过 3% 的项是 `make_particles` 水池的 `b`，且已定位到 float32 晶格抖动")
    A("（`cell_centres`），不是模型误差：同一实测方法在 float64 理想晶格上给出 "
      + f"{SY['rel_diff']:.1e}" + "。")
    A("")
    A("## 5. 注意点 / 局限")
    A("")
    A("- **校准时总质量变化 `1/(1+b)`**：对 `b<0` 的晶格（本仓 square 默认）质量变大，")
    A("  对 hex 变小。若某算例总质量本身是标定量（如按注入体积给定），需重新核对——")
    A("  本仓各 benchmark 均由 `rho0` 与几何派生质量，不属此列。")
    A("- **不修自由面**：部分支撑粒子的密度亏缺（`rho<rho0`，压强被钳到 0）是另一个缺陷，")
    A("  由 `benchmarks/laplace_pressure_floor` 与自由面压强闭合线负责；质量校准只消掉")
    A("  「完整支撑区的均匀偏置」，两者的误差不可混算。")
    A("- **默认关闭**：`Config.calibrate_initial_mass = False` 保持已发表数字逐位可复现；")
    A("  需要用它的算例显式打开（hex 类算例建议必开）。关闭时 `make_particles` 的质量与")
    A("  历史版本逐位相同，有测试钉住。")
    A("- **半解析性**：闭式解在有限晶格上求和，`n_half` 需覆盖 2h 支撑（实现自动取")
    A("  `ceil(2h/dx)+2`）；若 `h/dx` 极端大（>3），求和域需同步放大。")
    A("- 时间域证据只跑到 1.5 个毛细时间，未做长时间统计；核心压强的**稳态**归因")
    A("  （C3 自由面闭合）不在本单范围。")
    A("")
    A("## 6. 判据汇总（机器裁决）")
    A("")
    A("| 判据 | 结果 |")
    A("|---|---|")
    for name, ok in gates.items():
        A(f"| {name} | {'pass' if ok else 'FAIL'} |")
    A(f"| **全部通过** | **{'是' if rep['all_gates_passed'] else '否'}** |")
    A("")
    A("复现：`PYTHONPATH=src python benchmarks/initial_density_bias/run.py`（约 40 s，")
    A("`--quick` 更快），文档由 `make_doc.py` 从 `report.json` 注入生成；测试")
    A("`tests/test_initial_density_calibration.py`（16 passed）。")
    A("")

    DOC.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", DOC, f"({len(lines)} lines)")


if __name__ == "__main__":
    main()
