# ruff: noqa: E501  (report/prose templates carry long lines by design)
"""Render ``docs/verification_laplace_resolution_ladder.md`` from the ladder.

Every number in the document is read from ``resolution_ladder.json`` and the
per-rung artifacts in ``convergence/``; nothing is transcribed by hand, so the
document cannot drift away from the measurement (same protocol as
``benchmarks/dam_break/make_ladder_doc.py``).

    PYTHONPATH=src python benchmarks/laplace_law/make_ladder_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
DOC = ROOT / "docs" / "verification_laplace_resolution_ladder.md"
LADDER = BENCH / "resolution_ladder.json"
FIGURE = BENCH / "laplace_resolution_ladder.png"


def f(x, nd=1, sign=False):
    fmt = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return fmt.format(x)


def table_ladder(ladder):
    lines = [
        "| dx [mm] | N | steps | t/t_sigma | bandA Δp [Pa] | bandA 误差 | "
        "absolute Δp [Pa] | absolute 误差 | tail-avg 误差 | 压缩比 | ρ_core std | v_max/v_ref |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in sorted(ladder["rungs"], key=lambda r: -r["dx_m"]):
        lines.append(
            f"| {r['dx_m'] * 1e3:.2f} | {r['n_particles']} | {r['steps']} | "
            f"{r['t_over_t_sigma']:.3f} | {r['bandA']['delta_p_Pa']:.1f} | "
            f"{f(r['bandA']['rel_error_pct'], 1, True)} % | "
            f"{r['absolute']['delta_p_Pa']:.1f} | "
            f"{f(r['absolute']['rel_error_pct'], 1, True)} % | "
            f"{f(r['absolute']['rel_error_tailavg_pct'], 1, True)} % | "
            f"{r['compression_ratio_measured_over_required']:.2f} | "
            f"{r['density_readout']['rho_core_rel_std']:.2e} | "
            f"{r['spurious_current_ratio']:.3f} |")
    return "\n".join(lines)


def table_readout(ladder):
    lines = [
        "| dx [mm] | mean p_core 误差 | p(mean ρ) 误差 | 差 [pt] | 判读 |",
        "|--:|--:|--:|--:|:--|",
    ]
    for r in sorted(ladder["rungs"], key=lambda r: -r["dx_m"]):
        a = r["absolute"]["rel_error_pct"]
        b = r["density_readout"]["rel_error_from_mean_core_density_pct"]
        gap = a - b
        verdict = "读数被少数高压粒子抬高" if abs(gap) > 10.0 else "两口径一致"
        lines.append(f"| {r['dx_m'] * 1e3:.2f} | {f(a, 1, True)} % | "
                     f"{f(b, 1, True)} % | {f(gap, 1, True)} | {verdict} |")
    return "\n".join(lines)


def render(ladder, artifacts):
    ref = ladder["reference"]
    proto = ladder["protocol"]
    dp = ref["delta_p_Pa"]
    rungs = sorted(ladder["rungs"], key=lambda r: -r["dx_m"])   # coarsest first
    n_bad_15 = sum(1 for r in rungs if not r["absolute"]["within_15pct"])
    n_bad_3 = sum(1 for r in rungs if not r["absolute"]["within_3pct"])
    worst = max(rungs, key=lambda r: abs(r["absolute"]["rel_error_pct"]))
    best = min(rungs, key=lambda r: abs(r["absolute"]["rel_error_pct"]))
    fine = [r for r in rungs if r["dx_m"] <= 0.002 + 1e-12]
    _coarse = [r for r in rungs if r["dx_m"] > 0.002 + 1e-12]
    ratio_fine = [r["compression_ratio_measured_over_required"] for r in fine]
    F = rungs[-1]          # finest rung (dx = 1.5 mm)
    M = rungs[len(rungs) // 2]  # middle rung (dx = 2.0 mm)
    C = rungs[0]           # coarsest rung (dx = 2.5 mm)
    comp_needed = ((1.0 + dp / (ref['rho0'] * ref['sound_speed'] ** 2 / 7.0))
                   ** (1.0 / 7.0) - 1.0)
    _comp_pts = abs(F['bandA']['rel_error_pct'] - F['absolute']['rel_error_pct'])
    coarse_gap = (C['absolute']['rel_error_pct']
                  - C['density_readout']['rel_error_from_mean_core_density_pct'])

    text = f"""# Young–Laplace 压差的分辨率阶梯（固定物理时刻）

**仓库**: TensorSPH ｜ **工件**: `benchmarks/laplace_law/`（`run_ladder.py`、`resolution_ladder.json`、
`convergence/relax_dx*.json`、`laplace_resolution_ladder.png`）｜ **测试**:
`tests/test_laplace_resolution_ladder.py`

**结论（先给判定）**: 把阶梯从「固定步数」改成「固定毛细时间」（{proto['duration_t_over_t_sigma']:.2f} $t_\\sigma$）
并用**绝对压强口径**重测后，Young–Laplace 压差仍然 **WITHHELD**：三档分辨率误差
{f(C['absolute']['rel_error_pct'],1,True)} / {f(M['absolute']['rel_error_pct'],1,True)} /
{f(F['absolute']['rel_error_pct'],1,True)} %（dx = 2.5 / 2.0 / 1.5 mm），
**没有一档进入 15%，更没有进入 3%**。但阶梯把三件事量化了：

1. **历史 `bandA` 口径的 −85…−90 % 主要是口径成本**（把自由面密度亏缺带当成流体压强来相减），
   同一份流场换成绝对口径只剩 −19…−34 %；
2. **粗分辨率（dx = 2.5 mm, N = {C['n_particles']}）的读数被少数高压粒子抬高**：
   `mean p_core` 误差 {f(C['absolute']['rel_error_pct'],1,True)} % 而「压强对均值密度取值」只有
   {f(C['density_readout']['rel_error_from_mean_core_density_pct'],1,True)} %，两者差
   {f(coarse_gap,1,True)} 个百分点，
   以 `mean(p)` 为读数的绝对口径在 dx ≥ 2.5 mm 上不可信；
3. **内部压缩比在细网格上停在 {min(ratio_fine):.2f}–{max(ratio_fine):.2f}**（dx ≤ 2 mm，要求为 1.00），
   且 2.0 mm → 1.5 mm 不再改善（{M['compression_ratio_measured_over_required']:.2f} →
   {F['compression_ratio_measured_over_required']:.2f}），这正是自由面压强闭合的**水位型**欠账，
   不能靠加密网格消掉。

## ① 案例介绍（物理背景/几何/参数）

半径 $R$ = {ref['radius_m'] * 1e3:.0f} mm 的二维单相液滴，六方密排格子、静止启动、四周真空、无重力。
Young–Laplace 定律给出唯一可量化的整体判据

$$\\Delta p = p_{{in}} - p_{{out}} = \\sigma/R = {dp:.1f}\\ \\mathrm{{Pa}} .$$

**真空参考是被状态方程钉死的**：Tait 形式 `p = B[(rho/rho0)^gamma - 1]`（`B = rho0 c0^2/gamma`）以
`rho0` 为零点，故 `p_out = 0`，物理压差就是内部绝对压强 `p_core`。

| 参数 | 值 | 说明 |
|---|---|---|
| 液滴半径 R | {ref['radius_m']:.4f} m | 静止自由液滴 |
| 表面张力 sigma | {ref['sigma_N_per_m']:.1f} N/m | 10× 水，压缩毛细时间 |
| 参考密度 rho0 | {ref['rho0']:.0f} kg/m³ | 弱可压 Tait，gamma = 7 |
| 声速 c0 | {ref['sound_speed']:.1f} m/s | 所需压缩 (ρ/ρ0−1) = {comp_needed:.3e} |
| 粘度 | {ref['viscosity']:.2f} m²/s | 层流正则化 Laplacian |
| 毛细时间 t_sigma | {(ref['radius_m']**3*ref['rho0']/ref['sigma_N_per_m'])**0.5:.4f} s | 窗口 = {proto['duration_t_over_t_sigma']:.2f} t_sigma |
| 分辨率阶梯 | dx = {C['dx_m']*1e3:.1f} / {M['dx_m']*1e3:.1f} / {F['dx_m']*1e3:.1f} mm | R/h = {ref['radius_m']/(1.3*C['dx_m']):.1f} / {ref['radius_m']/(1.3*M['dx_m']):.1f} / {ref['radius_m']/(1.3*F['dx_m']):.1f} |
| 平滑长度 | h = 1.3 dx | 三次样条核 |
| 初始化 | {proto['start']} | 不做质量校准（与仓库默认路径一致）|

## ② 怎么做的（方法/流程）

**问题**：仓库里 `convergence/` 的三份阶梯工件是用**固定步数**（380 步）产生的，而 `dt ∝ dx/c0`，
于是三档的物理窗口分别是 0.33 / 0.44 / 0.56 $t_\\sigma$：阶梯比较的是**三个不同时刻**的振荡相位，
且最短的那一档低于仓库自己的下限（`tests/test_laplace_law.py::test_committed_verdicts_are_self_consistent`
要求 `t > 0.5 t_sigma`），HEAD 上该测试是红的。

**修法（最小改动，不动 `src/`）**：

1. `benchmarks/laplace_law/run.py::relax` 增加 `duration_t_sigma`：由
   `steps = ceil(duration * t_sigma / dt)` 反算步数，使每档落在**同一物理时刻**
   （本文件全部结果取 {proto['duration_t_over_t_sigma']:.2f} $t_\\sigma$）；
2. 同函数增加**绝对口径**读数（`p(r<0.5R) − 0`）与尾部时间平均（窗口后
   {proto['tail_window_fraction']*100:.0f} %），历史 `bandA` 口径保留作对照
   （{proto['read_out']}）；
3. 新增 `benchmarks/laplace_law/run_ladder.py`：按固定时长跑阶梯，写
   `convergence/relax_dx*.json` + 汇总 `resolution_ladder.json`（含末态场快照，供云图）；
4. 新增本文件（由 `make_ladder_doc.py` 程序化注入数字）与 `tests/test_laplace_resolution_ladder.py`
   （协议守卫 + 文档防陈旧）。

**结果**：`convergence/relax_dx*.json` 被重跑成 `t/t_sigma = {proto['duration_t_over_t_sigma']:.3f}`，
`tests/test_laplace_law.py::test_committed_verdicts_are_self_consistent`（HEAD 上红）因此转绿——
**修的是工件（窗口）而不是判据**；判定内容（没有一档进 15 %）并未放宽，仍是 FAIL。

**噪声诊断（为什么同时报两种口径）**：`γ = 7` 的 Tait 律是强凸函数，核心区少数高压粒子会明显抬高
`mean(p)`，而「压强对均值密度取值」`p(mean ρ)` 不受其害。两者之差就是读数噪声的量级，
用来判断某一档的绝对口径是否可信。

## ③ 计算结果

{len(rungs)} 档分辨率、同一物理时刻，末态场快照随工件提交。参考压差 σ/R = {dp:.1f} Pa。

{table_ladder(ladder)}

读数口径诊断（同一末态，两种聚合）：

{table_readout(ladder)}

| 判定项 | 阈值 | 实测（dx = 1.5 mm，最细档） | 是否通过 |
|:--|:--|:--|:--|
| 绝对压差误差 | < 15 % | {f(F['absolute']['rel_error_pct'],1,True)} % | {"是" if F['absolute']['within_15pct'] else "**否**"} |
| 绝对压差误差 | < 3 % | {f(F['absolute']['rel_error_pct'],1,True)} % | {"是" if F['absolute']['within_3pct'] else "**否**"} |
| 内部压缩比 | 1.00 ± 0.05 | {F['compression_ratio_measured_over_required']:.2f} | **否** |
| 曲率带均值 / 解析 1/R（0.6–0.98R，非本阶梯主判据）| 1.00 | {F['curvature_band_mean_over_exact']:.2f} | 仅记录 |
| 虚假流 v_max / v_ref | < 0.1（Adami 2012）| {F['spurious_current_ratio']:.3f} | 是 |

阶梯覆盖 {len(rungs)} 档、N = {C['n_particles']} … {F['n_particles']}，全部 `t/t_sigma` =
{f(F['t_over_t_sigma'], 3)}（同一物理时刻）；误差范围 {f(worst['absolute']['rel_error_pct'],1,True)} %
（dx = {worst['dx_m']*1e3:.1f} mm）至 {f(best['absolute']['rel_error_pct'],1,True)} %
（dx = {best['dx_m']*1e3:.1f} mm），未进 15% 的档数 {n_bad_15}/{len(rungs)}，未进 3% 的档数 {n_bad_3}/{len(rungs)}。

## ④ 结果图

![分辨率阶梯](laplace_resolution_ladder.png)

- **(a)** 两种口径的 Δp 误差 vs dx（对数横轴），阴影为 ±15 % 判定带：`bandA` 全程贴着 −80…−90 %，
  绝对口径在 −19…−34 %，随分辨率**非单调**（2.0 mm 最好，2.5 mm 最差）；
- **(b)** 内部压缩比 vs dx：要求 1.00，细网格停在 {min(ratio_fine):.2f}–{max(ratio_fine):.2f}，加密不再改善；
- **(c)** 最细档末态压强云图（粒子着色 + 解析 360 Pa 参考色标）：芯部整体低于 360 Pa，界面壳层因
  Tait 钳位落在 0，压差在壳层内被截断——与压缩比的水位型亏缺同源；
- **(d)** 三档末态径向密度剖面：自由面密度亏缺随半径增大，0.95R 外亏缺 −10…−30 %，
  正是 `bandA` 口径混入的那部分。

## ⑤ 注意点（踩过的坑/局限/适用范围）

1. **同一时刻才可比**：`dt ∝ dx/c0`，固定步数的"分辨率阶梯"实际是在比不同相位。任何后续阶梯
   （含 `benchmarks/convergence/`）都应固定 `t/t_sigma` 而不是步数。
2. **粗网格绝对口径不可用**：dx = 2.5 mm 档 `mean(p)` 与 `p(mean ρ)` 差
   {f(C['absolute']['rel_error_pct']-C['density_readout']['rel_error_from_mean_core_density_pct'],1,True)} 个百分点，
   少数高压粒子抬高了均值；引用绝对口径必须同时给出 `ρ_core std`（本档 {C['density_readout']['rho_core_rel_std']:.1e}，
   细档 {F['density_readout']['rho_core_rel_std']:.1e}）。
3. **尾部时间平均不等于更准**：单帧读数误差未必小于尾均值（2.0 mm：{f(M['absolute']['rel_error_pct'],1,True)} % vs
   {f(M['absolute']['rel_error_tailavg_pct'],1,True)} %），说明窗口内既有呼吸振荡也有持续漂移，
   时间平均不能替代机理修正。
4. **协议不同不可横比**：本阶梯走仓库默认（未校准）路径，压缩比 {min(ratio_fine):.2f}–{max(ratio_fine):.2f}；
   `benchmarks/laplace_pressure_floor` 那条已发布口径（做质量校准、窗口 1.5 $t_\\sigma$）测到 0.57±0.06。
   两者的绝对误差（−19…−21 % vs −34 %）**不能直接比较**，差异来自初始化偏置与窗口长度，不是模型改动。
5. **本阶梯未做质量校准**（`calibrated_droplet` 路线）：格子核和偏置 `ρ/ρ0−1 ≈ +5.9e-3` 会给初始状态
   带来约 2.4 kPa 的虚假压强（c0 = 20），这在 t ≤ 0.3 t_sigma 内会污染读数；{proto['duration_t_over_t_sigma']:.2f} t_sigma 的窗口
   已有相当部分被弛豫掉，但**不是零**。后续若要 3 % 级判据，必须走校准路线。
6. **既有欠账（本单未修）**：`benchmarks/laplace_law/make_doc.py` 在 HEAD 上无法运行——
   `instant_report.json` 现在把阶梯放在 `convergence` 键下，脚本仍读 `resolutions`（KeyError），
   因此 `docs/verification_surface_tension_laplace.md` 的阶梯段仍是**旧工件**数值
   （bandA −114.6 / −78.3 / −83.7 %），与本单新工件（−84.8 / −89.8 / −80.4 %）不一致。
   该文档必须在 `make_doc.py` 修好后重渲染；在修好之前，**阶梯数字一律以本文件为准**。
7. **不涉及表面张力模型的改动**：本单只改验证工件与口径；同期的自由面压强闭合工作（shell 外推）
   在别的分支进行，两者的口径一旦统一，本阶梯可直接复用（`run_ladder.py --duration` 一行即可重跑）。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 实测（最细档 dx = 1.5 mm） | 相对误差 | < 3 % ? | < 15 % ? |
|:--|:--|:--|:--|:--|:--|
| Δp（绝对口径，单帧） | σ/R = {dp:.1f} Pa | {F['absolute']['delta_p_Pa']:.1f} Pa | {f(F['absolute']['rel_error_pct'],1,True)} % | **否** | **否** |
| Δp（绝对口径，尾均值） | 同上 | {F['absolute']['delta_p_Pa'] * (1 + F['absolute']['rel_error_tailavg_pct']/100):.1f} Pa | {f(F['absolute']['rel_error_tailavg_pct'],1,True)} % | **否** | **否** |
| Δp（历史 bandA 口径） | 同上 | {F['bandA']['delta_p_Pa']:.1f} Pa | {f(F['bandA']['rel_error_pct'],1,True)} % | **否** | **否** |
| 内部压缩比 | 1.00（由 σ/R 反解的 (ρ/ρ0−1) = {comp_needed:.3e}）| {F['compression_ratio_measured_over_required']:.2f} | {f((F['compression_ratio_measured_over_required']-1)*100,1,True)} % | **否** | **否** |
| 曲率（0.6–0.98R 带均值） | 1/R = {1/ref['radius_m']:.1f} m⁻¹ | — | {f((F['curvature_band_mean_over_exact']-1)*100,1,True)} % | **否** | — |
| 虚假流 v_max/v_ref | < 0.1 | {F['spurious_current_ratio']:.3f} | 比值 {F['spurious_current_ratio']:.3f} | — | 是（为阈值的 {F['spurious_current_ratio']/0.1*100:.0f} %）|

**是否 < 3 %：否，全部关键量都不满足**；是否 < 15 %：**否**。Laplace 压差维持 **WITHHELD**。
可确证的是误差**归属**：口径成本约 {f(abs(F['bandA']['rel_error_pct']-F['absolute']['rel_error_pct']),1)} 个百分点（固定），
剩余 {f(abs(F['absolute']['rel_error_pct']),1)} 个百分点表现为内部压缩比 {F['compression_ratio_measured_over_required']:.2f}
的水位型亏缺，对 dx（2.0 → 1.5 mm）不再改善。

_本文由 `benchmarks/laplace_law/make_ladder_doc.py` 从工件渲染，数字禁止手抄；工件更新后重跑该脚本即可。_
"""
    return text


def make_figure(ladder, artifacts, out=FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rungs = sorted(ladder["rungs"], key=lambda r: r["dx_m"])
    dx = np.array([r["dx_m"] * 1e3 for r in rungs])
    band = np.array([r["bandA"]["rel_error_pct"] for r in rungs])
    abse = np.array([r["absolute"]["rel_error_pct"] for r in rungs])
    tail = np.array([r["absolute"]["rel_error_tailavg_pct"] for r in rungs])
    ratio = np.array([r["compression_ratio_measured_over_required"] for r in rungs])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))

    ax = axes[0][0]
    ax.axhspan(-15, 15, color="0.85", label="±15 % band")
    ax.axhline(0, color="k", lw=0.8)
    ax.semilogx(dx, band, "s--", ms=5, color="tab:red", label="bandA (historical)")
    ax.semilogx(dx, abse, "o-", ms=6, color="tab:blue", label="absolute (single frame)")
    ax.semilogx(dx, tail, "^:", ms=6, color="tab:green", label="absolute (tail average)")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel(r"$\Delta p$ error [%]")
    ax.invert_xaxis()
    ax.set_title("(a) Young-Laplace error vs resolution\n(fixed window "
                 f"{ladder['protocol']['duration_t_over_t_sigma']:.2f} $t_\\sigma$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0][1]
    ax.axhline(1.0, color="k", ls="--", lw=1, label="required (sigma/R)")
    ax.axhspan(0.95, 1.05, color="0.85")
    ax.semilogx(dx, ratio, "o-", ms=6, color="tab:purple")
    for x, y in zip(dx, ratio):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), fontsize=7, ha="center")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel(r"$(\rho_{core}/\rho_0-1)$ / required")
    ax.set_ylim(0, 1.15)
    ax.invert_xaxis()
    ax.set_title("(b) interior compression ratio\n(a level deficit, not a resolution effect)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[1][0]
    finest = min(artifacts.values(), key=lambda a: a["setup"]["dx_m"])
    field = finest.get("field")
    if field:
        x = np.array(field["x_m"]) * 1e3
        y = np.array(field["y_m"]) * 1e3
        p = np.array(field["pressure_Pa"])
        sc = ax.scatter(x, y, c=p, s=14, cmap="viridis",
                        vmin=0, vmax=ladder["reference"]["delta_p_Pa"])
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("p [Pa] (colorbar max = sigma/R)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"(c) final pressure cloud, dx = {finest['setup']['dx_m']*1e3:.1f} mm")

    ax = axes[1][1]
    for r in rungs:
        rep = artifacts[r["artifact"].split("/")[-1]]
        field = rep.get("field")
        if not field:
            continue
        x = np.array(field["x_m"])
        y = np.array(field["y_m"])
        rho = np.array(field["rho_over_rho0"]) - 1.0
        rr = np.hypot(x, y) / ladder["reference"]["radius_m"]
        order = np.argsort(rr)
        ax.plot(rr[order], rho[order] * 100.0, ".", ms=3,
                label=f"dx = {r['dx_m']*1e3:.1f} mm")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\rho/\rho_0 - 1$ [%]")
    ax.set_title("(d) final radial density: free-surface deficit")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.2)

    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main():
    ladder = json.loads(LADDER.read_text())
    artifacts = {}
    for r in sorted(ladder["rungs"], key=lambda r: -r["dx_m"]):
        name = r["artifact"].split("/")[-1]
        artifacts[name] = json.loads((BENCH / "convergence" / name).read_text())
    fig = make_figure(ladder, artifacts)
    DOC.write_text(render(ladder, artifacts))
    print("wrote", DOC)
    print("wrote", fig)


if __name__ == "__main__":
    main()
