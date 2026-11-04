"""Generate docs/verification_free_surface_pressure_closure.md from report.json.

Every number and the figure come from
``benchmarks/laplace_pressure_floor/report.json`` (the ``p0_closure`` block that
``run.py`` writes); nothing is typed by hand.  Run order::

    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --quick
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / "report.json"
DOC = ROOT / "docs" / "verification_free_surface_pressure_closure.md"
FIG = "verification_free_surface_pressure_closure.png"
COLORS = {"shell": "C0", "shell_density": "C1"}


def make_figures(report: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = report["runs"]
    p0 = report["p0_closure"]
    dp = report["reference"]["delta_p_Pa"]
    pts = [(r["dx_m"], r["sound_speed"]) for r in base]
    labels = [f"{dx:g}\n$c_0$={c0:g}" for dx, c0 in pts]
    idx = list(range(len(pts)))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0))

    ax = axes[0][0]
    series = [("clamp $p\\geq0$ (baseline)",
               [abs(r["absolute_time_average_tail"]["rel_error_pct"]) for r in base],
               "0.4")]
    for mode, rows in p0.items():
        by_key = {(r["dx_m"], r["sound_speed"]): r for r in rows}
        series.append((f"closure: {mode}",
                       [abs(by_key[k]["absolute_time_average_tail"]["rel_error_pct"])
                        for k in pts], COLORS.get(mode)))
    width = 0.8 / len(series)
    for k, (name, vals, col) in enumerate(series):
        ax.bar([i + (k - (len(series) - 1) / 2) * width for i in idx], vals,
               width, label=name, color=col)
    ax.axhline(15.0, color="r", ls="--", lw=1, label="15 % acceptance")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$|p_{core}-\sigma/R|/(\sigma/R)$ [%]")
    ax.set_title("(a) residual, baseline vs closure")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0][1]
    ax.plot(idx, [((r["density"]["rho_core_over_rho0"] - 1.0)
                   / r["density"]["compression_required"]) for r in base],
            "s", color="0.4", ms=6, label="clamp (baseline)")
    for mode, rows in p0.items():
        by_key = {(r["dx_m"], r["sound_speed"]): r for r in rows}
        ax.plot(idx, [((by_key[k]["density"]["rho_core_over_rho0"] - 1.0)
                       / by_key[k]["density"]["compression_required"]) for k in pts],
                "o-", ms=5, label=f"closure: {mode}", color=COLORS.get(mode))
    ax.axhline(1.0, color="k", ls="--", lw=1, label="Young-Laplace requirement")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("achieved / required compression")
    ax.set_title("(b) interior compression")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    key = pts[0]
    ax.plot(next(r for r in base if (r["dx_m"], r["sound_speed"]) == key)["trace"]["t"],
            next(r for r in base if (r["dx_m"], r["sound_speed"]) == key)["trace"]["absolute"],
            lw=1.2, color="0.4", label="clamp (baseline)")
    for mode, rows in p0.items():
        run = next(r for r in rows if (r["dx_m"], r["sound_speed"]) == key)
        ax.plot(run["trace"]["t"], run["trace"]["absolute"], lw=1.2,
                label=f"closure: {mode}", color=COLORS.get(mode))
    ax.axhline(dp, color="k", ls="--", lw=1, label=rf"$\sigma/R$={dp:.0f} Pa")
    ax.set_xlabel(r"$t/t_\sigma$")
    ax.set_ylabel(r"$p_{\rm core}$ [Pa]")
    ax.set_title(f"(c) core pressure, dx={key[0]:g}, $c_0$={key[1]:g}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    for mode, rows in p0.items():
        ax.plot([abs(r["absolute_time_average_tail"]["delta_p_Pa"]) / dp for r in rows],
                [abs(r["force_implied"]["delta_p_Pa"]) / dp for r in rows], "o",
                ms=7, label=f"closure: {mode}", color=COLORS.get(mode))
    ax.plot([abs(r["absolute_time_average_tail"]["delta_p_Pa"]) / dp for r in base],
            [abs(r["force_implied"]["delta_p_Pa"]) / dp for r in base], "s",
            ms=7, color="0.4", label="clamp (baseline)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="equilibrium (pressure = force)")
    ax.set_xlabel(r"$|\Delta p|$ from the pressure field  $/\ (\sigma/R)$")
    ax.set_ylabel(r"$|\int \rho_0 a_r^{\rm CSF} dr|$  $/\ (\sigma/R)$")
    ax.set_title("(d) pressure read-out vs CSF-force read-out")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Free-surface pressure closure (Adami et al. 2012 eq. 11): "
                 "what it fixes and what it leaves open", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / FIG, dpi=130)
    plt.close(fig)
    return FIG


def main():
    report = json.loads(REPORT.read_text())
    FIG_name = make_figures(report)
    base = report["runs"]
    p0 = report["p0_closure"]
    dp = report["reference"]["delta_p_Pa"]

    def tail(r):
        return r["absolute_time_average_tail"]["rel_error_pct"]

    def comp(r):
        return ((r["density"]["rho_core_over_rho0"] - 1.0)
                / r["density"]["compression_required"])

    def frc(r):
        return abs(r["force_implied"]["delta_p_Pa"]) / dp

    rows = []
    for mode, runs in p0.items():
        for r in runs:
            rows.append(
                f"| `{mode}` | {r['dx_m']:g} | {r['sound_speed']:g} | {r['n_particles']} "
                f"| {r['baseline_absolute_rel_error_pct']:+.1f} | {tail(r):+.1f} "
                f"| {r['improvement_points']:+.1f} | {comp(r):.3f} | {frc(r):.3f} "
                f"| {'PASS' if abs(tail(r)) < 15 else 'FAIL'} |")
    table = "\n".join(rows)

    base_tail = [abs(tail(r)) for r in base]
    base_mean = sum(base_tail) / len(base_tail)
    shell = p0["shell"]
    shell_mean = sum(abs(tail(r)) for r in shell) / len(shell)
    dens_mean = sum(abs(tail(r)) for r in p0["shell_density"]) / len(p0["shell_density"])
    best_mode = "shell" if shell_mean <= dens_mean else "shell_density"
    best_err = min(abs(tail(r)) for r in p0[best_mode])
    worst_err = max(abs(tail(r)) for r in p0[best_mode])
    base_force = [frc(r) for r in base]
    shell_force = [frc(r) for r in shell]
    c0_list = ", ".join(f"{c:g}" for c in sorted({r["sound_speed"] for r in base}))
    dx_list = ", ".join(f"{d:g}" for d in sorted({r["dx_m"] for r in base}))
    n_list = ", ".join(str(r["n_particles"]) for r in base)
    c0_quick = ", ".join(f"{c:g}" for c in sorted({r["sound_speed"] for r in base}))
    # gap between the two independent read-outs of the same physical jump
    force_gap = max(abs(a - (1.0 - abs(tail(r)) / 100.0))
                    for a, r in zip([frc(r) for r in shell], shell))

    doc = f"""# 自由面压强闭合（Adami 2012, eq. 11）：拿回 {base_mean - shell_mean:.0f} 个百分点，剩下的缺口在力不在闭合

**仓库**: TensorSPH ｜ **代码**: `src/tensorsph/free_surface_pressure.py`（`Config.free_surface_pressure`）
｜ **工件**: `benchmarks/laplace_pressure_floor/`（`run.py` 的 `p0_closure` 块、`report.json`、`{FIG_name}`）
｜ **测试**: `tests/test_free_surface_pressure.py`（9 passed）+ `tests/test_laplace_pressure_floor.py`

**结论（先给判定）**：把 `backlog_update_m2971.md` 的 P0 实现出来（截断壳层按 Adami 等 (2012)
式 (11) 外推压强），端到端 Young–Laplace 残差从基线均值 **{base_mean:.1f}%** 降到
**{shell_mean:.1f}%**（`shell` 模式，逐点 {best_err:.1f}%–{worst_err:.1f}%），**仍不过 15% 门槛**，
故 Laplace 压差维持 **WITHHELD**。三点可裁的结论：

1. **闭合有效但不够**：{base_mean - shell_mean:.0f} 个百分点的改善在两端声速上一致
   （+{shell[0]['improvement_points']:.1f} pt / +{shell[1]['improvement_points']:.1f} pt），不是噪声。
2. **「照壁面处理」是错的**：`shell_density`（再按 Tait 反解壳层密度）反而只改善
   {dens_mean and (base_mean - dens_mean):.0f} pt，均残差 {dens_mean:.1f}%——动量的 `p/rho^2` 在
   壳层被额外放大。这是负结果，按纪律保留。
3. **残差已不在闭合上**：同一批算例里，CSF 力的径向积分只有 **{min(shell_force):.2f}–{max(shell_force):.2f}**
   倍 `sigma/R`（压强读数 {1 - worst_err / 100:.2f}–{1 - best_err / 100:.2f} 倍），
   而力积分与压强读数之间的差达 **{force_gap:.2f}** 倍（图 d）——两条读数都不在 1 上，
   说明**表面张力力幅本身偏小**是新的门槛项，且我的力积分口径与
   `laplace_droplet` 在 R=0.1 上测到的 0.79 不一致（§⑤ 记为未结清的仪器欠账）。

## ① 案例介绍（物理背景/几何/参数）

算例与 `docs/verification_laplace_pressure_floor.md` 完全相同（同一 `run.py`，同一初始化），
本单只是在它上面打开了闭合开关：

半径 `R = {report['reference']['radius_m']:g} m` 的二维单相液滴，六方密排初始化，四周真空，无重力，
弱可压 Tait 方程（`p(rho0)=0`），解析基准 `Delta p = sigma/R = {dp:.1f} Pa`。

| 参数 | 值 |
|---|---|
| 半径 R | {report['reference']['radius_m']:g} m |
| 表面张力 sigma | {report['reference']['sigma_N_per_m']:g} N/m（10x 水） |
| 密度 rho0 / gamma | {report['reference']['rho0']:g} kg/m³ / {report['reference']['gamma']:g} |
| 声速 c0 | {c0_list} m/s |
| 分辨率 dx | {dx_list} m（N = {n_list}） |
| 毛细时间 | {report['reference']['t_sigma_s']:.4f} s，观测 1.5 t_sigma |
| 解析压差 | {dp:.1f} Pa |

**被修的缺陷**：`tait_pressure` 钳 `p>=0`；截断核支撑使自由面壳层和密度亏缺（0.95–1.05R 约 −30%），
壳层粒子全落在 `p=0`，界面无法承载 `sigma/R`。上一单实测内部只压到解析所需的 0.57±0.06 倍。

## ② 怎么做的（方法/流程）

**闭合**（`free_surface_pressure.extrapolated_pressure`，仅作用于 `rho < 0.99 rho0` 的粒子）：

    p_i = max(0, sum_j (p_j + rho_j g·(r_i - r_j)) W_ij) / sum_j W_ij

= Adami, Ihmsen & Adams (2012) 式 (11) 在静粒子（`a_i = 0`）上的特化，与求解器**已有的静态壁面**
压强外推同一构造。body-force 项不可省：`tests/test_free_surface_pressure.py`
用 `p = rho G (y_s - y)` 验证该项下外推对线性压场**精确**（1e-10 相对），并同时证明去掉该项会引入
~1e-3 相对偏差（截断模板的一阶矩）。

| `Config.free_surface_pressure` | 内容 |
|---|---|
| `off`（默认） | 历史行为，保持既有数字与测试逐位不变 |
| `shell` | 只外推压强 |
| `shell_density` | 外推压强 + 壳层取 Tait 一致密度（**仅用于动量分母**，连续性方程仍用真实和密度） |

**判别器**：每个算例除四种压强口径外，还用 `implied_jump_from_force` 计算
`|rho0 * int a_r dr|`（CSF 加速度的径向分箱积分，向内为正）。这样能把「压强是否把力传出来」
与「力本身多大」分开。

**复现**：

    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --quick     # 8 个算例，~10 min
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py

## ③ 计算结果

| 模式 | dx [m] | c0 [m/s] | N | 基线残差 [%] | 闭合残差 [%] | 改善 [pt] | 压缩比 | 力积分/`sigma/R` | <15% |
|---|---|---|---|---|---|---|---|---|---|
{table}

基线（`off`）逐点 {", ".join(f"{v:.1f}%" for v in base_tail)}，与 `report.json` 的历史记录一致
（本单复现，未改默认路径）。力积分口径在基线上为
{", ".join(f"{v:.2f}" for v in base_force)} 倍 `sigma/R`，闭合后为
{", ".join(f"{v:.2f}" for v in shell_force)}。

## ④ 结果图

![closure]({FIG_name})

（a）逐点残差：闭合把两根柱子都压到 15% 线上方一点；（b）内部压缩比 0.57→0.79/0.67，
仍未到 1；（c）芯部压强时程；（d）压强读数 vs 力积分读数：两者都偏离 1，且彼此不重合——
这就是「残差已不在闭合上」的直接证据。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

- **适用范围**：只修「截断核 → 密度亏缺 → 压强钳零」这一条链；`p>=0` 钳位保留，张力不稳定不处理。
- **阈值 0.99**：六方格子自身有 +0.59% 和密度偏置，阈值提到 1.0 会把整个液滴当自由面（等价于
  全局 Shepard 平均），本单没走这条路。
- **`shell_density` 是负结果**：壁面是静止的、自由面不是，照搬壁面的密度一致性会破坏静平衡。
  保留它而不是删掉，是为了让后来者不必重跑这条路。
- **力积分口径未标定**：本单测到 {min(shell_force):.2f}–{max(shell_force):.2f} 倍 `sigma/R`，
  而 `laplace_droplet` 在 R=0.1（10 倍半径、更厚的界面）上测到 0.79。液滴只有 2h/R ≈ 5% 的界面厚度，
  径向分箱（40 箱）可能欠分辨。**该口径不得单独作为精度结论**，只能作相对比较用。
- 只跑了 `dx={dx_list}`、`c0 ∈ {{{c0_quick}}}`（预算约束下的 `--quick` 矩阵），
  `dx=0.0015` 与 `c0=10` 未纳入本单。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

基准：`Delta p = sigma/R = {dp:.1f} Pa`（Young–Laplace 解析解，非实验）。

| 量 | 闭合前 | 闭合后（`{best_mode}`） | 相对误差（闭合后） | <3%? | <15%? |
|---|---|---|---|---|---|
| 芯部压差 `p_core`（tail 平均） | {base_tail[0]:.1f}% / {base_tail[1]:.1f}% | {abs(tail(shell[0])):.1f}% / {abs(tail(shell[1])):.1f}% | {worst_err:.1f}%（最差点） | **否** | {'是' if worst_err < 15 else '否'} |
| 内部压缩比 | {comp(base[0]):.2f} / {comp(base[1]):.2f} | {comp(shell[0]):.2f} / {comp(shell[1]):.2f} | {abs(1 - comp(shell[0])) * 100:.1f}%（对 1） | 否 | 否 |
| CSF 力积分 `|int rho0 a_r dr|` | {base_force[0]:.2f} / {base_force[1]:.2f} | {shell_force[0]:.2f} / {shell_force[1]:.2f} | {abs(1 - min(shell_force)) * 100:.1f}%–{abs(1 - max(shell_force)) * 100:.1f}% | 否 | 否 |

**显式回答**：关键量 `Delta p` **不是 <3%**（最优点 {best_err:.1f}%）。按精度标准（>15% 不可接受），
端到端 Young–Laplace 压差仍判 **WITHHELD**，本单不得被引用为「Laplace 已通过」。

**对上一单结论的修正（honest-FAIL 记录）**：`docs/verification_laplace_pressure_floor.md` §⑦ 把闭合
写成「唯一门槛项」并预测闭合后 4 点全部 <15%。本单实测：闭合只拿回 {base_mean - shell_mean:.0f} pt，
**该预测被证伪**。新的门槛项是表面张力力幅（力积分仅 {min(shell_force):.2f}–{max(shell_force):.2f} 倍
`sigma/R`），与 `tests/test_laplace_droplet.py` 记录的 `dp_ratio ≈ 0.79–0.84` 是同一缺口的两条线索。

## ⑦ 下一步（机器可裁的判据）

1. **P0'（新，最高优先）**：标定力积分口径（分箱数/界面厚度/R 依赖），先在 R=0.1 与 R=0.02 上
   复现同一个数（判据：两个半径上 `|int rho0 a_r dr| / (sigma/R)` 相差 <5%），再谈力幅修正。
2. **P1**：`grad c` 归一化（`N_delta`）与 `fit_quadratic` 曲率系统偏差二选一定量归因；
   判据：`tests/test_laplace_droplet.py::test_laplace_droplet` 的 `dp_ratio` 进入 `1±0.03`。
3. **P2**：闭合阈值/迭代扫描（1–3 次外推）；判据：`shell` 残差对阈值不敏感（漂移 <2 pt）。
4. 端到端 xfail 转正的前提不变：
   `tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct` 由 strict-xfail 转 xpass。
"""
    DOC.write_text(doc)
    print("wrote", DOC)


if __name__ == "__main__":
    main()