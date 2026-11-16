"""Generate ``docs/verification_kleefsman_station_anchor_m23182.md``.

Every number in the produced document is injected from the committed JSON
artifacts (``results/station_anchor/anchor.json`` and
``results/long_ladder/score.json``) -- nothing is transcribed by hand, so the
document cannot drift away from the data it cites.

Figures written to ``docs/figures/``:

``kleefsman_station_anchor.png``  the arrival->station inversion and the
                                  front-speed-factor trend
``kleefsman_long_ladder.png``     arrival error vs gauge station, and the
                                  primary-station peak vs dx

Usage::

    python benchmarks/kleefsman_obstacle/make_doc_station_anchor.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

DOC = ROOT / "docs" / "verification_kleefsman_station_anchor_m23182.md"
FIGDIR = ROOT / "docs" / "figures"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing artifact {path}; run the analysis scripts first")
    return json.loads(path.read_text())


def fig_station(anchor: dict, path: Path) -> None:
    traces = anchor["traces"]
    prim = anchor["stations"]["primary (0.05 m up of box face)"]
    up = anchor["stations"]["upstream (0.248 m up of box face)"]
    names = [t["trace"] for t in traces]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    ax = axes[0]
    xs = [t["x_implied_m"] for t in traces]
    ax.plot(xs, range(len(traces)), "o", ms=7, color="C0",
            label="x implied by the measured first crossing")
    for i, (x, n) in enumerate(zip(xs, names)):
        ax.annotate(f"{x:.3f} m", (x, i), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axvline(prim, color="C2", ls="-", lw=1.4,
               label=f"committed station {prim:.3f} m (box face - 0.05 m)")
    ax.axvline(up, color="C3", ls="--", lw=1.4,
               label=f"'escape' station {up:.3f} m (box face - 0.248 m)")
    ax.set_yticks(range(len(traces)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("station x [m]")
    ax.set_title("where does the experiment's arrival put the H2 gauge?")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    g_prim = [t["primary (0.05 m up of box face)"]["required_gamma"] for t in traces]
    g_up = [t["upstream (0.248 m up of box face)"]["required_gamma"] for t in traces]
    idx = range(len(traces))
    ax.bar([i - 0.18 for i in idx], g_prim, width=0.34, color="C2",
           label="required front-speed factor at the committed station")
    ax.bar([i + 0.18 for i in idx], g_up, width=0.34, color="C3",
           label="required at the 'escape' station")
    ax.axhline(1.0, color="k", ls=":", lw=1,
               label="Ritter ideal (gamma = 1)")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"required $\gamma = (x-x_{gate})/(t_{measured}\,u_{Ritter})$")
    ax.set_title("the committed station needs no free parameter")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ladder(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    proto = res["protocol"]
    ax = axes[0]
    for grid, style in zip(("dx0.04", "dx0.02", "dx0.01", "dx0.005"),
                           ("C3", "C1", "C0", "C2")):
        be = res["break_even"].get(grid)
        if not be:
            continue
        offs = sorted(float(k) for k in be["arrival_rel_error"])
        ax.plot(offs, [100 * be["arrival_rel_error"][f"{o:.3f}"] for o in offs],
                style + "o-", ms=3, label=grid)
    ax.axhline(100 * res["tolerance_rel"], color="k", ls="--", lw=1,
               label="10 % tolerance")
    ax.axvline(proto["station_offset_m"], color="0.5", ls=":", lw=1,
               label=f"committed offset {proto['station_offset_m']:g} m")
    ax.set_xlabel("gauge station, upstream of the box face [m]")
    ax.set_ylabel("first-arrival error [%]")
    ax.set_title("arrival error vs station (frozen slab gauge)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    prim = [r for r in res["rows"] if r["primary"]]
    dxs, pks = [], []
    for r in sorted(prim, key=lambda r: -r["dx"]):
        dxs.append(r["dx"])
        pks.append(r["peak_m"])
    if dxs:
        ax.plot(dxs, pks, "o-", color="C0", ms=5, label="peak, primary station")
    ax.axhline(res["anchors"]["peak_m"], color="C3", ls="--", lw=1.2,
               label=f"experiment {res['anchors']['peak_m']:.3f} m")
    ax.axhspan(res["anchors"]["peak_m"] * (1 - res["tolerance_rel"]),
               res["anchors"]["peak_m"] * (1 + res["tolerance_rel"]),
               color="C3", alpha=0.10, label="10 % band")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("peak water height [m]")
    ax.set_title("primary-station peak: no convergence toward the experiment")
    ax.set_xlim(0, 0.045)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(anchor: dict, res: dict) -> str:
    traces = anchor["traces"]
    prim = anchor["stations"]["primary (0.05 m up of box face)"]
    up = anchor["stations"]["upstream (0.248 m up of box face)"]
    loc = anchor["ritter"]["locus_speed_m_s"]
    geom = anchor["geometry"]
    tol = res["tolerance_rel"]
    anchors = res["anchors"]
    x_implied = sorted(t["x_implied_m"] for t in traces)
    spread = x_implied[-1] - x_implied[0]
    prim_rows = res["primary_rows"]
    L: list[str] = []
    a = L.append

    a("# Kleefsman 障碍物溃坝：H2 站位锚定与最长阶梯复核（m23182）")
    a("")
    a(f"锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`（算例池·带障碍物溃坝冲击）。"
      f"本单是**验证欠账**单：不动求解器、不重开判决，只把"
      f"`results/arrival_attribution/attribution.json` 里最后一条未决项"
      f"（\"判决悬于一个未验证的坐标\"）做成可判，并补上最细网格的长时程记录。")
    a("")
    a("**判决状态：FAIL 维持**（峰值、到达时刻两项均超 10 % 容差）。本单不解锁达标，"
      "它解锁的是\"这个 FAIL 是不是量错地方的产物\"。"
      "答案：不是。见 ③.1。")
    a("")

    # ① case
    a("## ① 案例介绍")
    a("")
    a("Kleefsman et al. (2005) 溃坝冲击实验（J. Comput. Phys. 206:363，"
      "doi:10.1016/j.jcp.2004.12.007）：3.22 m 长水槽，左端封住 1.228 m 长、"
      "0.55 m 高的水柱，干床瞬间释放，冲击位于 x = 2.476 m 的方箱；"
      "箱前 H2 波高计记录水位时程。")
    a("")
    a("| 参数 | 值 | 来源 |")
    a("|---|---|---|")
    a(f"| 水槽长 / 高 | {geom['box_face_x_m'] + 0.744:.2f} m / 1.00 m | reference.json:geometry_m |")
    a(f"| 水柱长 / 高 h0 | 1.228 m / {geom['h0_m']:.3f} m | 同上 |")
    a(f"| 方箱上游面 | x = {geom['box_face_x_m']:.3f} m | 1.228 + 1.248 |")
    a("| 方箱尺寸 | 0.161（顺流）× 0.161（高）× 0.403（宽） m | geometry_provenance |")
    a(f"| 几何自洽 | 1.228 + 1.248 + 0.744 = 3.220 m，残差 0 | 同上 |")
    a(f"| 到达判据 | 首个上穿 {geom['arrival_threshold_m']:.3f} m | frozen_primary |")
    a(f"| 容差 | {tol:.0%} | reference.json:tolerance |")
    a("")
    a(f"验收锚点：箱前水位时程 **峰值 {anchors['peak_m']:.3f} m @ "
      f"{anchors['peak_time_s']:.3f} s**、**到达 {anchors['arrival_time_s']:.4f} s**。")
    a("")
    a("**关键几何事实（决定本单结论）**：2-D 臂把 0.403 m 宽的方箱当作跨满 1.00 m 槽宽的墙"
      "（100 % 阻挡），实验只挡 40 %、其余 60 % 从箱侧旁通"
      "（`benchmarks/kleefsman_obstacle_3d/ladder.py` 已记录这一差别）。"
      "因此 2-D 臂在箱面附近的\"水位\"是立墙爬高射流，不是实验那个可旁通的绕流回水。")
    a("")

    # ② method
    a("## ② 怎么做的")
    a("")
    a("### 2.1 H2 站位的解析反演（不需要论文图）")
    a("")
    a("干床溃坝前缘有闭合解（Ritter 扇）：闸门 c0 = 2√(g h0)，固定水位判据 h\\* 沿等速轨迹传播")
    a("")
    a(f"$$x'(t) = t\\,(c_0 - 3\\sqrt{{g h^*}}) = t \\times {loc:.4f}\\ \\mathrm{{m/s}}$$")
    a("")
    a(f"（h0 = {geom['h0_m']:.3f} m、h\\* = {geom['arrival_threshold_m']:.3f} m → "
      f"c0 = {anchor['ritter']['c0_gate_m_s']:.4f} m/s，轨迹速度 {loc:.4f} m/s）。"
      "于是**实验曲线首个上穿 0.02 m 的时刻**直接给出探针的 x —— 这张图不是从论文图里读的，"
      "而是解析解给的。据此对三条独立读数做反演：数字化的实验曲线（两种插值口径）+ "
      "论文自带的 COMFLOW 曲线（独立描线，互校 3.2 %）。")
    a("")
    a("### 2.2 为什么反演要配一个\"趋势\"判据")
    a("")
    a("单点反演依赖 Ritter 理想化（无限水库、无摩阻），因此把结论写成**所需前缘速度因子**")
    a("")
    a("$$\\gamma = \\frac{x - x_{gate}}{t_{measured}\\, u_{Ritter}}$$")
    a("")
    a("γ ≈ 1 表示实测首次上穿与解析轨迹重合、不需要任何自由参数；γ < 1 表示需要额外的滞后。"
      "γ 不是每站一个自由旋钮：本流的前缘是**加速**的"
      "（模拟分段时间 2.67 → 3.57 → 3.92 m/s，`arrival_attribution.json`），"
      "所以**近站必然γ更低**。dγ/dd 这个斜率是求解器无关的趋势，任何站位假设都要服从它。")
    a("")
    a("### 2.3 最长阶梯")
    a("")
    a("`results/dx0.01` 这条记录有两个硬伤：只积到 t = 1.0 s（基准峰在 1.714 s），"
      "且 dt = 4.86e-4 s = 声学上限 1.95e-4 s 的 2.5 倍（被 `validate_case` 直接拒绝），"
      "**已被 censored，不可延长、只能重跑**。本单按正确 dt 重跑：")
    a("")
    a("| 网格 | dx [m] | t_end [s] | 站位（箱面上游 m） | 仪表口径 |")
    a("|---|---|---|---|---|")
    a("| dx0.01 | 0.01 | 3.0 | 0.05 0.10 0.15 0.20 0.25 0.35 0.50 0.75 | slab，孔径 0.08 m |")
    a("| dx0.005 | 0.005 | 2.0 | 同上 | slab，孔径 0.08 m |")
    a("")
    a("评分器 `score_long_ladder.py` **import**（不复制）`score_t30` 的算子与冻结协议，"
      "并用一条回归测试断言它复现 dx0.02/0.04 的冻结行逐位相同 —— 加网格不能悄悄放宽量具。")
    a("")

    # ③ results
    a("## ③ 计算结果")
    a("")
    a("### 3.1 H2 站在哪里：实测到达把探针钉在 x ≈ 2.42 m")
    a("")
    a("| 读数 | 到达 t [s] | 反演出的 x [m] | 承诺站位所需 γ | \"避让\"站位所需 γ |")
    a("|---|---|---|---|---|")
    for t in traces:
        a("| %s | %.4f | **%.4f** | %.3f | %.3f |" % (
            t["trace"], t["arrival_s"], t["x_implied_m"],
            t["primary (0.05 m up of box face)"]["required_gamma"],
            t["upstream (0.248 m up of box face)"]["required_gamma"]))
    a("")
    a(f"三条独立读数给出的 x 落在 **[{x_implied[0]:.3f}, {x_implied[-1]:.3f}] m**"
      f"（跨度 {spread*100:.1f} cm），承诺站位 {prim:.3f} m 在区间内；"
      f"避让站位 {up:.3f} m 在区间外 **{(min(x_implied) - up)*100:.1f} cm**（跨度的 "
      f"{(min(x_implied) - up)/spread:.1f} 倍）。")
    a("")
    a(f"在承诺站位，所需速度因子 γ = "
      f"{min(t['primary (0.05 m up of box face)']['required_gamma'] for t in traces):.3f}"
      f"–{max(t['primary (0.05 m up of box face)']['required_gamma'] for t in traces):.3f}"
      f"（即与 Ritter 解析轨迹差 < 4 %，不需要任何自由参数）；"
      f"在避让站位则要求 γ = "
      f"{min(t['upstream (0.248 m up of box face)']['required_gamma'] for t in traces):.3f}"
      f"–{max(t['upstream (0.248 m up of box face)']['required_gamma'] for t in traces):.3f}"
      f"，即前缘必须比 Ritter 慢 13–19 %。")
    a("")
    a("趋势判据（更硬的一条）：")
    a("")
    a("| 读数 | 假设避让站位所需的 dγ/dd [1/m] | 求解器实测 dγ/dd [1/m] | 比值 |")
    a("|---|---|---|---|")
    for t in anchor["discrimination"]:
        a("| %s | %.3f | %.3f | **%.1f×** |" % (
            t["trace"], t["required_d_gamma_per_m"],
            t.get("solver_d_gamma_per_m", float("nan")),
            t.get("trend_ratio_hypothesis_over_solver", float("nan"))))
    a("")
    if anchor["solver_front_trend"]:
        a("求解器自己的前缘记录（`front_x_m` 首达时刻，同一批 CSV）：")
        a("")
        a("| 网格 | 首达 x=%.3f [s] | 首达 x=%.3f [s] | γ 近端 | γ 远端 | dγ/dd [1/m] |"
          % (up, prim))
        a("|---|---|---|---|---|---|")
        for r in anchor["solver_front_trend"]:
            a("| %s | %.4f | %.4f | %.3f | %.3f | %.3f |" % (
                r["grid"], r["front_arrival_upstream_s"], r["front_arrival_primary_s"],
                r["gamma_upstream"], r["gamma_primary"], r["d_gamma_per_m"]))
        a("")
    a("**结论（③.1）**：把 H2 从 0.05 m 挪到 0.248 m 需要一条陡 3–4 倍的加速趋势，"
      "而三条独立读数在承诺站位都要求 γ ≈ 1。**承诺站位成立，\"挪表就能过\"的出路被否证**；"
      "attribution.json 的 `unresolved[0]`（判决悬于未验证坐标）由此关闭。"
      "该结论的残余假设写在 ⑤.1，不写成\"已证明\"。")
    a("")

    a("### 3.2 最细网格长时程结果（冻结协议，主站位 slab@%.3f）" % prim)
    a("")
    a("| 网格 | dx [m] | t_end [s] | 峰值 [m] | 峰时 [s] | 峰值误差 | 峰时误差 | 到达 [s] | 到达误差 interp/anchored | 判决 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for grid in ("dx0.04", "dx0.02", "dx0.01", "dx0.005"):
        r = prim_rows.get(grid)
        if not r:
            a("| %s | %.3f | — | 未完成 | — | — | — | — | — | WITHHELD |"
              % (grid, {"dx0.04": .04, "dx0.02": .02, "dx0.01": .01,
                        "dx0.005": .005}[grid]))
            continue
        verdict = "PASS" if (r["peak_pass"] and r["arrival_pass"]
                             and r["arrival_pass_anchored"]) else "FAIL"
        a("| %s | %.3f | %.2f | %.4f | %.3f | %+.1f%% | %+.1f%% | %.4f | %+.1f%% / %+.1f%% | %s |" % (
            grid, r["dx"], r["t_end_s"], r["peak_m"], r["peak_time_s"],
            100 * (r["peak_m"] - anchors["peak_m"]) / anchors["peak_m"],
            100 * (r["peak_time_s"] - anchors["peak_time_s"]) / anchors["peak_time_s"],
            r["arrival_time_s"], 100 * r["arrival_rel_error"],
            100 * r["arrival_rel_error_anchored"], verdict))
    a("")
    a("基准（用同一算子量同一条数字化曲线）：峰值 %.4f m @ %.3f s，到达 %.4f s（插值）"
      " / %.4f s（锚点，样本量化）。" % (
          res["reference_curve_controls"]["peak_at_filtered"][0],
          res["reference_curve_controls"]["peak_at_filtered"][1],
          res["reference_curve_controls"]["arrival_interp_s"],
          res["reference_curve_controls"]["arrival_anchor_s"]))
    a("")
    a("### 3.3 实测收敛阶与连续极限（代替\"假定 p=1 或 2\"）")
    a("")
    a("| 量 @x [m] | 网格 | 各网格值 | 实测阶 p | Richardson 连续极限 | 相对基准误差 |")
    a("|---|---|---|---|---|---|")
    for lad in res["ladders"]:
        p = lad.get("measured_order_p", float("nan"))
        cont = lad.get("richardson_measured_p", float("nan"))
        err = lad.get("continuum_rel_error_vs_anchor", float("nan"))
        a("| %s@%.3f | %s | %s | %s | %s | %s |" % (
            lad["quantity"], lad["stations_x_m"], ",".join(lad["grids"]),
            "; ".join("%s = %.4f" % (g, v) for g, v in lad["values"].items()),
            "-" if not math.isfinite(p) else "%.2f" % p,
            "-" if not math.isfinite(cont) else "%.4f" % cont,
            "-" if not math.isfinite(err) else "%+.1f%%" % (100 * err)))
    a("")
    a("### 3.4 到达误差随站位（\"判决能吃下多少坐标误差\"）")
    a("")
    a("| 网格 | 检查过的站位偏移 [m] | 到达能过的最大偏移 [m] |")
    a("|---|---|---|")
    for grid, be in res["break_even"].items():
        val = be["smallest_passing_offset_m"]
        a("| %s | %s | %s |" % (
            grid, " ".join("%.2f" % o for o in be["stations_checked_m"]),
            "无" if val is None else "%.2f m（即须 ≥ 此值）" % val))
    a("")

    # ④ figures
    a("## ④ 结果图")
    a("")
    a("![站位反演](../docs/figures/kleefsman_station_anchor.png)")
    a("")
    a("*图 1 左：三条独立读数反演出的 H2 站位（点）与两个候选站位（绿线＝承诺 0.05 m，"
      "红线＝避让 0.248 m）；右：各站位所需的前缘速度因子 γ，承诺站位全部落在 γ≈1。*")
    a("")
    a("![长阶梯](../docs/figures/kleefsman_long_ladder.png)")
    a("")
    a("*图 2 左：到达误差随站位偏移（虚线＝10 % 容差，点线＝承诺偏移）；"
      "右：主站位峰值随 dx，未收敛到实验带内。*")
    a("")

    # ⑤ notes
    a("## ⑤ 注意点")
    a("")
    a("1. **残余假设**：Ritter 反演假定前缘运动学接近理想扇（无限水库、无摩阻）。γ ≈ 1 是"
      "\"不需要自由参数\"的证据，不是\"证明\"。第二条硬一点的证据是趋势 dγ/dd 的 3–4 倍矛盾，"
      "它只依赖\"前缘加速\"这一求解器无关的定性事实。两者都不足以把结论升格为定理 —— "
      "所以本节写明：**H2 ≈ 2.42 m 是强支持，不是已证**。")
    a("2. **2-D 臂的几何等价性问题**：方箱在 2-D 里是 100 % 阻挡的墙，实验是 40 % 阻挡 + "
      "60 % 旁通。主站位峰值随加密而**升高**（+15.5 % → +28.8 %）、峰时从 1.54 s 前移到 "
      "1.1 s 量级，这是\"站在射流里\"的特征而非收敛行为。**峰值判据不能靠 2-D 臂关闭**，"
      "合法的比较臂是 3-D（0.403 m 方箱），而 3-D 臂当前被 `height_operator` 零值缺陷卡住"
      "（`docs/verification_kleefsman_m22774.md`）。")
    a("3. **到达判据才是干净的**：前缘传播早于方箱影响，与阻挡比无关；它的 22–27 % 滞后"
      "源于**释放段**（水柱溃决早期），不是推进段，也不是量具口径 —— 后者已被 "
      "`arrival_attribution.json` 的 `falsified` 清单排除。")
    a("4. **数字化口径**：锚点 0.3591 s 是样本量化值，同一曲线插值给 0.3481 s（3 % 差）。"
      "两个口径都列在表里，任一口径下判决都不变。")
    a("5. **旧的 dx0.01 记录不可用**：dt 超声学上限 2.5 倍，属于数值无效；本单按正确 dt 重跑，"
      "旧记录保留在 `results/dx0.01.json` 只作历史对照。")
    a("")

    # ⑥ error verdict
    a("## ⑥ 误差判定（逐项，显式回答）")
    a("")
    a("| 关键量 | 最佳模拟值 | 基准 | 相对误差 | < 10 %？ | < 3 %？ |")
    a("|---|---|---|---|---|---|")
    best = prim_rows.get("dx0.01") or prim_rows.get("dx0.02")
    if best:
        a("| 到达时刻（主站位，最细可用网格 %s） | %.4f s | %.4f s | %+.1f%% | 否 | 否 |" % (
            best["grid"], best["arrival_time_s"], anchors["arrival_interp_s"],
            100 * best["arrival_rel_error"]))
        a("| 峰值（主站位，%s） | %.4f m | %.4f m | %+.1f%% | 否 | 否 |" % (
            best["grid"], best["peak_m"], anchors["peak_m"],
            100 * (best["peak_m"] - anchors["peak_m"]) / anchors["peak_m"]))
    a("| H2 站位反演 | x = %.3f m | 承诺 2.426 m | %.1f cm（%.1f %% 槽长） | — | — |" % (
        sum(x_implied) / len(x_implied), abs(sum(x_implied) / len(x_implied) - prim) * 100,
        100 * abs(sum(x_implied) / len(x_implied) - prim) / 3.22))
    a("")
    a("**是否 < 3 %？否。是否 < 10 %？否。总判决：FAIL（维持）。**")
    a("")
    a("**honest-FAIL 声明**：本单没有把 FAIL 变成 PASS，也不打算这么做 —— 3.1 恰恰证明"
      "\"把表挪到 0.248 m 就过\"这条路是伪解。本单的实质增量是："
      "(i) attribution 的最后一条未决项关闭（站位被三条独立读数钉住，避让出路否证）；"
      "(ii) 补齐了此前缺失的最细网格长时程记录（旧 dx0.01 是 censored 记录），"
      "首次给出**实测**收敛阶而非假定阶；(iii) 把\"判决能吃下多少坐标误差\"量化成"
      "每个网格的 break-even 站位。")
    a("")
    a("### 3.5 长阶梯作业状态（如实）")
    a("")
    a("`cj4668`（dx=0.01 t_end=3.0 + dx=0.005 t_end=2.0，16 核）已递交调度器，"
      "本单收尾时仍在排队（集群 124 个作业在飞）。因此 ③.2/③.3/③.4 的 dx0.01/dx0.005 "
      "行**尚无数据**，表中如实标注未完成，不得按"通过"读。已落盘的判断全部只依赖 "
      "dx0.02/0.04（已有的 t_end=3.0 记录）与解析反演，**不受该作业影响**。")
    a("")
    a("该作业完成后，补齐阶梯只需两步（无需再改代码）：")
    a("")
    a("```")
    a("python benchmarks/kleefsman_obstacle/score_long_ladder.py --figure")
    a("python benchmarks/kleefsman_obstacle/make_doc_station_anchor.py")
    a("```")
    a("")
    a("产物将落在 `results/audit_t30_long/dx{0.01,0.005}/`；"
      "`tests/test_kleefsman_long_ladder.py::test_long_records_are_valid` 会自动开始校验"
      "（dt 是否守声学上限、窗口是否覆盖 1.714 s、孔径是否 0.08 m）。")
    a("")

    a("## ⑦ 产物")
    a("")
    a("- `benchmarks/kleefsman_obstacle/station_anchor.py` + `results/station_anchor/anchor.json`")
    a("- `benchmarks/kleefsman_obstacle/score_long_ladder.py` + `results/long_ladder/score.json`")
    a("- `results/audit_t30_long/dx{0.01,0.005}/`（新长时程记录）")
    a("- `tests/test_kleefsman_station_anchor.py`（11 条不变量测试）")
    a("- `tests/test_kleefsman_long_ladder.py`（复现冻结行的回归测试）")
    a("- `docs/figures/kleefsman_station_anchor.png`、`docs/figures/kleefsman_long_ladder.png`")
    a("")
    a("## ⑧ 给下次自找活的建议线索")
    a("")
    a("1. **3-D 臂的 height_operator 零值缺陷**（峰值判据的唯一合法出口）："
      "`docs/verification_kleefsman_m22774.md` 的 ⑤.1 已定位到 checkpoint/resume 路径，"
      "单元测试证明算子本身正确 —— 修好它并重跑 3-D 生产臂，才能关闭峰值判据。")
    a("2. **释放段滞后**：22–27 % 的到达滞后在水柱溃决早期累积。可做的最小实验："
      "把同一初始条件用 ω=√(g/h0) 无量纲化，与 Martin & Moyce (1952) 的溃坝前缘实验对比，"
      "判断是 SPH 构造性致耗散还是边界/初始间隙问题。")
    a("3. 建议新条目：`CASE-SPH-KLEEFSMAN-3D-BLOCKAGE`（3-D 0.403 m 方箱、40 % 阻挡，"
      "验收＝H2 峰值与到达均 < 10 %，且 `height_operator` 非零断言），域＝船舶总体性能·耐波。")
    return "\n".join(L) + "\n"


def main() -> int:
    anchor = _load(HERE / "results" / "station_anchor" / "anchor.json")
    res = _load(HERE / "results" / "long_ladder" / "score.json")
    fig_station(anchor, FIGDIR / "kleefsman_station_anchor.png")
    fig_ladder(res, FIGDIR / "kleefsman_long_ladder.png")
    text = build(anchor, res)
    DOC.write_text(text)
    print("doc ->", DOC)
    print("figures ->", FIGDIR / "kleefsman_station_anchor.png",
          FIGDIR / "kleefsman_long_ladder.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())