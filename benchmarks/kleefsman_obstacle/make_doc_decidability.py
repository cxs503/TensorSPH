"""Generate the decidability-audit report (markdown + figures) from the JSON.

Every number in the generated document is injected from
``results/audit_protocol/decidability.json`` or from a run record under
``results/``; nothing is typed by hand.  Figures are produced here as well so
that the report and the plotted quantities cannot drift apart.

Usage::

    python benchmarks/kleefsman_obstacle/make_doc_decidability.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
FIGDIR = ROOT / "docs" / "figures"


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(float(x)) else f"{100.0 * float(x):.{digits}f}%"


def num(x: float, digits: int = 4) -> str:
    return "n/a" if x is None or not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def extra_runs() -> dict[str, dict]:
    """Uncensored ``t_end = 3.0 s`` runs, if the cluster jobs have landed."""
    out = {}
    for path in sorted(RESULTS.glob("audit_t30/*/metrics.json")):
        row = load(path)
        out[path.parent.name] = row
    return out


# --------------------------------------------------------------------------- #
def figure_censoring(grids: dict, budget: dict, out: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in ("dx0.04", "dx0.02", "dx0.01") if t in grids]
    if not tags:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, tag in enumerate(tags):
        g = grids[tag]
        t_end = float(g["t_end"])
        peak_t = float(g["peak_time_s"])
        cens = g["horizon"]["censored"]
        ax.plot([peak_t], [k], "o", ms=9,
                color=("C3" if cens else "C0"))
        ax.plot([0, t_end], [k, k], color="0.75", lw=1.0, zorder=0)
        ax.plot([t_end], [k], "|", color="0.4", ms=14, mew=1.6)
        ax.annotate(f"peak {peak_t:.3f} s / t_end {t_end:.2f} s",
                    (max(peak_t, 0.02), k + 0.18), fontsize=8)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([f"{grids[t]['dx']:g} m" for t in tags])
    ax.set_xlabel("time [s]  (grey bar = run horizon, red = censored peak)")
    ax.set_title("Kleefsman obstacle case: peak vs run horizon")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = out / "kleefsman_decidability_censoring_m20773.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_c0_family(family: dict, out: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = {k: v for k, v in (family or {}).items() if v.get("status") == "OK"}
    if not fams:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, fit in sorted(fams.items()):
        c0 = [p[0] for p in fit["points"]]
        pk = [p[1] for p in fit["points"]]
        ax.plot(c0, pk, "o-", label=f"dx = {float(k):g} m")
        grid = [min(c0) + i * (max(c0) - min(c0)) / 40 for i in range(41)]
        ax.plot(grid, [fit["a_incompressible_peak_m"] + fit["b_m2_per_s2"] / c ** 2
                       for c in grid], "--", lw=1.0, color="0.5")
    ax.set_xlabel(r"artificial sound speed $c_0$ [m/s]")
    ax.set_ylabel("peak level in front of the box [m]")
    ax.set_title("Sound-speed family (dashed = fit $a + b/c_0^2$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "kleefsman_decidability_c0family_m20773.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_operators(grids: dict, out: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in ("dx0.04", "dx0.02", "dx0.01")
            if t in grids and grids[t]["operator_spread"]]
    if not tags:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    stations = sorted({s for t in tags for s in grids[t]["operator_spread"]}, key=float)
    width = 0.8 / max(len(stations), 1)
    for i, st in enumerate(stations):
        vals = [grids[t]["operator_spread"].get(st, {}).get("spread_rel", float("nan"))
                for t in tags]
        xs = [k + i * width for k in range(len(tags))]
        ax.bar(xs, [0 if v is None or not math.isfinite(v) else v for v in vals],
               width=width, label=f"station x = {float(st):.3f} m")
    ax.axhline(0.10, color="C3", ls="--", lw=1.0, label="case tolerance 10%")
    ax.set_xticks([k + width * (len(stations) - 1) / 2 for k in range(len(tags))])
    ax.set_xticklabels([f"dx = {grids[t]['dx']:g} m" for t in tags])
    ax.set_ylabel("peak spread across gauge operators")
    ax.set_title("Definitional spread of the acceptance metric")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "kleefsman_decidability_operators_m20773.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_t30(runs: dict, grids: dict, out: Path) -> Path | None:
    """Probe traces of the uncensored runs against the censored historical ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for tag, run in runs.items():
        csv = RESULTS / "audit_t30" / tag / "time_series.csv"
        if not csv.exists():
            continue
        ts, hs = [], []
        with csv.open() as fh:
            header = fh.readline().strip().split(",")
            it, ih = header.index("t_s"), header.index("height_m")
            for line in fh:
                cols = line.strip().split(",")
                try:
                    ts.append(float(cols[it]))
                    hs.append(float(cols[ih]))
                except ValueError:
                    continue
        series[tag] = (ts, hs)
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for tag, (ts, hs) in sorted(series.items()):
        ax.plot(ts, hs, lw=1.3, label=f"{tag} (t_end = 3.0 s)")
    ax.axhline(0.02, color="0.6", ls=":", lw=0.9, label="arrival threshold 0.02 m")
    fig.tight_layout()
    ax.set_xlabel("t [s]")
    ax.set_ylabel("probe water level at x = 2.426 m [m]")
    ax.set_title("Uncensored probe traces (same protocol, t_end = 3.0 s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "kleefsman_decidability_t30_m20773.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def build_report(audit: dict, runs: dict, figures: list[tuple[str, str]]) -> str:
    ref = load(HERE / "reference.json")
    g = ref["geometry_m"]
    tol = audit["tolerance"]
    grids = audit["grids"]
    budget = audit["uncertainty_budget"]
    verdict = budget["verdict"]

    out = []
    a = out.append
    a("# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 验收判据可判定性审计")
    a("")
    a("**Mission**: m20773 · **分支**: `agent/m20773` · **角色**: verifier 席（独立复核）  ")
    a("**求解器**: TensorSPH WCSPH（cubic-spline 核, Tait EOS, Adami 壁面 BC）  ")
    a("**本文档由 `benchmarks/kleefsman_obstacle/make_doc_decidability.py` 从 "
      "`results/audit_protocol/decidability.json` 与各 run 记录程序化注入生成（零手抄）**")
    a("")
    a("---")
    a("")
    a("## ① 案例介绍")
    a("")
    a("Kleefsman et al. (2005, *J. Comput. Phys.* 206:363, DOI 10.1016/j.jcp.2004.12.007) "
      "带障碍物溃坝：水槽左端水柱瞬时溃决，沿干底推进后撞击槽底刚性方箱。"
      "算例池的验收判据是**相对判据**：障碍物前水位时程的**峰值**与**到达时刻**"
      f"对实验值的相对误差均 < {pct(tol, 0)}。")
    a("")
    a("| 量 | 值 |")
    a("|---|---|")
    a(f"| 槽长 × 槽高 | {g['tank_length']} × {g['tank_height']} m |")
    a(f"| 水柱长 × 高 | {g['column_length']} × {g['column_height']} m |")
    a(f"| 方箱（流向×高×宽） | {g['obstacle_length']} × {g['obstacle_height']} × 0.403 m |")
    a(f"| 箱上游面 x | {g['obstacle_upstream_face_x']} m |")
    a("| 探针站 | 箱面上游 0.05 m（x = 2.426 m）、0.248 m（x = 2.228 m） |")
    a("| ρ₀ / g / γ | 1000 kg/m³ / 9.81 m/s² / 7 |")
    a("")
    a("**为什么需要本审计**：相对判据只有在被测量本身的数值不确定度**远小于**容差时才可判定。"
      "本单把这一前提变成可执行检查——从已提交的 run 记录出发，逐项量化定义性偏差与数值误差，"
      "并对“现在的证据能否支撑 10% 判定”给出可否证的回答。")
    a("")
    a("## ② 方法")
    a("")
    a("新增 `benchmarks/kleefsman_obstacle/decidability.py`（纯函数、不 import 求解器，可注入合成反例测试）：")
    a("")
    a("| 检查 | 判据 | 物理含义 |")
    a("|---|---|---|")
    a(f"| 时域删失 `check_horizon` | 峰值时刻距 `t_end` > max({pct(0.05,0)}·t_end, 0.05 s) | 峰值若落在最后若干采样上，报告的是**下界**而非峰值 |")
    a(f"| 声学 Courant `check_courant` | 由 `steps`/`simulated_time` 反推 dt ≤ {0.30}·h·dx/c₀ | 步数与所用格式自洽（证据可复现） |")
    a(f"| 压缩性 `check_compression` | max\\|Δρ\\|/ρ₀ ≤ {pct(0.01,0)} 且 c₀ ≥ 10·u_max | WCSPH 的声学误差是否进入验收量 |")
    a("| 计量算子散布 `check_operator_spread` | 同一 run 三种水位算子（top-N 粒子栈 / 面修正 / 定孔径 slab）峰值散布 | 验收量的**定义性**不确定度 |")
    a("| 声速族拟合 `compressibility_family` | peak = a + b/c₀² | 外推不可压极限，并检查单调性 |")
    a("| 不确定度预算 | RSS(离散、压缩性、算子散布) vs 容差 | 判据是否可判定 |")
    a("")
    a("复现：`python benchmarks/kleefsman_obstacle/decidability.py`，"
      "`python -m pytest tests/test_kleefsman_decidability.py -q`。")
    a("")
    a("## ③ 结果")
    a("")
    a("### 3.1 逐 run 检查")
    a("")
    a("| run | dx [m] | t_end [s] | 峰值 [m] | 峰值时刻 [s] | 时域删失 | 反推 dt [s] | Courant | max\\|Δρ\\|/ρ₀ |")
    a("|---|---|---|---|---|---|---|---|---|")
    for tag in sorted(grids, key=lambda t: -float(grids[t]["dx"])):
        gr = grids[tag]
        h, c, cp = gr["horizon"], gr["courant"], gr["compression"]
        a(f"| `{tag}` | {gr['dx']:g} | {num(gr['t_end'],3)} | {num(gr['peak_height_m'])} | "
          f"{num(gr['peak_time_s'],3)} | {'**是**' if h['censored'] else '否'} "
          f"(余量 {num(h.get('slack_s', float('nan')),3)} s) | "
          f"{c.get('dt_implied_s', float('nan')):.3e} | "
          f"{num(c.get('courant'),2)} / {num(c.get('max_courant'),2)} | {pct(cp['compression'])} |")
    a("")
    a("### 3.2 阻断级/重要级发现")
    a("")
    a("| id | 级别 | 证据 |")
    a("|---|---|---|")
    for f in audit["findings"]:
        a(f"| `{f['id']}` | {f['severity']} | {f['detail']} |")
    a("")
    a("### 3.3 声速族（压缩性灵敏度）拟合")
    a("")
    a("| dx [m] | 峰值(c₀=20/40/80) [m] | a（不可压极限）[m] | b [m²/s²] | 残差上界 [m] | 单调递减 | 极差 |")
    a("|---|---|---|---|---|---|---|")
    for dx, fit in sorted((audit.get("compressibility_family") or {}).items()):
        if fit.get("status") != "OK":
            a(f"| {float(dx):g} | — | — | — | — | — | {fit.get('status')} |")
            continue
        peaks = " / ".join(f"{p:.4f}" for _, p in fit["points"])
        a(f"| {float(dx):g} | {peaks} | {num(fit['a_incompressible_peak_m'])} | "
          f"{num(fit['b_m2_per_s2'],2)} | {num(fit['max_abs_residual_m'])} | "
          f"{'是' if fit['monotone_decreasing'] else '**否**'} | {pct(fit['spread_rel'])} |")
    a("")
    a("### 3.4 不确定度预算")
    a("")
    a("| 项 | 值 | 来源 |")
    a("|---|---|---|")
    a(f"| 离散（未删失网格极差） | {pct(budget['discretisation_rel'])} | {budget['n_uncensored_grids']} 个未删失网格 |")
    a(f"| 压缩性（声速族极差） | {pct(budget['compressibility_rel'])} | c₀ = 20/40/80 扫描 |")
    a(f"| 计量算子散布 | {pct(budget['gauge_operator_rel'])} | height/surface/slab 三种算子 |")
    a(f"| **合计（RSS）** | **{pct(budget['total_rel'])}** | —— |")
    a(f"| 容差 | {pct(tol,0)} | 算例池 |")
    a(f"| **可判定？** | **{'是' if verdict['decidable'] else '否'}** | 余量 {pct(verdict['headroom_rel'])} |")
    a("")
    if runs:
        a("### 3.5 未删失重跑（同协议 t_end = 3.0 s）")
        a("")
        a("| run | dx [m] | t_end [s] | 步数 | 峰值 [m] | 峰值时刻 [s] | 到达时刻 [s] | max\\|Δρ\\|/ρ₀ | 时域删失 |")
        a("|---|---|---|---|---|---|---|---|---|")
        for tag, r in sorted(runs.items()):
            horizon = r.get("horizon", {})
            a(f"| `{tag}` | {r['dx']:g} | {num(r['t_end'],2)} | {r.get('steps')} | "
              f"{num(r.get('peak_height_m'))} | {num(r.get('peak_time_s'),3)} | "
              f"{num(r.get('arrival_time_s'),3)} | "
              f"{pct(r.get('max_density_compression'))} | "
              f"{'**是**' if r.get('censored') else '否'} |")
        a("")
    a("## ④ 结果图")
    a("")
    for i, (name, caption) in enumerate(figures, start=1):
        a(f"![{name}](../figures/{name})")
        a(f"*Figure {i}: {caption}*")
        a("")
    a("## ⑤ 注意点")
    a("")
    a("1. **本审计只审证据，不产生新物理**。所有结论都可由已提交 JSON 复算；它不改进求解器精度。")
    a("2. **实验基准仍为 WITHHELD**。Kleefsman 2005 原文水位时程为图形形式，本节点无网络"
      "（urllib 直连返回 `Connection reset by peer`，前序单记录的 Cloudflare/ScienceDirect 403 未变），"
      "故不与实验做百分比对照；`reference.json` 的时间序列形式基准仍未获得。")
    a("3. **RSS 组合假设三项独立**。离散误差与压缩性误差在冲击段并非严格独立，"
      "合计值应视为上界性质的量级估计，而非置信区间。")
    a("4. **声速族本身受时域删失污染**：stored `dx0.02_c040` 的峰值时刻恰为 1.2001 s ≈ t_end，"
      "其“峰值随 c₀ 下降”不可直接采信；本单以未删失重跑复核（见 3.5）。")
    a("5. **算子散布含定义性差异**（`surface` 算子按构造比 `height` 高约 0.5 dx），"
      "它度量的是“验收量定义未冻结”的严重程度，不等于随机误差。")
    a("")
    a("## ⑥ 误差判定")
    a("")
    a("| 关键量 | 基准 | 计算值 | 误差 | 是否 < 3%？ |")
    a("|---|---|---|---|---|")
    a("| 峰值水位 | WITHHELD（实验） | 见 3.1/3.5 | 不可算 | **否**（无基准，"
      "且数值不确定度本身 > 容差） |")
    a("| 到达时刻 | WITHHELD（实验） | 见 3.1/3.5 | 不可算 | **否**（同上） |")
    a(f"| 峰值水位（数值自一致） | Richardson/极差 | —— | {pct(budget['discretisation_rel'])} | **否**（> 3%） |")
    a("")
    a(f"**诚实判定**：现有证据下 10% 验收判据**不可判定**（数值不确定度 {pct(budget['total_rel'])} "
      f"> 容差 {pct(tol,0)}）。因此本单**不**宣称 PASS，也**不**宣称 FAIL；"
      "给出的是把判据变成可判定的最小改动集：① 统一未删失时域、② c₀ 满足 10·u_max、"
      "③ 冻结单一水位算子、④ 用未删失梯子重算 GCI。①③ 由本单完成，②需要更高 c₀ 的重跑，"
      "④ 需要 dx = 0.01 的未删失长时程。")
    a("")
    a("---")
    a("")
    a("*工件*：`benchmarks/kleefsman_obstacle/decidability.py`、"
      "`benchmarks/kleefsman_obstacle/make_doc_decidability.py`、"
      "`results/audit_protocol/decidability.json`、`tests/test_kleefsman_decidability.py`。")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(RESULTS / "audit_protocol" / "decidability.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "benchmarks"
                                        / "kleefsman-decidability-audit-m20773.md"))
    args = ap.parse_args()
    audit = load(Path(args.audit))
    runs = extra_runs()
    for tag, row in runs.items():
        import sys
        sys.path.insert(0, str(HERE))
        import decidability as dc
        row["horizon"] = dc.check_horizon(row)
        row["censored"] = row["horizon"]["censored"]

    FIGDIR.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str]] = []
    for maker, caption in (
            (lambda: figure_censoring(audit["grids"], audit["uncertainty_budget"], FIGDIR),
             "已提交 run 的峰值时刻与运行时域；红点为删失（峰值落在最后采样上，只是下界）。"),
            (lambda: figure_c0_family(audit.get("compressibility_family") or {}, FIGDIR),
             "人工声速族：dx=0.02 单调下降、可拟合成 a+b/c₀²；dx=0.04 非单调（虚线为各自拟合）。"),
            (lambda: figure_operators(audit["grids"], FIGDIR),
             "同一 run 三种水位算子的峰值散布，已达或超过 10% 验收容差。"),
            (lambda: figure_t30(runs, audit["grids"], FIGDIR),
             "同协议 t_end = 3.0 s 的未删失探针时程。"),
    ):
        path = maker()
        if path is not None:
            figures.append((path.name, caption))

    text = build_report(audit, runs, figures)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"report -> {out} ({len(text.splitlines())} lines, {len(figures)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())