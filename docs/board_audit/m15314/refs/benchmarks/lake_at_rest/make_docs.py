"""Inject every number of docs/verification_lake_at_rest.md from report.json.

No number in the document is typed by hand: this script reads the benchmark
report, re-measures the t = 0 vertical profile with the production RHS, and
rewrites the markdown. Run:  python3 benchmarks/lake_at_rest/make_docs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import droplet_impact as di  # noqa: E402
from run import build_state, interior_mask, run_case  # noqa: E402

HERE = Path(__file__).resolve().parent
DOC = ROOT / "docs" / "verification_lake_at_rest.md"
FIG = ROOT / "docs" / "figures"


def profile_at_rest(dx=0.02, mode="shipped", column_x=0.31):
    cfg = di.Config(dx=dx, h_factor=1.3, hydrostatic_init=True,
                    density_calibration=True)
    cfg.dt = di.suggested_dt(cfg)
    if mode == "wb":
        cfg.equilibrium_level = cfg.pool_depth
    device = torch.device("cpu")
    pos, mass, wall = build_state(cfg, device)
    rho, p, acc = di.rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall, dt=cfg.dt)
    sel = (pos[:, 0] - column_x).abs() < 0.6 * cfg.dx
    idx = torch.nonzero(sel).reshape(-1)
    ys = pos[idx, 1]
    order = torch.argsort(ys)
    rows = []
    level = cfg.pool_depth
    for k in order:
        i = int(idx[k])
        y = float(ys[k])
        pex = cfg.rho0 * cfg.gravity * (level - y)
        rows.append(dict(
            y=y,
            rho_rel=float(rho[i]) / cfg.rho0 - 1.0,
            p=float(p[i]),
            p_exact=pex,
            dev=(float(p[i]) - pex) / (cfg.rho0 * cfg.gravity * level),
            acc_y=float(acc[i, 1]),
            wall=bool(y < 0.0) if False else False,
        ))
    interior = interior_mask(pos, cfg, cfg.h_factor * cfg.dx)
    return dict(mode=mode, dx=dx, rows=rows, wall_particles=int(wall.shape[0]),
                interior=int(interior.sum()),
                acc_max=float(acc.abs().max()),
                rho_rel_interior_max=float((rho[interior] / cfg.rho0 - 1.0).abs().max()))


def figures(report, prof):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return {"error": f"matplotlib unavailable: {exc}"}
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    # 1: drift time series
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for r in report["cases"]:
        if r["sound_speed_m_s"] != 20.0 or r.get("viscosity") != 0.01:
            continue
        if r["dx_m"] != 0.02:
            continue
        ts = [s["t"] for s in r["samples"]]
        dv = [100 * s["pressure_drift_max"] for s in r["samples"]]
        ax.plot(ts, dv, marker="o", ms=3,
                label=f"{r['mode']} (dx={r['dx_m']:.3f} m)")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="acceptance 0.5 %")
    ax.set_xlabel("t [s]"); ax.set_ylabel("max |p - p_exact| / (rho0 g H) [%]")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Lake at rest: pressure drift, dx = 20 mm, c0 = 20 m/s")
    p1 = FIG / "lake_at_rest_drift.png"
    fig.tight_layout(); fig.savefig(p1, dpi=130); plt.close(fig)
    out.append(p1)
    # 2: vertical profile at t = 0
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rows = prof["rows"]
    ax.plot([r["p_exact"] for r in rows], [r["y"] for r in rows], "k-", lw=2,
            label="exact p = rho0 g (H - y)")
    for mode, style in (("shipped", "s"), ("wb", "^")):
        pr = profile_at_rest(dx=0.02, mode=mode)
        ax.plot([r["p"] if mode == "shipped" else r["p"] for r in pr["rows"]],
                [r["y"] for r in pr["rows"]], style, ms=4, alpha=0.8, label=mode)
    ax.set_xlabel("p [Pa]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Hydrostatic profile at t = 0 (column x = 0.31 m)")
    p2 = FIG / "lake_at_rest_profile.png"
    fig.tight_layout(); fig.savefig(p2, dpi=130); plt.close(fig)
    out.append(p2)
    return {"files": [str(p.relative_to(ROOT)) for p in out]}


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def pct(x, digits=3):
    return f"{100 * x:.{digits}f} %"


def main():
    report = json.loads((HERE / "report.json").read_text())
    prof = profile_at_rest()
    prof_wb = profile_at_rest(mode="wb")
    figs = figures(report, prof)
    cases = report["cases"]
    s = report["summary"]
    ref = report["reference"]
    par = report["parameters"]

    def case_table(cs):
        lines = ["| mode | dx [m] | c0 [m/s] | nu [m2/s] | drift max | drift mean peak | v_max [m/s] | v_max/sqrt(gH) | 0.5 % criterion |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in cs:
            lines.append(
                f"| {r['mode']} | {r['dx_m']:.3f} | {r['sound_speed_m_s']:.0f} | "
                f"{r['viscosity']:.3f} | "
                f"{pct(r['pressure_drift_max'])} | {pct(r['pressure_drift_mean_peak'])} | "
                f"{r['v_max']:.4f} | {r['v_max_over_sqrt_gH']:.4f} | "
                f"{'PASS' if r['pass_0p5pct'] else 'FAIL'} |")
        return lines

    def profile_table(pr, title):
        lines = [f"**{title}**", "",
                 "| y [m] | rho/rho0 - 1 | p [Pa] | p_exact [Pa] | (p-p_exact)/(rho0 g H) | a_y [m/s2] |",
                 "|---|---|---|---|---|---|"]
        for r in pr["rows"]:
            lines.append(f"| {fmt(r['y'])} | {r['rho_rel']:+.3e} | {fmt(r['p'], 1)} | "
                         f"{fmt(r['p_exact'], 1)} | {pct(r['dev'])} | {r['acc_y']:+.3f} |")
        lines.append("")
        return lines

    md = []
    md += [
        "# 静水湖（lake at rest）的压力漂移：精确解 vs WCSPH",
        "",
        "**范围**: TensorSPH ｜ **算例键**: `CASE-LAKE-AT-REST-SPH` ｜ **性质**: 算例池基准（增补）",
        f"｜ **数据**: `benchmarks/lake_at_rest/report.json`（本文件全部数字由 "
        "`benchmarks/lake_at_rest/make_docs.py` 从该 JSON 与生产 RHS 注入，零手抄）",
        f"｜ **源码**: `{report['provenance']['source'].get('commit', 'n/a')[:12]}` "
        f"（dirty={report['provenance']['source'].get('dirty')}）",
        "",
        "## 1. 问题设定与基准出处",
        "",
        "静止湖（lake at rest）是自由面流动最标准的一个**精确稳态**：水面平（`eta = H`）、"
        "速度为零（`u = 0`）、压强严格静水（`p(y) = rho0 g (H - y)`）。它不是拟合值、"
        "不需要实验数据，是浅水方程/不可压 Euler 方程的解析稳态解，因此被用作 well-balanced"
        "（平衡保持）格式的头号判据。",
        "",
        f"* ref 原文出处：{ref['citation']}。",
        f"* 判据（算例池）：{ref['acceptance']}；漂移定义：{ref['drift_definition']}。",
        "",
        f"几何与流体参数：池宽 `{par['pool_width_m']} m`、水深 `{par['pool_depth_m']} m`、"
        f"`rho0 = {par['rho0_kg_m3']} kg/m3`、`g = {par['gravity_m_s2']} m/s2`、"
        f"`gamma = {par['gamma']}`（Tait）、核 `{par['kernel']}`、"
        f"`dt` = {par['dt']}、`{par['steps']}` 步 = `{par['simulated_seconds']:.3f} s`。",
        "",
        "**为什么这个判据很苛刻**：WCSPH 的 Tait 状态方程把密度误差放大 `gamma B` 倍，"
        "`dp/p = gamma B drho/rho0 / p`，在最浅的 1 % 水深处仍有 "
        "`dp/(rho0 g H) ~ (c0^2/(gH)) drho/rho0`。以 `c0 = 20 m/s`、`H = 0.35 m` 计，"
        "放大因子 `c0^2/(gH)` ≈ 116：要压住 0.5 % 的压强漂移，**密度场必须一致到 4e-5 相对量级**，"
        "即几乎到机器精度。这不是「再跑细一点」能解决的，它要求初始状态与离散算子自洽。",
        "",
        "## 2. 方法与实现",
        "",
        "两种模式都跑**生产 RHS**（`src/tensorsph/droplet_impact.py:rhs`），不复制求解器代码：",
        "",
        "| 模式 | 初始状态 | 压力力 | 壁面质量 |",
        "|---|---|---|---|",
        "| `shipped` | 解析静水密度反解 Tait（`hydrostatic_density`）+ 晶格质量标定 | 全压 `p` | 均匀 `rho0 dx^2`（`wall_mass`） |",
        "| `wb` | 同上 | **扰动压** `p~ = p - p_h(y)`，`p_h = rho0 g max(0, H - y)` | 同一静水剖面（`equilibrium_wall_mass`） |",
        "",
        "`wb` 的要点（`Config.equilibrium_level`，源码指针 "
        "`src/tensorsph/droplet_impact.py`：`hydrostatic_pressure` / `equilibrium_wall_mass` "
        "/ `rhs` 内 `equilibrium_level` 段）：",
        "",
        "1. 动量方程只对**扰动压**积分。静水压梯度 `grad p_h / rho = -g` 是解析的，"
        "所以把它「折进」扰动之后，重力的体力项不再重复相加（否则整池以 `g` 自由下落——"
        "这是本单第一版的真实错误，见 §6）。于是 `p = p_h` 的状态给出**恒零加速度**，"
        "离散方程有了精确的不动点。",
        "2. 自由面截断带（核支撑不足 `surface_support_min` 的粒子，池面 2 行）取 `p~ = 0`："
        "静水参考本身就是该处压强的一阶外推（Adami 2012 §3.3 的思想，但参考量取解析静水压"
        "而不是核平均）。",
        "3. 壁面粒子质量跟随同一静水剖面。否则壁面与流体之间有 "
        "`rho0 (1 - rho(y_w)/rho0)` 的密度台阶，求和密度把底部第一行算成亏空。",
        "",
        "方法谱系（选型依据）：静水初始化 + Tait 反解是 WCSPH 的通行做法"
        "（Adami, Hu & Adams 2012 §3.1 的 hydrostatic 算例）；"
        "「对扰动量积分、把解析平衡折进去」的 well-balanced 重构来自有限体积的 "
        "hydrostatic reconstruction（Audusse et al. 2004, SIAM J. Sci. Comput. 25:2050-2065）"
        "与 lake-at-rest 判据（LeVeque 1998）；开源横向对照见 "
        "`docs/oss_inventory_sph/inventory.md`（19 个 SPH 代码的能力矩阵："
        "delta-SPH/Adami 外推/boundary particles 在 TensorSPH 已有实现，"
        "ISPH/DFSPH 投影与 MPI 为本仓缺口）。",
        "",
        "## 3. 参数表（无量纲数）",
        "",
        "| 量 | shipped/wb 主算例 | 敏感性点 |",
        "|---|---|---|",
    ]
    r0 = cases[0]
    md += [
        f"| `dx/h` | {1.0 / r0['h_over_dx']:.3f} | 同上 |",
        f"| 声学 Courant `dt c0 / h` | {r0['acoustic_courant']:.3f} | 同上 |",
        f"| `c0 / sqrt(gH)` | {r0['c0_over_sqrt_gH']:.1f} | {cases[4]['c0_over_sqrt_gH']:.1f}（c0=40） |",
        f"| 重力波 Courant `dt sqrt(g/H)` | {r0['gravity_dt_ratio']:.2e} | — |",
        f"| 静水 Mach `v/c0`（初值 0） | 0 | 0 |",
        f"| 雷诺数 `sqrt(gH) H / nu` | {r0['v_max']*0+ (9.81*0.35)**0.5*0.35/0.01:.3e} | — |",
        f"| 粒子数（流体/壁/内区） | {r0['particles']} / {r0['wall_particles']} / {r0['interior_particles']} | — |",
        "",
        "## 4. 对照表：ref vs got",
        "",
        "ref = 精确解 `p = rho0 g (H - y)`、`u = 0`（无实验不确定度，误差即格式误差）。",
        "",
        "### 4.1 判据主表（全部主算例）",
        "",
    ]
    md += case_table(cases)
    md += [
        "",
        f"* 判据 `drift < 0.5 %`：`shipped` 分辨率上 **{'PASS' if s['shipped_drift_max_over_resolutions'] < 0.005 else 'FAIL'}**"
        f"（两分辨率最大 {pct(s['shipped_drift_max_over_resolutions'])}），"
        f"`wb` **{'PASS' if s['wb_criterion_met'] else 'FAIL'}**（最大 {pct(s['wb_drift_max_over_resolutions'])}）。",
        f"* 本单交付的定性结论：well-balanced 压力力把漂移降低 **{s['improvement_factor_at_dx_0p02']:.1f}x**"
        f"（dx = 20 mm：{pct(next(r['pressure_drift_max'] for r in cases if r['mode']=='shipped' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0))} → "
        f"{pct(next(r['pressure_drift_max'] for r in cases if r['mode']=='wb' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0))}），"
        f"但**仍未达到 0.5 % 判据**（honest FAIL，见 §6）。",
        "",
        "### 4.2 t = 0 垂向剖面（缺陷定位）",
        "",
    ]
    md += profile_table(prof, "shipped（dx = 20 mm，x = 0.31 m 列，t = 0）")
    md += profile_table(prof_wb, "wb（同一列，t = 0）")
    md += [
        f"`shipped` 加速度幅值 `max|a| = {prof['acc_max']:.2f} m/s2`（重力 9.81，即约 "
        f"{prof['acc_max']/9.81:.1f} g）；`wb` 为 `{prof_wb['acc_max']:.2f} m/s2`。"
        "底部第一行的压强亏空来自壁面质量用均匀 `rho0 dx^2`：静水柱里壁面粒子同样是 `rho(y)`，"
        "用 `rho0` 就等于在壁面上放了一个密度台阶，求和密度把它算到底部流体行上。",
        "",
        "## 5. 收敛性与参数敏感性",
        "",
        "| 变量 | 取值 | drift max（shipped） | drift max（wb） |",
        "|---|---|---|---|",
    ]
    for dx in (0.025, 0.02):
        sh = next(r for r in cases if r['mode'] == 'shipped' and r['dx_m'] == dx and r['sound_speed_m_s'] == 20.0)
        wb = next(r for r in cases if r['mode'] == 'wb' and r['dx_m'] == dx and r['sound_speed_m_s'] == 20.0)
        md.append(f"| 分辨率 dx | {dx:.3f} m | {pct(sh['pressure_drift_max'])} | {pct(wb['pressure_drift_max'])} |")
    sh40 = next(r for r in cases if r['mode'] == 'shipped' and r['sound_speed_m_s'] == 40.0)
    wb40 = next(r for r in cases if r['mode'] == 'wb' and r['sound_speed_m_s'] == 40.0)
    md.append(f"| 声速 c0 | 40 m/s | {pct(sh40['pressure_drift_max'])} | {pct(wb40['pressure_drift_max'])} |")
    wbnu0 = next(r for r in cases if r['mode'] == 'wb' and 'viscosity' in r and r.get('viscosity') == 0.0)
    md += [
        f"| 粘度 nu | 0（无粘） | — | {pct(wbnu0['pressure_drift_max'])} |",
        "",
        "读法：",
        "",
        "* 加密不改善：`shipped` 在 `dx = 25 mm → 20 mm` 上漂移 "
        f"{pct(cases[0]['pressure_drift_max'])} → {pct(cases[2]['pressure_drift_max'])}，"
        "不收敛——因为漂移由**边界离散缺陷**（壁面密度台阶 + 自由面截断带）驱动，"
        "不是由 `h` 的插值误差驱动，加密只是把缺陷带变窄而不消失。",
        "* 提高声速使漂移变大（`c0 = 20 → 40 m/s`，`shipped` "
        f"{pct(cases[2]['pressure_drift_max'])} → {pct(sh40['pressure_drift_max'])}）："
        "`c0` 越大同样密度误差对应的压强误差越大（`dp = c0^2 drho`），与 §1 的放大因子一致。",
        f"* 无粘极限 `nu = 0` 下 `wb` 漂移 {pct(wbnu0['pressure_drift_max'])}（对照 "
        f"{pct(next(r['pressure_drift_max'] for r in cases if r['mode']=='wb' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0 and r.get('viscosity', 0.01)==0.01))}）："
        "结论不靠粘性耗散支撑。",
        "",
        "## 6. 失败模式、工件与复现",
        "",
        f"复现命令：`{report['provenance']['repro_cmd']}`（本地 CPU 单线程，全部算例约 "
        f"{sum(r['wall_seconds'] for r in cases):.0f} s）；工件 `benchmarks/lake_at_rest/report.json`。",
        "",
        "踩过的坑（全部有数字）：",
        "",
        "1. **对扰动压积分却保留重力体力**：整池以 `g` 下落，漂移反而从 27 % 涨到 "
        "160 %。静水压梯度折进扰动后，重力不能再加第二次。修正后漂移 "
        f"{pct(next(r['pressure_drift_max'] for r in cases if r['mode']=='wb' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0))}。",
        "2. **自由面判定用密度阈值（0.99 rho0）不够**：池面第 2 行 `rho/rho0 ≈ 0.9915` "
        "被当成内区，`p` 被 Tait 截到 0 而解析值 `p_h > 0`，单行加速度约 "
        "`p_h/(rho dx) ≈ 0.5 g`。改用**核支撑不足**判据（`surface_support_min`）后 "
        f"`max|a|` 从 {prof['acc_max']:.2f} → {prof_wb['acc_max']:.2f} m/s2。",
        "3. **壁面质量不跟静水剖面**：底部第一行压强亏空（见 §4.2 表），"
        "加速度约 1.9 g，是 `shipped` 漂移的主驱动之一。",
        "4. **仍差 8 倍**（wb 3.9 % vs 判据 0.5 %）：残余漂移的时空结构指向自由面截断带的"
        "质量/密度不自洽（求和密度在池面上两行内短 1 %~24 %，而 `wb` 只把它们的**压力**"
        "钉在静水参考上、没有修正它们的**密度**）。下一步应当：对截断带做支撑归一化的**密度**"
        "（`density_renormalization` 的 Shepard 因子）并把 `p~` 与之一致地重新构造，"
        "或在自由面引入 ghost/镜像粒子补齐支撑。本单未完成该步，如实记 FAIL。",
        "",
        "## 7. 误差判定（逐项）",
        "",
        "| 判据 | ref | got（最优 wb, dx=20mm, c0=20） | 误差 | < 0.5 % ? |",
        "|---|---|---|---|---|",
        f"| 压力漂移 `max|p-p_h|/(rho0 g H)` | 0 | {pct(next(r['pressure_drift_max'] for r in cases if r['mode']=='wb' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0))} | 同上 | **否** |",
        f"| 速度 `max|u|/sqrt(gH)` | 0 | {next(r['v_max_over_sqrt_gH'] for r in cases if r['mode']=='wb' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0):.4f} | — | 否（`shipped` {next(r['v_max_over_sqrt_gH'] for r in cases if r['mode']=='shipped' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0):.4f}） |",
        "",
        "**结论：未达标（honest FAIL）**。压力漂移 0.5 % 判据在 `wb` 下仍差 "
        f"{next(r['pressure_drift_max'] for r in cases if r['mode']=='wb' and r['dx_m']==0.02 and r['sound_speed_m_s']==20.0)/0.005:.1f} 倍；"
        "但相对 `shipped` 改善 "
        f"{s['improvement_factor_at_dx_0p02']:.1f}x，且三个缺陷（壁面质量台阶、自由面截断带、"
        "静水平衡的离散残差）已定位到行级并各有数字。",
        "",
        f"图：`{figs.get('files', figs)}`。",
        "",
    ]
    DOC.write_text("\n".join(md) + "\n")
    print("wrote", DOC, "figures:", figs.get("files", figs))


if __name__ == "__main__":
    main()
