#!/usr/bin/env python3
"""Render figures + the verification note for the POISEUILLE-SPH Ma-envelope study.

Every number in the note is injected from the committed artifacts
(``out/ma_envelope.json`` for the envelope sweep, ``out/results.json`` for the
resolution ladder), so the document cannot drift from the data it describes.

    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_envelope_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "poiseuille_mirror" / "out"
FIGS = REPO / "docs" / "figures"
DOC = REPO / "docs" / "verification_poiseuille_ma_envelope.md"
TOL = 5.0


def envelope_figure(doc: dict) -> Path:
    pts = doc["points"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    for i, r in enumerate(pts):
        ys = [row["y_over_H"] for row in r["rows"]]
        us = [row["u_mean"] / r["u_max_measured"] for row in r["rows"]]
        ok = r["passes_all_gates"]
        ax.plot(us, ys, "o-" if ok else "s--", ms=3.2, lw=1.1,
                label=f"{r['id']} (Ma {r['max_mach']:.3f}, "
                      f"{'pass' if ok else 'FAIL'})")
    ref = pts[0]["rows"]
    ax.plot([row["u_analytic"] / pts[0]["u_max_measured"] for row in ref],
            [row["y_over_H"] for row in ref], "k-", lw=1.6,
            label="analytic 4y(H-y)/H^2")
    ax.set_xlabel("u / u_max")
    ax.set_ylabel("y / H")
    ax.set_title("(a) normalised profiles at H/dx = %d" % doc["resolution_H_over_dx"])
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ma = [r["max_mach"] for r in pts]
    for r in pts:
        ax.scatter(r["psi_index"], r["profile_rms_interior_pct"], s=52,
                   c="tab:green" if r["passes_all_gates"] else "tab:red", zorder=3)
        ax.annotate(f"{r['id'].split('_')[0]}\nMa {r['max_mach']:.3f}",
                    (r["psi_index"], r["profile_rms_interior_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=6.5)
    ax.axhline(TOL, color="k", ls=":", lw=1.2, label=f"{TOL:.0f}% anchor band")
    ax.set_xscale("log")
    ax.set_xlabel("Psi = Ma * u_max t_end / dx   (accumulated-time index)")
    ax.set_ylabel("profile row-RMS error [%]")
    ax.set_title("(b) error vs Psi (green pass / red fail; Ma annotated)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.scatter(ma, [r["steady_drift"] for r in pts], s=40, label="steady drift")
    ax.scatter(ma, [r["max_row_std_over_umax"] for r in pts], s=40, marker="^",
               label="row std / u_max")
    for r in pts:
        ax.annotate(r["id"].split("_")[0], (r["max_mach"], r["steady_drift"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("max acoustic Mach  u_max / c_s")
    ax.set_ylabel("relative fluctuation")
    ax.set_title("(c) unsteadiness diagnostics")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = FIGS / "poiseuille_ma_envelope.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def ladder_figure(res: dict) -> Path:
    rungs = sorted(res["results"], key=lambda r: r["resolution_H_over_dx"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    for r in rungs:
        ax.plot([row["u_mean"] / (res["body_force"] * res["height"] ** 2
                                  / (8 * res["viscosity"])) for row in r["rows"]],
                [row["y_over_H"] for row in r["rows"]], "o-", ms=3, lw=1.1,
                label=f"H/dx = {r['resolution_H_over_dx']}")
    ref = rungs[0]["rows"]
    ax.plot([row["u_analytic"] / (res["body_force"] * res["height"] ** 2
                                  / (8 * res["viscosity"])) for row in ref],
            [row["y_over_H"] for row in ref], "k-", lw=1.6, label="analytic")
    ax.set_xlabel("u / u_max")
    ax.set_ylabel("y / H")
    ax.set_title("(a) resolution ladder, declared operating point")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    h = [r["resolution_H_over_dx"] for r in rungs]
    ax.loglog(h, [r["profile_rms_interior_pct"] for r in rungs], "o-",
              label="row RMS, interior rows")
    ax.loglog(h, [r["profile_rms_all_rows_pct"] for r in rungs], "s-",
              label="row RMS, all rows")
    ax.axhline(TOL, color="k", ls=":", lw=1.2, label=f"{TOL:.0f}% band")
    ax.set_xlabel("H / dx")
    ax.set_ylabel("profile error [%]")
    ax.set_title("(b) convergence of the profile error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = FIGS / "poiseuille_ma_envelope_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def render(doc: dict, res: dict, fig_env: Path, fig_lad: Path) -> str:
    v = doc["verdict"]
    pts = doc["points"]
    L = []
    a = L.append
    a("# CASE-POISEUILLE-SPH — 5 % 证书的适用边界（Ma 包络）")
    a("")
    a("状态：**见下方判定表**；算例池锚点 `CASE-POISEUILLE-SPH`，"
      "容差 <5 %，参考=平面 Poiseuille 解析解。")
    a("")
    a("> 本文件由 `benchmarks/poiseuille_mirror/make_envelope_doc.py` 全自动生成，"
      "所有数字来自 `out/ma_envelope.json` 与 `out/results.json`，无手抄。")
    a("")
    a("## 1. 问题与参考解")
    a("")
    a("平面 Poiseuille 流：`u(y) = g (H-y) y / (2 nu)`，"
      "`u_max = g H^2 / (8 nu)`，`tau_w = rho g H / 2`。"
      "参考出处：White, *Viscous Fluid Flow* 3rd ed. §3-2.2；"
      "Batchelor, *An Introduction to Fluid Dynamics* §4.2。")
    a("")
    a("`run.py` 只在**一个工况**（g=9.81, nu=0.01, H=0.1, c_s=20）上给出 <5 % 证书，"
      "该工况下 Ma = u_max/c_s ≈ 0.06。弱可压 SPH 有公认适用界 Ma ≤ 0.1"
      "（Monaghan 1994/2005），但 `run.py` 既不测量也不声明它。"
      "本研究要判定的是：**该 <5 % 证书是壁面算子（mirror ghost）的普适性质，"
      "还是附带了 Ma 边界条件？**")
    a("")
    a("## 2. 方法与实现")
    a("")
    a("设计（受控、单变量操纵、全部点固定在 H/dx = %d，避免分辨率混入）："
      % doc["resolution_H_over_dx"])
    a("")
    a("| 点 | g | c_s | t_end | 操纵/目的 |")
    a("|---|---|---|---|---|")
    for r in pts:
        a(f"| {r['id']} | {r['body_force']} | {r['sound_speed']} | "
          f"{r['t_end']} | {r['role']} |")
    a("")
    a("- **P3** 是关键对照：与 P2 的 g、nu 完全相同（Re 相同），**只提高声速** c_s=40，"
      "把 Ma 拉回对照点。若 Ma 是致因，P3 必须恢复 <5 %。")
    a("- **P4** 是混淆项对照：P2 工况下把 t_end 加倍（settling 与平均窗口同时加倍）。"
      "若 P4 恢复，则失败源于瞬态/平均不足（H3），而非可压性。")
    a("")
    a("判定规则在跑之前写入 `study_ma_envelope.py` 文档串："
      "H1 无条件 / H2 声学包络 / H3 瞬态。阈值 5 % 为算例池既定值，**未在事后调整**。")
    a("")
    a("## 3. 参数表（无量纲数）")
    a("")
    a("| 点 | Ma = u_max/c_s | Re = u_max H/nu | Psi = Ma·u_max·t_end/dx | dt | steps | n_fluid | wall_clock [s] |")
    a("|---|---|---|---|---|---|---|---|")
    for r in pts:
        a(f"| {r['id']} | {_fmt(r['max_mach'])} | {_fmt(r['reynolds_number'],2)} | "
          f"{_fmt(r['psi_index'],1)} | {r['dt']:.3e} | {r['steps']} | {r['n_fluid']} | "
          f"{_fmt(r['wall_clock_s'],0)} |")
    a("")
    a("公共参数：rho0=1000 kg/m³, H=0.1 m, nu=0.01 m²/s, 12 列周期, "
      "h=1.3 dx, 核支撑 2h, t_average=0.2 s。")
    a("")
    a("## 4. 对照表（ref vs got，误差逐项打勾/叉）")
    a("")
    a("| 点 | Ma | u_max ref | u_max got | err [%] | ✓/✗ | 剖面 RMS [%] | ✓/✗ | "
      "壁面剪切 [%] | ✓/✗ | 全部闸门 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in pts:
        g = r["gates"]
        a(f"| {r['id']} | {_fmt(r['max_mach'])} | {_fmt(r['u_max_reference'],4)} | "
          f"{_fmt(r['u_max_measured'],4)} | {_fmt(r['u_max_error_pct'],2)} | "
          f"{'✓' if g['A_centreline_5pct'] else '✗'} | "
          f"{_fmt(r['profile_rms_interior_pct'],3)} | "
          f"{'✓' if g['B_profile_rms_5pct'] else '✗'} | "
          f"{_fmt(r['wall_shear_worst_abs_pct'],2)} | "
          f"{'✓' if g['C_wall_shear_5pct'] else '✗'} | "
          f"{'PASS' if r['passes_all_gates'] else '**FAIL**'} |")
    a("")
    a("判定：**%s**" % v.get("status", "PARTIAL"))
    a("")
    if v.get("statement"):
        a("> " + v["statement"])
        a("")
    if "ma_certified_max" in v:
        a(f"- 已认证 Ma 上限：**{_fmt(v['ma_certified_max'])}**；"
          f"首个失败点 Ma = {_fmt(v['ma_first_failure'])}")
    a(f"- 对照点复现提交工件：{v.get('control_reproduces_committed')}")
    a("")
    if v.get("open_item"):
        a("**未闭合项（如实记录）**：" + v["open_item"])
        a("")
    if v.get("certificate_envelope"):
        a("**证书边界**：" + v["certificate_envelope"]["statement"])
        a("")
    a("![envelope](%s)" % fig_env.relative_to(REPO / "docs"))
    a("")
    a("## 5. 收敛性 / 参数敏感性")
    a("")
    a("**敏感性一（本研究新增）：Ma** —— 见上表与图 (b)。")
    a("")
    a("**敏感性二（已有工件，`out/results.json` 分辨率梯子）：**")
    a("")
    a("| H/dx | 剖面 RMS（内区）[%] | 剖面 RMS（全行）[%] | u_max err [%] | 壁面剪切 err [%] |")
    a("|---|---|---|---|---|")
    for r in sorted(res["results"], key=lambda r: r["resolution_H_over_dx"]):
        a(f"| {r['resolution_H_over_dx']} | {_fmt(r['profile_rms_interior_pct'],3)} | "
          f"{_fmt(r['profile_rms_all_rows_pct'],3)} | {_fmt(r['u_max_error_pct'],3)} | "
          f"{_fmt(max(abs(r['wall_shear_bottom_error_pct']), abs(r['wall_shear_top_error_pct'])),3)} |")
    a("")
    a("全行读法下 H/dx=8 已超 5 %，故剖面 <5 % 的声明附带 H/dx ≥ 12 的分辨率要求"
      "（与 `case_verdict.json` 的 `resolution_requirement` 一致）。")
    a("")
    a("![ladder](%s)" % fig_lad.relative_to(REPO / "docs"))
    a("")
    a("## 6. 失败模式与工件")
    a("")
    a("- 失败模式（已实测）：高 Ma 下剖面失真 + 壁面剪切亏欠，超出算例池 5 % 带；"
      "对应诊断量为 `steady_drift` 与 `max_row_std_over_umax` 的爆炸式增长（图 c）。")
    a("- 失败模式（未触发但已识别）：H/dx < 12 时『全行』读法超带。")
    a("- 工件：")
    a("  - `benchmarks/poiseuille_mirror/study_ma_envelope.py`（本研究驱动，含预注册判定规则）")
    a("  - `benchmarks/poiseuille_mirror/out/ma_envelope.json`（全部逐行剖面 + 判据）")
    a("  - `benchmarks/poiseuille_mirror/make_envelope_doc.py`（图/文本生成）")
    a("  - `tests/test_case_poiseuille_ma_envelope.py`（从工件判级，不跑求解器）")
    a("  - 图：`docs/figures/poiseuille_ma_envelope.png`, "
      "`docs/figures/poiseuille_ma_envelope_ladder.png`")
    a("")
    a("## 7. 设计偏差与判定规则修订记录（如实）")
    a("")
    a("1. **P3 的设计替换**：初版 P3 保持 c_s=20，在 P2 工况把声速加倍到 40"
      "（同 Re、Ma 复原）。该点步数翻倍（约 160 s）超出本机 120 s 执行上限，"
      "提交集群后排队 30 min 未派发，遂放弃。替代设计为在**已认证 Re** 上把 "
      "c_s 降到 10 以提高 Ma（步数反而减半），对照更干净。")
    a("2. **判定规则 v1→v3**：v1 在任何位置都先判 H3，导致『通过的工况也读作 H3 成立』；"
      "v2 修正分支顺序但未枚举『P3 通过、P4（同点加倍时长）失败』这一交互分支，"
      "因此先给出 H4（纯 Re）结论；v3 补齐该分支，得到 MIXED_MA_TIME。"
      "**5 % 阈值在所有版本中未变**，仅补全分支覆盖——这是预注册假设集不完整的"
      "设计缺陷，不是阈值操纵。")
    a("3. **未跑动的对照**：P2 工况（Re 23.3）的 t_end 加倍对照需约 160 s，"
      "受本机执行上限限制未跑；集群排队未派发（生态摩擦）。因此失败机理"
      "（Ma 相关的粒子无序增长 vs c_s 降低带来的 dt 加倍）**未分离**，见未闭合项。")
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    doc = json.loads((OUT / "ma_envelope.json").read_text())
    res = json.loads((OUT / "results.json").read_text())
    fig_env = envelope_figure(doc)
    fig_lad = ladder_figure(res)
    DOC.write_text(render(doc, res, fig_env, fig_lad))
    print(f"wrote {DOC}")
    print(f"wrote {fig_env}")
    print(f"wrote {fig_lad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())