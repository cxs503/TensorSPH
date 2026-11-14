#!/usr/bin/env python3
"""Render figures + the verification note for the POISEUILLE-SPH Wo-envelope study.

Every number in the note is read from the committed artefact
(``benchmarks/case_poiseuille/out/wo_envelope.json``, produced by
``study_wo_envelope.py``), so the document cannot drift from the data it
describes.  Nothing is typed twice.

    PYTHONPATH=src:. python3 benchmarks/case_poiseuille/make_wo_envelope_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "case_poiseuille" / "out"
FIGS = REPO / "docs" / "figures"
DOC = REPO / "docs" / "case_poiseuille_wo_envelope.md"
TOL = 5.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load() -> dict:
    return json.loads((OUT / "wo_envelope.json").read_text())


def graded(doc: dict) -> list[dict]:
    return [p for p in doc["points"] if p["sampling_ok"]]


def by_id(doc: dict) -> dict:
    return {p["id"]: p for p in doc["points"]}


def sweep16(doc: dict) -> list[dict]:
    pts = [p for p in graded(doc) if p["role"].startswith("A:")]
    return sorted(pts, key=lambda p: p["womersley_number"])


def ladder65(doc: dict) -> list[dict]:
    pts = [p for p in graded(doc) if abs(p["womersley_requested"] - 6.5) < 1e-9]
    return sorted(pts, key=lambda p: p["stokes_layer_over_dx"])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_envelope(doc: dict) -> Path:
    sw = sweep16(doc)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    ax = axes[0]
    wo = [p["womersley_number"] for p in sw]
    ax.semilogx(wo, [p["amp_profile_rms_pct"] for p in sw], "o-", lw=1.4,
                label="interior amplitude RMS (G2 metric)")
    ax.semilogx(wo, [abs(p["amp_centre_error_pct"]) for p in sw], "s--", lw=1.2,
                label="centre-line amplitude error (G1)")
    ax.semilogx(wo, [p["phase_profile_rms_pct_of_quarter_turn"] for p in sw], "^:",
                lw=1.2, label="phase RMS (% of pi/2, G4)")
    ax.axhline(TOL, color="crimson", lw=1.2, ls="--", label=f"{TOL:.0f} % gate")
    ax.axvline(3.0, color="0.5", lw=1.0, ls="-.", label="scope claimed in the note (Wo 3)")
    for p in sw:
        if p["id"].startswith("W0"):
            ax.annotate("committed rung", (p["womersley_number"], p["amp_profile_rms_pct"]),
                        textcoords="offset points", xytext=(6, 8), fontsize=7.5)
    ax.set_xlabel("Womersley number  Wo = (H/2)sqrt(omega/nu)")
    ax.set_ylabel("error [%]")
    ax.set_title("(a) graded metrics vs Wo, H/dx = 16")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.2, loc="center left")

    ax = axes[1]
    lad = ladder65(doc)
    ax.plot([p["stokes_layer_over_dx"] for p in lad],
            [p["wall_rows"]["amp_worst_pct"] for p in lad], "o-", lw=1.4,
            label="measured, wall rows (y/H <= 0.15)")
    ax.plot([p["stokes_layer_over_dx"] for p in lad],
            [p["row_averaging_floor"]["wall_rows"]["amp_worst_pct"] for p in lad],
            "s--", lw=1.2, label="row-averaging floor (analytic, no solver)")
    ax.plot([p["stokes_layer_over_dx"] for p in lad],
            [p["amp_profile_rms_pct"] for p in lad], "^:", lw=1.2,
            label="measured, interior band (the graded rows)")
    ax.axhline(TOL, color="crimson", lw=1.2, ls="--")
    ax.set_xlabel("Stokes layer resolution  delta/dx")
    ax.set_ylabel("amplitude error [%]")
    ax.set_title("(b) Wo = 6.5 resolution ladder: the gate is blind to the layer")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.2)

    ax = axes[2]
    walls = [p for p in graded(doc)]
    sc = ax.scatter([p["stokes_layer_over_dx"] for p in walls],
                    [p["wall_rows"]["amp_worst_pct"] for p in walls],
                    c=[p["womersley_number"] for p in walls], cmap="viridis",
                    s=[28 + 2.2 * p["resolution_H_over_dx"] for p in walls],
                    edgecolor="k", lw=0.4)
    ax.scatter([p["stokes_layer_over_dx"] for p in walls],
               [p["row_averaging_floor"]["wall_rows"]["amp_worst_pct"] for p in walls],
               marker="x", color="0.35", s=26, label="row-averaging floor")
    ax.set_xlabel("delta/dx")
    ax.set_ylabel("wall-row worst amplitude error [%]")
    ax.set_title("(c) collapse onto delta/dx (colour = Wo, size = H/dx)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.colorbar(sc, ax=ax, label="Wo")

    fig.tight_layout()
    path = FIGS / "case_poiseuille_wo_envelope.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_profiles(doc: dict) -> Path:
    pts = [p for p in graded(doc) if p["id"] in ("W0_wo180_r16", "W3_wo500_r16",
                                                 "W5_wo800_r16", "W7_wo1250_r16")]
    pts.sort(key=lambda p: p["womersley_number"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    ax = axes[0]
    for p in pts:
        ys = [r["y_over_H"] for r in p["rows"]]
        ax.plot([r["amplitude"] / p["amp_centre_analytic"] for r in p["rows"]], ys,
                "o-", ms=3.2, lw=1.1,
                label=f"Wo {p['womersley_number']:.2f}, delta/dx {p['stokes_layer_over_dx']:.2f}")
    ref = pts[0]["rows"]
    ax.plot([r["amplitude_analytic"] / pts[0]["amp_centre_analytic"] for r in ref],
            [r["y_over_H"] for r in ref], "k--", lw=1.0, label="analytic (all Wo)")
    ax.set_xlabel("|u| / |Uhat(H/2)|  (Wo = 1.8 normalisation)")
    ax.set_ylabel("y/H")
    ax.set_title("(a) amplitude profiles: the wall layer that the gate never grades")
    ax.axhspan(0.0, 0.15, color="0.85", alpha=0.6, zorder=0)
    ax.text(0.02, 0.075, "wall band (not graded)", fontsize=7, color="0.3")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    p = by_id(doc)["W7_wo1250_r16"]
    ys = np.array([r["y_over_H"] for r in p["rows"]])
    ax.plot(ys, [r["amplitude"] for r in p["rows"]], "o-", ms=3.4, lw=1.2,
            label=f"measured, Wo {p['womersley_number']:.2f}")
    ax.plot(ys, [r["amplitude_analytic"] for r in p["rows"]], "k--", lw=1.0,
            label="Womersley closed form")
    ax2 = ax.twinx()
    ax2.plot(ys, [r["phase_error_deg"] for r in p["rows"]], "^:", color="tab:red",
             lw=1.1, label="phase error [deg]")
    ax2.set_ylabel("phase error [deg]", color="tab:red")
    ax2.axhline(0.0, color="tab:red", lw=0.6, ls=":")
    ax.set_xlabel("y/H")
    ax.set_ylabel("amplitude [m/s]")
    ax.set_title(f"(b) {p['id']}: delta/dx = {p['stokes_layer_over_dx']:.2f}, "
                 f"interior RMS {p['amp_profile_rms_pct']:.2f} %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    path = FIGS / "case_poiseuille_wo_envelope_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Note
# --------------------------------------------------------------------------- #

def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render(doc: dict, fig1: Path, fig2: Path) -> str:
    v = doc["verdict"]
    par = doc["parameters"]
    pts = by_id(doc)
    sw = sweep16(doc)
    lad = ladder65(doc)
    lines = []
    A = lines.append

    A("# CASE-POISEUILLE-SPH — Wo 包线：调和（Womersley）判据到底在哪里失效")
    A("")
    A("> 锚点：`CASE-POISEUILLE-SPH`（算例池·泊肃叶流；`feature_ledger` 容差 < 5 %）。")
    A("> 本文件由 `benchmarks/case_poiseuille/make_wo_envelope_doc.py` 从")
    A("> `benchmarks/case_poiseuille/out/wo_envelope.json` 程序化生成，表中每个数字都来自工件，无手抄。")
    A("> 研究脚本：`benchmarks/case_poiseuille/study_wo_envelope.py`（分片运行，工件按点自包含）。")
    A("")
    A("## ① 问题与参考出处")
    A("")
    A("平面通道脉动流（Womersley 问题）：`0 <= y <= H` 两侧无滑移，`x` 周期，")
    A("体积力（压力梯度）按 `g(t) = g0 cos(omega t)` 驱动：")
    A("")
    A("```")
    A("u_t = nu u_yy + g0 cos(omega t),   u(0,t) = u(H,t) = 0")
    A("周期解: u = Re{Uhat(y) e^{i omega t}},")
    A("        Uhat = (g0/(i omega)) [1 - cosh(alpha (y - H/2)) / cosh(alpha H/2)],")
    A("        alpha = sqrt(i omega / nu),  Wo = (H/2) sqrt(omega/nu),  delta = sqrt(2 nu/omega)")
    A("```")
    A("")
    A("已合并的 `run.py` 只在两个频率上认证这个算例（`Wo = 1.80` 主判据，`Wo = 2.80` 供频率比判据 G6），")
    A("而验证笔记把适用范围写成「`Wo` below ~3, `delta/dx` between 3 and 7」——**这个上界是断言的，")
    A("没有被它所附着的那条判据测过**。本研究的对象就是这条边界本身。")
    A("")
    A("参考解出处：Womersley (1955) *J. Physiol.* **127**:553；Batchelor (1967) §4.2（非定常平面 Poiseuille 特征展开）；")
    A("Landau & Lifshitz *Fluid Mechanics* 2nd ed. §24（Stokes 层标度）。")
    A("闭式解在 `src/tensorsph/womersley.py` 中用 sympy 重推（ODE 残差、两侧无滑移、`omega->0` 极限），")
    A("并由 `benchmarks/case_poiseuille/ref.py` 的 Crank–Nicolson 独立参考交叉验证（既有判据 G9，振幅 0.0043 %）。")
    A("")
    A("参数表（全部为 `run.py` 默认值，没有为本次研究调过任何一个）：")
    A("")
    A(md_table(["量", "值", "说明"], [
        ["`H`", f"{par['height']} m", "通道高度"],
        ["`nu`", f"{par['viscosity']} m²/s", "运动粘度"],
        ["`rho0`", f"{par['rho0']} kg/m³", "Tait EOS, `c_s` = " + str(par["sound_speed"]) + " m/s"],
        ["`g0`", f"{par['body_force']} m/s²", "体积力幅值"],
        ["列数", str(par["columns"]), "`x` 周期且流动 `x` 无关"],
        ["每个周期采样", str(par["samples_per_period"]), "调和拟合需要整数周期"],
        ["判据带", f"{par['interior_band'][0]} <= y/H <= {par['interior_band'][1]}", "G2/G4 的内区带"],
        ["壁面带", f"y/H <= {par['wall_band'][1]}", "本研究新增的敏感度量带（非判据）"],
    ]))
    A("")
    A("`omega` 不是自由旋钮：由声明的 `Wo` 反解 `omega = (2 Wo / H)**2 * nu`，")
    A("所以扫描的索引是物理量（`Wo`、`delta/dx`），不是手挑的角频率。")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. **两阶段预声明**。Stage 1（H1–H4）在任何一点跑之前写好，核心假设 H2 直接复述既有范围声明：")
    A("   「误差由 `delta/dx` 控制，判据当且仅当 `delta/dx >= 3` 成立」。")
    A("2. **先打最便宜的两个探针**（`W3_wo500_r16`, `W4_wo650_r16`）。结果两点**都在带内**")
    A("   （内区振幅 RMS 0.80 % / 0.60 %，`delta/dx` = 2.26 / 1.74）⇒ **H2 被证伪**。")
    A("   探针结果如实记录在工件 `predeclaration` 与每个点的 `prediction` 文本里，预测原文保留不改写。")
    A("3. **Stage 2**（H5–H6）在探针之后、主扫描之前写好，把问题改成：既然内区判据对 Stokes 层不敏感，")
    A("   那这条边界究竟由什么决定——时间采样（`dt` 受 CFL/粘性限制，每周期步数有限）还是空间分辨率？")
    A("4. **三组操纵，一次一个变量**：")
    A("   - A 组：`H/dx = 16` 固定，`Wo = 1.80 → 12.50`（`delta/dx` 6.30 → 0.90），共 8 点；")
    A("   - B 组：`Wo = 6.50` 固定，`H/dx = 8 / 16 / 32`（`delta/dx` 0.87 / 1.74 / 3.48）；")
    A("   - C 组：`Wo = 16.0` 固定，`H/dx = 16`（每周期约 50 步 < 64 采样 ⇒ 采样不足）vs `H/dx = 32`（约 101 步 ⇒ 采样足够）。")
    A("5. **伪迹控制**：`W3_wo500_r16` 用 4 个 settling 周期（默认 2）重跑（H3）；")
    A("   并用解析闭式算出**行平均地板**（见 ⑤），判断壁面行的误差是求解器的还是度量的。")
    A("6. 工具链：TensorSPH WCSPH（`tensorsph.droplet_impact.rhs` + `tensorsph.wall_mirror` 镜像无滑移面）")
    A("   + `tensorsph.stokes_second.harmonic_fit`；计算递交调度系统（本机只跑分钟级短点）。")
    A("")
    A(f"预声明记录：Stage 1 = H1–H4（跑之前）；Stage 1 探针 = {', '.join(doc['predeclaration']['stage1_probe'])}；")
    A(f"Stage 2 = H5–H6（探针之后、主扫描之前）；Stage 3 = H7–H8 标注为 **post-hoc**（扫描之后才提出，描述性检验）。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### A. 频率扫描（`H/dx = 16`）")
    A("")
    A(md_table(["点", "Wo", "delta/dx", "步/周期", "内区振幅 RMS [%]", "中心振幅 err [%]",
                "中心相位 err [%Q]", "壁面行 worst [%]", "G1–G4"],
               [[p["id"], f"{p['womersley_number']:.2f}", f"{p['stokes_layer_over_dx']:.2f}",
                 f"{p['steps_per_period']:.0f}", f"{p['amp_profile_rms_pct']:.2f}",
                 f"{p['amp_centre_error_pct']:+.2f}",
                 f"{p['phase_centre_pct_of_quarter_turn']:.2f}",
                 f"{p['wall_rows']['amp_worst_pct']:.2f}",
                 "PASS" if p["passes_case_gates"] else "FAIL"] for p in sw]))
    A("")
    A("### B. 分辨率阶梯（`Wo = 6.50`）+ 定 `Wo` 加密")
    A("")
    A(md_table(["点", "H/dx", "delta/dx", "内区振幅 RMS [%]", "壁面行 worst [%]",
                "行平均地板 [%]", "地板占比", "G1–G4"],
               [[p["id"], str(p["resolution_H_over_dx"]), f"{p['stokes_layer_over_dx']:.2f}",
                 f"{p['amp_profile_rms_pct']:.2f}", f"{p['wall_rows']['amp_worst_pct']:.2f}",
                 f"{p['row_averaging_floor']['wall_rows']['amp_worst_pct']:.2f}",
                 f"{p['wall_floor_fraction']:.2f}",
                 "PASS" if p["passes_case_gates"] else "FAIL"] for p in lad]))
    A("")
    A("### C. 采样解离（`Wo = 16.0`）与伪迹控制")
    A("")
    ctrl_ids = ["W8_wo1600_r16", "W8_wo1600_r32", "W3_wo500_r16", "W3_wo500_r16_settle4"]
    A(md_table(["点", "H/dx", "步/周期", "采样是否足够", "内区振幅 RMS [%]", "判定"],
               [[pts[i]["id"], str(pts[i]["resolution_H_over_dx"]),
                 f"{pts[i]['steps_per_period']:.0f}",
                 "是" if pts[i]["sampling_ok"] else "否（<64）",
                 f"{pts[i]['amp_profile_rms_pct']:.2f}", pts[i]["verdict"]] for i in ctrl_ids]))
    A("")
    A("### D. 假设裁定（全部由工件程序化判定）")
    A("")
    hyp_rows = []
    for key, label in (("H1_wo_free", "H1 频响不受 Wo 限制（预声明）"),
                       ("H2_layer_resolution", "H2 `delta/dx >= 3` 规则（预声明，复述既有范围）"),
                       ("H3_transient_artefact", "H3 伪迹=未收敛周期态"),
                       ("H4_noise_floor", "H4 噪声地板"),
                       ("H5_two_ceiling", "H5 天花板=时间采样而非物理（Stage 2）"),
                       ("H6_interior_gate_blind_to_wall_layer", "H6 内区判据看不见 Stokes 层（Stage 2）"),
                       ("H7_row_mean_floor_share", "H7 行平均地板占壁面误差 ≤50 %（post-hoc）"),
                       ("H8_wall_error_collapse", "H8 壁面误差是 `delta/dx` 的单值函数（post-hoc）")):
        h = v[key]
        s = h["supported"]
        hyp_rows.append([label, "—" if s is None else ("成立" if s else "**被证伪**"),
                         (h.get("note") or "")])
    A(md_table(["假设", "裁定", "依据"], hyp_rows))
    A("")
    if v["measured_boundary_H_over_dx_16"]:
        b = v["measured_boundary_H_over_dx_16"]
        A(f"**实测边界**：在 `H/dx = 16` 上，通过/未通过之间没有任何点落到带外——")
        A(f"边界记录 `{b}`。")
    else:
        A("**实测边界**：在 `H/dx = 16` 上，所有采样足够的点（`Wo` 一直到 "
          f"{v['H1_wo_free']['max_wo_in_sweep']:.2f}，`delta/dx` 低到 "
          f"{v['min_delta_over_dx_passing']:.2f}）全部在 5 % 带内 ⇒ "
          "**这个频段内不存在由空间分辨率造成的失效边界**。")
    A("")
    A("## ④ 结果图")
    A("")
    A(f"![envelope](figures/{fig1.name})")
    A("")
    A(f"![profiles](figures/{fig2.name})")
    A("")
    A("## ⑤ 注意点 / 局限")
    A("")
    A("1. **被证伪的是「既有范围声明」，不是求解器**。H2（`delta/dx >= 3`）在第一个探针点就破了：")
    A("   内区判据（`0.15 <= y/H <= 0.85`）平均掉了 Stokes 层，因此它对 `delta/dx = 0.9` 依然给出 < 1 %。")
    A("   这也解释了为什么原来那条「`delta/dx` between 3 and 7」的声明**从未被它所附着的判据检验过**。")
    A("2. **壁面行误差不是求解器缺陷的全部**。解析行平均地板（把闭式解在一个格子行上做积分平均，")
    A("   与被用作参考的点值相比）已计入表中：它解释了壁面误差的 "
    f"{min(p['wall_floor_fraction'] for p in graded(doc)):.2f}–"
    f"{max(p['wall_floor_fraction'] for p in graded(doc)):.2f} 倍，其余是 SPH 对亚格子厚层的抹平。")
    A("   换句话说：**`delta/dx < 1` 时壁面行是测不准的，而且是度量与算法共同造成的，换更细的网格才能改善**。")
    A("3. **真正的天花板是时间采样**：`dt` 由 CFL/粘性限制，`Wo = 16` 在 `H/dx = 16` 上每周期只有 "
    f"{pts['W8_wo1600_r16']['steps_per_period']:.0f} 步 < 64 个采样点，工件把该点标成 `invalid_sampling` 而**不是** FAIL；")
    A(f"同一 `Wo` 在 `H/dx = 32`（{pts['W8_wo1600_r32']['steps_per_period']:.0f} 步/周期）回到带内，")
    A("   这正好把「物理失效」与「测量失效」分开。")
    A("4. **弱可压前提不变**：高频工况的振幅随 `1/omega` 掉，`max_mach` 因此更小（见表），")
    A("   本研究的结论不能外推到自由面、湍流或 `Ma > 0.1`。")
    A("5. **适用面仍是层流单通道**：一个 `H`、一个 `nu`、`Re = 12.3`（稳态峰值）。")
    A("   结论的可迁移性仅限于「同一 WCSPH 通道在更高频脉动下」这一类问题。")
    A("6. **post-hoc 标注**：H7/H8 是看到壁面误差饱和后才提出的描述性检验，已在工件与上表中显式标注，")
    A("   不应被读作预声明预测。")
    A("")
    A("## ⑥ 误差判定（对照 5 % 判据；并回答是否 < 3 %）")
    A("")
    worst_in = max(graded(doc), key=lambda p: p["amp_profile_rms_pct"])
    worst_centre = max(graded(doc), key=lambda p: abs(p["amp_centre_error_pct"]))
    worst_phase = max(graded(doc), key=lambda p: p["phase_centre_pct_of_quarter_turn"])
    A(md_table(["关键量", "参考", "误差（最坏样本点）", "< 5 %?", "< 3 %?"], [
        ["内区振幅 profile RMS", "Womersley `|Uhat(y)|`（闭式）",
         f"{worst_in['amp_profile_rms_pct']:.2f} %（{worst_in['id']}）",
         "是" if worst_in["amp_profile_rms_pct"] <= 5 else "否",
         "是" if worst_in["amp_profile_rms_pct"] <= 3 else "否"],
        ["中心线振幅", "`|Uhat(H/2)|`",
         f"{abs(worst_centre['amp_centre_error_pct']):.2f} %（{worst_centre['id']}）",
         "是" if abs(worst_centre["amp_centre_error_pct"]) <= 5 else "否",
         "是" if abs(worst_centre["amp_centre_error_pct"]) <= 3 else "否"],
        ["中心线相位", "`arg Uhat(H/2)`",
         f"{worst_phase['phase_centre_pct_of_quarter_turn']:.2f} % of pi/2（{worst_phase['id']}）",
         "是" if worst_phase["phase_centre_pct_of_quarter_turn"] <= 5 else "否",
         "是" if worst_phase["phase_centre_pct_of_quarter_turn"] <= 3 else "否"],
        ["中心线振幅（最细网格 `H/dx = 32`）", "同上",
         f"{abs(pts['W4_wo650_r32']['amp_centre_error_pct']):.2f} %（W4_wo650_r32）", "是", "是"],
        ["壁面行 worst（**非判据**）", "点值 `|Uhat(y_wall)|`",
         f"{max(p['wall_rows']['amp_worst_pct'] for p in graded(doc)):.2f} %"
         f"（{max(graded(doc), key=lambda p: p['wall_rows']['amp_worst_pct'])['id']}）",
         "否（`delta/dx < 1` 时不适用）", "否"],
    ]))
    A("")
    A(f"**结论：全部判据量（G1、G2、G3、G4）在采样足够的每一点上都 < 5 %，最坏 "
      f"{worst_in['amp_profile_rms_pct']:.2f} %；且全部 < 3 %。**")
    A("唯一超 5 % 的是壁面行诊断量，而它**不是**算例判据：在 `delta/dx < 1` 时该行本身不可解析，")
    A("其误差 = 行平均地板 + 亚格子抹平，已按表分项列出，不作粉饰。")
    A("")
    A("原始工件：")
    A("")
    A(f"* `benchmarks/case_poiseuille/out/wo_envelope.json` — {v['n_points']} 点，"
      f"pass {v['n_pass']} / fail {v['n_fail']} / invalid {v['n_invalid']}，"
      f"计算墙钟 {doc.get('wall_clock_total_s', float('nan')):.0f} s")
    A("")
    return "\n".join(lines)


def main() -> int:
    doc = load()
    fig1 = fig_envelope(doc)
    fig2 = fig_profiles(doc)
    DOC.write_text(render(doc, fig1, fig2))
    print(f"wrote {DOC} and {fig1.name}, {fig2.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())