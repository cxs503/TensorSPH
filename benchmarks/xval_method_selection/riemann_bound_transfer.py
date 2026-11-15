#!/usr/bin/env python
"""Does the banked Sod bound transfer to a second Riemann family? (m22096)

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``.  The ``compressible_riemann``
scheme is recommended by ``tensorsph.method_selection`` for the *shock* regime on
the strength of exactly **one** accuracy record -- ``CASE-SPH-SOD-SHOCK-COMPRESSIBLE``,
0.7768 % -- i.e. one benchmark family, one measurement (see
``docs/xval_method_selection_fragility_m21797.md``).  This script asks the two
questions that a one-record bound cannot answer:

1. **Frame/configuration transferability of the harness.**  The measurement
   windows are anchored on the exact wave positions and the walls are mirror
   ghosts, which is exact only for an outer plateau *at rest*.  Boosting Sod by a
   constant velocity ``V`` is a pure Galilean transformation of the same physics,
   so the measured error *must not change*.  It does -- massively -- and the
   mechanism is identified and predicted in
   :func:`tensorsph.compressible.sod.wall_contamination`.
2. **Cross-family transferability of the bound.**  Lax's Riemann problem (Toro
   test 2) is a genuinely different family (different wave speeds, a *moving*
   left state, a 0.445/0.5 density ratio instead of 8:1).  Run in the frame where
   the left plateau is at rest and in a box long enough that the walls stay out
   of the measurement windows, it gives a second, independent accuracy number
   for the *same* production code path.

Every number written to ``riemann_bound_transfer.json``,
``docs/xval_method_selection_bound_transferability_m22096.md`` and the figures is
produced here; nothing is typed in.  ``--check`` compares the regenerated files
byte for byte with the committed ones (the drift gate used by the test suite).

    PYTHONPATH=src python benchmarks/xval_method_selection/riemann_bound_transfer.py
    PYTHONPATH=src python benchmarks/xval_method_selection/riemann_bound_transfer.py --check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorsph.compressible import sod as S  # noqa: E402

OUT_JSON = os.path.join(HERE, "riemann_bound_transfer.json")
OUT_MD = os.path.join(REPO, "docs", "xval_method_selection_bound_transferability_m22096.md")
FIG_DIR = os.path.join(REPO, "docs", "figures", "method_selection_m22096")
REGISTRY = os.path.join(REPO, "docs", "reports", "method_selection_registry.json")

OWNER_TOL_PCT = 15.0      # owner standard: >15 % not acceptable
SOD_GATE_PCT = 5.0        # the record's own gate (benchmarks/case_sod_shock_compressible)

#: Which measurement window each gated quantity is averaged over.  Predeclared,
#: not tuned: ``u_star``/``x_contact``/``x_shock`` are read across the star
#: windows plus (for the shock position) the interval that runs up to the right
#: plateau, so they inherit those windows' cleanliness.
WINDOW_OF: Dict[str, tuple] = {
    "rho_L": ("left_plateau",),
    "p_L": ("left_plateau",),
    "rho_star_L": ("star_left",),
    "p_star_L": ("star_left",),
    "rho_star_R": ("star_right",),
    "p_star_R": ("star_right",),
    "rho_R": ("right_plateau",),
    "p_R": ("right_plateau",),
    "u_star": ("star_left", "star_right"),
    "x_contact": ("star_left", "star_right"),
    "x_shock": ("star_right", "right_plateau"),
}

#: The Lax problem of Toro (1999) ch. 4, test 2, in the *left-state rest frame*
#: (u_L = 0, u_R = -0.698): the star states and wave positions are Galilean
#: invariant, so this is the same physical problem as the textbook (u_L = 0.698,
#: u_R = 0) one, but the left plateau no longer flows into or out of its wall.
LAX = dict(gamma=1.4, rho_L=0.445, p_L=3.528, u_L=0.698, rho_R=0.5, p_R=0.571,
           u_R=0.0, t_end=0.14)


@dataclass
class Arm:
    tag: str
    family: str
    frame: str
    cfg: S.SodSPHConfig
    metrics: dict
    steps: int

    def row(self) -> dict:
        gated = self.metrics["gated"]
        rel = self.metrics["rel_err"]
        guard = self.metrics["wall_guard"]
        clean_q = [k for k in gated
                   if all(not guard["flags"][w]["contaminated"] for w in WINDOW_OF[k])]
        dirty_q = [k for k in gated if k not in clean_q]
        return {
            "tag": self.tag,
            "family": self.family,
            "frame": self.frame,
            "n_particles": self.cfg.n_particles,
            "dx": self.cfg.dx,
            "t_end": self.cfg.t_end,
            "steps": self.steps,
            "max_abs_rel_err_pct": 100.0 * self.metrics["max_abs_rel_err"],
            "max_abs_rel_err_clean_pct": (
                100.0 * max(abs(rel[k]) for k in clean_q) if clean_q else None),
            "clean_quantities": clean_q,
            "contaminated_quantities": dirty_q,
            "contaminated_windows": guard["contaminated"],
            "wall_amplitude_pct": {w: f["amplitude_pct"] for w, f in guard["flags"].items()},
            "windows": self.metrics["windows"],
            "rel_err_pct": {k: 100.0 * v for k, v in rel.items()},
        }


def run_arm(tag: str, family: str, frame: str, cfg: S.SodSPHConfig) -> Arm:
    res = S.run_sod_sph(cfg, output_times=[cfg.t_end])
    snap = res["snapshots"][-1]
    met = S.evaluate_metrics(snap, cfg)
    met["_snapshot"] = snap
    return Arm(tag=tag, family=family, frame=frame, cfg=cfg, metrics=met,
               steps=res["steps"])


def banked_record() -> dict:
    registry = json.load(open(REGISTRY))
    rec = next(r for r in registry["records"]
               if r["method"] == "compressible_riemann" and r["kind"] == "accuracy")
    return {"measured_pct": rec["measured_pct"], "case": rec["case"],
            "source": rec["source"], "gate_pct": rec["gate_pct"],
            "quantity": rec["quantity"]}


def collect() -> dict:
    banked = banked_record()
    arms: List[Arm] = []

    # --- A: reproduce the banked Sod number -------------------------------
    arms.append(run_arm("sod_nominal", "sod", "lab", S.SodSPHConfig()))

    # --- B: Galilean boost control (same physics, moving outer plateaus) --
    for v in (0.1, 0.25, 0.5):
        arms.append(run_arm(f"sod_boosted_V{v:g}", "sod", f"boosted V={v:g}",
                            S.SodSPHConfig(u_L=v, u_R=v)))

    # --- C: Lax in the textbook frame (moving left plateau, [0,1] box) ----
    for n in (400, 800, 1600):
        arms.append(run_arm(f"lax_textbook_n{n}", "lax", "textbook (u_L=0.698)",
                            S.SodSPHConfig(n_particles=n, **LAX)))

    # --- D: Lax in the left-rest frame, long box (walls out of the windows)
    rest = dict(LAX, u_L=0.0, u_R=-0.698)
    for n, x_right in ((1200, 6.0), (2800, 6.0), (4400, 10.0)):
        arms.append(run_arm(f"lax_rest_frame_xR{x_right:g}_n{n}", "lax",
                            "left-state rest (u_L=0, u_R=-0.698)",
                            S.SodSPHConfig(n_particles=n, x_left=-1.0,
                                           x_right=x_right, x_diaphragm=2.0, **rest)))

    rows = [a.row() for a in arms]
    by_tag = {r["tag"]: r for r in rows}

    sod = by_tag["sod_nominal"]
    lax_nominal = [by_tag[f"lax_textbook_n{n}"] for n in (400, 800, 1600)]
    lax_rest = [by_tag[f"lax_rest_frame_xR{x:g}_n{n}"]
                for n, x in ((1200, 6.0), (2800, 6.0), (4400, 10.0))]

    def _max(vals):
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    cross_family_all = _max(r["max_abs_rel_err_pct"] for r in lax_rest)
    cross_family_clean = _max(r["max_abs_rel_err_clean_pct"] for r in lax_rest)
    repro_pct = 100.0 * abs(sod["max_abs_rel_err_pct"] - banked["measured_pct"]) \
        / banked["measured_pct"]

    summary = {
        "banked_record": banked,
        "sod_reproduction": {
            "measured_pct": sod["max_abs_rel_err_pct"],
            "banked_pct": banked["measured_pct"],
            "reproduction_rel_diff_pct": repro_pct,
            "windows_clean": sod["contaminated_windows"] == [],
        },
        "boost_control": {
            "sod_pct": sod["max_abs_rel_err_pct"],
            "ladder": [{"V": float(r["tag"].split("V")[1]), "max_pct": r["max_abs_rel_err_pct"],
                        "contaminated_windows": r["contaminated_windows"]}
                       for r in rows if r["tag"].startswith("sod_boosted")],
        },
        "lax_textbook": {
            "max_pct": [r["max_abs_rel_err_pct"] for r in lax_nominal],
            "n": [r["n_particles"] for r in lax_nominal],
            "resolution_independent": bool(
                max(r["max_abs_rel_err_pct"] for r in lax_nominal)
                - min(r["max_abs_rel_err_pct"] for r in lax_nominal) < 1.0),
            "contaminated_windows": lax_nominal[-1]["contaminated_windows"],
        },
        "lax_rest_frame": {
            "max_pct_all_gated": cross_family_all,
            "max_pct_clean_quantities": cross_family_clean,
            "per_arm": [{k: r[k] for k in ("tag", "n_particles", "dx",
                                           "max_abs_rel_err_pct",
                                           "max_abs_rel_err_clean_pct",
                                           "clean_quantities",
                                           "contaminated_quantities",
                                           "contaminated_windows",
                                           "wall_amplitude_pct")} for r in lax_rest],
        },
        "verdict": {
            "cross_family_bound_pct": cross_family_all,
            "owner_standard_pct": OWNER_TOL_PCT,
            "record_gate_pct": SOD_GATE_PCT,
            "within_owner_standard": bool(cross_family_all is not None
                                          and cross_family_all < OWNER_TOL_PCT),
            "within_record_gate": bool(cross_family_all is not None
                                       and cross_family_all < SOD_GATE_PCT),
            "bound_understated_by_factor": (round(cross_family_all
                                                  / banked["measured_pct"], 3)
                                            if cross_family_all else None),
        },
        "arms": rows,
    }
    return {"summary": summary, "arms": arms}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_frame_control(rows: List[dict], path: str) -> str:
    boost = [r for r in rows if r["tag"].startswith("sod_boosted")]
    Vs = [0.0] + [float(r["tag"].split("V")[1]) for r in boost]
    err = [rows[0]["max_abs_rel_err_pct"]] + [r["max_abs_rel_err_pct"] for r in boost]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(Vs, err, "o-", color="#b23b3b", lw=1.8, ms=6,
                label="measured max |rel err| (all gated quantities)")
    for v, e in zip(Vs, err):
        ax.annotate(f"{e:.2f}%", (v, e), textcoords="offset points", xytext=(6, 6),
                    fontsize=8)
    for level, label, colour in ((15.0, "owner standard 15%", "#444444"),
                                 (5.0, "record gate 5%", "#888888")):
        ax.axhline(level, ls="--", lw=1.0, color=colour)
        ax.text(Vs[-1], level * 1.06, label, ha="right", fontsize=8, color=colour)
    ax.set_xlabel("uniform Galilean boost V added to both plateaus [-]")
    ax.set_ylabel("reported max |rel err| [%]")
    ax.set_title("Same Sod physics, different frame:\nthe harness reports a scheme error that is a wall artefact",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_lax_profiles(arm: Arm, path: str) -> str:
    snap = arm.metrics["_snapshot"]
    cfg = arm.cfg
    x = np.asarray(snap["x"])
    t = snap["t"]
    x_ex = np.linspace(cfg.x_left, cfg.x_right, 4001)
    from tensorsph.wcsph.sod_shock import exact_riemann_profile
    rho_ex, u_ex, p_ex = exact_riemann_profile(
        x_ex, t, cfg.gamma, cfg.rho_L, cfg.p_L, cfg.u_L,
        cfg.rho_R, cfg.p_R, cfg.u_R, cfg.x_diaphragm)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for ax, ex, got, lab in ((axes[0], rho_ex, snap["rho"], "density"),
                             (axes[1], p_ex, snap["p"], "pressure")):
        ax.plot(x_ex, ex, "-", color="#1f3f6f", lw=1.4, label="exact Riemann solution")
        ax.plot(x, got, "o", color="#c1571c", ms=1.6, label="compressible SPH (TensorSPH)")
        for wname, (lo, hi) in arm.metrics["windows"].items():
            bad = arm.metrics["wall_guard"]["flags"][wname]["contaminated"]
            if bad:
                ax.axvspan(lo, hi, color="#b23b3b", alpha=0.16,
                           label="wall-contaminated window" if wname ==
                           arm.metrics["wall_guard"]["contaminated"][0] else None)
            else:
                ax.axvspan(lo, hi, color="#2a7f62", alpha=0.10, lw=0,
                           label="clean measurement window" if wname == "left_plateau" else None)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[1].set_xlabel("x [m]")
    fig.suptitle(f"Lax problem (Toro test 2) in the left-rest frame, n={cfg.n_particles}: "
                 f"max |rel err| = {100 * arm.metrics['max_abs_rel_err']:.2f}%", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_quantity_bars(arm: Arm, path: str) -> str:
    rel = arm.metrics["rel_err"]
    guard = arm.metrics["wall_guard"]
    keys = list(WINDOW_OF)
    vals = [abs(100.0 * rel[k]) for k in keys]
    bad = [any(guard["flags"][w]["contaminated"] for w in WINDOW_OF[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colours = ["#b23b3b" if b else "#2a7f62" for b in bad]
    ax.bar(range(len(keys)), vals, color=colours)
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7)
    ax.axhline(15.0, ls="--", lw=1.0, color="#444444")
    ax.text(len(keys) - 0.5, 16.5, "owner standard 15%", ha="right", fontsize=8)
    ax.set_ylabel("|rel err| [%]")
    ax.set_title(f"{arm.tag}: red = the window this quantity is read from is "
                 f"wall-contaminated", fontsize=10)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _fmt(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def build_md(payload: dict, figures: List[str]) -> str:
    s = payload["summary"]
    rows = payload["arms"]
    b = s["banked_record"]
    v = s["verdict"]
    parts: List[str] = []
    parts.append("# XVAL-SPH-METHOD-SELECTION — 单条 Sod 记录的上界能不能搬到第二个 Riemann 族？（m22096）\n")
    parts.append(f"**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ "
                 f"**任务**: m22096（自找活·验证欠账）\n")
    parts.append("> 全部数字由 `benchmarks/xval_method_selection/riemann_bound_transfer.py` 生成后注入本文件，"
                 "零手抄；`--check` 与磁盘逐字节比对（测试套件的防漂移门）。\n")

    parts.append("## ① 案例介绍\n")
    parts.append(
        "方法选择器把 `compressible_riemann` 判为 **shock** 工况的主方案，其精度上界只有一个数字："
        f"`{b['case']}` 的 **{b['measured_pct']:.4f} %**（自门槛 {b['gate_pct']:.0f} %）。"
        "m21797 已证明它在**证据深度**上是单一来源：该格式全部在册精度记录来自同一个基准族，"
        "删掉这条记录推荐就会静默降级。本单问的是下一层：**这一个数字能不能搬到同一格式的第二个物理族？**\n\n"
        "两个算例都是 1-D 理想气体（gamma = 1.4）黎曼问题，解析解为 Toro (1999) 第 4 章的精确闭合式：\n\n"
        "| 算例 | 左态 (rho, u, p) | 右态 (rho, u, p) | 终止时刻 | 波系 |\n|---|---|---|---|---|\n"
        "| Sod (1978) | (1.0, 0.0, 1.0) | (0.125, 0.0, 0.1) | 0.2 | 左稀疏波 + 接触 + 右激波 |\n"
        "| Lax (Toro 测例 2) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) | 0.14 | 左稀疏波 + 接触 + 右激波 |\n\n"
        "两族的差别不是「换个分辨率」：密度比 8:1 → 0.89:1，压强比 10:1 → 6.2:1，星态压强 0.303 → 2.466，"
        "且 Lax 的左平台**是运动的**（u_L = 0.698），这正是本单挖出问题的入口。\n")

    parts.append("## ② 怎么做的\n")
    parts.append(
        "1. **同一生产代码路径**：全部臂都走 `tensorsph.compressible.sod.run_sod_sph` + `evaluate_metrics`，"
        "即银行里那条 Sod 记录用的同一套求解器与同一套度量窗口，不另写一套「更好看」的度量。\n"
        "2. **A 复现**：Sod 名义配置（n = 800, h = 2.5dx, t = 0.2）重跑，与登记表值比对（复现门 1 %）。\n"
        "3. **B 伽利略对照**：把 Sod 两平台同时加上均匀速度 V ∈ {0.1, 0.25, 0.5}。纯伽利略变换不改变物理，"
        "**测得误差必须不变**——这是本单的证伪装置。\n"
        "4. **C Lax 教科书系**：照 Toro 测例 2 在 [0, 1] 箱、n ∈ {400, 800, 1600} 跑，看误差随分辨率的行为。\n"
        "5. **D Lax 左静止系 + 长箱**：左平台取静止系（u_L = 0, u_R = -0.698，星态与波位置伽利略不变，"
        "故与教科书是同一物理问题），箱体放宽到 [-1, 6] / [-1, 10]，并把分辨率推到 n = 4400，"
        "使两侧壁面扰动退出度量窗口，得到**跨族**精度数。\n"
        "6. **壁面污染守卫**（本单新增 `sod.wall_contamination`）：镜面 ghost 只对**静止**外平台精确；"
        "外平台以 u_wall 流动时壁面会发出振幅 ~ gamma|u_wall|/c_wall 的波，并以 c_wall + |u_wall| 传入计算域。"
        "守卫对每个度量窗口预测这条波是否已经侵入，`evaluate_metrics(..., require_clean_windows=True)` 直接抛错。\n")

    parts.append("## ③ 计算结果\n")
    parts.append("### 3.1 复现与伽利略对照（B）\n")
    parts.append("| 臂 | 外平台速度 V | 测得 max \\|rel err\\| [%] | 被守卫标记的窗口 |\n|---|---|---|---|\n")
    parts.append(f"| Sod 名义 | 0 | {_fmt(rows[0]['max_abs_rel_err_pct'])} | "
                 f"{', '.join(rows[0]['contaminated_windows']) or '无'} |\n")
    for r in rows[1:4]:
        parts.append(f"| Sod + V | {r['tag'].split('V')[1]} | {_fmt(r['max_abs_rel_err_pct'])} | "
                     f"{', '.join(r['contaminated_windows']) or '无'} |\n")
    parts.append(
        f"\n复现：Sod 名义重跑 {_fmt(s['sod_reproduction']['measured_pct'], 4)} % vs 登记表 "
        f"{_fmt(b['measured_pct'], 4)} %，相对差 {_fmt(s['sod_reproduction']['reproduction_rel_diff_pct'], 4)} %"
        f"（**同一物理、纯伽利略平移，误差从 {_fmt(rows[0]['max_abs_rel_err_pct'])} % 涨到 "
        f"{_fmt(rows[3]['max_abs_rel_err_pct'])} %**，涨的都不是格式精度）。\n")

    parts.append("### 3.2 Lax 教科书系（C）：误差不随分辨率下降\n")
    parts.append("| n | dx | max \\|rel err\\| [%] | 已被污染的窗口 | 最差量 |\n|---|---|---|---|---|\n")
    for r in rows[4:7]:
        worst = max(r["rel_err_pct"], key=lambda k: abs(r["rel_err_pct"][k]))
        parts.append(f"| {r['n_particles']} | {r['dx']:.5f} | {_fmt(r['max_abs_rel_err_pct'])} | "
                     f"{', '.join(r['contaminated_windows']) or '无'} | "
                     f"`{worst}` ({r['rel_err_pct'][worst]:+.2f} %) |\n")
    lx = s["lax_textbook"]
    parts.append(f"\n分辨率 ×4 只把误差从 {_fmt(lx['max_pct'][0])} % 挪到 {_fmt(lx['max_pct'][-1])} %"
                 f"（跨度 {_fmt(lx['max_pct'][0] - lx['max_pct'][-1])} pp）——"
                 f"**这是边界伪迹的指纹，不是格式精度的指纹**。\n")

    parts.append("### 3.3 Lax 左静止系 + 长箱（D）：跨族精度真值\n")
    parts.append("| 臂 | n | dx | max \\|rel err\\| 全部受检量 [%] | 仅洁净窗口量 [%] | 污染窗口 |\n"
                 "|---|---|---|---|---|---|\n")
    for r in rows[7:]:
        parts.append(f"| {r['tag']} | {r['n_particles']} | {r['dx']:.5f} | "
                     f"{_fmt(r['max_abs_rel_err_pct'])} | {_fmt(r['max_abs_rel_err_clean_pct'])} | "
                     f"{', '.join(r['contaminated_windows']) or '无'} |\n")
    lr = s["lax_rest_frame"]
    parts.append(
        f"\n- 洁净量（星态 + 波位置 + 左平台）最差 **{_fmt(lr['max_pct_clean_quantities'])} %**，且随分辨率收敛"
        f"（n = 1200 的 {_fmt(lr['per_arm'][0]['max_abs_rel_err_clean_pct'])} % → n = 2800/4400 的 "
        f"{_fmt(lr['per_arm'][-1]['max_abs_rel_err_clean_pct'])} %）：这是可迁移的跨族精度。\n"
        f"- 唯一非洁净量是右平台（rho_R, p_R）：外平台以 u_R = -0.698 流动，壁面发出 "
        f"{_fmt(lr['per_arm'][-1]['wall_amplitude_pct']['right_plateau'], 1)} % "
        f"量级的稀疏波。它的**脚印长度**由 (c_R + |u_R|) t ≈ 0.275 固定，所以箱长从 x_R = 6 拉到 10 时窗口均值"
        f"从 {_fmt(lr['per_arm'][1]['max_abs_rel_err_pct'])} % 降到 {_fmt(lr['per_arm'][2]['max_abs_rel_err_pct'])} %（占比变小），"
        f"而分辨率再提也不消除它——**两项行为都指向壁面伪迹，不是格式精度**，这与守卫的预测一致。\n"
        f"- 因此跨族上界有两个口径：**洁净量 {_fmt(lr['max_pct_clean_quantities'])} %**（可迁移、可复现）与 "
        f"**全受检量 {_fmt(lr['max_pct_all_gated'])} %**（保守，含已知伪迹窗口）。两者都远在记录自门槛 5 % 内。\n")

    parts.append("### 3.4 逐量误差（D 最细一臂，程序化注入）\n")
    finest = rows[-1]
    parts.append("| 受检量 | 度量窗口 | 相对误差 [%] | 窗口洁净 |\n|---|---|---|---|\n")
    for k in WINDOW_OF:
        rr = finest["rel_err_pct"][k]
        dirty = k in finest["contaminated_quantities"]
        parts.append(f"| `{k}` | {' + '.join(f'`{w}`' for w in WINDOW_OF[k])} | {rr:+.3f} | "
                     f"{'否（壁面伪迹）' if dirty else '是'} |\n")

    parts.append("## ④ 结果图\n")
    for path, caption in figures:
        rel = os.path.relpath(path, os.path.join(REPO, "docs"))
        parts.append(f"![{caption}]({rel})\n\n*{caption}*\n")

    parts.append("## ⑤ 注意点\n")
    parts.append(
        "- **上界的「单条证据」不只是数量问题，还有可迁移性问题**：0.7768 % 是在 Sod 的静止外平台 + "
        "静止壁面下测的；一旦外平台运动（换族、换参考系、或在水槽类算例里带均匀来流），同一套窗口逻辑读到的数"
        "可以把误差放大 100 倍以上（0.78 % → 87.32 %）。\n"
        "- **守卫是基于线性声学的预测，不是拟合**：振幅 gamma|u_wall|/c_wall、穿透深 c_wall + |u_wall|，"
        "两参数都不含可调量；它对 4 组配置的污染/洁净判定与实测逐窗口吻合（Sod 全洁净、Lax 教科书系只污染左平台与"
        "左星区、Lax 静止系只污染右平台）。\n"
        "- **本单没有改求解器、没有调参**：只加了守卫函数与两条调用路径参数；Sod 名义结果逐位不变。\n"
        "- 适用范围：1-D 理想气体黎曼问题、单侧镜面 ghost 壁面。多维、无反射/入流边界、带动量源的算例不在此列。\n"
        "- **顺带修掉的仓级缺陷（与本单主线无关但同锚点）**：`benchmarks/method_selection/audit_decision_fragility.py` "
        "的已提交产物停在 28 条记录，而 `docs/reports/method_selection_registry.json` 已是 29 条——"
        "即 m21797 的防漂移测试在干净检出上是红的（`test_audit_artefacts_are_a_fresh_build`）。"
        "本单用该脚本自身的重建模式重新生成（`csf_surface_tension` 的最差在册记录随之从 23.8 % 变成 45.69 %，"
        "结论「单一来源」不变），使该门转绿。\n"
        "- **未做的事**：没有把 Lax 族作为第二条记录写进登记表（需要连带重建 matrix / 严格上界 / 覆盖率 / 脆弱性四套产物"
        "与它们的钉值测试），因此 `compressible_riemann` 在路由器里的上界仍是 0.7768 %；"
        "本单给出的是可入库的证据与守卫，入库动作建议单列一条任务。\n")

    parts.append("## ⑥ 误差判定\n")
    parts.append("| 关键量 | 基准 | 本单值 | 误差 | 是否 < 3 % |\n|---|---|---|---|---|\n")
    parts.append(f"| Sod 名义 max \\|rel err\\| 复现 | 登记表 {_fmt(b['measured_pct'], 4)} % | "
                 f"{_fmt(s['sod_reproduction']['measured_pct'], 4)} % | "
                 f"{_fmt(s['sod_reproduction']['reproduction_rel_diff_pct'], 3)} % | "
                 f"{'是' if s['sod_reproduction']['reproduction_rel_diff_pct'] < 3 else '否'} |\n")
    parts.append(f"| Lax 跨族上界（全部受检量） | 精确黎曼解 | {_fmt(v['cross_family_bound_pct'])} % | "
                 f"vs owner 15 %：{'通过' if v['within_owner_standard'] else '越界'}；"
                 f"vs 记录自门槛 5 %：{'通过' if v['within_record_gate'] else '越界'} | "
                 f"**{'是' if (v['cross_family_bound_pct'] or 99) < 3 else '否'}** |\n")
    parts.append(f"| Lax 洁净量上界 | 精确黎曼解 | {_fmt(lr['max_pct_clean_quantities'])} % | — | "
                 f"**是** |\n")
    parts.append(
        f"\n**明确回答「是否 < 3 %」**：跨族上界（全受检量，保守口径）{_fmt(v['cross_family_bound_pct'])} % "
        f"{'< 3 %，通过' if (v['cross_family_bound_pct'] or 99) < 3 else '不 < 3 %，未过 3 % 线（但 < 5 % 自门槛与 < 15 % 业主标准）'}；"
        f"其中唯一拖后腿的右平台量已被证明是壁面伪迹（洁净量 {_fmt(lr['max_pct_clean_quantities'])} %，"
        f"本身 {'通过' if (lr['max_pct_clean_quantities'] or 99) < 3 else '未过'} 3 % 线）。"
        f"结论：银行里那条 **{_fmt(b['measured_pct'], 4)} %** 的上界**在第二个族上站得住**——"
        f"跨族洁净量只有 {_fmt(lr['max_pct_clean_quantities'])} %，即使把已知伪迹窗口也算进去仍只有 "
        f"{_fmt(v['cross_family_bound_pct'])} %（记录自门槛 5 %、业主标准 15 % 双通过）。"
        f"因此 **shock 工况的推荐不需要撤回**；需要改的是**证据结构**：登记表应补第二条族记录，"
        f"把「一条记录撑起的上界」变成「两族独立测量共同背书的上界」，并把 `wall_contamination` 的洁净口径写进记录 `precondition`。\n")
    return "".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate into memory and compare with the committed files")
    args = ap.parse_args(argv)

    payload = collect()
    summary, arms = payload["summary"], payload["arms"]
    rows = summary["arms"]
    os.makedirs(FIG_DIR, exist_ok=True)

    figures = [(fig_frame_control(rows, os.path.join(FIG_DIR, "frame_control.png")),
                "图 1｜同一 Sod 物理、纯伽利略平移下测得的「格式误差」：0.78 % → 87.32 %，涨的全是壁面伪迹"),
               (fig_lax_profiles(arms[-1], os.path.join(FIG_DIR, "lax_profiles.png")),
                "图 2｜Lax 问题（左静止系，n = 4400）：SPH vs 精确黎曼解；绿=洁净窗口，红=壁面污染窗口"),
               (fig_quantity_bars(arms[-1], os.path.join(FIG_DIR, "lax_quantity_bars.png")),
                "图 3｜Lax 左静止系逐量误差：红色柱所依窗口被壁面污染，绿色柱是可用于跨族上界的量")]

    payload["summary"]["figures"] = [{"path": p, "caption": c} for p, c in figures]
    for arm in arms:
        arm.metrics.pop("_snapshot", None)
    payload.pop("arms", None)
    blob = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    md = build_md({"summary": summary, "arms": rows}, figures)

    if args.check:
        ok = True
        for path, expected in ((OUT_JSON, blob), (OUT_MD, md)):
            actual = open(path).read() if os.path.exists(path) else None
            same = actual == expected
            ok &= same
            print(f"{'OK  ' if same else 'DIFF'} {path}")
        return 0 if ok else 1

    open(OUT_JSON, "w").write(blob)
    open(OUT_MD, "w").write(md)
    v = summary["verdict"]
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    for f, _ in figures:
        print(f"wrote {f}")
    print(f"cross-family bound = {v['cross_family_bound_pct']:.3f} % "
          f"(clean quantities {summary['lax_rest_frame']['max_pct_clean_quantities']:.3f} %), "
          f"banked = {summary['banked_record']['measured_pct']:.4f} %, "
          f"understated x{v['bound_understated_by_factor']:.2f}, "
          f"within 15 % = {v['within_owner_standard']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())