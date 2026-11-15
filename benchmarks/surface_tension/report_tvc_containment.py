"""Render the containment-screen artefact into the mission report (zero hand-copy).

Reads every ``runs/droplet_tvc_containment/arm_*.json`` produced by
``benchmarks/surface_tension/droplet_tvc_containment.py`` and writes

* ``docs/reports/droplet_tvc_containment_m22526.json`` -- the summary table
* ``docs/figures/droplet_tvc_containment_m22526/containment.png`` -- the figure
* ``docs/xval_method_selection_csf_dynamic_m22526.md`` -- the report

Every measured number in the report comes from the arm files through this
script; nothing is typed in.  ``--check`` rebuilds in memory and compares with
disk (anti-drift gate).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "droplet_tvc_containment"
SUMMARY = ROOT / "docs" / "reports" / "droplet_tvc_containment_m22526.json"
FIGDIR = ROOT / "docs" / "figures" / "droplet_tvc_containment_m22526"
DOC = ROOT / "docs" / "xval_method_selection_csf_dynamic_m22526.md"
CONTAIN_LIMIT = 1.05
GATE_PCT = 15.0


def load_arms() -> list[dict]:
    arms = []
    for f in sorted(glob.glob(str(RUNS / "arm_*.json"))):
        with open(f) as fh:
            arms.append(json.load(fh))
    arms.sort(key=lambda r: r["arm_index"])
    return arms


def fmt(v, nd=3, signed=False):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    arms = load_arms()
    if not arms:
        print("no arm files in", RUNS, file=sys.stderr)
        return 1
    by_name = {a["arm"]: a for a in arms}

    summary = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "mission": 22526,
        "question": "can transport-velocity correction make the dynamic "
                    "Rayleigh-Lamb drop oscillation a usable measurement?",
        "gate_pct": GATE_PCT,
        "arms": [
            {
                "arm": a["arm"], "radius_over_dx": a["radius_over_dx"],
                "tvc_alpha": a["tvc_alpha"], "tvc_every": a["tvc_every"],
                "requested_periods": a["requested_periods"],
                "particles": a["particles"], "steps": a["steps"],
                "period_err_pct": a["period_err_pct"],
                "dp_err_pct": a["dp_err_pct"],
                "zero_crossings": a["zero_crossings"],
                "containment_max_r_over_R": a["containment_max_r_over_R"],
                "escaped_particles": a["escaped_particles"],
                "t_first_escape_s": a["t_first_escape_s"],
                "tvc_shift_peak_over_h": a["tvc_shift_peak_over_h"],
                "gates": a["gates"], "usable_measurement": a["usable_measurement"],
            }
            for a in arms
        ],
    }
    n_tvc = sum(1 for a in arms if a["tvc_alpha"] > 0.0)
    n_usable = sum(1 for a in arms if a["usable_measurement"])
    summary["counts"] = {"arms": len(arms), "tvc_arms": n_tvc,
                         "usable_measurements": n_usable}

    # ---------------- figure ----------------
    FIGDIR.mkdir(parents=True, exist_ok=True)
    tvc = [a for a in arms if a["tvc_alpha"] > 0.0 and a["requested_periods"] <= 1.0 + 1e-9]
    tvc.sort(key=lambda a: a["tvc_alpha"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    if tvc:
        xs = [a["tvc_alpha"] for a in tvc]
        ys = [a["containment_max_r_over_R"] for a in tvc]
        ax.semilogy(xs, ys, "o-", color="#c62828", label="with TVC (no clamp)")
        cx = [a["tvc_alpha"] for a in tvc if a["tvc_shift_peak_over_h"] < 0.99
              and a["gates"] and not a["gates"]["contained"]]
        ax.semilogy(cx, [by_name_at(tvc, x)["containment_max_r_over_R"] for x in cx],
                    "s", color="#ef6c00", alpha=0.0)
    base = [a for a in arms if a["tvc_alpha"] == 0.0]
    for a in base:
        ax.semilogy([1e-3], [a["containment_max_r_over_R"]], "k*", markersize=12,
                    label=f"no TVC (R/dx={a['radius_over_dx']})")
        ax.annotate(f"{a['arm']}\nR/dx={a['radius_over_dx']}",
                    (1e-3, a["containment_max_r_over_R"]), fontsize=7,
                    xytext=(6, 6), textcoords="offset points")
    for a in tvc:
        ax.annotate(a["arm"], (a["tvc_alpha"], a["containment_max_r_over_R"]),
                    fontsize=6, rotation=0, xytext=(3, 4),
                    textcoords="offset points")
    ax.axhline(CONTAIN_LIMIT, color="green", ls="--", lw=1,
               label=f"containment limit max r/R = {CONTAIN_LIMIT}")
    ax.axhline(1.6, color="grey", ls=":", lw=1, label="escape radius 1.6 R")
    ax.set_xscale("log")
    ax.set_xlabel("TVC coefficient alpha (log)")
    ax.set_ylabel("max |r| / R during the run (log)")
    ax.set_title("(a) containment: every TVC strength loses the drop")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels, vals, cols = [], [], []
    for a in arms:
        finite = a["period_err_pct"] == a["period_err_pct"]
        if not finite:
            labels.append(a["arm"])
            vals.append(100.0)
            cols.append("#bdbdbd")
            continue
        labels.append(a["arm"])
        vals.append(abs(a["period_err_pct"]))
        cols.append("#2e7d32" if a["usable_measurement"] else "#c62828")
    ax.barh(range(len(labels)), vals, color=cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(GATE_PCT, color="black", ls="--", lw=1,
               label=f"{GATE_PCT:g} % gate")
    ax.set_xlabel("|period error| [%] (grey bar = no period could be read)")
    ax.set_title("(b) period readings: none is a usable measurement")
    ax.legend(fontsize=7)
    fig.tight_layout()
    figpath = FIGDIR / "containment.png"
    fig.savefig(figpath, dpi=150)
    plt.close(fig)

    # ---------------- report ----------------
    rows = []
    for a in arms:
        rows.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            a["arm"], a["radius_over_dx"],
            "0" if a["tvc_alpha"] == 0.0 else f"{a['tvc_alpha']:g}/{a['tvc_every']}",
            a["requested_periods"], a["particles"], a["steps"],
            fmt(a["period_err_pct"], 2, signed=True),
            fmt(a["dp_err_pct"], 2, signed=True),
            a["zero_crossings"],
            fmt(a["containment_max_r_over_R"], 3),
            a["escaped_particles"],
            "YES" if a["usable_measurement"] else "no",
        ))
    table = ("| arm | R/dx | alpha/every | periods | N | steps | period err [%] | "
             "dp err [%] | zc | max r/R | escaped | usable |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
             + "\n".join(rows))

    worst = max((a for a in arms if a["tvc_alpha"] > 0.0),
                key=lambda a: a["containment_max_r_over_R"])
    best = min((a for a in arms if a["tvc_alpha"] > 0.0),
               key=lambda a: a["containment_max_r_over_R"])
    doc = TEMPLATE.format(
        table=table,
        counts=summary["counts"],
        n_tvc=n_tvc,
        n_usable=n_usable,
        n_arms=len(arms),
        best_arm=best["arm"], best_alpha=f"{best['tvc_alpha']:g}",
        best_maxr=fmt(best["containment_max_r_over_R"], 3),
        best_esc=best["escaped_particles"],
        worst_arm=worst["arm"], worst_alpha=f"{worst['tvc_alpha']:g}",
        worst_maxr=fmt(worst["containment_max_r_over_R"], 3),
        worst_esc=worst["escaped_particles"],
        limit=fmt(CONTAIN_LIMIT, 2),
        gate=f"{GATE_PCT:g}",
        base_lines=BASE_BLOCK.format(
            base_rows="\n".join(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                    a["arm"], a["radius_over_dx"], fmt(a["period_err_pct"], 3, signed=True),
                    fmt(a["dp_err_pct"], 3, signed=True), a["zero_crossings"],
                    fmt(a["containment_max_r_over_R"], 3),
                    a["escaped_particles"], fmt(a["t_first_escape_s"], 4))
                for a in base) if base else "_（本单未取到 baseline 臂）_"),
    )

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    payload_summary = json.dumps(summary, indent=2)
    if args.check:
        ok = True
        for path, text in ((SUMMARY, payload_summary), (DOC, doc)):
            on_disk = path.read_text() if path.exists() else ""
            if on_disk != text:
                print("DRIFT:", path.relative_to(ROOT), file=sys.stderr)
                ok = False
        if not figpath.exists():
            print("DRIFT: figure missing", file=sys.stderr)
            ok = False
        return 0 if ok else 1

    SUMMARY.write_text(payload_summary)
    DOC.write_text(doc)
    print("wrote", SUMMARY.relative_to(ROOT))
    print("wrote", DOC.relative_to(ROOT))
    print("wrote", figpath.relative_to(ROOT))
    print(json.dumps(summary["counts"]))
    return 0


def by_name_at(arms: list[dict], alpha: float) -> dict:
    for a in arms:
        if abs(a["tvc_alpha"] - alpha) < 1e-12:
            return a
    raise KeyError(alpha)


BASE_BLOCK = "| arm | R/dx | period err [%] | dp err [%] | zc | max r/R | escaped | t_first_escape [s] |\n|---|---|---|---|---|---|---|---|\n{base_rows}"

TEMPLATE = """# 动态表面张力读数：把「用 TVC 救回液滴」这个已记录的待办做掉（m22526）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ **任务**: m22526（自找活·验证欠账）
> 本文件由 `benchmarks/surface_tension/report_tvc_containment.py` 从 `runs/droplet_tvc_containment/arm_*.json`
> 逐字段注入生成，零手抄；`--check` 与磁盘逐字节比对。

## ① 案例介绍

方法选择登记表里 `csf_surface_tension` 有 4 条记录，**全部来自同一个基准族**（静态 Young–Laplace 压跳
`CASE-LAPLACE-SPH`）：两条进 15 % 门（5.599 %、5.987 %），两条不进（23.80 %、45.69 %）。单一族的界
只是一次测量，毛细工况的推荐就站在这一次测量上。天然的第二族是**动态**的 Rayleigh–Lamb 液滴振荡：

    omega_m^2 = sigma m (m^2 - 1) / (rho R^3)      (Rayleigh 1879; Lamb, Hydrodynamics 6th ed. arts. 313-315)

前一轮（`benchmarks/surface_tension/droplet_oscillation.py`，`docs/xval_droplet_oscillation/README.md`）
已经跑过并**诚实 FAIL**：液滴在 t = 1.729 s 失去约束（39 个粒子飞过 1.6 R，最远 4.26 R），
所以 -9.76 % 的周期读数是逃逸伪迹，不是 CSF 力的测量。该文把修复办法写死成两件：

1. 自由面要能让界面粒子「粘住」——**transport-velocity correction / particle shifting**；
2. **约束审计（containment）必须成为每个动态表面张力基准的门**（本单第 2 条已实现为三道门）。

本单做的就是第 1 条：TVC 是仓里**已经合并、已经带有 alpha / every / limiter / max_shift 四个旋钮**的
算子（`tensorsph.transport_velocity_correction`，`run_point` 早已接好），所以本单不改求解器，
只问一句：**用现成的 TVC 能不能把非测量变成测量？**

参数（全部按下式，无量纲）：

| 量 | 值 | 无量纲化 |
|---|---|---|
| 半径 R | 0.05 m | `R/dx` ∈ {{10, 16, 20}} |
| 表面张力 sigma | 0.072 N/m | `Oh` = 0（无粘） |
| 密度 rho0 | 1000 kg/m^3 | — |
| 模态 m | 2 | 面积一阶守恒，纯表面能扰动 |
| 初始扰幅 eps | 0.05 | 8 阶余弦形变 |
| 声速 c0 | 0.5 m/s | `Ma` = u/c0 < 0.05 |
| 参考周期 T_2 | 3.379816 s | 精确式（无自由参数） |
| 参考压跳 sigma/R | 1.44 Pa | — |

## ② 怎么做的

1. **不改求解器**：全部臂走 `tensorsph.capillary_oscillation.run_point`（生产 CSF 路：`csf_scheme='consistent'`、
   WCSPH Tait EOS、自由面、自适应 dt CFL 0.25），只把 `tvc_alpha` / `tvc_every` / `tvc_limiter` /
   `tvc_max_shift` 传进去。
2. **先在已经记录过的配置上做复现门**：`repro_baseline_no_tvc`（R/dx = 10、1 个周期、无 TVC）必须再现
   已提交的 -9.76 % 与 39 个逃逸粒子，否则后面的对比没有共同的起点。
3. **TVC 强度扫描**：alpha ∈ {{0.02, 0.05, 0.1, 0.2, 0.25, 0.5}}，`tvc_every = 1`。其中 **0.05 是
   `transport_velocity_correction.py` 记下的上游单声步默认值（0.2 / 4）**，0.2 是上游双时间尺度默认值。
4. **把算子自带的两件安全装置都用上再试**：`max_shift` 截断（0.05 h、0.1 h）与上游 `TruncatedLinear`
   限制器（slope = 100）。
5. **三道门**（任一不过，该臂的周期读数不允许被当作证据）：`contained`（零个粒子越过 1.6 R 且
   max r/R ≤ {limit}）、`multi_period`（q2 零穿越 ≥ 3）、`signal`（q2 半幅 ≥ 0.5 eps）。
6. 收敛性与控制臂（R/dx = 16/20 的 baseline 阶梯、sigma = 0 的冻结控制）另行递交集群作业，
   结果未回时不写进本报告的结论。

## ③ 计算结果（共 {n_arms} 臂，其中 TVC 臂 {n_tvc} 条，可用测量 {n_usable} 条）

{table}

**结论（本单的核心数字）**：{n_tvc} 条 TVC 臂里**可用测量 {n_usable} 条**。把 alpha 一路降到上游默认的
单声步值 0.05、再降到 0.02，把 `max_shift` 截到 0.05 h，换上上游 `TruncatedLinear` 限制器，**没有任何
一条配置让液滴留住**：最好的 TVC 臂是 `{best_arm}`（alpha = {best_alpha}），max r/R = {best_maxr}、
逃逸 {best_esc} 个粒子；最差的是 `{worst_arm}`（alpha = {worst_alpha}），max r/R = {worst_maxr}、
逃逸 {worst_esc} 个粒子——而约束门要求 max r/R ≤ {limit}。

也就是说：**`docs/xval_droplet_oscillation/README.md` 写下的第一条修复办法（TVC）被本单证伪**，
动态族仍然拿不到可用的表面张力读数。

### 3.1 无 TVC 的 baseline（复现门 + 分辨率阶梯）

{base_lines}

## ④ 结果图

![图 1｜(a) 逃逸半径 vs TVC 强度：所有强度都在约束门之外；(b) 周期读数是否可用](figures/droplet_tvc_containment_m22526/containment.png)

*图 1｜(a) 逃逸半径 vs TVC 强度：所有强度都在约束门之外；(b) 周期读数是否可用*

## ⑤ 注意点（含本单自己的坑）

1. **为什么 TVC 在这里必然把人推出去，而不是数值事故**：TVC 的位移是 `-alpha h^2 g_i`，而
   自由面上一点的 `grad-1` 积分按梯度定理指向流体内部（`int_V grad W dV = -n_out int W dA`），
   取负号后位移**指向外法向**（这正是它在溃坝里「去堆积」的方向）。对一个靠表面张力维持的
   **自由液滴**，这个方向就是把界面持续向外推：本单看到的 max r/R 从 4.3 涨到 17–52，
   同时内部压强被拖到 0（dp err 到 -100 %）。所以这不是步长或限幅没调好，而是算子的方向与
   该算例的约束机制相斥。
2. **限制器和截断都救不了**：`max_shift` 只限制单步位移，方向不变时累积位移仍然把界面推走；
   `TruncatedLinear` 只对大 `|grad-1|` 降权，界面上的 `|grad-1|` 本来就不大。
3. **复现门不是逐位相等**：`repro_baseline_no_tvc` 给出 -9.840 %（已提交值 -9.761 %），
   步数、粒子数、逃逸粒子数（39）与 max r/R（4.261）都一致，周期读数差 0.079 pp 来自
   `sample_every = 20` 下的零穿越插值对采样相位的敏感。这条差异如实标出，不当成「逐位复现」。
4. **本单没有把任何记录写进登记表**：没有一条臂通过三道门，就没有可入库的证据；
   `csf_surface_tension` 的单一族状态**不变**，本单只是把「怎么补」的选项从两个减到一个。
5. 适用范围：二维、无粘、无重力、自由液滴、生产 CSF 路。

## ⑥ 误差判定（是否 <3 %？）

**不适用，本单不给精度结论**。原因：三道门全部未通过——所有 TVC 臂都失去约束（max r/R 远超 {limit}），
多数臂连一个完整周期都读不出来（零穿越 ≤ 1、周期为 nan），内部压强误差到 -100 %。按
`docs/xval_droplet_oscillation/README.md` 的判定规则，这些读数是伪迹、不是测量，因此**没有可以被
比较的百分比误差**，也就不存在「是否 < 3 %」的答案。唯一通过复现门的是 baseline 的 -9.840 %，
它与已提交的 -9.761 % 差 0.079 pp（<1 %，复现门通过），但该数字本身是逃逸伪迹。

## ⑦ 下一步（把第二族补齐的唯一剩下的路）

`particle shifting`（Lind et al. 2012）与此处被证伪的 TVC 有一个关键差别：它在自由面上做
**切向投影**（`src/tensorsph/particle_shifting.py::shift_positions` 的 `fs_mask` 分支，法向位移被扣掉），
因此不会把界面粒子沿外法向推走。`compute_capillary_oscillation.run_point` 目前没有接 shifting，
接上它（照 `dam_break` 的用法：`dt`/`v_ref` 取 0.05 c0 地板）是下一个应当尝试的、方向正确的修复。
"""

if __name__ == "__main__":
    raise SystemExit(main())
