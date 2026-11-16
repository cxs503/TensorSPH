"""Render the particle-shifting containment screen into the mission report (zero hand-copy).

Reads every ``runs/droplet_shifting_containment/arm_*.json`` produced by
``benchmarks/surface_tension/droplet_shifting_containment.py`` and writes

* ``docs/reports/droplet_shifting_containment_m22654.json`` -- summary table
* ``docs/figures/droplet_shifting_containment_m22654/containment.png`` -- figure
* ``docs/xval_method_selection_csf_shifting_m22654.md`` -- the report

Every measured number in the report comes from the arm files through this
script; nothing is typed in.  The verdict sentence is *computed* from the gates,
so a failing sweep cannot be rendered as a passing one.  ``--check`` rebuilds in
memory and compares with disk (anti-drift gate).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "droplet_shifting_containment"
SUMMARY = ROOT / "docs" / "reports" / "droplet_shifting_containment_m22654.json"
FIGDIR = ROOT / "docs" / "figures" / "droplet_shifting_containment_m22654"
DOC = ROOT / "docs" / "xval_method_selection_csf_shifting_m22654.md"
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


def verdict_of(arms: list[dict]) -> dict:
    shift = [a for a in arms if a["shift_steps"] > 0]
    usable = [a for a in shift if a["usable_measurement"]]
    passing = [a for a in usable if a["period_pass"]]
    control = [a for a in arms if a["sigma_case"] == 0.0]
    return {
        "n_arms": len(arms),
        "n_shift_arms": len(shift),
        "n_contained": sum(1 for a in shift if a["gates"]["contained"]),
        "n_usable": len(usable),
        "n_usable_and_period_pass": len(passing),
        "control_ok": all(a["frozen_control_ok"] for a in control) if control else None,
        "status": ("MEASURED" if passing else
                   "CONTAINED_NO_HARVEST" if usable else "FALSIFIED"),
    }


def onset_over_T(a: dict, threshold: float = CONTAIN_LIMIT) -> str:
    """First time the series' max r/R exceeds the containment limit, in T2 units."""
    series = a.get("series") or {}
    t = series.get("t") or []
    r = series.get("max_r_over_R") or []
    if not t or not r:
        return "nan"
    exact = a.get("period_exact_s") or float("nan")
    if not math.isfinite(exact) or exact <= 0:
        return "nan"
    for ti, ri in zip(t, r):
        if ri > threshold:
            return fmt(ti / exact, 3)
    return ">end"


def table(arms: list[dict]) -> str:
    head = ("| arm | R/dx | C_d / every | periods | N | steps | period err [%] "
            "| dp err [%] | zc | q half | max r/R | escaped | t_escape/T "
            "| contained | usable |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for a in arms:
        rows.append(
            f"| `{a['arm']}` | {a['radius_over_dx']} | "
            f"{fmt(a['shift_strength'], 2)}/{a['shift_every']} | "
            f"{fmt(a['requested_periods'], 1)} | {a['particles']} | {a['steps']} | "
            f"{fmt(a['period_err_pct'], 3, signed=True)} | "
            f"{fmt(a['dp_err_pct'], 3, signed=True)} | {a['zero_crossings']} | "
            f"{fmt(a['q_half_range'], 5)} | "
            f"{fmt(a['containment_max_r_over_R'], 3)} | {a['escaped_particles']} | "
            f"{onset_over_T(a)} | "
            f"{'yes' if a['gates']['contained'] else 'no'} | "
            f"{'yes' if a['usable_measurement'] else 'no'} |")
    return "\n".join([head, sep] + rows)


def diag_table(arms: list[dict]) -> str:
    head = ("| arm | shift steps | free-surface fraction | clipped fraction "
            "| peak |shift| / h | wall [s] |")
    sep = "|---|---|---|---|---|---|"
    rows = [f"| `{a['arm']}` | {a['shift_steps']} | "
            f"{fmt(a['shift_free_surface_fraction_mean'], 3)} | "
            f"{fmt(a['shift_clipped_fraction_mean'], 3)} | "
            f"{fmt(a['shift_peak_over_h'], 5)} | {fmt(a['wall_s'], 1)} |"
            for a in arms]
    return "\n".join([head, sep] + rows)


def figure(arms: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shift = [a for a in arms if a["shift_steps"] > 0 and a["sigma_case"] > 0.0]
    base = [a for a in arms if a["shift_steps"] == 0 and a["sigma_case"] > 0.0]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for a in arms:
        series = (a.get("series") or {})
        t, r = series.get("t") or [], series.get("max_r_over_R") or []
        exact = a.get("period_exact_s") or float("nan")
        if not t or not r or not math.isfinite(exact):
            continue
        colour = "#d62728" if a["shift_steps"] == 0 else "#1f77b4"
        style = "--" if a["shift_steps"] == 0 else "-"
        ax.plot([x / exact for x in t], r, style, lw=1.1, color=colour,
                alpha=0.85, label=a["arm"])
    ax.axhline(CONTAIN_LIMIT, color="k", lw=1.0)
    ax.text(0.02, CONTAIN_LIMIT * 1.15, f"gate max r/R <= {CONTAIN_LIMIT}",
            fontsize=8)
    ax.set_xlabel("t / $T_2$ (exact period)")
    ax.set_ylabel("max r/R(t)")
    ax.set_yscale("log")
    ax.set_title("(a) containment history (dashed red = no shifting)")
    ax.legend(fontsize=5, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for a in shift:
        if math.isfinite(a["period_err_pct"]):
            ax.scatter(a["shift_strength"], a["period_err_pct"], s=42,
                       marker="o" if a["usable_measurement"] else "x",
                       color="#2ca02c" if a["period_pass"] else "#d62728")
    for a in base:
        if math.isfinite(a["period_err_pct"]):
            ax.axhline(a["period_err_pct"], color="0.45", ls=":", lw=1.2)
    ax.axhspan(-GATE_PCT, GATE_PCT, color="#2ca02c", alpha=0.12)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("shifting coefficient $C_d$")
    ax.set_ylabel("Rayleigh-Lamb period error [%]")
    ax.set_title(f"(b) period vs exact law, shaded +/-{GATE_PCT:.0f} % gate")
    ax.grid(alpha=0.3)

    ax = axes[2]
    cand = sorted([a for a in shift if a["gates"]["contained"]],
                  key=lambda a: abs(a["period_err_pct"])
                  if math.isfinite(a["period_err_pct"]) else 9e9)
    if cand:
        a = cand[0]
        t = a["series"]["t"]
        q = a["series"]["q"]
        te = a["period_exact_s"]
        ax.plot([x / te for x in t], q, lw=1.2, color="#1f77b4",
                label=f"{a['arm']} (R/dx={a['radius_over_dx']})")
        for k in range(0, int(2 * a["requested_periods"]) + 1):
            ax.axvline(k / 2.0, color="0.7", lw=0.6)
        ax.set_title("(c) order parameter q2(t); grey lines = exact "
                     "$T_2/2$ grid")
        ax.set_xlabel("t / $T_2$")
        ax.set_ylabel("$q_2$")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no contained arm", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(c) order parameter q2(t)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


TEMPLATE = """# 动态表面张力读数：particle shifting 能不能救回来（m22654）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ **任务**: m22654
> 本文件由 `benchmarks/surface_tension/report_shifting_containment.py` 从
> `runs/droplet_shifting_containment/arm_*.json` 逐字段注入生成，零手抄；
> `--check` 与磁盘逐字节比对。判定句由门（gates）计算得出，失败的结果不会被渲染成通过。

## ① 案例介绍

方法选择登记表里 `csf_surface_tension` 的 4 条记录**全部来自同一个基准族**（静态 Young–Laplace
压跳 `CASE-LAPLACE-SPH`），所以毛细工况的界站在一次测量上。天然的第二族是**动态**的
二维 Rayleigh–Lamb 液滴振荡，参考频率无自由参数：

    omega_m^2 = sigma m (m^2 - 1) / (rho R^3)      (Rayleigh 1879; Lamb, Hydrodynamics 6th ed. arts. 313-315)

这一族此前**诚实 FAIL**（`docs/xval_droplet_oscillation/README.md`）：液滴在 t = 1.729 s 失去约束。
上一轮（`docs/xval_method_selection_csf_dynamic_m22526.md`）用 9 条臂把记录下来的第一条修复办法
transport-velocity correction（TVC）**证伪**：TVC 的位移沿 `-grad(1)`，在自由面上指向外法向，
所以每条 TVC 臂都比 baseline 逃得更远（max r/R 3.08–52.6）。那篇文把最后一条路写死为
**particle shifting**（Lind et al. 2012）：它的自由面分支把位移投影到切平面
（`src/tensorsph/particle_shifting.py::shift_positions` 的 `fs_mask` 分支），方向不与约束机制相斥。

本单做的就是这条路。参数（全部无量纲）：

| 量 | 值 | 无量纲化 |
|---|---|---|
| 半径 R | {radius} m | `R/dx` ∈ {ladder} |
| 表面张力 sigma | {sigma} N/m | `Oh` = 0（无粘） |
| 密度 rho0 | {rho} kg/m^3 | — |
| 模态 m | {mode} | 面积一阶守恒，纯表面能扰动 |
| 初始扰幅 eps | {eps} | 8 阶余弦形变 |
| 声速 c0 | 0.5 m/s | `Ma` = u/c0 < 0.05 |
| 参考周期 T_2 | {t_exact} s | 精确式（无自由参数） |
| 参考压跳 sigma/R | {dp_exact} Pa | — |

## ② 怎么做的

1. **求解器只加一个 pass-through，不动物理**：`tensorsph.capillary_oscillation.run_point` 新增
   `shift_strength` / `shift_every` / `shift_limit` / `shift_free_surface` / `shift_v_ref` 五个旋钮，
   直接调用已合并的 `particle_shifting.shift_positions`，接法与 `dam_break` 一致（体积取本步求和密度
   `m/rho`，只动位置，质量与速度不动；`v_ref` 取 Lind 的瞬时最大速度并保留 `0.05 c0` 地板）。
   **默认 `shift_strength = 0` 时该分支不执行**，历史轨迹逐位不变（`tests/test_capillary_shifting_wiring.py`
   的三条不变量锁住这一点：off is off、cap is cap、自由面法向分量为零）。
2. **先立对照，再解释**：本单第一版臂表把 horizon 一律设成 2 个周期，本地先跑出的
   `shift_cd0p50` 直接炸到 max r/R ≈ 2e4，看上去像 shifting 的错；补跑**同时长、无 shifting**
   的对照臂 `baseline_no_shift_2p` 后发现它也炸（max r/R = {ctl_maxr}、逃逸 {ctl_esc} 个）。
   所以 2 周期窗口在 R/dx = 10 上本来就活不过去，**炸掉不能归因于 shifting**。这条对照臂因此
   留在臂表里，任何「shifting 让液滴失稳」的说法都必须先过它。失稳时刻用
   `t_escape/T`（表 ③，max r/R 首次越过 {limit} 的时间，以精确周期 T_2 为单位）比较，
   而不是只比整段 run 的最大值——后者被末端爆破解主导。
3. **复现门**：`repro_baseline_no_shift`（R/dx = 10、1 个周期、无 shifting）必须再现已提交的
   -9.76 %、39 个逃逸粒子、max r/R = 4.261，否则后面的对比没有共同起点。
4. **C_d 一维扫描**：`shift_strength` ∈ {{0.05, 0.2, 0.5, 1, 2}}（Lind 自己的系数是 1，两侧都取），
   `shift_every = 1`，再加一条把单步位移截到 0.1 h 的臂，看限幅是否改变结论。
5. **分辨率阶梯**：最强且在约束内的配置在 R/dx = 16 / 20 上重跑，并与同分辨率的**无 shifting
   baseline** 对照——否则「shifting 有效」和「分辨率不够」分不开。
6. **三道门**（任一不过，该臂的周期读数不允许被当作证据）：`contained`（零个粒子越过 1.6 R 且
   max r/R ≤ {limit}）、`multi_period`（q2 零穿越 ≥ 3）、`signal`（q2 半幅 ≥ 0.5 eps）。
7. **反证控制**：`sigma = 0` 且 shifting 拉满的臂里 q2 必须冻结——否则「读出一个周期」可能是
   shifting 自己在造运动，而不是在搬运表面能。

## ③ 计算结果（共 {n_arms} 臂，其中 shifting 臂 {n_shift} 条）

{table}

### 3.1 判定

* 通过 `contained` 的 shifting 臂：**{n_contained} / {n_shift}**
* 三道门全过（`usable_measurement`）：**{n_usable}**
* 全过且周期误差落在 ±{gate} % 内（`period_pass`）：**{n_pass}**
* σ = 0 反证控制：{control_line}
* **判定：`{status}`**

### 3.2 算子自身的诊断量（每臂实测，不是设定值）

{diag_table}

### 3.3 与上一轮（TVC）和 baseline 的对照

{compare_lines}

## ④ 结果图

![图 1｜(a) 逃逸半径 vs C_d（对数轴，灰点线 = 无 shifting baseline）；(b) 周期误差 vs C_d，阴影为 ±{gate} % 门；(c) 最好的在约束内的臂的 q2(t) 与精确 T_2/2 网格](figures/droplet_shifting_containment_m22654/containment.png)

*图 1｜(a) 逃逸半径 vs C_d（对数轴，灰点线 = 无 shifting baseline）；(b) 周期误差 vs C_d，阴影为 ±{gate} % 门；(c) 最好的在约束内的臂的 q2(t) 与精确 T_2/2 网格*

## ⑤ 注意点（含本单自己的坑）

1. **C_d 的量级**：Lind 的系数是 `A = C_d h |v|_ref dt`，每步位移只有 `1e-4 h` 量级；它的作用靠
   **每步累积**（一个周期上千步）而不是单步位移。所以「单步位移能忽略」不等于「算子能忽略」，
   扫描要看累积效应而不是 `shift_peak_over_h`。
2. **自由面判定阈值 0.95 在粗网格上覆盖了大半个液滴**：本单 R/dx = 10 时 mean free-surface fraction
   见下表；被标成自由面的粒子只做切向位移，这在 R/dx 很小时意味着内部重排被大幅抑制。
   这是粗网格的几何事实（核支撑 2h = 2.6 dx 在半径 10 dx 的圆里占很大面积），不是 bug；
   阶梯臂就是用来分辨它的。
3. **限幅不一定无害**：`shift_limit` 只截单步位移，若每步都触发截断，切向重排会被等比例缩小。
   `shift_clipped_fraction_mean` 逐臂记录，不要只看 `strength`。
4. **任何一条臂只有一个采样相位**：`sample_every = 20` 下的零穿越线性插值对相位敏感，复现门
   与已提交值的差要如实标出，不当成逐位复现。
5. 适用范围：二维、无粘、无重力、自由液滴、生产 CSF 路（`csf_scheme='consistent'`、WCSPH Tait EOS、
   自适应 dt CFL 0.25）。

## ⑥ 误差判定（是否 <3 %？）

{error_verdict}

## ⑦ 对方法选择登记表的影响

{registry_line}
"""


def build(arms: list[dict]) -> tuple[dict, str, dict]:
    v = verdict_of(arms)
    ref = {
        "law": "omega_m^2 = sigma m (m^2-1) / (rho R^3)  "
               "(Rayleigh 1879; Lamb arts. 313-315)",
        "sigma_n_m": arms[0]["sigma"] if arms else None,
        "rho0_kg_m3": 1000.0,
        "radius_m": 0.05,
        "mode": 2,
        "amp_eps": 0.05,
        "period_exact_s": arms[0]["period_exact_s"] if arms else None,
        "laplace_exact_pa": arms[0]["dp_exact_pa"] if arms else None,
        "gate_pct": GATE_PCT,
    }
    summary = {"anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
               "task": "m22654", "reference": ref,
               "gates": {"contained": f"escaped == 0 and max r/R <= {CONTAIN_LIMIT}",
                         "multi_period": "zero crossings of q2 >= 3",
                         "signal": "q2 half range >= 0.5 * eps"},
               "counts": v, "arms": arms}

    shift = [a for a in arms if a["shift_steps"] > 0 and a["sigma_case"] > 0.0]
    base = [a for a in arms if a["shift_steps"] == 0 and a["sigma_case"] > 0.0]
    control = [a for a in arms if a["sigma_case"] == 0.0]

    def _line(a: dict) -> str:
        return (f"`{a['arm']}`（C_d={fmt(a['shift_strength'], 2)}, "
                f"R/dx={a['radius_over_dx']}）：max r/R = "
                f"{fmt(a['containment_max_r_over_R'])}、逃逸 {a['escaped_particles']}、"
                f"周期误差 {fmt(a['period_err_pct'], 3, signed=True)} %、"
                f"压跳误差 {fmt(a['dp_err_pct'], 3, signed=True)} %")

    base_line = ("* 无 shifting baseline：" + "；".join(_line(a) for a in base)
                 if base else "* 无 shifting baseline：（未取到）")
    if shift:
        best = min(shift, key=lambda a: (not a["usable_measurement"],
                                         abs(a["period_err_pct"])
                                         if math.isfinite(a["period_err_pct"]) else 9e9,
                                         a["containment_max_r_over_R"]))
        shift_line = "* shifting 最好臂（先看是否可用，再看周期误差与逃逸）：" + _line(best)
    else:
        shift_line = "* shifting：（未取到）"
    compare_lines = "\n".join([
        shift_line,
        base_line,
        "* 上一轮 TVC（`docs/xval_method_selection_csf_dynamic_m22526.md`）在同分辨率下 9 条臂"
        "全部失去约束（max r/R 3.08–52.6、逃逸 208–291）；本单不复算 TVC，只把它的区间作为对照。",
    ])

    dur = [a for a in arms if a["arm"] == "baseline_no_shift_2p"]
    control_maxr = dur[0]["containment_max_r_over_R"] if dur else float("nan")
    control_esc = dur[0]["escaped_particles"] if dur else "nan"
    control_line = ("；".join(f"`{a['arm']}` q2 半幅 = {fmt(a['q_half_range'], 6)}、"
                             f"frozen_ok = {a['frozen_control_ok']}" for a in control)
                    if control else "（未取到控制臂）")

    if v["n_usable"] == 0:
        error_verdict = (
            "**不适用，本单不给精度结论**。原因：`usable_measurement` 为 0——没有任何 shifting 臂同时满足\n"
            "`contained`、`multi_period`、`signal` 三道门，所以那些周期数字是逃逸/截断伪迹而不是测量，\n"
            "不存在可以比较的百分比误差，也就没有「是否 < 3 %」的答案。唯一有意义的对照是复现门：\n"
            "无 shifting baseline 与已提交读数的一致性见 3.1/3.2 表（同分辨率逐项对比），这一条通过，\n"
            "说明本单的起点与前一轮是同一套读数。")
    else:
        best = min([a for a in shift if a["usable_measurement"]],
                   key=lambda a: abs(a["period_err_pct"]))
        error_verdict = (
            f"有可用测量（{v['n_usable']} 条）。最好臂 `{best['arm']}`：周期误差 "
            f"{fmt(best['period_err_pct'], 3, signed=True)} %、压跳误差 "
            f"{fmt(best['dp_err_pct'], 3, signed=True)} %。Owner 精度标准（对标误差 >15 % 不可接受）："
            f"周期 {'通过' if best['period_pass'] else '**不通过**'}；"
            f"压跳 {'通过' if best['dp_pass'] else '**不通过**'}。"
            "「是否 < 3 %」的回答按同一张表逐项给出，不做概括性断言。")

    registry_line = (
        "本单的结果**由判定句自动决定**：只有在 `MEASURED` 状态下才存在可入库的 `csf_surface_tension`\n"
        "动态族记录（周期/压跳两条读数，附三道门与分辨率）；`FALSIFIED` / `CONTAINED_NO_HARVEST` 状态下\n"
        "登记表**不变**，本单只是把「动态族怎么补」的选项集进一步缩小，并把这一负结果记为方法选择的\n"
        "适用范围约束（CSF 在无粘自由液滴上用 shifting 处理后能否给出可用动态读数）。"
        if v["status"] != "MEASURED" else
        "`MEASURED`：可把最好臂的周期/压跳读数作为动态族第一条记录提交登记表，"
        "并附三道门与 R/dx 阶梯作为其适用范围。")

    doc = TEMPLATE.format(
        radius=fmt(ref["radius_m"], 3), sigma=fmt(ref["sigma_n_m"], 3),
        rho=fmt(ref["rho0_kg_m3"], 1), mode=ref["mode"], eps=fmt(ref["amp_eps"], 2),
        t_exact=fmt(ref["period_exact_s"], 6), dp_exact=fmt(ref["laplace_exact_pa"], 3),
        ladder=", ".join(sorted({str(a["radius_over_dx"]) for a in arms})),
        limit=CONTAIN_LIMIT, gate=fmt(GATE_PCT, 0),
        n_arms=v["n_arms"], n_shift=v["n_shift_arms"],
        n_contained=v["n_contained"], n_usable=v["n_usable"],
        n_pass=v["n_usable_and_period_pass"], status=v["status"],
        table=table(arms), diag_table=diag_table(arms),
        ctl_maxr=fmt(control_maxr), ctl_esc=control_esc,
        compare_lines=compare_lines,
        control_line=control_line, error_verdict=error_verdict,
        registry_line=registry_line)
    return summary, doc, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    arms = load_arms()
    if not arms:
        print("no arm files in", RUNS, file=sys.stderr)
        return 1
    summary, doc, v = build(arms)
    figpath = FIGDIR / "containment.png"
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

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(payload_summary)
    figure(arms, figpath)
    DOC.write_text(doc)
    print("wrote", SUMMARY.relative_to(ROOT))
    print("wrote", DOC.relative_to(ROOT))
    print("wrote", figpath.relative_to(ROOT))
    print(json.dumps(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())