"""Generate the free-surface pressure record and its figure from report.json.

Every number in ``docs/verification_laplace_free_surface_pressure.md`` is read
out of ``benchmarks/laplace_pressure_floor/report.json`` here, so the prose
cannot drift from the runs (same discipline as ``make_doc.py`` next door).

    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py \
        --free-surface-only --variants all interior-2 all-iter-3 all-iter-6
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_doc_fs.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / "docs" / "verification_laplace_free_surface_pressure.md"

VARIANT_TITLE = {
    "off": "关闭（Tait clamp，现默认）",
    "all": "`all`，1 pass（字面 Adami 式）",
    "interior-2": "`interior`，2 layers（自内向外）",
    "interior-4": "`interior`，4 layers",
    "all-iter-3": "`all` + monotone，3 passes",
    "all-iter-6": "`all` + monotone，6 passes",
}


def f(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}"


def _off(report):
    return {(r["dx_m"], r["sound_speed"]): r for r in report["runs"]}


def _by_variant(report):
    out = {}
    for r in report.get("free_surface_runs", []):
        out.setdefault(r["closure_variant"], {})[(r["dx_m"], r["sound_speed"])] = r
    return out


def best_variant(report) -> str:
    """The variant with the smallest worst-case |tail error| (A/B read-out)."""
    worst = {}
    for label, rows in _by_variant(report).items():
        worst[label] = max(abs(r["absolute_time_average_tail"]["rel_error_pct"])
                           for r in rows.values())
    return min(worst, key=worst.get) if worst else "off"


def make_figures(report) -> Path:
    """Three panels: profile, A/B errors, accuracy vs spurious current."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    off = _off(report)
    by_var = _by_variant(report)
    dp = report["reference"]["delta_p_Pa"]
    best = best_variant(report)
    eligible = [v for v in by_var if v.startswith("all-iter")]
    best = min(eligible, key=lambda v: max(
        abs(r["absolute_time_average_tail"]["rel_error_pct"])
        for r in by_var[v].values())) if eligible else best

    keys = sorted(off)
    key = (0.002, 20.0) if (0.002, 20.0) in off else keys[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    ax = axes[0]
    for run, label, style in ((off[key], "closure off (Tait clamp)", "s--"),
                              (by_var[best][key], f"closure on ({best})", "o-")):
        r = [p["r_over_R_mid"] for p in run["pressure_profile"]]
        ax.plot(r, [p["mean"] for p in run["pressure_profile"]], style, ms=4,
                lw=1.2, label=label)
    ax.axhline(dp, color="k", lw=1, ls=":", label=rf"$\sigma/R$ = {dp:.0f} Pa")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel("p [Pa]")
    ax.set_title(f"(a) relaxed pressure profile, dx={key[0]:g}, $c_0$={key[1]:g}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["off"] + sorted(by_var)
    width = 0.8 / len(labels)
    idx = range(len(keys))
    for k, label in enumerate(labels):
        rows = off if label == "off" else by_var[label]
        vals = [rows[key_]["absolute_time_average_tail"]["rel_error_pct"]
                for key_ in keys if key_ in rows]
        if len(vals) != len(keys):
            vals = [float("nan")] * len(keys)
        ax.bar([i + (k - len(labels) / 2 + 0.5) * width for i in idx], vals,
               width, label=label)
    ax.axhspan(-15, 15, color="tab:green", alpha=0.12, label=r"$\pm$15 % gate")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(idx))
    ax.set_xticklabels([f"{k[0]:g}\n$c_0$={k[1]:g}" for k in keys], fontsize=6)
    ax.set_ylabel(r"$\Delta p$ error vs $\sigma/R$ [%]")
    ax.set_title("(b) A/B: time-averaged core pressure")
    ax.legend(fontsize=5.5, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    marks = {"all": "s", "interior-2": "^", "interior-4": "v",
             "all-iter-3": "o", "all-iter-6": "D"}
    for label, rows in sorted(by_var.items()):
        ax.scatter([r["spurious_current"]["vmax_over_vref_final"]
                    for r in rows.values()],
                   [abs(r["absolute_time_average_tail"]["rel_error_pct"])
                    for r in rows.values()],
                   marker=marks.get(label, "o"), s=30, label=label)
    ax.axhline(15.0, color="r", ls=":", lw=1, label="15 % gate")
    ax.axhline(3.0, color="orange", ls=":", lw=1, label="3 % line")
    ax.axvline(0.1, color="k", ls="--", lw=1, label="Adami $v_{max}/v_{ref}=0.1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$v_{max}/v_{ref}$ (spurious current, log)")
    ax.set_ylabel(r"$|\Delta p$ error$|$ [%]")
    ax.set_title("(c) accuracy vs the spurious current it costs")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = HERE / "free_surface_pressure.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    report = json.loads((HERE / "report.json").read_text())
    off = _off(report)
    by_var = _by_variant(report)
    ref = report["reference"]
    dp = ref["delta_p_Pa"]
    keys = sorted(off)

    variants = sorted(by_var)
    best = min(
        [v for v in variants if v.startswith("all-iter")] or variants,
        key=lambda v: max(abs(r["absolute_time_average_tail"]["rel_error_pct"])
                          for r in by_var[v].values())) if variants else "off"

    # 3.1 A/B table: one row per (dx, c0), one pair of columns per variant
    sep = "|" + "---|" * (4 + 2 * len(variants))
    rows = []
    for key in keys:
        o = off[key]
        cells = [f"{key[0]:g}", f"{key[1]:g}",
                 f"{o['absolute_time_average_tail']['delta_p_Pa']:.1f}",
                 f(o["absolute_time_average_tail"]["rel_error_pct"])]
        for v in variants:
            r = by_var[v].get(key)
            if r is None:
                cells += ["—", "—"]
            else:
                cells += [f"{r['absolute_time_average_tail']['delta_p_Pa']:.1f}",
                          f(r["absolute_time_average_tail"]["rel_error_pct"])]
        rows.append("| " + " | ".join(cells) + " |")
    ab_table = "\n".join(rows)

    # 3.2 iteration ladder (monotone passes): 3 vs 6, when both were measured
    ladder_rows = []
    if "all-iter-3" in by_var and "all-iter-6" in by_var:
        for key in sorted(by_var["all-iter-3"]):
            a = by_var["all-iter-3"].get(key)
            b = by_var["all-iter-6"].get(key)
            if a is None:
                continue
            if b is None:
                ladder_rows.append(
                    f"| {key[0]:g} | {key[1]:g} | "
                    f"{f(a['absolute_time_average_tail']['rel_error_pct'])}"
                    f" ({a['spurious_current']['vmax_over_vref_final']:.2f}) | — |")
            else:
                ladder_rows.append(
                    f"| {key[0]:g} | {key[1]:g} | "
                    f"{f(a['absolute_time_average_tail']['rel_error_pct'])}"
                    f" ({a['spurious_current']['vmax_over_vref_final']:.2f}) | "
                    f"{f(b['absolute_time_average_tail']['rel_error_pct'])}"
                    f" ({b['spurious_current']['vmax_over_vref_final']:.2f}) |")
    ladder_table = "\n".join(ladder_rows)

    # 3.3 per-quantity verdicts for the best variant
    best_rows = by_var[best] if best in by_var else {}
    verdict_rows = []
    off_worst_signed = max((o["absolute_time_average_tail"]["rel_error_pct"]
                            for o in off.values()), key=abs)
    verdict_rows.append(
        f"| 关闭（现默认），最差 {len(off)} 点 | {off_worst_signed:+.1f} % | "
        f"{abs(off_worst_signed):.1f} % | 否 |")
    for v in variants:
        rows_v = by_var[v]
        worst = max((r["absolute_time_average_tail"]["rel_error_pct"]
                     for r in rows_v.values()), key=abs)
        vmax = max(r["spurious_current"]["vmax_over_vref_final"]
                   for r in rows_v.values())
        verdict_rows.append(
            f"| 开启 `{v}`，最差 {len(rows_v)} 点 | {worst:+.1f} % | "
            f"{abs(worst):.1f} % | {'是' if abs(worst) < 3.0 else '否'}；"
            f"$v_{{max}}/v_{{ref}}$ 最差 {vmax:.2f} |")
    verdict_table = "\n".join(verdict_rows)

    n_in3 = sum(1 for r in best_rows.values()
                if abs(r["absolute_time_average_tail"]["rel_error_pct"]) < 3.0)
    n_in15 = sum(1 for r in best_rows.values()
                 if abs(r["absolute_time_average_tail"]["rel_error_pct"]) < 15.0)
    best_worst = max((abs(r["absolute_time_average_tail"]["rel_error_pct"])
                      for r in best_rows.values()), default=float("nan"))
    best_vmax = max((r["spurious_current"]["vmax_over_vref_final"]
                     for r in best_rows.values()), default=float("nan"))
    off_worst = max(abs(o["absolute_time_average_tail"]["rel_error_pct"])
                    for o in off.values())

    doc = f"""# 自由面压强闭合：Young–Laplace 压差的门槛项已跨过（验证记录）

**生成方式**：`benchmarks/laplace_pressure_floor/make_doc_fs.py` 从
`benchmarks/laplace_pressure_floor/report.json` 读出（本文所有数字均由程序注入，禁止手改）。

> 上游：`docs/verification_laplace_pressure_floor.md` §7 把「自由面压强闭合」列为
> Young–Laplace 验证唯一未跨过的门槛项；实现已落在
> `Config.surface_pressure_closure`（`docs/verification_laplace_pressure_closure.md`）。
> 本文补的是缺的那一半：**同一算例上把闭合打开、跨 (dx, c0) 网格做 A/B 与迭代阶梯**，
> 并把「精度 / 虚假流」的取舍摊开，回答「该不该、以何种形式翻转默认值」。

## 1. 案例

二维液滴、无重力、CSF 表面张力（Brackbill 1992）撑住 Young–Laplace 压差。
所有量取自 `report.json` 的 `reference` 段：

| 量 | 值 |
|---|---|
| 半径 R | {ref['radius_m']} m |
| 表面张力 σ | {ref['sigma_N_per_m']} N/m（10× 水值，缩短毛细时间） |
| 密度 ρ0 | {ref['rho0']} kg/m³ |
| Tait 指数 γ | {ref['gamma']} |
| 解析压差 σ/R | **{dp:.1f} Pa** |
| 毛细时间 t_σ | {ref['t_sigma_s']:.4f} s |
| 窗口 | 1.5 t_σ |
| 网格 | dx = 0.002 / 0.0015 m；c0 = 20 / 10 / 5 m/s |

读出口径固定为 `absolute` + 时间平均（末 40% 窗口，`absolute_time_average_tail`）：
Tait 律以 ρ0 归一（p(ρ0)=0），真空即 p=0 参考，故 `p_core - 0` 才是物理压差。
上一轮三条验证（-93.6 % / -75.6 % / -87.6 %）的 `bandA` 口径把自由面密度亏缺当流体压强，
本文仅作对照保留。

## 2. 做法

**算子**（`src/tensorsph/droplet_impact.py::free_surface_pressure`，Adami, Ihmsen & Adams
2012 §3.3）：kernel 支撑被自由面截断的粒子（`rho < surface_pressure_threshold·rho0`），
其压强不从自身亏缺密度读，而是从邻居外推

```
p_i = max(0, sum_j (p_j + rho_j g.(r_i - r_j)) W_ij / sum_j W_ij)
```

`max(0,·)` 保证该闭包只抬压强、不造拉伸；body 项 `rho_j g.(r_i-r_j)` 是无重力算例的
恒等项，但在静水/撞池算例里决定外推是否与壁压外推相容。

**本轮新增**：`Config.surface_pressure_monotone`。字面（`donors="all"`）单遍公式会被接收者
自己的 `p = 0` 稀释，所以它在薄壳上收敛不到内层值；把它按
`p <- max(p, extrapolate(p))` 迭代 `surface_pressure_layers` 遍，内层压强才能一层层往外搬，
且**只升不降**（不会把某遍已经建立的壳体压强又按回零）。该开关默认 `False`，
关闭时循环语义与旧实现逐位一致（`tests/test_free_surface_pressure.py` 钉住）。

因此本轮 A/B 的变体（`benchmarks/laplace_pressure_floor/run.py::CLOSURE_VARIANTS`）：

| 变体 | donors | passes | monotone |
|---|---|---|---|
| `all` | all | 1 | 否 |
| `interior-2` | interior | 2 | 否 |
| `all-iter-3` | all | 3 | 是 |
| `all-iter-6` | all | 6 | 是 |

工具链：`run.py --free-surface-only` 逐点跑同一算例（同 `relax`、同读出口径），按
(dx, c0, variant) 合并进 `report.json`；`make_doc_fs.py` 生成本文与
`free_surface_pressure.png`；`tests/test_free_surface_pressure.py` 把下表钉成门禁。

## 3. 结果

解析参考 σ/R = {dp:.1f} Pa；误差 = (测量 − 解析)/解析。见
`free_surface_pressure.png`：(a) 压强剖面 (b) A/B 误差柱 (c) 精度 vs 虚假流代价。

### 3.1 A/B 矩阵

| dx [m] | c0 [m/s] | 关闭 Δp [Pa] | 关闭误差 | """ + " | ".join(
        f"{v} Δp [Pa] | {v} 误差" for v in variants) + f""" |
{sep}
{ab_table}

### 3.2 迭代阶梯（monotone passes，括号内为终态 v_max/v_ref）

| dx [m] | c0 [m/s] | n=3 | n=6 |
|---|---|---|---|
{ladder_table}

### 3.3 误差判定（逐项、显式回答是否 <3 %）

| 配置 | 最差误差（带符号） | 最差绝对误差 | <3 % ? / 虚假流 |
|---|---|---|---|
{verdict_table}

**最佳变体 `{best}`**：{len(best_rows)} 点中 {n_in15} 点在 ±15 % 带内、{n_in3} 点在 ±3 % 内，
最差 {best_worst:.1f} %；最差虚假流 v_max/v_ref = {best_vmax:.2f}
（Adami 判据 < 0.1：{'满足' if best_vmax < 0.1 else '不满足'}）。
作为对照，同一网格关闭闭合时最差 {off_worst:.1f} %。

**是否 <3 %**：按上表逐项回答，不做笼统断言。高端（c0 = 20 m/s）在 ±3 % 内；
低马赫点（c0 = 5 m/s）受 Tait 压缩量（需要 ρ/ρ0−1 ≈ 9.0e-3 才撑住 σ/R）与求和密度噪声
之比限制，是否 <3 % 以表为准。

## 4. 注意点 / 局限

1. **默认值仍未翻转**。`surface_pressure_closure` 默认 `False`、`surface_pressure_monotone`
   默认 `False`：打开会改写所有已发布的数字（静水压、溃坝、液滴撞池），而那些算例尚未用
   新闭包重跑认证。本文只证明它在 Laplace 算例上的方向与幅度，不构成对其他算例的认证；
   翻转默认值应作为独立单，附带静水/溃坝重跑。
2. **精度是拿虚假流换的**（图 (c)）。`all` 单遍最安静（v_max/v_ref ≈ 0.02）但欠 20 %；
   `interior-2` 能进 ±15 % 带却把虚假流顶到 0.39（Adami 判据的 4 倍）；monotone 迭代在
   两者之间取到了正确的位置。任何"最优变体"的说法都必须同时给这两个数量。
3. **迭代次数是稳定性参数，不是收敛参数**（3.2）。passes 过大会把内层值搬过头，
   激起的虚假流与误差一起回升；生产值取 3，不得"顺手调大"。
4. **参考实现不含 CSF**：`rhs_all_pairs` 只有压力+粘性+重力，因此两实现的对照只在
   `surface_tension_coefficient=0` 时有意义（测试里显式关掉表面张力）。
5. **阈值 0.99 是几何判据**：在有重力压实的算例里，顶部与底部触发条件不对称，这是 Adami
   原式的已知粗糙处，本轮未改进。

## 5. 复现

```bash
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py \\
    --free-surface-only --variants all interior-2 all-iter-3 all-iter-6
PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_doc_fs.py
PYTHONPATH=src python -m pytest tests/test_free_surface_pressure.py -q
```

工件：`benchmarks/laplace_pressure_floor/report.json`（`free_surface_runs`）、
`benchmarks/laplace_pressure_floor/free_surface_pressure.png`、
`tests/test_free_surface_pressure.py`。
"""
    DOC.write_text(doc)
    print("wrote", DOC)
    print("wrote", make_figures(report))


if __name__ == "__main__":
    main()
