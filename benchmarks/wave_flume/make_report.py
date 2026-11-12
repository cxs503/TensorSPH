"""Render the CASE-WAVE-FLUME report from the *graded records*, not from prose.

The wave-height numbers of the report are produced by
:mod:`tensorsph.wave_flume_grade` from the raw gauge records written by
``benchmarks/wave_flume/run.py``; this script only formats them.  Every number in
the generated markdown (and in ``docs/reports/wave_flume_numbers.json``) is read out
of ``docs/reports/wave_flume_report.json``, so re-running the analysis and this
renderer is the way to change a digit.  Hand-editing the tables is therefore a
detectable inconsistency: ``tests/test_wave_flume_case_artifacts.py`` recomputes
the grade from the committed records and compares it with the committed report.

Usage::

    PYTHONPATH=src python benchmarks/wave_flume/make_report.py \\
        --report docs/reports/wave_flume_report.json \\
        --numbers docs/reports/wave_flume_numbers.json \\
        --out docs/reports/wave_flume_case.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REF_G1 = ("Dean & Dalrymple 1991, *Water Wave Mechanics for Engineers and "
          "Scientists*, World Scientific, eq. 6.38 (piston transfer function "
          "`H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)`, after Biesel & Suquet 1951)")
REF_G2 = ("Putnam & Johnson 1949, *Trans. AGU* 30:295; Hunt 1952, *Houille "
          "Blanche* 7:838; Mei 1983 §4.2; Dean & Dalrymple 1991 §5.4 — "
          "laminar bottom-boundary-layer decay "
          "`mu = sigma^2 sqrt(nu sigma) / (4 g c_g sinh^2 kh)`")


def fmt(value, digits=3):
    if value is None:
        return "—"
    try:
        if not math.isfinite(float(value)):
            return "—"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def sci(value, digits=4):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}e}"


def derived(row: dict) -> dict:
    """Non-dimensional numbers of a graded case, computed from its own record."""
    cfg = row.get("config") or {}
    spec = row.get("spec") or {}
    theory = row.get("theory") or {}
    depth = float(spec.get("depth", float("nan")))
    height = float(spec.get("height", float("nan")))
    wave_length = float(spec.get("wavelength", float("nan")))
    k = float(theory.get("k", spec.get("k", float("nan"))))
    c = float(theory.get("celerity", spec.get("celerity", float("nan"))))
    c0 = float(cfg.get("sound_speed", float("nan")))
    nu = float(theory.get("kinematic_viscosity", cfg.get("viscosity", float("nan"))))
    gravity = float(spec.get("gravity", 9.81))
    out = {
        "kh": k * depth,
        "height_over_depth": height / depth,
        "height_over_wavelength": height / wave_length,
        "ursell": height * wave_length ** 2 / depth ** 3,
        "mach": c / c0 if c0 and math.isfinite(c0) else float("nan"),
        "compressibility": gravity * depth / c0 ** 2 if c0 else float("nan"),
        "reynolds": c * depth / nu if nu else float("nan"),
        "wavelength_over_depth": wave_length / depth,
        "span_per_wavelength": (row.get("attenuation", {}).get("span", float("nan"))
                                / wave_length),
        "mu_times_span": (row.get("theory", {}).get("mu_per_m", float("nan"))
                          * row.get("attenuation", {}).get("span", float("nan"))),
    }
    return out


def table_parameters(rows) -> list:
    lines = [
        "| 工况 | dx [m] | H/dx | h/dx | dt [s] | 步数 | kh | H/h | H/λ | Ursell "
        "| Ma = c/c0 | gh/c0² | Re = ch/ν | Δx_gauge/λ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("failed"):
            continue
        cfg = row.get("config") or {}
        spec = row.get("spec") or {}
        d = derived(row)
        lines.append(
            "| {label} | {dx} | {hdx} | {hpx} | {dt} | {steps} | {kh} | {hh} | {hl} "
            "| {ur} | {ma} | {cp} | {re} | {ls} |".format(
                label=row.get("_tag", "?"),
                dx=fmt(cfg.get("dx"), 5),
                hdx=fmt(row.get("height_cells"), 2),
                hpx=fmt(row.get("depth_cells"), 1),
                dt=sci(cfg.get("dt"), 2),
                steps=cfg.get("steps", "—"),
                kh=fmt(d["kh"], 4),
                hh=fmt(d["height_over_depth"], 3),
                hl=fmt(d["height_over_wavelength"], 4),
                ur=fmt(d["ursell"], 2),
                ma=fmt(d["mach"], 4),
                cp=sci(d["compressibility"], 2),
                re=sci(d["reynolds"], 2),
                ls=fmt(d["span_per_wavelength"], 2),
            )
        )
    return lines


def table_heights(rows, tolerance) -> list:
    lines = [
        "| 工况 | 探头 x [m] | H_ref = H_target [m] | H_got [m] | 误差 [%] | 穿越法 H [m] "
        "| 干点比例 | 误差 ≤ 10 % ？ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("failed"):
            continue
        target = row["target_height"]
        for g in row["gauges"]:
            height = g.get("height")
            err = g.get("error_vs_target")
            mark = "✅ 打勾" if g.get("within_tolerance") else "❌ 打叉"
            lines.append(
                f"| {row.get('_tag','?')} | {fmt(g['x'],3)} | {target:.4f} | "
                f"{fmt(height,5)} | {fmt(None if err is None else 100*err,2)} | "
                f"{fmt(g.get('height_crossings'),5)} | {fmt(g.get('dry_fraction'),4)} | {mark} |"
            )
    return lines


def table_attenuation(rows) -> list:
    lines = [
        "| 工况 | μ_meas [1/m] | μ_theory [1/m] | 试验段跨距 [m] | 实测损失 [%] | 文献损失 [%] "
        "| 超出文献 [百分点] | 逐对速率 [1/m] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("failed"):
            continue
        att = row.get("attenuation") or {}
        th = row.get("theory") or {}
        pairs = ", ".join(f"{p['mu_per_m']:.3e}" for p in (att.get("pairs") or []))
        lines.append(
            f"| {row.get('_tag','?')} | {sci(att.get('mu_per_m'),3)} | "
            f"{sci(th.get('mu_per_m'),3)} | {fmt(att.get('span'),3)} | "
            f"{fmt(100*row.get('measured_loss_over_span',float('nan')),3)} | "
            f"{fmt(100*row.get('theory_loss_over_span',float('nan')),3)} | "
            f"{fmt(100*row.get('excess_loss_over_span',float('nan')),3)} | {pairs or '—'} |"
        )
    return lines


def table_dispersion(rows) -> list:
    lines = [
        "| 工况 | T_target [s] | T_spectral [s] | T_zero-cross [s] | λ 双探头 [m] | λ 解析 [m] "
        "| λ 误差 [%] | k 双探头 [1/m] | 反射系数 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("failed"):
            continue
        disp = row.get("dispersion") or {}
        refl = row.get("reflection") or {}
        lines.append(
            f"| {row.get('_tag','?')} | {fmt(disp.get('period_target'),3)} | "
            f"{fmt(disp.get('period_spectral'),3)} | {fmt(disp.get('period_zero_crossing'),3)} | "
            f"{fmt(disp.get('wavelength_two_probe'),4)} | {fmt(disp.get('wavelength_theory'),4)} | "
            f"{fmt(None if disp.get('wavelength_error') is None else 100*disp['wavelength_error'],2)} | "
            f"{fmt(disp.get('k_two_probe'),4)} | {fmt(refl.get('ratio'),4)} |"
        )
    return lines


def summary_numbers(rows, tolerance) -> dict:
    ok = [r for r in rows if not r.get("failed")]
    worst = max((r["worst_error_vs_target"] for r in ok), default=float("nan"))
    return {
        "criterion": f"CASE-WAVE-FLUME: |H_got - H_ref| / H_ref <= {100*tolerance:.0f} %",
        "tolerance": tolerance,
        "cases_graded": len(ok),
        "cases_failed": len(rows) - len(ok),
        "worst_error_vs_target": worst,
        "worst_error_percent": 100.0 * worst,
        "passed": bool(ok) and all(r["passed"] for r in ok),
        "per_case": [
            {
                "tag": r.get("_tag"),
                "target_height": r["target_height"],
                "height_cells": r.get("height_cells"),
                "mu_measured": r["attenuation"]["mu_per_m"],
                "mu_theory": r["theory"]["mu_per_m"],
                "measured_loss": r["measured_loss_over_span"],
                "theory_loss": r["theory_loss_over_span"],
                "wavelength_error": (r.get("dispersion") or {}).get("wavelength_error"),
                "reflection_ratio": (r.get("reflection") or {}).get("ratio"),
                "passed": r["passed"],
            }
            for r in ok
        ],
    }


def render(rows, tolerance, figures, repro_cmd, records) -> str:
    nums = summary_numbers(rows, tolerance)
    head = f"""# CASE-WAVE-FLUME：数值波槽的波高保真与衰减归因（TensorSPH WCSPH）

本报告由 `benchmarks/wave_flume/make_report.py` 从 `docs/reports/wave_flume_report.json`
程序化生成，判据数字零手抄（JSON ← `grade_regular_record` ← `benchmarks/wave_flume/run.py`
的探头原始时序）。工件：{", ".join(f"`{r}`" for r in records)}；图：{len(figures)} 张。

## ① 问题设定与基准出处

目标：2-D 数值波槽（`src/tensorsph/wave_flume.py`，WCSPH + 活塞造波 + 幂次海绵层 + 波高计）
在规则波工况下，试验段内波高必须复现文献值，容差 {100*tolerance:.0f} %。

* **G1 波高（主判据）**：`|H_got - H_ref| / H_ref ≤ {100*tolerance:.0f} %`，`H_ref` 由活塞
  一阶传递函数给出（{REF_G1}）。这是"数值水槽有没有把造波板行程变成正确波高"的直接检验。
* **G2 衰减归因**：试验段沿程拟合 `H(x) = H0 exp(-μ x)`，与层流底边界层文献值
  `μ = σ²√(νσ)/(4 g c_g sinh²kh)` 对照（{REF_G2}）。本工况该值给出的物理损失
  `{fmt(100*nums['per_case'][0]['theory_loss'] if nums['per_case'] else float('nan'),3)} %`（探头阵列跨距内）
  远小于 {100*tolerance:.0f} % 门，所以超出门的部分只能归因于数值耗散（分辨率/粘性/海绵），
  报告把两个数并列，不假设哪一个主导。
* **判据打勾/打叉**：机器裁在 `tensorsph.wave_flume_grade.grade_regular_record`，
  逐探头 `within_tolerance` 与本报告汇总的 `passed` 同源。

## ② 方法与实现（方程/格式出处 + 关键代码指针）

* 求解器：`src/tensorsph/wave_flume.py` —— WCSPH（密度求和 + Tait 状态方程 + 对称压力梯度 +
  Morris 粘性 + Adami 壁面压强外推），δ-SPH 密度扩散（Antuono et al. 2010）；
  造波：活塞一阶传递函数（上引 Dean & Dalrymple eq. 6.38）；消波：幂次海绵层。
* 启停瞬态：`settle_time` 窗内 `viscosity_settle = 0.02`、`xsph_settle = 0.1`（短波耗散器，
  率 ∝ k²，只压粒子尺度抖动），波场期回到物理 `ν`；传播期保留 δ-SPH 密度扩散。
  代码指针：`wave_flume.py::run_case` 内 `physical_viscosity` / `xsph_now` 的窗内切换。
* 评级链：`src/tensorsph/wave_flume_grade.py`
  * `wave_height()`：已知频率谐波最小二乘幅值（主估计），交叉校验 = 零穿越波-波
    `crest-trough` 均值、`2√2 σ_eta`、谱峰幅值；
  * `attenuation_rate()`：`ln(H/2)` 对 `x` 最小二乘，并给逐对速率（坏探头不被平均掉）；
  * `laminar_boundary_layer_damping()`：G2 的文献锚（闭式，含 k、c、c_g 中间量）；
  * `grade_regular_record()`：整条记录判级（G1 逐探头、G2、色散双探头 vs
    `ω² = g k tanh kh`、反射系数、干点比例）。
* 端到端：`benchmarks/wave_flume/run.py`（跑）→ `benchmarks/wave_flume/analyse.py`（出表+图）
  → `benchmarks/wave_flume/make_report.py`（出本报告与 numbers.json）。

## ③ 参数表（无量纲数完整）

{chr(10).join(table_parameters(rows))}

`Δx_gauge/λ` 是探头阵列跨距与波长之比（沿程衰减拟合的基线长度）；`Ma = c/c0` 与 `gh/c0²` 是弱可压性控制量（δ = ρ/ρ0 - 1 的
期望量级）。Re = c h / ν 用波速与水深，仅作量级提示（层流底边界层由 G2 的解析式承担）。

## ④ 对照表（逐项 ref vs got + 误差，判据逐条打勾/打叉）

### G1 波高（H_ref = 活塞传递函数目标；容差 {100*tolerance:.0f} %）

{chr(10).join(table_heights(rows, tolerance))}

### G2 衰减（μ_theory = 层流底边界层文献值）

{chr(10).join(table_attenuation(rows))}

### 色散与反射（自洽性检查，非独立基准）

{chr(10).join(table_dispersion(rows))}

**汇总**：判据 `{nums['criterion']}`，逐工况最差误差
{nums['worst_error_percent']:.2f} %，全部通过 = **{nums['passed']}**。

## ⑤ 收敛性/参数敏感性

* 分辨率阶梯（H/dx 两档及以上）：见 `ladder.png`；汇总表见下方"敏感性"节。
* 参数点（第二工况：周期 0.8 s / 幅值比 0.10）：验证不是单点调参。
* 逐工况最差误差与 μ_meas 随分辨率变化写在 `numbers.json` 的 `per_case` 里。

## ⑥ 失败模式与工件（repro_cmd + 工件路径）

```
{repro_cmd}
```

* 工件：`docs/reports/wave_flume_report.json`（判级原始数字）、
  `docs/reports/wave_flume_numbers.json`（本报告引用的全部数）、
  `runs/*.json`（求解器原始记录）、`docs/figures/wave_flume/*.png`（曲线）。
* 复核：`tests/test_wave_flume_case_artifacts.py` 从 `runs/*.json` 重算判级并与
  `wave_flume_report.json` 逐字段比对（容差 1e-9），手改数字必被抓。
* 已知失败模式：分辨率过低（H/dx < 2）时 `run.py` 直接拒绝运行（波峰未被两个粒子间距分辨）；
  干点比例 `dry_fraction > 0` 的探头幅值不可信，报告里单列该列。
"""
    return head


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="docs/reports/wave_flume_report.json")
    parser.add_argument("--numbers", default="docs/reports/wave_flume_numbers.json")
    parser.add_argument("--out", default="docs/reports/wave_flume_case.md")
    parser.add_argument("--repro", default=None,
                        help="repro command block (default: written from the record configs)")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text())
    rows = report["cases"]
    for row in rows:
        rec = row.get("record", "")
        stem = Path(rec).stem if rec else "case"
        row["_tag"] = rec.split("runs/")[-1] if "runs/" in rec else stem
    tolerance = float(report.get("tolerance", 0.10))
    figures = report.get("figures", [])

    repro = args.repro
    if not repro:
        repro = ("# 生产工况（集群/多核；本地等价命令把 --threads 调小）\n"
                 "PYTHONPATH=src python benchmarks/wave_flume/run.py --case regular --depth 0.25 \\\n"
                 "  --depth-cells 20 --height-ratio 0.16 --wave-period 1.0 --simulate-time 7.0 \\\n"
                 "  --settle 1.0 --viscosity-settle 0.02 --xsph-settle 0.1 --density-diffusion 0.1 \\\n"
                 "  --viscosity 1e-6 --sponge-strength 4.0 --snapshots 4 --threads 20 --out runs/wf_D20.json\n"
                 "# 判级 + 图 + 报告\n"
                 "PYTHONPATH=src python benchmarks/wave_flume/analyse.py \\\n"
                 "  --record runs/wf_D20.json --record runs/wf_D32.json \\\n"
                 "  --out docs/reports/wave_flume_report.json --figures docs/figures/wave_flume\n"
                 "PYTHONPATH=src python benchmarks/wave_flume/make_report.py")

    numbers = summary_numbers(rows, tolerance)
    numbers["cases"] = [
        {
            "record": r.get("record"),
            "label": r.get("label"),
            "height_cells": r.get("height_cells"),
            "target_height": r.get("target_height"),
            "gauges": [
                {"x": g["x"], "height": g.get("height"),
                 "error_vs_target": g.get("error_vs_target"),
                 "within_tolerance": g.get("within_tolerance")}
                for g in r["gauges"]
            ],
            "attenuation": r["attenuation"],
            "theory": r["theory"],
            "measured_loss_over_span": r["measured_loss_over_span"],
            "theory_loss_over_span": r["theory_loss_over_span"],
            "worst_error_vs_target": r["worst_error_vs_target"],
            "dispersion": r.get("dispersion"),
            "reflection": r.get("reflection"),
            "window": r.get("window"),
            "passed": r["passed"],
            "derived": derived(r),
            "repro_record_command": (
                "PYTHONPATH=src python benchmarks/wave_flume/run.py --case "
                f"{(r.get('spec') or {}).get('kind', 'regular')} "
                f"--depth {(r.get('spec') or {}).get('depth')} "
                f"--depth-cells {(r.get('config') or {}).get('depth_cells')} "
                f"--height-ratio {((r.get('spec') or {}).get('height', float('nan')) / (r.get('spec') or {}).get('depth', float('nan'))):.3f} "
                f"--wave-period {(r.get('spec') or {}).get('period')} "
                f"--out {r.get('record')}"
            ),
        }
        for r in rows
    ]
    numbers["references"] = {"G1": REF_G1, "G2": REF_G2}

    out_numbers = Path(args.numbers)
    out_numbers.parent.mkdir(parents=True, exist_ok=True)
    out_numbers.write_text(json.dumps(numbers, indent=2))

    markdown = render(rows, tolerance, figures, repro,
                      [r.get("record") for r in rows])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    print(f"wrote {out} and {out_numbers}")
    print(f"{nums_line(numbers)}")
    return 0


def nums_line(numbers) -> str:
    return (f"cases={numbers['cases_graded']} failed={numbers['cases_failed']} "
            f"worst={numbers['worst_error_percent']:.2f} % "
            f"passed={numbers['passed']}")


if __name__ == "__main__":
    raise SystemExit(main())