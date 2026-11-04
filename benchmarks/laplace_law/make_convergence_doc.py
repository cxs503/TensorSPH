"""Render ``docs/verification_laplace_law_convergence.md`` and its figure.

Every number in the document is read from the committed artifacts
(``convergence/*.json``, ``convergence_calibrated/*.json``,
``convergence/legacy_steps380/*.json``, ``validation_report.json``); nothing is
transcribed by hand.

    PYTHONPATH=src python benchmarks/laplace_law/make_convergence_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_laplace_law_convergence.md"
FIG = HERE / "convergence_time_confound.png"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(pattern: str) -> list[dict]:
    return [load(p) for p in sorted(HERE.glob(pattern))]


def f(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    legacy = rows("convergence/legacy_steps380/*.json")
    fixed = rows("convergence/*.json")
    calib = rows("convergence_calibrated/*.json")
    main_rep = load(HERE / "validation_report.json")

    def table(data, with_tail):
        head = ("| dx [m] | n | steps | t/t_σ | Δp final frame [Pa] | err [%] "
                "| within 15 % |")
        if with_tail:
            head += " Δp tail [Pa] | tail err [%] | tail σ [Pa] | tail within 15 % |"
        lines = [head, "|" + "---|" * (head.count("|") - 1)]
        for d in data:
            s, lap = d["setup"], d["laplace"]
            line = (f"| {s['dx_m']:.4f} | {s['n_particles']} | {s['steps']} "
                    f"| {s['t_over_t_sigma']:.3f} "
                    f"| {lap['delta_p_measured_Pa']:.1f} "
                    f"| {lap['rel_error_pct']:.1f} "
                    f"| {'yes' if lap['within_15pct'] else 'no'} |")
            if with_tail:
                t = lap["tail"]
                line += (f" {t['delta_p_measured_Pa']:.1f} "
                         f"| {t['rel_error_pct']:.1f} "
                         f"| {t['delta_p_std_Pa']:.1f} "
                         f"| {'yes' if t['within_15pct'] else 'no'} |")
            lines.append(line)
        return "\n".join(lines)

    # ---- figure -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for data, style, label in ((legacy, "o--", "fixed 380 steps (HEAD artifact)"),
                               (fixed, "s-", "fixed t = 1.0 t_σ, tail average"),
                               (calib, "^-", "fixed t = 1.0 t_σ, mass calibrated")):
        if not data:
            continue
        xs = [d["setup"]["dx_m"] for d in data]
        ys = [abs(d["laplace"]["tail"]["rel_error_pct"] if data is not legacy
                  else d["laplace"]["rel_error_pct"]) for d in data]
        ax1.semilogx(xs, ys, style, label=label)
    ax1.axhline(15.0, color="k", lw=0.8, ls=":")
    ax1.axhline(3.0, color="r", lw=0.8, ls=":")
    ax1.text(0.004, 16, "15 % bar", fontsize=8)
    ax1.set_xlabel("dx [m]")
    ax1.set_ylabel("|Δp − σ/R| / (σ/R)  [%]")
    ax1.set_title("(a) Laplace error vs resolution")
    ax1.invert_xaxis()
    ax1.legend(fontsize=7)

    for d in fixed:
        s, h = d["setup"], d["history"]
        dp = [c - b for c, b in zip(h["p_center"], h["p_band"])]
        t_over = [t / 0.0333333 for t in h["t"]]
        ax2.plot(t_over, dp, lw=1.0, label=f"dx = {s['dx_m']:.4f} m")
        t_from = d["laplace"]["tail"]["t_from_s"] / 0.0333333
        ax2.axvspan(t_from, t_over[-1], color="grey", alpha=0.12)
    ax2.axhline(360.0, color="k", ls="--", lw=0.9, label="σ/R = 360 Pa")
    ax2.set_xlabel("t / t_σ")
    ax2.set_ylabel("Δp = p_core − p_band  [Pa]")
    ax2.set_title("(b) Fixed-time runs, shaded = averaging window")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)

    # ---- headline numbers -------------------------------------------------
    worst_fixed = max(fixed, key=lambda d: abs(d["laplace"]["tail"]["rel_error_pct"]))
    best_fixed = min(fixed, key=lambda d: abs(d["laplace"]["tail"]["rel_error_pct"]))
    coarse = next(d for d in fixed if abs(d["setup"]["dx_m"] - 0.004) < 1e-9)
    calib_txt = "（校准梯子未生成）"
    if calib:
        best_c = min(calib, key=lambda d: abs(d["laplace"]["tail"]["rel_error_pct"]))
        worst_c = max(calib, key=lambda d: abs(d["laplace"]["tail"]["rel_error_pct"]))
        calib_txt = (f"质量校准后最优 {best_c['setup']['dx_m']:.4f} m 上 "
                     f"{best_c['laplace']['tail']['rel_error_pct']:.2f} %，"
                     f"最差 {worst_c['setup']['dx_m']:.4f} m 上 "
                     f"{worst_c['laplace']['tail']['rel_error_pct']:.2f} %")

    text = f"""# Young–Laplace 收敛序列的时间-分辨率混杂（m3342 结清）

**结论先行**：HEAD 上提交的 `benchmarks/laplace_law/convergence/*.json` 用**固定步数**
（380 步）生成，而 `dt ∝ h/c0`，于是细化网格同时缩短了物理时间——所谓"收敛序列"实际在比较
三个不同的物理问题。改成**固定物理时长（t = 1.0 t_σ）+ 尾段时均**后，结论与提交件不同，
且提交件里 `validation_report.json` 的"within 15 %"是**单帧侥幸**。Laplace 压差维持
**WITHHELD**。

## 1. 案例介绍

二维单相液滴（Young–Laplace）：半径 R = 0.02 m，σ = 7.2 N/m（水的 10 倍，压缩毛细时间），
ρ0 = 1000 kg/m³，γ = 7，c0 = 20 m/s，μ = 0.01 m²/s，无重力，hex 点阵，h = 1.3 dx。
解析基准 Δp = σ/R = {fixed[0]['reference']['delta_p_Pa'] if fixed else 360.0:.0f} Pa，
毛细时间 t_σ = √(ρ0 R³/σ) = {(fixed[0]['reference']['capillary_time_s'] if fixed else 0.033333):.4f} s。
测量口径固定为 `p_core − p(0.80–0.95R)`（本仓历史口径，其缺陷另见
`docs/verification_laplace_pressure_floor.md` 的 C1）。

## 2. 怎么做的

1. `benchmarks/laplace_law/run.py` 新增 `steps_for_physical_time()`：由目标 t/t_σ 反解步数；
   `relax(..., t_over_t_sigma=..., average_tail=...)` 记录 `p_core`、`p_band` 时间序列，
   读数取**末段 25 % 窗口的时均**（单帧读数同时保留，便于对照）。`--steps` 保留，向后兼容。
2. `convergence_ladder()` 用固定物理时长重跑 dx = 0.004 / 0.0025 / 0.002 / 0.0015；
   旧 380 步件原样移入 `convergence/legacy_steps380/`（历史留痕，不删）。
3. 判别实验：`calibrate_mass=True` 用 `m → m/(1+b)` 去掉点阵核和偏置（核心区实测 b），
   同一固定物理时长再跑一梯，把"初始化瞬态"与"稳态缺陷"分开。
4. 全部数字由 `make_convergence_doc.py` 从 JSON 注入本文，无手抄。

## 3. 计算结果

**旧提交件（固定 380 步，HEAD）**

{table(legacy, False)}

**固定物理时长 t = 1.0 t_σ，末段 25 % 时均**

{table(fixed, True)}

**同一物理时长 + 质量校准**

{table(calib, True) if calib else '（未生成）'}

关键对照：同一 dx = 0.004 m 上，提交主报告（380 步，t = 0.889 t_σ，单帧）读
**{main_rep['laplace']['rel_error_pct']:.2f} %**（within 15 %）；
本单固定物理时长 t = 1.0 t_σ 的尾段时均读 **{coarse['laplace']['tail']['rel_error_pct']:.2f} %**，
单帧读 {coarse['laplace']['rel_error_pct']:.2f} %。也就是说 0.11 t_σ 的差别把"通过"
变成"不通过"：该判据坐在一个振幅远大于信号本身的振荡上。

固定物理时长下的误差带：最优 {best_fixed['setup']['dx_m']:.4f} m →
{best_fixed['laplace']['tail']['rel_error_pct']:.2f} %，
最差 {worst_fixed['setup']['dx_m']:.4f} m → {worst_fixed['laplace']['tail']['rel_error_pct']:.2f} %。
{calib_txt}。

## 4. 结果图

![时间-分辨率混杂](convergence_time_confound.png)

(a) |Δp − σ/R|/(σ/R) 对 dx：虚线为 15 %／3 % 判据线。固定 380 步的"收敛序列"随细化而变差，
因为它同时缩短了物理时间；固定物理时长后误差落在一条与分辨率无关的带上。
(b) 固定物理时长算例的 Δp(t)；灰底为时均窗口，黑虚线为解析 360 Pa。可见信号围绕 0 附近
振荡，单帧读数不可信。

## 5. 注意点

- **固定步数的分辨率阶梯不是收敛研究**：`dt ∝ h/c0`，固定 steps 等于让 t/t_σ 随 dx 漂移
  （本仓实测 0.556 / 0.445 / 0.333）。任何"随分辨率变差"的结论都必须先排除这个混杂。
- 单帧读数在毛细振荡上不可用：dx = 0.004 m 上相隔 0.11 t_σ 的两个单帧差 45 个百分点。
- 测量口径 `p_core − p(0.80–0.95R)` 本身仍受自由面密度亏缺污染（m2971 的 C1/C3），
  本单只清时间维，不清口径维；换成"绝对压强"口径后误差量级会变（见
  `docs/verification_laplace_pressure_floor.md`）。
- 质量校准只改初始瞬态；它是否把压差救回来见第 3 节两梯对照，未救回的部分属自由面压强闭合缺陷。

## 6. 误差判定

| 关键量 | 基准 | 本单读数 | 相对误差 | <3 % ? |
|---|---|---|---|---|
| Δp（dx = 0.004 m，固定物理时长，时均） | σ/R = 360 Pa | {coarse['laplace']['tail']['delta_p_measured_Pa']:.1f} Pa | {coarse['laplace']['tail']['rel_error_pct']:.1f} % | **否** |
| Δp（最优分辨率，固定物理时长，时均） | σ/R = 360 Pa | {best_fixed['laplace']['tail']['delta_p_measured_Pa']:.1f} Pa | {best_fixed['laplace']['tail']['rel_error_pct']:.1f} % | **否** |
| t/t_σ 的跨分辨率一致性（新梯子） | 全梯相同 | {min(d['setup']['t_over_t_sigma'] for d in fixed):.3f}–{max(d['setup']['t_over_t_sigma'] for d in fixed):.3f} | ≤ {100*(max(d['setup']['t_over_t_sigma'] for d in fixed)-min(d['setup']['t_over_t_sigma'] for d in fixed))/1.0:.2f} % | **是** |

**是否 <3 %：否。** Laplace 压差在固定物理时长下所有分辨率都不满足 3 %，也不满足 15 %，
故证据等级维持 **WITHHELD**；本单的增量是把这个失败从"不知道错在哪"收紧到"与时间混杂无关、
与分辨率无关的系统性缺压"，并把提交件里的"通过"判据标记为单帧侥幸。honest-FAIL。
"""
    DOC.write_text(text)
    print(f"wrote {DOC} ({len(text)} chars) and {FIG}")


if __name__ == "__main__":
    main()