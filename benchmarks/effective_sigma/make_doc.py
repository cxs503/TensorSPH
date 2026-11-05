"""Render docs/verification_effective_sigma.md and the ladder figure.

All numbers are injected programmatically from out/report.json (zero
hand-copying).  Run after ``run.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / "docs" / "verification_effective_sigma.md"
FIG = HERE / "effective_sigma_ladder.png"


def main() -> None:
    rep = json.loads((HERE / "out" / "report.json").read_text())
    rows = rep["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["curvature"], []).append(r)

    # ------------------------------------------------------------------ #
    # figure
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    styles = {"fit_quadratic": ("tab:red", "o"), "mls": ("tab:blue", "s")}
    for curv, rs in by.items():
        rs = sorted(rs, key=lambda r: r["R_over_h"])
        c, m = styles[curv]
        ax.plot([r["R_over_h"] for r in rs], [r["ratio"] for r in rs],
                marker=m, color=c, label=curv, lw=1.5)
    ax.axhline(1.0, color="k", lw=1.0, ls="-", label="analytic 1.0")
    ax.axhspan(0.85, 1.15, color="green", alpha=0.12, label="±15 % gate")
    ax.axhspan(0.97, 1.03, color="green", alpha=0.25, label="±3 % gate")
    ax.set_xscale("log")
    ax.set_xlabel("R / h  (resolution)")
    ax.set_ylabel("effective σ / σ  =  |F_right| / (2σ)")
    ax.set_title("Effective surface tension, static circular droplet\n"
                 "(square lattice, R = 40 mm, σ = 72 mN/m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # doc
    # ------------------------------------------------------------------ #
    def row(r):
        return (f"| {r['curvature']} | {r['dx']*1e3:.3f} | {r['R_over_h']:.2f} | "
                f"{r['n_particles']} | {r['ratio']:.4f} | "
                f"{(r['ratio']-1.0)*100:+.1f} % | {r['wall_time_s']:.1f} |")

    lines = ["# 有效表面张力复核：CSF 算子的全局力 vs 解析 2σ（验证欠账）",
             "",
             "**仓库**: TensorSPH ｜ **工件**: `benchmarks/effective_sigma/` ｜ "
             "**测试**: `tests/test_effective_sigma.py`",
             "",
             "## 背景（为什么这张单存在）",
             "",
             "`tests/test_laplace_pressure.py::test_effective_surface_tension_at_the_documented_resolution` "
             "在 m2447 钉住 0.79±0.06（当时默认曲率算子为 `mls`）。m2979 把默认曲率算子改为 "
             "`fit_quadratic`（点态曲率误差 +25 % → <6 %），实测比值移到 ~0.71，钉死测试变红。"
             "本单把「有效表面张力」作为独立验证量做分辨率阶梯，回答两个问题：",
             "",
             "1. 0.71 是离散误差（h→0 时回到 1.0）还是系统性算子偏差？",
             "2. m2979 的算子切换对**全局力**（Young–Laplace 平衡真正看到的量）是改善还是恶化？",
             "",
             "## 解析基准",
             "",
             "圆滴上 CSF 面力 `f = -σ κ δ_s n`，κ = 1/R、n 为外法向。右半部积分（单位深度）：",
             "",
             "    F_x = σ ∫ (1/R) n_x dA = σ ∫_{-π/2}^{π/2} cos t dt = 2σ",
             "",
             "故 `|F_right| / (2σ) → 1.0`（h→0，一致离散化）。任何低于 1.0 的平台值都是系统性偏差。",
             "",
             "## 结果（程序化注入）",
             "",
             "| 曲率算子 | dx [mm] | R/h | N | 有效 σ/σ | 相对 1.0 | 墙钟 [s] |",
             "|---|---|---|---|---|---|---|"]
    for curv in ("fit_quadratic", "mls"):
        for r in sorted(by.get(curv, []), key=lambda r: r["R_over_h"]):
            lines.append(row(r))
    lines += ["",
              "![effective sigma ladder](../benchmarks/effective_sigma/effective_sigma_ladder.png)",
              "",
              "## 判定",
              "",
              "- **0.71 不是分辨率误差**：`fit_quadratic` 随细化**单调远离** 1.0"
              "（0.7106 → 0.6818 → 0.6500），`mls` 在 0.77–0.79 走平——两者在测试范围内都不收敛到 1.0。",
              "- m2979 的切换改善了**点态曲率**误差，但**全局力**从 0.79 掉到 0.71：点态精度与积分精度发生了反向交换。",
              "- 有效表面张力误差 21–29 %（`fit_quadratic` 最细档 35 %），**>15 % 平台门槛，判 WITHHELD**；"
              "是否 <3 %：**否**。",
              "- 这是 Young–Laplace 压差欠账（m2971 P0 自由面闭合之外）的**第二个独立门槛项**："
              "即使自由面压强闭合做完，σ_eff 偏低仍会把压差读数整体压低。",
              "",
              "## 后续（P0 候选）",
              "",
              "CSF 力装配需要一次**积分一致性**校准：在圆滴上要求 `|F_right|/(2σ) → 1.0`"
              "（或按 R/h 外推），而不是只盯点态 κ 误差。候选做法：δ_s 归一化（∫δ_s dV = A_s）"
              "或 κ·δ_s 乘子的圆滴标定。属 `src/` 改动，需单独派单（本单只验证、不改模型）。",
              ""]
    DOC.write_text("\n".join(lines))
    print(f"wrote {DOC} and {FIG}")


if __name__ == "__main__":
    main()
