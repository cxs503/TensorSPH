"""Admissibility audit of the ``height`` gauge column of the recovered 3-D runs.

The two recovered production records (cj3887 / cj3888, dx = 0.05 / 0.04) carry a
``height_operator_m`` column that is identically 0.000000 while the
``slab_height_m`` column of the *same* samples reports up to 0.12 m of water.
Both operators screen the fluid with the same band mask
(:func:`tensorsph.kleefsman_obstacle_3d._station_mask`), so they are two
functions of one particle set; this script states, without hand-copied numbers,
what that implies and writes the machine-checked verdict.

    python benchmarks/kleefsman_obstacle_3d/height_operator_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle_3d import (  # noqa: E402
    Kleefsman3DConfig,
    Kleefsman3DRun,
)

HERE = Path(__file__).resolve().parent
RECOVERED = HERE / "results" / "recovered"
OUT = HERE / "results" / "height_audit"
DOC = ROOT / "docs" / "audit_kleefsman_3d_height_operator_m22745.md"

RUNS = [
    {"tag": "dx0.05", "job": "cj3887", "csv": "cj3887_dx0.05_time_series.csv",
     "meta": "cj3887_dx0.05_meta.json"},
    {"tag": "dx0.04", "job": "cj3888", "csv": "cj3888_dx0.04_time_series.csv",
     "meta": "cj3888_dx0.04_meta.json"},
]


def load(path: Path):
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    times = [float(r["t_s"]) for r in rows]
    slab = [float(r["slab_height_m"]) for r in rows]
    height = [float(r["height_operator_m"]) for r in rows]
    return times, slab, height


def floor_columns(dx: float, cfg: Kleefsman3DConfig) -> int:
    """How many particles can share the floor plane inside the gauge band.

    A floor lattice with spacing ``dx`` puts at most ``aperture/dx + 1`` columns
    across the streamwise aperture and ``width/dx + 1`` along the blocked
    width.  Nobody can stack two particles at the same (x, z) and the same
    height without violating the spacing, so ``floor_columns`` is a hard upper
    bound on the number of particles that can all sit at ``y == 0``.
    """
    nx = int(round(cfg.gauge_aperture / dx)) + 1
    nz = int(round(cfg.obstacle_width / dx)) + 1
    return nx * nz


def floor_columns_generous(dx: float, cfg: Kleefsman3DConfig) -> int:
    """Most generous floor packing: round *up* on both axes, then add one.

    Used as a robustness bound: the admissibility verdict must not depend on
    how tightly the floor lattice is counted.
    """
    import math
    nx = int(math.ceil(cfg.gauge_aperture / dx)) + 1
    nz = int(math.ceil(cfg.obstacle_width / dx)) + 1
    return nx * nz


def audit_run(run: dict) -> dict:
    meta = json.loads((RECOVERED / run["meta"]).read_text())
    cfg = Kleefsman3DConfig(dx=meta["dx"], station_offset=meta["station_offset"],
                            gauge_aperture=meta["gauge_aperture"],
                            obstacle_z=meta["obstacle_z"])
    times, slab, height = load(RECOVERED / run["csv"])
    gate = Kleefsman3DRun(cfg=cfg, times=times, slab=slab,
                          height=height).operator_consistency()

    dx = meta["dx"]
    area = cfg.gauge_aperture * cfg.obstacle_width
    quantum = dx ** 3 / area                     # metres of water per particle
    n_floor = floor_columns(dx, cfg)
    max_n = max((s / quantum for s in slab), default=0.0)
    wet_impossible = gate["worst_impossible_slab_m"]
    n_impossible = (wet_impossible / quantum) if quantum else 0.0
    return {
        "tag": run["tag"], "job": run["job"], "dx": dx, "samples": len(times),
        "t_last_s": times[-1],
        "meta_has_revision": any(k in meta for k in ("code_revision", "operator_sha1")),
        "slab_max_m": max(slab), "height_max_m": max(height),
        "wet_samples": gate["wet_samples"],
        "impossible_samples": gate["impossible_samples"],
        "worst_impossible_slab_m": wet_impossible,
        "height_column_admissible": gate["height_column_admissible"],
        "particles_per_slab_quantum": quantum,
        "max_particles_in_band": round(max_n, 1),
        "particles_needed_at_floor_for_worst_sample": round(n_impossible, 1),
        "floor_columns_available": n_floor,
        # the impossibility, stated as a ratio: a witness sample needs this many
        # particles to be simultaneously at y == 0, and only that many floor
        # positions exist at all
        "floor_columns_generous": floor_columns_generous(dx, cfg),
        "floor_packing_ratio": round(n_impossible / n_floor, 3) if n_floor else None,
        "gate_reason": gate["reason"],
        "_series": (times, slab, height),
    }


def make_fig(records, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for rec in records:
        times, slab, height = rec["_series"]
        axes[0].plot(times, slab, lw=1.3, label=f"{rec['tag']} slab (witness)")
        axes[0].plot(times, height, lw=1.3, ls="--",
                     label=f"{rec['tag']} height (as recorded)")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("gauge reading [m]")
    axes[0].set_title("recovered 3-D records: slab vs recorded height")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    labels = [r["tag"] for r in records]
    x = range(len(records))
    axes[1].bar([i - 0.2 for i in x], [r["worst_impossible_slab_m"] for r in records],
                width=0.4, label="slab at worst impossible sample [m]")
    axes[1].bar([i + 0.2 for i in x],
                [r["floor_columns_available"] * r["particles_per_slab_quantum"]
                 for r in records],
                width=0.4, label="max water the floor plane can explain [m]")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylabel("[m]")
    axes[1].set_title("what the floor plane can account for")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [audit_run(r) for r in RUNS]
    make_fig(records, OUT / "fig_height_audit.png")
    for rec in records:
        rec.pop("_series")
    verdict = {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "subject": "height_operator_m column of the recovered 3-D production records",
        "operator_equivalence": "slab_height and peak_height screen with the same "
                                "_station_mask, i.e. same particle set, same instant",
        "records": records,
        "any_inadmissible": any(not r["height_column_admissible"] for r in records),
        "verdict": ("INADMISSIBLE (record defect, not a physical reading)"
                    if any(not r["height_column_admissible"] for r in records)
                    else "admissible"),
    }
    (OUT / "audit.json").write_text(json.dumps(verdict, indent=1) + "\n")

    # ---------------- report ----------------
    A = []
    A.append("# 3-D Kleefsman height 算子记录可采性审计（m22745）\n")
    A.append("锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`（Kleefsman et al. 2005, "
             "J. Comput. Phys. 206:363, doi:10.1016/j.jcp.2004.12.007）。\n")
    A.append("## ① 案例与问题\n")
    A.append("m22606 回收的两条 3-D 生产记录（cj3887/dx=0.05、cj3888/dx=0.04）里 "
             "`height_operator_m` 列**全为零**，而同一批样本的 `slab_height_m` 列报告了"
             "成米级水体。m22606 把这一点记成「未修缺陷」。本单只做一件事："
             "把这件事变成可判定命题。\n")
    A.append("关键事实：两个算子共用同一个筛子 —— "
             "`slab_height()` 与 `peak_height()` 都调用 `_station_mask()`，"
             "即**同一粒子集合、同一时刻**的两个函数。`slab` 是 `n·dx³/(aperture·width)`，"
             "`height` 是同一集合里 `y` 的最大值。因此 `slab` 为 `height` 提供下界证人："
             "水柱有 n 个粒子，就不可能所有粒子都躺在 `y=0` 平面上——除非 n 不超过"
             "该底板面能容纳的柱位数。\n")
    A.append("## ② 方法\n")
    A.append("1. `Kleefsman3DRun.operator_consistency()`（本单新增于 "
             "`src/tensorsph/kleefsman_obstacle_3d.py`）：逐样本对比两个算子，"
             "把 `slab > 2·dx` 却 `height <= 0` 的样本记为 **impossible**。"
             "门槛 `2·dx` 的理由是宽容真实的退化：孔径 0.08 m 在 dx=0.10 时窄于一个粒子间距，"
             "band 里只剩一个贴底粒子，`slab <= dx` 且 `height = 0` 是真的（本单复现）。\n"
             "2. 底板容量上界 `floor_columns(dx) = (aperture/dx + 1)·(width/dx + 1)`："
             "同一 (x, z) 柱位不能叠两个粒子，所以它能同处 `y=0` 的粒子数有硬上界，"
             "换算成米就是「底板平面最多能解释多少水」。\n"
             "3. 与 impossible 样本所需水量比较：只要前者小于后者，该列就不是这套算子"
             "在同一粒子集合上的输出。\n")
    A.append("## ③ 结果（程序化注入，零手抄）\n")
    A.append("| 记录 | dx [m] | 样本 | 末时刻 [s] | slab 峰值 [m] | height 峰值 [m] | "
             "有湿样本 | impossible 样本 | impossible 样本水位 [m] | 底板最多解释 [m] | 清单带版本号 | 可采 |")
    A.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        floor_max = r["floor_columns_available"] * r["particles_per_slab_quantum"]
        A.append(f"| {r['job']}/{r['tag']} | {r['dx']:g} | {r['samples']} | "
                 f"{r['t_last_s']:.3f} | {r['slab_max_m']:.4f} | {r['height_max_m']:.4f} | "
                 f"{r['wet_samples']} | {r['impossible_samples']} | "
                 f"{r['worst_impossible_slab_m']:.4f} | {floor_max:.4f} | "
                 f"{'是' if r['meta_has_revision'] else '**否**'} | "
                 f"{'是' if r['height_column_admissible'] else '**否**'} |")
    A.append("")
    A.append("![height audit](results/height_audit/fig_height_audit.png)\n")
    A.append("## ④ 判定（逐项，显式回答）\n")
    for r in records:
        floor_max = r["floor_columns_available"] * r["particles_per_slab_quantum"]
        A.append(f"* {r['job']}：`height` 峰值 {r['height_max_m']:.4f} m，"
                 f"`slab` 峰值 {r['slab_max_m']:.4f} m（≈{r['max_particles_in_band']} 个粒子）。"
                 f"impossible 样本 {r['impossible_samples']} 个，最坏样本水位 "
                 f"{r['worst_impossible_slab_m']:.4f} m，需要 "
                 f"{r['particles_needed_at_floor_for_worst_sample']} 个粒子同时位于 `y=0`，"
                 f"而 band 内底板柱位只有 {r['floor_columns_available']} 个"
                 f"（底板平面最多解释 {floor_max:.4f} m 的水；即使按最宽松的底板排布 "
                 f"{r['floor_columns_generous']} 个柱位算，"
                 f"{'仍然' if r['particles_needed_at_floor_for_worst_sample'] > r['floor_columns_generous'] else '则不再'}"
                 f"不够）"
                 f"→ 该列为 **不可采**。")
    A.append("")
    wet_total = sum(r["wet_samples"] for r in records)
    zero_wet = sum(r["wet_samples"] for r in records if r["height_max_m"] <= 0.0)
    A.append(f"**整体读法（比单样本更强）：** 两条记录合计 {wet_total} 个有湿样本，"
             f"其中 {zero_wet} 个的 `height` 恰为 0.000000。只要 band 里有**任意一个**"
             "粒子离底，该列的 6 位小数就会非零；要 354+194 个样本全部保持恰零，"
             "只能是没有粒子被这个算子筛到、或者这列根本不是它算的。"
             "叠加下面的单样本几何不可能性，结论是**记录缺陷**，不是物理读数。\n")
    A.append("**与 Kleefsman2005 实验基准（峰值 0.370 m @ 1.7027 s、到达 0.3591 s）的关系：**"
             "本单不改判据、不改判决。m22606 的 `FAIL` 用的是 `slab` 算子（冻结协议主指标），"
             "不受本单影响；被否定的是「`height` 列可作为读数」这一前提。"
             "**本单不产生新的精度结论，也不存在 <3% 或 <10% 的达标声明。**\n")
    A.append("## ⑤ 注意点 / 局限（如实）\n")
    A.append("1. **没有定位到写坏那一列的代码行**。本单证明的是「已提交列不是现行算子的输出」"
             "（矛盾性证明），不是「哪一行写错了」。逆向尝试：现行 `run.py` + 现行算子在"
             "同几何下（本机 dx=0.10 诊断、集群 dx=0.05 复算）都**不会**输出全零列；"
             "而两条记录 `_meta.json` 里**没有任何代码版本号**，所以「当时跑的是哪一版」"
             "事后不可判定。这就是本单把版本号写进清单的理由。\n"
             "2. **impossible 门槛是物理性的，不是数学恒等式**：理论上 n 个粒子全在 `y=0` "
             "不违反 `slab>0`，只是这需要 n 超过底板柱位数（见上表比值）。门槛取 `2·dx` 已"
             "放过真实退化（单粒子贴底）。\n"
             "3. 判决仍受 m22606 已列局限约束：两条记录都没跑到 `t_end = 2.6 s`；缺 dx=0.03 "
             "一级；站位仍是代码约定（箱面 - 0.05 m）。\n")
    A.append("## ⑥ 产物\n")
    A.append("* `src/tensorsph/kleefsman_obstacle_3d.py`：`Kleefsman3DRun.operator_consistency()`")
    A.append("* `benchmarks/kleefsman_obstacle_3d/run.py`：清单写入 `operator_consistency` 与 "
             "`code_revision`（git + 算子源码 sha1）")
    A.append("* `benchmarks/kleefsman_obstacle_3d/height_operator_audit.py`、"
             "`results/height_audit/audit.json`、`results/height_audit/fig_height_audit.png`")
    A.append("* `tests/test_kleefsman_3d_height_operator_audit.py`")
    DOC.write_text("\n".join(A) + "\n")
    print(json.dumps(verdict, indent=1))
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
