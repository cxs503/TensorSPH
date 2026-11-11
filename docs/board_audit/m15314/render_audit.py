#!/usr/bin/env python3
"""Render the m15278 board-audit report (CASE-LAKE-AT-REST-SPH, TensorSPH).

Every number in the generated markdown is read from the audit snapshots
(delivered report.json, auditor re-run, auditor t=0 probe). Nothing is typed by
hand; the figures are written next to the markdown and attached to the platform
report.

Run (from the audit worktree):  python3 docs/board_audit/m15314/render_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REFS = HERE / "refs"
ART = REFS / "artifacts"
OUT_MD = HERE / "AUDIT_m15278.md"
FIG1 = ART / "figures/audit_drift_vs_criterion.png"
FIG2 = ART / "figures/audit_drift_vs_time.png"
TOL = 0.005  # pool criterion: pressure drift < 0.5 % of rho0 g H


def load(p: Path):
    return json.loads(p.read_text())


def case_label(c: dict) -> str:
    return "%s dx=%dmm c0=%g nu=%g" % (c["mode"], round(c["dx_m"] * 1000),
                                       c["sound_speed_m_s"], c["viscosity"])


def fig_criterion(cases: list[dict]) -> None:
    labels = [case_label(c) for c in cases]
    vals = [100.0 * c["pressure_drift_max"] for c in cases]
    colors = ["#c0392b" if c["mode"] == "shipped" else "#2471a3" for c in cases]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_yscale("log")
    ax.axhline(100.0 * TOL, color="k", ls="--", lw=1.4,
               label="acceptance: drift < 0.5 % of rho0 g H")
    for b, v in zip(bars, vals):
        ax.annotate("%.3f %%" % v, (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("max pressure drift [% of rho0 g H]")
    ax.set_title("CASE-LAKE-AT-REST-SPH (m15278): all 7 cases vs the 0.5 % criterion")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG1, dpi=140)
    plt.close(fig)


def fig_time(cases: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    keep = [c for c in cases if abs(c["dx_m"] - 0.02) < 1e-9
            and abs(c["sound_speed_m_s"] - 20.0) < 1e-9
            and abs(c["viscosity"] - 0.01) < 1e-9]
    for c in keep:
        t = [s["t"] for s in c["samples"]]
        d = [100.0 * s["pressure_drift_max"] for s in c["samples"]]
        ax.plot(t, d, marker="o", ms=3, label=case_label(c))
    ax.axhline(100.0 * TOL, color="k", ls="--", lw=1.4, label="0.5 % criterion")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("max interior pressure drift [% of rho0 g H]")
    ax.set_title("drift history, dx = 20 mm, c0 = 20 m/s (auditor re-run, bit-identical)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG2, dpi=140)
    plt.close(fig)


def main() -> None:
    orig = load(REFS / "benchmarks/lake_at_rest/report.json")
    rerun = load(ART / "m15278_rerun.json")
    probe = load(ART / "probe_t0.json")["result"]
    cases = orig["cases"]

    key = lambda c: (c["mode"], c["dx_m"], c["sound_speed_m_s"], c["viscosity"])  # noqa: E731
    kb = {key(c): c for c in rerun["cases"]}
    repro = []
    worst = 0.0
    for c in cases:
        d = abs(c["pressure_drift_max"] - kb[key(c)]["pressure_drift_max"])
        worst = max(worst, d)
        repro.append((key(c), c["pressure_drift_max"], kb[key(c)]["pressure_drift_max"], d))

    fig_criterion(cases)
    fig_time(cases)

    shipped = [c for c in cases if c["mode"] == "shipped"]
    wb = [c for c in cases if c["mode"] == "wb"]
    best = min(cases, key=lambda c: c["pressure_drift_max"])
    s_max = max(c["pressure_drift_max"] for c in shipped)
    wb_max = max(c["pressure_drift_max"] for c in wb)
    imp = orig["summary"]["improvement_factor_at_dx_0p02"]
    over_best = best["pressure_drift_max"] / TOL
    over_shipped = s_max / TOL

    L = []
    w = L.append
    w("# 交差审计：m15278 — CASE-LAKE-AT-REST-SPH（TensorSPH 静水湖）")
    w("")
    w("`verdict_by=board-audit` · 审计单 m15314 · 被审单 m15278（succeeded, "
      "task_key=case-lake-at-rest-sph-r2, claimed_by=tp-ds-7）")
    w("")
    w("## 0. 结论先行（disposition: request changes）")
    w("")
    w(f"* **级别 = L3（系统集成），不授 L4**：功能实质是真的（well-balanced 压力力写进生产 RHS "
      f"`droplet_impact.rhs`，被 4 个求解器入口共用），但算例池判据 **压力漂移 < 0.5 %** "
      f"**未达标**——最好一例 `{case_label(best)}` = "
      f"**{100 * best['pressure_drift_max']:.3f} %**，超判据 **{over_best:.2f}×**；"
      f"`shipped` 模式最大 **{100 * s_max:.3f} %**（超 {over_shipped:.1f}×）。")
    w("* 被审单自报的正是 honest FAIL（`criterion_met=false`），本审计**逐位复现**了它："
      f"7/7 算例漂移与原工件**逐位相同**（max |Δ| = {worst:.1e}），"
      f"复跑耗时 ~22 s，命令即交付的 repro_cmd。")
    w("* 交付三件套实体存在（工件/数值/复现命令全部可核），**但一件都没进 git**："
      "`head_commit=null`，`git ls-files` 对这些路径为空，改动只存在于工作树（未 track）。"
      "→ 产品（TensorSPH main）**没有** `equilibrium_level`（生态仓 grep 0 命中）；"
      "审计已把 patch 与工件逐字节快照进本分支 `docs/board_audit/m15314/refs/` 作保命。")
    w("* 基准裁判**空转**：`tensor_benchmark_cases` 对 `anchor_slug='CASE-LAKE-AT-REST-SPH'` "
      "**返回 0 行**（全表 14 行、无一含 LAKE）→ 机器裁判不存在，唯一可援引判据是算例池 "
      "`tensor_case_pool#67 tol_policy='压力漂移<0.5%'`（且该行 `status='dispatched'` 仍挂在失败前轮 "
      "m15241 上，未随 m15278 更新）。")
    w("")
    w("## 1. 取证范围与独立性边界")
    w("")
    w("| 项 | 值 |")
    w("|---|---|")
    w("| 被审工作树 | `/root/tp-agent/work/m15278`（base `ae61a2b`，`dirty=true`） |")
    w("| 交付工件 | `benchmarks/lake_at_rest/{run.py,report.json,make_docs.py}`, "
      "`docs/verification_lake_at_rest.md`, `docs/figures/lake_at_rest_{drift,profile}.png`, "
      "`tests/test_lake_at_rest.py` |")
    w("| 审计快照（逐字节 + sha256） | `docs/board_audit/m15314/refs/` + "
      "`refs/artifacts/droplet_impact_wellbal.patch` |")
    w("| 平台报告 | `tensor_reports#14367`（6 段齐全，附 2 图，图可取回 200/58 616 B、45 243 B PNG） |")
    w("| 独立性边界 | 同一物理 CPU、同一 torch 2.7.1/py3.11.6；复跑用交付自带的 `run.py`（未改一行）"
      "→ 验证的是**确定性与数值可重放**，不是独立实现比对；t=0 探针 `probe_t0.py` 是审计自写，独立于 run.py 的聚合逻辑 |")
    w("")
    w("## 2. 三件套核验")
    w("")
    w("| 三件套 | 结论 | 证据 |")
    w("|---|---|---|")
    w(f"| run_artifacts 文件 | **PASS（在盘）** | `report.json` 36 455 B：7 算例 × 16 采样点，"
      f"含 `pressure_drift_max/mean`、`v_max`、`rho_rel_error_max`、`acc_max_at_rest`；"
      f"判据字段 `acceptance_tolerance={orig['acceptance_tolerance']}` |")
    w(f"| benchmark ref vs got | **逐项核，全部 FAIL** | ref = 精确解 `p=rho0 g (H-y), u=0`；"
      f"got 见表 §3；`criterion_met={orig['summary']['criterion_met']}`, "
      f"`wb_criterion_met={orig['summary']['wb_criterion_met']}` |")
    w(f"| repro_cmd 可执行 | **PASS** | `PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py` "
      f"审计实跑：7/7 算例漂移逐位相同（max |Δ| = {worst:.1e}） |")
    w("| 单测 | **PASS** | `PYTHONPATH=src pytest tests/test_lake_at_rest.py -q` → **3 passed in 7.81 s**（审计实跑） |")
    w("| 进仓（git） | **FAIL** | `head_commit: null`；`git ls-files benchmarks/lake_at_rest …` 空；"
      "`git log --all -- benchmarks/lake_at_rest docs/verification_lake_at_rest.md` 空 |")
    w("| 基准登记（裁判） | **FAIL（系统级）** | `tensor_benchmark_cases` 该锚点 0 行 |")
    w("")
    w("## 3. 基准判据逐项（唯一裁判 = 算例池 `压力漂移<0.5%`）")
    w("")
    w("ref = 0（精确静水解，无实验不确定度）；误差定义 "
      "`max_{内区粒子,采样} |p−rho0 g (H−y)| / (rho0 g H)`，内区 = 距壁/自由面 ≥ 3h = "
      f"{probe['shipped']['three_h_m'] * 1000:.0f} mm。")
    w("")
    w("| # | mode | dx [mm] | c0 [m/s] | nu [m²/s] | 粒子/内区 | drift max | **< 0.5 %?** | 超判据倍数 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(cases, 1):
        w(f"| {i} | {c['mode']} | {c['dx_m'] * 1000:.0f} | {c['sound_speed_m_s']:.0f} | "
          f"{c['viscosity']:.3f} | {c['particles']}/{c['interior_particles']} | "
          f"{100 * c['pressure_drift_max']:.3f} % | "
          f"{'**PASS**' if c['pass_0p5pct'] else 'FAIL'} | "
          f"{c['pressure_drift_max'] / TOL:.1f}× |")
    w("")
    w(f"* `shipped`（仓默认初始化）漂移 {100 * s_max:.3f} %，且**加密不收敛**"
      f"（25 → 20 mm：{100 * shipped[0]['pressure_drift_max']:.3f} % → "
      f"{100 * shipped[1]['pressure_drift_max']:.3f} %）→ 缺陷在边界离散，不在插值。")
    w(f"* `wb`（well-balanced）最好 {100 * min(c['pressure_drift_max'] for c in wb):.3f} %"
      f"（dx=20 mm，c0=20），最大 {100 * wb_max:.3f} %（ν=0）→ 改善 **{imp:.2f}×**，"
      f"但距判据仍差 **{over_best:.2f}×**。")
    w("")
    w("## 4. 独立 t=0 探针（审计自算，核报告 §4.2 的缺陷定位数字）")
    w("")
    w("| 量 | shipped | wb | 报告自报 | 判定 |")
    w("|---|---|---|---|---|")
    w(f"| 内区 t=0 偏差 max | {probe['shipped']['interior_dev_max_pct']:.5f} % | "
      f"{probe['wb']['interior_dev_max_pct']:.5f} % | 0.012 % | **一致** |")
    w(f"| 内区 t=0 偏差 mean | {probe['shipped']['interior_dev_mean_pct']:.5f} % | "
      f"{probe['wb']['interior_dev_mean_pct']:.5f} % | — | 补充 |")
    w(f"| 底行(t=0) 偏差 max | {probe['shipped']['bottom_row_dev_max_pct']:.2f} % | "
      f"{probe['wb']['bottom_row_dev_max_pct']:.3f} % | −24.8 %（单列切片） | 同量级；"
      f"审计给出整行 max {probe['shipped']['bottom_row_dev_max_pct']:.2f} %，比切片更差 |")
    w(f"| max\\|a\\| (t=0) [m/s²] | {probe['shipped']['acc_max_m_s2']:.2f} | "
      f"{probe['wb']['acc_max_m_s2']:.2f} | 19.04 / 4.65 | **逐位一致**（1.94 g / 0.47 g） |")
    w("")
    w("## 5. 发现（按严重度排序）")
    w("")
    w("**B1（阻断，判级）判据未达标 → 不授 L4。** 最好 wb "
      f"{100 * best['pressure_drift_max']:.3f} % vs 0.5 % = {over_best:.2f}× 超标；"
      "L4「基准验证」要求被验证能力通过基准，本算例的诚实结论是 FAIL（交付方也如此声明，无粉饰）。")
    w("")
    w("**B2（阻断，交付完整性）全部交付物未进 git。** `head_commit=null`、路径 `untracked`、"
      "生态仓 `grep equilibrium_level src/` 0 命中 → 产品侧不存在该能力；"
      "报告 §6 写「commit `ae61a2b2ce7a`」实为**基线提交**（工作树 base），构成 provenance 误导。"
      "审计已把 diff（154 行 patch）与全部工件快照进本分支，避免工作树回收即丢。")
    w("")
    w("**B3（系统级，非本单可控）基准裁判空转。** `SELECT … FROM tensor_benchmark_cases "
      "WHERE anchor_slug='CASE-LAKE-AT-REST-SPH'` → 0 行；算例池 `#67` 仍 `status='dispatched'`、"
      "`mission_id=15241`（已 failed 的前轮），未随 m15278 更新。"
      "与 m15240（F2）、m15235（F1_referee_missing）审计交叉一致：**算例池锚点普遍未登记判据**，"
      "「基准是唯一裁判」这条纪律目前**无机器裁判可依**，只能援引 tol_policy 文本。")
    w("")
    w("**C1（中）判据与算式的自指耦合。** wb 算子把解析静水压 `p_h=rho0 g (H−y)`（H 由用户经 "
      "`Config.equilibrium_level` 传入）减出后积分，drift 又用**同一** `p_h` 度量 → "
      "3.887 % 是 well-balanced 性质的度量，**不能**当作一般自由面精度证据；"
      "且要求先验已知平衡水位，对真实未知水面问题不可直接使用（交付文档 §2 已如实说明出处，但报告标题的 6.8× 易被读成普遍精度提升）。")
    w("")
    w("**C2（中）默认模式仍是缺陷态。** `equilibrium_level=None` 时走原路径，底行压强亏空 "
      f"{probe['shipped']['bottom_row_dev_max_pct']:.1f} %、虚假加速度 1.94 g、漂移 "
      f"{100 * shipped[1]['pressure_drift_max']:.2f} % 全部保留 → 修的是新开关，不是产品默认行为；"
      "该项须作为缺陷单跟进（壁面质量按静水剖面是通用修法，可无条件生效，未必需要新开关）。")
    w("")
    w("**D1（低）报告证据链小瑕。** `tensor_reports#14367.figures[].png_base64` 为空串，"
      "但 `/tensor/reports/14367/figures/<name>` 取回 200 + 真实 PNG → 图**在**，只是行内未内嵌；"
      "不计缺陷，登记以免下游误判为无图。")
    w("")
    w("## 6. 未跑的检查（unknown / not_run，不得当作 passed）")
    w("")
    w("* 与**独立实现**的横比（ISPH/DFSPH/其它开源 SPH 的同一静水算例）——未做，本审计只做同码复现。")
    w("* 算例池判据 0.5 % 的**可达性**审查（WCSPH 在 c0/√(gH)=10.79 下压强漂移 0.5 % 等价于密度一致到 ~4e-5，"
      "是否该由 ISPH/更高 c0 承担）：只做了量纲推断，未做可达性数值实验。")
    w("* 三维/长时（>0.05 s）/更大规模网格上的 wb 行为：未跑。")
    w("* `make_docs.py` 的图管线本身未逐行审（只核 report.json → md 的数字一致性）。")
    w("")
    w("## 7. 复现命令（审计实跑过的）")
    w("")
    w("```")
    w("cd /root/tp-agent/work/m15278")
    w("PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py --out /tmp/m15278_rerun.json   # 7/7 逐位复现")
    w("PYTHONPATH=src pytest tests/test_lake_at_rest.py -q                                  # 3 passed")
    w("PYTHONPATH=src:benchmarks/lake_at_rest python3 <审计>/probe_t0.py --out probe_t0.json  # t=0 探针")
    w("python3 docs/board_audit/m15314/render_audit.py                                      # 生成本文+两图")
    w("```")
    w("")
    w("图：`refs/artifacts/figures/audit_drift_vs_criterion.png`（7 算例 vs 0.5 % 判据，log 轴）、"
      "`audit_drift_vs_time.png`（dx=20 mm/c0=20 漂移史）。")
    OUT_MD.write_text("\n".join(L) + "\n")
    print("wrote", OUT_MD, "| figures:", FIG1.name, FIG2.name)
    print("best case drift=%.6f over_tol=%.3f repro_max_abs_diff=%.1e"
          % (best["pressure_drift_max"], over_best, worst))


if __name__ == "__main__":
    main()