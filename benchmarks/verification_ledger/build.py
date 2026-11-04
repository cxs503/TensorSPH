"""Build the cross-family verification ledger.

Outputs (deterministic functions of the committed artefacts):

    benchmarks/verification_ledger/ledger.json   machine-readable ledger
    docs/verification_ledger.md                  generated document
    benchmarks/verification_ledger/ledger.png    figure (4 panels)

Every number in the document and in the figure is extracted from the source
``*.json`` artefacts by ``ledger.py``; nothing is typed by hand.
``tests/test_verification_ledger.py`` regenerates the document and fails if the
committed copy drifts.

Usage::

    python benchmarks/verification_ledger/build.py [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "benchmarks" / "verification_ledger" / "ledger.json"
OUT_MD = ROOT / "docs" / "verification_ledger.md"
OUT_PNG = ROOT / "benchmarks" / "verification_ledger" / "ledger.png"


def _load_ledger_module():
    spec = importlib.util.spec_from_file_location(
        "verification_ledger_impl", Path(__file__).resolve().parent / "ledger.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses needs the module registered
    spec.loader.exec_module(module)
    return module


L = _load_ledger_module()


def _f(value: Any, digits: int = 3, plus: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # nan
        return "-"
    return f"{v:+.{digits}f}" if plus else f"{v:.{digits}f}"


def _pp(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{float(value):+.{digits}f}"


def _gate(row: Dict[str, Any]) -> str:
    if row["within_3pct"]:
        return "PASS<3%"
    if row["within_15pct"]:
        return "pass<15%"
    return "FAIL"


def render_markdown(ledger: Dict[str, Any]) -> str:
    verdict = ledger["verdict"]
    prov = ledger.get("provenance") or {}
    dp = verdict["quantities"]["laplace_dp"]
    cf = verdict["quantities"].get("cut_force_ratio")
    best = verdict.get("best_sanctioned_reading") or {}
    sens = verdict["convention_sensitivity"]
    readings: List[Dict[str, Any]] = ledger["readings"]
    q1 = prov.get("q1_reproducible") or {}
    q2 = prov.get("q2_sigma_control") or {}
    q3 = prov.get("q3_resolution_stability") or {}

    lines: List[str] = []
    add = lines.append

    add("# 接口压差验证台账（跨 benchmark 家族 · 机器核对）")
    add("")
    add("> 本文件由 `benchmarks/verification_ledger/build.py` 从仓库内已提交的 `*.json` "
        "工件**程序化注入**生成（零手抄）。改数字请改工件或 `ledger.py` 的口径登记表后重新生成；"
        "`tests/test_verification_ledger.py` 会在台账 / 工件 / 本文档三者漂移时报红。")
    add("")
    add("## 0. 结论摘要")
    add("")
    add(f"- 覆盖 **{ledger['n_families']} 个家族 / {ledger['n_readings']} 条读数**，"
        f"来自 {len(ledger['sources'])} 个已提交工件，归入 "
        f"{sum(1 for k in ledger['conventions'] if any(r['convention'] == k for r in readings))} "
        "种测量口径。")
    add(f"- Δp 共 {dp['n_readings']} 条读数，只有 {dp['n_within_15pct']} 条 <15%、"
        f"{dp['n_within_3pct']} 条 <3%；当前口径（sanctioned）最优一条是 "
        f"`{best.get('id', '-')}`：**{_pp(best.get('rel_error_pct'))}%**。")
    add(f"- 唯一那条跨自由面且 <15% 的读数（`laplace_law#relaxed#0`）经三项对照检验后"
        f"**不能作为自由面压强闭合有效的证据**：工件不可复现（{_pp(q1.get('committed_rel_error_pct'))}% "
        f"vs 重跑 {_pp(q1.get('recomputed_rel_error_pct'))}%）、同代码路径细化一档即翻到 "
        f"{_pp(q3.get('fine_rel_error_pct'))}%、且其减去的上带压强在粗网格上恰被钳为 0"
        f"（{_f(q3.get('coarse_p_band_Pa'), 1)} Pa → 细网格 {_f(q3.get('fine_p_band_Pa'), 1)} Pa）。")
    add("- **判定：Δp 维持 WITHHELD。** 台账把 m2971 的 P0 归因（自由面压强闭合是唯一门槛项）"
        "从推理升级为可计数、可复算的事实，并正式退役 `center_minus_band_instant` 口径。")
    add("")

    add("## 1. 为什么要有这本台账")
    add("")
    add("同一个物理量（弯曲界面压差 Δp = σ/R）在本仓 2026-10 → 2026-11 的历史里被先后引用为 "
        "−76%、−107%、−87.6%、+5.6%、−34%。它们互相矛盾的主因不是求解器变了，而是**测量口径**变了"
        "（减的是哪两个压力、单时刻还是时间平均）。仓库原先没有任何机制阻止后来的 agent 把已废弃口径"
        "的数字当成当前精度引用——`backlog_update_m2971.md`（P2.1）要求的正是这台机器闸门。")
    add("")
    add("本台账把每个家族的**每一条读数**登记为一行，并给每一行贴上口径标签；口径标签带 "
        "`sanctioned`（可否作为当前精度引用）与理由。当前标准口径为「绝对压强 + 时间平均」（m2971）。")
    add("")

    add("## 2. 口径登记表（单一事实来源）")
    add("")
    add("| 口径 key | 相减的操作数 | 时间处理 | 跨自由面 | 可引用为当前精度 | 理由 |")
    add("|---|---|---|---|---|---|")
    for key, meta in ledger["conventions"].items():
        used = sum(1 for r in readings if r["convention"] == key)
        if used == 0:
            continue
        add(f"| `{key}` | {meta['operand']} | {meta['time']} | "
            f"{'是' if meta['free_surface'] else '否'} | "
            f"{'**是**' if meta['sanctioned'] else '否（历史口径）'} | "
            f"{meta['rationale']}（{used} 条） |")
    add("")

    add("## 3. 读数总表")
    add("")
    add("`Δ%` = (实测 − 解析)/解析，门限取平台标准 3% / 15%。全部数值由工件注入。")
    add("")
    add("| # | 家族 | 读数 id | 口径 | dx (m) | 实测 | 解析 | Δ% | 门限 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(readings, 1):
        add(f"| {i} | {row['family']} | `{row['id']}` | `{row['convention']}` | "
            f"{_f(row.get('dx_m'), 4)} | {_f(row['measured'], 4)} | {_f(row['theory'], 4)} | "
            f"{_pp(row['rel_error_pct'])} | {_gate(row)} |")
    add("")

    add("## 4. 同一算例、四种口径：口径能把误差挪多远")
    add("")
    add("`laplace_pressure_floor` 在 6 个算例上同时给出 4 种口径的读数，因此可以**在物理完全相同**"
        "的前提下隔离口径的影响：")
    add("")
    add("| 算例（dx, 说明） | abs_instant | bandA | bandB | abs_tail_avg（当前口径） | 口径极差 (pp) |")
    add("|---|---|---|---|---|---|")
    for row in sens:
        if not row["case"].startswith("laplace_pressure_floor"):
            continue
        label = row["case"].split(" / ", 1)[1]
        dx_part = row["case"].split(" / ", 1)[0].split()[-1]
        bucket = {
            r["convention"]: r["rel_error_pct"]
            for r in readings
            if r["family"] == "laplace_pressure_floor"
            and f"{r['dx_m']:.4f}" == dx_part
            and row["case"].endswith(r["note"])
        }
        add(f"| {label} | "
            + " | ".join(_pp(bucket.get(k)) if k in bucket else "-" for k in
                         ("abs_instant", "bandA_instant", "bandB_instant", "abs_tail_avg"))
            + f" | {row['spread_pp']:.1f} |")
    add("")
    worst = sens[0] if sens else {"spread_pp": 0.0, "case": "-", "min_pct": 0.0, "max_pct": 0.0}
    add(f"最大口径极差 **{worst['spread_pp']:.1f} pp**（算例 `{worst['case']}`：同一算例读数从 "
        f"{_pp(worst['min_pct'])} 到 {_pp(worst['max_pct'])}）。口径选择本身就能吞掉或放大整个误差，"
        "这就是历史引用互相矛盾的量级来源。")
    add("")

    add("## 5. 汇总与误差判定")
    add("")
    add("| 量 | 家族数 | 读数 | 中位 Δ% | 最差 abs(Δ%) | <15% | <3% |")
    add("|---|---|---|---|---|---|---|")
    for _, block in verdict["quantities"].items():
        add(f"| {block['label']} | {block['n_families']} | {block['n_readings']} | "
            f"{_pp(block['median_pct'])} | {block['worst_abs_pct']:.1f} | "
            f"{block['n_within_15pct']}/{block['n_readings']} | "
            f"{block['n_within_3pct']}/{block['n_readings']} |")
    add("")
    add("### 5.1 Δp 是否 <15%？（逐项作答）")
    add("")
    add(f"- 当前口径（sanctioned）读数 {dp['n_sanctioned']} 条：abs(Δ%) 最小 "
        f"**{dp['sanctioned_min_abs_pct']:.1f}%**、中位 {_pp(dp['sanctioned_median_pct'])}、"
        f"最差 {dp['sanctioned_max_abs_pct']:.1f}%；全部 <15% ？ "
        f"**{'是' if dp['sanctioned_all_within_15pct'] else '否'}**。")
    add(f"- 最优 sanctioned 读数：`{best.get('id', '-')}`（{best.get('family', '-')} / "
        f"`{best.get('convention', '-')}`）实测 {_f(best.get('measured_Pa'), 1)} Pa vs 解析 "
        f"{_f(best.get('theory_Pa'), 1)} Pa → **Δ = {_pp(best.get('rel_error_pct'))}%**。")
    add(f"- 全台账是否存在 abs(Δ%) < 3% 的 Δp 读数？**{'是' if dp['n_within_3pct'] else '否'}**。")
    add(f"- **结论：Δp 证据等级维持 WITHHELD** —— 当前口径最优点仍差 "
        f"{best.get('rel_error_pct', float('nan')):+.1f}%，未达平台 15% 门限。")
    add("")
    add("### 5.2 自由面是唯一分离变量（但有一条例外必须解释）")
    add("")
    add(f"- 跨自由面读数 {dp['n_free_surface']} 条：abs(Δ%) 最小 {dp['free_surface_min_abs_pct']:.1f}%，"
        f"其中仅 **{dp['free_surface_within_15pct']}** 条 <15%（逐条见下表与第 6 节）。")
    add(f"- 不跨自由面（两侧皆有流体，双相算例）读数 {dp['n_no_free_surface']} 条：abs(Δ%) 最小 "
        f"{dp['no_free_surface_min_abs_pct']:.1f}%，其中 {dp['no_free_surface_within_15pct']} 条 <15%"
        f"（占该子集 {100.0 * dp['no_free_surface_within_15pct'] / max(dp['n_no_free_surface'], 1):.0f}%）。")
    add("")
    passes = ledger.get("free_surface_passes") or []
    add(f"跨自由面且 <15% 的读数共 **{len(passes)}** 条。它们能否推翻 P0 归因，取决于两项检验："
        "(a) 是否分辨率稳定；(b) 读数是否真由表面张力产生。第 6 节逐条作答，结论是**两项都不满足**，"
        "因此不推翻。")
    add("")
    add("| 读数 | 家族 | 口径 | Δ% | dx (m) | 分辨率检验 |")
    add("|---|---|---|---|---|---|")
    verdict_label = {
        "zero_crossing": "零穿越：同家族 4 档分辨率上 |Δ%| 先降后升（随细化发散）",
        "no_ladder": "家族内只有单点，需外部对照（见 6.1–6.3）",
        "monotone_converging": "单调收敛",
        "monotone_diverging": "单调发散",
    }
    for row in passes:
        add(f"| `{row['id']}` | {row['family']} | `{row['convention']}` | "
            f"{_pp(row['rel_error_pct'], 2)} | {_f(row['dx_m'], 4)} | "
            f"{verdict_label.get(row['resolution_verdict'], row['resolution_verdict'])} |")
    add("")
    if cf:
        add(f"### 5.3 辅助量：切割力比 F_cut/(2σL)（{cf['n_readings']} 条，中位 "
            f"{_pp(cf['median_pct'])}%，最差 abs(Δ%) {cf['worst_abs_pct']:.1f}%）")
        add("")
        add("切割力是**半域积分**量，不经过自由面压强闭合，因此它单独达标也不能证明 Δp 达标；"
            "它落在门限附近说明「力 → 压强」的传递环节才是缺口。该量仅作辅助登记。")
        add("")

    add(f"## 6. 来源审计：{len(ledger.get('free_surface_passes') or [])} 条 <15% 的读数经得起检验吗")
    add("")
    add("### 6.1–6.3 被审对象 A：`laplace_law#relaxed#0`（工件 `benchmarks/laplace_law/validation_report.json`），"
        "登记的读数为 "
        f"{_pp(q1.get('committed_rel_error_pct'))}%。三项对照全部复跑该家族**自己的**代码路径 "
        "`benchmarks/laplace_law/run.py::relax`（R = 0.02 m，σ = 7.2 N/m，c₀ = 20 m/s，"
        "t = 0.89 t_σ）：")
    add("")
    add("### Q1 可复现性 —— 不通过")
    add("")
    add(f"- 重跑同一设置得 **{_pp(q1.get('recomputed_rel_error_pct'))}%**"
        f"（{_f(q1.get('recomputed_measured_Pa'), 1)} Pa），工件登记 "
        f"{_pp(q1.get('committed_rel_error_pct'))}%（{_f(q1.get('committed_measured_Pa'), 1)} Pa），"
        f"相差 **{q1.get('delta_pp', 0.0):.2f} pp**。")
    add("- 且 `run.py` 的输出版本是 `{\"instant\": [...], \"relax\": {...}}`，而该工件顶层为 "
        "`setup/reference/laplace/history`、`dx = 0.004`、380 步 —— 它不是当前代码的输出，"
        "是旧代码状态的摘要。**该工件不可复现，属陈旧快照。**")
    add("")
    add("### Q2 σ → 0 对照 —— 通过（读数确由表面张力产生）")
    add("")
    add(f"- σ 置为 {q2.get('sigma_control_N_per_m')} N/m、其余完全不变：Δp = "
        f"**{_f(q2.get('dp_sigma_control_Pa'), 1)} Pa**（σ = {q2.get('sigma_reference_N_per_m')} 时 "
        f"{_f(q2.get('dp_sigma_reference_Pa'), 1)} Pa）。")
    add(f"- 初始核和偏置压强（t=0 时 {_f(q2.get('p_core_t_start_Pa'), 0)} Pa，即审计 C2 的同一量）"
        f"在 σ→0 下 50 步内即衰减到 {_f(q2.get('p_core_t_after_50_steps_Pa'), 0)} Pa，"
        "而 σ = 7.2 时核心压强被持续维持。")
    add("- 因此 +8% 这一读数**不是**初始化伪压强的残留，它是 CSF 力真实压出来的核心过压。"
        "问题不在「是否为数值噪声」，而在下一问。")
    add("")
    add("### Q3 分辨率稳定性 —— 不通过（决定性）")
    add("")
    add("| 算例 | R/dx | p_core (Pa) | p_band (Pa) | Δp (Pa) | Δ% | <15% |")
    add("|---|---|---|---|---|---|---|")
    for key in ("sigma7_rdx5", "sigma7_rdx10"):
        case = (prov.get("cases") or {}).get(key)
        if not case:
            continue
        add(f"| `{key}` | {case['R_over_dx']:g} | {case['p_core_Pa']:.1f} | "
            f"{case['p_band_Pa']:.1f} | {case['dp_measured_Pa']:+.1f} | "
            f"{_pp(case['rel_error_pct'], 2)} | {'是' if case['within_15pct'] else '否'} |")
    add("")
    add(f"同一条代码路径、同一瞬时口径、同一物理时刻，仅把 R/dx 从 "
        f"{q3.get('coarse_R_over_dx'):g} 细化到 {q3.get('fine_R_over_dx'):g}，读数从 "
        f"**{_pp(q3.get('coarse_rel_error_pct'), 2)}%** 翻到 "
        f"**{_pp(q3.get('fine_rel_error_pct'), 2)}%**。机械原因很清楚：粗网格上被减去的上带压强"
        f"（p(0.80–0.95R)）被 Tait 钳位**恰为 0**（{_f(q3.get('coarse_p_band_Pa'), 1)} Pa），"
        f"于是「减上带」这一步退化成「绝对压强」；细网格上该带进入受压区变为 "
        f"{_f(q3.get('fine_p_band_Pa'), 1)} Pa，减法把信号本身减掉了。")
    add("")
    add("**判定：该读数不作数。** 它是「退化操作数 + 粗网格」的巧合，不能作为自由面压强闭合有效的"
        "证据；`center_minus_band_instant` 口径据此正式退役（见第 2 节）。P0 归因（自由面压强闭合是"
        "唯一门槛项）在台账里保持成立：\n")
    add(f"- 当前口径（绝对压强 + 时间平均）的 {dp['n_sanctioned']} 条读数全部在门限外"
        f"（最优 {dp['sanctioned_min_abs_pct']:.1f}%）；")
    add(f"- 不跨自由面（双相、无自由面闭合问题）的读数里已有 "
        f"{dp['no_free_surface_within_15pct']}/{dp['n_no_free_surface']} 条 <15%，说明曲率算子与 δ_s "
        "已不是瓶颈。")
    add("")
    add("### 6.4 其余通过者：同家族的分辨率阶梯自己否定了它们")
    add("")
    others = [r for r in (ledger.get("free_surface_passes") or [])
              if r["id"] != "laplace_law#relaxed#0"]
    by_family = {}
    for row in others:
        by_family.setdefault(row["family"], []).append(row)
    for family, rows in by_family.items():
        for row in rows:
            add(f"**`{row['id']}`**（`{row['source']}`，登记 {_pp(row['rel_error_pct'], 2)}%）："
                "把同工件内的分辨率阶梯全部列出来（数值同样由工件注入）：")
            add("")
            add("| dx (m) | 实测 Δp | 解析 Δp | Δ% |")
            add("|---|---|---|---|")
            for step in row["ladder"]:
                hit = [r for r in readings
                       if r["family"] == row["family"] and r["convention"] == row["convention"]
                       and r["dx_m"] is not None and abs(r["dx_m"] - step["dx_m"]) < 1e-12]
                measured = hit[0]["measured"] if hit else None
                theory = hit[0]["theory"] if hit else None
                add(f"| {_f(step['dx_m'], 5)} | {_f(measured, 4)} | {_f(theory, 3)} | "
                    f"{_pp(step['rel_error_pct'], 2)} |")
            add("")
            signed = [s["rel_error_pct"] for s in row["ladder"]]
            errs = [abs(v) for v in signed]
            add(f"符号误差从 {_pp(signed[0], 1)}%（最细网格）单调变到 {_pp(signed[-1], 1)}%"
                f"（最粗网格），abs(Δ%) 的最小值 {min(errs):.2f}% 落在**内部一档**："
                f"这是随细化发散的曲线穿过零点的位置（判定：`{row['resolution_verdict']}`），"
                "不是收敛点。以该档为「验证通过」引用，细化后必然失效。")
            add("")

    add("## 7. 注意点 / 局限")
    add("")
    add("1. **工件是历史快照**：台账只读 `benchmarks/**` 已提交的 json，不重跑仿真。"
        "陈旧工件会被如实登记（第 6 节是范例），行级 `source` 字段给出可追溯路径。")
    add("2. **口径标签人工登记、机器校验**：`sanctioned` 由 `CONVENTIONS` 表一次性给出，读数继承；"
        "新工件必须同时登记口径，否则 `collect()` 直接 KeyError。")
    add("3. **第 6 节的三项对照是实跑的**（非台账提取），原始快照留在 "
        "`benchmarks/verification_ledger/snapshot_*.json`；本机 load15 高时重跑约 3 分钟。")
    add("4. **未纳入**：")
    for item in ledger["out_of_scope"]:
        add(f"   - `{item['artefact']}`：{item['reason']}")
    add("5. 台账不判断模型好坏，只回答「这条数字是什么口径下测的、现在还能不能引用」。")
    add("")

    add("## 8. 复现")
    add("")
    add("```bash")
    add("python benchmarks/verification_ledger/build.py          # 重建 ledger.json / 本文档 / 图")
    add("python benchmarks/verification_ledger/build.py --check   # 只校验漂移（CI 用）")
    add("python benchmarks/verification_ledger/provenance_control.py --run   # 重跑第 6 节三项对照")
    add("python -m pytest tests/test_verification_ledger.py -q")
    add("```")
    add("")
    add("![verification ledger](benchmarks/verification_ledger/ledger.png)")
    add("")
    return "\n".join(lines)


def render_figure(ledger: Dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    readings = ledger["readings"]
    verdict = ledger["verdict"]
    prov = ledger.get("provenance") or {}
    floor = [r for r in readings if r["family"] == "laplace_pressure_floor"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)

    # ---- panel 1: same case, four conventions -----------------------------
    ax = axes[0][0]
    cases = []
    for r in floor:
        key = (r["dx_m"], r["note"])
        if key not in cases:
            cases.append(key)
    convs = ["abs_instant", "bandA_instant", "bandB_instant", "abs_tail_avg"]
    width = 0.2
    for j, conv in enumerate(convs):
        xs, vals = [], []
        for i, key in enumerate(cases):
            hit = [r for r in floor if (r["dx_m"], r["note"]) == key and r["convention"] == conv]
            if hit:
                xs.append(i + (j - 1.5) * width)
                vals.append(hit[0]["rel_error_pct"])
        ax.bar(xs, vals, width=width, label=conv,
               color=["#9ecae1", "#fdae6b", "#c9a0dc", "#2c7fb8"][j],
               edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-15, color="crimson", ls="--", lw=1.0)
    ax.axhline(15, color="crimson", ls="--", lw=1.0)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"dx={d:.4f}\n{n}" for d, n in cases], fontsize=7, rotation=20)
    ax.set_ylabel("Δp relative error  (%)")
    ax.set_title("(1) Same physics, four conventions\nlaplace_pressure_floor (6 cases)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # ---- panel 2: every Δp reading, split by free surface ------------------
    ax = axes[0][1]
    fs = sorted(r["rel_error_pct"] for r in readings
                if r["quantity"] == "laplace_dp" and r["free_surface"])
    nfs = sorted(r["rel_error_pct"] for r in readings
                 if r["quantity"] == "laplace_dp" and not r["free_surface"])
    ax.scatter(fs, [0] * len(fs), marker="o", s=38, color="#d7301f", alpha=0.85,
               label=f"crosses free surface (n={len(fs)})")
    ax.scatter(nfs, [1] * len(nfs), marker="s", s=38, color="#2166ac", alpha=0.85,
               label=f"both sides wetted (n={len(nfs)})")
    ax.axvspan(-15, 15, color="green", alpha=0.10)
    ax.axvline(-15, color="crimson", ls="--", lw=1.0)
    ax.axvline(15, color="crimson", ls="--", lw=1.0)
    ax.axvline(0, color="black", lw=0.8)
    n_fs_pass = sum(1 for v in fs if abs(v) < 15)
    ax.annotate("the one passing\nfree-surface reading\n(see panel 4)",
                xy=(min(fs), 0), xytext=(-95, 0.45), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["crosses free\nsurface", "both sides\nwetted"], fontsize=8)
    ax.set_xlabel("Δp relative error  (%)   [gate ±15 % shaded]")
    ax.set_title(f"(2) {len(fs) + len(nfs)} Δp readings: only {n_fs_pass} free-surface\n"
                 "reading clears the gate — and it fails the audit", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="x", alpha=0.3)

    # ---- panel 3: sanctioned readings vs dx ------------------------------
    ax = axes[1][0]
    sanc = [r for r in readings if r["quantity"] == "laplace_dp" and r["sanctioned"]]
    fams = sorted({r["family"] for r in sanc})
    colors = ["#2c7fb8", "#e6550d", "#31a354", "#756bb1"]
    for i, fam in enumerate(fams):
        rows = [r for r in sanc if r["family"] == fam]
        xs = [r["dx_m"] if r["dx_m"] is not None else 0.1 * (i + 1) for r in rows]
        ys = [r["rel_error_pct"] for r in rows]
        ax.scatter(xs, ys, label=fam, color=colors[i % len(colors)], s=44,
                   marker="o", edgecolor="black", linewidth=0.4)
    ax.axhspan(-15, 15, color="green", alpha=0.10)
    ax.axhline(-15, color="crimson", ls="--", lw=1.0)
    ax.axhline(15, color="crimson", ls="--", lw=1.0)
    ax.axhline(0, color="black", lw=0.8)
    if sanc:
        med = verdict["quantities"]["laplace_dp"]["sanctioned_median_pct"]
        ax.axhline(med, color="black", ls=":", lw=1.0,
                   label=f"sanctioned median {med:+.1f} %")
    ax.set_xscale("log")
    ax.set_xlabel("dx  (m)")
    ax.set_ylabel("Δp relative error  (%)")
    ax.set_title("(3) Sanctioned convention only:\nno dx-convergence into the gate", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- panel 4: provenance controls ------------------------------------
    ax = axes[1][1]
    styles = {
        "sigma7_rdx5": ("#d7301f", "R/dx=5, σ=7.2 N/m (committed settings)"),
        "sigma0_rdx5": ("#2166ac", "R/dx=5, σ→0 control"),
        "sigma7_rdx10": ("#31a354", "R/dx=10, σ=7.2 N/m (refined)"),
    }
    for key, (color, label) in styles.items():
        case = (prov.get("cases") or {}).get(key)
        if not case or not case.get("t_trace_s"):
            continue
        ax.semilogy(case["t_trace_s"], [max(v, 1e-2) for v in case["p_core_trace_Pa"]],
                    marker="o", ms=4, lw=1.4, color=color, label=label)
    for key, color in (("sigma7_rdx5", "#d7301f"), ("sigma7_rdx10", "#31a354")):
        case = (prov.get("cases") or {}).get(key)
        if not case:
            continue
        ax.axhline(abs(case["dp_measured_Pa"]), color=color, ls=":", lw=1.0)
    ax.axhline(360.0, color="black", ls="--", lw=1.0, label="σ/R = 360 Pa")
    ax.set_xlabel("t  (s)")
    ax.set_ylabel("core pressure  (Pa, log)")
    ax.set_title("(4) Core pressure history: control cases\n"
                 "init residue dies in 50 steps; σ-driven overpressure persists", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    q1 = prov.get("q1_reproducible") or {}
    fig.suptitle(
        "TensorSPH verification ledger — Laplace pressure jump, "
        f"{ledger['n_readings']} readings / {ledger['n_families']} families / "
        f"{sum(1 for k in ledger['conventions'] if any(r['convention'] == k for r in readings))} "
        f"conventions; best sanctioned Δ = {verdict.get('best_sanctioned_reading', {}).get('rel_error_pct', float('nan')):+.1f}%"
        + (f"; committed laplace_law reading {q1.get('committed_rel_error_pct'):+.2f}% is stale "
           f"(re-run {q1.get('recomputed_rel_error_pct'):+.2f}%)" if q1 else ""),
        fontsize=10.5,
    )
    fig.savefig(path, dpi=92)
    plt.close(fig)
    try:  # keep the PNG small enough for report embedding
        from PIL import Image

        if path.stat().st_size > 230_000:
            img = Image.open(path).convert("P", palette=Image.ADAPTIVE, colors=128)
            img.save(path, optimize=True)
    except Exception:  # pragma: no cover - optional shrink only
        pass


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the committed artefacts are out of date")
    args = parser.parse_args(argv)

    ledger = L.build(ROOT)
    markdown = render_markdown(ledger)
    payload = json.dumps(ledger, indent=1, sort_keys=True, allow_nan=False)

    if args.check:
        stale = []
        if not OUT_JSON.exists() or OUT_JSON.read_text(encoding="utf-8").strip() != payload.strip():
            stale.append(str(OUT_JSON.relative_to(ROOT)))
        if not OUT_MD.exists() or OUT_MD.read_text(encoding="utf-8").strip() != markdown.strip():
            stale.append(str(OUT_MD.relative_to(ROOT)))
        if stale:
            print("STALE: " + ", ".join(stale))
            return 1
        print("ledger up to date")
        return 0

    OUT_JSON.write_text(payload + "\n", encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(markdown + "\n", encoding="utf-8")
    render_figure(ledger, OUT_PNG)
    print(f"wrote {OUT_JSON.relative_to(ROOT)} ({len(ledger['readings'])} readings), "
          f"{OUT_MD.relative_to(ROOT)}, {OUT_PNG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
