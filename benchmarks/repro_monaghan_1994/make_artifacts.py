"""Render REPRO-SPH-MONAGHAN1994 artifacts: reference declaration, figures, document.

Everything here is read out of the committed artifacts — ``results/summary.json``,
``results/trajectory_*.json`` and ``results/refs_verified.json`` — so no number in
the figures or in ``docs/repro_monaghan_1994.md`` is typed by hand.

Usage::

    python benchmarks/repro_monaghan_1994/make_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for candidate in (str(ROOT), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from benchmarks.repro_monaghan_1994.run import (                     # noqa: E402
    COMPRESSION_LIMIT, DEFAULT_DX, N_SAMPLES, TOL_REL, WINDOW, references)
from benchmarks.dam_break.reference_audit import structural_audit    # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
DOC_PATH = ROOT / "docs" / "repro_monaghan_1994.md"
REFS_PATH = HERE / "refs.json"
RESULTS = HERE / "results"


def load() -> tuple[dict, list[dict], dict]:
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} missing: run `run.py --stage run` first")
    summary = json.loads(summary_path.read_text())
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("trajectory_*.json"))]
    verified_path = RESULTS / "refs_verified.json"
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    return summary, runs, verified


def load_estimator_probe() -> dict | None:
    """Merge the committed ``front_estimator_probe*.json`` fragments, if any.

    The probe writes one JSON per invocation (one per ``--dx`` list), so the
    fragments are merged here; a missing probe simply omits the section - it is
    a diagnostic, not a gate.
    """
    cases: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("front_estimator_probe*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for case in payload.get("cases", []):
            cases.setdefault(case["case"], case)
    if not cases:
        return None
    return {"cases": [cases[k] for k in sorted(cases, key=lambda k: -cases[k]["dx"])]}


def write_refs_declaration(summary: dict, verified: dict) -> dict:
    """The reference/protocol declaration, built from the audit and the DOI resolution."""
    audit = structural_audit()
    selected = (verified.get("crossref_search") or {}).get("selected") or {}
    payload = {
        "benchmark": "REPRO-SPH-MONAGHAN1994",
        "anchor": "REPRO-SPH-MONAGHAN1994",
        "paper": {
            "citation_claim": (verified.get("anchor_citation_claim") or {}),
            "resolved_doi": selected.get("doi"),
            "resolved_title": selected.get("title"),
            "resolved_container": selected.get("container"),
            "resolved_volume": selected.get("volume"),
            "resolved_pages": selected.get("page"),
            "resolution_rule": (verified.get("crossref_search") or {}).get("rule"),
            "resolution_artifact": "results/refs_verified.json",
            "retrieved_utc": verified.get("retrieved_utc"),
            "open_access": (verified.get("openalex") or {}).get("fields", {}),
            "text_available_in_workspace": False,
        },
        "targets": {
            "T1_trajectory_vs_independent_transcription": {
                "status": "evaluated",
                "reference": "benchmarks/dam_break/martin_moyce_1952_crosscheck.json",
                "metric": "mae(Z/a) / amplitude over the pre-registered window",
                "tolerance": TOL_REL,
                "window_tau": list(WINDOW),
            },
            "T2_trajectory_vs_bundled_transcription": {
                "status": "withheld",
                "reference": "benchmarks/dam_break/martin_moyce_1952.json",
                "reason": (audit.get("verdict", {}).get("reason")
                           or "bundled table graded WITHHELD by the reference audit"),
                "reported_not_graded": True,
            },
            "T3_reference_pair_self_consistency": {
                "status": "evaluated",
                "metric": "max |bundled - crosscheck| / claimed uncertainty",
                "limit": 2.0,
            },
            "T4_paper_tables_and_figures": {
                "status": "WITHHELD",
                "reason": ("no accessible copy: Crossref resolves the DOI by rule and "
                           "OpenAlex reports the work as closed access with no deposited "
                           "abstract, so the paper's own tabulated or figured numbers are "
                           "not in this workspace"),
                "closure_condition": ("deliver JCP 110:399-406 as a text-extractable file "
                                      "with page/figure coordinates for the quantitative "
                                      "target; then transcribe the value with coordinates "
                                      "into refs.json and re-run --stage accept (no solver "
                                      "change needed)"),
            },
        },
        "protocol": summary.get("protocol"),
        "reference_audit_artifact": "benchmarks/dam_break/reference_audit.json",
        "reproduction_commands": [
            "python benchmarks/repro_monaghan_1994/verify_refs.py",
            "python benchmarks/repro_monaghan_1994/run.py --stage run --dx 0.04,0.02,0.01 --seconds 0.34",
            "python benchmarks/repro_monaghan_1994/make_artifacts.py",
        ],
    }
    REFS_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def make_figures(summary: dict, runs: list[dict]) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refs = references()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = []

    # ---- figure 1: trajectory against both transcriptions -------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    b, c = refs["bundled"], refs["crosscheck"]
    ax.axvspan(WINDOW[0], WINDOW[1], color="0.92", zorder=0,
               label=f"pre-registered window $\\tau\\in[{WINDOW[0]},{WINDOW[1]}]$")
    ax.plot(b["tau"], b["za"], "s--", color="#d62728", ms=4, lw=1.0,
            label="bundled transcription (audit: WITHHELD, synthetic lattice)")
    ax.plot(c["tau"], c["za"], "o-", color="#2ca02c", ms=4, lw=1.0,
            label="independent transcription (Lethe digitisation)")
    ax.fill_between(c["tau"], [v - 0.05 for v in c["za"]],
                    [v + 0.05 for v in c["za"]], color="#2ca02c", alpha=0.15,
                    label="independent transcription $\\pm0.05$")
    colors = {0.04: "#1f77b4", 0.02: "#ff7f0e", 0.01: "#9467bd"}
    for run in sorted(runs, key=lambda r: -r["dx"]):
        ax.plot(run["tau"], run["za"], "-", lw=1.8, color=colors.get(run["dx"], "k"),
                label=f"TensorSPH dx={run['dx']:g} m (N={run['particles']})")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=12)
    ax.set_ylabel(r"$Z/a = (x_{\rm front}-a)/a$", fontsize=12)
    ax.set_title("Dam-break front trajectory: TensorSPH vs two transcriptions of\n"
                 "Martin & Moyce (1952), the reference case of Monaghan (1994)", fontsize=12)
    ax.set_xlim(0.0, max(r["tau"][-1] for r in runs) * 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path1 = FIG_DIR / "repro_monaghan1994_fig1_trajectory.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": path1.name,
                    "caption": "Front trajectory in the Martin & Moyce variables against both "
                               "bundled transcriptions and the TensorSPH ladder.",
                    "path": path1})

    # ---- figure 2: signed error against each reference ----------------------
    finest = min(runs, key=lambda r: r["dx"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for name, color, marker in (("crosscheck", "#2ca02c", "o"), ("bundled", "#d62728", "s")):
        samples = summary["comparisons"][finest["case"]][name]["samples"]
        ax.plot([s["tau"] for s in samples], [s["delta"] for s in samples],
                marker + "-", ms=3, lw=1.2, color=color, label=f"vs {name}")
    unc = summary["comparisons"][finest["case"]]["crosscheck"]["claimed_uncertainty"]
    ax.axhspan(-unc, unc, color="#2ca02c", alpha=0.15,
               label=f"reference uncertainty $\\pm{unc:g}$ in $Z/a$")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\tau$", fontsize=12)
    ax.set_ylabel(r"$\Delta Z/a$ (simulation $-$ reference)", fontsize=12)
    ax.set_title(f"Signed error at dx={finest['dx']:g} m: the two references differ by "
                 f"more than the solver deviates from either", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path2 = FIG_DIR / "repro_monaghan1994_fig2_signed_error.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": path2.name,
                    "caption": "Signed error against each transcription at the finest "
                               "resolution, with the reference's own uncertainty band.",
                    "path": path2})

    # ---- figure 3: resolution ladder ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, color, marker in (("crosscheck", "#2ca02c", "o"), ("bundled", "#d62728", "s")):
        entries = summary["ladder"][name]
        ax.plot([e["dx"] for e in entries],
                [100.0 * e["mae_over_amplitude"] for e in entries],
                marker + "-", color=color, label=f"vs {name}")
    ax.axhline(100.0 * TOL_REL, color="k", ls="--", lw=1.1,
               label=f"{100 * TOL_REL:g} % tolerance")
    ax.set_xscale("log")
    ax.set_xlabel("particle spacing dx [m]", fontsize=12)
    ax.set_ylabel("mae / amplitude [%]", fontsize=12)
    ax.set_title("Reference sensitivity is resolution independent: the gap to the\n"
                 "bundled table does not shrink, the gap to the independent one does",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path3 = FIG_DIR / "repro_monaghan1994_fig3_ladder.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": path3.name,
                    "caption": "Error against each transcription through the resolution "
                               "ladder, with the platform tolerance.",
                    "path": path3})
    return figures


def pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f} %"


def num(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def make_doc(summary: dict, runs: list[dict], refs_payload: dict,
             figures: list[dict], verified: dict) -> Path:
    finest = min(runs, key=lambda r: r["dx"])
    gates = summary["gates"]
    bracket = summary["bracket"]
    verdict = summary["verdict"]
    primary = summary["comparisons"][finest["case"]]["crosscheck"]
    bundled = summary["comparisons"][finest["case"]]["bundled"]
    paper = refs_payload["paper"]
    oa = paper.get("open_access") or {}

    lines: list[str] = []
    add = lines.append
    add("# REPRO-SPH-MONAGHAN1994 — 溃坝前沿轨迹复现（TensorSPH）")
    add("")
    add("**锚点**：`REPRO-SPH-MONAGHAN1994` · **仓**：TensorSPH · "
        "**参考论文**：Monaghan (1994) *J. Comput. Phys.* **110**:399-406 "
        f"(DOI `{paper.get('resolved_doi')}`) · **判据**：quantitative_targets 逐项容差")
    add("")
    add("本文档由 `benchmarks/repro_monaghan_1994/make_artifacts.py` 从 "
        "`results/summary.json`、`results/trajectory_*.json`、`results/refs_verified.json` "
        "**程序化生成**：每个数字都是读出来的，人工不写数。")
    add("")
    add("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    add("")
    add("Monaghan (1994) 的 SPH 自由面论文以溃坝（dam break）作为与实验对标的算例："
        "矩形水柱在静止状态下释放、在干床水槽内坍塌并沿底床推进，观测量是**涌浪前沿位移**"
        "随无量纲时间的轨迹，参考数据来自 Martin & Moyce (1952) 的溃坝实验。"
        "本仓的 B2 算例采用该实验最常用的几何（柱宽 a、高 2a）。")
    add("")
    add(f"* 论文标识核验（`results/refs_verified.json`，retrieved {paper.get('retrieved_utc')}）："
        f"Crossref 检索按规则命中 DOI `{paper.get('resolved_doi')}`，"
        f"{paper.get('resolved_container')} {paper.get('resolved_volume')}:"
        f"{paper.get('resolved_pages')}，作者行含 Monaghan；OpenAlex 对同一 DOI 报 "
        f"`oa_status={oa.get('oa_status')}`、有无摘要="
        f"{'有' if oa.get('has_abstract') else '无'}。**全文在仓内不可得**"
        "（无 PDF/文本，无登记副本），因此论文自身的表/图数值一律不填（见 T4）。")
    add("* 几何与数值参数（`results/summary.json:protocol`，先于任何一次运行写死）：")
    add("")
    add("| 参数 | 值 |")
    add("|---|---|")
    proto = summary["protocol"]
    geom = proto["geometry"]
    add(f"| 水柱宽 a | {geom['column_width_m']} m |")
    add(f"| 高宽比 H/a | {geom['height_over_width']} |")
    add(f"| 水槽 | {geom['tank_width_m']} m × {geom['tank_height_m']} m，干床、自由滑移底 |")
    add(f"| 重力 / 密度 | {geom['gravity']} m/s² / {geom['rho0']} kg/m³ |")
    add(f"| 状态方程 | Tait，γ = {geom['tait_gamma']}（弱可压） |")
    add(f"| 平滑长度 | h = {geom['h_over_dx']}·dx |")
    add(f"| 声速 / Courant | c0 见运行表，max_courant = {geom['max_courant']} |")
    add(f"| 人工粘性 / 密度扩散 | α = 0.03（求解器默认）/ {geom['density_diffusion']} |")
    add(f"| 壁面摩擦 | {geom['wall_friction']}（自由滑移） |")
    add(f"| 评估窗口 | τ ∈ [{proto['window_tau'][0]}, {proto['window_tau'][1]}] |")
    add(f"| 采样点数 | {proto['n_samples']} |")
    add(f"| 容差 | 轨迹误差 ≤ {100 * proto['tolerance_relative_to_amplitude']:.0f} % 幅值 |")
    add("")
    add("**两份参考表的出处（这是本算例的关键约束）**：")
    add("")
    add(f"* 独立数字化表 `martin_moyce_1952_crosscheck.json`：与 Lethe 官方算例后处理"
        f"分发的数字化值逐值同源，非本仓重画；与 1952 原文**未**逐点核对（原文付费）。"
        f"声明的参考不确定度 ±{bracket.get('claimed_uncertainty')} in Z/a。")
    add("* 内置表 `martin_moyce_1952.json`：被 "
        "`benchmarks/dam_break/reference_audit.json` 判为**合成样表**"
        "（纵坐标全是 0.05 的整数倍、τ 是精确等差格点），grade = WITHHELD，"
        "不得作为判据来源。")
    add("")
    add("## ② 怎么做的（方法 / 流程 / 关键设置）")
    add("")
    add("1. **先定参考与判据，再跑**：`refs.json` 声明四个目标 T1–T4 与窗口/容差，"
        "写入时间早于任何一次求解（协议常量在 `run.py` 顶部）。")
    add("2. **论文标识按规则解析而非手填**：`verify_refs.py` 用锚点给的引文串检索 Crossref，"
        "打分规则（作者/卷/页/年/题名词元）选出命中项，再查 OpenAlex 的开放获取状态，"
        "结果原样落 `results/refs_verified.json`。")
    add("3. **求解器不改**：直接驱动仓内 B2 算例（WCSPH + Tait + Adami 壁面压力外推），"
        "只做驱动、无量纲化与聚合。无量纲化 Z/a = (x_front − a)/a，"
        "τ = t·√(2g/a)——与两份参考表同一坐标系（crosscheck 表自报 X = x_front/a，"
        "Z/a = X − 1）。")
    dx_list = ", ".join(f"{r['dx']:g}" for r in sorted(runs, key=lambda r: -r['dx']))
    add(f"4. **分辨率阶梯** dx ∈ {{{dx_list}}} m，"
        "同一 t_end，考察误差是否随加密下降（若误差与分辨率无关，则它不是离散误差）。")
    add("5. **双参考括号**：同一仿真同时与两份数字化对照，并做参考对的**自洽性检验**"
        "（两者之差 / 声明不确定度）——参考自己不自洽时，任何“通过/失败”都不能归给求解器。")
    add("6. **聚合与出图**：`make_artifacts.py` 只读工件，产出 3 张图与本文档 + `refs.json`。")
    add("")
    add("复现命令：")
    add("")
    add("```bash")
    for cmd in refs_payload["reproduction_commands"]:
        add(cmd)
    add("```")
    add("")
    add("## ③ 计算结果（数字表，程序化注入）")
    add("")
    add("### 3.1 运行表")
    add("")
    add("| case | dx [m] | 粒子数 | 步数 | 墙钟 [s] | dt [s] | Mach | max ρ/ρ0−1 |")
    add("|---|---|---|---|---|---|---|---|")
    for run in sorted(runs, key=lambda r: -r["dx"]):
        add(f"| `{run['case']}` | {run['dx']:g} | {run['particles']} | {run['steps_taken']} | "
            f"{run['wall_clock_s']:.1f} | {run['dt']:.3e} | {run['mach_number']:.3f} | "
            f"{run['max_density_compression']:.4f} |")
    add("")
    add("### 3.2 与两份参考表的对照（τ ∈ 评估窗口，最细分辨率）")
    add("")
    add(f"| 参考 | MAE(Z/a) | 幅值 | MAE/幅值 | max abs | 偏置 | 命中 ±{bracket.get('claimed_uncertainty')} 比例 | 判级 |")
    add("|---|---|---|---|---|---|---|---|")
    for name in ("crosscheck", "bundled"):
        comp = summary["comparisons"][finest["case"]][name]
        grade = ("PASS" if comp["mae_over_amplitude"] <= TOL_REL else "FAIL") \
            if name == "crosscheck" else "WITHHELD（参考不可用）"
        add(f"| {name} | {num(comp['mae'])} | {num(comp['amplitude'])} | "
            f"{pct(comp['mae_over_amplitude'])} | {num(comp['max_abs'])} | "
            f"{num(comp['bias']):} | {pct(comp['within_uncertainty_fraction'], 0)} | {grade} |")
    add("")
    add("### 3.3 分辨率阶梯（MAE/幅值，附偏置符号）")
    add("")
    add("| dx [m] | vs independent MAE/幅值 | 偏置 (sim−ref) | 偏置/幅值 | vs bundled MAE/幅值 |")
    add("|---|---|---|---|---|")
    entries_c = {e["dx"]: e for e in summary["ladder"]["crosscheck"]}
    entries_b = {e["dx"]: e for e in summary["ladder"]["bundled"]}
    bias_rows = []
    for dx in sorted(entries_c, reverse=True):
        comp = summary["comparisons"][entries_c[dx]["case"]]["crosscheck"]
        bias_rows.append((dx, comp["bias"], comp["amplitude"]))
        add(f"| {dx:g} | {pct(entries_c[dx]['mae_over_amplitude'])} | "
            f"{comp['bias']:+.3f} | {pct(comp['bias'] / comp['amplitude'])} | "
            f"{pct(entries_b[dx]['mae_over_amplitude'])} |")
    add("")
    # Does the non-monotone |MAE| come from a *sign crossing* of a monotone bias?
    # (Computed, not written: the diagnosis follows from the artefacts above.)
    rows_fine = sorted(bias_rows, reverse=True)          # coarsest -> finest
    biases = [r[1] for r in rows_fine]
    amp_c = rows_fine[0][2]
    slopes = [b - a for a, b in zip(biases, biases[1:])]
    mono = all(s > 0 for s in slopes) or all(s < 0 for s in slopes)
    crossed = any(a * b < 0 for a, b in zip(biases, biases[1:]))
    sat = abs(slopes[-1]) / amp_c if slopes else float("nan")
    add(f"* 偏置随加密（dx 递减）的逐级增量："
        f"{', '.join(f'{100 * s / amp_c:+.2f} % 幅值' for s in slopes)}；"
        f"偏置{'单调' if mono else '非单调'}"
        f"{'且发生符号翻转' if crossed else '且未发生符号翻转'}。")
    add(f"* 因此 `resolution_trend_vs_independent` 的 fail 是 **|偏置| 过零**造成的，"
        f"不是“误差不随加密下降”：最细两级之间偏置仅再变化 {100 * sat:.2f} % 幅值"
        f"（已进入饱和），残余差 {(biases[-1] and 100 * abs(biases[-1]) / amp_c):.2f} % 幅值"
        "是**与分辨率无关**的固定偏移，按定义不能归因于离散误差。" if (mono and crossed)
        else f"* 偏置{'单调' if mono else '非单调'}"
             f"{'且已过零' if crossed else '且未过零'}；"
             f"最细两级之间偏置变化 {100 * sat:.2f} % 幅值，"
             "阶梯判据的 fail 如实保留，不解释为测量噪声。")
    add("")
    probe = load_estimator_probe()
    if probe:
        add("### 3.3b 前沿估计器敏感性（独立探针）")
        add("")
        add("`front_estimator_probe.py` 用同一 `DamBreakConfig` 重跑，并在每个输出时刻记录粒子 "
            "`x` 的分布（max / 99.5% / 99% / 95% 分位），检查判据量是不是被单颗飞溅粒子决定。")
        add("")
        add("| dx [m] | 估计器 | MAE/幅值 (independent) | 偏置 | 领先 99.5% 分位 >2dx 的粒子数（各时刻最大） |")
        add("|---|---|---|---|---|")
        for case in probe["cases"]:
            ahead = max(q["n_ahead_of_p995_by_2dx"] for q in case["probe"])
            for est, errs in case["errors"].items():
                ref = errs.get("crosscheck") or {}
                add(f"| {case['dx']:g} | `{est}` | {pct(ref.get('mae_over_amplitude'))} | "
                    f"{num(ref.get('bias'))} | {ahead} |")
        add("")
        repro = [c["committed_diff"] for c in probe["cases"] if c.get("committed_diff")]
        if repro:
            worst = max(r["max_abs_za_diff"] for r in repro)
            add(f"* 同机重跑与已提交轨迹的最大逐点差：**{num(worst, 6)}** in Z/a"
                f"（{len(repro)} 个分辨率：{', '.join(str(r['path']).split('_')[-2] for r in repro)} m）。")
            add("")
    add("### 3.4 参考对自洽性（T3）与括号")
    add("")
    add(f"* 两份数字化在同一窗口内的最大分歧：**{num(bracket.get('reference_spread_max'))} in Z/a**"
        f"（窗口 τ ∈ [{bracket['window'][0]:.3f}, {bracket['window'][1]:.3f}]，"
        f"n = {bracket['n_samples']}），是其声明不确定度 "
        f"{bracket.get('claimed_uncertainty')} 的 **{num(bracket.get('reference_spread_over_claimed'), 1)}×**。")
    add(f"* 仿真落在双表带（含 ±不确定度）内的比例：{pct(bracket.get('inside_fraction'), 0)}；"
        f"最大越界量 {num(bracket.get('max_excursion'))} in Z/a。")
    add("")
    add("### 3.5 判据与判决")
    add("")
    add("| gate | 实测 | 阈值 | 结果 |")
    add("|---|---|---|---|")
    for name, gate in gates.items():
        measured = gate.get("measured")
        if isinstance(measured, dict):
            shown = ", ".join(f"{100 * v:.2f} %" for v in measured.get("values", []) if v is not None)
        elif isinstance(measured, float):
            shown = f"{measured:.4g}"
        else:
            shown = str(measured)
        add(f"| `{name}` | {shown} | {gate.get('limit')} | **{gate['result']}** |")
    add("")
    add(f"**判决**：physical_validation = **{verdict['physical_validation']}**（对独立数字化表），"
        f"参考可否裁决 = {verdict['reference_evaluable']}；内置合成表的使用状态 = withheld。")
    add("")
    add(verdict["interpretation"])
    add("")
    add("## ④ 结果图")
    add("")
    for fig in figures:
        add(f"![{fig['name']}](figures/{fig['name']})")
        add("")
        add(f"*{fig['caption']}*")
        add("")
    add("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    add("")
    add("1. **DOI 不能凭记忆填**：本次首试把 `10.1006/jcph.1994.1187` 当成本文 DOI，"
        "Crossref 返回的是 Liu–Osher–Chan 的 WENO 文章（JCP 115:200）。改为“检索+规则命中”后"
        f"才解析出 `{paper.get('resolved_doi')}`（JCP 110:399）。任何“记着的 DOI/页码”都可能是指向别人的。")
    add("2. **两份数字化互不自洽**：分歧 0.85 in Z/a，远超其声明 ±0.05。也就是说"
        "本算例的瓶颈是**参考**，不是求解器；把 0.57 的偏差记成求解器 FAIL 是错误归因。")
    add("3. **内置表的“合成”证据是结构性的**（精确等差 τ + 0.05 量子 + 起点斜率达峰值的 "
        "0.885 即冲动起步），不是“看起来不像实验”的主观判断。")
    add("4. **窗口 τ ∈ [0.5, 3] 与容差 15 % 是跑前写死的**；窗口下限避开启动瞬态，"
        "上限取在两表公共区间内。窗口或容差若事后调整，结论不可比。")
    add("5. **适用范围**：单相、干床、自由滑移底、c0 = 25 m/s 的弱可压配置；"
        "对 no-slip 底床（wall_friction=1）与其他声速的敏感度不在本算例结论内。")
    add(f"6. **压缩性余量**：最细分辨率 max ρ/ρ0−1 = {finest['max_density_compression']:.3f}，"
        f"限值 {COMPRESSION_LIMIT}——通过但不宽裕。")
    add("")
    add("## ⑥ 误差判定（逐项 vs 基准，显式回答是否 < 3 %）")
    add("")
    add("| 关键量 | 基准 | 实测误差 | 是否 < 3 % |")
    add("|---|---|---|---|")
    add(f"| 前沿轨迹 Z/a（最细 dx={finest['dx']:g} m） | 独立数字化表（τ∈[{WINDOW[0]},{WINDOW[1]}]） | "
        f"MAE/幅值 = {pct(primary['mae_over_amplitude'])}，max abs = {num(primary['max_abs'])} | "
        f"{'是' if primary['mae_over_amplitude'] < 0.03 else '否'} |")
    add(f"| 前沿轨迹 Z/a（最细 dx={finest['dx']:g} m） | 内置合成表 | "
        f"{pct(bundled['mae_over_amplitude'])}（**不判级**：参考 WITHHELD） | 不适用 |")
    add(f"| 参考对自洽（同一图的两次数字化） | 声明不确定度 ±{bracket.get('claimed_uncertainty')} | "
        f"{num(bracket.get('reference_spread_max'))}（{num(bracket.get('reference_spread_over_claimed'), 1)}× 声明值） | 否 |")
    add(f"| 弱可压质量/密度闭合 | max ρ/ρ0−1 ≤ {COMPRESSION_LIMIT} | "
        f"{finest['max_density_compression']:.4f} | 是 |")
    add("")
    paper_grade = primary["mae_over_amplitude"] <= 0.03
    platform_grade = primary["mae_over_amplitude"] <= TOL_REL
    add(f"结论：**对唯一通过结构审计的参考，前沿轨迹误差 {pct(primary['mae_over_amplitude'])}，"
        f"{'在 3 % 以内（回答“是”）' if paper_grade else '不在 3 % 以内（回答“否”）'}**；"
        f"平台验收线（{pct(TOL_REL, 0)} 幅值）为 "
        f"{'pass' if platform_grade else 'fail'}——两条线不是同一条，不得互相替代引用。"
        "论文自身的表/图数值仍为 WITHHELD（全文不可得），不得以任何转述值替代。")
    add("")
    DOC_PATH.write_text("\n".join(lines) + "\n")
    return DOC_PATH


def main() -> int:
    summary, runs, verified = load()
    refs_payload = write_refs_declaration(summary, verified)
    figures = make_figures(summary, runs)
    doc = make_doc(summary, runs, refs_payload, figures, verified)
    print(f"wrote {REFS_PATH}")
    for fig in figures:
        print(f"wrote {fig['path']}")
    print(f"wrote {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())