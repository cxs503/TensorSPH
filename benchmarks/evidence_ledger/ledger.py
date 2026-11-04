"""Machine-checked ledger of the Young-Laplace / surface-tension evidence chain.

Why this exists
---------------
The same physical quantity -- the capillary pressure jump ``Delta p = sigma/R`` --
has been measured nine times in this repository over four rounds of work, with
three different surface-tension discretisations, two different *read-out
definitions* and references that differ by a factor of ten in sigma and R.
Each round wrote its own document, and each document quotes the number its own
run produced.  The result is a citation trap: ``docs/verification_surface_
tension_{laplace,consistent,curvature}.md`` still open with -93.6 %, -75.6 % and
-87.6 %, which are (a) *band-definition* numbers -- the read-out that
``benchmarks/laplace_pressure_floor`` later showed is contaminated by the
free-surface density deficit -- and (b) in one case no longer reproducible from
the JSON the same repository ships (-75.6 % vs -78.6 %).

A reader cannot tell, from any single document, which number is current, which
definition it used, or whether it still matches its own artifact.  This ledger is
the single place that answers those three questions, and it is *checked by
tests*: every number below is read out of the committed JSON at generation time
(no value is typed by hand), every quoted historical claim is re-matched against
the document that makes it, and ``tests/test_verification_ledger.py`` fails if
the committed document drifts from the artifacts.

Nothing here re-runs a simulation: this is an analysis of existing evidence.

Usage
-----
    PYTHONPATH=src python benchmarks/evidence_ledger/ledger.py           # write
    PYTHONPATH=src python benchmarks/evidence_ledger/ledger.py --check   # verify
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_ledger.md"

# Platform acceptance standard (owner standard, docs/PROJECT_CHARTER.md):
# a quantity that disagrees with experiment/reference by more than 15 % is not
# acceptable; 3 % is the target for a converged profile.
GRADE_3PCT = 3.0
GRADE_15PCT = 15.0


def grade(error_pct: float) -> str:
    """Evidence grade implied by one relative error, by the platform rule."""
    magnitude = abs(error_pct)
    if magnitude <= GRADE_3PCT:
        return "REAL (<3 %)"
    if magnitude <= GRADE_15PCT:
        return "REAL (<15 %)"
    return "WITHHELD (>15 %)"


# --------------------------------------------------------------------------- #
# artifact rows: one measured capillary quantity each
# --------------------------------------------------------------------------- #
def _rel(measured: float, reference: float) -> float:
    return (measured - reference) / reference * 100.0


def _laplace_law():
    """Single-resolution drop, m2174-era read-out (p_center - band)."""
    d = json.loads(
        (ROOT / "benchmarks/laplace_law/validation_report.json").read_text())
    lap = d["laplace"]
    return {
        "metric": "Delta p [Pa]",
        "measured": lap["delta_p_measured_Pa"],
        "reference": lap["delta_p_theory_Pa"],
        "error_pct": lap["rel_error_pct"],
        "definition": "p_center - p(interface band); the band measured 0 Pa, so this "
                      "is an absolute read-out",
    }


def _csf_consistent():
    d = json.loads(
        (ROOT / "benchmarks/csf_consistent/validation_report.json").read_text())
    lap = d["relax"]["laplace"]
    return {
        "metric": "Delta p [Pa]",
        "measured": lap["delta_p_measured_Pa"],
        "reference": d["reference"]["delta_p_Pa"],
        "error_pct": lap["rel_error_pct"],
        "definition": "p_center - p_interface_band (historical band read-out)",
    }


def _csf_curvature_fit():
    d = json.loads(
        (ROOT / "benchmarks/csf_curvature_fit/validation_report.json").read_text())
    row = next(r for r in d["relax"] if r["curvature"] == "mls")
    lap = row["laplace"]
    return {
        "metric": "Delta p [Pa]",
        "measured": lap["delta_p_measured_Pa"],
        "reference": row["reference"]["delta_p_Pa"],
        "error_pct": lap["rel_error_pct"],
        "definition": "p_center - p_interface_band (historical band read-out)",
    }


def _surface_tension_laplace():
    d = json.loads(
        (ROOT / "benchmarks/surface_tension_laplace/validation_report.json").read_text())
    cases = d["one_sided_free_surface"]["cases"]
    worst = max(cases, key=lambda c: abs(c["rel_error"]))
    return {
        "metric": "Delta p, one-sided free surface (worst dx)",
        "measured": worst["dp_Pa"],
        "reference": None,
        "error_pct": worst["rel_error"] * 100.0,
        "definition": "colour jump only partly resolved (no outside particles)",
        "detail": f"dx = {worst['dx_m']:g} m, colour jump "
                  f"{worst['resolved_color_jump']:.2f}",
    }


def _laplace_plateau():
    d = json.loads(
        (ROOT / "benchmarks/laplace_plateau/validation_report.json").read_text())
    v = d["verdict"]
    return {
        "metric": "Delta p [Pa] (worst resolution)",
        "measured": None,
        "reference": None,
        "error_pct": v["worst_rel_error_pct"],
        "definition": v["definition"],
        "detail": v["reason"],
    }


def _laplace_droplet():
    d = json.loads(
        (ROOT / "benchmarks/laplace_droplet/results.json").read_text())
    v = d["verdict"]
    return {
        "metric": "dp_model / dp_theory, worst of 8 cases",
        "measured": None,
        "reference": None,
        "error_pct": v["max_relative_error"] * 100.0,
        "definition": "instrument-A read-out of the merged CSF model (pre-refactor)",
        "detail": f"{v['n_within_tolerance']}/{v['n_cases']} cases inside 15 %; "
                  f"converges_under_refinement = {v['converges_under_refinement']}",
    }


def _laplace_pressure_cutforce():
    d = json.loads(
        (ROOT / "benchmarks/laplace_pressure/validation_report.json").read_text())
    errors = [(c["sigma_eff_over_sigma"] - 1.0) * 100.0 for c in d["cases"]]
    best = min(errors, key=abs)
    return {
        "metric": "sigma_eff / sigma - 1, cut-force identity (best case)",
        "measured": None,
        "reference": None,
        "error_pct": best,
        "definition": "global cut-force path, best of "
                      f"{len(errors)} (radius, dx) cases",
        "detail": f"range {min(errors):+.1f} % .. {max(errors):+.1f} %",
    }


def _floor_absolute():
    d = json.loads(
        (ROOT / "benchmarks/laplace_pressure_floor/report.json").read_text())
    runs = d["runs"]
    worst = max(runs, key=lambda r: abs(r["absolute_time_average_tail"]["rel_error_pct"]))
    return {
        "metric": "Delta p [Pa], absolute + tail average (worst run)",
        "measured": worst["absolute_time_average_tail"]["delta_p_Pa"],
        "reference": d["reference"]["delta_p_Pa"],
        "error_pct": worst["absolute_time_average_tail"]["rel_error_pct"],
        "definition": "p_core - 0, time-averaged over the last 40 % of the window",
        "detail": f"dx = {worst['dx_m']:g} m, c0 = {worst['sound_speed']:g} m/s, "
                  f"{len(runs)} runs in the c0/resolution ladder",
    }


def _floor_band():
    d = json.loads(
        (ROOT / "benchmarks/laplace_pressure_floor/report.json").read_text())
    runs = d["runs"]
    worst = max(runs, key=lambda r: abs(r["definition_bandA"]["rel_error_pct"]))
    return {
        "metric": "Delta p [Pa], band read-out p_core - p(0.80-0.95R)",
        "measured": worst["definition_bandA"]["delta_p_Pa"],
        "reference": d["reference"]["delta_p_Pa"],
        "error_pct": worst["definition_bandA"]["rel_error_pct"],
        "definition": "same runs, historical band read-out (contaminated by the "
                      "free-surface density deficit)",
    }


ARTIFACTS = [
    {
        "key": "laplace_law",
        "json": "benchmarks/laplace_law/validation_report.json",
        "era": "m2174 CSF (delta_s-weighted)",
        "extract": _laplace_law,
        "doc": "docs/verification_surface_tension_laplace.md",
    },
    {
        "key": "laplace_droplet",
        "json": "benchmarks/laplace_droplet/results.json",
        "era": "merged CSF, instrument A",
        "extract": _laplace_droplet,
        "doc": "docs/verification_laplace_pressure_floor.md",
    },
    {
        "key": "csf_consistent",
        "json": "benchmarks/csf_consistent/validation_report.json",
        "era": "consistent CSF (MLS operators)",
        "extract": _csf_consistent,
        "doc": "docs/verification_surface_tension_consistent.md",
    },
    {
        "key": "csf_curvature_fit",
        "json": "benchmarks/csf_curvature_fit/validation_report.json",
        "era": "consistent CSF + fit_quadratic curvature",
        "extract": _csf_curvature_fit,
        "doc": "docs/verification_surface_tension_curvature.md",
    },
    {
        "key": "surface_tension_laplace",
        "json": "benchmarks/surface_tension_laplace/validation_report.json",
        "era": "one-sided free surface",
        "extract": _surface_tension_laplace,
        "doc": "docs/validation/surface-tension-laplace.md",
    },
    {
        "key": "laplace_plateau",
        "json": "benchmarks/laplace_plateau/validation_report.json",
        "era": "relaxation plateau",
        "extract": _laplace_plateau,
        "doc": "docs/verification_laplace_plateau.md",
    },
    {
        "key": "laplace_pressure",
        "json": "benchmarks/laplace_pressure/validation_report.json",
        "era": "cut-force identity",
        "extract": _laplace_pressure_cutforce,
        "doc": "docs/validation/surface-tension-laplace.md",
    },
    {
        "key": "laplace_pressure_floor_band",
        "json": "benchmarks/laplace_pressure_floor/report.json",
        "era": "same runs as below, historical read-out",
        "extract": _floor_band,
        "doc": "docs/verification_laplace_pressure_floor.md",
    },
    {
        "key": "laplace_pressure_floor_absolute",
        "json": "benchmarks/laplace_pressure_floor/report.json",
        "era": "same runs, C1-corrected read-out",
        "extract": _floor_absolute,
        "doc": "docs/verification_laplace_pressure_floor.md",
    },
]


# --------------------------------------------------------------------------- #
# historical claims: a document quotes a number, the artifact says something else
# --------------------------------------------------------------------------- #
CLAIMS = [
    {
        "doc": "docs/verification_surface_tension_laplace.md",
        "quote": "-93.6%",
        "artifact": None,
        "artifact_value": None,
        "definition_note": "band read-out of the legacy colour heuristic; the run "
                           "that produced it is no longer shipped as a field",
    },
    {
        "doc": "docs/verification_surface_tension_consistent.md",
        "quote": "-75.6%",
        "artifact": "benchmarks/csf_consistent/validation_report.json",
        "artifact_value": lambda: _csf_consistent()["error_pct"],
        "definition_note": "band read-out of the consistent CSF",
    },
    {
        "doc": "docs/verification_surface_tension_curvature.md",
        "quote": "-87.6%",
        "artifact": "benchmarks/csf_curvature_fit/validation_report.json",
        "artifact_value": lambda: _csf_curvature_fit()["error_pct"],
        "definition_note": "band read-out of the fit_quadratic curvature operator",
    },
]

# Claims whose document number cannot be reproduced from the artifact it ships
# with.  Recorded rather than hidden: a document that quotes a number its own
# JSON contradicts is the exact failure mode this ledger exists to catch.
KNOWN_DRIFT_TOLERANCE_PCT = 0.5


def _quote_value(quote: str) -> float:
    return float(re.search(r"[-+]?\d+(?:\.\d+)?", quote).group(0))


def build() -> dict:
    rows = []
    for entry in ARTIFACTS:
        measured = entry["extract"]()
        path = ROOT / entry["json"]
        rows.append({
            "key": entry["key"],
            "json": entry["json"],
            "json_mtime": path.stat().st_mtime_ns,
            "era": entry["era"],
            "doc": entry["doc"],
            "grade": grade(measured["error_pct"]),
            **measured,
        })

    claims = []
    for claim in CLAIMS:
        doc_path = ROOT / claim["doc"]
        text = doc_path.read_text()
        quoted = _quote_value(claim["quote"])
        current = (claim["artifact_value"]() if claim["artifact_value"]
                   is not None else None)
        claims.append({
            "doc": claim["doc"],
            "quote": claim["quote"],
            "present_in_doc": claim["quote"] in text,
            "quoted_value_pct": quoted,
            "artifact": claim["artifact"],
            "artifact_value_pct": current,
            "delta_pct": (None if current is None else current - quoted),
            "mapped": current is not None,
            "drifted": (current is not None
                        and abs(current - quoted) > KNOWN_DRIFT_TOLERANCE_PCT),
            "definition_note": claim["definition_note"],
        })
    return {"artifacts": rows, "claims": claims}


def readout_cost() -> dict:
    """How much of the historical miss was the read-out definition (C1)."""
    d = json.loads(
        (ROOT / "benchmarks/laplace_pressure_floor/report.json").read_text())
    band = [r["definition_bandA"]["rel_error_pct"] for r in d["runs"]]
    absolute = [r["absolute_time_average_tail"]["rel_error_pct"] for r in d["runs"]]
    return {
        "band_mean_pct": sum(band) / len(band),
        "absolute_mean_pct": sum(absolute) / len(absolute),
        "cost_mean_pct": (sum(band) - sum(absolute)) / len(band),
        "n_runs": len(d["runs"]),
        "reference_pa": d["reference"]["delta_p_Pa"],
    }


def _fmt(value, spec="{:.4g}"):
    if value is None:
        return "--"
    return spec.format(value)


def render(built: dict | None = None) -> str:
    built = build() if built is None else built
    cost = readout_cost()
    rows = built["artifacts"]
    claims = built["claims"]

    out = []
    out.append("# Young-Laplace / 表面张力证据总账（机器校验）\n")
    out.append(
        "本文件由 `benchmarks/evidence_ledger/ledger.py` 从仓库内已提交的 JSON 工件生成，\n"
        "**没有手抄数字**；`tests/test_verification_ledger.py` 校验本文件与工件一致、"
        "历史引用仍在原文档中、且分级符合平台判据。"
        "重新生成：`PYTHONPATH=src python benchmarks/evidence_ledger/ledger.py`。\n")
    out.append(
        "分级规则（平台标准）：|误差| ≤ 3 % → **REAL (<3 %)**；≤ 15 % → **REAL (<15 %)**；"
        f"> 15 % → **WITHHELD (>15 %)**。所有误差均为 \"(实测-基准)/基准\"。\n")

    out.append("\n## ① 全部压差测量（按年代排序）\n")
    out.append("| # | 工件 | 年代/方案 | 量 | 实测 | 基准 | 误差 | 分级 | 读数口径 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for n, row in enumerate(rows, 1):
        out.append(
            f"| {n} | `{row['json']}` | {row['era']} | {row['metric']} | "
            f"{_fmt(row['measured'])} | {_fmt(row['reference'])} | "
            f"{row['error_pct']:+.1f} % | **{row['grade']}** | {row['definition']} |")

    out.append("\n### 读数口径的代价（C1）\n")
    out.append(
        f"`benchmarks/laplace_pressure_floor/report.json` 的 {cost['n_runs']} 个算例"
        "（c0 × 分辨率阶梯）用**同一份流场**同时读两种口径：\n")
    out.append(
        f"- 历史口径 `p_core − p(0.80–0.95R)`：平均 **{cost['band_mean_pct']:+.1f} %**；\n"
        f"- 绝对压强 `p_core − 0` + 时间平均：平均 **{cost['absolute_mean_pct']:+.1f} %**。\n"
        f"- 口径本身贡献 **{cost['cost_mean_pct']:+.1f} 个百分点**"
        f"（基准 σ/R = {cost['reference_pa']:.0f} Pa）。\n")
    out.append(
        "因此上表中所有标注 \"band read-out\" 的行，其数值**系统性偏负**，"
        "不能与 C1 校正后的行直接比较；它们的历史价值是证明离散方案演进的相对改善，"
        "而不是当前精度。\n")

    out.append("\n## ② 历史引用 vs 工件（漂移检查）\n")
    out.append("| 文档 | 文中引用 | 文内存在 | 工件当前值 | 差 | 判定 | 口径 |")
    out.append("|---|---|---|---|---|---|---|")
    for claim in claims:
        if not claim["mapped"]:
            verdict = "无法定位：文档是历史快照"
            artifact_cell = "仓库内无对应字段"
            delta_cell = "--"
        else:
            verdict = ("**DRIFT**（不可复现）" if claim["drifted"]
                       else "一致（但口径为 band）")
            artifact_cell = (f"`{claim['artifact'].split('/')[-1]}` = "
                             f"{claim['artifact_value_pct']:+.1f} %")
            delta_cell = f"{claim['delta_pct']:+.1f} pt"
        out.append(
            f"| `{claim['doc'].split('/')[-1]}` | `{claim['quote']}` | "
            f"{'是' if claim['present_in_doc'] else '否'} | "
            f"{artifact_cell} | {delta_cell} | {verdict} | {claim['definition_note']} |")

    drifted = [c for c in claims if c["drifted"]]
    out.append("\n### 结论（引用纪律）\n")
    out.append(
        "1. 上述三份 `verification_surface_tension_*.md` 里引用的 −93.6 % / −75.6 % / −87.6 % "
        "一律视为**历史 band 口径值**，不是当前精度，不得作为现状引用。\n"
        "2. 当前可用于引用的压差结论只有 C1 校正后的行（ledger ① 中 definition 为 "
        "\"absolute\" 的行）：仍 >15 %，即 **WITHHELD**，残差归属自由面压强闭合 "
        "（`backlog_update_m2971.md` P0）。\n")
    if drifted:
        out.append(
            "3. 本轮自动检查发现 " + "、".join(
                f"`{c['doc'].split('/')[-1]}`（引 {c['quote']}，工件 {c['artifact_value_pct']:+.1f} %）"
                for c in drifted) +
            " 已与自身工件不一致：文档是历史快照，工件随后被重跑过。"
            "该差已登记在本表，不在文档里静默修改。\n")
    else:
        out.append("3. 本轮自动检查未发现文档与其工件不一致。\n")

    out.append("\n## ③ 未结项\n")
    out.append(
        "- **P0 自由面压强闭合**（`backlog_update_m2971.md`）：本 ledger 不含其数值——"
        "该实现由并行在飞单负责，本单刻意不重复实现、也不预写它的结果。"
        "落地后需向 `ARTIFACTS` 追加一行（closure ladder），本表即自动纳入。\n"
        "- **表面张力 3D / 多相**：仓库内无工件，故不在表内（无证据不列表）。\n")

    out.append("\n## ④ 局限\n")
    out.append(
        "- 本表只覆盖**压差/切割面力**这一族量；曲率、伪流、δ_s 归一化等判据仍在各 benchmark 自己的 "
        "validation_report.json 内，未纳入本表。\n"
        "- 各行的参考量并不统一（有的按 σ/R，有的按 2σ 切割力，有的按归一化压强），"
        "表中 \"基准\" 列已如实给出该行自己的基准；跨行比较只在同 metric 内有效。\n"
        "- `benchmarks/laplace/report.json` 的 `checks` 只存偏差量、未记单位，"
        "无法诚实地归一化成 (实测-基准)/基准，故未纳入本表。\n"
        "- 本表不重跑仿真，因此无法发现\"工件本身已过期于当前代码\"的情况；"
        "它发现的是\"文档与工件不一致\"。\n")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the committed document instead of writing it")
    args = ap.parse_args()
    text = render()
    if args.check:
        current = DOC.read_text() if DOC.exists() else ""
        if current != text:
            print(f"OUT OF DATE: {DOC.relative_to(ROOT)}", file=sys.stderr)
            return 1
        print(f"OK: {DOC.relative_to(ROOT)} matches the artifacts")
        return 0
    DOC.write_text(text)
    print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())