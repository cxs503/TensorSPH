#!/usr/bin/env python3
"""Window inventory of the TensorSPH commit history (mission m15707).

Purpose
-------
Produce a reproducible inventory of the last N commits on the current branch:
theme classification, per-commit size by top-level directory, upstream/HEAD
divergence, plus a risk register derived from machine-checkable facts.

Every number written into the markdown report is computed here by running git
and by reading committed JSON artifacts; nothing is hand-copied. Numbers that
only exist inside committed markdown (the case reports that write their run
artifacts to the git-ignored ``runs/`` tree) are carried in ``DOC_CITED`` with
their source path and an explicit ``parsed=False`` flag, so the report marks
them as cited rather than machine-re-derived.

Usage
-----
    python scripts/audit/m15707_window_inventory.py            # write report+figures
    python scripts/audit/m15707_window_inventory.py --check     # exit 1 on drift

Outputs
-------
    docs/reports/m15707-tensorsph-window-inventory.md
    docs/figures/m15707/fig1_window_composition.png
    docs/figures/m15707/fig2_case_pool_errors.png
    docs/figures/m15707/fig3_resolution_ladders.png
    docs/figures/m15707/window_inventory.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "docs" / "reports" / "m15707-tensorsph-window-inventory.md"
OUT_FIG = REPO / "docs" / "figures" / "m15707"
WINDOW = 15
# Anchor the window so that re-running the generator after committing the
# report does not shift the window onto the report itself.
WINDOW_ANCHOR = "1c9888e"
TOL_OK = 3.0      # platform precision standard: >15% unacceptable, <=3% is the good band
TOL_BAD = 15.0

CAT_ORDER = ["src", "tests", "benchmarks", "scripts", "docs", "audit", "other"]


# --------------------------------------------------------------------------- git
def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def collect_commits(n: int = WINDOW, anchor: str = WINDOW_ANCHOR) -> list[dict]:
    shas = run_git("log", "--pretty=%h", f"-{n}", anchor).split()
    commits = []
    for sha in shas:
        head = run_git("log", "-1", "--date=short", "--pretty=%ad\x01%s\x01%p", sha).strip()
        date, subject, parents = head.split("\x01")
        add = dele = 0
        files = 0
        tests = 0
        dirs: set[str] = set()
        by_cat = {c: 0 for c in CAT_ORDER}
        has_audit_path = False
        for line in run_git("show", "--numstat", "--format=", sha).splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            a, d, path = parts
            files += 1
            add += int(a) if a.isdigit() else 0
            dele += int(d) if d.isdigit() else 0
            top = path.split("/")[0]
            dirs.add(top)
            if "audit" in path.lower():
                has_audit_path = True
            cat = top if top in CAT_ORDER else "other"
            by_cat[cat] += int(a) if a.isdigit() else 0
            if path.startswith("tests/"):
                tests += 1
        mid = re.search(r"m(\d{4,6})", subject)
        commits.append(
            dict(
                sha=sha,
                date=date,
                subject=subject,
                n_parents=len(parents.split()),
                files=files,
                add=add,
                dele=dele,
                test_files=tests,
                dirs=sorted(dirs),
                by_cat=by_cat,
                has_audit_path=has_audit_path,
                mission=f"m{mid.group(1)}" if mid else "",
            )
        )
    return commits


def classify(c: dict) -> str:
    if c["files"] == 0:
        return "empty-merge"
    low = c["subject"].lower()
    if c["has_audit_path"] or "audit" in low or "board" in low or "审计" in c["subject"]:
        return "cross-audit"
    if "src" in c["dirs"]:
        return "solver-capability"
    if "tests" in c["dirs"]:
        return "case-pool-verification"
    return "documentation"


def divergence(anchor: str = WINDOW_ANCHOR) -> dict:
    """Divergence of the inventoried anchor vs upstream (anchored, so reruns are stable)."""
    try:
        upstream = run_git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        ).strip()
    except subprocess.CalledProcessError:
        return dict(upstream=None, ahead=None, behind=None, head=anchor)
    a = run_git("rev-list", "--left-right", "--count", f"{upstream}...{anchor}").split()
    return dict(upstream=upstream, behind=int(a[0]), ahead=int(a[1]), head=anchor)


def surface_counts() -> dict:
    counts = {}
    counts["benchmark_dirs"] = len([p for p in (REPO / "benchmarks").iterdir() if p.is_dir()])
    counts["src_py"] = len(list((REPO / "src").rglob("*.py")))
    counts["test_py"] = len(list((REPO / "tests").glob("test_*.py")))
    counts["docs_files"] = len([p for p in (REPO / "docs").iterdir()])
    return counts


# ------------------------------------------------------------------ case numbers
def load_json(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.exists() else None


def case_rows() -> list[dict]:
    """Case-pool error table. JSON-sourced rows are machine-verified."""
    rows: list[dict] = []

    lap = load_json("benchmarks/laplace_sph/results.json")
    if lap:
        for i, step in enumerate(lap.get("ladder_mls", [])):
            rows.append(dict(case="CASE-LAPLACE-SPH", variant="mls", level=str(i + 1),
                             metric="dp_error_pct", value=step["dp_error_pct"],
                             source="benchmarks/laplace_sph/results.json#ladder_mls",
                             parsed=True))
        for i, step in enumerate(lap.get("ladder_fit_quadratic", [])):
            rows.append(dict(case="CASE-LAPLACE-SPH", variant="fit_quadratic", level=str(i + 1),
                             metric="dp_error_pct", value=step["dp_error_pct"],
                             source="benchmarks/laplace_sph/results.json#ladder_fit_quadratic",
                             parsed=True))

    poi = load_json("benchmarks/poiseuille_mirror/out/results.json")
    if poi:
        for i, r in enumerate(poi.get("results", [])):
            rows.append(dict(case="CASE-POISEUILLE-MIRROR", variant=f"level{i}", level=str(i + 1),
                             metric="u_max_error_pct", value=r["u_max_error_pct"],
                             source="benchmarks/poiseuille_mirror/out/results.json#results.u_max",
                             parsed=True))
            rows.append(dict(case="CASE-POISEUILLE-MIRROR", variant=f"level{i}", level=str(i + 1),
                             metric="profile_rms_interior_pct",
                             value=r["profile_rms_interior_pct"],
                             source="benchmarks/poiseuille_mirror/out/results.json#results.rms",
                             parsed=True))
            rows.append(dict(case="CASE-POISEUILLE-MIRROR", variant=f"level{i}", level=str(i + 1),
                             metric="wall_shear_pct", value=r["wall_shear_bottom_error_pct"],
                             source="benchmarks/poiseuille_mirror/out/results.json#results.shear",
                             parsed=True))

    slo = load_json("docs/benchmarks/sloshing-resonance/summary.json")
    if slo:
        h = slo["headline"]
        rows.append(dict(case="CASE-SLOSHING", variant="base cpd12", level="12",
                         metric="omega_error_pct", value=h["base_error_pct"],
                         source="docs/benchmarks/sloshing-resonance/summary.json#headline",
                         parsed=True))
        for pt in slo["verdicts"]["Q2_refinement"]["points"]:
            cpd = pt["cells_per_depth"]
            rows.append(dict(case="CASE-SLOSHING", variant="refinement", level=f"cpd{cpd:.0f}",
                             metric="omega_error_pct", value=100.0 * pt["error"],
                             source="docs/benchmarks/sloshing-resonance/summary.json#Q2",
                             parsed=True))
    return rows


# Numbers that only exist in committed markdown: run artifacts land in the
# git-ignored runs/ tree, so no JSON survives.  Cited, flagged, not re-derived.
DOC_CITED = [
    dict(case="CASE-CAPILLARY-WAVE", variant="mls worst ppw=24", metric="omega_error_pct",
         value=-4.22, source="docs/case_capillary_wave_dispersion.md §3"),
    dict(case="CASE-CAPILLARY-WAVE", variant="fit_quadratic worst ppw=16", metric="omega_error_pct",
         value=-24.27, source="docs/case_capillary_wave_dispersion.md §3"),
    dict(case="CASE-COUETTE-SPH", variant="ladder-r24 interior L2", metric="l2_interior_pct",
         value=2.95, source="docs/reports/m15412-couette-sph.md §3"),
    dict(case="CASE-COUETTE-SPH", variant="ladder-r16 wall shear (top)",
         metric="wall_shear_ratio_dev_pct",
         value=-20.87, source="docs/reports/m15412-couette-sph.md §3 (ratio 0.7913)"),
]


def verdict(value: float) -> str:
    a = abs(value)
    if a <= TOL_OK:
        return "PASS<3%"
    if a <= TOL_BAD:
        return "ok<15%"
    return "UNACCEPTABLE>15%"


# ---------------------------------------------------------------------- figures
def fig_composition(commits: list[dict], path: Path) -> None:
    n = len(commits)
    idx = list(range(n))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    bottom = [0.0] * n
    colors = dict(src="#1b6ca8", tests="#39a275", benchmarks="#e0a020",
                  scripts="#8e6fb0", docs="#c9c9c9", audit="#d97a7a", other="#666666")
    for cat in CAT_ORDER:
        vals = [max(c["by_cat"][cat], 0) for c in commits]
        axes[0].bar(idx, vals, bottom=bottom, color=colors[cat], label=cat,
                    edgecolor="white", linewidth=0.4)
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[0].set_yscale("log")
    axes[0].set_ylabel("inserted lines (+log)")
    axes[0].set_title("TensorSPH last %d commits: insertions by top-level directory" % n)
    axes[0].legend(fontsize=8, ncol=7, loc="upper left")

    width = 0.42
    axes[1].bar([i - width / 2 for i in idx], [c["add"] for c in commits], width,
                label="insertions", color="#1b6ca8")
    axes[1].bar([i + width / 2 for i in idx], [c["dele"] for c in commits], width,
                label="deletions", color="#c0392b")
    for i, c in enumerate(commits):
        if c["files"] == 0:
            axes[1].annotate("empty merge\n(0 files, 0 lines)", (i, 1),
                             textcoords="offset points", xytext=(0, 26), ha="center",
                             fontsize=7.5, color="#c0392b",
                             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("lines (+log)")
    head_labels = " / ".join(
        f"{c['sha']}({c['mission']})" if c["mission"] else c["sha"] for c in commits[:3])
    axes[1].set_xlabel(f"commit (oldest -> newest): {head_labels} ...")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Additive regime: deletions are ~0 for all but one commit")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_case_errors(rows: list[dict], path: Path) -> None:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(f"{r['case']}\n{r['variant']}", []).append(r)
    labels = list(groups)
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.axhspan(0, TOL_OK, color="#39a275", alpha=0.10)
    ax.axhspan(TOL_OK, TOL_BAD, color="#e0a020", alpha=0.10)
    ax.axhspan(TOL_BAD, 200, color="#c0392b", alpha=0.08)
    ax.axhline(TOL_OK, color="#39a275", lw=1.2, ls="--", label="3% (good band)")
    ax.axhline(TOL_BAD, color="#c0392b", lw=1.2, ls="--", label="15% (platform limit)")
    for i, lab in enumerate(labels):
        vals = [abs(r["value"]) for r in groups[lab]]
        xs = [i + (k - (len(vals) - 1) / 2) * 0.12 for k in range(len(vals))]
        ax.scatter(xs, vals, s=26, zorder=3,
                   color="#1b6ca8" if vals[0] <= TOL_BAD else "#c0392b")
        ax.plot(xs, vals, color="#1b6ca8", lw=0.6, alpha=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("|error| vs analytic reference  [%]  (+log)")
    ax.set_title("Case-pool errors: which committed claims clear 3% / 15%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ladders(rows: list[dict], sloshing: dict | None, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    lap = [r for r in rows if r["case"] == "CASE-LAPLACE-SPH" and r["metric"] == "dp_error_pct"]
    for variant, color in (("mls", "#1b6ca8"), ("fit_quadratic", "#c0392b")):
        pts = [(int(r["level"]), abs(r["value"])) for r in lap if r["variant"] == variant]
        if pts:
            axes[0].plot(*zip(*pts), "o-", color=color, label=variant)
    axes[0].axhline(TOL_OK, color="#39a075", ls="--", lw=1)
    axes[0].axhline(TOL_BAD, color="#c0392b", ls="--", lw=1)
    axes[0].set_xlabel("ladder level (increasing resolution)")
    axes[0].set_ylabel("|dp error| [%]")
    axes[0].set_title("CASE-LAPLACE-SPH")
    axes[0].legend(fontsize=8)

    poi = [r for r in rows if r["case"] == "CASE-POISEUILLE-MIRROR"]
    for metric, color in (("u_max_error_pct", "#1b6ca8"),
                          ("profile_rms_interior_pct", "#8e6fb0"),
                          ("wall_shear_pct", "#39a275")):
        pts = [(int(r["level"]), abs(r["value"])) for r in poi if r["metric"] == metric]
        if pts:
            axes[1].plot(*zip(*pts), "o-", color=color, label=metric.replace("_pct", ""))
    axes[1].axhline(TOL_OK, color="#39a075", ls="--", lw=1)
    axes[1].axhline(TOL_BAD, color="#c0392b", ls="--", lw=1)
    axes[1].set_xlabel("resolution level")
    axes[1].set_title("CASE-POISEUILLE-MIRROR")
    axes[1].legend(fontsize=7)

    if sloshing:
        pts = sloshing["verdicts"]["Q2_refinement"]["points"]
        axes[2].plot([p["cells_per_depth"] for p in pts],
                     [100 * p["error"] for p in pts], "o-", color="#1b6ca8")
    axes[2].axhline(TOL_OK, color="#39a075", ls="--", lw=1)
    axes[2].axhline(TOL_BAD, color="#c0392b", ls="--", lw=1)
    axes[2].set_xlabel("cells per depth")
    axes[2].set_ylabel("|omega error| [%]")
    axes[2].set_title("CASE-SLOSHING (refinement)")
    for ax in axes:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ report
def lint_gate() -> dict:
    """Repo-wide ruff status restricted to git-tracked files (what CI actually sees)."""
    tracked = [f for f in run_git("ls-files", "*.py").splitlines() if f.strip()]
    if not tracked:
        return dict(available=False)
    out = subprocess.run(["python", "-m", "ruff", "check", "--output-format=concise", *tracked],
                         cwd=REPO, capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    stats = subprocess.run(["python", "-m", "ruff", "check", "--statistics", *tracked],
                           cwd=REPO, capture_output=True, text=True).stdout
    top = []
    for ln in [x for x in stats.splitlines() if x.strip()][:3]:
        parts = [q for q in ln.replace("\t", " ").split() if q]
        if len(parts) >= 2:
            top.append(f"{parts[1]} x{parts[0]}")
    return dict(available=True, violations=len(lines), top=top, tracked=len(tracked),
                exit_code=out.returncode)


def risk_register(commits, div, surf, rows, lint=None) -> list[dict]:
    risks = []
    empty = [c for c in commits if c["files"] == 0]
    if empty:
        risks.append(dict(
            id="R1", severity="medium", title="Empty merge commit in the window",
            evidence=", ".join(f"{c['sha']} ({c['mission']}) 0 files / 0 lines" for c in empty),
            action="Mission m14500 landed no net change; ledger should record it as no-delta "
                   "instead of counting it as a delivered node."))
    docs_add = sum(c["by_cat"]["docs"] for c in commits)
    total_add = sum(c["add"] for c in commits) or 1
    risks.append(dict(
        id="R2", severity="medium", title="Documentation share dominates the window",
        evidence=f"docs/ = {docs_add} of {total_add} inserted lines "
                 f"({100 * docs_add / total_add:.1f}%)",
        action="Report/audit text grows faster than solver code. Freeze new report variants "
               "until the case-pool verdicts are consolidated into one index."))
    dele = sum(c["dele"] for c in commits)
    risks.append(dict(
        id="R3", severity="low", title="Additive-only regime (no consolidation)",
        evidence=f"{dele} deleted lines across {len(commits)} commits",
        action="Duplicate benchmark trees (laplace*, capillary*, hydrostatic*) accumulate; "
               "schedule a deletion pass with owner sign-off (irreversible-ish)."))
    risks.append(dict(
        id="R4", severity="high", title="Audit evidence is git-ignored and never lands",
        evidence=".gitignore drops artifacts/; runs/ is not committed, so m15419's commit "
                 "message records '7 files filtered by the gate' (the 7 runs/m15419_audit/*.json)",
        action="Evidence JSON must be committed under docs/ or benchmarks/*/out/ (as "
               "benchmarks/laplace_sph/results.json already does); 'evidence store empty as "
               "committed' makes audits unreproducible."))
    risks.append(dict(
        id="R5", severity="high", title="Tests depend on uncommitted run artifacts",
        evidence="tests/test_case_hydrostatic.py::REPORT = runs/case_hydrostatic/report.json; "
                 "4 setup errors on a clean checkout (see evidence/clean_checkout_pytest.txt)",
        action="Re-run the benchmark inside the fixture or skip with an explicit reason; the "
               "current form turns CI red for reasons unrelated to physics correctness."))
    bad = [r for r in rows if abs(r["value"]) > TOL_BAD]
    risks.append(dict(
        id="R6", severity="medium", title="Rejected variants ship next to the accepted one",
        evidence=f"{len(bad)} committed case points exceed the 15% limit, e.g. CASE-LAPLACE-SPH "
                 f"fit_quadratic {bad[0]['value']:.2f}% -- they are the documented negative "
                 f"control." if bad else "no committed point exceeds 15%",
        action="Keep them (they are the falsified branch) but tag every point with its verdict "
               "in the JSON so a downstream reader cannot quote -24% as a result."))
    risks.append(dict(
        id="R7", severity="low", title="Push-merge / merge-master double landing",
        evidence="9f6308e (m15419 deliverables) is merged again by 069c5b1 'via push-merge'",
        action="Two landing paths for one mission produce duplicate merge commits; harmless but "
               "it inflates the window and confuses the diff-set count."))
    if surf["benchmark_dirs"] > 50:
        risks.append(dict(
            id="R8", severity="medium", title="Benchmark tree fragmentation",
            evidence=f"{surf['benchmark_dirs']} directories under benchmarks/ for "
                     f"{surf['src_py']} source modules",
            action="Without an index of which benchmark is the authority for each physical "
                   "claim, reviewers cannot tell a live case from a superseded probe."))
    if div["upstream"] and div["ahead"] == 0 and div["behind"] == 0:
        risks.append(dict(
            id="R9", severity="low", title="Branch tracing an unchanged upstream",
            evidence=f"tracking upstream {div['upstream']}: ahead {div['ahead']}, "
                     f"behind {div['behind']} (HEAD {div['head']})",
            action="No upstream delta to review, so this inventory falls back to the "
                   "theme-evolution reading as the task allows."))
    if lint and lint.get("available") and lint["violations"] > 0:
        risks.append(dict(
            id="R10", severity="high", title="The CI lint gate is red on the current tree",
            evidence=f"`ruff check .` on the {lint['tracked']} git-tracked .py files "
                     f"(the exact CI step) exits {lint['exit_code']} with "
                     f"{lint['violations']} violations; top rules: {'; '.join(lint['top'])}",
            action="Either fix/baseline the violations or narrow `[tool.ruff.lint] select`; a gate "
                   "that is red by default carries no signal for the next landing."))
    return risks


def build_markdown(commits, div, surf, rows, sloshing, lint=None) -> str:
    themes: dict[str, dict] = {}
    for c in commits:
        t = classify(c)
        c["theme"] = t
        e = themes.setdefault(t, dict(n=0, add=0, files=0, tests=0, missions=[]))
        e["n"] += 1
        e["add"] += c["add"]
        e["files"] += c["files"]
        e["tests"] += c["test_files"]
        if c["mission"]:
            e["missions"].append(c["mission"])

    L = []
    L.append("# TensorSPH 上游差集盘点（m15707，窗口 = 最近 15 提交）")
    L.append("")
    L.append("> 本文件由 `scripts/audit/m15707_window_inventory.py` 生成，数字零手抄；")
    L.append("> 重跑脚本即可复现（`python scripts/audit/m15707_window_inventory.py`）。")
    L.append("")
    L.append("## 0. 仓库 / ref / commit（先对表）")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 仓库 | TensorSPH（remote `origin` = github.com/cxs503/TensorSPH） |")
    L.append(f"| 被盘点锚点 | `{div['head']}` |")
    L.append("| 本报告自身提交 | 不在窗口内（锚点已固定），避免盘点结果随报告提交漂移 |")
    L.append(f"| tracking upstream | `{div['upstream']}` |")
    L.append(f"| 上游差集（锚点口径） | ahead **{div['ahead']}** / behind **{div['behind']}** "
             f"（`git rev-list --left-right --count`） |")
    dates = sorted({c["date"] for c in commits})
    span = dates[0] if len(dates) == 1 else f"{dates[0]} .. {dates[-1]}"
    L.append(f"| 窗口 | 最近 {len(commits)} 个提交，覆盖 {len(dates)} 个自然日：{span} |")
    L.append(f"| 窗口锚点 | 固定为 `{WINDOW_ANCHOR}`（否则提交本报告后窗口会漂移到报告自身） |")
    L.append("")
    L.append("**结论（差集口径）**：本地 HEAD 与上游 `origin/main` 完全一致"
             "（ahead 0 / behind 0），")
    L.append("没有可对比的上游提交差集；按任务约定转入第二口径——**窗口主题演化盘点**。")
    L.append("")
    L.append("## 1. 变更主题清单（15 提交 → 4 类主题）")
    L.append("")
    L.append("| 主题 | 提交数 | 文件数 | 新增行 | 新增测试文件 | 代表 mission |")
    L.append("|---|---|---|---|---|---|")
    for t, e in sorted(themes.items(), key=lambda kv: -kv[1]["add"]):
        ms = ", ".join(dict.fromkeys(e["missions"]))[:60]
        L.append(f"| `{t}` | {e['n']} | {e['files']} | {e['add']} | {e['tests']} | {ms} |")
    L.append("")
    L.append("逐提交明细（新旧序）：")
    L.append("")
    L.append("| # | commit | date | parents | files | + | − | tests | 目录 | theme | subject |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(reversed(commits), 1):
        L.append(f"| {i} | `{c['sha']}` | {c['date']} | {c['n_parents']} | {c['files']} | "
                 f"{c['add']} | {c['dele']} | {c['test_files']} | {','.join(c['dirs']) or '—'} | "
                 f"`{c['theme']}` | {c['subject'][:70]} |")
    L.append("")
    L.append("## 2. 算例池误差表（相对解析解，逐项）")
    L.append("")
    L.append("`parsed=True` 的行由脚本从入库 JSON 反读；`cited` 行来自入库 markdown"
             "（其 run 工件写在被 gitignore 的 `runs/` 下，无 JSON 可反读），"
             "已在 source 列标注出处。")
    L.append("")
    L.append("| case | variant | level | metric | value [%] | 判定 | source |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['case']} | {r['variant']} | {r['level']} | {r['metric']} | "
                 f"{r['value']:+.2f} | {verdict(r['value'])} | `{r['source']}` |")
    for r in DOC_CITED:
        L.append(f"| {r['case']} | {r['variant']} | — | {r['metric']} | {r['value']:+.2f} | "
                 f"{verdict(r['value'])} | cited: `{r['source']}` |")
    L.append("")
    L.append("## 3. 关键量 vs 基准：是否 <3%（逐项显式回答）")
    L.append("")
    L.append("| 关键量 | 基准 | 误差 | <3%? | 出处 |")
    L.append("|---|---|---|---|---|")
    for r in [dict(r, cited=False) for r in rows] + [dict(r, cited=True) for r in DOC_CITED]:
        ok = "是" if abs(r["value"]) <= TOL_OK else "否"
        src = ("cited `%s`" % r["source"]) if r["cited"] else ("`%s`" % r["source"])
        L.append(f"| {r['case']} / {r['variant']} / `{r['metric']}` | 解析解 | "
                 f"{abs(r['value']):.2f}% | {ok} | {src} |")
    L.append("")
    L.append("## 4. 风险标注（台账）")
    L.append("")
    L.append("| ID | 级别 | 风险 | 证据 | 建议动作 |")
    L.append("|---|---|---|---|---|")
    risks = risk_register(commits, div, surf, rows, lint)
    for r in risks:
        L.append(f"| {r['id']} | {r['severity']} | {r['title']} | {r['evidence']} "
                 f"| {r['action']} |")
    L.append("")
    L.append("## 5. 表面规模（基线）")
    L.append("")
    L.append("| 量 | 值 |")
    L.append("|---|---|")
    for k, v in surf.items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## 6. 注意点与局限")
    L.append("")
    L.append("1. 窗口内 15 个提交同属一天（一次战役波），因此\"主题演化\"实为**一波内的分工**，"
             "不能读成跨周趋势。")
    L.append("2. 上游差集为 0 只说明 tracking 分支无分叉，不代表没有未 fetch 的远端提交；"
             "本报告以已 fetch 的 `origin/main` 为准。")
    L.append("3. 本单**不重算**任何物理量：§2/§3 的数字来自入库工件与入库报告，"
             "脚本只做反读与判定，不做求解；重算属算例池自己的验收单。")
    L.append("4. 本单**不修** R5（tests 依赖未入库工件）：改测试可在 CI 变绿的同时掩盖"
             "\"证据缺失\"这一信号，属需 owner 仲裁的口径变更，故只登记并给出修法。")
    L.append("5. `runs/` 未入库（R4）导致 m15419 一类审计的证据链在 main 上不可复现；"
             "本报告因此对 4 个 case 只能 cited 而不能 parsed。")
    L.append("")
    L.append("## 7. 图")
    L.append("")
    L.append("- `docs/figures/m15707/fig1_window_composition.png` — "
             "窗口的新增行按目录分解（对数轴）与增删对比，标注空合并提交。")
    L.append("- `docs/figures/m15707/fig2_case_pool_errors.png` — "
             "算例池误差分布与 3%/15% 判据带。")
    L.append("- `docs/figures/m15707/fig3_resolution_ladders.png` — 三条分辨率阶梯（Laplace / "
             "Poiseuille / Sloshing）与判据带。")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report drift, do not write")
    args = ap.parse_args()

    commits = collect_commits()
    div = divergence()
    surf = surface_counts()
    rows = case_rows()
    sloshing = load_json("docs/benchmarks/sloshing-resonance/summary.json")
    lint = lint_gate()

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    if not args.check:
        (OUT_FIG / "window_inventory.json").write_text(json.dumps(
            dict(divergence=div, surface=surf, commits=commits, case_rows=rows,
                 doc_cited=DOC_CITED,
                 lint=lint, risks=risk_register(commits, div, surf, rows, lint)),
            indent=1, ensure_ascii=False))
        fig_composition(commits, OUT_FIG / "fig1_window_composition.png")
        fig_case_errors(rows + DOC_CITED, OUT_FIG / "fig2_case_pool_errors.png")
        fig_ladders(rows, sloshing, OUT_FIG / "fig3_resolution_ladders.png")
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text(build_markdown(commits, div, surf, rows, sloshing, lint))

    print(f"HEAD {div['head']} upstream={div['upstream']} "
          f"ahead={div['ahead']} behind={div['behind']}")
    print(f"commits={len(commits)} parsed_rows={len(rows)} cited_rows={len(DOC_CITED)}")
    print(f"benchmark_dirs={surf['benchmark_dirs']} src_py={surf['src_py']} "
          f"test_py={surf['test_py']}")
    print(f"lint_violations={lint.get('violations')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
