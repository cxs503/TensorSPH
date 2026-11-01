#!/usr/bin/env python3
"""Upstream diff-set audit for TensorSPH.

Compares the remote-tracking upstream refs (``origin/*``) with the local HEAD and
produces, fully programmatically:

* ahead/behind divergence per remote branch,
* the set of commits present upstream but absent from HEAD,
* patch-id equivalence (``git cherry``) for those commits,
* merge-ability against HEAD (``git merge-tree --write-tree`` conflict surface),
* a *semantic* coverage check: are the symbols introduced by an unmerged branch
  already present in HEAD's sources?  (patch-id equivalence and semantic
  equivalence are different questions -- see docs/upstream-diff-m1633.md),
* a test-coverage delta: which branch test functions have no HEAD counterpart,
* two figures and a markdown table bundle.

No torch import is needed: this is a pure git/PLY analysis, so it runs even when
the CUDA stack is unavailable.

Usage::

    python3 benchmarks/audit/upstream_diff.py \
        --repo . --out docs/audit/m1633
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# git plumbing
# --------------------------------------------------------------------------- #


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def resolve_head(repo: Path) -> dict:
    return {
        "sha": git(repo, "rev-parse", "HEAD").strip(),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "committed": git(repo, "log", "-1", "--format=%cI").strip(),
    }


def remote_branches(repo: Path) -> list[str]:
    out = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/")
    return [line.strip() for line in out.splitlines() if line.strip()]


def ahead_behind(repo: Path, rev: str) -> tuple[int, int]:
    """(commits in HEAD not in rev, commits in rev not in HEAD)."""
    out = git(repo, "rev-list", "--left-right", "--count", f"HEAD...{rev}").split()
    return int(out[0]), int(out[1])


def behind_base(repo: Path, rev: str) -> int:
    base = git(repo, "merge-base", "HEAD", rev).strip()
    return int(git(repo, "rev-list", "--count", f"{base}..HEAD").strip())


def unique_commits(repo: Path, rev: str) -> list[dict]:
    fmt = "%H%x1f%h%x1f%aI%x1f%an%x1f%s"
    out = git(repo, "log", f"--format={fmt}", f"HEAD..{rev}")
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, short, date, author, subject = line.split("\x1f")
        commits.append(
            {
                "sha": sha,
                "short": short,
                "date": date[:10],
                "author": author,
                "subject": subject,
            }
        )
    return commits


def cherry_equivalence(repo: Path, rev: str) -> dict[str, bool]:
    """sha -> True if an equivalent patch is already in HEAD (``-`` in git cherry)."""
    out = git(repo, "cherry", "-v", "HEAD", rev)
    result: dict[str, bool] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        marker, sha = line.split()[:2]
        result[sha] = marker == "-"
    return result


def merge_conflicts(repo: Path, rev: str) -> dict:
    proc = subprocess.run(
        ["git", "merge-tree", "--write-tree", "HEAD", rev],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    text = proc.stdout + proc.stderr
    conflicted = sorted({m.group(1) for m in re.finditer(r"Merge conflict in (\S+)", text)})
    return {"clean": proc.returncode == 0 and not conflicted,
                "returncode": proc.returncode,
                "conflicted_files": conflicted}


def changed_files(repo: Path, rev: str) -> list[str]:
    base = git(repo, "merge-base", "HEAD", rev).strip()
    out = git(repo, "diff", "--name-only", f"{base}..{rev}")
    return [line for line in out.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# semantic coverage: are the branch's new symbols already in HEAD?
# --------------------------------------------------------------------------- #

DEF_RE = re.compile(r"^\+\s*(?:async\s+)?def\s+([A-Za-z_]\w*)")
TEST_RE = re.compile(r"^\+\s*def\s+(test_\w+)")


def added_names(repo: Path, rev: str, paths: list[str]) -> dict[str, list[str]]:
    base = git(repo, "merge-base", "HEAD", rev).strip()
    out = git(repo, "diff", f"{base}..{rev}", "--", *paths)
    defs, tests = [], []
    for line in out.splitlines():
        if m := DEF_RE.match(line):
            (tests if m.group(1).startswith("test_") else defs).append(m.group(1))
    return {"defs": sorted(set(defs)), "tests": sorted(set(tests))}


def head_symbol_index(repo: Path) -> dict[str, str]:
    """name -> file, for every def/class in HEAD's src/ tree."""
    index: dict[str, str] = {}
    for path in sorted((repo / "src").rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", line)
            if m:
                index.setdefault(m.group(1), str(path.relative_to(repo)))
    return index


def rename_candidates(absent: list[str], pool: list[str], cutoff: float = 0.55) -> dict:
    """Best fuzzy match in ``pool`` for every name in ``absent`` (difflib ratio)."""
    import difflib

    out = {}
    for name in absent:
        best, best_r = None, 0.0
        for cand in pool:
            r = difflib.SequenceMatcher(None, name, cand).ratio()
            if r > best_r:
                best, best_r = cand, r
        if best and best_r >= cutoff:
            out[name] = {"match": best, "ratio": round(best_r, 3)}
    return out


def head_test_names(repo: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted((repo / "tests").glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^def\s+(test_\w+)", line)
            if m:
                names.add(m.group(1))
    return names


# --------------------------------------------------------------------------- #
# theme classification of commits
# --------------------------------------------------------------------------- #

# Ordered: first match wins.  Purely lexical, deterministic and auditable.
THEMES: list[tuple[str, str]] = [
    ("merge-integration", r"^merge (pull request|branch|remote|:)|^merge:"),
    ("surface-tension", r"surface.?tension|csf|brackbill"),
    ("wall-boundary", r"wall pressure|\bwall\b|adami|no.?slip|friction|boundary"),
    ("free-surface", r"free.?surface|tensile|density diffusion|antuono"),
    ("time-integration", r"adaptive|\bdt\b|time step|courant|stable_dt"),
    ("neighbor-search", r"neighbou?r|linked.?cell|verlet|skin|pair"),
    ("dam-break", r"dam.?break"),
    ("droplet-impact", r"droplet|impact|surface tension droplet"),
    ("gpu-scaling", r"gpu|cuda|scaling|speedup"),
    ("validation-gate", r"gate|acceptance|manifest|regression|验证"),
    ("audit-review", r"audit|review|审计|评审|follow|跟进"),
    ("docs-report", r"^docs?:|report|readme|文档|报告|backlog"),
    ("benchmark", r"bench|sweep|基准"),
    ("test-suite", r"test|pytest|比较"),
    ("release", r"^\d+\.\d+\.\d+|release|version"),
]


def classify(subject: str) -> str:
    low = subject.lower()
    for name, pattern in THEMES:
        if re.search(pattern, low):
            return name
    return "other"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(rows: list[dict], theme_rows: list[dict], figs: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    produced: list[dict] = []

    # --- Figure 1: divergence of every remote branch vs local HEAD -----------
    live = [r for r in rows if r["ahead"] or r["behind"]]
    labels = [r["branch"] for r in live]
    ahead = [r["ahead"] for r in live]
    behind = [r["behind"] for r in live]

    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(live) + 1.9))
    y = range(len(live))
    ax.barh([i + 0.2 for i in y], ahead, height=0.38, color="#c0392b",
            label="ahead of HEAD (unmerged)")
    ax.barh([i - 0.2 for i in y], behind, height=0.38, color="#2c7fb8",
            label="behind HEAD (stale base)")
    for i, r in enumerate(live):
        ax.text(r["ahead"] + 0.4, i + 0.2, str(r["ahead"]), va="center", fontsize=8)
        ax.text(r["behind"] + 0.4, i - 0.2, str(r["behind"]), va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("commits")
    ax.set_title("TensorSPH: remote upstream vs local HEAD divergence", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p1 = figs / "upstream_divergence_m1633.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    produced.append({"name": p1.name, "path": str(p1),
                     "caption": "Branch divergence (ahead/behind) vs local HEAD"})

    # --- Figure 2: theme evolution timeline ---------------------------------
    themes = sorted({r["theme"] for r in theme_rows})
    colors = {t: c for t, c in zip(themes, plt.cm.tab20.colors)}
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    order = {t: i for i, t in enumerate(themes)}
    for r in theme_rows:
        x = datetime.fromisoformat(r["date"])
        yv = order[r["theme"]]
        marker = {"upstream-unmerged": "*", "head-15": "D"}.get(r["scope"], "o")
        size = {"upstream-unmerged": 130, "head-15": 60}.get(r["scope"], 30)
        ax.scatter(x, yv, color=colors[r["theme"]], marker=marker, s=size,
                   edgecolor="k", linewidth=0.5, alpha=0.9, zorder=3)
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels(themes, fontsize=8)
    ax.set_xlabel("commit date")
    ax.set_title(
        "TensorSPH theme evolution: HEAD top-15 (D) / upstream-unmerged (*) / history (o)",
        fontsize=10)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    p2 = figs / "theme_timeline_m1633.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    produced.append({"name": p2.name, "path": str(p2),
                     "caption": "Theme evolution timeline of commits"})
    return produced


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="docs/audit/m1633")
    ap.add_argument("--figs", default=None, help="defaults to <out>/figs")
    ap.add_argument("--head-window", type=int, default=15)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = Path(args.figs) if args.figs else out / "figs"

    head = resolve_head(repo)
    branches = remote_branches(repo)
    index = head_symbol_index(repo)
    htests = head_test_names(repo)

    rows: list[dict] = []
    unmerged: list[dict] = []
    for br in branches:
        a, b = ahead_behind(repo, br)
        rows.append({"branch": br, "ahead": a, "behind": b})
        if b == 0:
            continue
        commits = unique_commits(repo, br)
        equiv = cherry_equivalence(repo, br)
        conflicts = merge_conflicts(repo, br)
        names = added_names(repo, br, ["src", "tests"])
        present = [n for n in names["defs"] if n in index]
        missing = [n for n in names["defs"] if n not in index]
        tests_missing = [t for t in names["tests"] if t not in htests]
        renames = rename_candidates(missing, sorted(index))
        test_renames = rename_candidates(tests_missing, sorted(htests))
        unmerged.append(
            {
                "branch": br,
                "ahead": b,
                "behind_base": behind_base(repo, br),
                "commits": [
                    {**c, "patch_already_in_head": equiv.get(c["sha"], False)}
                    for c in commits
                ],
                "patch_equivalent": sum(1 for c in commits if equiv.get(c["sha"])),
                "patch_novel": sum(1 for c in commits if not equiv.get(c["sha"])),
                "changed_files": changed_files(repo, br),
                "merge_clean": conflicts["clean"],
                "conflicted_files": conflicts["conflicted_files"],
                "branch_defs": names["defs"],
                "defs_present_in_head": present,
                "defs_absent_from_head": missing,
                "branch_tests_unmatched_in_head": tests_missing,
                "defs_rename_candidates": renames,
                "tests_rename_candidates": test_renames,
                "defs_unmatched_no_rename": [
                    n for n in missing if n not in renames]
            }
        )

    # ---- machine self-check: independent recount of every headline number ---
    selfcheck = {"checks": [], "failed": 0}

    def check(name: str, got, want) -> None:
        ok = got == want
        selfcheck["checks"].append({"name": name, "got": got, "want": want, "ok": ok})
        if not ok:
            selfcheck["failed"] += 1

    for u in unmerged:
        check(f"{u['branch']}: rev-list ahead == logged commit count",
              len(u["commits"]), u["ahead"])
        check(f"{u['branch']}: patch_equivalent + patch_novel == ahead",
              u["patch_equivalent"] + u["patch_novel"], u["ahead"])
        check(f"{u['branch']}: defs classified present+absent == new defs",
              len(u["defs_present_in_head"]) + len(u["defs_absent_from_head"]),
              len(u["branch_defs"]))
    main_ref = "origin/main" if "origin/main" in branches else None
    if main_ref:
        a, b = ahead_behind(repo, main_ref)
        check("HEAD vs origin/main: ahead", a, 0)
        check("HEAD vs origin/main: behind", b, 0)

    # theme timeline rows: HEAD window + every unmerged commit
    head_window = []
    fmt = "%H%x1f%aI%x1f%s"
    for line in git(repo, "log", f"--format={fmt}", f"-{args.head_window}").splitlines():
        if not line.strip():
            continue
        sha, date, subject = line.split("\x1f")
        head_window.append({"sha": sha[:7], "date": date[:10], "subject": subject})

    theme_rows = [
        {**c, "scope": "head-15", "theme": classify(c["subject"])} for c in head_window
    ]
    for item in unmerged:
        for c in item["commits"]:
            theme_rows.append(
                {**c, "scope": "upstream-unmerged", "branch": item["branch"],
                 "theme": classify(c["subject"])}
            )
    # merged-history context (beyond the head window, for the figure only)
    for line in git(repo, "log", f"--format={fmt}", f"--skip={args.head_window}").splitlines():
        if not line.strip():
            continue
        sha, date, subject = line.split("\x1f")
        theme_rows.append({"sha": sha[:7], "date": date[:10], "subject": subject,
                           "scope": "history", "theme": classify(subject)})

    figures = make_figures(rows, theme_rows, figs)

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": str(repo),
        "head": head,
        "remote_branch_count": len(branches),
        "remote_branches": rows,
        "merged_into_head": sorted(r["branch"] for r in rows if r["behind"] == 0),
        "diverged": unmerged,
        "totals": {
            "unmerged_commits": sum(u["ahead"] for u in unmerged),
            "patch_equivalent_commits": sum(u["patch_equivalent"] for u in unmerged),
            "patch_novel_commits": sum(u["patch_novel"] for u in unmerged),
            "conflicted_files": sum(len(u["conflicted_files"]) for u in unmerged),
            "stale_base_max": max((u["behind_base"] for u in unmerged), default=0),
        },
        "selfcheck": selfcheck,
        "head_window": head_window,
        "head_window_themes": {
            t: sum(1 for r in theme_rows if r["scope"] == "head-15" and r["theme"] == t)
            for t in sorted({r["theme"] for r in theme_rows if r["scope"] == "head-15"})
        },
        "figures": figures,
    }

    payload = out / "upstream_diff_m1633.json"
    payload.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- markdown table bundle (numbers are injected here, never hand-copied)
    lines: list[str] = []
    lines.append(f"Generated: {summary['generated_utc']} · HEAD `{head['sha'][:7]}` "
                 f"on `{head['branch']}`\n")
    lines.append("### T1. Remote branch divergence vs local HEAD\n")
    lines.append("| remote branch | ahead (unmerged) | behind (stale by) | class |")
    lines.append("|---|---:|---:|---|")
    for r in sorted(rows, key=lambda x: (-x["ahead"], -x["behind"])):
        cls = "merged" if r["behind"] == 0 else ("diverged" if r["ahead"] else "stale-only")
        lines.append(f"| `{r['branch']}` | {r['ahead']} | {r['behind']} | {cls} |")
    lines.append("")
    lines.append("### T2. Unmerged commit inventory (patch-id `git cherry` verdict)\n")
    lines.append("| branch | sha | date | subject | patch in HEAD? |")
    lines.append("|---|---|---|---|---|")
    for u in unmerged:
        for c in u["commits"]:
            flag = "yes (equivalent)" if c["patch_already_in_head"] else "no (novel text)"
            lines.append(
                f"| `{u['branch']}` | `{c['short']}` | {c['date']} | {c['subject']} "
                f"| {flag} |")
    lines.append("")
    lines.append("### T3. Merge-ability against HEAD\n")
    lines.append("| branch | stale by (commits) | merge clean? | conflicted files |")
    lines.append("|---|---:|---|---|")
    for u in unmerged:
        lines.append(
            f"| `{u['branch']}` | {u['behind_base']} "
            f"| {'clean' if u['merge_clean'] else 'CONFLICT'} "
            f"| {len(u['conflicted_files'])} |")
    lines.append("")
    lines.append("### T4. Semantic coverage of branch-introduced symbols in HEAD\n")
    lines.append("| branch | new defs | present in HEAD | absent from HEAD |")
    lines.append("|---|---:|---:|---|")
    for u in unmerged:
        absent = ", ".join(u["defs_absent_from_head"]) or "—"
        lines.append(f"| `{u['branch']}` | {len(u['branch_defs'])} "
                     f"| {len(u['defs_present_in_head'])} | {absent} |")
    lines.append("")
    lines.append("### T4b. Name-level absence explained by API rename\n")
    lines.append("| branch | branch symbol | closest HEAD symbol | ratio |")
    lines.append("|---|---|---|---:|")
    for u in unmerged:
        for name, hit in u["defs_rename_candidates"].items():
            lines.append(f"| `{u['branch']}` | `{name}` | `{hit['match']}` | {hit['ratio']} |")
    lines.append("")
    lines.append("### T5. Branch test functions with no HEAD counterpart\n")
    lines.append("| branch | unmatched test functions |")
    lines.append("|---|---|")
    for u in unmerged:
        parts = []
        for t in u["branch_tests_unmatched_in_head"]:
            hit = u["tests_rename_candidates"].get(t)
            parts.append(f"`{t}`" + (f" (~`{hit['match']}`, {hit['ratio']})" if hit else ""))
        lines.append(f"| `{u['branch']}` | {', '.join(parts) or '—'} |")
    lines.append("")
    lines.append("### T6. HEAD top-%d commit themes\n" % args.head_window)
    lines.append("| theme | commits |")
    lines.append("|---|---:|")
    for t, n in sorted(summary["head_window_themes"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {t} | {n} |")
    lines.append("")
    (out / "upstream_diff_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary["totals"], indent=2))
    print(f"wrote {payload}")
    print(f"wrote {out / 'upstream_diff_tables.md'}")
    for f in figures:
        print(f"wrote {f['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
