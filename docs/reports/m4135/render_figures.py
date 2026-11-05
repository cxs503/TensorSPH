"""Render the two figures of ``docs/reports/m4135-ci-lint-fix.md``.

All numbers are computed, never hand-copied:

* panel A -- ``ruff check`` violation counts per file, before (= commit
  ``3420d0a`` from git) vs after (= working tree);
* panel B -- the CI step chain of run ``3420d0a`` reconstructed from the
  check-run annotations (one failure annotation => exactly one failing step,
  and the ``if: always()`` upload step found no artifacts => every step that
  produces artifacts never ran).

Usage: ``python docs/reports/m4135/render_figures.py`` (writes next to itself).
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_COMMIT = "3420d0a"

# The four files touched by the fix; "before" is read out of git, "after" from
# the working tree, so both columns come from real ruff runs.
FILES = [
    "benchmarks/viscosity_shear_wave/make_doc.py",
    "benchmarks/viscosity_shear_wave/run.py",
    "tests/test_viscosity_shear_wave.py",
]


def ruff_counts(text: str, suffix: str) -> dict[str, int]:
    """Count ``ruff check --stdin-filename`` findings by rule code."""
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", "-q",
         "--output-format=concise", "--stdin-filename", f"probe{suffix}", "-"],
        input=text, capture_output=True, text=True, check=False,
    )
    out = proc.stdout + proc.stderr
    counts: dict[str, int] = {}
    for line in out.splitlines():
        m = re.match(r"^probe\S*:\d+:\d+: (\w+)", line)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def panel_a() -> dict:
    rows = []
    for rel in FILES:
        before = subprocess.run(
            ["git", "show", f"{BASE_COMMIT}:{rel}"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
        after = (ROOT / rel).read_text()
        b = ruff_counts(before, pathlib.Path(rel).suffix)
        a = ruff_counts(after, pathlib.Path(rel).suffix)
        rows.append({"file": rel, "before": sum(b.values()), "after": sum(a.values()),
                     "rules_before": b, "rules_after": a})

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    labels = [pathlib.Path(r["file"]).name for r in rows]
    xs = range(len(rows))
    ax.bar([x - 0.2 for x in xs], [r["before"] for r in rows], width=0.4,
           color="#c0392b", label=f"before ({BASE_COMMIT[:7]})")
    ax.bar([x + 0.2 for x in xs], [r["after"] for r in rows], width=0.4,
           color="#27ae60", label="after (this fix)")
    for x, r in zip(xs, rows):
        ax.text(x - 0.2, r["before"] + 0.06, str(r["before"]), ha="center", fontsize=8)
        ax.text(x + 0.2, r["after"] + 0.06, str(r["after"]), ha="center", fontsize=8)
        if r["rules_before"]:
            ax.text(x, -0.55, ",".join(sorted(r["rules_before"])), ha="center",
                    fontsize=7, color="#555555")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ruff findings")
    ax.set_ylim(0, max(r["before"] for r in rows) + 1.0)
    ax.set_title("ruff check . violations in the files added by the PR head", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig1_ruff_before_after.png")
    plt.close(fig)
    return rows


# CI steps of .github/workflows/tests.yml (the pytest job), in order, with the
# status deduced from the check-run annotations of the failed head: one failure
# annotation and an empty artifact upload.
STEPS = [
    ("checkout + pip install", "pass"),
    ("Lint: ruff check .", "fail"),
    ("Run unit and smoke tests (pytest)", "skipped"),
    ("tiny e2e droplet_impact_2d", "skipped"),
    ("manifest contract assertions", "skipped"),
    ("neighbour search / pair-RHS refs", "skipped"),
    ("free-surface side-by-side", "skipped"),
    ("pair-list reuse neutrality", "skipped"),
    ("hydrostatic gates", "skipped"),
    ("ringing-pool negative control", "skipped"),
    ("dam-break convergence", "skipped"),
    ("fail-fast exit status", "skipped"),
    ("upload smoke artifacts (always)", "ran, empty"),
]


def panel_b() -> None:
    colors = {"pass": "#27ae60", "fail": "#c0392b", "skipped": "#bdc3c7",
              "ran, empty": "#e67e22"}
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=140)
    ys = list(range(len(STEPS)))[::-1]
    for y, (name, status) in zip(ys, STEPS):
        ax.barh(y, 1.0, color=colors[status], height=0.7)
        ax.text(1.03, y, status, va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in STEPS], fontsize=7.5)
    ax.set_xlim(0, 1.55)
    ax.set_xticks([])
    ax.set_title("PR #156 CI (head 3420d0a): the lint step blocked the chain",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_ci_chain.png")
    plt.close(fig)


def main() -> int:
    rows = panel_a()
    panel_b()
    (HERE / "figures.json").write_text(json.dumps(
        {"base_commit": BASE_COMMIT, "ruff": rows,
         "ci_steps": [{"step": s, "status": t} for s, t in STEPS]},
        indent=2, ensure_ascii=False))
    print(json.dumps(rows, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
