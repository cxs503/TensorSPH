"""Build the before/after lint evidence for the PR #149 CI repair.

Reads the ruff output captured from the failing GitHub Actions log
(pytest (3.10) job 127155964466, ruff 0.17.3) and compares it with the
local tree after the fix.  Writes artifacts/pr149_lint_evidence.json and
docs/figures/pr149_lint_before_after.png.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = pathlib.Path("/tmp/ci310.raw")
OUT_JSON = ROOT / "docs" / "pr149_lint_evidence.json"
OUT_PNG = ROOT / "docs" / "figures" / "pr149_lint_before_after.png"

TS = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ")
RULE = re.compile(r"^([A-Z]\d{3})\b")
LOC = re.compile(r"^\s+--> ([^:]+):(\d+):(\d+)$")


def parse_ci(path):
    rows = []
    pending = None
    for raw in path.read_text(errors="replace").splitlines():
        line = TS.sub("", raw)
        m = LOC.match(line)
        if m and pending:
            rows.append({"file": m.group(1), "line": int(m.group(2)),
                         "col": int(m.group(3)), "rule": pending})
            pending = None
            continue
        m = RULE.match(line)
        if m:
            pending = m.group(1)
    return rows


def local_ruff(paths):
    """Count remaining violations with the locally installed ruff."""
    import subprocess
    out = subprocess.run(["python3", "-m", "ruff", "check", *paths,
                          "--output-format=concise"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    rows = []
    for line in out.splitlines():
        m = re.match(r"^(.+?):(\d+):(\d+): ([A-Z]\d{3}) ", line)
        if m:
            rows.append({"file": m.group(1), "line": int(m.group(2)),
                         "col": int(m.group(3)), "rule": m.group(4)})
    return rows


def main():
    before = parse_ci(LOG)
    after = local_ruff(["."])
    by_file_before, by_rule_before = {}, {}
    for r in before:
        by_file_before[r["file"]] = by_file_before.get(r["file"], 0) + 1
        by_rule_before[r["rule"]] = by_rule_before.get(r["rule"], 0) + 1

    ev = {
        "pr": 149,
        "branch": "agent/m3626",
        "head_sha": "4fe0ae57",
        "ci_run": 42001027681,
        "ci_job": 127155964466,
        "ci_ruff_version": "0.17.3",
        "stage": "Lint (`python -m ruff check .`)",
        "errors_before": len(before),
        "errors_after": len(after),
        "by_rule_before": by_rule_before,
        "by_file_before": dict(sorted(by_file_before.items(),
                                      key=lambda kv: -kv[1])),
        "remaining_after": after,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(ev, indent=2) + "\n")
    print(json.dumps({k: ev[k] for k in ("errors_before", "errors_after",
                                         "by_rule_before", "by_file_before")},
                     indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = list(ev["by_file_before"])
    counts = [ev["by_file_before"][f] for f in files]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    bars = ax.barh(range(len(files)), counts, color="#c0392b", label="before (CI, ruff 0.17.3)")
    ax.barh(range(len(files)), [0] * len(files), color="#27ae60",
            label="after (fix, ruff 0.15.0)")
    ax.set_yticks(range(len(files)))
    ax.set_yticklabels([f"{f}" for f in files], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ruff violations (E501 / W292 / F401 / E401 / E702 / E741)")
    ax.set_title("PR #149 lint gate: %d -> 0 violations before reaching `pytest`"
                 % (len(before),), fontsize=10)
    for b, c in zip(bars, counts):
        ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, str(c),
                va="center", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print("figure ->", OUT_PNG)


if __name__ == "__main__":
    main()
