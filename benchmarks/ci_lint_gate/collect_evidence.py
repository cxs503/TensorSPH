"""Collect the before/after evidence for the m3227 CI lint-gate fix.

Usage::

    python benchmarks/ci_lint_gate/collect_evidence.py <base-checkout> \
        [<pytest-before.log> <pytest-after.log>]

Inputs:
  * a pristine checkout of the base commit (for the "before" ruff run)
  * the local worktree (for the "after" ruff run)
  * pytest logs for both states (defaults to the paths used for m3227)

Outputs: benchmarks/ci_lint_gate/*.json (committed, so make_doc.py can rebuild
the document without hand-typed numbers).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

GATE = Path("benchmarks/ci_lint_gate")
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/m3227/base2")
PYTEST_BEFORE = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/m3227/pytest_baseline.log")
PYTEST_AFTER = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/m3227/pytest_after.log")


def ruff_json(cwd):
    out = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format=json"],
        cwd=cwd, capture_output=True, text=True,
    ).stdout
    items = json.loads(out)
    rows = []
    for d in items:
        rel = Path(d["filename"]).resolve().relative_to(Path(cwd).resolve())
        rows.append({
            "file": str(rel),
            "row": d["location"]["row"],
            "col": d["location"]["column"],
            "code": d["code"],
            "message": d["message"],
            "fixable": bool(d.get("fix")),
        })
    rows.sort(key=lambda r: (r["file"], r["row"], r["code"]))
    return rows


def pytest_summary(log):
    text = Path(log).read_text()
    failed = sorted(set(re.findall(r"^FAILED (\S+)", text, re.M)))
    errors = sorted(set(re.findall(r"^ERROR (\S+)", text, re.M)))
    m = re.search(r"^(?:=+ )?([\d]+ (?:failed|passed).*?) in ([\d.]+)s", text, re.M)
    tail = m.group(1) if m else ""
    counts = {}
    for k in ("passed", "failed", "xfailed", "xpassed", "error", "errors", "skipped"):
        mm = re.search(rf"(\d+) {k}\b", tail)
        if mm:
            counts[k.rstrip("s")] = int(mm.group(1))
    return {"failed": failed, "errors": errors, "counts": counts, "summary": tail}


def main():
    GATE.mkdir(parents=True, exist_ok=True)
    before = ruff_json(BASE)
    after = ruff_json(Path("."))
    for name, rows in (("ruff_before.json", before), ("ruff_after.json", after)):
        (GATE / name).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
        print(name, len(rows))
    pb, pa = pytest_summary(PYTEST_BEFORE), pytest_summary(PYTEST_AFTER)
    (GATE / "pytest_before.json").write_text(
        json.dumps(pb, indent=1, ensure_ascii=False) + "\n")
    (GATE / "pytest_after.json").write_text(
        json.dumps(pa, indent=1, ensure_ascii=False) + "\n")
    print("pytest before:", pb["summary"])
    print("pytest after :", pa["summary"])


if __name__ == "__main__":
    main()
