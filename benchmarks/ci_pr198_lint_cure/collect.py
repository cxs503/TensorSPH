"""Programmatic evidence collector for the TensorSPH PR #198 CI cure.

PR #198 (branch ``agent/m3584``) was red on all six matrix legs
(``pytest (3.10)``, ``(3.11)``, ``(3.12)`` on two runs each) *before* a single
test ran: the ``Lint`` step -- ``python -m ruff check .`` -- failed, so
``Run unit and smoke tests`` and every later step were skipped.  The violations
come from the files that ``main`` gained after the PR branched (the benchmark
generators and the ``initial_density`` module), not from the PR itself, but a
PR is judged on the merged tree, so the branch inherits them.

This script re-derives every number quoted in ``docs/ci_cure_pr198.md`` from
artifacts instead of transcription:

* ``before`` -- the pinned ``ruff check . --output-format=json`` snapshot of the
  failing tree (``ruff_before.json``, produced on ``origin/main``),
* ``after``  -- the same command run on the current tree,
* ``tests``  -- the pytest summary of a full-suite log (``--pytest-log``),
* ``pins``   -- the mode-2 frequency measurements pinned in
  ``tests/test_oscillating_droplet.py`` (read from the module, not re-typed).

Outputs (written next to this file): ``summary.json``, ``summary.md`` and
``ci_cure_summary.png``.  Usage::

    python benchmarks/ci_pr198_lint_cure/collect.py --pytest-log /tmp/full.log
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GATE_PCT = 15.0          # platform acceptance bar for a frequency claim
STRICT_PCT = 3.0         # strict "publication grade" bar


def ruff_json(path: Path) -> list[dict]:
    """``ruff check`` violations for a tree, as a list of normalised records."""
    out = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format=json"],
        cwd=path, capture_output=True, text=True, check=False)
    payload = json.loads(out.stdout or "[]")
    records = []
    for item in payload:
        filename = Path(item["filename"]).resolve()
        try:
            rel = filename.relative_to(Path(path).resolve())
        except ValueError:                      # pragma: no cover - foreign path
            rel = Path(filename.name)
        records.append({"file": rel.as_posix(), "code": item["code"],
                        "row": item["location"]["row"]})
    return records


def pytest_summary(log: Path) -> dict:
    """Test counts from the final ``pytest`` summary line of a log file."""
    text = log.read_text(errors="replace")
    passed = failed = xfailed = errors = 0
    for pattern, key in ((r"(\d+) passed", "passed"), (r"(\d+) failed", "failed"),
                         (r"(\d+) xfailed", "xfailed"), (r"(\d+) error", "errors")):
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            passed, failed, xfailed, errors = (
                value if key == "passed" else passed,
                value if key == "failed" else failed,
                value if key == "xfailed" else xfailed,
                value if key == "errors" else errors)
    return {"passed": passed, "failed": failed, "xfailed": xfailed,
            "errors": errors,
            "green": failed == 0 and errors == 0 and passed > 0,
            "tail": text.strip().splitlines()[-1] if text.strip() else ""}


def mode2_pins() -> dict:
    """The architecture-resolved mode-2 pins, read from the test module."""
    spec = importlib.util.spec_from_file_location(
        "test_oscillating_droplet", REPO / "tests" / "test_oscillating_droplet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)             # importing defines pins only
    return dict(module.RECORDED_MODE2_ERR_PCT)


def render(records: list[dict]) -> dict:
    per_rule = Counter(r["code"] for r in records)
    per_file = Counter(r["file"] for r in records)
    generators = sum(n for f, n in per_file.items() if f.endswith("make_doc.py"))
    return {
        "total": len(records),
        "per_rule": dict(sorted(per_rule.items())),
        "per_file": dict(sorted(per_file.items(), key=lambda kv: -kv[1])),
        "doc_generator_violations": generators,
    }


def figure(summary: dict, out: Path) -> None:
    """Two panels: the lint gate, and the architecture-fragile mode-2 pin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_lint, ax_phys) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    before = summary["lint"]["before"]["per_rule"]
    after = summary["lint"]["after"]["per_rule"]
    codes = sorted(before, key=lambda c: -before[c])
    xs = range(len(codes))
    ax_lint.bar([x - 0.2 for x in xs], [before[c] for c in codes], width=0.4,
                label="before cure (merged tree)", color="#c0392b")
    ax_lint.bar([x + 0.2 for x in xs], [after.get(c, 0) for c in codes], width=0.4,
                label="after cure", color="#27ae60")
    for x, code in zip(xs, codes):
        ax_lint.text(x - 0.2, before[code], str(before[code]),
                     ha="center", va="bottom", fontsize=8)
    ax_lint.set_xticks(list(xs))
    ax_lint.set_xticklabels(codes)
    ax_lint.set_ylabel("violations")
    ax_lint.set_title(f"`ruff check .` gate: {summary['lint']['before']['total']} "
                      f"-> {summary['lint']['after']['total']}")
    ax_lint.legend(frameon=False, fontsize=8)
    ax_lint.grid(axis="y", alpha=0.3)

    pins = summary["mode2_pins_pct"]
    names = list(pins)
    values = [pins[n] for n in names]
    ax_phys.axhspan(-GATE_PCT, GATE_PCT, color="#27ae60", alpha=0.12,
                    label=f"+-{GATE_PCT:.0f} % acceptance band")
    ax_phys.axhline(STRICT_PCT, color="#2980b9", ls=":", lw=1,
                    label=f"+{STRICT_PCT:.0f} % strict band")
    ax_phys.axhline(-STRICT_PCT, color="#2980b9", ls=":", lw=1)
    ax_phys.axhline(0.0, color="0.4", lw=0.8)
    ax_phys.plot(range(len(names)), values, "o", ms=9, color="#8e44ad",
                 label="mode-2 frequency error")
    for i, (name, value) in enumerate(zip(names, values)):
        ax_phys.annotate(f"{value:+.2f} %", (i, value), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8)
    ax_phys.axhline(summary["reference_case_error_pct"], color="#e67e22", ls="--",
                    lw=1.2, label="reference case R = 2 cm "
                    f"({summary['reference_case_error_pct']:+.1f} %)")
    ax_phys.set_xticks(range(len(names)))
    ax_phys.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax_phys.set_ylabel("mode-2 frequency error [%]")
    ax_phys.set_title("why the pin has to be multi-valued")
    ax_phys.legend(frameon=False, fontsize=7, loc="lower right")
    ax_phys.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-log", type=Path, default=None,
                        help="log of a full-suite run to summarise")
    parser.add_argument("--reference-case", type=float, default=None,
                        help="documented reference-case frequency error [%%]")
    args = parser.parse_args()

    before = json.loads((HERE / "ruff_before.json").read_text())
    summary = {
        "lint": {"before": render(before), "after": render(ruff_json(REPO))},
        "mode2_pins_pct": mode2_pins(),
    }
    summary["mode2_pins_spread_pct"] = (
        max(summary["mode2_pins_pct"].values()) - min(summary["mode2_pins_pct"].values()))
    summary["reference_case_error_pct"] = args.reference_case
    if args.pytest_log is not None:
        summary["tests"] = pytest_summary(args.pytest_log)
    figure(summary, HERE / "ci_cure_summary.png")

    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| rule | before | after |", "|---|---|---|"]
    for code in summary["lint"]["before"]["per_rule"]:
        lines.append(f"| `{code}` | {summary['lint']['before']['per_rule'][code]} "
                     f"| {summary['lint']['after'].get(code, 0)} |")
    lines.append(f"| **total** | **{summary['lint']['before']['total']}** "
                 f"| **{summary['lint']['after']['total']}** |")
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
