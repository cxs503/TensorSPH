#!/usr/bin/env python3
"""Reproduce the PR #118 merge evidence figure and its numbers.

Panel (a) -- the content the PR is about: |Δp| tail-averaged error against σ/R
for the same grid with the free-surface pressure closure off (``runs``) and on
(``closure_runs``), read from ``benchmarks/laplace_pressure_floor/report.json``.
Panel (b) -- the merge evidence: pytest outcome counts of the merge result
versus the ``main`` baseline, parsed from the two pytest logs so that no number
is transcribed by hand.

Usage
-----
    python tools/make_pr118_merge_figure.py --base-log /tmp/base.log \
        --merge-log /tmp/merge.log [--out docs/figs/pr118_merge_evidence.png]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "laplace_pressure_floor" / "report.json"
SUMMARY_RE = re.compile(r"(?:(\d+) failed, )?(\d+) passed, (?:(\d+) xfailed|(\d+) error)")


def parse_log(path: Path) -> dict:
    text = path.read_text()
    m = SUMMARY_RE.search(text)
    if not m:
        raise SystemExit(f"no pytest summary line in {path}")
    return {"failed": int(m.group(1) or 0), "passed": int(m.group(2)),
            "xfailed": int(m.group(3) or 0),
            "failed_tests": re.findall(r"^FAILED (\S+)", text, re.M)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-log", required=True)
    ap.add_argument("--merge-log", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs" / "figs" / "pr118_merge_evidence.png"))
    ap.add_argument("--json", default=str(ROOT / "docs" / "merge_pr118_test_evidence.json"))
    args = ap.parse_args()

    rep = json.loads(REPORT.read_text())
    base = parse_log(Path(args.base_log))
    merged = parse_log(Path(args.merge_log))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for key, marker, colour, label in (("runs", "o", "tab:red", "closure off"),
                                       ("closure_runs", "s", "tab:blue", "closure on")):
        runs = rep.get(key) or []
        if not runs:
            continue
        # x = sound speed, one series per resolution
        for dx in sorted({r["dx_m"] for r in runs}):
            sel = sorted((r for r in runs if r["dx_m"] == dx),
                         key=lambda r: r["sound_speed"])
            ax.plot([r["sound_speed"] for r in sel],
                    [r["absolute_time_average_tail"]["rel_error_pct"] for r in sel],
                    marker=marker, color=colour, ls="-" if dx == 0.002 else "--",
                    label=f"{label}, dx={dx * 1e3:g} mm")
    ax.axhspan(-15, 15, color="0.85", zorder=0, label="±15 % acceptance band")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"sound speed $c_0$ [m/s]")
    ax.set_ylabel(r"$|\Delta p|$ error vs $\sigma/R$ [%] (tail average)")
    ax.set_title("(a) what the PR fixes: Laplace pressure closure")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["main\n(baseline)", "merge\nagent/m3293"]
    passed = [base["passed"], merged["passed"]]
    failed = [base["failed"], merged["failed"]]
    xfailed = [base["xfailed"], merged["xfailed"]]
    ax.bar(labels, passed, color="tab:green", label="passed")
    ax.bar(labels, failed, bottom=passed, color="tab:red", label="failed")
    ax.bar(labels, xfailed, bottom=[p + f for p, f in zip(passed, failed)],
           color="tab:orange", label="xfailed")
    for i, (p, f, x) in enumerate(zip(passed, failed, xfailed)):
        ax.text(i, p / 2, str(p), ha="center", color="w", fontweight="bold")
        ax.text(i, p + f + x + 4, f"failed={f}, xfail={x}", ha="center", fontsize=8)
    ax.set_ylabel("tests")
    ax.set_ylim(0, max(p + f + x for p, f, x in zip(passed, failed, xfailed)) * 1.2)
    ax.set_title("(b) full test suite: merge vs main baseline")
    ax.legend(fontsize=8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("TensorSPH PR #118 (agent/m3293 → main): union merge evidence", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)

    evidence = {
        "baseline_main": base,
        "merge_agent_m3293": merged,
        "same_failure_set": base["failed_tests"] == merged["failed_tests"],
        "closure_acceptance": rep.get("closure_acceptance"),
        "worst_closure_off_pct": max(abs(r["absolute_time_average_tail"]["rel_error_pct"])
                                     for r in rep["runs"]),
        "worst_closure_on_pct": max(abs(r["absolute_time_average_tail"]["rel_error_pct"])
                                    for r in rep["closure_runs"]),
        "figure": out.name,
    }
    Path(args.json).write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
