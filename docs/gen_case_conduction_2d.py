#!/usr/bin/env python3
"""Anti-drift check for docs/case-conduction-2d.md against the committed artefact.

The Markdown table and the headline numbers are *rendered* from
``benchmarks/case_conduction_2d/result.json``; this script re-derives them and
fails (exit 1) if the document disagrees, so a hand-edited number is detectable.
Same spirit as ``benchmarks/method_selection/build_registry.py --check``.

Usage::

    PYTHONPATH=src python3 docs/gen_case_conduction_2d.py [--check]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = ROOT / "benchmarks" / "case_conduction_2d" / "result.json"
DOC = ROOT / "docs" / "case-conduction-2d.md"

ARMS = (("none", "euler"), ("lattice_1d", "euler"), ("k0_2d", "rk4"),
        ("k0_2d", "euler"), ("mean_2d", "rk4"), ("mode_2d", "rk4"))


def render_table(doc: dict) -> str:
    ladder = doc["ladder"]
    ns = sorted({row["n"] for row in ladder})
    lines = ["| arm | integrator | " + " | ".join(f"n={n}" for n in ns) + " |",
             "|---|---|" + "---|" * len(ns)]
    for cal, integ in ARMS:
        cells = []
        for n in ns:
            hit = [r for r in ladder
                   if r["calibration"] == cal and r["integrator"] == integ and r["n"] == n]
            cells.append(f"{hit[0]['max_normalised_error'] * 100:.3g} %" if hit else "—")
        lines.append(f"| `{cal}` | {integ} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(doc: dict) -> list[str]:
    main = doc["main_run"]
    return [
        f"**{main['max_normalised_error'] * 100:.4g} %**",
        f"{main['max_error_over_initial_amplitude'] * 100:.4g} %",
        f"{doc['convergence']['order_log_log']:.2f}",
        f"{main['factor']:.6g}",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    doc = json.loads(ARTEFACT.read_text())
    text = DOC.read_text()
    table = render_table(doc)
    problems = []
    if table not in text:
        problems.append("Markdown ladder table drifted from the artefact")
    for item in headline(doc):
        if item not in text:
            problems.append(f"headline {item!r} missing from the document")
    if args.check and problems:
        for p in problems:
            print("DRIFT:", p)
        return 1
    print("docs/case-conduction-2d.md matches the artefact" if not problems
          else "\n".join(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
