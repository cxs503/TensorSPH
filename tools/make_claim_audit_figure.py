"""Figure for docs/claim_audit.md: what the ledger grades, and the one gate that fails.

Panel A: evidence grade of every claim in docs/claims/ledger.json (counts annotated).
Panel B: dam-break gates from benchmarks/dam_break/resolution_ladder.json, plotted as
measured/limit so that every gate shares one axis; 1.0 is the acceptance limit.

Run: python tools/make_claim_audit_figure.py [--out docs/claims/claim_audit.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRADE_COLOURS = {"REAL": "#2c7fb8", "WITHHELD": "#f0a30a", "DRIFT": "#d62728"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/claims/claim_audit.png")
    args = parser.parse_args(argv)

    report = json.loads((ROOT / "docs/claims/report.json").read_text(encoding="utf-8"))
    ladder = json.loads((ROOT / "benchmarks/dam_break/resolution_ladder.json")
                        .read_text(encoding="utf-8"))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 3.6), dpi=110)

    # ---- Panel A: grade per claim -----------------------------------------
    claims = report["claims"]
    grades = [c["grade"] for c in claims]
    order = sorted(range(len(claims)), key=lambda i: (grades[i] != "REAL", claims[i]["id"]))
    ys = list(range(len(order)))
    ax_a.barh(ys, [1] * len(order),
              color=[GRADE_COLOURS[grades[i]] for i in order], height=0.72)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([claims[i]["id"] for i in order], fontsize=7.5)
    ax_a.set_xlim(0, 1.32)
    ax_a.set_xticks([])
    for y, i in zip(ys, order):
        label = grades[i] if claims[i]["kind"] != "prose_only" else "WITHHELD (prose only)"
        ax_a.text(1.02, y, label, va="center", fontsize=7,
                  color=GRADE_COLOURS[grades[i]])
    counts = report["summary"]
    ax_a.set_title(f"A · claim ledger: {counts['real']} REAL / {counts['withheld']} WITHHELD"
                   f" / {counts['drift']} DRIFT (n={counts['claims']})", fontsize=9, loc="left")
    ax_a.invert_yaxis()
    for side in ("top", "right", "bottom"):
        ax_a.spines[side].set_visible(False)

    # ---- Panel B: dam-break gates as measured/limit ------------------------
    gates = ladder["gates"]
    rows = []
    for name, gate in gates.items():
        limit = gate.get("limit")
        if isinstance(limit, (int, float)) and limit:
            rows.append((name, gate["measured"] / limit, gate["result"]))
    rows.sort(key=lambda r: r[1])
    xs = [r[1] for r in rows]
    ys_b = list(range(len(rows)))
    ax_b.barh(ys_b, xs,
              color=["#d62728" if r[2] == "fail" else "#2c7fb8" for r in rows], height=0.6)
    ax_b.axvline(1.0, color="k", ls="--", lw=1.0)
    ax_b.text(1.02, len(rows) - 0.35, "acceptance limit", fontsize=7.5)
    ax_b.set_yticks(ys_b)
    ax_b.set_yticklabels([r[0] for r in rows], fontsize=8)
    for y, (name, ratio, result) in zip(ys_b, rows):
        ax_b.text(ratio + 0.05, y, f"{ratio:.2f}× ({result})", va="center", fontsize=7.5,
                  color="#d62728" if result == "fail" else "#2c7fb8")
    ax_b.set_xlim(0, max(xs) * 1.45)
    ax_b.set_xlabel("measured / limit  (1.0 = gate)", fontsize=8)
    ax_b.set_title("B · B2 dam-break gates: the experimental gate is the only failure",
                   fontsize=9, loc="left")
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)

    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
