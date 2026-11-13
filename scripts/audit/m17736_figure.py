#!/usr/bin/env python3
"""Figure for audit m17736: report prose vs run artifact, and the ladder itself.

Reads only artifacts/audit_m17715/*.json (no number is typed here).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "audit_m17715"
consist = json.loads((OUT / "report_consistency.json").read_text())
repro = json.loads((OUT / "repro.json").read_text())


def num(txt):
    return float(str(txt).replace("−", "-").replace("+", ""))


fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))

# (a) ladder: report table vs artifact
for tag, arm, marker in (("ours", "ours[mls", "o"), ("oss", "oss[paper", "s")):
    rows = [r for r in consist["ladder_rows"]
            if r["doc"]["arm"].startswith(arm) and r["artifact"]]
    x = [r["doc"]["R_over_dx"] for r in rows]
    ax[0].plot(x, [num(r["doc"]["err_pct"]) for r in rows], marker + "--",
               label=f"{tag}: report table")
    ax[0].plot(x, [r["artifact"]["err_pct"] for r in rows], marker + "-",
               label=f"{tag}: results.json")
ax[0].axhspan(-15, 15, color="0.88", zorder=0, label="owner +-15% gate")
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xlabel("R/dx"); ax[0].set_ylabel("dp error vs sigma/R  [%]")
ax[0].set_title("(a) ladder ④: prose vs artifact")
ax[0].legend(fontsize=7)

# (b) sensitivity rows, doc vs artifact
rows = [r for r in consist["sensitivity_rows"] if r["artifact"]]
labels = [f"{r['doc']['vary']}={r['doc']['value']}" for r in rows]
docvals, artvals = [], []
for r in rows:
    d = r["doc"]["ours_pct"]
    if d in ("—", ""):
        d = r["doc"]["oss_pct"]
    a = r["artifact"].get("ours_pct")
    if a is None:
        a = r["artifact"].get("oss_pct")
    docvals.append(num(d)); artvals.append(a)
yy = range(len(rows))
ax[1].barh([y + 0.2 for y in yy], docvals, height=0.38, label="report table")
ax[1].barh([y - 0.2 for y in yy], artvals, height=0.38, label="results.json")
ax[1].set_yticks(list(yy)); ax[1].set_yticklabels(labels, fontsize=6)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_xlabel("dp error [%]  (symlog)")
ax[1].set_xscale("symlog", linthresh=1.0)
ax[1].set_title("(b) sensitivity ⑤: prose vs artifact")
ax[1].legend(fontsize=7)

# (c) audit gate scoreboard
checks = [("artifacts regenerate", repro["json_leaf_diff_count"] == 0),
          ("PNG byte-identical",
           repro["png_sha256_delivered"] == repro["png_sha256_repro"]),
          ("repro_cmd runs", repro["harness_rc"] == 0),
          ("integration caller", repro["integration_caller"]["rc"] == 0),
          ("report == artifact",
           consist["ladder_mismatches"] == 0 and consist["sensitivity_mismatches"] == 0),
          ("registered referee", False),
          ("committed to a branch", False)]
ax[2].barh([c[0] for c in checks], [1 if c[1] else 0 for c in checks],
           color=["#2a8" if c[1] else "#c33" for c in checks])
ax[2].set_xlim(0, 1.15); ax[2].set_xlabel("pass=1 / fail=0")
ax[2].set_title("(c) audit scoreboard m17715 (level 2)")
for i, c in enumerate(checks):
    ax[2].text(0.02, i, "PASS" if c[1] else "FAIL", va="center", fontsize=8,
               color="w" if c[1] else "w", fontweight="bold")
fig.tight_layout()
p = OUT / "m17736_audit.png"
fig.savefig(p, dpi=150)
print(p)