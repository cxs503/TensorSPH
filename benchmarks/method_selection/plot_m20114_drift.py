"""Drift evidence for the m20114 artefacts: how stale were they, and what moved.

Reads the git history of ``docs/reports/method_selection_registry.json`` and
recomputes, for every revision, the two things the m20114 deliverables print:

* ``records`` -- the count that appears in the tables/summary, and
* the per-method ``(best, worst)`` pair the coverage figure draws as a bar + red
  marker.

The m20114 artefacts were committed at the revision where the registry held 12
records; the registry has grown since.  This script quantifies the gap (it is the
evidence behind ``docs/dev_report_xval_method_selection_m20114.md`` §drift
closure) and draws it.

Usage::

    PYTHONPATH=src python benchmarks/method_selection/plot_m20114_drift.py

Outputs
-------
``docs/figures/method_selection_m20114/figure_drift_closure.png``
``docs/reports/method_selection_m20114_drift.json``
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.method_selection import OWNER_TOLERANCE_PCT  # noqa: E402

REGISTRY = "docs/reports/method_selection_registry.json"
#: the commit that merged the m20114 deliverables (registry then held 12 records)
ANCHOR_COMMIT = "9f79e31f"
FIG_PATH = ROOT / "docs" / "figures" / "method_selection_m20114" / "figure_drift_closure.png"
JSON_PATH = ROOT / "docs" / "reports" / "method_selection_m20114_drift.json"


def _sh(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd}: {out.stderr.strip()}")
    return out.stdout


def _bars(doc: dict) -> dict[str, tuple[float, float]]:
    methods = {r["method"] for r in doc.get("records", [])}
    return {m: (min(r["measured_pct"] for r in doc["records"] if r["method"] == m),
                max(r["measured_pct"] for r in doc["records"] if r["method"] == m))
            for m in methods}


def history() -> list[dict]:
    revs = _sh(f"git log --reverse --format=%H -- {REGISTRY}").split()
    rows = []
    for rev in revs:
        doc = json.loads(_sh(f"git show {rev}:{REGISTRY}"))
        rows.append({
            "commit": rev[:8],
            "date": _sh(f"git log -1 --format=%cs {rev}"),
            "records": len(doc.get("records", [])),
            "methods_with_evidence": len(doc.get("methods_with_evidence", [])),
            "bars": _bars(doc),
        })
    return rows


def main() -> int:
    rows = history()
    anchor = next(r for r in rows if r["commit"] == ANCHOR_COMMIT)
    current = rows[-1]
    old_bars, new_bars = anchor["bars"], current["bars"]

    def _fmt(pair, method):
        return {"best": pair[method][0], "worst": pair[method][1]} if method in pair else None

    drift = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "artefact_commit": ANCHOR_COMMIT,
        "artefact_commit_date": anchor["date"],
        "records_at_artefact_commit": anchor["records"],
        "records_now": current["records"],
        "records_banked_after": current["records"] - anchor["records"],
        "hidden_fraction": 1.0 - anchor["records"] / current["records"],
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
        "methods_at_artefact_commit": len(old_bars),
        "methods_now": len(new_bars),
        "methods_new_since": sorted(set(new_bars) - set(old_bars)),
        "per_method": {m: {"at_artefact_commit": _fmt(old_bars, m),
                           "now": _fmt(new_bars, m)} for m in sorted(new_bars)},
        "records_over_owner_tolerance": sorted(
            ({"method": m, "at_artefact_commit": _fmt(old_bars, m) is not None
              and old_bars[m][1] > OWNER_TOLERANCE_PCT,
              "now": new_bars[m][1] > OWNER_TOLERANCE_PCT} for m in sorted(new_bars)),
            key=lambda d: d["method"]),
        "history": [{"commit": r["commit"], "date": r["date"], "records": r["records"]}
                    for r in rows],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax = axes[0]
    ax.plot([r["records"] for r in rows], "o-", color="#2e8b57", lw=1.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["date"][5:] for r in rows], rotation=45, fontsize=7.5)
    ia = rows.index(anchor)
    ax.axhline(anchor["records"], color="crimson", ls="--", lw=1.0,
               label=f"m20114 artefacts committed (n={anchor['records']})")
    ax.annotate(f"+{current['records'] - anchor['records']} records banked later",
                xy=(len(rows) - 1, current["records"]),
                xytext=(max(ia, 1), current["records"] - 4), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_ylabel("evidence records in the registry")
    ax.set_title("registry growth after the m20114 commit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    methods = sorted(new_bars, key=lambda m: new_bars[m][0])
    y = range(len(methods))
    for k, m in enumerate(methods):
        if m in old_bars:
            ax.plot([old_bars[m][1]], [k], "x", color="0.55", ms=7,
                    label="worst record at m20114 commit" if k == 0 else None)
        ax.plot([new_bars[m][1]], [k], "rx", ms=7,
                label="worst record now" if k == 0 else None)
        ax.plot([new_bars[m][0]], [k], "o", color="#2e8b57", ms=5,
                label="best record now" if k == 0 else None)
    ax.axvline(OWNER_TOLERANCE_PCT, color="crimson", lw=1.2, ls="--",
               label=f"owner standard {OWNER_TOLERANCE_PCT:g}%")
    ax.set_yticks(list(y))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("measured error vs its reference (%)")
    ax.set_title("what the stale coverage figure did not show")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    JSON_PATH.write_text(json.dumps(drift, indent=2))
    print(f"wrote {FIG_PATH}\nwrote {JSON_PATH}")
    print(f"records {anchor['records']} -> {current['records']} "
          f"({drift['hidden_fraction']*100:.1f}% of the evidence was invisible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())