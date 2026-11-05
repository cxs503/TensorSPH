"""Audit of the m2926 (PR #86) heal merge: content preservation + test parity.

The numbers written to ``docs/reports/m3850-pr86-merge-audit.json`` and drawn in
the figures are computed here, never typed by hand:

* for every file both sides touched, the lines each side *added* relative to the
  merge base are looked up in the merged worktree -- the count of missing lines
  is the mechanical answer to "did the merge silently drop content?";
* the pytest summaries of the merged tree and of a clean ``origin/main``
  worktree are parsed from their logs and compared test by test.

Usage
-----
    python benchmarks/merge_audit/run.py \
        --merged-log /tmp/merged.log --base-log /tmp/base.log \
        --ours origin/agent/m2926 --theirs origin/main

writes ``docs/reports/m3850-pr86-merge-audit.json`` and
``docs/reports/m3850-pr86-merge-audit.png``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCS = ROOT / "docs" / "reports"
DELTA_REPORT = ROOT / "benchmarks" / "delta_s_normalization" / "validation_report.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def changed_files(base: str, rev: str) -> dict[str, str]:
    out = git("diff", "--name-status", base, rev).strip().split("\n")
    return {line.split("\t")[1]: line.split("\t")[0] for line in out if line}


def added_lines(base: str, rev: str, path: str) -> list[str]:
    out = git("diff", base, rev, "--", path).split("\n")
    return [line[1:].strip() for line in out
            if line.startswith("+") and not line.startswith("+++")]


def preservation(base: str, ours: str, theirs: str) -> list[dict]:
    o, t = changed_files(base, ours), changed_files(base, theirs)
    rows = []
    for path in sorted(set(o) & set(t)):
        merged = (ROOT / path).read_text(encoding="utf-8")
        row = {"path": path}
        for side, rev in (("m2926", ours), ("main", theirs)):
            added = added_lines(base, rev, path)
            row[side] = {
                "added_lines": len(added),
                "missing_in_merged": sum(1 for line in added if line and line not in merged),
            }
        rows.append(row)
    return rows


def parse_pytest(log: str) -> dict:
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    # pytest prints "<n> failed, <n> passed, ..." (bare with xdist, wrapped in
    # "=" banners when sequential); keep the last such line of the log.
    lines = [ln.strip().strip("=").strip() for ln in text.splitlines()]
    summary = [ln for ln in lines if re.search(r"\d+ (passed|failed|error)", ln)]
    counts: dict[str, int] = {}
    for key in ("passed", "failed", "error", "errors", "xfailed", "xpassed", "skipped"):
        found = re.findall(rf"(\d+) {key}\b", summary[-1] if summary else "")
        if found:
            counts[key] = int(found[0])
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)))
    return {"summary_line": summary[-1] if summary else "", "counts": counts,
            "failed": failed}


def figure(payload: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    report = json.loads(DELTA_REPORT.read_text(encoding="utf-8"))
    combos = [(r["delta_s_scheme"], r["force_density"]) for r in report["instant"]]
    seen: list[tuple[str, str]] = []
    for combo in combos:
        if combo not in seen:
            seen.append(combo)
    styles = {"two_sided": "--", "kernel_sum": "-"}
    markers = {"local": "o", "reference": "s"}
    for scheme, dens in seen:
        rows = [r for r in report["instant"]
                if r["delta_s_scheme"] == scheme and r["force_density"] == dens]
        rows.sort(key=lambda r: r["dx_m"])
        dx = [r["dx_m"] * 1e3 for r in rows]
        ident = [r["delta_s"]["vector_identity"] for r in rows]
        cut = [r["cut_force"]["ratio_measured_over_theory"] for r in rows]
        label = f"{scheme}/{dens}"
        ax.plot(dx, ident, styles[scheme], marker=markers[dens], ms=4, lw=1.2, label=label)
        ax2.plot(dx, cut, styles[scheme], marker=markers[dens], ms=4, lw=1.2, label=label)
    for axis, title, ref in (
        (ax, r"$\sum\delta_s n_x\,dV/2R$ (target 1)", 1.0),
        (ax2, r"$F_{cut}/2\sigma$ (target 1)", 1.0),
    ):
        axis.axhline(ref, color="k", lw=1.0)
        axis.axhspan(0.97 * ref, 1.03 * ref, color="tab:green", alpha=0.12,
                     label="±3 %")
        axis.axhspan(0.85 * ref, 1.15 * ref, color="tab:orange", alpha=0.10,
                     label="±15 %")
        axis.set_xlabel("dx [mm]")
        axis.set_title(title, fontsize=9)
        axis.invert_xaxis()
        axis.grid(alpha=0.3)
    ax.set_ylabel("measured / exact")
    ax.legend(fontsize=6, loc="best")

    fig.suptitle("m2926 delta_s / cut-force identity (left) and CSF force density "
                 "normalisation (right)\nmerged tree, benchmarked by "
                 "benchmarks/delta_s_normalization", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=95)

    fig2, bx = plt.subplots(figsize=(7.2, 3.0))
    paths = [row["path"].split("/")[-1] for row in payload["preservation"]]
    left = [row["m2926"]["added_lines"] for row in payload["preservation"]]
    right = [row["main"]["added_lines"] for row in payload["preservation"]]
    idx = range(len(paths))
    bx.bar([i - 0.2 for i in idx], left, width=0.4, label="lines added by m2926")
    bx.bar([i + 0.2 for i in idx], right, width=0.4, label="lines added by main")
    lost = sum(row[side]["missing_in_merged"]
               for row in payload["preservation"] for side in ("m2926", "main"))
    bx.set_xticks(list(idx))
    bx.set_xticklabels(paths, fontsize=7)
    bx.set_ylabel("added lines relative to merge base")
    bx.set_title(f"co-modified files: {len(paths)}, added lines lost in the merge: {lost}",
                 fontsize=8)
    bx.legend(fontsize=7)
    bx.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out.with_name(out.stem + "-preservation.png"), dpi=110)
    print("figures:", out, out.with_name(out.stem + "-preservation.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="origin/agent/m2926")
    ap.add_argument("--theirs", default="origin/main")
    ap.add_argument("--merged-log", default="/tmp/merged.log")
    ap.add_argument("--base-log", default="/tmp/base.log")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    base = git("merge-base", args.theirs, args.ours).strip()
    payload = {
        "mission": 3850,
        "task": "heal PR #86 (agent/m2926) against main: semantic-union merge",
        "shas": {"merge_base": base, "main": git("rev-parse", args.theirs).strip(),
                 "pr_head_old": git("rev-parse", args.ours).strip()},
        "conflicts": {
            "files": ["src/tensorsph/csf_consistent.py",
                      "src/tensorsph/droplet_impact.py"],
            "hunks": 3,
            "other_co_modified_auto_merged": "src/tensorsph/surface_tension.py",
        },
        "preservation": preservation(base, args.ours, args.theirs),
        "tests": {"merged": parse_pytest(args.merged_log),
                  "main_baseline": parse_pytest(args.base_log)},
    }
    payload["preservation_total_missing"] = sum(
        row[side]["missing_in_merged"]
        for row in payload["preservation"] for side in ("m2926", "main"))
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "m3850-pr86-merge-audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)
    print("preservation_total_missing =", payload["preservation_total_missing"])
    print("merged tests:", payload["tests"]["merged"]["summary_line"])
    print("base   tests:", payload["tests"]["main_baseline"]["summary_line"])
    if not args.no_figures:
        figure(payload, DOCS / "m3850-pr86-merge-audit.png")


if __name__ == "__main__":
    main()
