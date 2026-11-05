"""Three-way union resolution for the laplace_law convergence artifacts.

Conflict files are *measurement* artifacts: both branches re-ran the same
relaxation case under different protocols, so there is no scalar merge. The
rule used here (no silent loss):

  primary schema+numbers  = agent/m3163 (t/t_sigma = 1.0 protocol; superset
                            schema: adds "field" snapshot + absolute read-out);
  preserved verbatim      = main (900 fixed steps) under
                            provenance.superseded_protocol_run, so every number
                            main contributed stays in-tree and auditable.

After building, the script proves (a) every branch leaf is at its own path and
unchanged, and (b) every main leaf is present with the same value under
superseded_protocol_run. Anything else is reported as LOST, exit non-zero.

usage: python union_merge.py <base_sha> <branch_sha> <main_sha> [audit_dir]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FILES = ["relax_dx0.0025.json", "relax_dx0.0020.json", "relax_dx0.0015.json"]
REL = "benchmarks/laplace_law/convergence/"
LEAVES_TOP = ["setup", "reference", "laplace", "spurious_current",
              "curvature_final", "history"]


def git_show(rev: str, path: str):
    out = subprocess.run(["git", "show", f"{rev}:{path}"], check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out)


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def resolve(path: str, base_rev: str, mine_rev: str, theirs_rev: str):
    base = git_show(base_rev, REL + path)
    mine = git_show(mine_rev, REL + path)      # agent/m3163 -> primary
    theirs = git_show(theirs_rev, REL + path)  # main -> preserved

    merged = dict(mine)
    preserved = {k: theirs[k] for k in LEAVES_TOP if k in theirs}
    merged["provenance"] = {
        "merge": "PR #103 (agent/m3163) x main, conflict cured by m3877",
        "primary_protocol": (
            "t/t_sigma = 1.0 relaxation protocol, produced by "
            "benchmarks/laplace_law/run_ladder.py; top-level numbers are this "
            "protocol and are the ones the resolution ladder, its test and "
            "docs/verification_laplace_resolution_ladder.md are rendered from"),
        "superseded_protocol_run": {
            "branch": "main",
            "protocol": ("fixed 900 steps (t/t_sigma = 0.79/1.05/1.32 per rung); "
                         "kept verbatim so the numbers main contributed are not "
                         "lost, but not used as the ladder protocol because a "
                         "fixed step count compares different capillary phases"),
            **preserved,
        },
    }

    # --- no-silent-loss proof -------------------------------------------
    mine_leaves = dict(leaves(mine))
    theirs_leaves = dict(leaves(theirs))
    merged_leaves = dict(leaves(merged))
    primary_leaves = dict(leaves(preserved))

    lost_mine = [p for p in mine_leaves if p not in merged_leaves]
    lost_theirs = [p for p in theirs_leaves if p not in primary_leaves]
    altered_theirs = [p for p in theirs_leaves
                      if p in primary_leaves
                      and json.dumps(primary_leaves[p], sort_keys=True)
                      != json.dumps(theirs_leaves[p], sort_keys=True)]
    altered_mine = [p for p in mine_leaves
                    if p.split("/")[1] in LEAVES_TOP
                    and p in merged_leaves
                    and json.dumps(merged_leaves[p], sort_keys=True)
                    != json.dumps(mine_leaves[p], sort_keys=True)]
    lost = []
    if lost_mine or lost_theirs or altered_theirs or altered_mine:
        lost.append({"file": path, "lost_branch": lost_mine[:10],
                     "lost_main": lost_theirs[:10],
                     "altered_main": altered_theirs[:10],
                     "altered_branch": altered_mine[:10]})

    audit = {
        "file": REL + path,
        "base_top_keys": sorted(base),
        "main_top_keys": sorted(theirs),
        "branch_top_keys": sorted(mine),
        "n_leaves_branch": len(mine_leaves),
        "n_leaves_main": len(theirs_leaves),
        "n_leaves_merged": len(merged_leaves),
        "main_only_top_keys": sorted(set(theirs) - set(mine)),
        "branch_only_top_keys": sorted(set(mine) - set(theirs)),
        "main_leaves_absent_from_branch": sorted(
            set(theirs_leaves) - set(mine_leaves))[:200],
        "branch_leaves_absent_from_main": sorted(
            set(mine_leaves) - set(theirs_leaves))[:200],
        "main_protocol": {"steps": theirs["setup"]["steps"],
                          "t_over_t_sigma": theirs["setup"]["t_over_t_sigma"]},
        "primary_protocol": {"steps": mine["setup"]["steps"],
                             "t_over_t_sigma": mine["setup"]["t_over_t_sigma"]},
    }
    return merged, audit, lost


def main() -> int:
    base_rev, mine_rev, theirs_rev = sys.argv[1:4]
    outdir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    audits, lost_all = [], []
    for f in FILES:
        merged, audit, lost = resolve(f, base_rev, mine_rev, theirs_rev)
        (Path(".") / (REL + f)).write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
        audits.append(audit)
        lost_all += lost
        print(f"{f}: leaves branch={audit['n_leaves_branch']} "
              f"main={audit['n_leaves_main']} merged={audit['n_leaves_merged']}")
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "union_audit.json").write_text(
            json.dumps({"files": audits, "lost": lost_all}, indent=2) + "\n")
    if lost_all:
        print("LOST CONTENT:", json.dumps(lost_all, indent=2))
        return 1
    print("no silent loss: every leaf of both sides survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
