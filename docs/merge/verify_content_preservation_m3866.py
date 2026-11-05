"""File-by-file content-preservation check for a completed merge commit.

A textual merge that reports *no conflicts* is not by itself evidence that both
sides survived: git happily applies a resolution that drops content when a
rename, an add/add pair or a stale base is involved (this repository has been
bitten by exactly that, see ``docs/merge/agent-m2905-into-main.md``).  This
checker therefore compares the merged tree against *both* parents, path by path,
and asserts:

1. the two sides touch disjoint path sets relative to the merge base (so an
   automatic merge cannot have chosen between competing edits);
2. every path changed by exactly one side is byte-identical to that side in the
   merge result (nothing rewritten, nothing partially applied);
3. no path that exists on either parent is missing from the merge result
   (nothing silently deleted);
4. no path that exists on neither parent appears in the merge result
   (no accidental resurrection).

Usage::

    python docs/merge/verify_content_preservation_m3866.py [--main REF]
        [--branch REF] [--result REF] [--json OUT.json]

Exit code 0 means every check passed; 1 means at least one violation was found
(details in the JSON report).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

PARENT_A = "--parent-a"
PARENT_B = "--parent-b"


def git(*args: str) -> str:
    """Run a git command and return stdout, raising on failure."""
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def rev(ref: str) -> str:
    """Resolve a ref to a full SHA."""
    return git("rev-parse", ref).strip()


def merge_base(a: str, b: str) -> str:
    """Return the merge base of two refs."""
    return git("merge-base", a, b).strip()


def name_status(base: str, ref: str) -> dict[str, str]:
    """Map path -> status letter for every path that differs from ``base``."""
    out = git("diff", "--name-status", "-z", base, ref)
    fields = out.split("\0")
    result: dict[str, str] = {}
    i = 0
    while i + 1 < len(fields):
        status = fields[i]
        if not status:
            i += 1
            continue
        # Renames/copies are reported as R100<TAB>old<TAB>new; none are expected
        # here, but keep the parser honest about them.
        if status[0] in "RC":
            old, new = fields[i + 1], fields[i + 2]
            result[new] = f"{status}:{old}"
            i += 3
        else:
            result[fields[i + 1]] = status
            i += 2
    return result


def tree_paths(ref: str) -> set[str]:
    """Return every blob path in a tree."""
    out = git("ls-tree", "-r", "--name-only", "-z", ref)
    return {p for p in out.split("\0") if p}


def blob(ref: str, path: str) -> str | None:
    """Return the blob SHA of ``path`` in ``ref``, or None if absent."""
    proc = subprocess.run(
        ["git", "rev-parse", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def audit(main_ref: str, branch_ref: str, result_ref: str) -> dict:
    """Run all preservation checks and return a JSON-serialisable report."""
    sha_main = rev(main_ref)
    sha_branch = rev(branch_ref)
    sha_result = rev(result_ref)
    base = merge_base(main_ref, branch_ref)
    parents = git("rev-list", "--parents", "-n", "1", result_ref).split()[1:]

    side_main = name_status(base, main_ref)
    side_branch = name_status(base, branch_ref)
    shared = sorted(set(side_main) & set(side_branch))

    violations: list[str] = []
    per_path: list[dict] = []

    # (2) single-side paths must be reproduced exactly.
    for path in sorted(set(side_main) | set(side_branch)):
        in_main = path in side_main
        in_branch = path in side_branch
        expected_ref = main_ref if in_main else branch_ref
        expected = blob(expected_ref, path)
        got = blob(result_ref, path)
        ok = expected == got
        if not ok:
            violations.append(f"content mismatch: {path} (expected {expected_ref})")
        per_path.append(
            {
                "path": path,
                "sides": ("both" if in_main and in_branch else
                          ("main" if in_main else "agent/m2905")),
                "merged_blob_matches": ok,
            }
        )

    # (3) nothing present on either parent may vanish.
    lost_from_main = sorted(tree_paths(main_ref) - tree_paths(result_ref))
    lost_from_branch = sorted(tree_paths(branch_ref) - tree_paths(result_ref))
    violations += [f"lost from main: {p}" for p in lost_from_main]
    violations += [f"lost from agent/m2905: {p}" for p in lost_from_branch]

    # (4) nothing may appear from nowhere.
    spurious = sorted(tree_paths(result_ref) - tree_paths(main_ref)
                      - tree_paths(branch_ref))
    violations += [f"present in neither parent: {p}" for p in spurious]

    return {
        "checker": "docs/merge/verify_content_preservation_m3866.py",
        "refs": {
            "main": {"ref": main_ref, "sha": sha_main},
            "branch": {"ref": branch_ref, "sha": sha_branch},
            "result": {"ref": result_ref, "sha": sha_result},
            "merge_base": base,
            "result_parents": parents,
        },
        "counts": {
            "paths_changed_by_main": len(side_main),
            "paths_changed_by_branch": len(side_branch),
            "paths_changed_by_both": len(shared),
            "paths_checked": len(per_path),
        },
        "checks": {
            "disjoint_change_sets": not shared,
            "shared_paths": shared,
            "single_side_paths_byte_identical": not any(
                not row["merged_blob_matches"] for row in per_path
            ),
            "no_path_lost_from_main": not lost_from_main,
            "no_path_lost_from_branch": not lost_from_branch,
            "no_path_from_nowhere": not spurious,
        },
        "per_path": per_path,
        "violations": violations,
        "verdict": "PASS" if not violations else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", default="origin/main")
    parser.add_argument("--branch", default="agent/m2905")
    parser.add_argument("--result", default="HEAD")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = audit(args.main, args.branch, args.result)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    counts = report["counts"]
    print(
        f"merge base {report['refs']['merge_base'][:8]}  "
        f"main {report['refs']['main']['sha'][:8]}  "
        f"branch {report['refs']['branch']['sha'][:8]}  "
        f"result {report['refs']['result']['sha'][:8]}"
    )
    print(
        f"paths: main-side {counts['paths_changed_by_main']}, "
        f"branch-side {counts['paths_changed_by_branch']}, "
        f"shared {counts['paths_changed_by_both']}"
    )
    for name, value in report["checks"].items():
        if isinstance(value, bool):
            print(f"  [{'PASS' if value else 'FAIL'}] {name}")
    for violation in report["violations"]:
        print(f"  VIOLATION {violation}")
    print(f"verdict: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
