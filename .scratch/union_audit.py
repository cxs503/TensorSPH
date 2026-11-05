"""Merge verification for PR agent/m2926 -> main (mission m3790).

Guards against the historical failure mode *"mechanical overlay silently drops
one side's additions"*.  The tree is classified the only way a 3-way merge can
be wrong:

  A. paths changed only by main   -> merged file must be byte-identical to main
  B. paths changed only by the PR -> merged file must be byte-identical to the PR
  C. paths changed by BOTH sides  -> merged file must contain every line and every
     ``def``/``class``/field name each side added relative to the merge base
     (multiset containment, so a duplicated union is caught too)

Exit code 0 = no silent loss, 1 = loss detected.

Usage: python3 .scratch/union_audit.py <merge-base> <main> <pr>
"""
from __future__ import annotations

import ast
import collections
import subprocess
import sys

BASE, MAIN, PR = sys.argv[1:4]


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True,
                          check=True).stdout


def show(rev: str, path: str) -> bytes | None:
    p = subprocess.run(("git", "show", f"{rev}:{path}"), capture_output=True)
    return p.stdout if p.returncode == 0 else None


def text(b: bytes | None) -> str | None:
    if b is None:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def norm(lines) -> collections.Counter:
    return collections.Counter(l.strip() for l in lines if l.strip())


def names(src: str) -> set[str]:
    """Function/class names + assigned/annotated names (dataclass fields)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, ast.Assign):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return out


def main() -> int:
    main_files = set(git("diff", "--name-only", BASE, MAIN).split())
    pr_files = set(git("diff", "--name-only", BASE, PR).split())
    both = sorted(main_files & pr_files)
    only_main = sorted(main_files - pr_files)
    only_pr = sorted(pr_files - main_files)

    failures: list[str] = []
    print(f"both-sides: {len(both)}  main-only: {len(only_main)}  pr-only: {len(only_pr)}")

    for path in only_pr:
        pr_b, got_b = show(PR, path), show("HEAD", path)
        try:
            got_b = open(path, "rb").read()
        except (FileNotFoundError, IsADirectoryError):
            pass
        if got_b != pr_b:
            failures.append(f"{path}: pr-only file differs from the PR version "
                            f"({len(got_b or b'')} vs {len(pr_b or b'')} bytes)")

    for path in only_main:
        main_b, got_b = show(MAIN, path), show("HEAD", path)
        try:
            got_b = open(path, "rb").read()
        except (FileNotFoundError, IsADirectoryError):
            pass
        if got_b != main_b:
            failures.append(f"{path}: main-only file differs from the main version")

    print(f"\n{'both-sides path':40s} {'+main':>6s} {'+pr':>6s} {'lost':>5s}")
    for path in both:
        base_t, main_t, pr_t = (text(show(BASE, path)), text(show(MAIN, path)),
                                text(show(PR, path)))
        try:
            work_t = open(path, encoding="utf-8").read()
        except (FileNotFoundError, UnicodeDecodeError):
            failures.append(f"{path}: missing/binary in merged tree")
            continue
        b = norm(base_t.split("\n"))
        have = norm(work_t.split("\n"))
        lost = 0
        for label, side_t in (("main", main_t), ("pr", pr_t)):
            if side_t is None:
                continue
            added = norm(side_t.split("\n")) - b
            lack = added - have
            for line, n in lack.items():
                failures.append(f"{path}: {label} line lost ({n}x): {line[:100]}")
                lost += n
            for sym in names(side_t) - names(work_t):
                failures.append(f"{path}: {label} symbol lost: {sym}")
                lost += 1
        add_m = len(norm((main_t or "").split("\n")) - b)
        add_p = len(norm((pr_t or "").split("\n")) - b)
        print(f"{path:40s} {add_m:6d} {add_p:6d} {lost:5d}")

    if failures:
        print(f"\nFAIL ({len(failures)} findings):")
        for f in failures[:40]:
            print("  " + f)
        return 1
    print("\nOK: no added line, symbol or side-only file was dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())