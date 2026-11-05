#!/usr/bin/env python3
"""union_audit -- machine check that a three-way merge kept *both* sides.

Motivation (TensorSPH PR #118 / mission m3982): mechanical overlays have a
history in this repository of resolving a conflict by taking one side and
silently dropping hunks of the other.  A clean ``git merge`` exit code does not
prove anything about content: it only proves that the *recorded* resolution of
an earlier merge touched no further lines.  This tool therefore audits the
merge result against both parents, line by line and symbol by symbol.

For every path that both parents touched (relative to a common base) it reports

  * ``deleted``   -- paths present in a parent but missing from the merge,
  * ``missing_ours`` / ``missing_theirs`` -- normalised source lines that a
    parent contributes but the merge does not contain.  A resolution is a
    *union* when this list is empty; when it is not empty every entry has to be
    adjudicated by hand (a line that was deliberately rewritten shows up here),
  * ``symbols_ours`` / ``symbols_theirs`` -- for Python files, top-level
    ``def``/``class`` names contributed by each parent that the merge lacks.
    For a union resolution this list must be empty -- a rewritten line is a
    judgement call, a vanished function is content loss.

Usage
-----
    union_audit.py --repo . --base <sha> --ours <sha> --theirs <sha> \
                   --merged <sha> [--json out.json] [--max-lines 40]

Exit status is 0 when no path was deleted and no symbol disappeared, 1
otherwise.  Missing *lines* do not fail the run by themselves -- they are
printed for adjudication, because the conflict being resolved usually is a
deliberate rewrite of a few lines.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".cfg", ".txt", ".yml", ".yaml",
                 ".sh", ".in", ".rst"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def tree(repo: Path, sha: str) -> dict[str, str]:
    """path -> blob sha for every blob in ``sha``."""
    out = git(repo, "ls-tree", "-r", "-z", sha)
    entries: dict[str, str] = {}
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, otype, oid = meta.split()
        if otype == "blob":
            entries[path] = oid
    return entries


def blob(repo: Path, oid: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "blob", oid],
                          check=True, capture_output=True).stdout


def changed(repo: Path, base: str, side: str) -> set[str]:
    out = git(repo, "diff", "--name-only", "-z", base, side)
    return {p for p in out.split("\0") if p}


def norm_lines(data: bytes) -> set[str]:
    """Normalised non-trivial source lines (indentation and blank lines ignored)."""
    keep: set[str] = set()
    for raw in data.decode("utf-8", "replace").splitlines():
        line = " ".join(raw.split())
        if len(line) < 8:            # punctuation-only lines (``):`` ``else:`` ...)
            continue
        keep.add(line)
    return keep


def py_symbols(data: bytes) -> set[str]:
    import ast
    try:
        mod = ast.parse(data.decode("utf-8", "replace"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def audit(repo: Path, base: str, ours: str, theirs: str, merged: str,
          max_lines: int = 40) -> dict:
    t_merged = tree(repo, merged)
    c_ours = changed(repo, base, ours)
    c_theirs = changed(repo, base, theirs)
    both = sorted(c_ours & c_theirs)

    report: dict = {
        "base": base, "ours": ours, "theirs": theirs, "merged": merged,
        "ours_only": sorted(c_ours - c_theirs),
        "theirs_only": sorted(c_theirs - c_ours),
        "both": both, "deleted": [], "files": {}, "symbol_loss": [],
        "line_loss": [],
    }

    for path in sorted(c_ours | c_theirs):
        if path in c_ours and path not in t_merged:
            report["deleted"].append({"path": path, "lost_from": "ours"})
        if path in c_theirs and path not in t_merged:
            report["deleted"].append({"path": path, "lost_from": "theirs"})

    for path in both:
        if path not in t_merged:
            continue                      # already reported as deleted
        suffix = Path(path).suffix
        if suffix not in TEXT_SUFFIXES:
            report["files"][path] = {"binary": True}
            continue
        mbytes = blob(repo, t_merged[path])
        o = blob(repo, tree(repo, ours)[path]) if path in tree(repo, ours) else b""
        t = blob(repo, tree(repo, theirs)[path]) if path in tree(repo, theirs) else b""
        m_lines, o_lines, t_lines = norm_lines(mbytes), norm_lines(o), norm_lines(t)
        entry = {
            "missing_ours": sorted(o_lines - m_lines)[:max_lines],
            "missing_theirs": sorted(t_lines - m_lines)[:max_lines],
            "n_missing_ours": len(o_lines - m_lines),
            "n_missing_theirs": len(t_lines - m_lines),
            "n_lines_merged": len(m_lines),
        }
        if suffix == ".py":
            m_sym, o_sym, t_sym = py_symbols(mbytes), py_symbols(o), py_symbols(t)
            entry["symbols_missing_ours"] = sorted(o_sym - m_sym)
            entry["symbols_missing_theirs"] = sorted(t_sym - m_sym)
            entry["symbols_merged"] = len(m_sym)
            if m_sym - (o_sym | t_sym):
                entry["symbols_new_in_merge"] = sorted(m_sym - (o_sym | t_sym))
            for side, missing in (("ours", entry["symbols_missing_ours"]),
                                  ("theirs", entry["symbols_missing_theirs"])):
                for sym in missing:
                    report["symbol_loss"].append(
                        {"path": path, "side": side, "symbol": sym})
        if entry["n_missing_ours"]:
            report["line_loss"].append({"path": path, "side": "ours",
                                        "n": entry["n_missing_ours"]})
        if entry["n_missing_theirs"]:
            report["line_loss"].append({"path": path, "side": "theirs",
                                        "n": entry["n_missing_theirs"]})
        report["files"][path] = entry

    report["ok"] = not report["deleted"] and not report["symbol_loss"]
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--json")
    ap.add_argument("--max-lines", type=int, default=40)
    args = ap.parse_args(argv)

    rep = audit(Path(args.repo), args.base, args.ours, args.theirs,
                args.merged, args.max_lines)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("ok", "deleted", "symbol_loss",
                                         "line_loss", "both")}, indent=2))
    for path, entry in rep["files"].items():
        if entry.get("binary"):
            print(f"{path}: binary (not line-audited)")
            continue
        print(f"{path}: merged_lines={entry['n_lines_merged']} "
              f"missing_ours={entry['n_missing_ours']} "
              f"missing_theirs={entry['n_missing_theirs']} "
              f"symbols_missing={entry.get('symbols_missing_ours')}"
              f"{entry.get('symbols_missing_theirs')}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
