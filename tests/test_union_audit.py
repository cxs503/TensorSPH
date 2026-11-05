"""Negative controls for ``tools/union_audit.py`` (PR #118 merge discipline).

The auditor exists because a clean ``git merge`` exit code says nothing about
content.  These tests build throw-away repositories with three branches and
check that the auditor

* passes a genuine union resolution,
* fails when one side's function silently disappears (the historical failure
  mode of mechanical overlays), and
* fails when a path contributed by one side is deleted.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "union_audit.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("union_audit_m3982", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def commit(repo: Path, msg: str) -> str:
    git(repo, "add", "-A")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def make_repo(tmp_path: Path, merged_text: str, delete_shared: bool = False):
    """base -> {ours, theirs}; both touch module.py, ours adds a second file."""
    repo = tmp_path / "r"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    (repo / "module.py").write_text(
        "def common():\n    return 1\n\n\ndef ours_only():\n    return 2\n")
    base = commit(repo, "base")

    git(repo, "checkout", "-qb", "theirs")
    (repo / "module.py").write_text(
        "def common():\n    return 1\n\n\ndef ours_only():\n    return 2\n\n\n"
        "def theirs_only():\n    return 3\n")
    theirs = commit(repo, "theirs side")

    git(repo, "checkout", "-q", "main")
    (repo / "module.py").write_text(
        "def common():\n    return 10\n\n\ndef ours_only():\n    return 2\n")
    (repo / "extra.txt").write_text("ours side note, long enough to be audited\n")
    ours = commit(repo, "ours side")

    git(repo, "checkout", "-qb", "merged")
    (repo / "module.py").write_text(merged_text)
    if delete_shared:
        (repo / "extra.txt").unlink()
    else:
        (repo / "extra.txt").write_text("ours side note, long enough to be audited\n")
    merged = commit(repo, "merge result")
    return repo, base, ours, theirs, merged


UNION = ("def common():\n    return 10\n\n\ndef ours_only():\n    return 2\n\n\n"
         "def theirs_only():\n    return 3\n")


def test_union_resolution_passes(tmp_path):
    repo, base, ours, theirs, merged = make_repo(tmp_path, UNION)
    rep = load_tool().audit(repo, base, ours, theirs, merged)
    assert rep["ok"] is True, rep
    assert rep["deleted"] == []
    assert rep["symbol_loss"] == []
    assert rep["both"] == ["module.py"]


def test_side_pick_losing_a_symbol_is_caught(tmp_path):
    """Overlay resolution (ours wins verbatim) must be reported as symbol loss."""
    ours_only = "def common():\n    return 10\n\n\ndef ours_only():\n    return 2\n"
    repo, base, ours, theirs, merged = make_repo(tmp_path, ours_only)
    rep = load_tool().audit(repo, base, ours, theirs, merged)
    assert rep["ok"] is False
    assert {"path": "module.py", "side": "theirs", "symbol": "theirs_only"} in \
        rep["symbol_loss"]
    # the lost body shows up as a missing line as well
    entry = rep["files"]["module.py"]
    assert any("theirs_only" in line for line in entry["missing_theirs"])


def test_deleted_file_is_caught(tmp_path):
    repo, base, ours, theirs, merged = make_repo(tmp_path, UNION, delete_shared=True)
    rep = load_tool().audit(repo, base, ours, theirs, merged)
    assert rep["ok"] is False
    assert rep["deleted"] == [{"path": "extra.txt", "lost_from": "ours"}]


def test_cli_exit_codes(tmp_path):
    repo, base, ours, theirs, merged = make_repo(tmp_path, UNION)
    tool = load_tool()
    assert tool.main(["--repo", str(repo), "--base", base, "--ours", ours,
                      "--theirs", theirs, "--merged", merged]) == 0
    bad = make_repo(tmp_path / "bad",
                    "def common():\n    return 10\n\n\ndef ours_only():\n    return 2\n")
    assert tool.main(["--repo", str(bad[0]), "--base", bad[1], "--ours", bad[2],
                      "--theirs", bad[3], "--merged", bad[4]]) == 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
