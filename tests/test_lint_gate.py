"""Gate: ``ruff check .`` (the CI step that runs *before* pytest) must be clean.

Why this file exists
--------------------
The lint gate has gone red on `main` three times for the same reason -- unrelated
benchmark/documentation scripts -- and each time it was cured by hand (m3804,
m4638, this change).  Nothing in the test suite noticed, so the debt could be
re-introduced silently.  These two tests make the gate machine-checkable:

* ``test_ruff_check_is_clean`` runs the exact CI command;
* ``test_ruff_waiver_is_narrow`` pins *how* the remaining noise is waived, so
  the waiver cannot silently widen to the whole ``benchmarks`` tree.  The
  benchmark runners are real code: the ``F841`` cleared here
  (``contact_angle_geometry/run.py``) is precisely the kind of bug a blanket
  ``exclude = ["benchmarks"]`` would have hidden forever.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def test_ruff_check_is_clean():
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:  # pragma: no cover - ruff is in the [test] extra
        pytest.skip("ruff not installed: pip install -e '.[test]'")

    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "ruff check . is red -- CI lints before pytest, so this blocks every PR:\n"
        f"{proc.stdout}{proc.stderr}"
    )


def test_ruff_waiver_is_narrow():
    cfg = tomllib.loads(PYPROJECT.read_text())
    lint = cfg["tool"]["ruff"]["lint"]
    ignores = lint.get("per-file-ignores", {})

    # The waiver must not cover a tree-wide pattern, and it must be E501 only:
    # pyflakes (F) and W stay enforced everywhere.
    assert list(ignores) == ["benchmarks/**/make_doc.py"], ignores
    for pattern, rules in ignores.items():
        assert rules == ["E501"], (pattern, rules)
        assert pattern.endswith("/make_doc.py"), pattern

    # A blanket `exclude = ["benchmarks"]` would delete the lint coverage of the
    # benchmark runners (where the dead local was found) -- keep it out.
    excluded = cfg["tool"]["ruff"].get("exclude", [])
    assert "benchmarks" not in excluded, excluded
    assert set(lint["select"]) == {"E", "F", "W"}
