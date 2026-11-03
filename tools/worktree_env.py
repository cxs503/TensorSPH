"""Worktree-pinned import guard for TensorSPH verification harnesses.

Why this module exists
----------------------
A single host Python environment carries *editable* installs of the solver
packages whose ``.pth`` files point at one particular worktree, e.g.

    /usr/lib/python3.11/site-packages/__editable__.tensorsph-0.8.0.pth
        -> /root/tp-agent/work/m2420/src

Because ``sys.path`` therefore already contains another agent's worktree,
``import tensorsph`` inside worktree ``m2730`` silently resolves to ``m2420``
(source line: ``python -c "import tensorsph; print(tensorsph.__file__)"``).
Every test run, benchmark run and ``python -m tensorsph`` smoke run then
grades *a different branch*: false reds waste budget, and -- worse -- false
greens certify someone else's numbers as yours, since the numbers are
attributable to no worktree at all.

``ensure_worktree_source()`` removes the ambiguity: it puts ``<repo>/src`` at
the front of ``sys.path`` and then *verifies* where the package really came
from, so a stale environment fails loudly instead of quietly measuring the
wrong code.

Usage (tests/conftest.py, benchmark entry points)::

    from tools.worktree_env import ensure_worktree_source
    ensure_worktree_source(__file__)
"""
from __future__ import annotations

import sys
from pathlib import Path


def repo_root_for(anchor: str | Path) -> Path:
    """Repo root for ``anchor``: the nearest ancestor that contains ``src``."""
    path = Path(anchor).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "src").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"no repository root (src/ + pyproject.toml) above {path}")


def foreign_source_error(module_file: str | Path | None, repo_root: str | Path) -> str | None:
    """Pure predicate: why ``module_file`` is *not* this worktree's source.

    Returns ``None`` when the module really lives in ``<repo_root>/src``,
    otherwise a human-readable diagnosis.  Kept free of imports so the negative
    control in ``tests/test_harness_integrity.py`` can exercise it directly.
    """
    root = Path(repo_root).resolve()
    if module_file is None:
        return "package resolved to a namespace package (no __file__)"
    origin = Path(module_file).resolve()
    src = (root / "src").resolve()
    if origin == src or src in origin.parents:
        return None
    return (f"tensorsph was imported from {origin}, which is outside this "
            f"worktree ({src}); a stale editable install or a foreign "
            f"PYTHONPATH entry is shadowing the source under test.")


def ensure_worktree_source(anchor: str | Path, package: str = "tensorsph",
                           verbose: bool = True) -> Path:
    """Pin ``package`` to the worktree containing ``anchor``.

    Prepends ``<repo>/src`` to ``sys.path`` (idempotent) and raises
    ``RuntimeError`` if the package still resolves elsewhere.  Returns the
    resolved ``__init__.py``.
    """
    root = repo_root_for(anchor)
    src = root / "src"
    entry = str(src)
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

    module = __import__(package)
    error = foreign_source_error(getattr(module, "__file__", None), root)
    if error is not None:
        raise RuntimeError(
            f"{package}: {error}\n"
            f"Fix: run with PYTHONPATH={src}, or reinstall the editable package "
            f"from {root} (pip install -e . --no-deps).")
    if verbose:
        print(f"[worktree_env] {package} <- {module.__file__}", file=sys.stderr)
    return Path(module.__file__).resolve()