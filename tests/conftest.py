"""Pytest configuration for TensorSPH test suite.

Marks simulation-heavy tests as ``slow`` so they can be skipped in quick CI
runs (``pytest -m 'not slow'``).  Also sets per-test timeouts that reflect
the actual wall-clock cost on the reference platform (CPU, torch 2.7+).

The baseline at m281 (torch 2.14.1, Python 3.12) ran 104/104 in 18.55 s.
The current environment (torch 2.7.1, Python 3.11) is measurably slower on
large particle counts, so the timeouts below are generous enough to pass on
both platforms while still catching genuine hangs.
"""
import pytest

# --------------------------------------------------------------------------- #
# Worktree import guard
# --------------------------------------------------------------------------- #
# The host has several agent worktrees registered on sys.path through .pth files
# in site-packages (m57, m708, m991, m1213, m2420 at the time of writing). Whichever
# one comes first wins, so a bare ``import tensorsph`` inside this worktree can
# silently exercise another agent's (possibly older or newer) source tree, and a
# green suite then says nothing about the code under test. Putting this worktree's
# ``src`` first, and failing loudly if the package still comes from elsewhere,
# makes the suite score the code that is actually being committed.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)

_imported = sys.modules.get("tensorsph")
if _imported is not None and getattr(_imported, "__file__", ""):
    if not Path(_imported.__file__).resolve().is_relative_to(_REPO_ROOT):
        raise RuntimeError(
            f"tensorsph was imported from {_imported.__file__}, not from {_SRC}; "
            "the suite would be scoring another worktree"
        )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: simulation-heavy tests that take >10 s"
    )


# Tests that run full SPH simulations and need a longer timeout.
# Wall-clock times measured on the current platform (torch 2.7.1, CPU):
#   test_pool_with_walls_stays_bounded_and_does_not_clump  ~35 s
#   test_no_slip_reproduces_the_stokes_shear_decay_rate    ~67 s
#   test_the_adaptive_step_does_not_change_the_answer_much ~58 s
#   test_an_accelerating_flow_cannot_exceed_the_limit      ~83 s
#   test_compression_before_wall_impact_is_separated_...   ~45 s
SLOW_TIMEOUT = 120  # seconds, generous margin over measured times


def pytest_collection_modifyitems(config, items):
    """Auto-mark known slow tests so ``pytest -m 'not slow'`` works."""
    slow_names = {
        "test_pool_with_walls_stays_bounded_and_does_not_clump",
        "test_no_slip_reproduces_the_stokes_shear_decay_rate",
        "test_the_adaptive_step_does_not_change_the_answer_much",
        "test_an_accelerating_flow_cannot_exceed_the_limit",
        "test_compression_before_wall_impact_is_separated_from_the_impact_spike",
    }
    for item in items:
        if item.originalname in slow_names or item.name in slow_names:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.timeout(SLOW_TIMEOUT))
