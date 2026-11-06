"""Pipeline E2E smoke test for mission m4616 (TensorSPH).

Verifies the deliverable chain: repo-root marker module is importable from
the test tree and carries the mission-id constant, plus a trivial arithmetic
sanity check. No solver code is exercised.
"""
import sys
from pathlib import Path

# The smoke marker lives at the repository root; make it importable
# regardless of how pytest is invoked (plain `pytest` vs `python -m pytest`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smoke_marker_m4616  # noqa: E402


def test_smoke_marker_constant_matches_mission_id():
    assert smoke_marker_m4616.PIPELINE_SMOKE_M4616 == "4616"


def test_arithmetic_sanity():
    assert 1 + 1 == 2
