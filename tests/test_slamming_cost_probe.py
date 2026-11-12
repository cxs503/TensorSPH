"""Tests for ``benchmarks/slamming/cost_probe.py`` (campaign affordability probe).

The probe answers an arithmetic question -- does one campaign case fit a
wall-clock budget? -- so what is pinned here is the *arithmetic*, never a
measured runtime (a test that asserted a wall clock would be a flaky test).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "benchmarks" / "slamming" / "cost_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("slamming_cost_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_campaign_steps_uses_the_campaign_window():
    """Anchor on the real solver: c_end = 50 mm, dx = 4 mm, beta = 20 deg."""
    mod = _load()
    steps, dt = mod.campaign_steps(4.0e-3, 20.0, 2.0, 0.050, 41)
    assert steps > 0 and dt > 0.0
    t_end = 2.0 * 0.050 * math.tan(math.radians(20.0)) / (math.pi * 2.0)
    assert math.isclose(steps * dt, t_end, rel_tol=0.05)


def test_projection_record_is_self_consistent():
    """`projected_wall_s` must equal mean step cost x campaign step count.

    The record is produced with a one-step probe (cheap); the assertion is the
    identity between the three numbers, not their size, so it cannot flake.
    """
    mod = _load()
    rec = mod.probe(4.0e-3, 1, 20.0, 2.0, 0.010, 2, 0.1, True, 1)
    assert rec["probe_steps"] >= 1
    assert rec["per_step_s"] > 0.0
    projected = rec["per_step_s"] * rec["campaign_case"]["steps"]
    assert math.isclose(rec["campaign_case"]["projected_wall_s"], projected,
                        rel_tol=1e-12)
    assert rec["campaign_case"]["steps"] >= rec["probe_steps"]
    assert "extrapolation" in rec["campaign_case"]
