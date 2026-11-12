"""Pins the Couette wall-shear reading audit (docs/audit_couette_wall_shear_reading.md).

The audited artefact ``benchmarks/couette/out_full/shear_reading.json`` is
committed, so these assertions re-read it and can never re-derive a number by
hand.  What is pinned:

1. the published artefact really does contain the -32.6 % one-spacing secant
   reading at ``H/dx = 8`` and ``t_end = 0.15 s`` (0.3 viscous times);
2. with the predeclared terminal time (``t_end = 1.2 H^2/nu = 0.60 s``) the same
   estimator on the same lattice is inside the 5 % gate -- i.e. the published
   "wall shear under-predicted ~33 %" is a startup transient, not a wall-model
   deficit;
3. the improvement is monotone in ``t_end`` at fixed ``H/dx``;
4. at the reference resolution both shear readings (secant and interior fit) are
   inside the 5 % gate, and the residual relaxation (steady drift) is recorded
   honestly rather than gated away.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmarks" / "couette" / "out_full" / "shear_reading.json"


@pytest.fixture(scope="module")
def audit():
    if not AUDIT.exists():
        pytest.skip(f"audit artefact missing: {AUDIT}")
    return json.loads(AUDIT.read_text())


def _by_label(audit, label):
    for row in audit["rows"]:
        if row["label"] == label:
            return row
    raise AssertionError(f"label {label!r} not in {[r['label'] for r in audit['rows']]}")


def test_artifact_declares_the_design_terminal_time(audit):
    assert audit["design_t_end_s"] == pytest.approx(0.6, abs=1e-9)
    gate = audit["gates"]
    assert gate["wall_shear_rel"] == pytest.approx(0.05)
    assert gate["steady_drift"] == pytest.approx(0.01)


def test_published_quick_run_reproduces_the_recorded_deficit(audit):
    """The committed --quick artefact is the source of the -32.6 % claim."""
    pub = _by_label(audit, "ladder-r8")
    assert pub["t_end"] == pytest.approx(0.15)
    assert pub["wall_shear_ratio_secant"] == pytest.approx(0.6735, abs=0.002)
    assert pub["steady_drift"] > 0.05  # the transient that produced it


def test_design_time_removes_the_deficit_at_the_same_resolution(audit):
    short = _by_label(audit, "ladder-r8")
    long_ = _by_label(audit, "ladder-r8-steady")
    assert long_["t_end"] == pytest.approx(0.6)
    # same lattice, same estimator, only t_end changed
    assert long_["n_fluid"] == short["n_fluid"]
    assert abs(long_["wall_shear_ratio_secant"] - 1.0) < 0.05
    assert abs(long_["wall_shear_ratio_secant"] - 1.0) < \
        abs(short["wall_shear_ratio_secant"] - 1.0) - 0.2


def test_reference_resolution_both_readings_inside_the_gate(audit):
    r16 = _by_label(audit, "ladder-r16-steady")
    assert abs(r16["wall_shear_ratio_secant"] - 1.0) < 0.05
    assert abs(r16["wall_shear_ratio_fit"] - 1.0) < 0.05
    assert r16["interior_l2"] < 0.05
    assert abs(r16["flux_ratio"] - 1.0) < 0.05


def test_relaxation_is_recorded_not_hidden(audit):
    """1.2 viscous times is still not steady; the audit says so, so the test does."""
    r16 = _by_label(audit, "ladder-r16-steady")
    assert r16["steady_drift"] > audit["gates"]["steady_drift"]
    assert r16["steady_drift"] < 0.03


def test_figure_is_committed(audit):
    fig = AUDIT.parent / "shear_reading.png"
    assert fig.exists() and fig.stat().st_size > 10_000
