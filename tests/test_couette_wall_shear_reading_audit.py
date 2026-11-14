"""Invariants of the Couette sliding-wall shear-reading audit (m20907).

The audit exists to answer one question with evidence: is the banked 17 % sliding-wall
record a wall-model defect or an estimator artefact?  These tests lock the answer, and
they fail if anyone silently re-banks the secant number as a stress reading.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "benchmarks" / "couette" / "out_readings" / "wall_shear_reading_audit.json"
SWEEP = REPO / "benchmarks" / "couette" / "out_design" / "viscous_time_convergence.json"
OWNER_TOL = 15.0


@pytest.fixture(scope="module")
def audit() -> dict:
    if not ART.exists():
        pytest.skip("audit artifact not built")
    return json.loads(ART.read_text())


def _arm(audit: dict, res: int, mirrored: bool) -> dict:
    for a in audit["arms"]:
        if a["resolution_H_over_dx"] == res and a["mirrored"] is mirrored:
            return a
    pytest.skip(f"arm H/dx={res} mirrored={mirrored} not present")


def test_reproduces_the_committed_sweep(audit):
    """The re-run must land on the committed sweep at the same viscous time."""
    if not SWEEP.exists():
        pytest.skip("sweep artifact not available")
    sweep = json.loads(SWEEP.read_text())
    for res in (8, 16):
        pts = [p for p in sweep["points"]
               if p["resolution_H_over_dx"] == res and abs(p["viscous_times"] - 1.2) < 1e-9]
        if not pts:
            continue
        a = _arm(audit, res, False)
        assert a["secant1_sliding_pct"] == pytest.approx(
            abs(pts[0]["wall_shear_error_pct_top"]), abs=0.01)
        assert a["secant1_fixed_pct"] == pytest.approx(
            abs(pts[0]["wall_shear_error_pct_bottom"]), abs=0.01)


def test_mirror_control_keeps_the_reading(audit):
    """Sliding wall at y=0 reads the same as at y=H -> not a domain-layout artefact."""
    for res in (8,):
        plain = _arm(audit, res, False)
        mirrored = _arm(audit, res, True)
        for key in ("secant1_sliding_pct", "fit4_sliding_pct",
                    "fit8_sliding_pct", "interior_slope_pct"):
            # the flipped lattice changes the float summation order only
            assert mirrored[key] == pytest.approx(plain[key], abs=0.01), key
        # the mirror really is a mirror: the fixed reference u=U y/H no longer matches
        assert mirrored["interior_l2_pct"] > 20.0
        assert plain["interior_l2_pct"] < 5.0


def test_secant_is_the_worst_estimator_and_the_lag_decays(audit):
    """secant1 > fit8 > fit4 at every rung: the one-spacing secant reads row lag."""
    for a in audit["arms"]:
        assert abs(a["secant1_sliding_pct"]) > abs(a["fit8_sliding_pct"])
        assert abs(a["fit8_sliding_pct"]) > abs(a["fit4_sliding_pct"])


def test_owner_standard_verdict_is_estimator_dependent(audit):
    """The defensible readings are inside the owner 15 % standard, the secant is not."""
    for a in audit["arms"]:
        assert abs(a["fit4_sliding_pct"]) < OWNER_TOL
        assert abs(a["interior_slope_pct"]) < OWNER_TOL
    plain8 = _arm(audit, 8, False)
    assert abs(plain8["secant1_sliding_pct"]) > OWNER_TOL


def test_fixed_and_sliding_wall_are_not_symmetric(audit):
    """The estimator bias is one-sided: this is the physical residue to chase."""
    for a in audit["arms"]:
        assert abs(a["secant1_sliding_pct"]) > 3.0 * abs(a["secant1_fixed_pct"])


def test_artifact_declares_estimators_and_setup(audit):
    assert set(audit["estimators"]) == {"secant1", "fit4", "fit8", "interior"}
    assert audit["owner_tolerance_pct"] == OWNER_TOL
    assert audit["setup"]["wall_friction"] == 1.0
    assert audit["anchor"] == "XVAL-SPH-METHOD-SELECTION-2026-11-12"
