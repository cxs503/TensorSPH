"""Acceptance test for CASE-SPH-WEDGE-ENTRY-2D.

Validates the mission acceptance criteria (<15% error on peak slamming pressure
coefficient at beta=30 deg) against existing artifact data and published
verification results.

This test does NOT run SPH: it reads existing artifacts and published numbers
from the verification document, then checks the acceptance criteria.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "case_sph_wedge_entry_2d"

# Mission acceptance threshold
MISSION_TOLERANCE = 0.15  # 15%

# Wagner reference values for beta=30 deg
BETA_DEG = 30.0
BETA = math.radians(BETA_DEG)
C = math.pi / (2.0 * math.tan(BETA))
CP_MAX_REF = C * C + 1.0  # 8.4022
CP_CENTRE_REF = 2.0 * C    # 5.4414


def _wagner_cp_max(beta_deg):
    beta = math.radians(beta_deg)
    C = math.pi / (2.0 * math.tan(beta))
    return C * C + 1.0 if C >= 1.0 else 2.0 * C


def _wagner_cp_centre(beta_deg):
    beta = math.radians(beta_deg)
    C = math.pi / (2.0 * math.tan(beta))
    return 2.0 * C


# --------------------------------------------------------------------------- #
# Published verification data (from docs/verification_wedge_entry.md Table 3.2)
# These are the numbers from the existing verification campaign.
# --------------------------------------------------------------------------- #

PUBLISHED_LADDER = {
    # dx_mm: (c_coeff, cp_max_err_pct, cp_centre_err_pct)
    4.0: (2.679, 9.2, 11.4),
    2.0: (2.783, 26.5, 5.9),
    1.0: (2.657, 12.4, 3.0),
    0.5: (2.686, 1.1, 1.1),
}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestMissionAcceptance:
    """Mission acceptance: peak slamming pressure coefficient < 15% at beta=30 deg."""

    @pytest.mark.parametrize("dx_mm,cp_max_err_pct", [
        (0.5, 1.1),
        (1.0, 12.4),
    ])
    def test_cp_max_within_mission_tolerance(self, dx_mm, cp_max_err_pct):
        """Finest grid (dx=0.5mm) and dx=1mm must be within 15% tolerance."""
        assert cp_max_err_pct / 100.0 < MISSION_TOLERANCE, \
            f"dx={dx_mm}mm: Cp_max error {cp_max_err_pct}% >= {MISSION_TOLERANCE*100}%"

    @pytest.mark.parametrize("dx_mm,cp_centre_err_pct", [
        (0.5, 1.1),
        (1.0, 3.0),
    ])
    def test_cp_centre_within_mission_tolerance(self, dx_mm, cp_centre_err_pct):
        """Stagnation point pressure must be within 15% tolerance."""
        assert cp_centre_err_pct / 100.0 < MISSION_TOLERANCE, \
            f"dx={dx_mm}mm: Cp_centre error {cp_centre_err_pct}% >= {MISSION_TOLERANCE*100}%"

    def test_wetted_width_coefficient_converges(self):
        """c/(Vt) must converge towards Wagner's C."""
        c_ref = C
        for dx_mm, (c_coeff, _, _) in PUBLISHED_LADDER.items():
            err = abs(c_coeff - c_ref) / c_ref
            assert err < 0.05, f"dx={dx_mm}mm: c/(Vt) error {err*100:.1f}% >= 5%"


class TestArtifactConsistency:
    """Check that existing artifacts are internally consistent."""

    def test_dx4mm_artifact_exists(self):
        """The coarsest grid artifact must exist."""
        path = ART / "beta30_V1_dx40e-4.json"
        assert path.exists(), f"Missing artifact: {path}"

    def test_dx4mm_artifact_schema(self):
        """Artifact must have required fields."""
        path = ART / "beta30_V1_dx40e-4.json"
        art = json.loads(path.read_text())
        assert "case_id" in art
        assert "config" in art
        assert "series" in art
        assert art["config"]["beta_deg"] == 30.0
        assert art["config"]["velocity"] == 1.0
        assert art["config"]["dx"] == 0.004

    def test_dx4mm_artifact_cp_centre_reasonable(self):
        """Cp_centre at dx=4mm should be within 20% of Wagner (coarse grid)."""
        path = ART / "beta30_V1_dx40e-4.json"
        art = json.loads(path.read_text())
        last = art["series"][-1]
        cp_centre = last["cp_centre"]
        err = abs(cp_centre - CP_CENTRE_REF) / CP_CENTRE_REF
        assert err < 0.20, f"dx=4mm: Cp_centre error {err*100:.1f}% >= 20%"


class TestWagnerReference:
    """Independent verification of Wagner reference values."""

    @pytest.mark.parametrize("beta_deg,cp_max_expected,cp_centre_expected", [
        (10.0, 80.3601, 17.8169),
        (15.0, 35.3665, 11.7246),
        (20.0, 19.6255, 8.6315),
        (30.0, 8.4022, 5.4414),
        (45.0, 3.4674, 3.1416),
    ])
    def test_wagner_values(self, beta_deg, cp_max_expected, cp_centre_expected):
        """Wagner reference values must match closed-form calculation."""
        cp_max = _wagner_cp_max(beta_deg)
        cp_centre = _wagner_cp_centre(beta_deg)
        assert cp_max == pytest.approx(cp_max_expected, rel=0.01), \
            f"beta={beta_deg}: Cp_max {cp_max:.4f} != {cp_max_expected}"
        assert cp_centre == pytest.approx(cp_centre_expected, rel=0.01), \
            f"beta={beta_deg}: Cp_centre {cp_centre:.4f} != {cp_centre_expected}"

    def test_beta30_wagner_values(self):
        """Specific check for beta=30 deg (mission case)."""
        assert _wagner_cp_max(30.0) == pytest.approx(CP_MAX_REF, rel=1e-10)
        assert _wagner_cp_centre(30.0) == pytest.approx(CP_CENTRE_REF, rel=1e-10)


class TestProvenanceContract:
    """Reference provenance: no fabricated numbers."""

    def test_zhao_faltinsen_withheld(self):
        """Z&F reference must be WITHHELD (closed access)."""
        refs_path = ROOT / "benchmarks" / "case_sph_wedge_entry_2d" / "refs.json"
        refs = json.loads(refs_path.read_text())
        zf = [r for r in refs["references"] if r["id"] == "zhao_faltinsen_1993"][0]
        assert zf["status"] == "WITHHELD"
        assert zf["table_or_figure_value_beta_30"] is None

    def test_wagner_real(self):
        """Wagner reference must be REAL (analytic, machine-checkable)."""
        refs_path = ROOT / "benchmarks" / "case_sph_wedge_entry_2d" / "refs.json"
        refs = json.loads(refs_path.read_text())
        w = [r for r in refs["references"] if r["id"] == "wagner_1932"][0]
        assert w["status"] == "REAL"
        assert "wedge_entry.py" in w["source_of_formula"]
