"""Guarding test for the CASE-POISEUILLE-SPH anchor verdict.

Anchor: CASE-POISEUILLE-SPH -- ref: analytic parabola u = g y (H-y) / (2 nu),
tolerance <5 %.

The anchor is graded from committed artifacts only (no solver is run here), so the
test has to do two jobs that an ordinary "read the number and compare" test does not:

1. **Binding** -- every published derived number must be reproducible from the raw
   per-row means.  `benchmarks/poiseuille_mirror/gauge_audit.py` recomputes them
   independently of `run.py`; here we re-run that recomputation and require
   agreement to 1e-6 relative with the stored values, and re-hash the source
   artifact so the verdict cannot drift away from the data it claims to grade.

2. **No over-certification** -- the published gate C is written on one gauge (the
   slope of a *global* parabola fit), which reproduces the analytic wall slope for
   any parabola of the right height.  This test therefore refuses a verdict that
   passes only on that gauge: it requires the momentum budget to agree as well, and
   it pins the *resolution requirement* that the stricter gauges carry (the
   bias-corrected near-wall secant needs H/dx >= 16; the all-row profile rms needs
   H/dx >= 12).  If a future run claims <5 % at a coarser rung, this test fails.

Regenerate the verdict with:
    python benchmarks/poiseuille_mirror/gauge_audit.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "benchmarks" / "poiseuille_mirror" / "out" / "results.json"
VERDICT = REPO / "benchmarks" / "poiseuille_mirror" / "case_verdict.json"

sys.path.insert(0, str(REPO / "benchmarks" / "poiseuille_mirror"))
import gauge_audit as ga  # noqa: E402  (path insertion must precede the import)

TOL_PCT = 5.0


@pytest.fixture(scope="module")
def data():
    assert SRC.exists(), f"missing committed artifact {SRC}"
    return json.loads(SRC.read_text())


@pytest.fixture(scope="module")
def verdict():
    assert VERDICT.exists(), (
        f"missing anchor verdict {VERDICT}; regenerate with "
        "python benchmarks/poiseuille_mirror/gauge_audit.py")
    return json.loads(VERDICT.read_text())


def _gate(verdict, gid):
    for g in verdict["gates"]:
        if g["id"] == gid:
            return g
    raise AssertionError(f"gate {gid} missing from verdict")


# --------------------------------------------------------------------------- #
# 1. binding: artifact identity and independent recomputation
# --------------------------------------------------------------------------- #

def test_verdict_is_bound_to_the_committed_artifact(verdict):
    """The verdict must name the exact bytes it was computed from."""
    digest = hashlib.sha256(SRC.read_bytes()).hexdigest()
    assert verdict["source_sha256"] == digest, (
        "verdict was built from a different results.json than the one committed "
        f"(verdict {verdict['source_sha256'][:12]}, file {digest[:12]})")


def test_anchor_metadata(verdict):
    assert verdict["anchor"] == "CASE-POISEUILLE-SPH"
    assert verdict["tolerance_pct"] == TOL_PCT
    assert len(verdict["runs"]) == 3, "the resolution ladder must keep all three rungs"


def test_stored_numbers_are_reproducible_from_raw_rows(verdict):
    """Every published derived number is recomputed from the raw row means.

    This is the anti-fabrication check: a stored value that cannot be rebuilt from
    `results[*].rows` (the only raw observable in the file) is unverifiable.
    """
    block = verdict["consistency"]
    assert block["all_agree_1e-6"] is True
    for rung, pairs in block["stored_vs_recomputed"].items():
        for name, rec in pairs.items():
            if rec["agree"] is None:
                continue
            assert rec["agree"], f"H/dx={rung}: {name} stored {rec['stored']} != recomputed {rec['recomputed']}"


def test_recomputation_is_live_not_copied(data, verdict):
    """Re-run the audit here and match the committed verdict to 1e-9.

    Guards against a hand-edited verdict: the numbers must fall out of the rows.
    """
    rho, h = data["rho0"], data["height"]
    for run, rec in zip(sorted(data["results"], key=lambda r: r["resolution_H_over_dx"]),
                        verdict["runs"]):
        prof = ga.profile_metrics(run["rows"], run["body_force"], h, run["viscosity"])
        assert prof["u_max_error_pct"] == pytest.approx(rec["profile"]["u_max_error_pct"], abs=1e-9)
        assert prof["rms_all_rel_pct"] == pytest.approx(rec["profile"]["rms_all_rel_pct"], abs=1e-9)
        wall = ga.wall_gauges(run["rows"], run["body_force"], h, run["viscosity"], rho)
        g4 = wall["gauges"]["G4_global_fit_slope"]["ratio"]
        rung = str(rec["resolution_H_over_dx"])
        stored_g4 = _gate(verdict, "C_wall_shear_battery")["per_ratio"][rung]["G4_global_fit_slope"]["ratio"]
        assert g4 == pytest.approx(stored_g4, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. the anchor gates
# --------------------------------------------------------------------------- #

def test_gate_A_centreline_within_tolerance(verdict):
    g = _gate(verdict, "A_centreline")
    assert g["status"] == "PASS"
    assert abs(g["measured"]) <= TOL_PCT
    # the published rung values, so a regression shows the ladder, not one number
    for rec in verdict["runs"]:
        assert abs(rec["profile"]["u_max_error_pct"]) <= TOL_PCT, (
            f"H/dx={rec['resolution_H_over_dx']} centreline error "
            f"{rec['profile']['u_max_error_pct']:.3f} % exceeds {TOL_PCT} %")


def test_gate_B_interior_profile_within_tolerance(verdict):
    g = _gate(verdict, "B_profile_rms_interior")
    assert g["status"] == "PASS"
    assert g["measured"] <= TOL_PCT


def test_gate_B2_declares_its_resolution_requirement(verdict):
    """The all-row profile reading only meets 5 % from H/dx = 12 upward."""
    g = _gate(verdict, "B2_profile_rms_all_rows")
    per_rung = g["measured_per_rung"]
    assert per_rung["8"] > TOL_PCT, (
        "H/dx=8 must be recorded as failing the all-row reading; the interior-only "
        "gate hides the wall rows")
    assert per_rung["12"] <= TOL_PCT and per_rung["16"] <= TOL_PCT
    assert g["min_resolution_passing"] == 12
    assert verdict["resolution_requirement"]["profile_all_rows"] == "H/dx >= 12"


def test_gate_D_wall_slip_within_tolerance(verdict):
    g = _gate(verdict, "D_wall_slip")
    assert g["status"] == "PASS"
    assert g["measured"] <= 0.05


# --------------------------------------------------------------------------- #
# 3. no over-certification on the wall-shear claim
# --------------------------------------------------------------------------- #

def test_wall_shear_passes_on_more_than_the_published_gauge(verdict):
    """G4 (global fit) alone must not carry the PASS: the budget must agree."""
    per = _gate(verdict, "C_wall_shear_battery")["per_ratio"]
    for rung, row in per.items():
        assert row["G4_global_fit_slope"]["pass_5pct"], f"H/dx={rung}: published gauge fails"
        assert "G5_budget_drag" in row, f"H/dx={rung}: momentum-budget gauge missing"
        assert row["G5_budget_drag"]["pass_5pct"], f"H/dx={rung}: budget drag off by >5 %"
        # the budget is a different physical route (d<u>/dt = g - D/M), not a fit
        assert abs(row["G5_budget_drag"]["error_pct"]) <= TOL_PCT


def test_uncorrected_stencils_are_reported_as_failing(verdict):
    """Honest bookkeeping: the raw near-wall stencils are outside the band.

    If someone later "fixes" the verdict by dropping these gauges, this test fails.
    """
    c = _gate(verdict, "C_wall_shear_battery")
    assert "G2_secant_row_pair" in c["gauges_failing_uncorrected"]
    assert "G3_three_point_quad" in c["gauges_failing_uncorrected"]
    per = c["per_ratio"]
    for rung in ("8", "12", "16"):
        assert abs(per[rung]["G2_secant_row_pair"]["error_pct"]) > TOL_PCT
        # ... and the deficit is the stencil, not the flow: corrected it is inside
        assert abs(per[rung]["G2_secant_row_pair"]["error_pct_bias_corrected"]) <= 1.0


def test_strictest_local_gauge_sets_the_resolution_requirement(verdict):
    """The bias-corrected wall-plane secant needs H/dx >= 16 to be honest.

    Raw G1 passes at every rung, but only because its stencil bias
    (1 - dx/2H) partly cancels the numerical wall layer.  Corrected, the coarse
    rungs overshoot by 9.3 % (H/dx=8) and 6.1 % (H/dx=12).
    """
    per = _gate(verdict, "C_wall_shear_battery")["per_ratio"]
    assert not per["8"]["G1_secant_wall_plane"]["pass_5pct_bias_corrected"]
    assert not per["12"]["G1_secant_wall_plane"]["pass_5pct_bias_corrected"]
    assert per["16"]["G1_secant_wall_plane"]["pass_5pct_bias_corrected"]
    assert _gate(verdict, "C_wall_shear_battery")["gauges_failing_bias_corrected"] == [
        "G1_secant_wall_plane"]


def test_negative_control_fails_the_no_slip_gates(verdict):
    """A gauge that also passes the free-slip channel measures nothing."""
    ctrl = verdict["negative_control"]
    assert ctrl["fails_centreline_gate"] and ctrl["fails_wall_shear_gate"]
    assert _gate(verdict, "E_negative_control")["status"] == "PASS"


def test_overall_status_is_honest(verdict):
    """PASS_WITH_CAVEAT, not PASS, while a gauge is outside the band."""
    assert verdict["overall"] == "PASS_WITH_CAVEAT"
    assert verdict["counts"]["fail"] == 0
    assert verdict["counts"]["pass_with_caveat"] == 1

def test_budget_gauge_is_calibrated_on_the_free_slip_control(verdict):
    """G5 reads d<u>/dt, so the reading itself is calibrated on a known answer.

    Free slip has D = 0 and hence d<u>/dt = g exactly; the measured bias of that
    reading (here 1.0e-6) is removed from the channel reading before it is graded.
    """
    ctrl = verdict["negative_control"]
    assert ctrl["drift_bias"] is not None
    assert abs(ctrl["drift_bias"]) < 1e-2, "free-slip calibration drifted by more than 1 %"
    per = _gate(verdict, "C_wall_shear_battery")["per_ratio"]
    for rung, row in per.items():
        corr = row["G5_budget_drag"]["error_pct_control_corrected"]
        assert corr is not None
        assert abs(corr) <= TOL_PCT, f"H/dx={rung}: calibrated budget drag off by {corr:.3f} %"
