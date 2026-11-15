"""Regression gates for the independent Poiseuille refinement ladder.

Two layers, deliberately separated:

* **pure logic** -- the convergence estimator, the power-law fit and the
  resolution extrapolation are graded on synthetic data with a known order, so
  a broken estimator cannot silently certify a real ladder.
* **artifact invariants** -- on `benchmarks/poiseuille_mirror/out/refinement.json`
  (produced by `refinement_study.py`), every summary number must be
  re-derivable from the stored per-rung readings, and the same-session control
  rung must reproduce the committed ladder at
  `out/results.json`.  These assert *bookkeeping*, not the outcome: an honest
  FAIL of a 5 % band must still be recorded faithfully, so the test checks that
  the recorded verdict matches the recorded numbers instead of demanding a pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.poiseuille_mirror.refinement_study import (
    READINGS,
    analyse,
    fit_power_law,
    flatten,
    observed_orders,
    resolution_for,
    verdict,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "benchmarks/poiseuille_mirror/out/refinement.json"


# --------------------------------------------------------------------------- #
# pure logic
# --------------------------------------------------------------------------- #

def test_observed_order_recovers_known_exponent():
    rs = [8, 16, 32, 64]
    es = [1.0 * (r / 8.0) ** -2.0 for r in rs]
    orders = observed_orders(rs, es)
    assert orders[0] == pytest.approx(2.0, abs=1e-12)
    assert all(o == pytest.approx(2.0, abs=1e-12) for o in orders)


def test_observed_order_flags_a_plateau():
    """A constant error (no decay) must read as order 0, not as a small pass."""
    assert observed_orders([8, 16], [3.0, 3.0]) == [pytest.approx(0.0, abs=1e-12)]


def test_fit_power_law_and_extrapolation_round_trip():
    rs = [16, 24, 32, 48]
    model = fit_power_law(rs, [10.0 * (r / 16.0) ** -1.5 for r in rs])
    assert model["p"] == pytest.approx(1.5, rel=1e-10)
    r5 = resolution_for(model, 5.0)
    # C r^-p = 5 with C = 10 * 16^1.5
    assert r5 == pytest.approx((10.0 * 16.0 ** 1.5 / 5.0) ** (1 / 1.5), rel=1e-10)


def test_resolution_for_refuses_a_non_decaying_series():
    model = fit_power_law([16, 32], [3.0, 3.0])
    assert model["p"] == pytest.approx(0.0, abs=1e-12)
    assert resolution_for(model, 5.0) is None


def test_flatten_reading_keys_are_all_analysed():
    """Every reading the analysis fits must exist in the flattened rung."""
    synthetic = {
        "resolution_H_over_dx": 8,
        "profile": {"u_max_error_pct": 0.1, "rms_interior_rel_pct": 0.2,
                    "rms_all_rel_pct": 0.3, "rms_umax_pct": 0.4,
                    "max_row_rel_pct": 0.5},
        "wall": {"nu_eff_rel_pct": 1.0,
                 "gauges": {name: {"error_pct": 2.0, "error_pct_bias_corrected": 1.0}
                            for name in ("G1_secant_wall_plane",
                                         "G2_secant_row_pair",
                                         "G3_three_point_quad",
                                         "G4_global_fit_slope")}},
        "slip_over_umax": 0.01,
    }
    flat = flatten(synthetic)
    assert set(READINGS).issubset(flat)


# --------------------------------------------------------------------------- #
# artifact invariants
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT.exists():
        pytest.skip(f"refinement artifact not present: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


def test_artifact_covers_past_the_published_ladder(artifact):
    rs = artifact["analysis"]["resolutions"]
    assert rs == sorted(rs)
    assert len(rs) >= 3
    # the published ladder stops at 16; this study must extend it
    assert max(rs) > 16, f"ladder did not extend past H/dx=16: {rs}"
    # the control rung must be the published top rung, so the two ladders join
    assert 16 in rs


def test_summary_numbers_are_rederivable_from_rungs(artifact):
    """No hand-typed summary: recompute the analysis from the stored rungs."""
    stored = artifact["analysis"]["readings"]
    rungs = artifact["rungs"]
    flat = [flatten(r) for r in rungs]
    for key in READINGS:
        assert stored[key]["values"] == pytest.approx([f[key] for f in flat]), key


def test_gate_flags_agree_with_the_recorded_numbers(artifact):
    for key, entry in artifact["analysis"]["readings"].items():
        finite = [abs(v) for v in entry["values"] if v is not None]
        assert entry["pass_5pct_all_rungs"] is all(v <= 5.0 for v in finite), key
        assert entry["pass_3pct_all_rungs"] is all(v <= 3.0 for v in finite), key


def test_same_session_control_reproduces_the_committed_rung(artifact):
    repro = artifact["reproducibility_vs_committed_ladder"]
    if not repro.get("available"):
        pytest.skip(repro.get("reason", "committed ladder unavailable"))
    assert repro["checks"], "no overlapping resolution with the committed ladder"
    for row in repro["checks"]:
        for key in ("u_max_error_pct", "rms_interior_rel_pct", "rms_all_rel_pct"):
            entry = row[key]
            assert entry["abs_diff_pct_points"] <= 0.5, (row["resolution_H_over_dx"], key,
                                                         entry)


def test_verdict_is_derived_not_asserted(artifact):
    """Recompute the verdict block and require the stored one to match."""
    an = analyse(artifact["rungs"])
    assert verdict(an)["questions"] == artifact["verdict"]["questions"]


def test_wall_layer_question_is_answered_with_numbers(artifact):
    q1 = artifact["verdict"]["questions"]["Q1_wall_layer_keeps_decaying_above_16"]
    assert isinstance(q1["decays"], bool)
    assert q1["observed_order"] is None or math.isfinite(q1["observed_order"])
    # Q2/Q3 must state either a resolution or an explicit non-answer
    for key in ("Q2_resolution_for_5pct_wall_layer", "Q3_resolution_for_3pct_wall_layer"):
        q = artifact["verdict"]["questions"][key]
        assert "H_over_dx" in q
        if q["H_over_dx"] is not None:
            assert q["H_over_dx"] > 0