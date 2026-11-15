"""Gates for the consistent-operator study of `CASE-SPH-LAKE-AT-REST`.

`docs/verification_lake_at_rest_wall_gap.md` left one open item: *a
well-balanced/consistent momentum operator is required* to close the closed-tank
rest gap.  `docs/verification_lake_at_rest_wb.md` decides that item with
measurements.  These gates pin the decision so it cannot rot:

1. the re-implementation reproduces the shipped operator's numbers exactly
   (otherwise every comparison below is about the wrong operator);
2. **no** consistent-gradient variant reaches the anchor tolerance, and the best
   one is still within a factor 2 of the shipped residual (i.e. the cure is not
   "almost there");
3. the variants that cannot even hold the *vertical* balance are reported as such
   rather than silently dropped;
4. the measurement is sensitive to its own inputs (zero pressure -> |a|/g = 1,
   a 1 % pressure error moves the residual), so "rest" is not manufactured;
5. the cancellation diagnostic is decided: cancelling the residual would need an
   own-pressure of the wrong sign and several times the hydrostatic value, so no
   admissible pressure closure can do it.

Numbers come from `tests/data/lake_at_rest/wb_operator.json`, written by
`benchmarks/lake_at_rest_wb/run.py`; nothing is hand-copied.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).parent / "data" / "lake_at_rest" / "wb_operator.json"
ANCHOR_TOLERANCE = 1.0e-9


@pytest.fixture(scope="module")
def evidence() -> dict:
    with open(EVIDENCE) as handle:
        return json.load(handle)


def test_evidence_channel_is_the_closed_tank(evidence):
    assert evidence["case"] == "CASE-SPH-LAKE-AT-REST"
    assert evidence["geometry"]["periodic_x"] is False
    assert evidence["anchor_tolerance"] == ANCHOR_TOLERANCE
    assert [row["dx"] for row in evidence["ladder"]] == [0.050, 0.040]
    # the third resolution carries the cancellation diagnostic only (the mass
    # solve is not needed for it, and its ladder run was left in the cluster queue)
    assert "0.025" in evidence["cancellation_extra"]


def test_reimplementation_reproduces_the_shipped_residual(evidence):
    """The baseline variant *is* the shipped operator: same numbers, same solve."""
    for row in evidence["ladder"]:
        base = next(v for v in row["variants"]
                    if v["name"] == "shipped_reimplementation")
        shipped = row["shipped"]["max_ax_over_g"]
        at_mass = base["at_shipped_mass"]["max_ax_over_g"]
        assert at_mass == pytest.approx(shipped, rel=1e-9), row["dx"]
        assert base["retuned"]["max_ay_over_g"] < ANCHOR_TOLERANCE, row["dx"]


def test_no_variant_reaches_the_anchor_tolerance(evidence):
    """The open item is decided: the gap survives every consistent gradient."""
    for row in evidence["ladder"]:
        for variant in row["variants"]:
            assert variant["retuned"]["max_a_over_g"] > ANCHOR_TOLERANCE, (
                row["dx"], variant["name"])


def test_the_best_variant_is_not_almost_good_enough(evidence):
    """A 15 % gain is not "converging to the tolerance" -- pin the factor."""
    for row in evidence["ladder"]:
        shipped = row["shipped"]["max_ax_over_g"]
        best = min(v["retuned"]["max_ax_over_g"] for v in row["variants"])
        assert best > 0.5 * shipped, row["dx"]
        # ... and the one variant that does improve the horizontal number is the
        # own-pressure reconstruction, by less than 3x at every resolution
        own = next(v for v in row["variants"]
                   if v["name"] == "own_pressure_reconstructed")
        assert own["retuned"]["max_ax_over_g"] > shipped / 3.0, row["dx"]


def test_variants_that_lose_the_vertical_balance_are_reported_not_dropped(evidence):
    """`difference_consistent` and `lsq_plane_gradient` cannot hold a_y: say so."""
    for row in evidence["ladder"]:
        bad = [v["name"] for v in row["variants"]
               if v["retuned"]["max_ay_over_g"] > 1.0e-2]
        assert sorted(bad) == ["difference_consistent", "lsq_plane_gradient"], row["dx"]
        for variant in row["variants"]:
            if variant["name"] in bad:
                assert not variant["retuned"]["converged"], (row["dx"], variant["name"])


def test_operator_responds_to_its_own_inputs(evidence):
    """Controls: a rest state that ignores the pressure data would fail these."""
    for row in evidence["ladder"]:
        controls = row["controls"]
        assert controls["zero_pressure_max_a_over_g"] == pytest.approx(1.0, rel=1e-9)
        assert controls["pressure_scaled_1pct_max_ax_over_g"] > 1.0e-3
        assert controls["gravity_off_max_a_over_g"] > 1.0e-3


def test_cancellation_needs_an_inadmissible_pressure(evidence):
    """No own-pressure value can cancel it: the demand is negative (tensile)."""
    for row in evidence["ladder"]:
        cancel = row["cancellation"]
        assert cancel["worst_p_eos"] == 0.0, row["dx"]
        assert cancel["worst_p_physical"] > 0.0, row["dx"]
        assert cancel["worst_p_required"] < 0.0, row["dx"]
        assert abs(cancel["worst_required_over_physical"]) > 2.0, row["dx"]
        x = cancel["worst_position"][0]
        assert min(x, evidence["geometry"]["width"] - x) < 2.5 * row["h"], (
            "the worst particle must be in a wall column", row["dx"])


def test_cancellation_extends_to_the_third_resolution(evidence):
    extra = evidence["cancellation_extra"]["0.025"]
    assert extra["worst_p_eos"] == 0.0
    assert extra["worst_p_required"] < 0.0
    assert abs(extra["worst_required_over_physical"]) > 2.0
    assert extra["wall_band_residual_ax_max_over_g"] > 1.0e-2


def test_summary_is_honest(evidence):
    summary = evidence["summary"]
    assert summary["verdict"] == ("no consistent-gradient variant reaches the "
                                 "anchor tolerance")
    assert summary["best_over_tolerance"] > 1.0e6