"""Tests for the CASE-POISEUILLE-SPH uncertainty/validation layer.

Two distinct jobs, and they are not interchangeable:

1. **Machinery** -- :mod:`tensorsph.verification.uncertainty` must recover a
   prescribed order from synthetic data whose order is known exactly, and it must
   *refuse* to produce an order when the data cannot support one (sign change,
   stall, disagreeing pair orders).  A verification tool that always returns a
   number is worse than no tool: it certifies noise.
2. **Published verdict** -- the committed ``out/uncertainty.json`` must be
   reproducible from the raw ladder artifact, must be bound to that artifact by
   hash, and must not over-certify: no order claim where the pair orders
   disagree, no Richardson extrapolation there either, and the scale of every
   percentage pinned to the declared acceptance scale (relative for amplitudes,
   quarter turn for phases).

Run:  pytest tests/test_case_poiseuille_uncertainty.py -v
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_poiseuille"))

from tensorsph.verification import uncertainty as U  # noqa: E402
import uncertainty as case_unc  # noqa: E402  (the driver)

RESULT = REPO / "benchmarks" / "case_poiseuille" / "out" / "result.json"
UNC = REPO / "benchmarks" / "case_poiseuille" / "out" / "uncertainty.json"

pytestmark = pytest.mark.skipif(not RESULT.is_file(), reason="ladder artifact absent")


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def payload(result: dict) -> dict:
    return case_unc.analyse(result)


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads(UNC.read_text())


# --------------------------------------------------------------------------- #
# 1. machinery: synthetic orders
# --------------------------------------------------------------------------- #
def _ladder(truth: float, order: float, spacings, coeff: float = 1.0) -> list[U.Rung]:
    return [U.Rung(h, truth + coeff * h ** order, f"h={h:g}") for h in spacings]


@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_prescribed_order_is_recovered(order: float) -> None:
    """A clean ``h**p`` error must be identified as ``p`` with equal ratios."""
    spacings = [1 / 16, 1 / 8, 1 / 4]          # r = 2 between every pair
    rungs = _ladder(1.0, order, spacings)
    rep = U.order_from_errors(rungs, 1.0)
    assert rep["status"] == "converging", rep
    assert rep["p_best"] == pytest.approx(order, abs=1e-9)
    assert rep["monotone_magnitude"] is True
    # Richardson on the two finest must return the truth for a pure power law.
    ex = U.richardson_extrapolate(rungs[0], rungs[1], rep["p_best"])
    assert ex == pytest.approx(1.0, abs=1e-12)


def test_sign_change_is_oscillatory_and_refuses_an_order() -> None:
    rungs = [U.Rung(1 / 16, 1.0 + 0.01, "f"), U.Rung(1 / 8, 1.0 - 0.02, "m"),
             U.Rung(1 / 4, 1.0 + 0.04, "c")]
    rep = U.order_from_errors(rungs, 1.0)
    assert rep["status"] == "oscillatory"
    assert rep["p_best"] is None
    assert all(p["p"] is None for p in rep["pair_orders"])


def test_stalled_and_diverging_are_distinguished() -> None:
    stalled = [U.Rung(1 / 16, 1.10, "f"), U.Rung(1 / 8, 1.01, "m"),
               U.Rung(1 / 4, 1.02, "c")]        # |e| 0.10, 0.01, 0.02
    rep = U.order_from_errors(stalled, 1.0)
    assert rep["status"] == "stalled"

    diverging = [U.Rung(h, 1.0 + 0.3 / h * 1e-2, "x") for h in (1 / 16, 1 / 8, 1 / 4)]
    rep2 = U.order_from_errors(diverging, 1.0)
    assert rep2["status"] == "diverging"


def test_unequal_ratios_can_leave_the_order_unresolved() -> None:
    """The real 8/12/16 geometry: two unequal ratios, one clean power law.

    Even for an exact ``h**2`` error the two pair orders differ, because each
    pair samples a different interval; the status says so instead of pretending
    to a single order.
    """
    truth, p = 0.5, 2.0
    spacings = [1 / 16, 1 / 12, 1 / 8]
    rungs = _ladder(truth, p, spacings, coeff=20.0)   # coefficient changes nothing
    rep = U.order_from_errors(rungs, truth)
    assert rep["status"] in {"converging", "order_unresolved"}
    if rep["status"] == "order_unresolved":
        assert rep["order_spread"] > U.ORDER_AGREEMENT
        assert rep["p_best"] == pytest.approx(p, abs=0.5)


# --------------------------------------------------------------------------- #
# 2. machinery: GCI, scale, validation
# --------------------------------------------------------------------------- #
def test_gci_matches_hand_computation() -> None:
    fine, coarse = U.Rung(0.5, 1.00, "f"), U.Rung(1.0, 1.10, "c")   # r = 2
    g = U.gci_percent(fine, coarse, order=2.0, safety=1.25)
    # GCI = 1.25 * 0.10 / (2**2 - 1) = 0.041666...
    assert g["gci_percent"] == pytest.approx(100.0 * 1.25 * 0.10 / 3.0)
    assert g["r"] == pytest.approx(2.0)
    assert g["order_convention"] == "observed p"
    # no order -> conservative p=1, and it must be *wider* than the order-2 band
    g1 = U.gci_percent(fine, coarse, order=None, safety=1.25)
    assert g1["gci_percent"] > g["gci_percent"]
    assert g1["order_convention"] == "conservative p=1"


def test_gci_scale_is_the_declared_denominator() -> None:
    fine, coarse = U.Rung(0.5, 0.60, "f"), U.Rung(1.0, 0.66, "c")
    rel = U.gci_percent(fine, coarse, order=2.0)
    declared = U.gci_percent(fine, coarse, order=2.0, scale=1.0)
    assert rel["scale_basis"] == "|fine value|"
    assert declared["scale_basis"] == "declared"
    # |0.60| is the default denominator, so the declared-scale band is larger/lower
    # exactly by that factor -- no silent rescaling.
    assert declared["gci_percent"] == pytest.approx(rel["gci_percent"] * 0.60)


def test_validation_verdicts_and_tolerance_support() -> None:
    ok = U.validation(1.002, 1.0, u_num_percent=0.5, tolerance_percent=5.0)
    assert ok["E_percent"] == pytest.approx(0.2)
    assert ok["verdict"] == "validated"
    assert ok["within_tolerance"] is True
    assert ok["supports_tolerance"] is True

    over = U.validation(1.10, 1.0, u_num_percent=0.5, tolerance_percent=5.0)
    assert over["verdict"] == "not_established"
    assert over["within_tolerance"] is False

    # A lucky point error inside the tolerance whose band is *wider* than the
    # tolerance: passes the tolerance, cannot certify it.
    wide = U.validation(1.001, 1.0, u_num_percent=8.0, tolerance_percent=5.0)
    assert wide["within_tolerance"] is True
    assert wide["supports_tolerance"] is False


def test_repeatability_reports_half_range() -> None:
    rep = U.repeatability([1.0, 1.02])
    assert rep["half_range"] == pytest.approx(0.01)
    assert rep["half_range_percent_of_mean"] == pytest.approx(100 * 0.01 / 1.01)


# --------------------------------------------------------------------------- #
# 3. machinery: reference alignment
# --------------------------------------------------------------------------- #
def test_per_rung_truth_travels_with_the_rung() -> None:
    rungs = [U.Rung(1 / 8, 0.8, "coarse"), U.Rung(1 / 16, 0.9, "fine"),
             U.Rung(1 / 12, 0.85, "medium")]
    truth = [0.81, 0.91, 0.86]        # aligned with the *input* order, not sorted
    rs, ts = U.align_truth(rungs, truth)
    assert [r.label for r in rs] == ["fine", "medium", "coarse"]
    assert ts == [0.91, 0.86, 0.81]


def test_mismatched_truth_length_is_rejected() -> None:
    rungs = [U.Rung(1 / 8, 1.0), U.Rung(1 / 16, 1.0)]
    with pytest.raises(ValueError):
        U.align_truth(rungs, [1.0])


def test_a_degenerate_refinement_ratio_is_rejected() -> None:
    """``pair_order`` must refuse ``r <= 1`` rather than return a negative order."""
    with pytest.raises(ValueError):
        U.pair_order(0.01, 0.02, r=1.0)
    # and a mis-ordered ladder is sorted by the alignment helper, not silently
    # used with a reversed ratio
    coarse, fine = U.Rung(1 / 4, 1.0, "c"), U.Rung(1 / 16, 1.0, "f")
    rs, _ = U.align_truth([coarse, fine], 0.9)
    assert [r.label for r in rs] == ["f", "c"]


# --------------------------------------------------------------------------- #
# 4. the published verdict
# --------------------------------------------------------------------------- #
def test_published_verdict_is_bound_to_the_ladder_artifact(published: dict) -> None:
    assert published["source_sha256"] == hashlib.sha256(RESULT.read_bytes()).hexdigest()


def test_published_verdict_reproduces_from_the_artifact(payload: dict,
                                                       published: dict) -> None:
    """Re-running the analysis must give the committed numbers, to round-off."""
    for key, pub in published["quantities"].items():
        got = payload["quantities"][key]
        assert got["order_analysis"]["status"] == pub["order_analysis"]["status"]
        assert got["validation"]["E_percent"] == pytest.approx(
            pub["validation"]["E_percent"], rel=1e-9)
        assert got["validation"]["U_V_percent"] == pytest.approx(
            pub["validation"]["U_V_percent"], rel=1e-9)
        assert got["error_percent_per_rung"] == pytest.approx(
            pub["error_percent_per_rung"], rel=1e-12)
    assert payload["gates"] == published["gates"]
    # honest FAIL: the R1 steady band does not support the 5 % acceptance
    # (U3b), while every graded R2 band and every R1 point/rung check passes
    assert payload["all_gates_pass"] is False
    assert payload["gates"]["U3b_R1_band_supports_5pct"] is False


def test_reference_matches_the_library_closed_form(payload: dict) -> None:
    """The analytic reference used by the case *is* ``tensorsph.womersley``."""
    ref = payload["reference_consistency"]
    assert ref["pass"] is True
    assert ref["max_relative_deviation"] < 1e-9


def test_scale_of_every_percentage_is_the_declared_one(published: dict) -> None:
    """Phase percentages are of a quarter turn; amplitude ones are relative."""
    quarter = math.pi / 2
    for key, q in published["quantities"].items():
        if key.startswith("phase"):
            assert q["scale"] == pytest.approx(1.0)
            assert "quarter turn" in q["scale_text"]
        else:
            assert q["scale"] is None
    # cross-check phase_centre against the raw radians in the ladder artifact
    raw = json.loads(RESULT.read_text())["harmonic_runs"]
    finest = max(raw, key=lambda r: r["resolution_H_over_dx"])
    expected = 100 * abs(finest["phase_centre_rad"]
                         - finest["phase_centre_analytic_rad"]) / quarter
    assert published["quantities"]["phase_centre"]["validation"]["E_percent"] == (
        pytest.approx(expected, rel=1e-9))


def test_errors_follow_the_report_order(published: dict) -> None:
    """The finest rung is the first row: mis-sorting here labels the wrong rung."""
    for key, q in published["quantities"].items():
        spacings = q["spacings"]
        assert spacings == sorted(spacings), f"{key}: not fine -> coarse"
        recomputed = [
            100.0 * abs(v - t) / (abs(q["scale"]) if q["scale"] else abs(t))
            for v, t in zip(q["values"], q["truth_per_rung"])
        ]
        assert q["error_percent_per_rung"] == pytest.approx(recomputed, rel=1e-12)
        # entry 0 is the *finest* rung, i.e. exactly the error the validation
        # block compares against the reference
        assert q["error_percent_per_rung"][0] == pytest.approx(
            q["validation"]["E_percent"], rel=1e-12)


def test_profile_reference_is_resolution_independent(published: dict) -> None:
    """The norm functionals must have one fixed reference; the point ones cannot.

    That asymmetry is what makes a GCI legitimate for the norm functionals and
    is also why the row centres (and hence the point references) move with dx.
    """
    for key in ("amp_profile_l2", "phase_profile_l2"):
        truths = published["quantities"][key]["truth_per_rung"]
        assert max(truths) == pytest.approx(min(truths), rel=1e-12)
    for key in ("amp_centre", "phase_centre"):
        truths = published["quantities"][key]["truth_per_rung"]
        assert max(truths) != pytest.approx(min(truths), rel=1e-12)


def test_no_order_is_claimed_where_the_pairs_disagree(published: dict) -> None:
    """Over-certification guard: unresolved/oscillatory ladders get no Richardson."""
    for key, q in published["quantities"].items():
        status = q["order_analysis"]["status"]
        if status == "converging":
            continue
        assert q["p_identified"] is False, key
        assert q["richardson"] is None, key
        assert "no order of accuracy claimed" in q["order_claim"]
    # the phase functional in this case really is oscillatory -- that is the
    # honest finding, not a regression
    assert published["quantities"]["phase_centre"]["order_analysis"]["status"] == "oscillatory"


def test_every_graded_functional_is_inside_the_5pct_acceptance(published: dict) -> None:
    for key, q in published["quantities"].items():
        assert q["worst_rung_error_percent"] <= published["tolerance_percent"], key
        assert q["error_percent_per_rung"][0] <= 0.5, key   # finest rung, < 0.5 %
        assert q["validation"]["verdict"] == "validated", key
        assert q["validation"]["supports_tolerance"] is True, key


def test_r1_anchor_is_graded_on_a_ladder(published: dict) -> None:
    """R1 used to be a single rung ("status: insufficient"); it is a ladder now.

    The anchor rung is still H/dx = 12 and still has to be inside the tolerance,
    but the status must no longer claim that no order/band can be estimated.
    """
    single = published["single_rung"]
    assert single["status"] == "ladder"
    assert "single resolution" not in single["reason"]
    assert single["ladder_resolutions_H_over_dx"] == [8, 12, 16]
    assert single["steady_profile_rms"]["within_tolerance"] is True
    assert single["startup_profile_rms_at_0.40s"]["within_tolerance"] is True
    assert published["r1_quantities"], "R1 ladder must be published"


def test_legacy_artifact_without_ladder_is_still_insufficient() -> None:
    """The honest fallback must survive: no ladder -> no order claim, no pass."""
    legacy = {"constant_run": {"resolution_H_over_dx": 12}}
    assert case_unc.r1_quantities(legacy) == {}
    assert case_unc.r1_input_uncertainty(legacy) == {}


def test_gci_band_is_never_narrower_than_the_conservative_band(published: dict) -> None:
    for key, q in published["quantities"].items():
        g = q["gci_fine_pairs"]
        cons = g["conservative"]["gci_percent"]
        obs = g["observed_order"]["gci_percent"] if g["observed_order"] else None
        assert q["U_num_percent"] >= cons - 1e-12, key
        if obs is not None:
            assert q["U_num_percent"] >= obs - 1e-12, key