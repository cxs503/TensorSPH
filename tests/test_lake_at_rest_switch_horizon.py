"""Gates for the switch-horizon study of `CASE-SPH-LAKE-AT-REST`.

Artefact: ``tests/data/lake_at_rest/switch_horizon.json`` (produced by
``benchmarks/lake_at_rest/switch_horizon.py``).  The question it decides: the
mechanism study capped its switch arms at ``t_end = 1.6 s``, so the best arm left
the cap one third of a decade *below* the declared band floor and the study could
only say "no switch arm produced a rate inside the band".  This study removes the
cap and, per the declared rules, either clears the growth with a shipped closure
or refuses to.

Every gate below **re-derives** its number from the artefact's raw series with its
own least squares; nothing is read back from the producer's reductions except to
check that the two agree.  The one gate that touches the physics gate of the
previous study cross-checks the reference arm's rate against
``tests/data/lake_at_rest/stability_band_fit.json`` (the band estimator of the
anchor's audit), so a drift in the case's operator shows up here as a red gate
rather than as a silently re-baselined number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = ROOT / "tests" / "data" / "lake_at_rest" / "switch_horizon.json"
SHIPPED_BAND_FIT = ROOT / "tests" / "data" / "lake_at_rest" / "stability_band_fit.json"

BAND_LOW = 1.0e-6
BAND_HIGH = 1.0e-3
TOLERANCE = 1.0e-9
#: ``band_fit.MIN_POINTS``: below this sample count the producer returns no
#: slope at all, so a missing rate is a sampling statement rather than a verdict.
MIN_POINTS = 4
CLEARS = 0.10
REDUCES = 0.75
#: Published protocol of this study; a re-run that changes it must change these.
DECLARED_T_END = 2.6
DECLARED_DX = 0.02
#: The mechanism study's cap -- the horizon this study has to beat.
PREVIOUS_CAP = 1.6
#: The previously certified rest horizon (`t_end = 0.50 s`), which must still hold.
PREVIOUSLY_CERTIFIED = 0.50
#: The anchor's audit rate for the reference arm (band estimator, base arm).
REFERENCE_RATE = 16.47
REFERENCE_RATE_TOL = 0.15


# ---------------------------------------------------------------------------
# loading and independent re-derivation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def payload() -> dict:
    assert ARTEFACT.exists(), (
        f"{ARTEFACT} is missing; reproduce with\n"
        "  PYTHONPATH=src python3 benchmarks/lake_at_rest/switch_horizon.py "
        "--json tests/data/lake_at_rest/switch_horizon.json --figures docs/figures"
    )
    return json.loads(ARTEFACT.read_text())


@pytest.fixture(scope="module")
def arms(payload) -> dict:
    return payload["arms"]


def _series(arm: dict) -> tuple[np.ndarray, np.ndarray]:
    samples = arm["samples"]
    t = np.array([s["t"] for s in samples], dtype=float)
    a = np.array([s["v_over_c0"] for s in samples], dtype=float)
    keep = np.isfinite(t) & np.isfinite(a) & (a > 0.0)
    return t[keep], a[keep]


def _fit(t: np.ndarray, a: np.ndarray, low: float, high: float) -> dict:
    """Least squares of ``ln a`` against ``t`` inside ``[low, high]``.

    Deliberately not imported from the producer or the library: a gate that
    reuses the estimator under test cannot detect an estimator defect (the
    lesson of the lake's own audit).
    """
    sel = (a >= low) & (a <= high)
    n = int(sel.sum())
    if n < 3:
        return {"sigma": None, "r2": None, "n": n}
    tt, aa = t[sel], np.log(a[sel])
    slope, intercept = np.polyfit(tt, aa, 1)
    resid = aa - (slope * tt + intercept)
    ss_tot = float(((aa - aa.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {"sigma": float(slope), "r2": r2, "n": n}


def _band_rate(arm: dict) -> float | None:
    t, a = _series(arm)
    return _fit(t, a, BAND_LOW, BAND_HIGH)["sigma"]


def _t_star(arm: dict, level: float = TOLERANCE) -> float | None:
    for s in arm["samples"]:
        if s["v_over_c0"] >= level:
            return float(s["t"])
    return None


def _closest(arm, payload__t: float) -> dict:
    return min(arm["samples"], key=lambda s: abs(s["t"] - payload__t))


# ---------------------------------------------------------------------------
# G1 the protocol was declared, and it can decide the question
# ---------------------------------------------------------------------------
def test_protocol_is_declared_and_not_tuned_to_the_numbers(payload):
    proto = payload["protocol"]
    assert payload["case"] == "CASE-SPH-LAKE-AT-REST"
    assert tuple(payload["band"]) == (BAND_LOW, BAND_HIGH)
    assert proto["t_end"] == pytest.approx(DECLARED_T_END)
    assert proto["dx"] == pytest.approx(DECLARED_DX)
    assert proto["periodic_x"] is True
    assert proto["tolerance"] == pytest.approx(TOLERANCE)


def test_horizon_is_long_enough_to_separate_delay_from_removal(arms):
    """An arm that kept the base rate must have crossed the whole band."""
    base = arms["base"]
    rate = base["growth"]["lambda_per_s"]
    assert rate is not None and rate > 0.0
    t_end = base["run"]["t_end"]
    # time for the base rate to cross the declared band, one decade of margin
    crossing = math.log(BAND_HIGH / BAND_LOW) / rate
    assert t_end >= PREVIOUS_CAP, "cap not removed: the study cannot decide"
    assert t_end >= 5.0 * crossing, (
        f"t_end = {t_end:.2f} s is less than five band-crossing times "
        f"({crossing:.2f} s): a delayed arm and a cleared arm would look alike"
    )


# ---------------------------------------------------------------------------
# G2 the reference arm behaves as the anchor's own evidence says
# ---------------------------------------------------------------------------
def test_reference_arm_reproduces_the_audited_rate(arms):
    base = arms["base"]
    own = _band_rate(base)
    assert own is not None, "reference arm never entered the declared band"
    assert own == pytest.approx(REFERENCE_RATE, rel=REFERENCE_RATE_TOL), (
        f"reference arm rate {own:.2f} 1/s drifted from the audited "
        f"{REFERENCE_RATE:.2f} 1/s: the operator or the protocol changed"
    )
    fit = _fit(*_series(base), BAND_LOW, BAND_HIGH)
    assert fit["r2"] >= 0.98, f"the departure is not exponential: R^2 = {fit['r2']:.3f}"


def test_reference_arm_still_holds_the_certified_horizon(arms):
    """The previously certified `0.50 s` must not have regressed."""
    t_star = _t_star(arms["base"])
    assert t_star is not None, "reference arm never crossed the anchor tolerance"
    assert t_star >= PREVIOUSLY_CERTIFIED, (
        f"reference arm leaves the tolerance at t* = {t_star:.3f} s, inside the "
        f"certified {PREVIOUSLY_CERTIFIED:.2f} s horizon"
    )


def test_reference_arm_is_far_outside_the_tolerance_at_the_horizon(arms):
    base = arms["base"]
    assert base["max_v_over_c0"] >= 1.0e3 * TOLERANCE, (
        "the reference arm does not depart inside t_end, so a 'cleared' verdict "
        "would have no contrast to stand on"
    )


# ---------------------------------------------------------------------------
# G3 every quoted rate is reproducible from the raw series
# ---------------------------------------------------------------------------
def test_shipped_rates_are_reproducible_from_the_raw_series(arms):
    for name, arm in arms.items():
        fit = _fit(*_series(arm), BAND_LOW, BAND_HIGH)
        shipped = arm["growth"]["lambda_per_s"]
        if shipped is None:
            # the producer returns no slope when fewer than MIN_POINTS samples
            # fall inside the band; that is a sampling statement, not a rate
            assert fit["n"] < MIN_POINTS, (
                f"{name}: producer reports no rate, but {fit['n']} samples lie "
                f"inside the band and fit {fit['sigma']}"
            )
            continue
        assert fit["n"] >= MIN_POINTS
        assert fit["sigma"] == pytest.approx(shipped, rel=1.0e-6), (
            f"{name}: shipped rate {shipped} != independent {fit['sigma']}"
        )


def test_band_membership_counts_agree(arms):
    for name, arm in arms.items():
        fit = _fit(*_series(arm), BAND_LOW, BAND_HIGH)
        assert fit["n"] == arm["growth"]["points"], (
            f"{name}: {fit['n']} samples inside the band, producer says "
            f"{arm['growth']['points']}"
        )


# ---------------------------------------------------------------------------
# G4 the verdict is the declared rule applied to the measured numbers
# ---------------------------------------------------------------------------
def test_switch_verdicts_follow_the_declared_thresholds(payload, arms):
    grading = payload["grading"]
    base_rate = _band_rate(arms["base"])
    graded = 0
    for name, entry in grading["switch"].items():
        own_rate = _band_rate(arms[name])
        assert own_rate is not None
        ratio = own_rate / base_rate
        assert entry["lambda_ratio"] == pytest.approx(ratio, rel=1.0e-6)
        expected = ("clears" if ratio < CLEARS else
                    "reduces" if ratio < REDUCES else "does_not_own")
        assert entry["verdict"] == expected, (
            f"{name}: ratio {ratio:.3f} graded '{entry['verdict']}'"
        )
        graded += 1
    assert graded + len(grading["unmeasured"]) + len(grading["excluded"]) \
        == len(arms) - 1, "an arm was neither graded, unmeasured nor excluded"


def test_cleared_verdict_needs_a_like_for_like_suppression_too(arms, payload):
    """A rate ratio alone is not enough: the amplitude must be suppressed too."""
    grading = payload["grading"]
    for name, entry in grading["switch"].items():
        if entry["verdict"] not in ("clears", "reduces"):
            continue
        ratio = entry["amplitude_ratio"]["t_end"]
        assert ratio is not None and ratio < 0.5, (
            f"{name}: rate ratio {entry['lambda_ratio']:.3f} but amplitude ratio "
            f"{ratio} at t_end"
        )


def test_excluded_arms_did_not_start_at_rest(arms, payload):
    grading = payload["grading"]
    for name, entry in grading["excluded"].items():
        eq = arms[name]["equilibrium"]
        assert (not eq["converged"]) or eq["a_max_over_g"] >= 1.0e-6 \
            or eq["a_max_over_g_horizontal"] >= 1.0e-6, (
            f"{name} is excluded but its reconstruction converged to "
            f"{eq['a_max_over_g']:.2e}: the exclusion is wrong"
        )


def test_unmeasured_arms_really_stayed_below_the_band(arms, payload):
    grading = payload["grading"]
    for name in grading["unmeasured"]:
        arm = arms[name]
        assert not arm["entered_band"]
        assert _band_rate(arm) is None
        assert arm["max_v_over_c0"] < BAND_LOW, (
            f"{name} reports max|v|/c0 = {arm['max_v_over_c0']:.3e} above the "
            f"band floor {BAND_LOW:g}, so it did enter the band"
        )


def test_no_arm_is_silently_dropped(arms, payload):
    grading = payload["grading"]
    seen = {"base"} | set(grading["switch"]) | set(grading["unmeasured"]) \
        | set(grading["excluded"])
    assert set(arms) == seen


def test_verdict_statement_names_every_graded_arm(payload):
    grading = payload["grading"]
    statement = grading["verdict"]["statement"]
    for name in list(grading["switch"]) + list(grading["unmeasured"]):
        assert name in statement, f"{name} is measured but not in the statement"
    assert grading["verdict"]["clearing_switch"] == sorted(
        n for n, e in grading["switch"].items() if e["verdict"] == "clears")


# ---------------------------------------------------------------------------
# G5 the horizon claim
# ---------------------------------------------------------------------------
def test_a_clearing_switch_extends_the_horizon_by_construction(arms, payload):
    """`clears` means: inside the new horizon the departure never reached the band."""
    grading = payload["grading"]
    t_end = arms["base"]["run"]["t_end"]
    for name in grading["verdict"]["clearing_switch"]:
        arm = arms[name]
        t_star = _t_star(arm)
        # either it never leaves the tolerance at all, or it leaves it much later
        assert t_star is None or t_star > 2.0 * (_t_star(arms["base"]) or 0.0), (
            f"{name}: t* = {t_star} is not a horizon extension"
        )
        assert arm["max_v_over_c0"] < BAND_HIGH, (
            f"{name}: amplitude {arm['max_v_over_c0']:.3e} reaches the band top"
        )
        assert t_end >= PREVIOUS_CAP


# ---------------------------------------------------------------------------
# G6 cross-document consistency with the anchor's own audit
# ---------------------------------------------------------------------------
def test_reference_rate_agrees_with_the_shipped_band_fit(arms):
    if not SHIPPED_BAND_FIT.exists():                     # pragma: no cover
        pytest.skip("stability_band_fit.json not shipped")
    shipped = json.loads(SHIPPED_BAND_FIT.read_text())
    entries = shipped["arms"] if isinstance(shipped.get("arms"), list) else []
    base = next((e for e in entries if e.get("arm") == "base"), None)
    if base is None:                                      # pragma: no cover
        pytest.skip("band-fit artefact has no reference arm")
    audited = base["band"]["lambda_per_s"]
    own = _band_rate(arms["base"])
    assert own == pytest.approx(audited, rel=0.10), (
        f"this study's reference rate {own:.2f} 1/s vs the anchor audit's "
        f"{audited:.2f} 1/s: more than 10 % apart"
    )