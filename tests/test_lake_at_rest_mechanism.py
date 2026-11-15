"""Gates on the mechanism-identification study of `CASE-SPH-LAKE-AT-REST`.

The artefact is ``tests/data/lake_at_rest/mechanism.json`` (written by
``benchmarks/lake_at_rest/mechanism.py``).  These gates are deliberately the
kind that *decide* something rather than merely execute the file:

* the band estimator and the derived exponents are **re-derived here** from the
  shipped raw series, so a defect in the producer's reduction cannot ship
  silently;
* the verdict flags are **recomputed from the declared rules** and compared with
  the stored ones, so the document's headline sentence cannot drift from its own
  numbers;
* the "never entered the band" statement is checked against each arm's own
  series (it is the only way to make a *no-growth* claim from a finite run);
* the produced document must quote the stored verdict, not a stronger one.

Run with ``-m "not slow"`` to skip the schema-independent physics gates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "tests" / "data" / "lake_at_rest"
MECHANISM = DATA / "mechanism.json"
STABILITY = DATA / "stability.json"
DOC = HERE / "docs" / "verification_lake_at_rest_mechanism.md"

sys.path.insert(0, str(HERE / "benchmarks" / "lake_at_rest"))
import band_fit  # noqa: E402

BAND_LOW = band_fit.BAND_LOW
BAND_HIGH = band_fit.BAND_HIGH
TAIL = 10          # the last samples before the run stops for saturation


def payload() -> dict:
    if not MECHANISM.is_file():
        pytest.skip(f"missing {MECHANISM}")
    return json.loads(MECHANISM.read_text())


# ---------------------------------------------------------------------------
# shape of the artefact
# ---------------------------------------------------------------------------
def test_case_identity_and_band():
    data = payload()
    assert data["case"] == "CASE-SPH-LAKE-AT-REST"
    assert data["band"] == [BAND_LOW, BAND_HIGH], (
        "the study must use the declared exponential band, not a private window")
    assert data["protocol"]["periodic_x"] is True


def test_every_declared_arm_is_present():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mechanism", HERE / "benchmarks" / "lake_at_rest" / "mechanism.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorded = set(payload()["arms"])
    declared = set(module.ARMS)
    # a reduced sweep is allowed (the artefact says so); every recorded arm must
    # still be a declared one, and the reference arm must be present.
    assert recorded <= declared, f"undeclared arms: {recorded - declared}"
    assert "base" in recorded, "the reference arm is not optional"
    for name in recorded:
        arm = payload()["arms"][name]
        for key in ("config", "growth", "mode", "oscillation", "localisation",
                    "samples", "max_v_over_c0", "entered_band"):
            assert key in arm, f"arm {name} lacks {key}"
        assert arm["samples"], f"arm {name} ships no raw series"


def test_arm_series_are_consistent_with_their_own_claims():
    for name, arm in payload()["arms"].items():
        series = arm["samples"]
        assert series == sorted(series, key=lambda s: s["t"]), name
        peak = max(s["v_over_c0"] for s in series)
        assert peak == pytest.approx(arm["max_v_over_c0"], rel=1e-12), name
        if not arm.get("equilibrium_valid", True):
            continue          # starts off-equilibrium: no band claim either way
        if arm["entered_band"]:
            assert peak >= BAND_LOW, name
        else:
            # "never entered the band" is exactly the statement that the whole
            # series stayed below the band floor: that is what the doc may claim.
            assert peak < BAND_LOW, name


def test_localisation_is_a_distribution():
    for name, arm in payload()["arms"].items():
        local = arm["localisation"]
        assert 0.0 <= local["ke_surface_fraction"] <= 1.0 + 1e-9, name
        assert 0.0 <= local["ke_bottom_fraction"] <= 1.0 + 1e-9, name
        assert 0.0 <= local["fastest_row_height"] <= 1.0 + 1e-9, name
        assert len(local["row_speed_mean"]) == arm["run"]["rows"], name


# ---------------------------------------------------------------------------
# re-derivation: the numbers the document quotes
# ---------------------------------------------------------------------------
def test_band_rates_re_derive_from_the_shipped_series():
    for name, arm in payload()["arms"].items():
        again = band_fit.fit_lambda(arm["samples"], BAND_LOW, BAND_HIGH)
        stored = arm["growth"]
        assert again["points"] == stored["points"], name
        if stored["lambda_per_s"] is None:
            assert again["lambda_per_s"] is None, name
            continue
        assert again["lambda_per_s"] == pytest.approx(stored["lambda_per_s"],
                                                      rel=1e-12), name
        if not arm.get("equilibrium_valid", True):
            continue          # starts off-equilibrium: not a stability arm
        assert stored["r2_exp"] >= 0.99, (
            f"{name}: an exponential rate is only a rate if the fit is "
            f"exponential (R^2 = {stored['r2_exp']:.4f})")


def test_base_rate_agrees_with_the_shipped_horizon_study():
    """Two independent runs of the reference arm must agree on the rate."""
    if not STABILITY.is_file():
        pytest.skip("no stability artefact")
    stability = json.loads(STABILITY.read_text())
    shipped = {arm["arm"]: arm for arm in stability["arms"]}
    if "base" not in shipped:
        pytest.skip("shipped study has no base arm")
    reference = band_fit.fit_lambda(
        band_fit.samples_of(shipped["base"]), BAND_LOW, BAND_HIGH)
    measured = payload()["arms"]["base"]["growth"]["lambda_per_s"]
    assert reference["lambda_per_s"] is not None and measured is not None
    assert measured == pytest.approx(reference["lambda_per_s"], rel=0.10), (
        f"reference arm rate drifted: {measured:.3f} vs shipped "
        f"{reference['lambda_per_s']:.3f} 1/s")


def test_exponents_re_derive_from_the_arm_rates():
    """Re-derive every stored exponent from the arm rates it claims to use."""
    import math as _math

    data = payload()
    arms = data["arms"]
    exponent = data["reductions"]["exponent"]

    def rate(name):
        if name not in arms or not arms[name].get("equilibrium_valid", True):
            return None
        value = arms[name]["growth"]["lambda_per_s"]
        return value if value and value > 0.0 else None

    for key, field, lo, hi in (("gravity", "gravity", "g_quarter", "g_quad"),
                               ("sound_speed", "sound_speed", "c0_half", "c0_double"),
                               ("h", "h", "h_lo", "h_hi")):
        present = [name for name in (lo, hi) if name in arms]
        if len(present) == 2 and rate(lo) and rate(hi):
            expected = _math.log(rate(hi) / rate(lo)) / _math.log(
                arms[hi]["config"][field] / arms[lo]["config"][field])
        elif present and rate("base") and rate(present[0]):
            other = present[0]
            expected = _math.log(rate(other) / rate("base")) / _math.log(
                arms[other]["config"][field] / arms["base"]["config"][field])
        else:
            assert exponent.get(key) is None, (key, "must be None", exponent)
            continue
        assert exponent[key] == pytest.approx(expected, rel=1e-9), key


def test_switch_ratios_re_derive_from_the_arm_rates():
    arms = payload()["arms"]
    base = (arms["base"]["growth"]["lambda_per_s"]
            if arms["base"].get("equilibrium_valid", True) else None)
    for name, ratio in payload()["reductions"]["switch_ratio"].items():
        value = (arms[name]["growth"]["lambda_per_s"]
                 if arms[name].get("equilibrium_valid", True) else None)
        if value is None or base is None:
            assert ratio is None, name
            continue
        assert ratio == pytest.approx(value / base, rel=1e-9), name


# ---------------------------------------------------------------------------
# the verdict: rules, not prose
# ---------------------------------------------------------------------------
def test_verdict_flags_follow_the_declared_rules():
    data = payload()
    verdict = data["verdict"]
    exponent = data["reductions"]["exponent"]
    p_g, p_c = exponent.get("gravity"), exponent.get("sound_speed")
    if p_g is not None:
        assert verdict["gravity_wave_branch"] == (abs(p_g - 0.5) <= 0.15)
        assert verdict["kernel_lattice_mode"] == (
            abs(p_g) < 0.15 and p_c is not None and abs(p_c) < 0.15)
    if p_c is not None:
        assert verdict["acoustic_stiffness"] == (abs(p_c - 1.0) <= 0.25)
    cleared = sorted(name for name, value
                     in data["reductions"]["switch_ratio"].items()
                     if value is not None and value < 0.1)
    assert sorted(verdict["switch_that_clears"]) == cleared


def test_document_quotes_the_stored_verdict_and_not_a_stronger_one():
    if not DOC.is_file():
        pytest.skip("document not generated")
    text = DOC.read_text()
    statement = payload()["verdict"]["statement"]
    assert statement in text, "the document must carry the artefact's own verdict"
    # The anchor is a machine-precision identity; nothing here may upgrade it.
    assert "**FAIL**" in text or "FAIL as an unconditional statement" in text
    for forbidden in ("anchor now passes", "verdict is PASS", "proves lambda = 0"):
        assert forbidden not in text


def test_no_growth_claim_is_bounded_by_the_run_length():
    """An arm with no band entry must not be described as 'stable'."""
    if not DOC.is_file():
        pytest.skip("document not generated")
    text = DOC.read_text()
    arms = payload()["arms"]
    quiet = [name for name, arm in arms.items() if not arm["entered_band"]]
    if quiet:
        assert "not** the same as `lambda = 0`" in text.replace("*", "**", 1) \
            or "no growth was measured" in text, (
            "the document must say that 'never entered the band' is not "
            "lambda = 0")