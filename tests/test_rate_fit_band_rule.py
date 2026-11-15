"""Gates for the band-rule rate estimator (``tensorsph.rate_fit``).

The rule these gates enforce is the one ``CASE-SPH-LAKE-AT-REST`` had to learn
twice (m21507): an exponential rate is only a rate where the quantity *is*
exponential, and a window that shows that must be validated -- not declared.

Two kinds of gate here, and the difference matters:

* **synthetic** series with a known ``sigma``: the estimator must recover it
  where it is measurable and must *refuse* it where it is not.  A refusal is a
  PASS here, because "no rate to quote" is the correct answer for a series with
  no exponential regime -- the failure mode being gated is a confident number.
* **shipped lake data** (``tests/data/lake_at_rest/stability.json``, manifest
  hashed): the module must reproduce the frozen rates to 1e-3 relative, and must
  reproduce the *documented failure* of the legacy window (``0.887`` / ``3.724``
  1/s against ``16.47``).  That binds this generic rule to evidence that was
  already frozen and adversarially reviewed, instead of to a fresh opinion.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorsph import rate_fit as rf

DATA = Path(__file__).resolve().parent / "data" / "lake_at_rest"


# --------------------------------------------------------------------------- #
# synthetic series with a known rate
# --------------------------------------------------------------------------- #
def exact_exponential(sigma=3.7, a0=1e-10, t_end=1.0, n=300):
    t = np.linspace(0.0, t_end, n)
    return t, a0 * np.exp(sigma * t)


def transient_exponential(seed=1e-6, sigma=16.0, tau=0.10, a_end=1e-2,
                          n=400, floor=1e-13):
    """Exponential envelope joined after a start-up of duration ``~tau``.

    This is the lake's shape: the quantity sits near a floor, then joins the
    exponential mode.  A window placed in the join measures the transient.
    """
    t_end = math.log(a_end / seed) / sigma
    t = np.linspace(0.0, t_end, n)
    a = seed * np.exp(sigma * t) * (1.0 - np.exp(-t / tau)) + floor
    return t, a


def saturating_exponential(a_sat=1e-3, sigma=6.0, a0=1e-12, n=600):
    t_end = math.log(a_sat / a0) / sigma * 1.15
    t = np.linspace(0.0, t_end, n)
    return t, np.minimum(a0 * np.exp(sigma * t), a_sat)


# --------------------------------------------------------------------------- #
# 1. the estimator is exact where it should be exact
# --------------------------------------------------------------------------- #
def test_exact_exponential_is_recovered_to_roundoff():
    t, a = exact_exponential()
    fit = rf.fit_band(t, a, None, None)
    assert fit["valid"]
    assert abs(fit["sigma"] / 3.7 - 1.0) < 1e-9
    assert fit["r2"] > 1.0 - 1e-12


def test_plateau_of_a_pure_exponential_spans_the_whole_range():
    t, a = exact_exponential(t_end=4.0)   # sigma * T / ln 10 = 6.4 decades
    p = rf.plateau(t, a)
    assert p["valid"]
    assert abs(p["sigma"] / 3.7 - 1.0) < 1e-6
    assert p["plateau_decades"] >= 6.0
    lo, hi = rf.auto_band(t, a)
    assert lo < hi and lo >= a.min() and hi <= a.max()


# --------------------------------------------------------------------------- #
# 2. the two ends of a window: transient (low) and saturation (high)
# --------------------------------------------------------------------------- #
def test_a_window_in_the_start_up_transient_is_not_a_rate():
    t, a = transient_exponential()
    inside = (float(np.interp(0.005, t, a)), float(np.interp(0.15, t, a)))
    legacy = rf.fit_band(t, a, *inside)
    p = rf.plateau(t, a)
    true_sigma = 16.0
    # the transient window is measurably biased and not exponential ...
    assert legacy["r2"] < 0.98
    assert abs(legacy["sigma"] / true_sigma - 1.0) > 0.20
    # ... while the plateau recovers the rate the transient hid
    assert p["valid"]
    assert abs(p["sigma"] / true_sigma - 1.0) < 0.03
    assert rf.verdict(rf.report(t, a, declared=inside))["declared_band_legal"] is False


def test_a_window_reaching_saturation_is_not_a_rate():
    t, a = saturating_exponential()
    p = rf.plateau(t, a)
    assert p["valid"]
    high = rf.fit_band(t, a, 1e-6, 1e-3)
    ok = rf.fit_band(t, a, 1e-9, 1e-7)
    assert high["sigma"] / p["sigma"] - 1.0 < -0.15
    assert abs(ok["sigma"] / p["sigma"] - 1.0) < 1e-6
    assert rf.verdict(rf.report(t, a, declared=(1e-6, 1e-3)))["declared_band_legal"] is False
    assert rf.verdict(rf.report(t, a, declared=(1e-9, 1e-7)))["declared_band_legal"] is True


# --------------------------------------------------------------------------- #
# 3. honest refusal: no exponential regime, and not-enough-samples
# --------------------------------------------------------------------------- #
def test_a_power_law_has_no_rate_even_though_R2_looks_acceptable():
    t = np.linspace(0.0, 20.0, 240)
    a = (1.0 + t) ** 3
    declared = (5.0, 200.0)
    fit = rf.fit_band(t, a, *declared)
    sweep = rf.sub_band_spread(t, a, *declared)
    v = rf.verdict(rf.report(t, a, declared=declared))
    # R^2 alone would have passed this window: it is the criterion that fails
    assert fit["r2"] > 0.97
    assert sweep["spread_rel"] > 0.5
    assert v["declared_band_legal"] is False
    assert v["plateau_valid"] is False
    assert v["reason"]


def test_a_sparse_series_is_unknown_not_passed():
    t = np.linspace(0.0, 8.0, 13)
    a = 1e-12 * np.exp(2.5 * t)
    p = rf.plateau(t, a)
    assert p["reason_kind"] == "sparse"
    assert p["valid"] is False
    assert rf.auto_band(t, a) is None


# --------------------------------------------------------------------------- #
# 4. the shipped lake numbers: agreement and the documented failure
# --------------------------------------------------------------------------- #
def _lake_arms():
    payload = json.loads((DATA / "stability.json").read_text())
    out = {}
    for arm in payload["arms"]:
        series = arm.get("samples") or arm["run"]["samples"]
        out[arm["arm"]] = ([s["t"] for s in series],
                           [s["v_over_c0"] for s in series])
    return out


def test_lake_declared_band_is_reproduced_from_the_raw_series():
    """The generic module must reproduce the shipped producer's band fits."""
    shipped = json.loads((DATA / "stability_band_fit.json").read_text())
    arms = _lake_arms()
    checked, thin = 0, []
    for entry in shipped["arms"]:
        t, a = arms[entry["arm"]]
        band = entry["band"]
        if not band["valid"] or band["lambda_per_s"] is None:
            continue
        mine = rf.fit_band(t, a, band["window"][0], band["window"][1])
        if band["points"] < rf.MIN_POINTS:
            # the producer ships this rate on fewer points than the band rule
            # accepts as a slope (its MIN_POINTS is 4, this module's is 6):
            # recorded as a finding, not silently reproduced
            thin.append((entry["arm"], band["points"]))
            assert math.isnan(mine["sigma"])
            # this arm is under-sampled for a one-decade band (103 samples over
            # 12 decades = 6.9 points/decade), so the ladder refuses it -- the
            # shipped 5-point rate is therefore *not* an independent check
            p = rf.plateau(t, a)
            assert p["valid"] is False
            assert p["reason_kind"] in ("sparse", "not_exponential")
            assert p["points_per_decade"] < 10.0
            continue
        assert mine["n_points"] == band["points"], entry["arm"]
        assert abs(mine["sigma"] / band["lambda_per_s"] - 1.0) < 1e-9, entry["arm"]
        checked += 1
    assert checked == 7, f"expected 7 full band fits, checked {checked}"
    assert thin == [("saturation_8s", 5)], thin


def test_lake_legacy_window_failure_is_reproduced_and_flagged():
    """The legacy absolute window returns 0.887 / 3.724 1/s -- not rates."""
    arms = _lake_arms()
    expected = {"seed_1e-6": 0.887, "seed_1e-8": 3.724}
    for name, documented in expected.items():
        t, a = arms[name]
        rec = rf.report(t, a, declared=(1e-10, 1e-5))
        v = rec["verdict"]
        assert abs(v["declared_sigma"] / documented - 1.0) < 2e-3, name
        assert v["declared_band_legal"] is False, name
    # ... and on the base arm the same window happens to be fine, which is why a
    # single-arm check never found the defect
    t, a = arms["base"]
    assert abs(rf.fit_band(t, a, 1e-10, 1e-5)["sigma"] / 16.299 - 1.0) < 2e-3


def test_lake_seed_relative_window_is_not_scale_free():
    """A ``[3a0, 25a0]`` window -- the form used by liquid_column -- is biased,
    and biased by a different amount on every arm: it tracks the seed, not the
    exponential."""
    arms = _lake_arms()
    biases, widths = {}, {}
    for name, (t, a) in arms.items():
        d = rf.relative_seed_defect(t, a)
        biases[name] = d["bias_rel"]
        widths[name] = math.log10(d["band"][1] / d["band"][0])
    finite = {k: v for k, v in biases.items() if v is not None}
    assert len(finite) >= 6
    assert (max(finite.values()) - min(finite.values())) > 0.30
    # r2 alone does not catch it: at least one arm fits "well" and is still
    # off by more than 15 %
    blind = 0
    for name, (t, a) in arms.items():
        d = rf.relative_seed_defect(t, a)
        if (d["legacy"]["r2"] is not None and d["legacy"]["r2"] >= 0.98
                and d["bias_rel"] is not None and abs(d["bias_rel"]) > 0.15):
            blind += 1
    assert blind >= 1
    # the windows themselves slide over seven orders of magnitude across arms
    assert max(widths.values()) - min(widths.values()) < 1e-9  # same width ...
    assert (max(biases[k] or 0.0 for k in finite) - min(finite.values())) > 0.3


def test_lake_declared_band_is_legal_where_the_estimator_holds():
    """The anchor's a-priori band passes on every unseeded arm.

    The seeded arm ``seed_1e-6`` is reported (not silently dropped) as marginal:
    its own plateau starts at ``1.9e-6``, so the band's lower edge sits in its
    start-up and the sub-band sweep is 6 % instead of 1 %.  Its *rate* is still
    within 1 % of the plateau, which is what the seed-independence claim needs;
    the *band* is what is tight there.
    """
    arms = _lake_arms()
    verdicts = {name: rf.verdict(rf.report(t, a, declared=(1e-6, 1e-3)))
                for name, (t, a) in arms.items()}
    for name in ("base", "courant_half", "resolution_dx025", "pairs_rebuilt",
                 "viscosity_1e-6", "seed_1e-8"):
        assert verdicts[name]["declared_band_legal"] is True, name
    marginal = verdicts["seed_1e-6"]
    assert marginal["declared_band_legal"] is False
    assert abs(marginal["declared_bias_rel"]) < 0.01  # rate fine, band tight
    # the sparse 8 s arm cannot be laddered at all: unknown, not passed
    assert verdicts["saturation_8s"]["plateau_valid"] is False


def test_a_plateau_rate_is_the_median_not_the_best_band():
    """No cherry-picking: the reported rate is the median over the plateau."""
    t, a = exact_exponential()
    p = rf.plateau(t, a)
    sigmas = [s for s in p["sigma_ladder"] if s is not None and not math.isnan(s)]
    assert p["sigma"] == pytest.approx(float(np.median(
        [s for s in sigmas if abs(s / 3.7 - 1) < 0.05])), rel=1e-9)
    assert p["spread_rel"] <= rf.SPREAD_MAX

# --------------------------------------------------------------------------- #
# 5. the two sibling implementations are wired to the rule, not around it
# --------------------------------------------------------------------------- #
def test_liquid_column_banded_growth_recovers_the_rate():
    from tensorsph.threed import liquid_column as lc
    t = np.linspace(0.0, 1.2, 300)
    a = 1e-9 * np.exp(15.0 * t)
    out = lc.fit_growth_rate(t, a, None, None)
    assert out["band_source"] == "auto"
    assert abs(out["omega"] / 15.0 - 1.0) < 1e-9


def test_liquid_column_seed_relative_window_can_reach_saturation():
    """The default ``(3, 25) a0`` window is the seed's window, not the mode's.

    With the same physics, a window quoted in multiples of the first sample lands
    past pinch-off as soon as the run starts near it -- and the fit then reports
    a fraction of the true rate.
    """
    from tensorsph.threed import liquid_column as lc
    a0, sigma = 1e-3, 15.0
    t = np.linspace(0.0, 1.0, 400)
    a = np.minimum(a0 * np.exp(sigma * t), 8.0 * a0)   # saturates at 8 a0
    out = lc.fit_growth_rate(t, a, 3.0, 25.0)
    assert out["band_source"] == "declared"
    assert out["omega"] / sigma - 1.0 < -0.15          # measurably diluted
    assert out["declared_legal"] is False
    assert out["plateau_omega"] is not None


def test_rayleigh_taylor_fit_carries_the_band_verdict():
    from tensorsph import rayleigh_taylor as rt
    t = np.linspace(0.0, 1.2, 300)
    a = 1e-9 * np.exp(15.0 * t)
    good = rt.fit_growth_rate(t, a, 1e-7, 1e-5)
    assert good["band_rule"]["declared_band_legal"] is True
    assert abs(good["sigma"] / 15.0 - 1.0) < 1e-9
    auto = rt.fit_growth_rate(t, a, None, None)
    assert auto["band_source"] == "auto"
    # a window in the start-up is now flagged where the number is produced
    t2, a2 = transient_exponential(seed=1e-5, sigma=15.0)
    bad = rt.fit_growth_rate(t2, a2, float(np.interp(0.005, t2, a2)),
                             float(np.interp(0.12, t2, a2)))
    assert bad["band_rule"]["declared_band_legal"] is False
