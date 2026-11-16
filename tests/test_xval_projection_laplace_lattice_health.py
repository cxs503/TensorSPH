"""Guards for the lattice-health diagnosis of the diverging marched arm.

The anchor ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` left one question open
(``docs/xval_projection_laplace_horizon_instability.md`` ⑤): when the marched
projection arm blows up, did the **lattice** break or is the **traction**
mis-projected?  ``benchmarks/laplace_sph/diagnose_lattice_health.py`` answers it
with the ``lattice_health`` instrument that was added for it but never run.

This module pins the parts that must not drift:

* the classifier is a pure function of a sample and its thresholds are
  pre-registered -- a later unit cannot quietly widen them to call a broken
  lattice healthy (and vice versa);
* the instrument actually records the **breaking** step.  Before this unit the
  diverged history entry carried only ``umax_over_vref``: the one step whose
  lattice health decides the question was the one step with no lattice block.
  That is asserted here on a deliberately blown-up 2-step march, not assumed;
* the shipped (``stability_report=False``) records stay byte-reproducible --
  the diverged entry must gain nothing unless the instrument is switched on.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
REC = BENCH / "results_dynamic_ladder"
DOC = ROOT / "docs" / "xval_projection_laplace_lattice_health.md"

sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import diagnose_lattice_health as dlh  # noqa: E402


# --------------------------------------------------------------------------- #
# the classifier (pure, no compute)
# --------------------------------------------------------------------------- #
HEALTHY = {"lattice": {
    "nn_min": 6.0, "nn_median": 18.0, "pair_min_over_dx": 1.0,
    "pair_p01_over_dx": 1.0, "color_min": 0.0, "color_max": 1.0,
    "kappa_R_max": 2.31, "gradc_dx_max": 0.48, "acc_over_aref_max": 14.97}}


def test_classify_passes_the_measured_healthy_lattice():
    """The values a freshly built ``R/dx = 16`` lattice actually reports."""
    broken, fired = dlh.classify(HEALTHY)
    assert broken is False and fired == []


@pytest.mark.parametrize("key,value", [
    ("pair_min_over_dx", 0.1),        # particles overlap
    ("color_min", -0.2),              # colour leaves [0, 1]
    ("color_max", 1.4),
    ("kappa_R_max", 12.0),            # single-pair curvature spike
    ("gradc_dx_max", 9.0),
    ("acc_over_aref_max", 900.0),     # traction that no pressure solve balances
])
def test_classify_fires_on_each_declared_rule(key, value):
    lat = dict(HEALTHY["lattice"])
    lat[key] = value
    broken, fired = dlh.classify({"lattice": lat})
    assert broken is True
    assert key in fired


def test_classify_treats_a_neighbour_collapse_as_broken():
    lat = dict(HEALTHY["lattice"])
    lat["nn_min"] = 2.0                       # 2 / 18 = 0.11 < 0.20
    assert "nn_min_over_median" in dlh.classify({"lattice": lat})[1]


def test_an_unmeasurable_lattice_counts_as_broken():
    """NaN is not health: a block that cannot be measured must fire, not pass."""
    lat = {k: float("nan") for k in dlh.BREAK_RULES}
    broken, fired = dlh.classify({"lattice": lat})
    assert broken is True and fired


def test_no_lattice_block_is_not_a_pass():
    """A sample with no lattice block must not be silently classified healthy."""
    broken, fired = dlh.classify({"step": 1})
    assert broken is True and fired == ["no_lattice_block"]   # fail-closed
    # ...and the record-level scan does not mistake such a sample for a break
    assert dlh.first_break({"history": [{"step": 1}]}) == (None, [])


def test_thresholds_are_pre_registered():
    """Guard the declared numbers: widening one is a change of verdict."""
    assert dlh.BREAK_RULES["pair_min_over_dx"]["value"] == pytest.approx(0.25)
    assert dlh.BREAK_RULES["nn_min_over_median"]["value"] == pytest.approx(0.20)
    assert dlh.BREAK_RULES["kappa_R_max"]["value"] == pytest.approx(5.0)
    assert dlh.BREAK_RULES["gradc_dx_max"]["value"] == pytest.approx(5.0)
    assert dlh.BREAK_RULES["acc_over_aref_max"]["value"] == pytest.approx(50.0)
    assert dlh.BREAK_RULES["color_min"]["value"] == pytest.approx(-0.05)
    assert dlh.BREAK_RULES["color_max"]["value"] == pytest.approx(1.05)


# --------------------------------------------------------------------------- #
# the instrument: the breaking step must carry the diagnosis
# --------------------------------------------------------------------------- #
def _forced_blow_up(stability_report: bool):
    """A 2-step march made to trip the divergence guard by an absurd dt."""
    from tensorsph.projection_adjoint import relax_projection_consistent
    return relax_projection_consistent(
        radius=0.02, dx=0.02 / 4.0, sigma=0.072, rho0=1000.0, outer=0.012,
        steps=2, dt_factor=1.0e9, nu_factor=0.0, sample_every=1,
        cg_max_iter=200, stability_report=stability_report)


def test_diverged_step_carries_the_lattice_block_when_the_instrument_is_on():
    rec = _forced_blow_up(True)
    assert rec["diverged"] is True
    broke = [h for h in rec["history"] if h.get("diverged")]
    assert broke, "no diverged entry to diagnose"
    entry = broke[-1]
    assert entry["finite_pos_in"] is True
    assert entry["finite_acc"] is True
    assert "lattice" in entry, (
        "the breaking step is the only decisive sample and it must carry the "
        "lattice block when stability_report=True")
    for key in ("pair_min_over_dx", "nn_min_over_median", "kappa_R_max",
                "gradc_dx_max", "acc_over_aref_max"):
        if key == "nn_min_over_median":
            assert math.isfinite(entry["lattice"]["nn_median"])
            continue
        assert math.isfinite(entry["lattice"][key]), key


def test_shipped_records_are_untouched_by_the_instrument():
    """stability_report=False must not grow the diverged entry (byte-repro)."""
    rec = _forced_blow_up(False)
    entry = [h for h in rec["history"] if h.get("diverged")][-1]
    assert set(entry) == {"step", "diverged", "umax_over_vref"}


# --------------------------------------------------------------------------- #
# the records (skipped when the arm has not been run in this checkout)
# --------------------------------------------------------------------------- #
def _load(name: str, sub: str = ""):
    p = (REC / sub / f"{name}.json") if sub else (REC / f"{name}.json")
    if not p.exists():
        pytest.skip(f"record {p} not in the tree")
    return json.loads(p.read_text())


def test_fine_level_march_is_reproducible_and_healthy():
    """The ``R/dx = 16`` arm the ladder record reports as dead on step 3."""
    rec = _load("lattice_health_dx16", "health_local_t24")
    health = [h for h in rec["history"] if "lattice" in h]
    assert health, "the diagnosis arm must carry lattice blocks"
    # every sampled step of the head of the march has a measurable lattice
    for s in health[:5]:
        broken, fired = dlh.classify(s)
        assert broken is False, (s["step"], fired)
    assert rec["first_break_step"] is None
    # and the Laplace jump is the anchor's acceptance number
    assert abs(rec["dp_rel_error_pct"]) < 3.0


def test_the_instrument_is_off_in_the_shipped_ladder_records():
    rec = _load("dx16_nu01")
    assert rec["diverged"] is True          # the old record is preserved as-is
    broke = [h for h in rec["history"] if h.get("diverged")][-1]
    assert "lattice" not in broke


def test_doc_page_exists_and_states_the_verdict():
    assert DOC.exists(), "the diagnosis needs a page"
    text = DOC.read_text()
    for needle in ("diagnose_lattice_health.py", "BREAK_RULES", "< 3 %"):
        assert needle in text, "page does not mention %r" % needle