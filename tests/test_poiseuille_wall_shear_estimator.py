"""Gates for the Poiseuille wall-shear *estimator* audit.

The audit itself is a post-processing of ``benchmarks/poiseuille/out/results.json``
(no SPH is advanced), so the whole record can be re-derived inside the test in a
second.  What is pinned here:

* the momentum-budget reading of the transmitted drag -- the quantity that
  refutes the published "carries only 50 % of the wall shear" sentence -- still
  closes to a few percent, and the free-slip control still calibrates the drift
  estimator to better than 1 %;
* the near-wall stencil bias is still what the record says it is: the 1-``dx``
  secant on the *exact* analytic parabola reads ``1 - 2 dx/H``, and on the
  numerical profile it reads roughly half -- so the "-50 %" number may never be
  quoted as a drag deficit;
* the generated record and artifact are in sync with the stored run (no
  hand-copied numbers, no drift between JSON and markdown).

If a future run changes the wall model, these gates fire and the record has to
be re-derived rather than silently inherited.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "poiseuille"
RESULTS = BENCH / "out" / "results.json"
ARTIFACT = BENCH / "out" / "wall_shear_estimators.json"
DOC = ROOT / "docs" / "verification_poiseuille_wall_shear_estimator.md"
CHANNEL_DOC = ROOT / "docs" / "verification_poiseuille_channel.md"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "wall_shear_estimators", BENCH / "wall_shear_estimators.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wse = _load_module()


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def audit(data) -> dict:
    return wse.analyse(data)


def test_artifact_matches_a_fresh_derivation(audit):
    """The checked-in JSON must be reproducible from the stored run."""
    stored = json.loads(ARTIFACT.read_text())
    fresh = json.loads(json.dumps(audit))
    assert stored["summary"] == pytest.approx(fresh["summary"], rel=1e-12, abs=1e-12)
    assert len(stored["cases"]) == len(fresh["cases"])
    for a, b in zip(stored["cases"], fresh["cases"]):
        assert a["budget"]["drag_over_weight_control_corrected"] == pytest.approx(
            b["budget"]["drag_over_weight_control_corrected"], rel=1e-12)
        assert a["estimator_ratios"] == pytest.approx(
            b["estimator_ratios"], rel=1e-12, abs=1e-12)


def test_control_calibrates_the_drift_estimator(audit):
    """Free slip: D = 0, so the drift estimator must reproduce d<u>/dt = g."""
    control = audit["control"]
    assert control["rate_over_g"] == pytest.approx(1.0, abs=0.01), control
    assert control["residual_rms"] < 1e-3


def test_transmitted_drag_closes_the_momentum_budget(audit):
    """The wall removes the injected momentum: this refutes a drag deficit."""
    assert audit["summary"]["budget_gate_2pct"], audit["summary"]
    for case in audit["cases"]:
        ratio = case["budget"]["drag_over_weight_control_corrected"]
        assert 0.97 <= ratio <= 1.03, (case["resolution_H_over_dx"], ratio)
        # the 3 % gate is met at every rung, not only on average
        assert abs(ratio - 1.0) < 0.03


def test_the_secant_bias_is_the_stencil_and_the_wall_layer(audit):
    """`-50 %` is a reading, and its size is reproducible from geometry."""
    for case in audit["cases"]:
        dx_ratio = 2.0 * case["dx"] / audit["inputs"]["height"]
        assert case["estimator_ratios"]["secant_1dx_on_analytic_parabola"] == \
            pytest.approx(1.0 - dx_ratio, abs=1e-12)
        secant = case["estimator_ratios"]["secant_1dx"]
        assert secant < 0.7, (case["resolution_H_over_dx"], secant)   # published defect
        # the drag reading is nevertheless close to 1 -> the secant under-reads
        assert case["budget"]["drag_over_weight_control_corrected"] > secant + 0.25


def test_higher_order_near_wall_estimator_is_unstable(audit):
    """The 3-point quadratic is no better than the secant: acoustic scatter."""
    worst = max(abs(c["estimator_ratios"]["three_point_quadratic"] - 1.0)
                for c in audit["cases"])
    assert worst > 0.3, "3-point estimator unexpectedly stable: re-derive the record"
    scatter = max(c["max_row_std_over_umax"] for c in audit["cases"])
    assert scatter > 0.05, "near-wall rows unexpectedly quiet: re-derive the record"


def test_interior_viscosity_is_within_the_documented_band(audit):
    for case in audit["cases"]:
        nu_eff = case["interior"]["nu_eff_over_nu"]
        assert 0.9 < nu_eff < 1.4, (case["resolution_H_over_dx"], nu_eff)


def test_record_doc_quotes_the_artifact(audit):
    text = DOC.read_text()
    s = audit["summary"]
    for needle in (
        f"{s['drag_ratio_finest']:.4f}",
        f"{s['secant_ratio_finest']:.3f}",
        f"{s['control_bias_pct']:.2f}",
    ):
        assert needle in text, (needle, "number missing from the record")
    assert "withdraw" in text.lower()
    assert "figures/poiseuille_wall_shear.png" in text
    assert (ROOT / "docs" / "figures" / "poiseuille_wall_shear.png").exists()


def test_channel_doc_no_longer_claims_a_drag_deficit(audit):
    """The published record must point at the audit and drop the wrong sentence."""
    text = CHANNEL_DOC.read_text()
    assert "4.1 What the -50 % wall-shear reading measures" in text
    assert "verification_poiseuille_wall_shear_estimator.md" in text
    assert "carries only" not in text
    # the drag reading is quoted with its numbers, from the artifact
    assert f"{audit['summary']['drag_ratio_finest']:.4f}" in text


def test_channel_rows_still_reproduce_the_published_secant(data):
    """The audit must not have moved the measured profile it audits."""
    for res in data["results"]:
        rows = res["rows"]
        dx = res["dx"]
        secant = 1000.0 * data["viscosity"] * (rows[1]["u_mean"] - rows[0]["u_mean"]) / dx
        assert secant == pytest.approx(res["wall_shear_measured"], rel=1e-9)
        assert np.isfinite(res["wall_row_slip_over_umax"])
