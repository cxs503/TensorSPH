"""Recorded-evidence gate for the sessile-droplet contact-angle verification.

The benchmark (``benchmarks/contact_angle/run.py``) integrates a drop resting on
a wall with ``Config.contact_angle`` prescribed and measures the angle the
interface actually adopts.  It is not run inside the test suite - a relaxation
is ~1 min of CPU - so these tests read the committed record
``benchmarks/contact_angle/contact_angle.json`` and pin what it says.  Re-running
the benchmark regenerates both the record and ``docs/benchmarks/contact-angle.md``.

Status of the shipped model (mission m2621, 2026-11-03): the integrated interface
does **not** adopt the prescribed angle, so the physical gate is a
``xfail(strict=True)``.  Fixing the model makes it XPASS and fails the suite -
that is the intended forcing function for updating the record.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "contact_angle"
RECORD = BENCH / "contact_angle.json"


def _load():
    if not RECORD.exists():                              # pragma: no cover
        pytest.skip("benchmark record missing; run benchmarks/contact_angle/run.py")
    return json.loads(RECORD.read_text())


def test_record_is_well_formed():
    """Every run carries the quantities the document quotes."""
    rep = _load()
    assert rep["benchmark"] == "contact_angle_sessile_droplet"
    assert rep["runs"], "no runs recorded"
    for run in rep["runs"]:
        for key in ("theta_target_deg", "theta_fit_deg", "theta_corr_deg",
                    "theta_geom_deg", "p_in", "p_theory", "spurious_ratio",
                    "steps", "result", "checks"):
            assert key in run, f"{key} missing from run {run.get('theta_target_deg')}"
        assert run["result"] in ("pass", "fail")
        # the verdict must follow the individual gates, not be asserted by hand
        assert run["result"] == ("pass" if all(run["checks"].values()) else "fail")


def test_target_angle_is_reported_in_sync_with_the_case():
    """The record's target angles are the ones the harness was asked to hit."""
    rep = _load()
    targets = sorted(r["theta_target_deg"] for r in rep["runs"])
    assert all(0.0 < t < 180.0 for t in targets)
    assert any(abs(t - 60.0) < 1e-6 for t in targets), targets


def test_estimator_gauge_is_small():
    """The angle estimator bias on a synthetic exact cap stays below 2 deg.

    If this fails the measurement apparatus, not the model, is what moved: the
    synthetic cap is an exact circular cap, so any bias here is discretisation
    and one-particle-thick contact-line error.
    """
    rep = _load()
    for run in rep["runs"]:
        assert abs(run["gauge_bias_deg"]) < 2.0, run["gauge_bias_deg"]


def test_spurious_currents_are_reported():
    rep = _load()
    for run in rep["runs"]:
        # the level is a measurement, not an assertion; it must be finite so the
        # report can quote it
        assert math.isfinite(run["spurious_ratio"])


@pytest.mark.xfail(strict=True, reason=(
    "m2621 measured the merged contact-angle BC in an integrated run: a "
    "theta = 60 deg prescribed angle produced a fitted 115 deg interface "
    "(2 atan(H/a) gives 93.5 deg), i.e. the drop never spread, and the Laplace "
    "pressure came out 60 % low.  The BC rotates the colour gradient locally; "
    "it exerts no force on the contact line, so the cap keeps the angle of its "
    "initial condition.  This test is the gate on the physical claim."))
def test_measured_angle_matches_prescription():
    """Measured (gauge-corrected) angle within 15 % of the prescribed one."""
    rep = _load()
    for run in rep["runs"]:
        assert run["theta_fit_error"] <= rep["acceptance_limit"], (
            run["theta_target_deg"], run["theta_corr_deg"], run["theta_fit_error"])


@pytest.mark.xfail(strict=True, reason=(
    "same measurement: interior (apex-band) pressure 26.6 Pa against "
    "sigma/R = 65.9 Pa, -60 %.  Unchanged from the m2174 finding that the "
    "single-phase colour function carries only half of the 1 -> 0 jump at a "
    "wall-truncated interface."))
def test_laplace_pressure_matches_cap_curvature():
    rep = _load()
    for run in rep["runs"]:
        assert run["p_error"] <= rep["acceptance_limit"], (
            run["theta_target_deg"], run["p_in"], run["p_theory"], run["p_error"])