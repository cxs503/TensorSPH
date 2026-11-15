"""Unit-consistency guards for ``CASE-POISEUILLE-SPH`` phase reporting.

Background (m21208 audit): the SPH driver stores per-row phases in **radians**
(``tensorsph.stokes_second.harmonic_fit`` returns radians), but the summary
fields were emitted into keys named ``*_deg`` without conversion, and the two
figure/notes consumers treated them as degrees.  The consequence was a factor
``180/pi = 57.3`` error in a headline number of the verification note
("-0.95 deg against -0.96 deg" for a phase lag that is actually -54.7 deg) and a
phase panel whose y-range was set by degrees-valued FD points while the
SPH/closed-form curves were radians.

These tests pin the conventions:

* ``*_rad`` fields hold radians, ``*_deg`` fields hold degrees, and the two
  agree to round-off;
* every ``*_deg`` field lies in ``[-180, 180]`` (a phase of -0.95 "deg" for a
  Stokes layer of ``delta/H = 0.39`` is not physical, so the range check alone
  would have caught the defect);
* the analytic centre-line phase recovered from the artifact matches the closed
  form :func:`tensorsph.womersley.phase` in degrees;
* the frequency-discrimination phase difference is reported in degrees and
  matches the analytic difference (it must not be a raw radian difference
  mislabelled).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorsph import womersley as wm

RESULT = Path(__file__).resolve().parents[1] / "benchmarks" / "case_poiseuille" / "out" / "result.json"

HEIGHT = 0.1
VISCOSITY = 0.01
BODY_FORCE = 9.81
REL = 1e-9


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.exists():                       # pragma: no cover - artifact optional
        pytest.skip("result.json not generated")
    return json.loads(RESULT.read_text())


def _graded(result: dict) -> dict:
    return result["harmonic_runs"][-1]


# --------------------------------------------------------------------------- #
# Schema / unit conventions of the per-run summary
# --------------------------------------------------------------------------- #

def test_summary_phase_fields_are_unit_explicit(result):
    run = _graded(result)
    for stem in ("phase_centre", "phase_centre_analytic"):
        assert f"{stem}_rad" in run, f"missing unit-explicit field {stem}_rad"
        assert f"{stem}_deg" in run, f"missing unit-explicit field {stem}_deg"
        rad, deg = run[f"{stem}_rad"], run[f"{stem}_deg"]
        assert deg == pytest.approx(math.degrees(rad), rel=REL, abs=1e-9)


def test_summary_phase_degrees_are_in_physical_range(result):
    run = _graded(result)
    for stem in ("phase_centre", "phase_centre_analytic"):
        assert abs(run[f"{stem}_deg"]) <= 180.0
    # A phase lag in *degrees* that is under a degree while the Stokes layer
    # spans 39 % of the channel is the signature of a radian value in a
    # degrees-named field.
    assert abs(run["phase_centre_analytic_deg"]) > 1.0


def test_per_row_phase_fields_are_unit_explicit(result):
    for run in result["harmonic_runs"] + [result["harmonic_fast"]]:
        for row in run["rows"]:
            assert row["phase_deg"] == pytest.approx(math.degrees(row["phase"]), rel=REL, abs=1e-9)
            assert row["phase_analytic_deg"] == pytest.approx(
                math.degrees(row["phase_analytic"]), rel=REL, abs=1e-9)


def test_analytic_centre_phase_matches_the_closed_form_in_degrees(result):
    run = _graded(result)
    rows = run["rows"]
    centre = min(rows, key=lambda r: abs(r["y_over_H"] - 0.5))
    expected = math.degrees(float(wm.phase(centre["y"], HEIGHT, BODY_FORCE,
                                           VISCOSITY, run["omega"])))
    assert centre["phase_analytic_deg"] == pytest.approx(expected, abs=1e-6)
    assert run["phase_centre_analytic_deg"] == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# Frequency discrimination (G6 companion quantities)
# --------------------------------------------------------------------------- #

def test_frequency_phase_difference_is_reported_in_degrees(result):
    freq = result["frequency_discrimination"]
    assert "phase_difference_measured_deg" in freq
    assert "phase_difference_measured_rad" in freq
    deg, rad = freq["phase_difference_measured_deg"], freq["phase_difference_measured_rad"]
    assert deg == pytest.approx(math.degrees(rad), rel=1e-9, abs=1e-9)
    ana = freq["phase_difference_analytic_deg"]
    assert ana == pytest.approx(
        math.degrees(freq["phase_difference_analytic_rad"]), rel=1e-9, abs=1e-9)
    # the analytic difference must equal the closed forms evaluated at the two
    # frequencies, in degrees
    slow = result["geometry"]
    h, nu, g0 = slow["height"], slow["viscosity"], slow["body_force"]
    # the analytic values are reported on the discrete centre row, so compare at
    # that ordinate rather than at y = H/2
    base = next(r for r in result["harmonic_runs"] if r["resolution_H_over_dx"] == 12)
    y = min((r["y"] for r in base["rows"]), key=lambda yy: abs(yy / h - 0.5))
    a_slow = math.degrees(float(wm.phase(y, h, g0, nu, freq["omega_slow"])))
    a_fast = math.degrees(float(wm.phase(y, h, g0, nu, freq["omega_fast"])))
    assert ana == pytest.approx(a_fast - a_slow, abs=1e-4)
    # a quasi-steady response would give ~0 deg; the real lag is tens of degrees
    assert abs(ana) > 15.0

# --------------------------------------------------------------------------- #
# The emitted schema, exercised on a synthetic case that *is* the closed form
# --------------------------------------------------------------------------- #

def test_evaluate_harmonic_emits_unit_explicit_fields():
    """No solver needed: feed ``evaluate_harmonic`` a case whose samples are the
    closed form, and require the summary to report that phase in degrees."""
    import numpy as np

    from benchmarks.case_poiseuille import run as chrun

    height, nu, g0, omega = HEIGHT, VISCOSITY, BODY_FORCE, 13.0
    n = 8
    dx = height / n
    y = (np.arange(n) + 0.5) * dx
    period = 2.0 * math.pi / omega
    t = np.arange(0.0, 2.0 * period, period / 64.0)
    u = wm.periodic_velocity(y, t, height, g0, nu, omega).T      # (nt, ny)
    case = {"sample_times": t, "u_samples": u, "y_rows": y,
            "dt": float(t[1] - t[0]), "steps": int(t.size),
            "n_fluid": 2 * n, "n_ghost": 0,
            "trace": [{"max_mach": 0.01}], "wall_clock_s": 0.0}
    out = chrun.evaluate_harmonic(case, omega, height, n)

    centre_y = min(y, key=lambda yy: abs(yy / height - 0.5))
    phase_deg = math.degrees(float(wm.phase(centre_y, height, g0, nu, omega)))
    assert out["phase_centre_deg"] == pytest.approx(math.degrees(out["phase_centre_rad"]))
    assert out["phase_centre_deg"] == pytest.approx(phase_deg, abs=1e-6)
    # the Womersley lag at Wo = 1.8 is tens of degrees, not a fraction of one
    assert abs(out["phase_centre_deg"]) > 30.0
    assert out["phase_centre_pct_of_quarter_turn"] == pytest.approx(0.0, abs=1e-6)
    assert all(out["gates"][k] for k in out["gates"])
