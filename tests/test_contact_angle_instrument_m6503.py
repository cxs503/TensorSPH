"""The contact-angle *instrument*, measured against exact interfaces (m6503).

Why this file exists: debt E3 (``docs/verification_contact_angle_m6453.md``)
recorded that the merged contact-angle BC had a harness that executed but an
instrument that returned ``nan`` for 8 of its 10 analytic rows, so no angle
could be read at all.  ``benchmarks/contact_angle/run.py`` has since been
repaired (see ``docs/verification_contact_angle_m6503.md``); these tests keep
the three layers apart and pin the honest state of each.

Layers:

1. **analytic** -- on *exact* interfaces (straight line, circular arc, sampled
   densely along the curve) the device must reproduce the set angle to far
   better than the 2 deg instrument gate.  If this fails, nothing below it
   means anything.
2. **resolution** -- on a *particle lattice* cap at production ``R/dx`` the
   device has a discretisation floor; it is asserted to be small and to shrink
   with resolution, and a reading with too few envelope points must come back
   as ``nan`` rather than as a number.
3. **physics** -- the relaxed drop of the shipped configuration is recorded
   with its admissibility gates (wall penetration, spurious current).  The
   reading is *not* graded as a contact-angle capability today: the shipped
   ``R/dx = 6.7`` run lets fluid reach below the wall by about one spacing, so
   the contact line is not where the fluid meets the wall.  When a future
   change makes the run admissible, the pin below turns red and forces this
   file (and the doc) to be rewritten.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "contact_angle"


def _load_bench():
    path = BENCH / "run.py"
    spec = importlib.util.spec_from_file_location("contact_angle_bench_m6503", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_bench()


@pytest.fixture(scope="module")
def report():
    return json.loads((BENCH / "report.json").read_text())


# --------------------------------------------------------------------------- #
# 1. Analytic layer: the device on exact interfaces
# --------------------------------------------------------------------------- #

def test_instrument_reproduces_exact_interfaces(bench):
    """Every analytic row must be recovered far inside the 2 deg gate."""
    inst = bench.instrument_check()
    assert len(inst["rows"]) == 10
    assert all(r["model"] in {"line", "circle"} for r in inst["rows"]), \
        [(r["kind"], r["theta_set_deg"], r["model"]) for r in inst["rows"]]
    assert inst["all_within_gate"] is True
    # the gate is 2 deg; the device is exact to ~1e-3 deg on exact input, so pin
    # the tighter bound -- a regression that biases the estimator by even 0.1 deg
    # must show up here, not only at the gate.
    assert inst["worst_abs_error_deg"] < 0.05


def test_straight_interface_uses_the_line_model(bench):
    """A straight interface must not be forced through the circle branch.

    The Kasa circle fit is singular on collinear points; the collinearity
    selection is what makes theta = 90 deg (a vertical surface) come out as
    90.000 deg instead of an arbitrary arc.
    """
    rows = [r for r in bench.instrument_check()["rows"] if r["kind"] == "line"]
    assert {r["model"] for r in rows} == {"line"}
    vertical = next(r for r in rows if r["theta_set_deg"] == 90.0)
    assert abs(vertical["error_deg"]) < 1e-6


def test_non_wetting_interface_is_not_measured_as_ninety(bench):
    """theta > 90 overhangs the contact line: the old x-envelope read ~89 deg.

    The repaired instrument traces the wall-normal envelope as well, so the
    super-hydrophobic rows must land near their set angle, not near 90.
    """
    rows = [r for r in bench.instrument_check()["rows"] if r["kind"] == "arc"]
    for theta in (120.0, 150.0):
        row = next(r for r in rows if r["theta_set_deg"] == theta)
        assert abs(row["error_deg"]) < 0.05, row


# --------------------------------------------------------------------------- #
# 2. Resolution layer: the device on a particle lattice
# --------------------------------------------------------------------------- #

def test_resolution_floor_shrinks_with_resolution(bench):
    """Discretisation error at R/dx = 6.7 is a few degrees and it falls off."""
    res = bench.resolution_check(dx_list=(0.003, 0.0015))
    floor = res["worst_abs_error_deg_by_resolution"]
    coarse = floor[str(bench.RADIUS / 0.003)]
    fine = floor[str(bench.RADIUS / 0.0015)]
    assert coarse < 5.0, coarse           # usable, but with a degree-level bar
    assert fine < coarse                  # and it is discretisation, not bias


def test_reading_with_too_few_envelope_points_is_nan(bench):
    """No fabricated angles: a shallow cap at coarse resolution has no reading."""
    x, y = bench.lattice_cap(30.0, 0.003)
    out = bench.measure_contact_angle(x, y, 0.45, 0.003)
    assert out["theta_deg"] != out["theta_deg"]        # nan
    assert out["model"] == "none"
    assert "envelope points" in out.get("reason", "")


# --------------------------------------------------------------------------- #
# 3. Physics layer: the shipped reading and its admissibility gates
# --------------------------------------------------------------------------- #

def test_physics_rows_are_in_degrees(report):
    """``error_deg`` used to subtract a radian set point from a degree reading.

    The shipped revision called ``relax(math.radians(t))`` and stored that
    radian value as ``theta_set_deg``, so ``error_deg = reading_deg - 1.047``
    was a unit salad and every row was structurally ungradable.  Degrees on
    both sides, and the identity must hold.
    """
    physics = report["physics"]
    assert [r["theta_set_deg"] for r in physics] == [60.0, 90.0, 120.0]
    for r in physics:
        assert r["error_deg"] == pytest.approx(
            r["theta_measured_deg"] - r["theta_set_deg"], abs=1e-9)
        assert r["within_gate"] is (abs(r["error_deg"]) <= report["gate_deg"])


def test_physics_readings_carry_their_admissibility_gates(report):
    """A reading is quoted only together with the run-level gates."""
    for r in report["physics"]:
        assert r["wall_penetration_dx"] >= 0.0
        assert r["spurious_current_ratio"] >= 0.0
        assert r["admissible"] is (
            r["wall_penetration_dx"] < report["physics"][0]["wall_penetration_gate_dx"]
            and r["spurious_current_ratio"] < report["physics"][0]["spurious_current_gate"]
        )


def test_shipped_physics_run_is_recorded_as_withheld(report):
    """Honest pin: at R/dx = 6.7 the shipped run is *not* admissible.

    Fluid particles reach below the wall by about one spacing, so the contact
    line is not where the fluid meets the wall and the angle cannot be graded
    either way.  The dev report states the same thing; if a future change makes
    the run admissible, this assertion fails and forces the record to be
    updated instead of the number being quietly reused.
    """
    physics = report["physics"]
    assert [r["admissible"] for r in physics] == [False, False, True]
    assert physics[0]["wall_penetration_dx"] > 0.5      # fluid below the wall
    # no row earns the gate, admissible or not: the BC does not reach the set
    # angle here, and the two inadmissible rows may not be read either way
    assert not any(r["within_gate"] for r in physics)
    assert all(abs(r["error_deg"]) > report["gate_deg"] for r in physics)


def test_doc_numbers_come_from_the_artifact(report):
    """The doc must not hard-code a claim the artifact contradicts.

    The shipped revision's generator wrote "instrument: 10/10 synthetic
    interfaces inside 2 deg" as a literal while ``all_within_2deg`` was false in
    the very same JSON.  The repaired generator injects the measured worst
    error; check that the number in the doc is the number in the report.
    """
    doc = (REPO / "docs" / "verification_contact_angle.md").read_text()
    worst = report["instrument"]["worst_abs_error_deg"]
    assert f"{worst:.3f}" in doc
    assert report["instrument"]["all_within_gate"] is True
    # the false sentence the shipped generator wrote as a literal, next to a
    # JSON in which 8 of the 10 rows were NaN
    assert "仪器层 10/10 合成界面全部在 2° 内" not in doc
