"""Gates for the free-surface shell closure record (mission m3289).

``benchmarks/laplace_pressure_floor/closure_report.json`` is the measured A/B of
the Adami/Hu/Adams (2012) shell closure against the clamped Tait law, on the
Young-Laplace drop. The unit-level properties of the closure are pinned in
``tests/test_free_surface_pressure.py``; what is pinned here is the physics
record, so that a silent regression in the pressure assembly, in the measurement
definition or in the acceptance arithmetic cannot slip past the suite.

The historical decomposition of the *pre-closure* error sources stays where it
was: ``benchmarks/laplace_pressure_floor/report.json`` (closure off) is read by
``tests/test_laplace_pressure_floor.py`` and is deliberately not rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
RECORD = BENCH / "closure_report.json"


@pytest.fixture(scope="module")
def record():
    if not RECORD.exists():
        pytest.skip("run outputs/m3289_make_report.py first")
    return json.loads(RECORD.read_text())


def test_the_record_has_the_full_cross(record):
    points = record["points"]
    assert len(points) == 6
    assert {(p["dx_m"], p["sound_speed"]) for p in points} == {
        (0.002, 20.0), (0.002, 10.0), (0.002, 5.0),
        (0.0015, 20.0), (0.0015, 10.0), (0.0015, 5.0)}
    assert record["reference"]["delta_p_Pa"] == pytest.approx(360.0)


def test_the_closure_improves_every_point_by_more_than_20_points(record):
    """The defect was a closure defect, so no corner may be left behind."""
    for p in record["points"]:
        assert p["improvement_pt"] >= 20.0, (p["dx_m"], p["sound_speed"])
        assert abs(p["closure_on"]["rel_error_pct"]) < abs(
            p["closure_off"]["rel_error_pct"])


def test_the_reference_resolution_is_inside_the_15pct_bar(record):
    """dx = 0.002 (the resolution the drop benchmark uses) at every c0."""
    coarse = [p for p in record["points"] if p["dx_m"] == 0.002]
    assert len(coarse) == 3
    for p in coarse:
        assert abs(p["closure_on"]["rel_error_pct"]) < 15.0, p["sound_speed"]


def test_the_compression_ratio_moves_towards_the_required_value(record):
    """Closure means the interface can hold the jump, so the interior must not
    stay at the 0.57 fraction of ``sigma/R`` compression."""
    for p in record["points"]:
        assert p["closure_on"]["compression_ratio"] > p["closure_off"]["compression_ratio"]
        assert p["closure_off"]["compression_ratio"] == pytest.approx(0.57, abs=0.15)


def test_the_closure_does_not_excite_the_drop(record):
    """A closure that inflates the surface would show up as spurious motion.

    The gate is Adami et al.'s: |v|_max / sqrt(sigma / (rho0 R)) < 0.1 for a drop
    that is supposed to be at rest.
    """
    for p in record["points"]:
        assert p["closure_on"]["vmax_over_vref"] < 0.1, (p["dx_m"], p["sound_speed"])


def test_at_most_one_point_misses_the_15pct_bar_and_it_is_the_fine_corner(record):
    """Honest bookkeeping of the open item, pinned so it cannot drift silently.

    ``dx = 0.0015, c0 = 20`` measured -15.1 %: it still improves by 23 pt over the
    clamped law, but it misses the bar. The assertion is two-sided on purpose --
    if the corner is fixed the test fails and the record must be updated, which
    is the same "no silent drift" contract the xfail in
    ``tests/test_laplace_pressure_floor.py`` uses.
    """
    misses = [(p["dx_m"], p["sound_speed"], p["closure_on"]["rel_error_pct"])
              for p in record["points"]
              if abs(p["closure_on"]["rel_error_pct"]) >= 15.0]
    assert misses == [(0.0015, 20.0, pytest.approx(-15.1, abs=0.5))], misses


def test_the_figure_exists(record):
    for name in record["figures"]:
        path = BENCH / name
        assert path.exists() and path.stat().st_size > 10_000