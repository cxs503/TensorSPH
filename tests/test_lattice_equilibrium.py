"""Machine-checkable record for the lattice discrete-equilibrium offset.

The claim under test is narrow and quantitative: on a regular lattice the
summation density is ``(lattice mass) * sum_j W_ij``, so the offset is geometric
-- it depends on the lattice type and ``h/dx``, never on ``c0`` -- and the
calibrated mass ``rho0 / sum_j W`` removes it.  Each test below either measures
the offset from *real particles* (not from the helper it is validating) or
checks an acceptance ratio that a reviewer can recompute.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "lattice_equilibrium"
REPORT = BENCH / "report.json"


def _load():
    spec = importlib.util.spec_from_file_location("lattice_equilibrium_run",
                                                 BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


le = _load()


@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip("run benchmarks/lattice_equilibrium/run.py first")
    return json.loads(REPORT.read_text())


# --------------------------------------------------------------------------- #
# L1: the offset is geometric, and the helper reproduces it from particles
# --------------------------------------------------------------------------- #
def test_square_lattice_at_default_h_is_nearly_exact():
    """h = 1.3 dx on a square lattice is a lucky near-zero of the discrete sum."""
    bias = le.bias_curve([le.DEFAULT_H_FACTOR])[0]["square_bias"]
    assert bias == pytest.approx(-5.32e-5, abs=2e-6)


def test_hexagonal_lattice_carries_the_offset_the_backlog_quoted():
    """The 5.88e-3 of backlog_update_m2971 belongs to the hexagonal lattice."""
    bias = le.bias_curve([le.DEFAULT_H_FACTOR])[0]["hexagonal_bias"]
    assert bias == pytest.approx(5.88e-3, abs=2e-4)


def test_helper_reproduces_measured_particle_density():
    """``sum_j W`` must predict the density of real particles, not itself."""
    for hf in (1.2, le.DEFAULT_H_FACTOR, 1.5):
        predicted = le.bias_curve([hf])[0]["square_bias"]
        measured = le.production_state(hf, 20.0, calibrated=False)[
            "interior_rel_density_mean"]
        assert measured == pytest.approx(predicted, abs=2e-7), (hf, predicted, measured)


def test_offset_is_sound_speed_independent():
    """b is geometric: same value at c0 = 20 and c0 = 5 (only its price differs)."""
    a = le.production_state(1.5, 20.0, calibrated=False)
    b = le.production_state(1.5, 5.0, calibrated=False)
    assert a["interior_rel_density_mean"] == pytest.approx(
        b["interior_rel_density_mean"], abs=1e-9)


# --------------------------------------------------------------------------- #
# L2: its price through the Tait law, against the Laplace signal
# --------------------------------------------------------------------------- #
def test_trap_h_over_dx_injects_a_kilopascal():
    """h = 1.5 dx costs 1.39 kPa at c0 = 20 -- 3.9x a 20 mm drop's sigma/R."""
    p = le.spurious_pressure(le.bias_curve([le.TRAP_H_FACTOR])[0]["square_bias"], 20.0)
    assert p == pytest.approx(1391.0, rel=0.02)
    assert p > 3.0 * 360.0


def test_default_lattice_does_not_carry_the_kilopascal():
    """The backlog's universal 2.4 kPa claim does not hold for the square lattice."""
    p = le.spurious_pressure(le.bias_curve([le.DEFAULT_H_FACTOR])[0]["square_bias"], 20.0)
    assert p == 0.0                      # b < 0, clamped by the free-surface rule


# --------------------------------------------------------------------------- #
# L3: acceptance ratio of m2971 P1
# --------------------------------------------------------------------------- #
def test_calibration_meets_the_1e_minus_4_acceptance():
    for hf in (le.DEFAULT_H_FACTOR, le.TRAP_H_FACTOR):
        for c0 in (20.0, 5.0):
            run = le.production_state(hf, c0, calibrated=True)
            assert run["p_max_over_rho0c02"] < 1.0e-4, (hf, c0, run)
            assert run["interior_rel_density_max_abs"] < 1.0e-5, (hf, c0, run)


def test_calibration_is_a_no_op_at_the_default_smoothing():
    """At h = 1.3 dx the mass moves by 5.3e-5 relative, so nothing is re-baselined."""
    run = le.production_state(le.DEFAULT_H_FACTOR, 20.0, calibrated=True)
    ratio = run["mass_unit"] / run["mass_unit_uncorrected"]
    assert abs(ratio - 1.0) < 1.0e-4


# --------------------------------------------------------------------------- #
# L4: the artefact is real motion, and the calibration removes it
# --------------------------------------------------------------------------- #
def test_resting_pool_moves_without_calibration_and_is_still_with_it(report):
    rows = {(r["h_over_dx"], r["calibrated"]): r for r in report["dynamic"]}
    trap_raw = rows[(le.TRAP_H_FACTOR, False)]
    trap_cal = rows[(le.TRAP_H_FACTOR, True)]
    assert trap_raw["v_max_m_per_s"] > 0.05          # self-propelled at rest
    assert trap_cal["v_max_m_per_s"] < 1.0e-3        # 100x quieter
    assert trap_raw["v_max_m_per_s"] > 100.0 * trap_cal["v_max_m_per_s"]
    for row in report["dynamic"]:
        assert row["finite"]


# --------------------------------------------------------------------------- #
# L5: hydrostatic initialisation
# --------------------------------------------------------------------------- #
def test_hydrostatic_init_is_accurate_only_with_the_calibrated_unit():
    raw = le.hydrostatic_profile_error(le.DEFAULT_H_FACTOR, calibrated=False)
    cal = le.hydrostatic_profile_error(le.DEFAULT_H_FACTOR, calibrated=True)
    assert raw["rel_error_rms"] < 1.0e-3            # already good ...
    assert cal["rel_error_rms"] < 1.0e-6            # ... 80x better
    assert raw["rel_error_rms"] > 50.0 * cal["rel_error_rms"]


# --------------------------------------------------------------------------- #
# recorded evidence
# --------------------------------------------------------------------------- #
def test_report_is_complete(report):
    assert len(report["bias_curve"]) >= 6
    assert report["production"] and report["dynamic"]
    for row in report["bias_curve"]:
        assert math.isfinite(row["square_bias"]) and math.isfinite(row["hexagonal_bias"])
    assert report["figures"] and report["figures"][0]["name"].endswith(".png")
