"""Geometry precondition of ``CASE-HYDROSTATIC`` (audit finding F4, M15419).

``cell_centres`` fits ``floor(extent / dx)`` cells inside ``[0, extent]``, so a
tank declared as ``depth = 0.4 m`` at ``dx = 0.03 m`` is only 0.39 m deep in the
lattice while the analytic reference keeps ``H = 0.4 m``.  Every band then reads
low by the geometric ratio (2.5 % here) *and* the wall lattice (built at
``top = depth``) no longer sits on the fluid surface -- a configuration defect
that shows up as a solver error.  The board audit measured that config at 9.30 %
against 2.25 % for the representable ``depth = 0.39 m`` and asked for a guard.

These tests pin the guard:

* the *shipped* grid must be representable (an invariant of the case pool);
* a non-representable tank is refused with the numbers that make it
  non-representable (lattice top / reference bias), not with a generic error;
* ``allow_misaligned=True`` still runs it and records the bias, so the spurious
  FAIL stays reproducible and attributable;
* the lattice extent the guard reports is the one the solver actually builds.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_hydrostatic import (DEPTH, GEOMETRY_RTOL,  # noqa: E402
                                      HydrostaticGeometryError, check_geometry,
                                      run_case, run_initial_state)
from tensorsph.droplet_impact import cell_centres  # noqa: E402


def _harness():
    path = REPO / "benchmarks" / "case_hydrostatic" / "run.py"
    spec = importlib.util.spec_from_file_location("case_hydrostatic_harness_geom", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The shipped grid must pass its own precondition
# --------------------------------------------------------------------------- #

def test_every_shipped_config_is_representable_by_the_lattice():
    h = _harness()
    for name, overrides in h.CONFIGS.items():
        kw = h.spec(name)
        geometry = check_geometry(dx=kw["dx"], tank_width=kw["tank_width"],
                                  depth=DEPTH, bands=kw["bands"])
        assert geometry["aligned"] is True, (name, geometry["problems"])
        assert geometry["reference_bias_pct"] == pytest.approx(0.0, abs=1e-12), name
        assert math.isclose(geometry["lattice_depth"], DEPTH, rel_tol=0, abs_tol=1e-12)
        assert geometry["gated_bands"], name


# --------------------------------------------------------------------------- #
# Misaligned tanks are refused, with the arithmetic that makes them misaligned
# --------------------------------------------------------------------------- #

def test_misaligned_depth_is_refused_with_the_lattice_top_and_the_bias():
    with pytest.raises(HydrostaticGeometryError) as err:
        check_geometry(dx=0.03, tank_width=0.99, depth=0.4)
    message = str(err.value)
    # the lattice top and the reference bias are computed, not asserted blindly
    assert "0.39" in message
    assert f"{abs(1.0 - 0.39 / 0.4) * 100:.2f} %" in message


def test_misaligned_width_is_refused():
    with pytest.raises(HydrostaticGeometryError) as err:
        check_geometry(dx=0.03, tank_width=1.0, depth=0.39)
    message = str(err.value)
    assert "tank_width" in message and "0.99" in message


def test_both_extents_are_reported_at_once():
    """The audited dx = 0.03 / 1.0 x 0.4 tank violates both extents; the guard
    must name both rather than stop at the first."""
    with pytest.raises(HydrostaticGeometryError) as err:
        check_geometry(dx=0.03, tank_width=1.0, depth=0.4)
    assert "depth" in str(err.value) and "tank_width" in str(err.value)


def test_allow_misaligned_records_the_reference_bias_instead_of_raising():
    geometry = check_geometry(dx=0.03, tank_width=0.99, depth=0.4,
                              allow_misaligned=True)
    assert geometry["aligned"] is False
    assert geometry["problems"]
    assert geometry["lattice_depth"] == pytest.approx(0.39, abs=1e-12)
    assert geometry["reference_bias_pct"] == pytest.approx(-2.5, abs=1e-9)
    assert geometry["gated_bands"], "a misaligned tank is still measurable"


def test_band_thinner_than_one_cell_is_refused():
    with pytest.raises(HydrostaticGeometryError) as err:
        check_geometry(dx=0.02, tank_width=1.0, depth=0.4, bands=40)
    assert "thinner than one cell" in str(err.value)


def test_tank_shallower_than_the_kernel_support_is_refused():
    with pytest.raises(HydrostaticGeometryError) as err:
        check_geometry(dx=0.02, tank_width=0.1, depth=0.02, bands=2)
    assert "overlap rule" in str(err.value)


def test_non_positive_extent_is_refused():
    for kw in (dict(dx=0.0), dict(dx=0.02, depth=0.0),
               dict(dx=0.02, tank_width=-1.0)):
        with pytest.raises(HydrostaticGeometryError):
            check_geometry(**kw)


def test_alignment_tolerance_is_relative_not_exact_float_equality():
    """A representable tank can still be off by float dust (0.39 / 0.03 is
    13.000000000000002): the guard must accept it, not reject the arithmetic."""
    assert 0.39 / 0.03 != round(0.39 / 0.03)     # the float really is off
    geometry = check_geometry(dx=0.03, tank_width=0.99, depth=0.39)
    assert geometry["aligned"] is True
    assert geometry["rows"] == 13 and geometry["columns"] == 33
    # ... but the dust is far below the tolerance it is compared against
    assert abs(0.39 - 13 * 0.03) < GEOMETRY_RTOL * 0.39


# --------------------------------------------------------------------------- #
# The guard describes the lattice the solver actually builds
# --------------------------------------------------------------------------- #

def test_reported_lattice_extents_are_the_ones_the_solver_uses():
    dx, width, depth = 0.05, 0.5, 0.4
    geometry = check_geometry(dx=dx, tank_width=width, depth=depth, bands=5)
    record = run_initial_state(dx=dx, tank_width=width, depth=depth, bands=5)
    assert record["geometry"] == geometry
    assert geometry["columns"] * geometry["rows"] == record["config"]["fluid_particles"]
    # lattice top = last row centre + dx / 2, as built by cell_centres
    ys = cell_centres(depth, dx, torch.device("cpu"))
    xs = cell_centres(width, dx, torch.device("cpu"))
    # cell_centres returns float32, so compare at its own precision
    assert float(ys.max()) + dx / 2 == pytest.approx(geometry["lattice_depth"], abs=1e-6)
    assert float(xs.max()) + dx / 2 == pytest.approx(geometry["lattice_width"], abs=1e-6)
    assert geometry["reference_bias_pct"] == pytest.approx(0.0, abs=1e-12)


def test_case_runners_refuse_a_misaligned_tank_before_simulating():
    with pytest.raises(HydrostaticGeometryError):
        run_case(dx=0.03, tank_width=1.0, depth=0.4, seconds=0.05)
    with pytest.raises(HydrostaticGeometryError):
        run_initial_state(dx=0.03, tank_width=1.0, depth=0.4)


# --------------------------------------------------------------------------- #
# End to end: the audited false FAIL and its representable counterpart
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_misaligned_dx003_is_a_spurious_fail_and_the_guard_refuses_it():
    """Reproduce the audited configuration (dx = 0.03, 1.0 x 0.4 m, 2 s) with
    the guard switched off: the gate reads several times the geometric bias, so
    the FAIL cannot be blamed on the 2.5 % reference shift alone."""
    record = run_case(dx=0.03, tank_width=1.0, depth=0.4, seconds=2.0,
                      allow_misaligned=True)
    geometry, gate = record["geometry"], record["gate"]
    assert geometry["aligned"] is False
    assert geometry["reference_bias_pct"] == pytest.approx(-2.5, abs=1e-9)
    assert gate["max_band_error_pct"] > 5.0
    assert gate["max_band_error_pct"] > 2.0 * abs(geometry["reference_bias_pct"])
    assert gate["pass"] is False


@pytest.mark.slow
def test_representable_dx003_stays_within_three_percent():
    """Same dx, same particle count (33 x 13 = 429), representable geometry:
    the error drops to the coarse-resolution level instead of the spurious FAIL."""
    record = run_case(dx=0.03, tank_width=0.99, depth=0.39, seconds=2.0)
    assert record["geometry"]["aligned"] is True
    assert record["config"]["fluid_particles"] == 429
    assert record["gate"]["max_band_error_pct"] < 3.0