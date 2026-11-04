"""Regression gates for the Young-Laplace error-source decomposition.

The physics runs are far too slow for a unit test suite (they advance the SPH
drop for 1.5 capillary times); what is pinned here is (a) the *initialisation*
facts that the decomposition rests on -- measured directly, they take
milliseconds -- and (b) the numbers recorded in
``benchmarks/laplace_pressure_floor/report.json``, so that a silent change in
the pressure closure, the EOS or the measurement definitions cannot slip
through unnoticed.

The last test is ``xfail(strict=True)``: it states the acceptance criterion for
the still-open P0 (free-surface pressure closure, see
``docs/verification_laplace_pressure_floor.md`` §7).  When that work lands the
test turns into an ``xpass`` and the suite fails on purpose, forcing the
verification record to be updated instead of drifting.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

import importlib.util

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
REPORT = BENCH / "report.json"


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "laplace_pressure_floor_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lpf = _load_run_module()


@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py first")
    return json.loads(REPORT.read_text())


# --------------------------------------------------------------------------- #
# initialisation facts (C2 / C3)
# --------------------------------------------------------------------------- #
def test_hex_lattice_sum_density_bias_is_systematic():
    """h = 1.3 dx makes the discrete kernel sum 1 + b; b ~ 5.9e-3, not noise."""
    st = lpf.density_structure(dx=0.002, sound_speeds=[20.0])
    assert st["lattice_bias_rel_density"] == pytest.approx(0.00588, abs=2e-4)
    # A perfect lattice has no scatter: the offset is a *bias*, not noise.
    assert st["lattice_bias_std"] < 1e-9


def test_mass_calibration_removes_the_bias():
    _, _, _, _, bias, after = lpf.calibrated_droplet(dx=0.002)
    assert bias > 1e-3
    assert after == pytest.approx(1.0, abs=1e-12)


def test_spurious_initial_pressure_grows_as_sound_speed_squared():
    """p_init = B[(1+b)^gamma-1] with B ~ c0^2, so p_init(20)/p_init(5) ~ 16."""
    st = lpf.density_structure(dx=0.002, sound_speeds=[20.0, 5.0])
    p20 = st["per_sound_speed"]["20"]["initial_spurious_pressure_Pa"]
    p5 = st["per_sound_speed"]["5"]["initial_spurious_pressure_Pa"]
    assert p20 > 2000.0                     # 6.6 x the 360 Pa Laplace signal
    assert p20 / p5 == pytest.approx(16.0, rel=0.05)


def test_free_surface_density_deficit_and_its_resolution_dependence():
    """The deficit is what band-A differencing mistakes for a pressure."""
    coarse = lpf.density_structure(dx=0.002, sound_speeds=[20.0])
    fine = lpf.density_structure(dx=0.0015, sound_speeds=[20.0])
    for st in (coarse, fine):
        assert st["bands"]["0.85-0.95"]["mean_rel_density"] < -0.03
        assert st["bands"]["0.95-1.05"]["mean_rel_density"] < -0.20
    # finer resolution -> thinner deficit shell
    assert (fine["bands"]["0.85-0.95"]["mean_rel_density"]
            > coarse["bands"]["0.85-0.95"]["mean_rel_density"])


# --------------------------------------------------------------------------- #
# recorded evidence (C1 / C3 / C4)
# --------------------------------------------------------------------------- #
def test_report_is_complete_and_finite(report):
    assert report["runs"], "no runs recorded"
    for run in report["runs"]:
        assert run["finite"], run
        for key in ("definition_bandA", "definition_bandB", "definition_absolute",
                    "absolute_time_average_tail"):
            assert math.isfinite(run[key]["rel_error_pct"])
        assert run["spurious_current"]["vmax_over_vref_final"] < 0.1


def test_band_differencing_costs_tens_of_percentage_points(report):
    """C1: the historical p_core - p(shell) read-out is contaminated by C3."""
    for run in report["runs"]:
        gap = (run["definition_bandA"]["rel_error_pct"]
               - run["definition_absolute"]["rel_error_pct"])
        assert gap < -30.0, (run["dx_m"], run["sound_speed"], gap)


def test_compression_ratio_is_c0_and_resolution_independent(report):
    """C3: the interior compresses to a fixed fraction of sigma/R -- closure, not noise."""
    ratios = []
    for run in report["runs"]:
        r = ((run["density"]["rho_core_over_rho0"] - 1.0)
             / run["density"]["compression_required"])
        ratios.append(r)
        assert 0.40 < r < 0.80, (run["dx_m"], run["sound_speed"], r)
    assert max(ratios) - min(ratios) < 0.25


def test_residual_is_not_explained_by_the_eos_noise_floor(report):
    """C4 is falsified: at low c0 the predicted pressure noise is tiny while the
    residual stays put."""
    worst = []
    for run in report["runs"]:
        if run["sound_speed"] > 5.5:
            continue
        dp = report["reference"]["delta_p_Pa"]
        predicted = (report["reference"]["rho0"] * run["sound_speed"] ** 2
                     * run["density"]["rho_core_rel_std"]) / dp
        residual = abs(run["absolute_time_average_tail"]["rel_error_pct"]) / 100.0
        assert predicted < 0.15, (run["dx_m"], predicted)
        assert residual > 0.25, (run["dx_m"], residual)
        worst.append(residual - predicted)
    assert worst and min(worst) > 0.15


def test_open_loop_still_undershoots_by_about_a_third(report):
    """Record of the *open* (closure-off) configuration, kept as the baseline.

    This used to be a strict xfail asserting the 15 % criterion, i.e. it encoded
    the pre-closure failure.  The closure landed in the same mission that added
    the ladder (``Config.free_surface_pressure``,
    ``docs/verification_free_surface_pressure_closure.md``), so the failure fact
    is now pinned as a number instead: every open-loop case stays 30-40 % low.
    The acceptance gate moved to
    ``tests/test_free_surface_pressure.py::test_three_sweeps_meet_the_p0_criterion``.
    """
    assert report["runs"]
    for run in report["runs"]:
        err = run["absolute_time_average_tail"]["rel_error_pct"]
        assert err < -25.0, (run["dx_m"], run["sound_speed"], err)