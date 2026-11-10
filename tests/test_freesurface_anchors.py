"""Tests for the three analytic anchors and one fast end-to-end run each.

The analytic side is checked first: a reference that is wrong by a factor is
worse than no reference, and the 8/27 in the released-volume comparison is
exactly the kind of constant that is easy to get wrong (the whole fan is
``h0 c0 t``, the part downstream of the dam is ``8 h0 c0 t / 27``).

The simulations here are deliberately tiny (coarse lattice, a fraction of a
period) and marked ``slow``: they exist so the integration path cannot rot, not
to measure accuracy -- the accuracies come from the run recorded in
``docs/benchmarks/free-surface/report.json`` and are pinned in
``test_freesurface_regression.py``.
"""
import contextlib
import math

import pytest
import torch

from tensorsph import ritter
from tensorsph.freesurface import ritter_anchor, sloshing_mode, solitary_anchor


@contextlib.contextmanager
def few_threads(count: int = 4):
    """Cap torch's thread pool for the smoke runs.

    These lattices have a few hundred particles: at that size the per-step cost
    is dominated by synchronisation, and torch's default of one thread per core
    makes the run slower, not faster (measured on the 128-core reference box:
    15 s with four threads against >100 s with the default pool).
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(min(count, previous))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# --------------------------------------------------------------------------- #
# Analytic anchors
# --------------------------------------------------------------------------- #

def test_analytic_front_at_depth_sits_where_the_exact_depth_equals_dx():
    case = ritter_anchor.RitterCase(cells_per_depth=10.0)
    time, threshold = 0.25, case.dx
    x = ritter_anchor.analytic_front_at_depth(time, threshold, case)
    depth = float(ritter.depth(x, time, case.depth0, dam_x=case.column_width,
                               gravity=case.gravity))
    assert depth == pytest.approx(threshold, rel=1e-9)


def test_analytic_front_at_depth_is_behind_the_zero_depth_front():
    case = ritter_anchor.RitterCase()
    time = 0.5 * case.validity_window
    resolved = ritter_anchor.analytic_front_at_depth(time, case.dx, case)
    zero = case.column_width + 2.0 * case.wave_speed * time
    assert resolved < zero
    # and the gap is the part of the front that no lattice can hold
    travel = 2.0 * case.wave_speed * time
    assert 0.0 < (zero - resolved) / travel < 0.5


def test_released_downstream_volume_is_eight_twenty_sevenths_of_the_fan():
    """Integrate the exact fan downstream of the dam; must give 8 h0 c0 t / 27."""
    depth0, gravity, time, dam = 0.10, 9.81, 0.3, 0.5
    c0 = ritter.wave_speed(depth0, gravity)
    x = torch.linspace(dam, dam + 2.0 * c0 * time, 200001, dtype=torch.float64)
    profile = ritter.depth(x, time, depth0, dam_x=dam, gravity=gravity)
    volume = float(torch.trapezoid(profile, x))
    exact = 8.0 * depth0 * c0 * time / 27.0
    assert volume == pytest.approx(exact, rel=1e-6)
    # and it is not the fan total, which is 3.375x larger
    assert ritter.released_volume(time, depth0, gravity) == pytest.approx(
        volume * 27.0 / 8.0, rel=1e-6)


def test_ritter_case_timeline_stays_inside_the_validity_window():
    case = ritter_anchor.RitterCase()
    assert case.end_time < case.validity_window
    assert max(case.snapshot_times) <= case.end_time + 1e-12
    assert case.shallow_ratio == pytest.approx(case.depth0 / case.column_width)
    # the shallow-water anchor is only meaningful for a wide reservoir
    assert case.shallow_ratio <= 0.25


def test_ritter_case_sets_an_acoustically_stable_step():
    case = ritter_anchor.RitterCase(cells_per_depth=10.0)
    cfg = case.dam_break_config()
    h = cfg.h_factor * cfg.dx
    courant = cfg.dt * cfg.sound_speed / h
    assert courant <= cfg.max_courant
    # the step actually integrates the requested time
    assert cfg.steps * cfg.dt == pytest.approx(case.end_time, rel=0.02)


def test_sloshing_analytic_frequency_limits():
    g, length = 9.81, 0.60
    shallow = sloshing_mode.analytic_omega(length, 0.002 * length, 1, g)
    deep = sloshing_mode.analytic_omega(length, 20.0 * length, 1, g)
    # shallow limit: omega -> sqrt(g k^2 H) = (pi / L) sqrt(g H); note the pi
    # squared, i.e. sqrt(pi g H / L^2) as sometimes quoted is off by sqrt(pi)
    assert shallow == pytest.approx((math.pi / length) * math.sqrt(g * 0.002 * length),
                                    rel=5e-3)
    # deep limit: omega -> sqrt(pi g / L)
    assert deep == pytest.approx(math.sqrt(math.pi * g / length), rel=1e-4)
    # second mode is sqrt(2) times the first in deep water
    assert sloshing_mode.analytic_omega(length, 20 * length, 2, g) == pytest.approx(
        math.sqrt(2.0) * deep, rel=1e-6)


def test_sloshing_case_reports_its_compressibility_budget():
    case = sloshing_mode.SloshingModeCase()
    assert case.compressibility == pytest.approx(9.81 * case.water_depth
                                                 / case.sound_speed ** 2)
    assert case.compressibility < 1.0e-2       # below the 1 % level it reports


def test_solitary_case_length_scales_are_the_first_order_ones():
    case = solitary_anchor.SolitaryCase()
    alpha = math.sqrt(3.0 * case.height / (4.0 * case.depth ** 3))
    assert case.alpha == pytest.approx(alpha)
    assert case.wavelength == pytest.approx(2.0 * math.pi / alpha)
    assert case.celerity == pytest.approx(math.sqrt(9.81 * (case.depth + case.height)))
    # first-order validity: the shipped case sits at H/h = 0.25, the largest
    # ratio at which the crest is still two particle spacings tall at the
    # 8-cells-per-depth floor of the flume module
    assert 0.0 < case.height / case.depth <= 0.3


def test_solitary_crest_fit_refines_the_peak_off_the_sample_grid():
    times = [i * 0.01 for i in range(11)]
    signal = [math.exp(-((t - 0.047) / 0.05) ** 2) for t in times]
    height, when = solitary_anchor._crest(signal, times)
    assert when == pytest.approx(0.047, abs=3e-3)
    assert height == pytest.approx(1.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# Integration smoke runs (coarse, cheap)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_ritter_smoke_run_produces_a_shallow_water_consistent_state():
    case = ritter_anchor.RitterCase(cells_per_depth=5.0, viscosity=1e-4,
                                    time_fraction=0.12, sample_fractions=(0.12,),
                                    sound_speed_ratio=15.0, label="smoke")
    with few_threads():
        result = ritter_anchor.run_ritter(case)
    assert len(result.rows) == 1
    row = result.rows[0]
    # the water has to leave the reservoir and stay inside the tank
    assert row["released_volume"] > 0.0
    assert row["front_sim"] > case.column_width
    assert row["depth_l2"] < 0.6
    # mass: SPH conserves particle mass exactly, nothing may escape
    assert result.escaped == 0
    assert abs(result.mass_drift) < 1e-9 * 1e3
    # the released volume is already within 40 % of the exact value at this
    # resolution; a gross normalisation error would be far larger
    assert row["released_volume_error"] < 0.4


@pytest.mark.slow
def test_sloshing_smoke_run_oscillates_at_the_first_mode():
    case = sloshing_mode.SloshingModeCase(cells_per_depth=6.0, periods=1.0,
                                          amplitude=0.02, viscosity=1e-3,
                                          sound_speed_ratio=80.0, label="smoke")
    with few_threads():
        result = sloshing_mode.run_sloshing_mode(case)
    modal = result.modal
    assert len(modal) > 20
    assert max(modal) > 0.004          # it actually moves
    assert min(modal) < -0.0005        # and swings the other way
    # the released mode loses energy, it never gains any
    assert abs(modal[-1]) <= max(abs(v) for v in modal) + 1e-9
    assert result.escaped == 0


@pytest.mark.slow
def test_solitary_smoke_run_generates_a_crest_at_the_paddle():
    case = solitary_anchor.SolitaryCase(depth=0.10, height=0.02,
                                        cells_per_depth=10.0, flume_length=0.8,
                                        sponge_length=0.2, duration=0.12,
                                        gauge_positions=(0.15, 0.30, 0.45),
                                        viscosity=1e-4, xsph=0.2, label="smoke")
    with few_threads():
        result = solitary_anchor.run_solitary(case)
    summary = result.summary()
    assert summary["gauges"][0]["x"] == pytest.approx(0.15)
    assert summary["propagation_distance"] == pytest.approx(0.30)
    # Structural smoke only: at 0.12 s the crest has not yet reached the first
    # gauge (the Goring stroke lasts ~0.6 s), so this checks the plumbing --
    # finite records, a configured wavemaker, conserved mass -- and the accuracy
    # claims live in the recorded run and test_freesurface_regression.py.
    assert all(math.isfinite(v) for v in result.elevations[0])
    assert len(result.elevations) == 3
    assert result.paddle_stroke > 0.0
    assert summary["particles"] > 0
    assert summary["wavelength"] > summary["characteristic_width"]
    assert result.escaped == 0
