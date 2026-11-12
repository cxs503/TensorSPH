"""Gates for the sloshing first-mode verification (``CASE-SPH-SLOSHING``).

Three layers, because the merged capability had a hole exactly between them:

1. **Reference layer** -- the closed form
   ``omega^2 = g k tanh(k H)`` against an independent Galerkin eigen-solve
   (:func:`benchmarks.sloshing_first_mode.run.variational_reference`).  The
   repository previously tested this formula against *itself*, which cannot
   catch a transcription error.
2. **Measurement layer** -- the modal projection :func:`modal_amplitude` must
   recover a known ``cos`` amplitude from a synthetic surface, and the surface
   quantisation model must bracket the analytic height.  Without this, a period
   error could hide inside the reading.
3. **Dynamics layer** (``slow``) -- the shipped WCSPH solver released from the
   first mode must oscillate at the linear natural period.  This is the claim
   ``tensorsph.sloshing``'s docstring has always made and never tested.

The dynamics test asserts an *interval*, not "it moved": the removed assertion
``eta_max - eta_min > 1e-6 m`` passed on free-surface roughening alone.
"""

import math

import numpy as np
import pytest
import torch

from tensorsph.sloshing import (SloshingConfig, make_water_block, run_case,
                                validate_case)
from benchmarks.sloshing_first_mode.run import (MODE, modal_amplitude,
                                               quantised_surface, read_period,
                                               run_columns, variational_reference)

# --------------------------------------------------------------------------- #
# 1. Reference layer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("H_over_L", [0.05, 0.4, 2.0])
def test_closed_form_matches_variational_eigenvalue_solve(H_over_L):
    """The tanh formula is confirmed by an independent discretisation."""
    L, H, g = 1.0, H_over_L, 9.81
    k = MODE * math.pi / L
    closed = math.sqrt(g * k * math.tanh(k * H))
    numer = variational_reference(L, H, g)
    assert numer == pytest.approx(closed, rel=1e-4), (
        f"H/L={H_over_L}: closed form {closed:.6f} vs variational {numer:.6f}")


def test_shallow_and_deep_limits_are_approached_from_the_right_side():
    """Dispersion limits -- a guard on the formula's direction of approach."""
    L, g = 1.0, 9.81
    k = MODE * math.pi / L
    shallow = math.sqrt(g * k * math.tanh(k * 0.01)) / (k * math.sqrt(g * 0.01))
    deep = math.sqrt(g * k * math.tanh(k * 20.0)) / math.sqrt(g * k)
    assert shallow == pytest.approx(1.0, abs=2e-3)
    assert deep == pytest.approx(1.0, abs=2e-4)


# --------------------------------------------------------------------------- #
# 2. Measurement layer
# --------------------------------------------------------------------------- #


def test_modal_projection_recovers_a_released_amplitude():
    """Projection of an exact cos surface returns the amplitude it carries."""
    dx, L, H, a = 0.01, 1.0, 0.4, 0.02
    cfg = SloshingConfig(tank_length=L, pool_width=L, water_depth=H,
                         initial_mode=1, initial_amplitude=a)
    x = run_columns(cfg)
    eta = [H + a * math.cos(math.pi * xi / L) for xi in x]
    # The recorded columns cover [dx/2, L - dx/2], so the projection drops the
    # two wall half-cells of the integral: for a pure cos mode that costs
    # ``(4/pi) dx / L`` of the amplitude (1.3 % here).  It shrinks with dx and
    # cancels between the measured and the quantised-reference release, so the
    # band is set by that truncation, not by the solver.
    loss = (4.0 / math.pi) * dx / L
    assert modal_amplitude(x, [eta], H, 1, L) == pytest.approx(a, rel=2.5 * loss)


def test_quantised_surface_brackets_the_analytic_height():
    """A column shows a height within one spacing below the true surface."""
    dx = 0.025
    eta = np.linspace(0.3, 0.5, 101)
    q = quantised_surface(eta, dx)
    assert np.all(q <= eta + 1e-12)
    assert np.all(q > eta - dx - 1e-12)
    # and it is on the lattice
    assert np.allclose((q / dx) * 2.0, np.round((q / dx) * 2.0))


def test_period_reader_measures_a_known_period():
    """zcr / fft / fit all recover a synthetic damped sinusoid's period."""
    t_ref = 1.23
    t = np.arange(0.0, 4.0 * t_ref, t_ref / 110.0)
    y = 0.014 * np.exp(-0.15 * t) * np.cos(2.0 * math.pi * t / t_ref + 0.3) + 0.4
    r = read_period(t, y)
    assert r["period"] == pytest.approx(t_ref, rel=1e-2)
    assert r["method_spread_pct"] < 2.0, r


# --------------------------------------------------------------------------- #
# 3. Release state (fast) and dynamics (slow)
# --------------------------------------------------------------------------- #


def test_release_state_is_the_mode_not_a_flat_surface():
    """make_water_block must actually tilt the free surface.

    The first implementation laid the lattice over ``crest = H + a``; because
    ``cell_centres`` floors the cell count, the top row sat a spacing below the
    crest and every column was clipped to the same height -- a flat surface one
    spacing too shallow, which cannot oscillate.
    """
    dx, H, a = 0.025, 0.4, 0.02
    cfg = SloshingConfig(tank_length=1.0, pool_width=1.0, tank_height=0.6,
                         water_depth=H, dx=dx, initial_mode=1,
                         initial_amplitude=a)
    pos, mass = make_water_block(cfg, torch.device("cpu"))
    prof = []
    for x_probe in run_columns(cfg):
        col = pos[(pos[:, 0] - x_probe).abs() < 0.5 * dx]
        prof.append(float(col[:, 1].max()) if col.numel() else -1.0)
    prof = np.asarray(prof)
    assert prof.min() > 0.0, "every column must hold fluid"
    span = prof.max() - prof.min()
    # Crest-to-trough of the mode is 2a; the column maximum can flatten it by at
    # most one spacing per column.
    assert span > 2.0 * a - 2.0 * dx, f"released surface is flat: span={span:.4f}"
    assert span <= 2.0 * a + 2.0 * dx
    assert float(mass.sum()) > 0.0


def test_recorded_profiles_have_one_column_per_cell():
    cfg = SloshingConfig(tank_length=1.0, pool_width=1.0, water_depth=0.4,
                         dx=0.025, initial_mode=1, initial_amplitude=0.02,
                         steps=1, output_every=1)
    run = run_case(cfg)
    assert run.profile_x is not None and len(run.profile_x) == 40
    assert run.surface_profiles and len(run.surface_profiles[0]) == 40


def test_forced_excitation_fails_loudly_instead_of_doing_nothing():
    """A silent ``pass`` used to integrate an unforced tank; it must raise now."""
    cfg = SloshingConfig(tank_length=1.0, pool_width=1.0, water_depth=0.4,
                         dx=0.05, steps=10, excitation_frequency=5.0)
    with pytest.raises(NotImplementedError):
        run_case(cfg)


def test_tank_length_must_match_pool_width():
    """wall_lattice/boundary_box span pool_width; a mismatch is silent."""
    cfg = SloshingConfig(tank_length=2.0, pool_width=1.0, water_depth=0.4)
    with pytest.raises(ValueError, match="pool_width"):
        validate_case(cfg)


def test_release_amplitude_is_bounded_by_linear_theory():
    cfg = SloshingConfig(tank_length=1.0, pool_width=1.0, water_depth=0.4,
                         dx=0.025, initial_mode=1, initial_amplitude=0.3)
    with pytest.raises(ValueError, match="amplitude"):
        validate_case(cfg)


@pytest.mark.slow
def test_released_first_mode_oscillates_at_the_linear_natural_period():
    """The dynamical claim: the solver's period matches linear theory.

    5 % band: the ladder in ``benchmarks/sloshing_first_mode`` measures
    +0.2 % at ``dx = 0.025``, so this is a regression envelope with room for
    platform variation, not a tuned number.
    """
    torch.set_num_threads(4)
    L, H, a, dx = 1.0, 0.4, 0.02, 0.025
    cfg = SloshingConfig(tank_length=L, pool_width=L, tank_height=0.6,
                         water_depth=H, dx=dx, initial_mode=1,
                         initial_amplitude=a, probe_locations=[L / 4],
                         device="cpu")
    h = cfg.h_factor * dx
    cfg.dt = cfg.max_courant * h / cfg.sound_speed
    t_end = 2.0 * cfg.natural_period(1)
    cfg.output_every = 25
    cfg.steps = int(math.ceil(t_end / cfg.dt))
    run = run_case(cfg)

    amp = modal_amplitude(run.profile_x, run.surface_profiles, H, 1, L)
    reading = read_period(np.asarray(run.times), amp)
    assert reading["period"] is not None
    err = reading["period"] / cfg.natural_period(1) - 1.0
    assert abs(err) < 0.05, f"period error {100 * err:+.2f} % (band 5 %)"
    assert run.escaped == 0
    assert run.max_density_compression < 0.03
    # The release amplitude must survive the projection: a run that starts from
    # a flat surface reads A_1(0) ~ 0 and cannot be measuring the mode.
    assert amp[0] > 0.4 * a, f"release amplitude {amp[0]:.4f} m vs {a} m"