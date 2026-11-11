"""Case-level tests: geometry, front tracking, sloshing reference, collisions."""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.wcsph import (LinearSloshing, front_at, front_positions,
                             lattice_box, natural_frequency, particle_pair)
from tensorsph.wcsph.cases import box_walls, relative_front_error


def test_lattice_box_is_cell_centred_and_counts_correctly():
    pts = lattice_box([0.0, 0.0], [0.05, 0.02], 0.01, 2)
    assert pts.shape == (10, 2)
    assert float(pts[:, 0].min()) == pytest.approx(0.005)
    assert float(pts[:, 1].max()) == pytest.approx(0.015)


def test_lattice_box_3d():
    pts = lattice_box([0.0] * 3, [0.02, 0.03, 0.01], 0.01, 3)
    assert pts.shape == (2 * 3 * 1, 3)


def test_box_walls_lie_outside_the_domain():
    lower, upper, dx = (0.0, 0.0), (0.1, 0.05), 0.01
    wall = box_walls(lower, upper, dx, 2, layers=2)
    assert wall.shape[0] > 0
    outside = ((wall[:, 0] < lower[0]) | (wall[:, 0] > upper[0])
               | (wall[:, 1] < lower[1]) | (wall[:, 1] > upper[1]))
    assert bool(outside.all())


def test_box_walls_3d_stay_outside():
    wall = box_walls([0.0] * 3, [0.02] * 3, 0.01, 3, layers=1)
    assert wall.shape[0] > 0
    assert float(wall.min()) < 0.0
    assert float(wall.max()) > 0.02


def test_front_positions_separate_the_toe_from_flying_particles():
    pos = torch.tensor([[0.10, 0.005], [0.20, 0.005], [0.35, 0.30]],
                       dtype=torch.float64)
    fronts = front_positions(pos, 0.01)
    assert fronts["front_max"] == pytest.approx(0.35)
    assert fronts["front_toe"] == pytest.approx(0.20)


def test_front_positions_handle_an_empty_set():
    fronts = front_positions(torch.zeros((0, 2), dtype=torch.float64), 0.01)
    assert fronts == {"front_max": 0.0, "front_toe": 0.0}


def test_front_at_interpolates_and_refuses_to_extrapolate():
    times = [0.0, 0.1, 0.2]
    front = [0.0, 1.0, 3.0]
    assert front_at(times, front, 0.05) == pytest.approx(0.5)
    assert front_at(times, front, 0.15) == pytest.approx(2.0)
    assert front_at(times, front, 0.3) is None


def test_relative_front_error_uses_the_column_edge():
    class Run:
        times = [0.0, 0.1, 0.2]
        front_toe = [0.2, 0.3, 0.4]

    rows = relative_front_error(Run(), [0.1], 0.2, z_of_t=lambda t: 0.5)["rows"]
    assert rows[0]["sim"] == pytest.approx(0.5)
    assert rows[0]["relative_error"] == pytest.approx(0.0)


def test_natural_frequency_matches_the_shallow_and_deep_limits():
    depth, length = 0.5, 1.0
    omega = natural_frequency(length, depth)
    k = math.pi / length
    assert omega == pytest.approx(math.sqrt(9.81 * k * math.tanh(k * depth)))
    shallow = natural_frequency(length, 1.0e-4)
    assert shallow == pytest.approx(math.sqrt(9.81 * k * k * 1.0e-4), rel=1e-2)


def test_linear_sloshing_has_zero_pressure_at_its_own_free_surface():
    """The reference must satisfy p = 0 on the surface it predicts.

    This is the check that the elevation and the pressure come from the same
    potential: evaluating the pressure at ``y = H + eta`` has to give zero to
    first order, for every x and t.
    """
    ref = LinearSloshing(length=1.0, depth=0.5, amplitude=0.01, frequency=0.5)
    for t in (0.3, 1.1, 2.7):
        eta = ref.wall_elevation(t)
        p = ref.pressure(ref.length, ref.depth + eta, t)
        scale = ref.rho0 * ref.gravity * abs(eta)
        assert abs(p) < 0.02 * scale


def test_linear_sloshing_pressure_is_hydrostatic_in_the_static_limit():
    ref = LinearSloshing(length=1.0, depth=0.5, amplitude=0.0, frequency=1.0)
    p = ref.pressure(1.0, 0.1, 0.0)
    assert p == pytest.approx(ref.rho0 * ref.gravity * (0.5 - 0.1))


def test_linear_sloshing_response_peaks_near_the_first_mode():
    a = LinearSloshing(length=1.0, depth=0.5, amplitude=0.01, frequency=0.5)
    b = LinearSloshing(length=1.0, depth=0.5, amplitude=0.01,
                       frequency=a.omega1 / (2 * math.pi))
    assert b.response_amplitude(periods=5.0) > 3.0 * a.response_amplitude(periods=1.0)


def test_linear_sloshing_resonant_limit_stays_finite():
    ref = LinearSloshing(length=1.0, depth=0.5, amplitude=0.01,
                         frequency=natural_frequency(1.0, 0.5) / (2 * math.pi))
    assert math.isfinite(ref.wall_elevation(1.0))


def test_particle_pair_conserves_momentum_and_energy():
    # cfl = 0.1 resolves the stiff head-on collision finely enough that the
    # energy error is the integrator's, not the step size's
    run = particle_pair(v0=0.5, c0=20.0, alpha_av=0.0, steps=800, cfl=0.1)
    assert run.momentum_drift < 1.0e-10
    assert run.energy_drift < 0.01
    # the pair must actually have interacted, not just flown apart
    assert min(run.separation) < 0.99 * run.separation[0]


def test_artificial_viscosity_dissipates_without_touching_momentum():
    elastic = particle_pair(v0=0.5, c0=20.0, alpha_av=0.0, steps=400)
    damped = particle_pair(v0=0.5, c0=20.0, alpha_av=0.2, steps=400)
    assert damped.momentum_drift < 1.0e-10
    assert damped.energy[-1] <= elastic.energy[-1] + 1.0e-12