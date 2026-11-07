"""3-D shear-wave dynamics and the conservation meters.

The run is short (a few hundred steps at 1000 particles) so the test suite stays
usable: what is asserted is the *structure* of the answer -- momentum at
round-off, mass exactly constant, amplitude decaying with a rate of the right sign
and magnitude, kinetic energy falling twice as fast as the amplitude -- while the
tight numbers live in ``benchmarks/threed_shear/report.json`` and are guarded by
``tests/test_threed_artifacts.py``.
"""

import math
import os

import pytest
import torch

from tensorsph.threed import (
    analytic_mode_decay,
    centre_of_mass,
    internal_energy,
    kinetic_energy,
    momentum,
    relative_drift,
    run_shear_wave,
    shear_wave_field,
)


@pytest.fixture(autouse=True, scope="module")
def _modest_thread_count():
    """Cap BLAS threads: on a loaded 128-core host torch's default 128 threads make
    a 5 s run take a minute through oversubscription alone."""
    previous = torch.get_num_threads()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    yield
    torch.set_num_threads(previous)


def _short_run(**overrides):
    params = dict(n=10, box=1.0, h_factor=1.3, nu=0.05, amplitude=0.1,
                  sound_speed=10.0, steps=60, sample_every=5)
    params.update(overrides)
    return run_shear_wave(**params)


def test_linear_momentum_is_conserved_to_roundoff():
    """Periodic box, central pair forces: ``sum m v`` may only move at round-off."""
    result = _short_run()
    first = result.diagnostics[0]
    last = result.diagnostics[-1]
    assert result.momentum_scale > 0.0
    # The drift is compared with the momentum the flow *could* carry (sum m|v|),
    # never with the initial momentum: that one is zero by symmetry.
    assert last["momentum_magnitude"] < 1e-12 * result.momentum_scale
    assert abs(last["mass"] - first["mass"]) == 0.0


def test_shear_wave_amplitude_decays_near_the_analytic_rate():
    result = _short_run()
    analytic = analytic_mode_decay(0.05, 2.0 * math.pi)
    assert abs(result.fit_rate - analytic) / analytic < 0.40
    # The SPH viscosity operator under-predicts slightly at this resolution; the
    # sign of the residual is part of the record, so pin it.
    assert result.fit_rate < analytic


def test_kinetic_energy_falls_twice_as_fast_as_the_amplitude():
    """``E ~ a^2`` for a single Fourier mode, with no internal energy returned."""
    result = _short_run()
    kinetic = [d["kinetic_energy"] for d in result.diagnostics]
    amplitude = result.amplitudes
    ratio_ke = kinetic[-1] / kinetic[0]
    ratio_a2 = (amplitude[-1] / amplitude[0]) ** 2
    assert abs(ratio_ke - ratio_a2) < 0.1 * ratio_a2


def test_the_driver_rejects_an_unresolved_mode_and_an_unstable_step():
    with_error = 0
    try:
        run_shear_wave(n=6, box=1.0, h_factor=1.3, steps=1)
    except ValueError:
        with_error += 1
    try:
        run_shear_wave(n=10, box=1.0, h_factor=1.3, nu=1.0, dt=1.0, steps=1)
    except ValueError:
        with_error += 1
    assert with_error == 2


def test_initial_field_is_the_analytic_shear_wave():
    pos = torch.zeros((5, 3), dtype=torch.float64)
    pos[:, 1] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    vel = shear_wave_field(pos, 0.2, 2.0 * math.pi)
    expected = 0.2 * torch.sin(2.0 * math.pi * pos[:, 1]) * torch.sin(
        2.0 * math.pi * torch.zeros(5, dtype=torch.float64))
    assert float((vel[:, 0] - expected).abs().max()) < 1e-15
    assert float(vel[:, 1:].abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# conservation meters
# --------------------------------------------------------------------------- #
def test_circular_centre_of_mass_survives_a_boundary_crossing():
    mass = torch.ones(2, dtype=torch.float64)
    pos = torch.tensor([[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]], dtype=torch.float64)
    com = centre_of_mass(mass, pos, periodic_box=1.0)
    assert float(com[0]) < 0.05 or float(com[0]) > 0.95
    plain = centre_of_mass(mass, pos)
    assert abs(float(plain[0]) - 0.5) < 1e-12  # what the naive mean would say


def test_internal_energy_is_zero_at_rest_density_and_positive_when_compressed():
    rho0, c0, gamma = 1000.0, 20.0, 7.0
    at_rest = internal_energy(torch.tensor([rho0], dtype=torch.float64), rho0, c0, gamma)
    compressed = internal_energy(torch.tensor([rho0 * 1.01], dtype=torch.float64),
                                 rho0, c0, gamma)
    assert abs(float(at_rest)) < 1e-14
    assert float(compressed) > 0.0


def test_momentum_of_a_symmetric_pair_of_particles_is_zero():
    mass = torch.ones(2, dtype=torch.float64)
    vel = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    assert float(momentum(mass, vel).abs().max()) == 0.0
    assert abs(kinetic_energy(mass, vel) - 1.0) < 1e-15


def test_relative_drift_reports_the_denominator_it_used():
    row = relative_drift([0.0, 1e-12, -2e-12], scale=1.0)
    assert row["scale"] == 1.0
    assert row["delta"] == -2e-12
    assert row["relative_drift"] == -2e-12
    unbounded = relative_drift([0.0, 0.0])
    assert unbounded["relative_drift"] is None
