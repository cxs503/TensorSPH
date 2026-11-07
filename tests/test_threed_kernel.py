"""3-D kernel: normalisation, Laplacian limit, lattice bias, mass calibration.

Every assertion here is against an exact value that can be derived by hand --
``int W dV = 1``, ``int r^2 laplacian W dV = 6``, ``laplacian W(0) = -9 sigma/h^2``,
``C`` independent of ``dx`` -- never against a number recorded from a previous
run of this code.
"""

import math

import torch

from tensorsph.threed import (
    SUPPORT_FACTOR,
    calibrated_particle_mass,
    cubic_spline_gradient,
    cubic_spline_kernel,
    cubic_spline_laplacian,
    kernel_normalisation,
    lattice_kernel_sum,
    sigma3,
)


def test_kernel_integrates_to_one():
    """``4 pi int_0^{2h} W r^2 dr = 1`` for the 3-D cubic spline."""
    for h in (0.01, 0.013, 0.05):
        integral, _ = kernel_normalisation(h, n_r=8000)
        assert abs(integral - 1.0) < 1e-9


def test_kernel_is_compact_and_positive_inside_support():
    h = 0.02
    r = torch.linspace(0.0, SUPPORT_FACTOR * h, 500, dtype=torch.float64)
    w = cubic_spline_kernel(r, h)
    assert bool((w >= 0.0).all())
    assert float(w[0]) == sigma3(h)  # W(0) = 1/(pi h^3)
    outside = torch.tensor([2.0001 * h, 3.0 * h], dtype=torch.float64)
    assert float(cubic_spline_kernel(outside, h).abs().max()) == 0.0


def test_second_moment_of_the_laplacian_is_six():
    """``int r^2 laplacian W dV = 6`` -- the identity the operator relies on."""
    h = 0.013
    r = torch.linspace(0.0, SUPPORT_FACTOR * h, 400001, dtype=torch.float64)
    delta = torch.stack((r, torch.zeros_like(r), torch.zeros_like(r)), dim=-1)
    integral = float((4.0 * math.pi * cubic_spline_laplacian(delta, h)
                      * r.pow(4)).sum() * (r[1] - r[0]))
    assert abs(integral - 6.0) < 1e-6


def test_laplacian_at_the_origin_is_the_analytic_limit():
    """``f'(0) = 0`` must not turn ``-9 sigma/h^2`` into ``-3 sigma/h^2``."""
    h = 0.013
    value = float(cubic_spline_laplacian(torch.zeros(1, 3, dtype=torch.float64), h))
    assert abs(value - (-9.0 * sigma3(h) / (h * h))) < 1e-3 * abs(value)


def test_gradient_matches_finite_differences():
    h = 0.02
    radius = torch.tensor([[0.004, 0.0, 0.0], [0.017, 0.0, 0.0], [0.03, 0.0, 0.0]],
                          dtype=torch.float64)
    grad = cubic_spline_gradient(radius, h)
    for row, delta in enumerate((0.004, 0.017, 0.03)):
        step = 1e-7
        # float64 explicitly: torch.tensor(0.004) is float32 by default, and a
        # 1e-7 step on a float32 argument loses the whole difference.
        plus = torch.tensor(delta + step, dtype=torch.float64)
        minus = torch.tensor(delta - step, dtype=torch.float64)
        fd = (float(cubic_spline_kernel(plus, h))
              - float(cubic_spline_kernel(minus, h))) / (2 * step)
        assert abs(float(grad[row, 0]) - fd) < 1e-4 * abs(fd)


def test_lattice_bias_is_a_lattice_constant_not_a_resolution_effect():
    """``C = dx^3 sum W`` is a sum in units of ``h``: it cannot move with ``dx``."""
    values = [lattice_kernel_sum(dx, 1.3 * dx) for dx in (0.02, 0.01, 0.005)]
    assert max(values) - min(values) < 1e-14


def test_calibrated_mass_removes_the_lattice_bias():
    """After calibration a full-support particle sees ``rho = rho0`` to machine zero."""
    for h_over_dx in (1.0, 1.3, 1.5):
        dx, rho0 = 0.01, 1000.0
        h = h_over_dx * dx
        mass = calibrated_particle_mass(dx, h, rho0)
        residual = mass * lattice_kernel_sum(dx, h) / (rho0 * dx ** 3) - 1.0
        assert abs(residual) < 1e-14


def test_bias_of_the_default_spacing_is_the_documented_one():
    """The shipped 3-D default ``h = 1.3 dx`` is on the tension side, ``b < 0``."""
    bias = lattice_kernel_sum(0.01, 0.013) - 1.0
    assert -5e-3 < bias < 0.0
    # A negative bias is clamped to p = 0 by the Tait law, so it may not be cited
    # as a spurious pressure -- only as a density offset.
    assert bias < 0.0 if abs(bias) > 1e-6 else True
