"""Analytic gate for the diffusive term: a plane density wave decays at ``D k^2``.

Why this gate exists
--------------------
``density_diffusion_rate`` (the diffusive term of Antuono, Colagrossi & Marrone
2012, J. Comput. Phys. 231:946) is the only operator in ``rhs`` that was never
compared with a closed form.  ``tests/test_free_surface.py`` pins its *bookkeeping*
-- a uniform field is untouched, ``sum_i V_i drho_i/dt = 0``, the variance falls,
the pair and dense implementations agree -- but not a single number.  A term that
conserves volume and always smooths something would pass all of those tests after
any sign flip of the diffusion coefficient, a factor-of-two error, or a wrong
``h``.  This file supplies the missing *quantitative* reference: the exact decay
rate of a plane wave under a Laplacian.

The closed form
---------------
On fluid pairs the term is

    drho_i/dt = 2 delta c0 h * sum_j V_j (rho_i - rho_j) L_ij ,
    L_ij = -(r_ij . grad_i W_ij) / |r_ij|^2  > 0 ,

which is the second-derivative kernel estimate (Brookshaw 1985, J. Comput. Phys.
61:43): ``lap rho_i ~ 2 sum_j V_j (rho_i - rho_j) L_ij``.  Hence

    drho/dt = D lap rho ,      D = delta * c0 * h ,      h = h_factor * dx .

Notice that the diffusion coefficient is fixed by the model constants alone --
``delta``, the numerical sound speed ``c0 = 20 m/s`` and the smoothing length --
it is not a tuned parameter.  For the plane wave ``rho = rho0 (1 + A cos kx)``
this is exactly

    drho/dt = -gamma * rho0 A cos kx ,      gamma = D k^2 ,

so the two comparisons below are statements about the *continuum* operator with
no free constant in between.

Tolerance basis (why 3 %)
-------------------------
The measured residual of this gate is 0.3 % to 1.0 % (the full ladders are in
``benchmarks/density_diffusion_decay/decay_rate.json``).  It is *not* a defect of
the coefficient: at a fixed mode resolution the residual moves between 0.44 % and
0.96 % when nothing but ``h/dx`` changes (1.3 -> 1.6 -> 2.0), so what is left is
the lattice quadrature error of the kernel Laplacian -- the Brookshaw sum
reproduces ``lap rho`` with cell volumes only up to the second-moment consistency
of W on the chosen lattice -- plus a ``(k dx)^2`` symbol term that is 0.3 % at
``k dx = 0.25``.

The gate is 3 %: about 3x the largest measured residual, 5x below the 15 %
engineering acceptance bar of the platform, and still tight enough to reject every
structural error this term can carry -- dropping the factor 2 of the pair sum
(-50 %), passing ``dx`` where ``h = h_factor dx`` belongs (-23 %), halving
``delta`` (-50 %), or flipping the sign (+200 %).  The finite-time gate is 3.5 %:
same residual plus the first-order error of the explicit substep (measured
0.48 %).

What the gates do *not* cover
-----------------------------
The reference is the diffusion coefficient of the operator, not its usefulness
against B1/B2 (that is measured in ``docs/benchmarks/free-surface-treatment.md``).
The solver still obtains density by summation, so this term is a rate applied to
the summation density over one step, not the delta-SPH continuity equation.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_gradient,
    density_diffusion_rate,
)
from tensorsph.neighbor_search import cell_linked_pairs

DX = 0.01
WAVELENGTH = 0.5
NY = 20
AMPLITUDE = 1.0e-3
DELTA = 1.0
SOUND_SPEED = 20.0
H_FACTOR = 1.3
RHO0 = 1000.0
RTOL_RATE = 0.03
RTOL_FINITE_TIME = 0.035
DTYPE = torch.float64


@pytest.fixture(autouse=True)
def single_threaded_dispatch():
    """Cap torch at one intra-op thread for this module.

    Every call here is a few thousand pairs, i.e. dispatch-bound, not
    compute-bound: on the 128-thread reference host one 9.2k-pair call of
    ``density_diffusion_rate`` takes 33 ms with the default pool and 0.9 ms with
    one thread.  The time-stepping gate below needs a few hundred calls, so the
    thread count decides between 16 s and 0.2 s of wall clock for identical
    arithmetic.  The previous setting is restored afterwards.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def lattice(nx: int, ny: int, dx: float) -> torch.Tensor:
    """Square lattice of cell centres ``(i + 1/2) dx``."""
    xs = (torch.arange(nx, dtype=DTYPE) + 0.5) * dx
    ys = (torch.arange(ny, dtype=DTYPE) + 0.5) * dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)


def plane_wave_state(dx: float = DX, wavelength: float = WAVELENGTH, ny: int = NY,
                     amplitude: float = AMPLITUDE, delta: float = DELTA,
                     sound_speed: float = SOUND_SPEED,
                     min_interior_fraction: float = 0.5):
    """Lattice carrying ``rho = rho0 (1 + A cos kx)`` together with its pair geometry.

    The particle mass is ``rho0 dx^2``, so the volumes ``V_j = m_j / rho_j`` in the
    pair sum are the lattice cells and the discrete operator has no mass bias.
    """
    cfg = Config(dx=dx, h_factor=H_FACTOR, rho0=RHO0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=0.0, density_diffusion=delta,
                 device="cpu")
    h = cfg.h_factor * cfg.dx
    nx = int(round(wavelength / dx))
    pos = lattice(nx, ny, dx)
    k = 2.0 * math.pi / wavelength

    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    separation = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(separation, h)
    distance = torch.linalg.vector_norm(separation, dim=1)
    radial = -(separation * grad_w).sum(dim=1)

    rho = cfg.rho0 * (1.0 + amplitude * torch.cos(k * pos[:, 0]))
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, dtype=DTYPE)

    # Only particles with a complete kernel support see the interior operator: at
    # the domain rim the pair sum is truncated and the Laplacian is not the one
    # whose symbol the closed form describes.
    interior = ((pos[:, 0] > 2.0 * h) & (pos[:, 0] < nx * dx - 2.0 * h)
                & (pos[:, 1] > 2.0 * h) & (pos[:, 1] < ny * dx - 2.0 * h))
    assert int(interior.sum()) > min_interior_fraction * pos.shape[0], (
        "the domain is too small for a rim-free band"
    )
    return cfg, pos, rho, mass, (distance, radial, (i, j)), k, interior, amplitude


def theory_decay_rate(delta: float, sound_speed: float, h: float, k: float) -> float:
    """``gamma = D k^2`` with ``D = delta c0 h`` -- the closed form under test."""
    return delta * sound_speed * h * k * k


def cosine_amplitude(field: torch.Tensor, pos: torch.Tensor, k: float,
                     interior: torch.Tensor) -> float:
    """Coefficient of ``cos(kx)`` in a least-squares fit of ``[1, cos kx, sin kx]``.

    A plain ``sum(field cos kx) / sum(cos^2 kx)`` is only the amplitude when the
    band contains a whole number of half waves; the rim-free band does not, and
    the constant part ``rho0`` then leaks into the projection (measured: it made
    the projection -303 instead of 1).  Removing the constant and the sine term
    with a three-column fit makes the estimator exact for the field under test.
    """
    x = pos[interior, 0]
    basis = torch.stack((torch.ones_like(x), torch.cos(k * x), torch.sin(k * x)), dim=1)
    coefficients = torch.linalg.lstsq(basis, field[interior]).solution
    return float(coefficients[1])


def measured_decay_rate(cfg, pos, rho, mass, geometry, k, interior,
                        amplitude: float) -> float:
    """Decay rate of the mode amplitude implied by the operator, in 1/s.

    ``cosine_amplitude`` extracts the amplitude of the cosine mode from the rate
    field; dividing by ``rho0 A`` and negating turns it into ``gamma``.
    """
    rate = density_diffusion_rate(rho, mass, geometry[0], geometry[1],
                                  geometry[2], cfg)
    return -cosine_amplitude(rate, pos, k, interior) / (cfg.rho0 * amplitude)


def assert_rate_matches_theory(rtol: float = RTOL_RATE, **state_kw) -> tuple[float, float]:
    """Compare the operator's decay rate with ``D k^2``; returns (measured, theory)."""
    cfg, pos, rho, mass, geometry, k, interior, amplitude = plane_wave_state(**state_kw)
    measured = measured_decay_rate(cfg, pos, rho, mass, geometry, k, interior, amplitude)
    theory = theory_decay_rate(cfg.density_diffusion, cfg.sound_speed,
                               cfg.h_factor * cfg.dx, k)
    relative = abs(measured - theory) / theory
    assert relative < rtol, (
        f"plane-wave decay rate {measured:.6g} 1/s differs from the analytic "
        f"D k^2 = {theory:.6g} 1/s by {100.0 * relative:.2f} % (gate {100.0 * rtol:g} %)"
    )
    return measured, theory


def test_the_diffusive_term_decays_a_plane_wave_at_the_analytic_rate():
    """D = delta c0 h is fixed by the model constants, so gamma = D k^2 is a prediction."""
    cfg, *_ = plane_wave_state()
    measured, theory = assert_rate_matches_theory()
    # A second, independent spelling of the same gate: the ratio must sit on 1.
    assert measured == pytest.approx(theory, rel=RTOL_RATE)
    assert cfg.density_diffusion == DELTA


def test_the_decay_rate_scales_with_the_square_of_the_wavenumber():
    """Halving the wavelength must quadruple the rate: the operator is a Laplacian."""
    long_wave = assert_rate_matches_theory(wavelength=WAVELENGTH)[0]
    short_wave = assert_rate_matches_theory(wavelength=0.5 * WAVELENGTH)[0]
    ratio = short_wave / long_wave
    assert ratio == pytest.approx(4.0, rel=0.05), (
        f"rate ratio k -> 2k is {ratio:.4f}, not the Laplacian's 4"
    )


def test_the_mode_amplitude_follows_the_exponential_over_a_finite_time():
    """Explicit integration of the operator over T must give exp(-gamma T)."""
    # A coarser lattice than the rate gates: the explicit step is limited by the
    # stiffest lattice mode, dt < 2 / (D k_max^2) with k_max = pi/dx, and the cost
    # of this gate is the number of steps.  Here D = 0.52 m^2/s and the limit is
    # 1.6e-4 s, so dt = 1.2e-4 s sits 1.3x inside it; 100 steps reach T = 12 ms
    # over which the analytic amplitude falls by 22 % -- far outside the gate.
    dx, wavelength, ny = 0.02, 1.0, 12
    cfg, pos, rho, mass, geometry, k, interior, amplitude = plane_wave_state(
        dx=dx, wavelength=wavelength, ny=ny, min_interior_fraction=0.40
    )
    theory = theory_decay_rate(cfg.density_diffusion, cfg.sound_speed,
                               cfg.h_factor * cfg.dx, k)
    dt = 1.2e-4
    steps = 100

    initial = cosine_amplitude(rho, pos, k, interior) / (cfg.rho0 * amplitude)
    field = rho.clone()
    for _ in range(steps):
        rate = density_diffusion_rate(field, mass, geometry[0], geometry[1],
                                      geometry[2], cfg)
        field = field + dt * rate
    final = cosine_amplitude(field, pos, k, interior) / (cfg.rho0 * amplitude)

    expected = initial * math.exp(-theory * dt * steps)
    relative = abs(final - expected) / abs(expected)
    assert relative < RTOL_FINITE_TIME, (
        f"mode amplitude after {steps} x {dt:g} s is {final:.6g}, not "
        f"exp(-gamma T) = {expected:.6g} ({100.0 * relative:.2f} % off, "
        f"gate {100.0 * RTOL_FINITE_TIME:g} %)"
    )


def test_the_gate_is_red_when_the_reference_rate_is_perturbed(monkeypatch):
    """Negative control, shipped so the gate's sensitivity is self-evident.

    A 10 % error in the analytic rate -- the size of a factor-of-two mistake in
    ``h`` versus ``h/2`` read as an ``h`` error, or of dropping the ``2`` in the
    pair sum -- must be reported as a failure, not absorbed by the tolerance.
    """
    import sys

    module = sys.modules[__name__]
    for perturbation in (1.10, 0.90):
        monkeypatch.setattr(
            module, "theory_decay_rate",
            lambda delta, c0, h, k, f=perturbation: f * delta * c0 * h * k * k,
        )
        with pytest.raises(AssertionError, match="differs from the analytic"):
            assert_rate_matches_theory()
