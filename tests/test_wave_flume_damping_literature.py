"""Independent re-derivation of the laminar-bottom-boundary-layer damping anchor.

``tensorsph.wave_flume_grade.laminar_boundary_layer_damping`` returns ``mu`` from
a closed form that is transcribed from the literature.  A transcription of a
closed form can only be checked against something that is *not* the same closed
form, so this file rebuilds ``mu`` from the physics:

1. solve the Airy dispersion relation by its own Newton iteration,
2. write the first-order Stokes boundary-layer velocity field
   ``u(z,t) = u_b [cos(sigma t) - exp(-z/delta) cos(sigma t - z/delta)]``,
   ``delta = sqrt(2 nu / sigma)``, ``u_b = a sigma / sinh(kh)``,
3. form the gradient of that field with **finite differences on the grid**
   (not the analytic derivative, which would re-introduce the algebra under
   test) and integrate ``nu <(du/dz)^2>`` over the depth by the trapezoid rule,
4. convert the dissipation ``D`` into a spatial decay of amplitude with the wave
   energy balance ``dF/dx = -D``, ``F = (1/2) rho g a^2 c_g``, ``da/dx = -mu a``
   -> ``mu = D / (rho g a^2 c_g)``.

The interesting number is the ``sqrt(2)``: the dissipation per unit bed area is
written both as ``(1/2) rho u_b^2 sqrt(nu sigma / 2)`` (dissipation integral) and
as ``(1/4) rho u_b^2 sqrt(2 nu sigma)`` (wall shear times velocity); they are
the same value, but a closed form that keeps ``sqrt(nu sigma)`` and the ``1/4``
is ``sqrt(2)`` too small.  ``test_missing_sqrt_two_form_is_forty_one_percent_low``
pins that, so a revert to the understated anchor fails here rather than quietly
moving published numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorsph.wave_flume_grade import laminar_boundary_layer_damping

GRAVITY = 9.81
NU = 1.0e-6


def _dispersion(period: float, depth: float, gravity: float = GRAVITY) -> dict:
    """Airy dispersion solved independently of the module under test."""
    sigma = 2.0 * math.pi / period
    k = sigma ** 2 / gravity
    for _ in range(200):
        kh = k * depth
        f = gravity * k * math.tanh(kh) - sigma ** 2
        df = gravity * (math.tanh(kh) + kh / math.cosh(kh) ** 2)
        step = f / df
        k -= step
        if abs(step) <= 1e-15 * k:
            break
    kh = k * depth
    celerity = sigma / k
    group = 0.5 * celerity * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    return {"sigma": sigma, "k": k, "kh": kh, "celerity": celerity,
            "group_velocity": group, "sinh_kh": math.sinh(kh)}


def _numeric_mu(period: float, depth: float, amplitude: float = 1.0,
                nu: float = NU, gravity: float = GRAVITY,
                z_points: int = 6001, t_points: int = 400) -> float:
    """``mu`` [1/m] from a numerical quadrature of the Stokes boundary layer."""
    d = _dispersion(period, depth, gravity)
    sigma = d["sigma"]
    u_bed = amplitude * sigma / d["sinh_kh"]
    delta = math.sqrt(2.0 * nu / sigma)
    z = np.linspace(0.0, 10.0 * delta, z_points)
    times = np.linspace(0.0, 2.0 * math.pi / sigma, t_points, endpoint=False)
    mean_gradient_square = np.zeros_like(z)
    for t in times:
        u = u_bed * (np.cos(sigma * t)
                     - np.exp(-z / delta) * np.cos(sigma * t - z / delta))
        du_dz = np.gradient(u, z)
        mean_gradient_square += du_dz ** 2 / t_points
    dissipation_per_density = nu * float(np.trapezoid(mean_gradient_square, z))
    # dF/dx = -D with F = (1/2) rho g a^2 c_g and da/dx = -mu a.
    return dissipation_per_density / (gravity * amplitude ** 2 * d["group_velocity"])


@pytest.mark.parametrize("period,depth", [(1.0, 0.25), (0.8, 0.15), (2.0, 1.0)])
def test_closed_form_matches_numerical_boundary_layer_dissipation(period, depth):
    numeric = _numeric_mu(period, depth)
    closed = laminar_boundary_layer_damping(period, depth, GRAVITY, NU)["mu_per_m"]
    assert closed == pytest.approx(numeric, rel=1e-3), (
        f"T={period}s h={depth}m: closed form {closed:.6e} 1/m vs numerical "
        f"Stokes-layer quadrature {numeric:.6e} 1/m")


@pytest.mark.parametrize("period,depth", [(1.0, 0.25), (2.0, 1.0)])
def test_missing_sqrt_two_form_is_forty_one_percent_low(period, depth):
    """The classic transcription slip: D = (1/4) rho u_b^2 sqrt(nu sigma).

    Keeping ``sqrt(nu sigma)`` where the dissipation integral gives
    ``(1/2) rho u_b^2 sqrt(nu sigma / 2)`` loses a factor ``sqrt(2)``.  The
    numerical layer and the module must both sit at the larger value.
    """
    d = _dispersion(period, depth)
    sigma, group, sinh_kh = d["sigma"], d["group_velocity"], d["sinh_kh"]
    understated = (sigma ** 2 * math.sqrt(NU * sigma)
                   / (4.0 * GRAVITY * group * sinh_kh ** 2))
    stated = laminar_boundary_layer_damping(period, depth, GRAVITY, NU)["mu_per_m"]
    numeric = _numeric_mu(period, depth)
    assert stated / understated == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert numeric / understated == pytest.approx(math.sqrt(2.0), rel=2e-3)
    assert stated > understated


def test_physical_budget_over_a_lab_flume_is_well_inside_the_ten_percent_gate():
    """A 1 s wave in 0.25 m of water loses <0.5 % of its height over 2 m."""
    mu = laminar_boundary_layer_damping(1.0, 0.25, GRAVITY, NU)["mu_per_m"]
    assert mu == pytest.approx(1.65e-3, rel=2e-3)   # independent derivation above
    assert 1.0 - math.exp(-mu * 2.0) < 5.0e-3


def test_damping_scales_with_sqrt_of_viscosity_and_with_bed_velocity_squared():
    """``mu`` must be linear in ``sqrt(nu)`` and in ``(a sigma / sinh kh)^2``."""
    base = laminar_boundary_layer_damping(1.0, 0.25, GRAVITY, NU)["mu_per_m"]
    four_nu = laminar_boundary_layer_damping(1.0, 0.25, GRAVITY, 4.0 * NU)["mu_per_m"]
    assert four_nu == pytest.approx(2.0 * base, rel=1e-12)
    # Same sigma, same h, same k: quadrupling the viscosity is the only change.
    assert _numeric_mu(1.0, 0.25, nu=4.0 * NU) == pytest.approx(
        2.0 * _numeric_mu(1.0, 0.25), rel=2e-3)