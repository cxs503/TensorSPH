"""Invariants of the 2-D drop capillary-oscillation reading.

The order parameter, the exact Rayleigh reference and the period estimator are
locked here so that the numbers in
``docs/xval_droplet_oscillation/README.md`` cannot drift silently.  Every
assertion is a property of the *formulation* (an exact law or an algebraic
identity), not a fit to whatever the implementation happens to produce; the one
measurement-based test uses a deliberately loose band and is labelled as such.

Reference: Rayleigh (1879); Lamb, *Hydrodynamics*, 6th ed., arts. 313-315.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.capillary_oscillation import (
    interior_pressure, make_droplet, mode_amplitude, period_from_crossings,
    rayleigh_2d, run_point,
)
from tensorsph.droplet_impact import Config, rhs, suggested_dt
from tensorsph.neighbor_search import PairListCache

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)   # keep the suite's wall time predictable

SIGMA = 0.072
RHO = 1000.0
RADIUS = 0.05


class TestRayleighReference:
    """The reference itself, independently anchored through its neutral mode."""

    def test_mode_two_frequency(self):
        omega = rayleigh_2d(SIGMA, RHO, RADIUS, 2)
        # omega^2 = 6 sigma / (rho R^3)
        assert omega == pytest.approx(math.sqrt(6.0 * SIGMA / (RHO * RADIUS ** 3)),
                                      rel=1e-12)
        assert 2.0 * math.pi / omega == pytest.approx(3.3798155633, rel=1e-9)

    def test_mode_one_is_neutral(self):
        """m = 1 is a pure translation: the relation must give zero frequency."""
        with pytest.raises(ValueError):
            rayleigh_2d(SIGMA, RHO, RADIUS, 1)

    def test_mode_scaling(self):
        """omega_m^2 ~ m (m^2-1) -- checked against the closed form for m=3, 4."""
        for m in (3, 4):
            assert (rayleigh_2d(SIGMA, RHO, RADIUS, m) ** 2
                    == pytest.approx(SIGMA * m * (m * m - 1) / (RHO * RADIUS ** 3),
                                     rel=1e-12))

    def test_frequency_scales_as_radius_pow_minus_three_halves(self):
        a = rayleigh_2d(SIGMA, RHO, RADIUS, 2)
        b = rayleigh_2d(SIGMA, RHO, 2.0 * RADIUS, 2)
        assert b / a == pytest.approx(2.0 ** -1.5, rel=1e-12)


class TestModeAmplitude:
    """``q2`` must be linear in the perturbation and vanish for a circle."""

    def test_zero_for_circle(self):
        pos, mass = make_droplet(RADIUS / 20.0, 0.0, "cpu", radius=RADIUS, mode=2)
        assert abs(mode_amplitude(pos, mass, 2)) < 2.0e-3

    def test_linear_in_perturbation(self):
        """Analytic value of the order parameter for ``r = R(1 + eps cos 2theta)``
        over a uniform mass distribution is ``2 eps / (1 + 3 eps^2)``.

        Measured on the lattice at ``R/dx = 40`` the ratio to the analytic value
        is 1.0096 (eps = 0.05), 0.9990 at R/dx = 80 and 1.0023 at R/dx = 160, so
        the estimator itself is exact and the residual is the boundary
        quantisation of the initial mask.  (eps = 0.02 at R/dx = 40 lands on an
        unlucky lattice alignment, ratio 0.891 -- which is why the assertion is
        made at the amplitude actually used in the benchmark, not at an
        arbitrary smaller one.)
        """
        eps = 0.05
        pos, mass = make_droplet(RADIUS / 40.0, eps, "cpu", radius=RADIUS, mode=2)
        expected = 2.0 * eps / (1.0 + 3.0 * eps ** 2)
        assert mode_amplitude(pos, mass, 2) == pytest.approx(expected, rel=0.05)

    def test_other_modes_see_nothing(self):
        """An m=2 perturbation must not leak into m=3 (zero by orthogonality) and
        only quadratically into m=4: analytically ``q4 = 1.5 eps^2``, i.e.
        ``q4 / q2 = 0.75 eps = 3.8 %``; measured 5.1 %.  A *linear* leak would
        show up as ``q4 / q2 ~ O(1)``, so the 10 % bound is the real assertion."""
        pos, mass = make_droplet(RADIUS / 30.0, 0.05, "cpu", radius=RADIUS, mode=2)
        q2 = mode_amplitude(pos, mass, 2)
        assert abs(mode_amplitude(pos, mass, 3)) < 1.0e-6
        assert abs(mode_amplitude(pos, mass, 4)) < 0.10 * abs(q2)


class TestPeriodEstimator:
    """The estimator must recover the period of a damped cosine exactly.

    This is the property the measurement relies on: the zero-crossing spacing is
    ``T/2`` no matter how strongly the amplitude decays, so numerical damping
    cannot bias the reported frequency.
    """

    def test_undamped(self):
        omega, n = 2.0 * math.pi / 3.0, 2000
        t = [k * 0.01 for k in range(n)]
        q = [0.1 * math.cos(omega * x) for x in t]
        period, crossings = period_from_crossings(t, q)
        assert period == pytest.approx(3.0, rel=2e-3)
        assert crossings >= 4

    def test_strong_damping_does_not_bias_period(self):
        omega, gamma, n = 2.0 * math.pi / 3.0, 1.5, 2000
        t = [k * 0.01 for k in range(n)]
        q = [0.1 * math.exp(-gamma * x) * math.cos(omega * x) for x in t]
        period, _ = period_from_crossings(t, q)
        assert period == pytest.approx(3.0, rel=2e-3)

    def test_no_crossings_reports_nan(self):
        t = [k * 0.01 for k in range(50)]
        q = [0.1 * math.exp(-0.1 * x) for x in t]
        period, crossings = period_from_crossings(t, q)
        assert math.isnan(period) and crossings == 0


class TestInteriorPressure:
    def test_selects_the_core_only(self):
        pos, mass = make_droplet(RADIUS / 20.0, 0.0, "cpu", radius=RADIUS, mode=2)
        r = torch.linalg.norm(pos, dim=1)
        pressure = 1.0 + r                       # synthetic linear field
        inside = r < 0.5 * RADIUS
        assert interior_pressure(pos, pressure, RADIUS) == pytest.approx(
            float(pressure[inside].mean()), rel=1e-12)


class TestSurfaceTensionProducesInwardForce:
    """Short integration: the CSF force must compress the drop, and sigma = 0
    must leave a relaxed cloud alone.  This is a *sign* test, not a gate."""

    @staticmethod
    def _cfg(dx: float, sigma: float) -> Config:
        cfg = Config(dx=dx, h_factor=1.3, rho0=RHO, sound_speed=0.5, gamma=7.0,
                     gravity=0.0, viscosity=1.0e-6,
                     pool_width=2.4 * RADIUS, pool_depth=2.4 * RADIUS,
                     surface_tension_coefficient=sigma, device="cpu",
                     boundary="clamp", adaptive_dt=True, max_courant=0.25,
                     density_calibration=True)
        cfg.dt = suggested_dt(cfg)
        return cfg

    def _radial_accel(self, sigma: float) -> float:
        dx = RADIUS / 10.0
        cfg = self._cfg(dx, sigma)
        pos, mass = make_droplet(dx, 0.05, "cpu", radius=RADIUS, mode=2)
        vel = torch.zeros_like(pos)
        h = cfg.h_factor * dx
        cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                              fluid_count=pos.shape[0])
        pairs, _ = cache.get(pos)
        _, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=cfg.dt)
        r = torch.linalg.norm(pos, dim=1).clamp_min(1e-12)
        return float((acc * (pos / r.unsqueeze(1))).sum(dim=1).mean())

    def test_force_is_inward_with_surface_tension(self):
        assert self._radial_accel(SIGMA) < 0.0

    def test_no_force_without_surface_tension(self):
        assert abs(self._radial_accel(0.0)) < 1.0e-3


class TestDynamicPoint:
    """Short dynamic point: the CSF force must do work against surface energy.

    Two robust consequences of a *dynamical* surface tension, both checked on one
    coarse run (R/dx = 8, 0.4 of the exact Rayleigh period):

    * the mode-2 order parameter must shrink by a measurable fraction -- the
      force compresses the interface, it does not merely exist pointwise;
    * the drop core must be *pressurised* -- the Laplace jump has to be carried
      by the EOS, not only by the force field.

    A crossing of ``q2`` (i.e. a full oscillation) is deliberately not asserted
    here: at this resolution the transient is strongly damped, and the
    period measurement is the science in
    ``benchmarks/surface_tension/droplet_oscillation.py``, not a unit test.
    """

    def test_force_compresses_and_pressurises_the_core(self):
        r = run_point(8, SIGMA, 0.4, sample_every=5)
        q0, q1 = r["series"]["q"][0], r["series"]["q"][-1]
        assert q0 > 0.0
        assert q1 < 0.7 * q0, "surface tension must compress the perturbed interface"
        assert r["p_interior_mean_pa"] > 0.25 * r["dp_exact_pa"], \
            "the Laplace jump must be carried by the EOS pressure"
        assert r["dp_exact_pa"] == pytest.approx(SIGMA / RADIUS, rel=1e-12)
        assert math.isfinite(r["omega_exact_rad_s"])
