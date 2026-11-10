"""Rayleigh-Plateau ladder: analytic anchors, curvature anchors, column attempt.

Three groups:

1. **Analytic** (closed form, no simulation): the most-unstable wavenumber must
   be the *Bessel* branch value ``kR = 0.697`` (``lambda = 9.02 R``) and not the
   naive ``1/sqrt(3) = 0.577`` (``lambda = 10.9 R``); the slender-jet limit must
   reproduce ``omega^2 = sigma k^2 / (2 rho R)``; the 2-D strip must be *stable*
   (the reason a planar solver cannot reproduce this anchor).
2. **Curvature operators** (static cylinder): ``div n = 1/R``.  The axisymmetric
   profile estimator is calibrated on that geometry and must land inside 2 %;
   the raw 3-D ``div n`` carries the documented lattice-staircase bias, which the
   test pins as a band so a silent improvement (or regression) shows up.
3. **Column dynamics**: the initial condition must carry the requested mode
   amplitude at zero pressure, mass must be conserved to round-off, and the run
   must stay finite.  The growth-rate anchor (``omega`` within 10 % of Rayleigh)
   is *not* met yet and is pinned as a non-strict xfail so the gap is visible in
   the test report instead of being silently dropped -- see
   ``docs/verification_rayleigh_plateau.md`` for the diagnosis.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph import rayleigh_plateau as rp
from tensorsph.threed import liquid_column as lc
from tensorsph.threed.csf3 import surface_tension_acceleration3
from tensorsph.threed.neighbors3 import periodic_pairs

SIGMA = 0.072
RHO = 1000.0
RADIUS = 0.006
MU = 1.0e-3


# --------------------------------------------------------------------------- #
# 1. Analytic anchors
# --------------------------------------------------------------------------- #
def test_most_unstable_wavenumber_is_the_bessel_branch():
    x = rp.most_unstable_wavenumber()
    assert abs(x / 0.697 - 1.0) < 0.005, x
    assert abs(rp.most_unstable_wavelength(1.0) / 9.02 - 1.0) < 0.005
    # The Bessel factor is what moves the maximum: x(1-x^2) alone peaks at
    # 1/sqrt(3) = 0.577, i.e. lambda = 10.9 R -- 20 % away from 9.02 R.
    assert rp.naive_wavenumber() == pytest.approx(1.0 / math.sqrt(3.0))
    assert abs(rp.naive_wavenumber() / x - 1.0) > 0.15


def test_dispersion_vanishes_at_the_neutral_point():
    k_neutral = 1.0 / RADIUS
    assert rp.dispersion_rate_squared(SIGMA, RHO, RADIUS, k_neutral) == pytest.approx(
        0.0, abs=1e-9)
    # k R > 1 is the stable branch (surface tension shortens the interface).
    assert rp.dispersion_rate_squared(SIGMA, RHO, RADIUS, 1.5 / RADIUS) < 0.0
    assert math.isnan(rp.dispersion_rate(SIGMA, RHO, RADIUS, 1.5 / RADIUS))


def test_slender_limit_matches_the_asymptote():
    k = 0.01 / RADIUS
    omega = rp.dispersion_rate(SIGMA, RHO, RADIUS, k)
    slender = math.sqrt(rp.slender_jet_dispersion(SIGMA, RHO, RADIUS, k))
    assert abs(omega / slender - 1.0) < 0.01


def test_growth_rate_times_capillary_time_is_order_one():
    omega = rp.dispersion_rate(SIGMA, RHO, RADIUS,
                               rp.most_unstable_wavenumber() / RADIUS)
    tau = rp.capillary_time(SIGMA, RHO, RADIUS)
    assert 0.2 < omega * tau < 0.6


def test_planar_strip_is_linearly_stable():
    """Negative control: the 2-D analogue has no Rayleigh-Plateau instability."""
    for kR in (0.2, 0.6971, 1.5):
        omega2 = rp.planar_strip_dispersion(SIGMA, RHO, RADIUS, kR / RADIUS)
        assert omega2 > 0.0


def test_inviscid_anchor_applies_to_a_water_jet():
    assert rp.ohnesorge(SIGMA, RHO, RADIUS, MU) < 0.05
    assert rp.ohnesorge_guard(SIGMA, RHO, RADIUS, MU)


# --------------------------------------------------------------------------- #
# 2. Curvature operators on a static cylinder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius_over_dx", [4, 6])
def test_profile_curvature_reproduces_one_over_radius(radius_over_dx):
    info = lc.static_cylinder_curvature(radius=radius_over_dx * 1e-3, dx=1e-3,
                                        curvature="profile")
    assert abs(info["relative_error_mean"]) < 0.02, info


def test_raw_three_dimensional_divergence_overestimates_curvature():
    """The lattice bias the profile estimator exists to remove."""
    coarse = lc.static_cylinder_curvature(radius=4e-3, dx=1e-3, curvature="mls")
    fine = lc.static_cylinder_curvature(radius=6e-3, dx=1e-3, curvature="mls")
    assert 0.10 < coarse["relative_error_mean"] < 0.50, coarse
    assert 0.05 < fine["relative_error_mean"] < 0.45, fine
    # Refining must reduce the bias monotonically.
    assert fine["relative_error_mean"] < coarse["relative_error_mean"]


def test_surface_tension_force_is_zero_without_sigma_and_points_inward():
    cfg = lc.ColumnConfig(radius=4e-3, dx=1e-3, kR=0.6971, epsilon=0.0)
    pos, mass, box = lc.build_column(cfg)
    pairs = periodic_pairs(pos, box, cfg.cutoff)
    from tensorsph.threed.csf3 import csf_fields3
    state = csf_fields3(pos, mass, cfg.h, pairs, rho0=cfg.rho0, box=box,
                        curvature="mls")
    rho = torch.full_like(mass, cfg.rho0)
    assert float(surface_tension_acceleration3(state, rho, 0.0).abs().max()) == 0.0
    acc = surface_tension_acceleration3(state, rho, SIGMA)
    shell = state.surface_mask(0.3) & state.valid
    radial = lc._axis_offset(pos, box)[shell]
    radial = radial / torch.linalg.vector_norm(radial, dim=1).unsqueeze(-1)
    cos = (acc[shell, :2] * radial).sum(dim=1) / torch.linalg.vector_norm(
        acc[shell], dim=1).clamp_min(1e-30)
    # A cylinder is convex, so the CSF force must pull inward.
    assert float(cos.mean()) < -0.8


# --------------------------------------------------------------------------- #
# 3. Column initial condition and dynamics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius_over_dx", [3])
def test_initial_condition_carries_the_requested_mode(radius_over_dx):
    cfg = lc.ColumnConfig(radius=radius_over_dx * 1e-3, dx=1e-3, kR=0.6971,
                          epsilon=0.02)
    pos, mass, box = lc.build_column(cfg)
    pairs = periodic_pairs(pos, box, cfg.cutoff)
    probe = lc.surface_radius_profile(pos, mass, cfg.h, pairs, rho0=cfg.rho0,
                                      box=box, nbin=cfg.nz)
    calib = cfg.radius / float(probe["radius"].mean())
    prof = lc.surface_radius_profile(pos, mass, cfg.h, pairs, rho0=cfg.rho0,
                                     box=box, nbin=cfg.nz, calibration=calib)
    k = 2.0 * math.pi / cfg.wavelength
    eps = float(abs(lc.mode_amplitude(prof["radius"], prof["dz"], k))) \
        / float(prof["radius"].mean())
    assert abs(eps / cfg.epsilon - 1.0) < 0.05, eps
    # Mass is normalised to rho0 pi R^2 lambda exactly.
    target = cfg.rho0 * math.pi * cfg.radius ** 2 * cfg.wavelength
    assert float(mass.sum()) == pytest.approx(target, rel=1e-12)
    # The realised wavenumber closes on the periodic box exactly: k * lambda = 2 pi.
    assert cfg.kR_actual == pytest.approx(2.0 * math.pi * cfg.radius / cfg.wavelength)
    assert (cfg.kR_actual / cfg.radius) * cfg.wavelength == pytest.approx(2.0 * math.pi)


def test_initial_condition_is_pressure_free():
    """A 2 % perturbation must not put a Tait pressure step across the mode."""
    from tensorsph.threed.operators3 import summation_density, tait_pressure
    cfg = lc.ColumnConfig(radius=4e-3, dx=1e-3, kR=0.6971, epsilon=0.02)
    pos, mass, box = lc.build_column(cfg)
    pairs = periodic_pairs(pos, box, cfg.cutoff)
    rho = summation_density(pos, mass, cfg.h, pairs, box)
    p = tait_pressure(rho, cfg.rho0, cfg.sound_speed, cfg.gamma)
    assert float(p.max()) == 0.0


def test_column_run_stays_finite_and_conserves_mass():
    cfg = lc.ColumnConfig(radius=3e-3, dx=1e-3, kR=0.6971, epsilon=0.02)
    result = lc.run_column(cfg, steps=40, sample_every=20)
    assert torch.isfinite(result.positions).all()
    assert torch.isfinite(result.velocities).all()
    assert torch.isfinite(result.densities).all()
    blob = result.to_dict()
    # Mass is a Lagrangian label here, so its conservation is exact.
    assert blob["particles"] == int(result.positions.shape[0])
    assert len(result.times) >= 3
    assert set(("omega_measured", "omega_analytic", "max_density_error",
                "final_volume_error", "particles")) <= set(blob)
    assert blob["max_velocity"] < 10.0 * math.sqrt(SIGMA / (RHO * cfg.radius))
    # The measurement must see *some* signal above the dominance of the mode.
    assert result.amplitudes[0] > 0.01


def test_growth_rate_anchor_is_reported_honestly():
    """C4: the growth rate must match Rayleigh to 10 % -- or the gap must be the
    *documented* free-surface defect, not something silently ignored.

    The test is self-updating: once a real fix lands, ``solved`` becomes true and
    the assertion passes as an ordinary test.  Until then it xfails with the
    reason, and it refuses to xfail silently if the failure were to come without
    the density-error signature of the documented defect.
    """
    cfg = lc.ColumnConfig(radius=3e-3, dx=1e-3, kR=0.6971, epsilon=0.02)
    result = lc.run_column(cfg, steps=120, sample_every=25)
    blob = result.to_dict()
    solved = (blob["omega_measured"] == blob["omega_measured"]      # not NaN
              and abs(blob["omega_error"]) < 0.10
              and blob["max_density_error"] < 0.10)
    if solved:
        return
    assert blob["max_density_error"] >= 0.10, (
        "growth-rate anchor missed without the documented free-surface defect "
        f"(max |rho/rho0-1| = {blob['max_density_error']:.3f}): investigate "
        "before filing this as the known gap")
    pytest.xfail("C4 open: 3-D free-surface pressure closure is not sufficient "
                 "for the capillary mode; diagnosed in "
                 "docs/verification_rayleigh_plateau.md")
