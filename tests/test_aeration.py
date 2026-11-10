"""Tests for the air-entrainment / bubble-transport / aerated-drag chain (P165W2-N04).

Layout mirrors the physics chain: inception criterion (C1), void-fraction profile
against the erf anchor (C2), aerated friction reduction against the Chisholm
multiplier (C3), bubble rise velocity against Mendelson (C4), and air-volume
conservation through the Lagrangian -> SPH mapping (C5).
"""
import math

import pytest
import torch

from tensorsph.aeration import (
    AIR_20C,
    C_CRIT,
    CAVITY_FRACTION,
    WATER_20C,
    aeration_active,
    aerated_friction_factor,
    analytic_void_fraction,
    chanson_parameter,
    chisholm_multiplier,
    chisholm_reduction_rate,
    derived_crit_constant,
    drag_balance_rise_velocity,
    entrainment_flux,
    erf_anchor,
    inception_depth,
    inception_froude,
    inception_velocity,
    l2_relative_error,
    lockhart_martinelli_x,
    loglaw_reduction_rate,
    mendelson_rise_velocity,
    onset_velocity_by_bisection,
    rise_velocity,
    solve_void_fraction_field,
    stokes_rise_velocity,
    transport_bubbles,
    void_fraction_from_bubbles,
    wood_kappa_effective,
)

DEPTH = 0.05
ANGLE = math.radians(15.0)


# --------------------------------------------------------------------------- #
# C1 -- inception criterion
# --------------------------------------------------------------------------- #

def test_chanson_parameter_reduces_to_the_definition():
    v, d = 3.0, DEPTH
    expected = v * math.sin(ANGLE) / math.sqrt(9.81 * d * math.cos(ANGLE))
    assert chanson_parameter(v, d, ANGLE) == pytest.approx(expected, rel=1e-12)


def test_inception_velocity_lands_exactly_on_the_threshold():
    v_i = inception_velocity(DEPTH, ANGLE)
    assert chanson_parameter(v_i, DEPTH, ANGLE) == pytest.approx(C_CRIT, rel=1e-12)


def test_inception_depth_inverts_inception_velocity():
    v_i = inception_velocity(DEPTH, ANGLE)
    assert inception_depth(v_i, ANGLE) == pytest.approx(DEPTH, rel=1e-12)


def test_inception_froude_reaches_unity_where_cos_equals_sin_squared():
    # Fr_i = C_crit sqrt(cos)/sin is unity where cos(theta) = sin^2(theta),
    # i.e. sin(theta) = sqrt((sqrt(5)-1)/2) -> theta = 51.83 deg.
    angle = math.asin(math.sqrt((math.sqrt(5.0) - 1.0) / 2.0))
    assert inception_froude(angle) == pytest.approx(C_CRIT, rel=1e-12)
    assert inception_froude(math.radians(45.0)) > C_CRIT


def test_inception_froude_grows_as_the_chute_flattens():
    steep = inception_froude(math.radians(30.0))
    shallow = inception_froude(math.radians(10.0))
    assert shallow > steep > C_CRIT


def test_energy_balance_reproduces_the_empirical_constant():
    """C1: derived C_crit vs Chanson's 1.0 -- error must stay inside 10%."""
    derived = derived_crit_constant(CAVITY_FRACTION)
    assert abs(derived - C_CRIT) / C_CRIT <= 0.10


def test_operational_onset_matches_the_algebraic_inception_velocity():
    numerical = onset_velocity_by_bisection(DEPTH, ANGLE)
    algebraic = inception_velocity(DEPTH, ANGLE)
    assert abs(numerical - algebraic) / algebraic <= 1e-6


def test_entrainment_flux_vanishes_below_onset_and_is_positive_above():
    v_i = inception_velocity(DEPTH, ANGLE)
    assert float(entrainment_flux(0.9 * v_i, DEPTH, ANGLE)) == 0.0
    assert float(entrainment_flux(1.1 * v_i, DEPTH, ANGLE)) > 0.0


def test_aeration_mask_follows_the_criterion():
    v_i = inception_velocity(DEPTH, ANGLE)
    assert not aeration_active(chanson_parameter(0.99 * v_i, DEPTH, ANGLE))
    assert aeration_active(chanson_parameter(1.01 * v_i, DEPTH, ANGLE))


# --------------------------------------------------------------------------- #
# C4 -- bubble rise velocity
# --------------------------------------------------------------------------- #

def test_stokes_limit_matches_the_analytic_formula():
    radius = 1.0e-5
    expected = 2.0 * 9.81 * radius ** 2 * (WATER_20C["rho"] - AIR_20C["rho"]) / (
        9.0 * WATER_20C["mu"]
    )
    assert stokes_rise_velocity(radius) == pytest.approx(expected, rel=1e-12)


def test_drag_balance_reproduces_stokes_in_the_creeping_flow_limit():
    radius = 1.0e-5
    stokes = stokes_rise_velocity(radius)
    balanced = drag_balance_rise_velocity(radius)
    assert abs(balanced - stokes) / stokes <= 0.01


def test_mendelson_matches_its_wave_theory_definition():
    radius = 1.0e-3
    expected = math.sqrt(
        2.0 * WATER_20C["sigma"] / (WATER_20C["rho"] * radius) + 2.0 * 9.81 * radius
    )
    assert mendelson_rise_velocity(radius) == pytest.approx(expected, rel=1e-12)


def test_rise_velocity_agrees_with_mendelson_over_the_ellipsoidal_band():
    """C4: R = 0.5..5 mm against Mendelson, error <= 10% (this closure)."""
    radii = torch.linspace(5.0e-4, 5.0e-3, 12, dtype=torch.float64)
    model = rise_velocity(radii)
    reference = mendelson_rise_velocity(radii)
    worst = float(((model - reference).abs() / reference).max())
    assert worst <= 0.10


def test_rise_velocity_is_continuous_across_the_regime_blend():
    # The hand-over at R = 0.5 mm must not leave a step: the blended value has
    # to approach Mendelson's branch from below and match it exactly at the end.
    just_below = rise_velocity(5.0e-4 * (1.0 - 1e-9))
    at_switch = rise_velocity(5.0e-4)
    assert abs(just_below - at_switch) / at_switch <= 1e-6


def test_rigid_sphere_branch_is_steeper_than_the_mobile_interface_branch():
    radii = torch.linspace(5.0e-5, 2.0e-4, 8, dtype=torch.float64)
    values = rise_velocity(radii)
    assert torch.all(torch.diff(values) > 0.0)
    # A mobile interface rises faster than a rigid sphere of the same size --
    # the documented reason the rigid-sphere branch is abandoned above 0.5 mm.
    radius = 5.0e-4
    assert float(mendelson_rise_velocity(radius)) > 2.0 * drag_balance_rise_velocity(radius)


def test_rise_velocity_uses_the_rigid_sphere_branch_below_the_blend():
    radius = 1.0e-4
    assert rise_velocity(radius) == pytest.approx(drag_balance_rise_velocity(radius), rel=1e-12)


# --------------------------------------------------------------------------- #
# C2 -- aeration-layer void fraction profile
# --------------------------------------------------------------------------- #

def _field(drift=0.0, cells=200, depth=1.0, transit=2.0, diffusivity=0.02):
    return solve_void_fraction_field(
        depth=depth, cells=cells, diffusivity=diffusivity, drift=drift, transit_time=transit
    )


def test_drift_free_profile_matches_the_erf_anchor():
    """C2: L2 error against C(y)=1-erf(y/(Y90/sqrt(2))) must stay under 15%."""
    y, c = _field()
    dy = float(y[1] - y[0])
    y90 = 2.0 * math.sqrt(2.0) * math.sqrt(0.02 * 2.0)
    reference = erf_anchor(y - 0.5 * dy, y90)
    assert l2_relative_error(c, reference) <= 0.15


def test_erf_anchor_is_normalised_at_the_surface_and_decays():
    y = torch.linspace(0.0, 2.0, 21, dtype=torch.float64)
    profile = erf_anchor(y, 0.5)
    assert float(profile[0]) == pytest.approx(1.0, rel=1e-12)
    assert float(profile[-1]) < 0.01


def test_buoyant_drift_profile_matches_the_closed_form_solution():
    y, c = _field(drift=0.15)
    reference = analytic_void_fraction(y - 0.5 * float(y[1] - y[0]), 2.0, 0.02, 0.15)
    assert l2_relative_error(c, reference) <= 0.15


def test_drift_shallows_the_aeration_layer():
    # Buoyancy lifts bubbles back towards the surface: less air at depth.
    _, still = _field(drift=0.0)
    _, rising = _field(drift=0.15)
    assert float(rising[-40]) < float(still[-40])


def test_analytic_void_fraction_reduces_to_erfc_without_drift():
    y = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    closed = analytic_void_fraction(y, 2.0, 0.02, 0.0)
    direct = torch.erfc(y / (2.0 * math.sqrt(0.02 * 2.0)))
    assert l2_relative_error(closed, direct) <= 1e-12


# --------------------------------------------------------------------------- #
# C3 -- aerated friction reduction
# --------------------------------------------------------------------------- #

def test_chisholm_multiplier_matches_the_algebra_at_x_equals_one():
    assert chisholm_multiplier(1.0) == pytest.approx(1.0 + 20.0 + 1.0, rel=1e-12)


def test_lockhart_martinelli_parameter_falls_as_void_fraction_grows():
    x_dilute = lockhart_martinelli_x(4.9, 0.1)
    x_dense = lockhart_martinelli_x(3.5, 1.5)
    assert x_dense < x_dilute


def test_loglaw_friction_factor_tracks_blasius():
    re = 1.0e5
    from tensorsph.aeration import friction_factor_loglaw

    blasius = 0.3164 * re ** -0.25
    assert abs(friction_factor_loglaw(re) - blasius) / blasius <= 0.10


def test_wood_kappa_reduces_with_void_fraction():
    assert wood_kappa_effective(0.41, 0.1, 0.5) < 0.41
    assert wood_kappa_effective(0.41, 0.1, 0.0) == pytest.approx(0.41)


def test_aerated_mixture_carries_less_wall_shear_despite_a_higher_friction_factor():
    # The mixture Reynolds number falls with void fraction, so *f* itself rises;
    # the wall shear proxy f * rho_m * u^2 is what must fall.
    f_clear = aerated_friction_factor(5.0, 0.0, 0.5)
    f_aerated = aerated_friction_factor(5.0, 0.2, 0.5)
    assert 0.0 < f_clear < f_aerated
    tau_clear = f_clear * WATER_20C["rho"]
    tau_aerated = f_aerated * (WATER_20C["rho"] * 0.8 + AIR_20C["rho"] * 0.2)
    assert tau_aerated < tau_clear


def test_chisholm_and_loglaw_reduction_rates_agree():
    """C3: reduction rate vs the Chisholm multiplier, error <= 20%."""
    worst = 0.0
    for alpha in (0.12, 0.16, 0.20, 0.25, 0.30):
        reference = chisholm_reduction_rate(alpha, 5.0, 0.5)
        model = loglaw_reduction_rate(alpha, 5.0, 0.5)
        assert reference > 0.0 and model > 0.0
        worst = max(worst, abs(model - reference) / reference)
    assert worst <= 0.20


# --------------------------------------------------------------------------- #
# C5 -- air volume conservation
# --------------------------------------------------------------------------- #

def test_lagrangian_transport_conserves_air_volume():
    generator = torch.Generator().manual_seed(9315)
    depths = torch.rand(4000, dtype=torch.float64, generator=generator) * 0.2
    moved, _ = transport_bubbles(
        depths, steps=400, dt=1.0e-3, drift=0.15, diffusivity=0.02, generator=generator
    )
    # Reflecting surface: every bubble stays in the domain, so the count (and
    # with equal bubble volumes the total air volume) is unchanged.
    assert int(moved.numel()) == int(depths.numel())
    assert float(moved.min()) >= 0.0


def test_sph_void_fraction_summation_recovers_the_prescribed_fraction():
    """C5: the Lagrangian -> SPH map is unbiased for a uniform bubble cloud.

    Kernel summation of discrete bubbles is a shot-noise estimator, so the
    *particle-averaged* void fraction is what has to reproduce the prescribed
    one; the per-particle scatter is a resolution property, not an error.
    """
    generator = torch.Generator().manual_seed(9315)
    target = 0.05
    box = 0.5
    count = 20000
    bubble_vol = target * box * box / count
    bubbles = torch.rand(count, 2, dtype=torch.float64, generator=generator) * box
    axis = torch.linspace(0.07, box - 0.07, 12, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    fluid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    alpha = void_fraction_from_bubbles(bubbles, torch.full((count,), bubble_vol), fluid, 0.02)
    assert float((alpha > 0.0).float().mean()) > 0.9
    assert abs(float(alpha.mean()) - target) / target <= 0.05


def test_bubble_cloud_spreads_under_diffusion():
    generator = torch.Generator().manual_seed(7)
    depths = torch.zeros(2000, dtype=torch.float64)
    moved, _ = transport_bubbles(
        depths, steps=200, dt=1.0e-3, drift=0.0, diffusivity=0.02, generator=generator
    )
    # Random walk from the surface: rms depth = sqrt(2 D t) to within 10%.
    expected = math.sqrt(2.0 * 0.02 * 0.2)
    rms = float(torch.sqrt(torch.mean(moved.square())))
    assert abs(rms - expected) / expected <= 0.10


def test_escaping_bubbles_are_flagged():
    generator = torch.Generator().manual_seed(11)
    depths = torch.zeros(500, dtype=torch.float64)
    _, escaped = transport_bubbles(
        depths,
        steps=50,
        dt=1.0e-3,
        drift=0.0,
        diffusivity=0.02,
        generator=generator,
        surface="escape",
    )
    assert bool(escaped.any())
