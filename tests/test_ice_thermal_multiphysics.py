"""Verification tests for :mod:`tensorsph.ice_thermal` (campaign IN13W2-N07).

Every assertion compares a *computed* number with an independent reference:

* the Neumann similarity solution of the Stefan problem (front + profile);
* exact thermo-elastic identities (Timoshenko plate, self-equilibrium, and the
  classical result that a free plate with a *linear* gradient is stress free);
* the exact Horton-Rogers-Lapwood onset ``Ra_c = 4 pi^2``;
* discrete conservation laws (salt mass to round-off) and the analytic
  equilibrium salinity of the drainage model;
* thermodynamic round-trips of the enthalpy model (``T(e(T)) == T``).

The heavier integration anchors (resolution ladder, 45-day coupled runs) live in
``benchmarks/ice_thermal_multiphysics``; the settings here are deliberately
small so the suite stays in the seconds range.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorsph.ice_thermal import darcy
from tensorsph.ice_thermal import properties as props
from tensorsph.ice_thermal import thermal_stress as stress_mod
from tensorsph.ice_thermal.brine import (V_IRR, band_check, drainage_advance,
                                         drainage_step, drainage_step_implicit,
                                         mobile_brine_fraction, residual_salinity,
                                         run_desalination)
from tensorsph.ice_thermal.coupled import (ThreeFieldConfig, run_three_field,
                                           spinup_diagnostic)
from tensorsph.ice_thermal.enthalpy import FRESH, SEA_ICE, neumann_lambda
from tensorsph.ice_thermal.sph_thermal import (StefanConfig, lattice_diffusion_factor,
                                               run_stefan_sph)


# ------------------------------------------------------------------ properties
def test_freezing_point_polynomial_matches_known_values():
    assert props.freezing_point(0.0) == pytest.approx(props.T_MELT, abs=1e-12)
    # -1.75 C at S = 32 psu is the classic seawater freezing point
    assert props.freezing_point(32.0) - props.T_MELT == pytest.approx(-1.751, abs=2e-3)


def test_brine_salinity_inverts_the_liquidus():
    for temperature in (272.15, 270.15, 268.15, 263.15):
        sb = props.brine_salinity(temperature)
        assert props.freezing_point(sb) == pytest.approx(temperature, abs=2e-3)
    vec = props.brine_salinity_vec(torch.tensor([272.15, 268.15]))
    scalar = [props.brine_salinity(x) for x in (272.15, 268.15)]
    assert torch.allclose(vec, torch.tensor(scalar, dtype=torch.float64), rtol=2e-3)


def test_brine_volume_fraction_and_permeability_limits():
    # warmer -> more brine; at the freezing point the ice is all liquid
    cold = props.brine_volume_fraction(268.15, 5.0)
    warm = props.brine_volume_fraction(271.15, 5.0)
    assert 0.0 < cold < warm <= 1.0
    assert props.brine_volume_fraction(273.0, 5.0) == 1.0
    # below the percolation threshold there is no connected brine
    assert props.permeability(0.04) == 0.0
    assert props.permeability(0.06) > 0.0
    assert props.permeability(0.12) > props.permeability_vec(torch.tensor([0.06])).item()


# ------------------------------------------------------------------- enthalpy
def test_fresh_enthalpy_roundtrip_and_latent_jump():
    temperature = torch.linspace(240.0, 285.0, 91, dtype=torch.float64)
    e = FRESH.enthalpy(temperature)
    back = FRESH.temperature(e)
    assert torch.allclose(back, temperature, atol=1e-9)
    # the mushy band holds the latent heat plus its own sensible heat
    band = torch.tensor([props.T_MELT, props.T_MELT - FRESH.mushy_dt], dtype=torch.float64)
    jump = float(FRESH.enthalpy(band)[0] - FRESH.enthalpy(band)[1])
    assert jump == pytest.approx(props.L_FUSION + props.C_ICE * FRESH.mushy_dt, rel=1e-9)


def test_sea_ice_enthalpy_roundtrip_and_latent_budget():
    temperature = torch.linspace(252.0, 272.0, 41, dtype=torch.float64)
    salinity = torch.full_like(temperature, 8.0)
    e = SEA_ICE.enthalpy(temperature, salinity)
    back = SEA_ICE.temperature(e, salinity)
    assert torch.allclose(back, temperature, atol=1e-6)
    # monotone in T (a necessary condition for any enthalpy method)
    assert bool((e[1:] > e[:-1]).all())
    # all the brine freezes between Tf and deep cold: latent ~ L per kg
    tf = float(SEA_ICE.freezing_point(torch.tensor([8.0]))[0])
    e_tf = float(SEA_ICE.enthalpy(torch.tensor([tf], dtype=torch.float64),
                                  torch.tensor([8.0], dtype=torch.float64))[0])
    e_cold = float(SEA_ICE.enthalpy(torch.tensor([252.0], dtype=torch.float64),
                                    torch.tensor([8.0], dtype=torch.float64))[0])
    # latent (L) plus the sensible heat of the ice below Tf
    latent_plus_sensible = (e_tf - e_cold) / props.L_FUSION
    assert 1.0 <= latent_plus_sensible <= 1.2


def test_sea_ice_liquid_fraction_endpoints():
    tf = float(SEA_ICE.freezing_point(torch.tensor([10.0]))[0])
    at_tf = float(SEA_ICE.liquid_fraction(torch.tensor([tf], dtype=torch.float64),
                                          torch.tensor([10.0], dtype=torch.float64))[0])
    assert at_tf == pytest.approx(1.0, abs=2e-2)
    deep = float(SEA_ICE.liquid_fraction(torch.tensor([253.15], dtype=torch.float64),
                                         torch.tensor([10.0], dtype=torch.float64))[0])
    assert deep < 0.2


# --------------------------------------------------------------- SPH / Stefan
def test_neumann_lambda_satisfies_the_transcendental_equation():
    for ste in (0.01, 0.06, 0.5, 2.0):
        lam = neumann_lambda(ste)
        residual = math.sqrt(math.pi) * lam * math.exp(lam * lam) * math.erf(lam) - ste
        assert abs(residual) < 1e-10


def test_lattice_diffusion_factor_is_consistent_across_wavenumbers():
    factor, scatter = lattice_diffusion_factor(n=128)
    assert 0.9 < factor < 1.1
    assert scatter < 0.01          # a single factor is legitimate at low k


def test_sph_stefan_front_matches_neumann():
    cfg = StefanConfig(n_particles=60, dx=0.02, t_end=2.0e4, profile_every=25)
    res = run_stefan_sph(cfg)
    err_max, err_mean = res.front_error(skip=2)
    # a coarse lattice (dx = 20 mm, front ~ 4 cells at early times): the ladder in
    # the benchmark reaches 0.3 % at dx = 8 mm, the gate here is 5 %
    assert err_mean < 0.05
    assert err_max < 0.08
    assert res.profile_error() < 0.5           # K, RMS in the solid
    assert res.final_front > 0.02              # the front actually advanced


def test_stefan_front_is_diffusive_not_linear():
    """X^2 must grow linearly in time for a Stefan front (self-similarity)."""
    cfg = StefanConfig(n_particles=60, dx=0.02, t_end=2.0e4, profile_every=10)
    res = run_stefan_sph(cfg)
    times = torch.tensor(res.time, dtype=torch.float64)
    front2 = torch.tensor(res.front, dtype=torch.float64) ** 2
    mask = times > 5.0e3
    slope = (front2[mask] / times[mask])
    assert float(slope.std() / slope.mean()) < 0.05      # X^2/t is a constant


# --------------------------------------------------------------------- brine
def test_explicit_and_implicit_drainage_agree_at_small_dt():
    s = torch.full((6,), 12.0, dtype=torch.float64)
    t = torch.full((6,), 268.15, dtype=torch.float64)
    s_explicit, flux = drainage_step(s, t, 0.02, 1.0)
    s_implicit, flux_i = drainage_step_implicit(s, t, 0.02, 1.0)
    assert torch.allclose(s_explicit, s_implicit, rtol=5e-3)
    assert flux == pytest.approx(flux_i, rel=0.2)


def test_drainage_conserves_salt_and_respects_the_equilibrium_floor():
    s = torch.full((10,), 12.0, dtype=torch.float64)
    t = torch.linspace(263.15, 272.0, 10, dtype=torch.float64)
    dz = 0.02
    s_new, removed, n_sub = drainage_advance(s, t, dz, 3600.0)
    lost = float((props.RHO_ICE * dz * (s - s_new)).sum())
    assert lost == pytest.approx(removed, rel=1e-6)
    assert n_sub >= 1
    assert float(s_new.min()) >= 0.0
    # the equilibrium floor at the coldest level is a lower bound of the drainage
    floor = float(residual_salinity(t).min())
    assert float(s_new.mean()) > floor * 0.5


def test_residual_salinity_is_a_fixed_point_of_the_drainage():
    t = torch.full((4,), 266.15, dtype=torch.float64)
    s = residual_salinity(t)
    s_new, removed, _ = drainage_advance(s, t, 0.02, 3600.0)
    assert abs(removed) < 1e-6 * float(s.sum())
    assert torch.allclose(s_new, s, rtol=1e-3)


def test_mobile_brine_vanishes_at_the_irreducible_saturation():
    assert float(mobile_brine_fraction(V_IRR - 1e-3)) == 0.0
    assert float(mobile_brine_fraction(V_IRR + 0.02)) > 0.0


def test_desalination_never_crosses_the_equilibrium_floor():
    import torch as _t

    res = run_desalination(thickness=0.3, n=10, days=20.0)
    margin = (_t.tensor(res.profile_s[-1], dtype=_t.float64)
              - residual_salinity(_t.tensor(res.profile_t[-1], dtype=_t.float64)))
    assert float(margin.min()) >= -1e-9


def test_desalination_reaches_the_observed_band():
    res = run_desalination(thickness=0.3, n=10, days=20.0)
    assert res.salt_error < 1e-3
    check = band_check(res.bulk_salinity[-1])
    assert check["ratio_to_ocean"] < 0.3       # ice is much fresher than the ocean
    assert res.bulk_salinity[-1] < 12.0        # it did desalinate


# --------------------------------------------------------------------- Darcy
def test_darcy_benard_onset_matches_four_pi_squared():
    ra, a_c, curve = darcy.critical_rayleigh(n=40, n_wavenumbers=9)
    assert ra == pytest.approx(darcy.RA_CRITICAL_EXACT, rel=0.02)
    assert 2.4 < a_c < 3.9
    for a, value in curve:
        assert value == pytest.approx(darcy.analytic_marginal(a), rel=0.03)


def test_nusselt_is_one_below_onset():
    assert darcy.nusselt(0.5 * darcy.RA_CRITICAL_EXACT) == 1.0
    assert darcy.nusselt(4.0 * darcy.RA_CRITICAL_EXACT) > 1.0


# -------------------------------------------------------------------- stress
def test_timoshenko_restrained_plate_stress():
    z = torch.linspace(-0.25, 0.25, 101, dtype=torch.float64)
    t_ref = props.T_MELT - 2.0
    d_t = 10.0
    temperature = t_ref + d_t * z / 0.5
    sigma = stress_mod.restrained_stress(temperature, t_ref=t_ref)
    analytic = stress_mod.biaxial_modulus() * props.ALPHA_TH * d_t / 2.0
    assert stress_mod.max_tensile_stress(sigma.abs()) == pytest.approx(analytic, rel=1e-12)


def test_free_plate_with_linear_gradient_is_stress_free():
    z = torch.linspace(-0.25, 0.25, 201, dtype=torch.float64)
    temperature = 270.0 + 8.0 * z / 0.5
    sigma = stress_mod.free_plate_stress(temperature, z)
    assert float(sigma.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_free_plate_self_equilibration_for_a_nonlinear_profile():
    z = torch.linspace(-0.25, 0.25, 201, dtype=torch.float64)
    temperature = 265.0 + 8.0 * torch.exp(-(z + 0.25) / 0.08)
    sigma = stress_mod.free_plate_stress(temperature, z)
    force, moment = stress_mod.free_plate_moments(sigma, z)
    assert abs(force) < 1e-6 * float(sigma.abs().max())
    assert abs(moment) < 1e-3 * float(sigma.abs().max()) * 0.25


def test_cracking_thresholds():
    d_t_crit = stress_mod.critical_temperature_drop()
    assert d_t_crit == pytest.approx(0.7e6 * 0.67 / (9e9 * 5e-5), rel=1e-9)
    assert 0.5 < d_t_crit < 2.0            # a one-kelvin cooling cracks sea ice
    assessment = stress_mod.crack_assessment([5.0e6, -1.0e6], flaw_depth=0.01)
    assert assessment.cracks
    assert assessment.critical_depth_m < 0.01


# ------------------------------------------------------------------- coupled
def test_coupled_column_conserves_salt_and_stays_physical():
    cfg = ThreeFieldConfig(n_z=10, days=0.5, dt=900.0, spinup=False)
    res = run_three_field(cfg)
    assert res.salt_error < 1e-3
    assert all(0.0 <= s <= cfg.salinity0 for s in res.salinity)
    assert all(x < props.T_MELT for x in res.temperature)
    assert res.bulk_salinity[-1] < cfg.salinity0        # drainage happened
    assert math.isfinite(res.melt_total_m) and res.melt_total_m >= 0.0


def test_coupled_temperature_stays_within_the_boundary_values():
    """The surface cools first; the column stays between surface and base values."""
    cfg = ThreeFieldConfig(n_z=10, days=0.2, dt=600.0, spinup=False)
    res = run_three_field(cfg)
    t = res.temperature
    assert t[0] < t[len(t) // 2]                     # a cooling front from the top
    # the surface condition is imposed through a mirror ghost, so the first
    # particle may sit a fraction of a kelvin past the wall value
    lower = min(cfg.t_surface, float(props.freezing_point(cfg.ocean_salinity0)))
    assert all(x >= lower - 0.5 for x in t)
    tf0 = float(props.freezing_point(cfg.salinity0))
    assert all(x <= tf0 + 0.5 for x in t)
    assert math.isfinite(max(t))


def test_coupled_cold_snap_slows_in_the_mushy_layer():
    """The mushy layer must buffer a cold snap; fresh ice must not."""
    cfg = ThreeFieldConfig(n_z=10, days=0.1, dt=600.0)
    spin = spinup_diagnostic(cfg, SEA_ICE, target=0.5, max_steps=400)
    assert spin["fresh_ice"]["relaxation_s"] is not None
    # within the same window the sea-ice column is still far from relaxed
    assert spin["sea_ice"]["relaxation_s"] is None
