"""Verification of the added-mass correction to the reduced floe launch model.

The engineering claim under test is the closed form, not the grid solver: a
2-D circular section in an unbounded fluid has ``C_m = 1`` (Lamb §71), so the
inertia that resists an impulse is ``rho_ice pi r^2 + C_m rho_w pi r^2`` and the
dry-mass launch speed of :func:`tensorsph.ice_splash.floe_launch_estimate` is an
upper bound by ``1 + C_m rho_ice_ratio``. The potential-flow solve in
:mod:`tensorsph.added_mass` is graded against that closed form with the
tolerance its first-order staircase boundary can actually hold (the measured
bias is recorded in ``docs/verification_ice_floe_added_mass.md``).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.added_mass import (
    DISC_ADDED_MASS_COEFFICIENT,
    cylinder_added_mass,
    speed_ratio_with_added_mass,
)
from tensorsph.ice_splash import (
    IceSplashConfig,
    build_dem_exchange,
    floe_layout,
    floe_launch_estimate,
    validate_config,
)

# Cheap grids for the suite; the production ladder is run by the benchmark.
CFG_KW = dict(dx=0.025, column_width=0.20, column_height=0.28, pool_width=1.0,
              tank_height=0.60, sheet_start=0.20, sheet_angle_deg=25.0,
              deck_height=0.28, sound_speed=20.0, viscosity=0.0, device="cpu",
              output_every=5, max_courant=0.3, n_floes=5, seed=7)


def _config(**overrides) -> IceSplashConfig:
    base = dict(CFG_KW)
    base.update(overrides)
    target = float(base.pop("t_target", 0.10))
    cfg = IceSplashConfig(**base)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(target / cfg.dt))
    return cfg


class _StubRun:
    """Minimal stand-in for :class:`IceSplashRun` (only the probe outputs)."""

    def __init__(self, impulses, forces, peak_ke):
        self.floe_impulse = list(impulses)
        self.floe_peak_force = list(forces)
        self.peak_splash_kinetic_energy = float(peak_ke)


# --------------------------------------------------------------------------- #
# Closed form and algebra
# --------------------------------------------------------------------------- #

def test_disc_added_mass_coefficient_is_the_lamb_value():
    assert DISC_ADDED_MASS_COEFFICIENT == 1.0


def test_speed_ratio_is_the_exact_inertia_ratio():
    rho_ice, rho_w, cm = 917.0, 1000.0, 1.0
    ratio = speed_ratio_with_added_mass(rho_ice, rho_w, cm)
    assert ratio == pytest.approx(rho_ice / (rho_ice + cm * rho_w))
    # The ice/water density ratio makes the correction large: the dry-mass
    # reading over-predicts the speed by more than a factor of two.
    assert 1.0 / ratio == pytest.approx((rho_ice + rho_w) / rho_ice, rel=1e-12)
    assert 1.0 / ratio == pytest.approx(2.0905, abs=5e-4)
    # No entrained inertia recovers the dry-mass reading exactly.
    assert speed_ratio_with_added_mass(rho_ice, rho_w, 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        speed_ratio_with_added_mass(0.0, rho_w)
    with pytest.raises(ValueError):
        speed_ratio_with_added_mass(rho_ice, rho_w, -1.0)


# --------------------------------------------------------------------------- #
# Potential-flow solve against the closed form
# --------------------------------------------------------------------------- #

def test_unbounded_solve_reproduces_the_closed_form_within_the_staircase_bias():
    # Cold start (initial="zero") is the honest solve: the dipole initial guess
    # is the analytic answer and would only re-test the quadrature.
    result = cylinder_added_mass(cells_per_radius=8, box_radii=6.0,
                                 initial="zero", tolerance=1e-5)
    assert result.boundary == "unbounded"
    assert result.residual < 1e-4
    # First-order staircase boundary: measured bias is -13 % on this grid
    # (docs/verification_ice_floe_added_mass.md, section 3.1).
    assert result.coefficient == pytest.approx(1.0, abs=0.20)
    assert result.coefficient > 0.5


def test_solve_is_isotropic_for_a_circular_section():
    coefficient_x = cylinder_added_mass(cells_per_radius=8, box_radii=6.0,
                                       direction="x", tolerance=1e-5).coefficient
    coefficient_y = cylinder_added_mass(cells_per_radius=8, box_radii=6.0,
                                       direction="y", tolerance=1e-5).coefficient
    assert coefficient_x == pytest.approx(coefficient_y, rel=1e-9)


def test_both_planes_recover_the_free_field_at_a_wide_gap():
    free = cylinder_added_mass(cells_per_radius=8, box_radii=6.0,
                               tolerance=1e-5).coefficient
    for boundary in ("rigid_wall", "free_surface"):
        result = cylinder_added_mass(cells_per_radius=8, box_radii=6.0,
                                     boundary=boundary, gap=4.0, tolerance=1e-5)
        assert result.gap_radii == pytest.approx(4.0)
        # A plane 4 radii away must not change the inertia by more than the
        # method's own noise floor (measured 1.5-3.2 % in the benchmark).
        assert result.coefficient / free == pytest.approx(1.0, abs=0.08)


def test_solver_rejects_inconsistent_inputs():
    with pytest.raises(ValueError):
        cylinder_added_mass(boundary="somewhere_else")
    with pytest.raises(ValueError):
        cylinder_added_mass(direction="z")
    with pytest.raises(ValueError):
        cylinder_added_mass(initial="random")
    with pytest.raises(ValueError):
        cylinder_added_mass(cells_per_radius=2)
    with pytest.raises(ValueError):
        cylinder_added_mass(gap=-1.0, boundary="rigid_wall")


# --------------------------------------------------------------------------- #
# Launch estimate and DEM hand-off
# --------------------------------------------------------------------------- #

def test_launch_estimate_reports_both_readings_and_their_ratio():
    cfg = _config()
    run = _StubRun(impulses=[2.0, 0.5], forces=[70.0, 20.0], peak_ke=4.0)
    floes = floe_layout(cfg, torch.device("cpu"))
    estimate = floe_launch_estimate(run, cfg, floes)
    m_dry = cfg.floe_rho * math.pi * cfg.floe_radius ** 2
    m_added = cfg.floe_added_mass_coefficient * cfg.rho0 * math.pi * cfg.floe_radius ** 2
    assert estimate["added_mass_coefficient"] == pytest.approx(1.0)
    for row, impulse in zip(estimate["floes"], run.floe_impulse):
        assert row["mass_kg_per_m"] == pytest.approx(m_dry)
        assert row["added_mass_kg_per_m"] == pytest.approx(m_added)
        assert row["launch_speed_m_s"] == pytest.approx(impulse / m_dry)
        assert row["launch_speed_with_added_mass_m_s"] == pytest.approx(
            impulse / (m_dry + m_added))
        assert row["speed_ratio_with_added_mass"] == pytest.approx(m_dry / (m_dry + m_added))
        # Ballistic apex scales with the square of the speed.
        assert row["ballistic_apex_with_added_mass_m"] == pytest.approx(
            row["ballistic_apex_m"] * (m_dry / (m_dry + m_added)) ** 2, rel=1e-12)


def test_dry_mass_mode_is_available_and_degenerate_ratio_is_one():
    cfg = _config(floe_added_mass_coefficient=0.0)
    run = _StubRun(impulses=[2.0], forces=[70.0], peak_ke=4.0)
    floes = floe_layout(cfg, torch.device("cpu"))
    estimate = floe_launch_estimate(run, cfg, floes)
    row = estimate["floes"][0]
    assert row["added_mass_kg_per_m"] == 0.0
    assert row["launch_speed_with_added_mass_m_s"] == pytest.approx(row["launch_speed_m_s"])
    assert estimate["dry_mass_speed_ratio"] == pytest.approx(1.0)


def test_added_mass_coefficient_is_validated():
    validate_config(_config())
    validate_config(_config(floe_added_mass_coefficient=0.0))
    with pytest.raises(ValueError, match="floe_added_mass_coefficient"):
        validate_config(_config(floe_added_mass_coefficient=-0.1))
    with pytest.raises(ValueError, match="floe_added_mass_coefficient"):
        validate_config(_config(floe_added_mass_coefficient=2.5))


def test_dem_payload_carries_the_added_mass_corrected_speed():
    cfg = _config(n_floes=2)
    impulses = [1.5, 0.4]
    run = _StubRun(impulses=impulses, forces=[30.0, 10.0], peak_ke=1.0)
    run.final_positions = torch.tensor([[0.1, 0.1], [0.2, 0.05]], dtype=torch.float64)
    run.final_velocity = torch.zeros((2, 2), dtype=torch.float64)
    payload = build_dem_exchange(run, cfg)
    assert payload["source"]["floe_added_mass_coefficient"] == pytest.approx(1.0)
    ice = [k for k, m in enumerate(payload["particles"]["material"]) if m == "ice"]
    m_dry = cfg.floe_rho * math.pi * cfg.floe_radius ** 2
    m_added = cfg.rho0 * math.pi * cfg.floe_radius ** 2
    for slot, impulse in zip(ice, impulses):
        speed = payload["particles"]["velocity_m_s"][slot][1]
        assert speed == pytest.approx(impulse / (m_dry + m_added))
        # Which is what the launch estimate reports as the recommended reading.
        assert speed < impulse / m_dry
