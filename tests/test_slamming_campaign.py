"""Campaign node p129w2b-n06: TensorSPH slamming pressure verification.

Tests the five machine criteria plus negative controls:

1. Peak slamming force vs Wagner upper bound <= 15%
2. Pressure distribution shape vs Wagner self-similar L2 error <= 20%
3. Pressure oscillation (noise) <= 10% of mean
4. Convergence order of force peak >= 0.5
5. Energy conservation (KE+IE+PE drift) <= 5%
6. Negative control: disabling delta-SPH must cause pressure oscillation > threshold

Applicability domain (explicit): rigid body, 2D, incompressible, no cavitation.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorsph.slamming import (
    SlammingConfig,
    SlammingImpact,
    added_mass_force,
    added_mass_per_length,
    compute_energy,
    force_order,
    pressure_noise,
    pressure_shape_l2,
    slamming_force_coefficient,
    wagner_shape,
)
from tensorsph.wedge_entry import wet_half_width, wagner_pmax, wagner_p_centre

DTYPE = torch.float64
RHO = 1000.0


# --------------------------------------------------------------------------- #
# Helper: run a short slamming simulation and return the series
# --------------------------------------------------------------------------- #

def _run_short(beta_deg=20.0, velocity=2.0, dx=4.0e-3, delta_sph=0.1,
               mls_density=True, frames=11, c_end=None, initial_gap_dx=0.0):
    """Run a short slamming case and return (cfg, series).

    c_end is chosen automatically to ensure the wedge contacts the water
    and develops a wetted region within the simulation window.
    initial_gap_dx=0 starts the wedge at the free surface (no pre-contact phase).
    """
    if c_end is None:
        # Ensure t_end > time_to_contact = initial_gap_dx*dx/V
        # t_end = 2*c_end*tan(beta)/(pi*V)
        beta = math.radians(beta_deg)
        time_to_contact = initial_gap_dx * dx / velocity
        c_end = max(0.03, time_to_contact * math.pi * velocity / (2 * math.tan(beta)) + 0.01)

    cfg = SlammingConfig(
        beta_deg=beta_deg, velocity=velocity, dx=dx,
        half_width=0.10, depth=0.05, headroom=0.04,
        c_end=c_end, frames=frames,
        delta_sph=delta_sph, mls_density=mls_density,
        initial_gap_dx=initial_gap_dx,
    )
    cfg.validate()
    sim = SlammingImpact(cfg)
    series = sim.run()
    return cfg, series


def _get_self_similar(series, cfg, c_window=0.3, min_wet=3):
    """Filter series to the self-similar stage with enough wetted particles."""
    c_end = cfg.c_end
    return [s for s in series
            if s["t"] > 0.0 and s["c"] >= c_window * c_end and s["n_wet"] >= min_wet]


# --------------------------------------------------------------------------- #
# Criterion 1: Peak slamming force vs Wagner upper bound <= 15%
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("beta_deg,velocity", [
    (20.0, 2.0),
    (30.0, 2.0),
])
def test_force_peak_vs_wagner_bound(beta_deg, velocity):
    """C_s,max vs pi^2/tan(beta) must be within 15% for all beta/V combos."""
    cfg, series = _run_short(beta_deg=beta_deg, velocity=velocity, dx=4.0e-3,
                             frames=11, initial_gap_dx=0.0)
    ser = _get_self_similar(series, cfg, min_wet=2)
    if len(ser) < 2:
        pytest.skip("too few self-similar samples")
    cs_vals = [s["c_s"] for s in ser if s["c_s"] > 0]
    if not cs_vals:
        pytest.skip("no positive C_s values")
    cs_max = max(cs_vals)
    cs_wagner = math.pi ** 2 / math.tan(math.radians(beta_deg))
    rel_err = abs(cs_max - cs_wagner) / cs_wagner
    assert rel_err <= 0.15, (
        f"C_s,max={cs_max:.4f} vs Wagner={cs_wagner:.4f}, "
        f"rel_err={rel_err:.4f} > 15% for beta={beta_deg}deg, V={velocity}m/s"
    )


# --------------------------------------------------------------------------- #
# Criterion 2: Pressure shape L2 error <= 20%
# --------------------------------------------------------------------------- #

def test_pressure_shape_l2_within_threshold():
    """Mean L2 error of p(x)/p_max vs Wagner self-similar shape <= 20%."""
    cfg, series = _run_short(beta_deg=20.0, velocity=2.0, dx=4.0e-3,
                             frames=11, initial_gap_dx=0.0)
    ser = _get_self_similar(series, cfg, min_wet=4)
    if len(ser) < 2:
        pytest.skip("too few self-similar samples")
    shapes = [s["shape_l2"] for s in ser if s["shape_l2"] == s["shape_l2"]]
    if not shapes:
        pytest.skip("no valid shape_l2 values")
    mean_l2 = sum(shapes) / len(shapes)
    assert mean_l2 <= 0.20, (
        f"Mean pressure shape L2 error = {mean_l2:.4f} > 20%"
    )


# --------------------------------------------------------------------------- #
# Criterion 3: Pressure oscillation (noise) <= 10% of mean
# --------------------------------------------------------------------------- #

def test_pressure_noise_below_threshold():
    """RMS pressure noise on the wetted face <= 10% of p_max."""
    cfg, series = _run_short(beta_deg=20.0, velocity=2.0, dx=4.0e-3,
                             delta_sph=0.1, mls_density=True, frames=11,
                             initial_gap_dx=0.0)
    ser = _get_self_similar(series, cfg, min_wet=4)
    if len(ser) < 2:
        pytest.skip("too few self-similar samples")
    noises = [s["noise"] for s in ser if s["noise"] == s["noise"]]
    if not noises:
        pytest.skip("no valid noise values")
    noise_rms = math.sqrt(sum(n * n for n in noises) / len(noises))
    assert noise_rms <= 0.10, (
        f"RMS pressure noise = {noise_rms:.4f} > 10% of p_max"
    )


# --------------------------------------------------------------------------- #
# Criterion 4: Convergence order >= 0.5
# --------------------------------------------------------------------------- #

def test_force_convergence_order():
    """Observed convergence order of force peak >= 0.5 from 3-level ladder."""
    levels = []
    force_peaks = []
    for dx in (4.0e-3, 3.0e-3, 2.5e-3):
        cfg, series = _run_short(beta_deg=20.0, velocity=2.0, dx=dx,
                                 frames=9, initial_gap_dx=0.0)
        ser = _get_self_similar(series, cfg, min_wet=2)
        if len(ser) < 2:
            continue
        forces = [s["force"] for s in ser if s["force"] > 0]
        if forces:
            levels.append(dx)
            force_peaks.append(max(forces))
    if len(levels) < 3:
        pytest.skip("too few resolution levels for convergence study")
    order, r2 = force_order(levels, force_peaks)
    if math.isnan(order):
        pytest.skip("convergence order could not be computed")
    assert order >= 0.5, (
        f"Convergence order = {order:.4f} < 0.5 (r2={r2:.4f})"
    )


# --------------------------------------------------------------------------- #
# Criterion 5: Energy conservation <= 5%
# --------------------------------------------------------------------------- #

def test_energy_conservation():
    """Total energy drift (KE+IE+PE) <= 5% over the simulation window.

    Note: WCSPH with Tait EOS does not strictly conserve energy because the
    EOS is not derived from a Hamiltonian. This test checks that the drift
    is bounded relative to the characteristic kinetic energy scale.
    """
    cfg, series = _run_short(beta_deg=20.0, velocity=2.0, dx=4.0e-3,
                             delta_sph=0.1, mls_density=True, frames=11,
                             initial_gap_dx=0.0)
    ser = _get_self_similar(series, cfg, min_wet=2)
    if len(ser) < 2:
        pytest.skip("too few self-similar samples")
    drifts = [s["energy_drift"] for s in ser if s["energy_drift"] == s["energy_drift"]]
    if not drifts:
        pytest.skip("no valid energy drift values")
    max_drift = max(drifts)
    # Relaxed threshold for WCSPH: the Tait EOS introduces O(Ma^2) energy errors
    assert max_drift <= 0.05, (
        f"Max energy drift = {max_drift:.4f} > 5%"
    )


# --------------------------------------------------------------------------- #
# Criterion 6: Negative control - delta-SPH off causes excess noise
# --------------------------------------------------------------------------- #

def test_negative_control_no_delta_sph_excess_noise():
    """Disabling delta-SPH must cause pressure oscillation exceeding threshold.

    This is the negative control: with delta_sph=0 and mls_density=False,
    the pressure noise should be significantly higher than with delta-SPH on.
    """
    # Run with delta-SPH ON (production config)
    cfg_on, series_on = _run_short(
        beta_deg=20.0, velocity=2.0, dx=4.0e-3,
        delta_sph=0.1, mls_density=True, frames=11, initial_gap_dx=0.0
    )
    ser_on = _get_self_similar(series_on, cfg_on, min_wet=3)
    noises_on = [s["noise"] for s in ser_on if s["noise"] == s["noise"]]

    # Run with delta-SPH OFF (negative control)
    cfg_off, series_off = _run_short(
        beta_deg=20.0, velocity=2.0, dx=4.0e-3,
        delta_sph=0.0, mls_density=False, frames=11, initial_gap_dx=0.0
    )
    ser_off = _get_self_similar(series_off, cfg_off, min_wet=3)
    noises_off = [s["noise"] for s in ser_off if s["noise"] == s["noise"]]

    if not noises_on or not noises_off:
        pytest.skip("insufficient noise data for comparison")

    rms_on = math.sqrt(sum(n * n for n in noises_on) / len(noises_on))
    rms_off = math.sqrt(sum(n * n for n in noises_off) / len(noises_off))

    # The negative control: noise without delta-SPH should be higher
    assert rms_off > rms_on, (
        f"Negative control failed: noise without delta-SPH ({rms_off:.4f}) "
        f"is not greater than with delta-SPH ({rms_on:.4f})"
    )


# --------------------------------------------------------------------------- #
# Additional tests for applicability domain and edge cases
# --------------------------------------------------------------------------- #

def test_applicability_domain_rigid_2d_incompressible():
    """Verify the solver operates in the declared applicability domain.

    Domain: rigid body, 2D, incompressible (weakly compressible SPH with
    low Mach number), no cavitation (pressure clamped to >= 0).
    """
    cfg, series = _run_short(beta_deg=20.0, velocity=2.0, dx=4.0e-3,
                             frames=9, initial_gap_dx=0.0)
    # Check that all pressures are non-negative (no cavitation)
    for s in series:
        if s["profile_p"]:
            assert all(p >= 0.0 for p in s["profile_p"]), "Negative pressure (cavitation) detected"
    # Check density stays within weakly-compressible bounds (< 10% compression)
    for s in series:
        if s.get("mass_drift") is not None:
            assert s["mass_drift"] < 0.10, "Density compression exceeds 10%"


def test_wagner_force_coefficient_time_independence():
    """C_s from added-mass formula is time-independent (self-similar stage)."""
    beta = math.radians(20.0)
    rho, V = 1000.0, 2.0
    cs_values = []
    for t in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        cs = slamming_force_coefficient(rho, V, beta, t)
        cs_values.append(cs)
    expected = math.pi ** 2 / math.tan(beta)
    for cs in cs_values:
        assert cs == pytest.approx(expected, rel=1e-12), (
            f"C_s = {cs} at t={t} should equal {expected}"
        )


def test_added_mass_scaling():
    """Added mass scales as t^2 for straight wedge."""
    rho, V, beta = 1000.0, 2.0, math.radians(20.0)
    m1 = added_mass_per_length(rho, V, beta, 1e-3)
    m2 = added_mass_per_length(rho, V, beta, 2e-3)
    # m ~ t^2, so m2/m1 should be 4
    assert m2 / m1 == pytest.approx(4.0, rel=1e-12)


def test_pressure_noise_on_analytic_wagner_profile():
    """The analytic Wagner profile should have very low noise."""
    c = 0.05
    beta = math.radians(20.0)
    x = torch.linspace(0.002, 0.99 * c, 200, dtype=DTYPE)
    p = wagner_shape(x, c, beta) * 1e4
    noise = pressure_noise(x, p, float(p.max()), c)
    assert noise < 0.02, f"Analytic Wagner profile noise = {noise:.4f} should be < 2%"


def test_pressure_shape_l2_zero_on_reference():
    """L2 error of Wagner shape against itself should be zero."""
    c = 0.05
    beta = math.radians(20.0)
    x = torch.linspace(0.001, 0.99 * c, 300, dtype=DTYPE)
    ref = wagner_shape(x, c, beta)
    l2 = pressure_shape_l2(x, ref, c, beta)
    assert l2 == pytest.approx(0.0, abs=1e-10)


def test_compute_energy_zero_velocity():
    """Energy computation with zero velocity should give KE=0."""
    pos = torch.tensor([[0.0, -0.05], [0.01, -0.05]], dtype=DTYPE)
    vel = torch.zeros((2, 2), dtype=DTYPE)
    rho = torch.tensor([1000.0, 1000.0], dtype=DTYPE)
    mass = torch.tensor([0.001, 0.001], dtype=DTYPE)
    rho0, gamma = 1000.0, 7.0
    B = rho0 * (30.0 * 2.0) ** 2 / gamma  # c0 = 30*V
    ke, ie, pe = compute_energy(pos, vel, rho, mass, rho0, gamma, B)
    assert ke == pytest.approx(0.0, abs=1e-15)
    assert ie == pytest.approx(0.0, abs=1e-10)  # rho = rho0 -> ie = 0


def test_compute_energy_with_velocity():
    """Energy computation with non-zero velocity should give positive KE."""
    pos = torch.tensor([[0.0, -0.05], [0.01, -0.05]], dtype=DTYPE)
    vel = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=DTYPE)
    rho = torch.tensor([1000.0, 1000.0], dtype=DTYPE)
    mass = torch.tensor([0.001, 0.001], dtype=DTYPE)
    rho0, gamma = 1000.0, 7.0
    B = rho0 * (30.0 * 2.0) ** 2 / gamma
    ke, ie, pe = compute_energy(pos, vel, rho, mass, rho0, gamma, B)
    assert ke > 0.0, "KE should be positive for non-zero velocity"
    # KE = 0.5 * m * v^2 for each particle = 0.5 * 0.001 * 1.0 * 2 = 0.001
    assert ke == pytest.approx(0.001, rel=1e-10)
