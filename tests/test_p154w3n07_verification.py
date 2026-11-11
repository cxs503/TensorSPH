"""P154W3-N07: Free-surface breaking wave & comprehensive verification tests.

Campaign node: SPH自由面破碎波与液舱晃荡验证面
Four verification anchors:
  (1) Ritter dry-bed dam break: x(t) = 2*sqrt(g*h0)*t
  (2) Solitary wave sech² propagation: eta = A*sech²[alpha*(x-ct)]
  (3) Liquid tank sloshing: omega_1 = sqrt(pi*g/L * tanh(pi*h/L))
  (4) Breaking wave: H/L > 0.14 threshold, energy dissipation

Acceptance criteria (from gates.py):
  C1: Ritter front error < 10%
  C2: Solitary wave decay < 5%
  C3: Sloshing frequency error < 10%
  C4: Convergence slope ~ 1.0 ± 0.2
  C5: Mass drift < 1%
  C6: Breaking differential dissipation > 5%
"""

import math
import pytest
import torch

# ============================================================================
# (1) Ritter dry-bed dam break analytical anchor
# ============================================================================

class TestRitterDamBreak:
    """Ritter (1892) shallow-water solution for dry-bed dam break."""

    def test_ritter_front_formula(self):
        """Ritter front position: x_front = dam + 2*sqrt(g*h0)*t."""
        from tensorsph.freesurface.ritter_anchor import RitterCase
        case = RitterCase(cells_per_depth=10.0)
        g = case.gravity
        h0 = case.depth0
        c0 = math.sqrt(g * h0)
        # Front should advance at 2*c0 from the dam position
        dam_x = case.column_width  # dam is at the column edge
        t = 0.1
        expected_front = dam_x + 2.0 * c0 * t
        assert expected_front > dam_x, "front must advance downstream"
        # Verify the formula x(t) = 2*sqrt(g*h0)*t (distance from dam)
        distance = 2.0 * c0 * t
        assert distance == pytest.approx(2.0 * math.sqrt(g * h0) * t, rel=1e-10)

    def test_ritter_fan_structure(self):
        """Ritter solution has a rarefaction fan from tail to front."""
        from tensorsph.freesurface import ritter_anchor
        g = 9.81
        h0 = 0.4
        c0 = math.sqrt(g * h0)
        dam = 0.5
        t = 0.15
        # Fan extends from x_tail = dam - c0*t to x_front = dam + 2*c0*t
        x_tail = dam - c0 * t
        x_front = dam + 2.0 * c0 * t
        assert x_tail < dam, "fan tail must be upstream of dam"
        assert x_front > dam, "fan front must be downstream of dam"
        # Depth at dam: h(dam, t) = (2*c0/(3*g))^2 * g = 4*h0/9
        depth_at_dam = ritter_anchor.ritter.depth(
            torch.tensor([dam], dtype=torch.float64), t, h0,
            dam_x=dam, gravity=g)
        expected = 4.0 * h0 / 9.0
        assert float(depth_at_dam[0]) == pytest.approx(expected, rel=1e-6)

    def test_ritter_released_volume(self):
        """Volume released by Ritter solution: V = h0*c0*t."""
        from tensorsph.freesurface import ritter_anchor
        g = 9.81
        h0 = 0.4
        c0 = math.sqrt(g * h0)
        t = 0.2
        vol = ritter_anchor.ritter.released_volume(t, h0, g)
        expected = h0 * c0 * t
        assert vol == pytest.approx(expected, rel=1e-10)

    def test_ritter_case_construction(self):
        """RitterCase builds with correct parameters."""
        from tensorsph.freesurface.ritter_anchor import RitterCase
        case = RitterCase(cells_per_depth=10.0, viscosity=1e-4)
        assert case.cells_per_depth == 10.0
        assert case.dx > 0
        assert case.depth0 > 0
        assert case.gravity > 0

    def test_ritter_velocity_profile(self):
        """Ritter velocity in fan: u = (2/3)*(c0 + xi) where xi = (x-dam)/t."""
        g = 9.81
        h0 = 0.4
        c0 = math.sqrt(g * h0)
        dam = 0.5
        t = 0.1
        # At the front (x = dam + 2*c0*t): xi = 2*c0, u = (2/3)*(c0 + 2*c0) = 2*c0
        x_front = dam + 2.0 * c0 * t
        xi_front = (x_front - dam) / t
        u_front = (2.0 / 3.0) * (c0 + xi_front)
        assert u_front == pytest.approx(2.0 * c0, rel=1e-10)
        # At the dam (x = dam): xi = 0, u = (2/3)*c0
        u_dam = (2.0 / 3.0) * c0
        assert u_dam > 0


# ============================================================================
# (2) Solitary wave sech² propagation
# ============================================================================

class TestSolitaryWave:
    """Solitary wave: eta(x,t) = A*sech²[alpha*(x-ct)], c = sqrt(g*(h+A))."""

    def test_solitary_wave_elevation_formula(self):
        """Verify sech² profile shape."""
        from tensorsph.wave_flume import solitary_wave_elevation, solitary_alpha
        g = 9.81
        h = 0.5
        A = 0.1  # wave amplitude
        alpha = solitary_alpha(A, h)
        # At x = 0, t = 0: eta = A * sech²(0) = A
        eta_0 = solitary_wave_elevation(0.0, 0.0, A, h, g)
        assert eta_0 == pytest.approx(A, rel=1e-10)
        # At large x: eta -> 0
        eta_far = solitary_wave_elevation(100.0, 0.0, A, h, g)
        assert abs(eta_far) < 1e-6

    def test_solitary_wave_celerity(self):
        """Solitary wave speed: c = sqrt(g*(h+A))."""
        from tensorsph.wave_flume import solitary_wave_celerity
        g = 9.81
        h = 0.5
        A = 0.1
        c = solitary_wave_celerity(A, h, g)
        expected = math.sqrt(g * (h + A))
        assert c == pytest.approx(expected, rel=1e-10)

    def test_solitary_alpha_parameter(self):
        """alpha = sqrt(3*A/(4*h³))."""
        from tensorsph.wave_flume import solitary_alpha
        A = 0.1
        h = 0.5
        alpha = solitary_alpha(A, h)
        expected = math.sqrt(3.0 * A / (4.0 * h ** 3))
        assert alpha == pytest.approx(expected, rel=1e-10)

    def test_solitary_wave_symmetry(self):
        """Solitary wave is symmetric about its crest."""
        from tensorsph.wave_flume import solitary_wave_elevation
        g = 9.81
        h = 0.5
        A = 0.08
        # At t=0, crest at x=0
        eta_left = solitary_wave_elevation(-1.0, 0.0, A, h, g)
        eta_right = solitary_wave_elevation(1.0, 0.0, A, h, g)
        assert eta_left == pytest.approx(eta_right, rel=1e-10)

    def test_solitary_case_construction(self):
        """SolitaryCase builds correctly."""
        from tensorsph.freesurface.solitary_anchor import SolitaryCase
        case = SolitaryCase(height=0.05, cells_per_depth=15.0)
        assert case.height > 0
        assert case.dx > 0
        assert case.celerity > 0
        assert case.alpha > 0


# ============================================================================
# (3) Liquid tank sloshing first-order frequency
# ============================================================================

class TestSloshingFrequency:
    """Sloshing: omega_n = sqrt(k_n * g * tanh(k_n * h)) where k_n = n*pi/L."""

    def test_sloshing_frequency_formula(self):
        """Verify the analytical frequency formula."""
        from tensorsph.freesurface.sloshing_mode import SloshingModeCase, analytic_omega
        case = SloshingModeCase(cells_per_depth=20.0)
        L = case.tank_length
        h = case.water_depth
        g = case.gravity
        # omega_1 = sqrt(k_1 * g * tanh(k_1 * h)) where k_1 = pi/L
        k1 = math.pi / L
        omega_analytic = math.sqrt(k1 * g * math.tanh(k1 * h))
        assert case.omega1 == pytest.approx(omega_analytic, rel=1e-10)
        # Also verify via analytic_omega function
        assert case.omega1 == pytest.approx(analytic_omega(L, h, 1, g), rel=1e-10)

    def test_sloshing_period_formula(self):
        """Period T_1 = 2*pi/omega_1."""
        from tensorsph.freesurface.sloshing_mode import SloshingModeCase
        case = SloshingModeCase(cells_per_depth=20.0)
        T1 = 2.0 * math.pi / case.omega1
        assert case.period1 == pytest.approx(T1, rel=1e-10)

    def test_sloshing_shallow_water_limit(self):
        """Shallow water: omega_1 -> sqrt(k_1 * g * k_1 * h) = k_1 * sqrt(g*h)."""
        g = 9.81
        L = 2.0
        h = 0.05  # h/L = 0.025 << 1
        k1 = math.pi / L
        # tanh(k1*h) ≈ k1*h for small k1*h
        omega_approx = k1 * math.sqrt(g * h)
        omega_exact = math.sqrt(k1 * g * math.tanh(k1 * h))
        assert omega_approx == pytest.approx(omega_exact, rel=0.02)

    def test_sloshing_deep_water_limit(self):
        """Deep water: omega_1 -> sqrt(k_1 * g) (tanh -> 1)."""
        g = 9.81
        L = 0.5
        h = 5.0  # h/L = 10 >> 1
        k1 = math.pi / L
        omega_approx = math.sqrt(k1 * g)
        omega_exact = math.sqrt(k1 * g * math.tanh(k1 * h))
        assert omega_approx == pytest.approx(omega_exact, rel=1e-6)

    def test_sloshing_mode_case_parameters(self):
        """SloshingModeCase has correct physical parameters."""
        from tensorsph.freesurface.sloshing_mode import SloshingModeCase
        case = SloshingModeCase(
            tank_length=2.0, water_depth=0.5, cells_per_depth=15.0)
        assert case.tank_length == 2.0
        assert case.water_depth == 0.5
        assert case.omega1 > 0
        assert case.period1 > 0
        assert case.amplitude > 0

    def test_sloshing_multiple_modes_frequency_ratio(self):
        """Mode n frequency: omega_n = sqrt(k_n * g * tanh(k_n * h)).
        
        For finite depth, higher modes have higher frequency.
        """
        from tensorsph.freesurface.sloshing_mode import analytic_omega
        L = 2.0
        h = 0.5
        g = 9.81
        # Mode 1
        omega1 = analytic_omega(L, h, 1, g)
        # Mode 2
        omega2 = analytic_omega(L, h, 2, g)
        # omega2 > omega1 for any depth
        assert omega2 > omega1, "higher modes must have higher frequency"
        # The ratio depends on depth
        ratio = omega2 / omega1
        assert ratio > 1.0


# ============================================================================
# (4) Breaking wave verification
# ============================================================================

class TestBreakingWave:
    """Breaking wave: McCowan (1894) H/L > 0.14 criterion."""

    def test_breaking_threshold_formula(self):
        """Breaking threshold: H_break = 0.14 * L."""
        from tensorsph.freesurface.breaking_wave import breaking_threshold_height
        L = 1.5
        H_break = breaking_threshold_height(0.5, L)
        assert H_break == pytest.approx(0.14 * L, rel=1e-10)

    def test_steepness_criterion(self):
        """Wave is breaking when H/L > 0.14."""
        from tensorsph.freesurface.breaking_wave import BreakingCase
        # Sub-critical
        case_sub = BreakingCase(depth=0.5, kh=2.0, wave_height=0.05)
        assert not case_sub.is_breaking
        assert case_sub.steepness < 0.14
        # Super-critical
        case_super = BreakingCase(depth=0.5, kh=2.0, wave_height=0.25)
        assert case_super.is_breaking
        assert case_super.steepness > 0.14

    def test_group_velocity_formula(self):
        """Group velocity: c_g = c * (1/2 + kh/sinh(2kh))."""
        from tensorsph.freesurface.breaking_wave import group_velocity
        g = 9.81
        h = 0.5
        k = 2.0 / h  # kh = 2
        cg = group_velocity(k, h, g)
        # Deep water limit: c_g = c/2
        omega = math.sqrt(g * k * math.tanh(k * h))
        c = omega / k
        kh = k * h
        n = 0.5 + kh / math.sinh(2 * kh)
        expected = c * n
        assert cg == pytest.approx(expected, rel=1e-10)

    def test_energy_flux_formula(self):
        """Energy flux: P = (1/8)*rho*g*H²*c_g."""
        from tensorsph.freesurface.breaking_wave import linear_energy_flux
        H = 0.05
        k = 4.0
        h = 0.5
        rho = 1000.0
        g = 9.81
        P = linear_energy_flux(H, k, h, rho, g)
        from tensorsph.freesurface.breaking_wave import group_velocity
        cg = group_velocity(k, h, g)
        expected = 0.125 * rho * g * H ** 2 * cg
        assert P == pytest.approx(expected, rel=1e-10)

    def test_wave_height_measurement(self):
        """Wave height from gauge: H = 4*std(eta)."""
        from tensorsph.freesurface.breaking_wave import measure_wave_height
        # Monochromatic wave: eta = A*sin(omega*t)
        A = 0.05
        omega = 2.0 * math.pi / 1.0  # T = 1s
        n = 1000
        times = [i * 0.01 for i in range(n)]
        series = [A * math.sin(omega * t) for t in times]
        H_meas = measure_wave_height(series, times, skip_fraction=0.1)
        # For a sine wave, 4*std = 4*A/sqrt(2) = 2*sqrt(2)*A ≈ 2.83*A
        # But H_significant for monochromatic = 2*A (crest to trough)
        # 4*std of sin = 4*A/sqrt(2) ≈ 2.83*A (slightly overestimates)
        assert H_meas > 0
        assert H_meas == pytest.approx(4 * A / math.sqrt(2), rel=0.05)

    def test_breaking_case_config(self):
        """BreakingCase builds a valid WaveTankConfig."""
        from tensorsph.freesurface.breaking_wave import BreakingCase
        case = BreakingCase(depth=0.5, kh=2.0, wave_height=0.05,
                            cells_per_depth=30.0)  # dx = 0.0167 < 4% of depth
        cfg = case.config()
        assert cfg.depth == 0.5
        assert cfg.wave_height == 0.05
        assert cfg.dx > 0

    def test_breaking_pair_structure(self):
        """Sub-critical vs super-critical: verify structure without full run."""
        from tensorsph.freesurface.breaking_wave import BreakingCase
        # Use fine resolution for wave tank validity (dx < 4% of depth)
        case_sub = BreakingCase(depth=0.5, kh=2.0, wave_height=0.03,
                                cells_per_depth=30.0, tank_lengths=8.0)
        case_super = BreakingCase(depth=0.5, kh=2.0, wave_height=0.25,
                                  cells_per_depth=30.0, tank_lengths=8.0)
        # Verify case properties
        assert case_sub.steepness < 0.14
        assert case_super.steepness > 0.14
        assert case_sub.wavelength == case_super.wavelength
        # Verify configs are valid
        cfg_sub = case_sub.config()
        cfg_super = case_super.config()
        assert cfg_sub.dx <= 0.04 * cfg_sub.depth  # dx <= 4% of depth
        assert cfg_super.dx <= 0.04 * cfg_super.depth


# ============================================================================
# (5) Gates and acceptance criteria
# ============================================================================

class TestGates:
    """Acceptance gates evaluation."""

    def test_all_criteria_defined(self):
        """All six criteria are defined."""
        from tensorsph.freesurface.gates import CRITERIA
        ids = [c.identifier for c in CRITERIA]
        assert "C1" in ids  # Ritter front
        assert "C2" in ids  # Solitary decay
        assert "C3" in ids  # Sloshing frequency
        assert "C4" in ids  # Convergence slope
        assert "C5" in ids  # Mass conservation
        assert "C6" in ids  # Breaking dissipation

    def test_evaluate_pass(self):
        """All-pass evaluation."""
        from tensorsph.freesurface.gates import evaluate, maturity
        measured = {"C1": 0.05, "C2": 0.02, "C3": 0.05, "C4": 1.0,
                    "C5": 0.001, "C6": 0.03}
        report = evaluate(measured)
        verdict = maturity(report)
        assert verdict["verdict"] == "PASS"

    def test_evaluate_fail(self):
        """One failing criterion."""
        from tensorsph.freesurface.gates import evaluate, maturity
        measured = {"C1": 0.15, "C2": 0.02, "C3": 0.05, "C4": 1.0,
                    "C5": 0.001, "C6": 0.03}
        report = evaluate(measured)
        verdict = maturity(report)
        assert verdict["verdict"] == "HONEST_FAIL_OPEN"
        assert "C1" in verdict["failed"]

    def test_evaluate_unmeasured(self):
        """Missing criterion is UNMEASURED."""
        from tensorsph.freesurface.gates import evaluate
        measured = {"C1": 0.05}
        report = evaluate(measured)
        c2_row = [r for r in report["gates"] if r["id"] == "C2"][0]
        assert c2_row["status"] == "UNMEASURED"


# ============================================================================
# (6) Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests across all four verification anchors."""

    def test_all_modules_importable(self):
        """All four verification modules import cleanly."""
        from tensorsph.freesurface import (
            ritter_anchor,
            solitary_anchor,
            sloshing_mode,
            breaking_wave,
            gates,
            profiles,
        )
        assert ritter_anchor is not None
        assert solitary_anchor is not None
        assert sloshing_mode is not None
        assert breaking_wave is not None

    def test_profiles_measurement_operators(self):
        """Measurement operators work on synthetic data."""
        from tensorsph.freesurface import profiles
        # Synthetic dam-break fan: linear depth profile
        n = 100
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        depth = torch.where(x < 0.5, 0.4 * (1 - x / 0.5), 0.0)
        mass = depth * 0.01 * 1000.0  # rho0 * dx * depth
        positions = torch.stack([x, torch.zeros_like(x)], dim=1)
        centres, d = profiles.binned_depth(positions, mass, 1000.0, 0.01)
        assert centres.numel() > 0
        assert d.sum() > 0

    def test_loglog_slope(self):
        """Log-log slope computation."""
        from tensorsph.freesurface.profiles import loglog_slope
        # Perfect first-order convergence: error = C * dx
        dx = [0.04, 0.02, 0.01]
        errors = [0.04, 0.02, 0.01]
        slope = loglog_slope(dx, errors)
        assert slope == pytest.approx(1.0, abs=0.01)

    def test_oscillation_frequency(self):
        """Frequency from time series (returns angular frequency omega)."""
        from tensorsph.freesurface.profiles import oscillation_frequency
        # Sine wave at 1 Hz -> omega = 2*pi rad/s
        T = 1.0
        times = [i * 0.01 for i in range(500)]
        signal = [math.sin(2 * math.pi * t / T) for t in times]
        omega = oscillation_frequency(times, signal)
        # oscillation_frequency returns angular frequency
        assert omega == pytest.approx(2 * math.pi, rel=0.05)
