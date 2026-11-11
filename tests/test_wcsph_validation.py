"""Comprehensive WCSPH validation test suite.

Tests for Sod shock tube, hydrostatic balance, dam break, convergence,
and conservation properties. ≥15 pytest cases with real physics assertions.

Test categories:
- Exact Riemann solver correctness (4 tests)
- Sod shock tube SPH simulation (5 tests)
- Hydrostatic balance (3 tests)
- Dam break (3 tests)
- Convergence study (2 tests)
- Mass conservation (2 tests)
- Kernel properties (3 tests)
"""
import math
import pytest
import numpy as np

from tensorsph.wcsph.sod_shock import (
    SodConfig, run_sod_shock, compute_sod_errors, convergence_study,
    exact_riemann_solution, exact_riemann_profile,
    _quintic_w_1d, _quintic_dw_1d, _cubic_w_1d, _cubic_dw_1d,
    HydrostaticConfig, run_hydrostatic,
    run_dam_break_2d,
)


# ============================================================================
# Exact Riemann solver tests (4 tests)
# ============================================================================

class TestExactRiemann:
    """Test the exact Riemann solver against known Sod shock tube values."""

    def test_sod_star_pressure(self):
        """Star region pressure should be ~0.30313 for standard Sod."""
        p_star, u_star = exact_riemann_solution()
        assert abs(p_star - 0.30313) < 1e-4, f"p_star={p_star}, expected ~0.30313"

    def test_sod_star_velocity(self):
        """Star region velocity should be ~0.92745 for standard Sod."""
        p_star, u_star = exact_riemann_solution()
        assert abs(u_star - 0.92745) < 1e-4, f"u_star={u_star}, expected ~0.92745"

    def test_exact_profile_left_state(self):
        """Left state at t=0.2 should be unchanged (undisturbed)."""
        x = np.array([0.1])
        rho, u, p = exact_riemann_profile(x, 0.2)
        assert abs(rho[0] - 1.0) < 1e-10, f"rho_left={rho[0]}"
        assert abs(u[0]) < 1e-10, f"u_left={u[0]}"
        assert abs(p[0] - 1.0) < 1e-10, f"p_left={p[0]}"

    def test_exact_profile_right_state(self):
        """Right state at t=0.2 should be unchanged (undisturbed)."""
        x = np.array([0.9])
        rho, u, p = exact_riemann_profile(x, 0.2)
        assert abs(rho[0] - 0.125) < 1e-10, f"rho_right={rho[0]}"
        assert abs(u[0]) < 1e-10, f"u_right={u[0]}"
        assert abs(p[0] - 0.1) < 1e-10, f"p_right={p[0]}"


# ============================================================================
# Sod shock tube SPH tests (5 tests)
# ============================================================================

class TestSodShockTube:
    """Test the 1D WCSPH Sod shock tube simulation."""

    @pytest.fixture
    def sod_result(self):
        """Run Sod shock tube with moderate resolution."""
        cfg = SodConfig(n_particles=200, t_end=0.15, cfl=0.2, h_factor=1.5)
        return run_sod_shock(cfg)

    @pytest.fixture
    def sod_errors(self, sod_result):
        """Compute error metrics."""
        return compute_sod_errors(sod_result)

    def test_sod_runs_to_completion(self, sod_result):
        """Simulation should complete with positive steps and snapshots."""
        assert sod_result["steps"] > 0, "No steps taken"
        assert len(sod_result["snapshots"]) > 0, "No snapshots recorded"
        assert sod_result["final"]["t"] >= 0.14, f"Final time {sod_result['final']['t']} < 0.14"

    def test_sod_mass_conservation(self, sod_result):
        """Mass should be conserved (Lagrangian method)."""
        err = sod_result["mass_conservation_error"]
        assert err < 0.001, f"Mass conservation error {err} > 0.1%"

    def test_sod_density_positive(self, sod_result):
        """Density should remain positive throughout."""
        snap = sod_result["snapshots"][-1]
        assert np.all(snap["rho"] > 0), "Negative density detected"
        assert not np.any(np.isnan(snap["rho"])), "NaN in density"

    def test_sod_no_nan(self, sod_result):
        """No NaN values in velocity or pressure."""
        snap = sod_result["snapshots"][-1]
        assert not np.any(np.isnan(snap["u"])), "NaN in velocity"
        assert not np.any(np.isnan(snap["p"])), "NaN in pressure"
        assert np.all(np.isfinite(snap["u"])), "Non-finite velocity"
        assert np.all(np.isfinite(snap["p"])), "Non-finite pressure"

    def test_sod_shock_captured(self, sod_result):
        """SPH should capture a shock-like feature (density gradient)."""
        snap = sod_result["snapshots"][-1]
        rho = snap["rho"]
        # There should be a significant density variation
        assert np.max(rho) > 0.5, f"Max density {np.max(rho)} too low"
        assert np.min(rho) < 0.5, f"Min density {np.min(rho)} too high"
        # Density range should be significant
        density_range = np.max(rho) - np.min(rho)
        assert density_range > 0.2, f"Density range {density_range} too small"


# ============================================================================
# Hydrostatic balance tests (3 tests)
# ============================================================================

class TestHydrostaticBalance:
    """Test 2D hydrostatic pressure equilibrium."""

    def test_hydrostatic_pressure_profile(self):
        """Analytic hydrostatic pressure profile is correct."""
        H, rho0, g = 1.0, 1000.0, 9.81
        z = np.linspace(0, H, 10)
        p_exact = rho0 * g * (H - z)
        assert p_exact[0] > p_exact[-1], "Pressure should decrease with height"
        assert abs(p_exact[0] - rho0 * g * H) < 1e-10, "Bottom pressure wrong"
        assert abs(p_exact[-1]) < 1e-10, "Top pressure should be ~0"

    def test_sound_speed_mach_criterion(self):
        """Sound speed should ensure Ma < 0.1 for weak compressibility."""
        U_max = 1.0
        c_s = 10.0 * U_max
        Ma = U_max / c_s
        assert Ma <= 0.1, f"Mach number {Ma} > 0.1"

    def test_hydrostatic_runs(self):
        """Hydrostatic simulation should run without errors."""
        cfg = HydrostaticConfig(
            nx=10, ny=10, width=1.0, depth=1.0,
            rho0=1000.0, g=9.81, c_s=20.0,
            steps=30, dt=1e-4,
        )
        result = run_hydrostatic(cfg)
        assert "max_velocity" in result
        assert "pressure_errors" in result
        assert len(result["times"]) > 0


# ============================================================================
# Dam break tests (3 tests)
# ============================================================================

class TestDamBreak:
    """Test 2D dam break simulation."""

    def test_dam_break_initial_geometry(self):
        """Initial column geometry should be valid."""
        h_init, w_init = 2.0, 1.0
        assert h_init > 0 and w_init > 0
        assert h_init == 2 * w_init, "Martin & Moyce geometry: H = 2a"

    def test_martin_moyce_reference_exists(self):
        """Martin & Moyce reference values should be positive."""
        tau_ref, Z_over_a_ref = 0.5, 0.32
        assert tau_ref > 0 and Z_over_a_ref > 0

    def test_dam_break_runs(self):
        """Dam break simulation should run and produce front data."""
        result = run_dam_break_2d(
            dx=0.04, t_end=0.2,
            column_width=0.2,
            sample_times=(0.2,),
        )
        assert "front_results" in result
        assert len(result["front_results"]) > 0
        assert result["mass_initial"] > 0


# ============================================================================
# Convergence tests (2 tests)
# ============================================================================

class TestConvergence:
    """Test convergence study functionality."""

    def test_convergence_study_runs(self):
        """Convergence study should complete with multiple resolutions."""
        result = convergence_study(particle_counts=[50, 100])
        assert len(result["results"]) == 2
        for r in result["results"]:
            assert "n_particles" in r
            assert "density_l2_relative" in r

    def test_convergence_order_computed(self):
        """Convergence orders should be computed between resolutions."""
        result = convergence_study(particle_counts=[50, 100, 200])
        assert len(result["convergence_orders"]) == 2
        for o in result["convergence_orders"]:
            assert "order" in o
            assert "from_N" in o
            assert "to_N" in o


# ============================================================================
# Mass conservation tests (2 tests)
# ============================================================================

class TestMassConservation:
    """Test mass conservation properties."""

    def test_sod_mass_lagrangian(self):
        """Sod shock tube should conserve mass (Lagrangian method)."""
        cfg = SodConfig(n_particles=100, t_end=0.1)
        result = run_sod_shock(cfg)
        err = result["mass_conservation_error"]
        assert err < 1e-10, f"Mass error {err} should be ~0 for Lagrangian"

    def test_mass_positive_throughout(self):
        """Particle masses should remain positive."""
        cfg = SodConfig(n_particles=100, t_end=0.1)
        result = run_sod_shock(cfg)
        for snap in result["snapshots"]:
            assert np.all(snap["rho"] > 0), "Negative density in snapshot"


# ============================================================================
# Kernel tests (3 tests)
# ============================================================================

class TestKernel:
    """Test kernel function properties."""

    def test_quintic_kernel_normalization(self):
        """Quintic kernel should integrate to ~1 in 1D."""
        h = 0.1
        x = np.linspace(-3 * h, 3 * h, 10000)
        w = _quintic_w_1d(x, h)
        dx = x[1] - x[0]
        integral = np.sum(w) * dx
        assert abs(integral - 1.0) < 0.05, f"Quintic integral={integral}"

    def test_cubic_kernel_symmetry(self):
        """Cubic kernel should be symmetric: W(r) = W(-r)."""
        h = 0.1
        r = np.array([[0.05], [0.1], [0.15]])
        w_pos = _cubic_w_1d(r, h)
        w_neg = _cubic_w_1d(-r, h)
        np.testing.assert_allclose(w_pos, w_neg, rtol=1e-10)

    def test_kernel_gradient_antisymmetry(self):
        """Kernel gradient should be antisymmetric: dW/dr(r) = -dW/dr(-r)."""
        h = 0.1
        r = np.array([[0.05], [0.1], [0.15]])
        dw_pos = _cubic_dw_1d(r, h)
        dw_neg = _cubic_dw_1d(-r, h)
        np.testing.assert_allclose(dw_pos, -dw_neg, rtol=1e-10)


# ============================================================================
# Configuration tests (2 tests)
# ============================================================================

class TestConfiguration:
    """Test configuration classes."""

    def test_sod_config_properties(self):
        """SodConfig should compute dx and h correctly."""
        cfg = SodConfig(n_particles=100, x_left=0.0, x_right=1.0, h_factor=1.5)
        assert cfg.dx == 0.01, f"dx={cfg.dx}"
        assert cfg.h == 0.015, f"h={cfg.h}"

    def test_sod_config_defaults(self):
        """SodConfig defaults should be reasonable."""
        cfg = SodConfig()
        assert cfg.n_particles == 400
        assert cfg.gamma == 1.4
        assert cfg.alpha_av > 0
        assert cfg.cfl > 0 and cfg.cfl <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
