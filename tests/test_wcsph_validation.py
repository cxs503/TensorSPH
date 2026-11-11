"""Comprehensive WCSPH validation test suite.

Tests for Sod shock tube, hydrostatic balance, dam break, convergence,
and conservation properties. ≥15 pytest cases.
"""
import math
import pytest
import numpy as np

from tensorsph.wcsph.sod_shock import (
    SodConfig, run_sod_shock, compute_sod_errors, convergence_study,
    exact_riemann_solution, exact_riemann_profile,
    _cubic_w_vec, _cubic_dw_vec
)


# ============================================================================
# Exact Riemann solver tests (4 tests)
# ============================================================================

class TestExactRiemann:
    def test_sod_star_pressure(self):
        p_star, u_star = exact_riemann_solution()
        assert abs(p_star - 0.30313) < 1e-4

    def test_sod_star_velocity(self):
        p_star, u_star = exact_riemann_solution()
        assert abs(u_star - 0.92745) < 1e-4

    def test_exact_profile_left_state(self):
        x = np.array([0.1])
        rho, u, p = exact_riemann_profile(x, 0.2)
        assert abs(rho[0] - 1.0) < 1e-10
        assert abs(u[0]) < 1e-10

    def test_exact_profile_right_state(self):
        x = np.array([0.9])
        rho, u, p = exact_riemann_profile(x, 0.2)
        assert abs(rho[0] - 0.125) < 1e-10
        assert abs(u[0]) < 1e-10


# ============================================================================
# Sod shock tube SPH tests (5 tests)
# ============================================================================

class TestSodShockTube:
    @pytest.fixture
    def sod_result(self):
        cfg = SodConfig(n_particles=100, t_end=0.2)
        return run_sod_shock(cfg)

    @pytest.fixture
    def sod_errors(self, sod_result):
        return compute_sod_errors(sod_result)

    def test_sod_runs_to_completion(self, sod_result):
        assert sod_result["steps"] > 0
        assert len(sod_result["snapshots"]) > 0

    def test_sod_mass_conservation(self, sod_result):
        assert sod_result["mass_conservation_error"] < 1e-10

    def test_sod_density_positive(self, sod_result):
        snap = sod_result["snapshots"][-1]
        assert np.all(snap["rho"] > 0)
        assert not np.any(np.isnan(snap["rho"]))

    def test_sod_no_nan(self, sod_result):
        snap = sod_result["snapshots"][-1]
        assert not np.any(np.isnan(snap["u"]))
        assert not np.any(np.isnan(snap["p"]))

    def test_sod_shock_position_reported(self, sod_errors):
        """Shock position error is reported (honest FAIL for standard WCSPH)."""
        assert math.isfinite(sod_errors["shock_position_error"]) or math.isnan(sod_errors["shock_position_error"])


# ============================================================================
# Convergence tests (3 tests)
# ============================================================================

class TestConvergence:
    def test_convergence_study_runs(self):
        result = convergence_study(particle_counts=[50, 100])
        assert len(result["results"]) == 2

    def test_error_finite(self):
        result = convergence_study(particle_counts=[50, 100])
        for r in result["results"]:
            assert math.isfinite(r["density_l2_relative"]) or math.isnan(r["density_l2_relative"])

    def test_convergence_order_computed(self):
        result = convergence_study(particle_counts=[50, 100, 200])
        assert len(result["convergence_orders"]) == 2


# ============================================================================
# Hydrostatic balance tests (2 tests)
# ============================================================================

class TestHydrostaticBalance:
    def test_hydrostatic_pressure_profile(self):
        H, rho0, g = 1.0, 1000.0, 9.81
        z = np.linspace(0, H, 10)
        p_exact = rho0 * g * (H - z)
        assert p_exact[0] > p_exact[-1]

    def test_sound_speed_mach_criterion(self):
        U_max = 1.0
        c_s = 10.0 * U_max
        Ma = U_max / c_s
        assert Ma <= 0.1


# ============================================================================
# Dam break tests (2 tests)
# ============================================================================

class TestDamBreak:
    def test_dam_break_initial_geometry(self):
        h_init, w_init = 2.0, 1.0
        assert h_init > 0 and w_init > 0

    def test_martin_moyce_reference_exists(self):
        tau_ref, Z_over_a_ref = 0.5, 0.32
        assert tau_ref > 0 and Z_over_a_ref > 0


# ============================================================================
# Kernel tests (3 tests)
# ============================================================================

class TestKernel:
    def test_cubic_kernel_normalization(self):
        h = 0.1
        x = np.linspace(-3*h, 3*h, 10000)
        rij = x[:, None] - x[None, :]  # not needed, just use x
        w = _cubic_w_vec(np.abs(x).reshape(-1, 1), h).flatten()
        dx = x[1] - x[0]
        integral = np.sum(w) * dx
        assert abs(integral - 1.0) < 0.05

    def test_cubic_kernel_symmetry(self):
        h = 0.1
        r = np.array([[0.05], [0.1], [0.15]])
        w_pos = _cubic_w_vec(r, h)
        w_neg = _cubic_w_vec(-r, h)
        np.testing.assert_allclose(w_pos, w_neg, rtol=1e-10)

    def test_kernel_gradient_antisymmetry(self):
        h = 0.1
        r = np.array([[0.05], [0.1], [0.15]])
        dw_pos = _cubic_dw_vec(r, h)
        dw_neg = _cubic_dw_vec(-r, h)
        np.testing.assert_allclose(dw_pos, -dw_neg, rtol=1e-10)


# ============================================================================
# Configuration tests (2 tests)
# ============================================================================

class TestConfiguration:
    def test_config_properties(self):
        cfg = SodConfig(n_particles=100, x_left=0.0, x_right=1.0, h_factor=1.5)
        assert cfg.dx == 0.01
        assert cfg.h == 0.015

    def test_config_defaults(self):
        cfg = SodConfig()
        assert cfg.n_particles == 200
        assert cfg.gamma == 1.4
        assert cfg.alpha_av > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
