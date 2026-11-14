"""Tests for the Boussinesq startup series implementation."""
import math
import numpy as np
import pytest

from benchmarks.poiseuille_startup.run import boussinesq_series


class TestBoussinesqSeries:
    """Verify the analytical startup series against known properties."""

    def test_starts_from_rest(self):
        """At t=0, u(y,0) = 0 everywhere (series cancels steady state)."""
        y = np.linspace(0.001, 0.099, 20)
        u = boussinesq_series(y, 0.0, 0.1, 1.0, 0.01, n_terms=500)
        assert np.max(np.abs(u)) < 1e-3, f"u(y,0) should be ~0, got max={np.max(np.abs(u))}"

    def test_decays_to_steady(self):
        """At large t, u(y,t) → u_s(y) = g y (H-y) / (2ν)."""
        y = np.linspace(0.01, 0.09, 10)
        t_large = 10.0  # >> τ = H²/(νπ²) ≈ 0.1
        u = boussinesq_series(y, t_large, 0.1, 1.0, 0.01, n_terms=200)
        u_steady = 1.0 * y * (0.1 - y) / (2.0 * 0.01)
        rel_err = np.max(np.abs(u - u_steady)) / np.max(u_steady)
        assert rel_err < 1e-4, f"Series should decay to steady: rel_err={rel_err}"

    def test_no_slip_walls(self):
        """u(0,t) = u(H,t) = 0 for all t."""
        for t in [0.0, 0.1, 0.5, 1.0]:
            y_walls = np.array([0.0, 0.1])
            u = boussinesq_series(y_walls, t, 0.1, 1.0, 0.01)
            assert np.max(np.abs(u)) < 1e-9, f"No-slip violated at t={t}: u={u}"

    def test_positive_and_monotone(self):
        """Centreline velocity increases monotonically from 0 to u_max,∞."""
        u_max_inf = 1.0 * 0.1**2 / (8.0 * 0.01)  # = 1.25
        times = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        centre_vals = []
        for t in times:
            u = boussinesq_series(np.array([0.05]), t, 0.1, 1.0, 0.01)
            centre_vals.append(u[0])

        # Monotone increase
        for i in range(len(centre_vals) - 1):
            assert centre_vals[i+1] > centre_vals[i], \
                f"Centreline not monotone: {centre_vals[i]} -> {centre_vals[i+1]}"

        # Bounded by steady state
        assert centre_vals[-1] <= u_max_inf * (1.0 + 1e-6)

    def test_truncation_convergence(self):
        """Series converges with n_terms."""
        y = np.array([0.05])
        t = 0.1
        u_50 = boussinesq_series(y, t, 0.1, 1.0, 0.01, n_terms=50)
        u_200 = boussinesq_series(y, t, 0.1, 1.0, 0.01, n_terms=200)
        rel_diff = abs(u_50[0] - u_200[0]) / abs(u_200[0])
        assert rel_diff < 1e-6, f"Truncation not converged: rel_diff={rel_diff}"

    def test_known_value(self):
        """Check against a hand-computed value at a specific point."""
        # At y=H/2, t=τ=H²/(νπ²), the series gives a specific value
        H, nu, g = 0.1, 0.01, 1.0
        tau = H**2 / (nu * math.pi**2)
        y = np.array([H/2])
        u = boussinesq_series(y, tau, H, g, nu, n_terms=500)
        u_max_inf = g * H**2 / (8.0 * nu)
        # At t=τ, centreline should be ~63% of steady (1 - 1/e for slowest mode)
        expected_fraction = 1.0 - (32 / math.pi**3) * math.exp(-1)  # dominant term
        assert abs(u[0] / u_max_inf - expected_fraction) < 0.05, \
            f"Centreline at t=τ: got {u[0]/u_max_inf:.3f}, expected ~{expected_fraction:.3f}"
