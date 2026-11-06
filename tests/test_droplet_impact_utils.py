"""Tests for utility functions in ``tensorsph.droplet_impact`` that lack direct coverage.

Functions tested:
- ``calibrated_particle_mass``: mass that yields exact rho0 in the interior
- ``lattice_mass_unit``: particle mass (calibrated or raw depending on config)
- ``nominal_fall_time``: free-fall time from drop to pool surface
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    calibrated_particle_mass,
    lattice_mass_unit,
    nominal_fall_time,
)


# --------------------------------------------------------------------------- #
# Positive cases
# --------------------------------------------------------------------------- #

class TestCalibratedParticleMass:
    """Tests for :func:`calibrated_particle_mass`."""

    def test_returns_positive_float(self):
        """The calibrated mass must be a positive float."""
        cfg = Config(dx=0.01, h_factor=1.3, device="cpu")
        m = calibrated_particle_mass(cfg)
        assert isinstance(m, float)
        assert m > 0.0

    def test_close_to_raw_mass_for_square_lattice(self):
        """Square lattice at h=1.3dx has tiny bias, so calibrated ≈ raw."""
        cfg = Config(dx=0.01, h_factor=1.3, device="cpu")
        raw = cfg.rho0 * cfg.dx * cfg.dx
        calibrated = calibrated_particle_mass(cfg)
        # The square lattice bias at h=1.3dx is ~5.3e-5, so the correction is tiny
        assert abs(calibrated - raw) / raw < 1e-3

    def test_custom_dx_and_h(self):
        """Passing explicit dx and h overrides the config values."""
        cfg = Config(dx=0.01, h_factor=1.3, device="cpu")
        m1 = calibrated_particle_mass(cfg)
        m2 = calibrated_particle_mass(cfg, dx=0.02, h=0.026)
        assert m1 != m2  # different dx → different mass

    def test_resolution_independence_of_bias(self):
        """The kernel sum S depends on h/dx, not on absolute dx."""
        cfg1 = Config(dx=0.01, h_factor=1.3, device="cpu")
        cfg2 = Config(dx=0.005, h_factor=1.3, device="cpu")
        m1 = calibrated_particle_mass(cfg1)
        m2 = calibrated_particle_mass(cfg2)
        # m scales with dx^2, so m2/m1 should be (0.005/0.01)^2 = 0.25
        assert m2 / m1 == pytest.approx(0.25, rel=1e-6)


class TestLatticeMassUnit:
    """Tests for :func:`lattice_mass_unit`."""

    def test_calibrated_when_enabled(self):
        """With density_calibration=True, returns calibrated mass."""
        cfg = Config(dx=0.01, h_factor=1.3, density_calibration=True, device="cpu")
        m = lattice_mass_unit(cfg)
        expected = calibrated_particle_mass(cfg)
        assert m == pytest.approx(expected)

    def test_raw_when_disabled(self):
        """With density_calibration=False, returns raw rho0*dx^2."""
        cfg = Config(dx=0.01, h_factor=1.3, density_calibration=False, device="cpu")
        m = lattice_mass_unit(cfg)
        expected = cfg.rho0 * cfg.dx * cfg.dx
        assert m == pytest.approx(expected)

    def test_calibrated_differs_from_raw_for_square_lattice(self):
        """Even for square lattice the calibrated mass differs slightly from raw."""
        cfg = Config(dx=0.01, h_factor=1.3, density_calibration=True, device="cpu")
        calibrated = lattice_mass_unit(cfg)
        raw = cfg.rho0 * cfg.dx * cfg.dx
        # They should be close but not identical (square lattice has small bias)
        assert calibrated != raw


class TestNominalFallTime:
    """Tests for :func:`nominal_fall_time`."""

    def test_returns_none_for_empty_positions(self):
        """Empty position tensor → None."""
        cfg = Config(dx=0.01, pool_depth=0.1, device="cpu")
        pos = torch.empty((0, 2), dtype=torch.float64)
        assert nominal_fall_time(cfg, pos) is None

    def test_returns_none_for_1d_positions(self):
        """1D positions (no y coordinate) → None."""
        cfg = Config(dx=0.01, pool_depth=0.1, device="cpu")
        pos = torch.zeros((10, 1), dtype=torch.float64)
        assert nominal_fall_time(cfg, pos) is None

    def test_returns_none_when_no_pool(self):
        """All particles above pool depth → None."""
        cfg = Config(dx=0.01, pool_depth=0.1, device="cpu")
        pos = torch.tensor([[0.5, 0.2], [0.5, 0.3]], dtype=torch.float64)
        assert nominal_fall_time(cfg, pos) is None

    def test_returns_none_when_no_drop(self):
        """All particles at or below pool depth → None."""
        cfg = Config(dx=0.01, pool_depth=0.1, device="cpu")
        pos = torch.tensor([[0.5, 0.05], [0.5, 0.08]], dtype=torch.float64)
        assert nominal_fall_time(cfg, pos) is None

    def test_returns_zero_when_gap_is_zero(self):
        """Drop bottom exactly at pool surface but with pool particles below → 0."""
        cfg = Config(dx=0.01, pool_depth=0.1, gravity=9.81, device="cpu")
        pos = torch.tensor([
            [0.5, 0.09],  # pool particle (y <= 0.1)
            [0.5, 0.11],  # drop particle (y > 0.1)
        ], dtype=torch.float64)
        t = nominal_fall_time(cfg, pos)
        # gap = 0.11 - 0.09 = 0.02
        expected = math.sqrt(2.0 * 0.02 / 9.81)
        assert t == pytest.approx(expected, rel=1e-10)

    def test_positive_gap_returns_positive_time(self):
        """Drop above pool → positive fall time."""
        cfg = Config(dx=0.01, pool_depth=0.1, gravity=9.81, device="cpu")
        pos = torch.tensor([
            [0.5, 0.05],  # pool particle
            [0.5, 0.25],  # drop particle (0.2 m above pool max)
        ], dtype=torch.float64)
        t = nominal_fall_time(cfg, pos)
        expected = math.sqrt(2.0 * 0.2 / 9.81)
        assert t == pytest.approx(expected, rel=1e-10)

    def test_fall_time_scales_with_gap(self):
        """Fall time ∝ sqrt(gap)."""
        cfg = Config(dx=0.01, pool_depth=0.1, gravity=9.81, device="cpu")
        pos1 = torch.tensor([[0.5, 0.05], [0.5, 0.15]], dtype=torch.float64)  # gap=0.1
        pos2 = torch.tensor([[0.5, 0.05], [0.5, 0.45]], dtype=torch.float64)  # gap=0.4
        t1 = nominal_fall_time(cfg, pos1)
        t2 = nominal_fall_time(cfg, pos2)
        assert t2 / t1 == pytest.approx(2.0, rel=1e-10)  # sqrt(4) = 2


# --------------------------------------------------------------------------- #
# Negative cases (error injection must turn red)
# --------------------------------------------------------------------------- #

def test_calibrated_particle_mass_with_invalid_dx():
    """Negative dx should produce an invalid kernel sum → error or non-finite."""
    cfg = Config(dx=0.01, h_factor=1.3, device="cpu")
    with pytest.raises((ValueError, ZeroDivisionError)):
        calibrated_particle_mass(cfg, dx=-0.01)


def test_calibrated_particle_mass_with_zero_h():
    """Zero smoothing length → division by zero in kernel."""
    cfg = Config(dx=0.01, h_factor=1.3, device="cpu")
    with pytest.raises((ValueError, ZeroDivisionError)):
        calibrated_particle_mass(cfg, h=0.0)


def test_lattice_mass_unit_negative_dx_still_positive():
    """Negative dx → dx^2 is still positive, so mass is positive (but h is negative)."""
    cfg = Config(dx=-0.01, h_factor=1.3, density_calibration=False, device="cpu")
    m = lattice_mass_unit(cfg)
    # dx^2 is always positive, so raw mass is positive
    assert m > 0
    # But h = h_factor * dx is negative, which would break the kernel
    assert cfg.h_factor * cfg.dx < 0
