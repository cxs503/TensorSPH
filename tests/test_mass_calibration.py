"""Regression tests for the lattice mass calibration (backlog_update_m2971.md P1).

The instrument is the analytic lattice kernel sum,
``tensorsph.droplet_impact.lattice_density_ratio``; the facts it is checked
against are the bulk density ratios measured on the initial states the solver
actually ships.  Two trajectories are pinned:

* the calibrated states must start at ``rho0`` (spurious Tait pressure
  ``p/(rho0 c0^2) < 1e-4``, the P1 acceptance floor);
* the *uncalibrated* states must keep their historical values, so the record of
  what was wrong before the fix stays visible and cannot silently drift.

Numbers come from ``benchmarks/init_mass_calibration/run.py``; the cases are
re-measured here (each is one density summation, no time stepping) so the tests
do not depend on a stale JSON.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    calibrated_particle_mass,
    lattice_area_per_particle,
    lattice_density_ratio,
    lattice_offsets,
    make_particles,
    wall_mass,
)

_ROOT = Path(__file__).resolve().parents[1]
_RUN = _ROOT / "benchmarks" / "init_mass_calibration" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("init_mass_calibration_run", _RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load()


@pytest.fixture(scope="module")
def pool_off():
    return H.pool_case(calibrated=False)


@pytest.fixture(scope="module")
def pool_on():
    return H.pool_case(calibrated=True)


@pytest.fixture(scope="module")
def hex_off():
    return H.drop_case("hex", 0.01, calibrated=False)


@pytest.fixture(scope="module")
def hex_on():
    return H.drop_case("hex", 0.01, calibrated=True)


# --------------------------------------------------------------------------- #
# The instrument
# --------------------------------------------------------------------------- #
def test_analytic_bias_is_a_lattice_property_not_a_resolution_choice():
    """``1 + b`` is scale invariant: same lattice + same h/dx -> same value."""
    for lattice in ("square", "hex"):
        ratios = [lattice_density_ratio(dx, 1.3 * dx, lattice)
                  for dx in (0.05, 0.025, 0.01, 0.005, 0.002)]
        assert max(ratios) - min(ratios) < 1e-12
        # and it does depend on the kernel width, so it is not a constant
        assert abs(lattice_density_ratio(0.01, 1.5 * 0.01, lattice)
                   - lattice_density_ratio(0.01, 1.2 * 0.01, lattice)) > 1e-4


def test_analytic_bias_matches_measured_pool_core(pool_off):
    """The closed-form lattice sum reproduces the measured bulk density."""
    rel = abs(pool_off["core_ratio_mean"] - pool_off["analytic_ratio"]) / pool_off["analytic_ratio"]
    assert pool_off["core_particles"] > 100, "mask must cover a real interior"
    assert rel < 1e-5, f"analytic vs measured bulk ratio off by {rel:.2e}"


def test_hex_lattice_has_the_m2971_bias(hex_off):
    """The hex drop reproduces m2971's +0.588 % pre-compression, both ways."""
    assert abs(hex_off["core_ratio_mean"] - 1.00588) < 5e-5
    assert abs(hex_off["analytic_ratio"] - 1.00588) < 5e-5
    # and that offset is visible against the acoustic scale: 2.4 kPa at c0=20
    assert hex_off["p_max"] > 2000.0
    assert hex_off["p_max_over_pressure_scale"] > H.PRESSURE_FLOOR


def test_square_lattice_is_not_precompressed(pool_off):
    """Honest record: the square lattice the solver ships sits *below* rho0.

    Tait clamps pressure at zero there, so the 2.4 kPa spurious pressure is a
    hex-lattice defect, not a solver-wide one. Pinned so nobody re-chases it.
    """
    assert pool_off["core_ratio_mean"] < 1.0
    assert abs(pool_off["core_ratio_mean"] - 1.0) < 1e-4
    assert pool_off["p_max"] == 0.0


# --------------------------------------------------------------------------- #
# The fix
# --------------------------------------------------------------------------- #
def test_calibration_removes_the_bulk_offset(pool_on, hex_on):
    for case in (pool_on, hex_on):
        assert abs(case["core_ratio_mean"] - 1.0) < 1e-6, case["case"]


def test_initial_pressure_below_the_p1_floor(pool_on, hex_on):
    """P1 acceptance: p_max/(rho0 c0^2) < 1e-4 on the shipped initial state."""
    for case in (pool_on, hex_on):
        assert case["p_max_over_pressure_scale"] < 1e-4, case["case"]


def test_calibration_is_a_sub_percent_mass_change():
    """Guard against a wrongly tuned correction: b is < 1 % for every lattice."""
    cfg = Config(dx=0.01)
    for lattice in ("square", "hex"):
        ratio = lattice_density_ratio(cfg.dx, cfg.h_factor * cfg.dx, lattice)
        assert 0.99 < ratio < 1.01
        mass = calibrated_particle_mass(cfg, lattice)
        assert mass == pytest.approx(cfg.rho0 * lattice_area_per_particle(cfg.dx, lattice) / ratio)


def test_flag_off_reproduces_lattice_constant_mass():
    """``mass_calibration=False`` is the pre-fix baseline, bit for bit."""
    cfg = Config(dx=0.025, mass_calibration=False)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    assert torch.allclose(mass, torch.full_like(mass, cfg.rho0 * cfg.dx * cfg.dx))
    assert torch.allclose(wall_mass(cfg, 4, torch.device("cpu")),
                          torch.full((4,), cfg.rho0 * cfg.dx ** 2, dtype=torch.float32))
    cfg_on = Config(dx=0.025)
    _, mass_on = make_particles(cfg_on, torch.device("cpu"))
    assert not torch.allclose(mass, mass_on)


def test_hydrostatic_init_carries_the_calibration():
    """The analytical profile is a density; its mass conversion must use the
    same calibration as the uniform lattice, otherwise the pool starts with a
    depth-dependent offset."""
    off = Config(dx=0.025, hydrostatic_init=True, mass_calibration=False)
    on = Config(dx=0.025, hydrostatic_init=True, mass_calibration=True)
    _, mass_off = make_particles(off, torch.device("cpu"))
    _, mass_on = make_particles(on, torch.device("cpu"))
    scale = calibrated_particle_mass(on) / (on.rho0 * on.dx * on.dx)
    assert scale != pytest.approx(1.0)
    assert torch.allclose(mass_on, mass_off * scale, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_unknown_lattice_and_bad_spacing_are_rejected():
    with pytest.raises(ValueError):
        lattice_density_ratio(0.01, 0.013, "fcc")
    with pytest.raises(ValueError):
        lattice_area_per_particle(0.01, "fcc")
    with pytest.raises(ValueError):
        lattice_offsets(0.0, 0.013)
    with pytest.raises(ValueError):
        lattice_offsets(0.01, -1.0)


def test_offsets_stay_inside_the_kernel_support():
    dx, h = 0.01, 0.013
    offsets = lattice_offsets(dx, h, "hex")
    r = torch.linalg.vector_norm(offsets, dim=-1)
    assert float(r.min()) == 0.0
    assert float(r.max()) > 2.0 * h  # the stencil reaches past the support ...
    from tensorsph.droplet_impact import cubic_spline_kernel
    assert float(cubic_spline_kernel(r[r > 2.0 * h], h).abs().max()) == 0.0  # ... weight 0