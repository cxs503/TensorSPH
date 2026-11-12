"""Gates for the case-layer lattice mass calibration (``Config.density_calibration``).

The kernel-sum bias ``b = m * sum_j W / rho0 - 1`` of a regular lattice is a
property of ``(h/dx, geometry)`` only (``tensorsph.initial_density``).  The
solver path (``droplet_impact.make_particles``) has honoured
``Config.density_calibration`` since m3211; the *case* layer did not, so
``density_calibration=True`` on a ``DamBreakConfig``/``SloshingConfig``/3-D case
was silently ignored and the block started with ``b`` baked in as a spurious
pressure.  These tests pin the behaviour of each case builder against the
analytic bias and against the flag.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph import dam_break, sloshing
from tensorsph.dam_break import DamBreakConfig
from tensorsph.dam_break_3d import DamBreak3DConfig
from tensorsph.droplet_impact import Config
from tensorsph.initial_density import (
    calibration_factor,
    lattice_bias,
    measure_complete_support_bias,
)

torch.set_num_threads(2)

DX = 0.02
H_FACTORS = (1.3, 1.5)


def _block_bias(cfg) -> float:
    """Measured ``b`` of a uniform-mass block built by the case builder."""
    pos, mass = dam_break.make_column(cfg, torch.device("cpu"))
    return measure_complete_support_bias(
        pos, mass, cfg.h_factor * cfg.dx, cfg.rho0)["bias"]


def _sloshing_bias(cfg) -> float:
    pos, mass = sloshing.make_water_block(cfg, torch.device("cpu"))
    return measure_complete_support_bias(
        pos, mass, cfg.h_factor * cfg.dx, cfg.rho0)["bias"]


@pytest.mark.parametrize("hf", H_FACTORS)
def test_dam_break_block_matches_analytic_bias(hf):
    """Uncalibrated block reproduces the closed-form ``b(h/dx)`` (G1)."""
    cfg = DamBreakConfig(dx=DX, h_factor=hf, hydrostatic_init=False,
                         density_calibration=False)
    measured = _block_bias(cfg)
    analytic = lattice_bias(DX, hf * DX)
    assert abs(measured - analytic) < 1.0e-5, (measured, analytic)
    # The point of the sweep: at h/dx = 1.5 the offset is real, not rounding.
    if hf == 1.5:
        assert measured > 2.0e-3


@pytest.mark.parametrize("hf", H_FACTORS)
def test_dam_break_calibration_flag_is_honoured(hf):
    """``density_calibration=True`` removes the offset (G2)."""
    cfg = DamBreakConfig(dx=DX, h_factor=hf, hydrostatic_init=False,
                         density_calibration=True)
    assert abs(_block_bias(cfg)) < 1.0e-5


@pytest.mark.parametrize("hf", H_FACTORS)
def test_sloshing_block_calibration_flag_is_honoured(hf):
    off = sloshing.SloshingConfig(dx=DX, h_factor=hf, hydrostatic_init=False,
                                  density_calibration=False)
    on = sloshing.SloshingConfig(dx=DX, h_factor=hf, hydrostatic_init=False,
                                 density_calibration=True)
    assert abs(_sloshing_bias(off) - lattice_bias(DX, hf * DX)) < 1.0e-5
    assert abs(_sloshing_bias(on)) < 1.0e-5


def test_hydrostatic_profile_is_scaled_not_replaced():
    """A hydrostatic start keeps its shape: mass = profile * (m_unit / rho0)."""
    cfg = DamBreakConfig(dx=DX, h_factor=1.5, hydrostatic_init=True,
                         density_calibration=True)
    pos, mass = dam_break.make_column(cfg, torch.device("cpu"))
    profile = dam_break.hydrostatic_density(pos[:, 1], cfg,
                                           surface=cfg.column_height)
    factor = float(mass[0]) / float(profile[0])
    # mass = profile * (m_unit / rho0), i.e. the profile times dx^2 times the
    # calibration scalar: the *shape* is untouched, only the density it encodes.
    assert factor == pytest.approx(DX * DX * calibration_factor(DX, cfg.h_factor * cfg.dx),
                                  rel=1e-6)
    # Depth ordering preserved (mass increases downwards, unlike a uniform fill)
    order = torch.argsort(pos[:, 1], descending=True)
    assert torch.all(mass[order][:-1] <= mass[order][1:] + 1e-18)


def test_default_resolution_change_is_negligible():
    """At the production ``h/dx = 1.3`` the fix moves mass by 5.3e-5 (G3).

    Every published B1/B2 number was produced with the uncalibrated block, so
    the calibration must be invisible there -- only the flag stays honest.
    """
    cfg = DamBreakConfig(dx=DX, h_factor=1.3, hydrostatic_init=False,
                         density_calibration=True)
    _, mass_on = dam_break.make_column(cfg, torch.device("cpu"))
    rel = abs(float(mass_on[0]) / (cfg.rho0 * DX * DX) - 1.0)
    assert rel < 1.0e-4


def test_three_d_block_uses_the_three_d_calibration():
    """The 3-D builder calibrates with the *3-D* kernel, not the 2-D one (G4)."""
    from tensorsph.threed.kernel3 import calibrated_particle_mass, lattice_bias as b3

    cfg = DamBreak3DConfig(dx=0.04, hydrostatic_init=False, density_calibration=True)
    _, mass_on = dam_break_3d_column(cfg)
    cfg_off = DamBreak3DConfig(dx=0.04, hydrostatic_init=False, density_calibration=False)
    _, mass_off = dam_break_3d_column(cfg_off)
    h = cfg.h_factor * cfg.dx
    expected = calibrated_particle_mass(cfg.dx, h, cfg.rho0)
    assert float(mass_on[0]) == pytest.approx(expected, rel=1e-6)
    # Uncalibrated mass is the raw rho0 dx^3 and the 3-D bias is the ratio back.
    bias = float(mass_off[0]) / expected - 1.0
    assert abs(bias - b3(cfg.dx, h)) < 1e-6
    assert abs(bias) < 0.05, "3-D bias should stay a small correction"


def dam_break_3d_column(cfg):
    from tensorsph.dam_break_3d import make_column_3d
    pos, mass = make_column_3d(cfg, torch.device("cpu"))
    return pos, mass


def test_documented_bias_numbers_match_the_computation():
    """The module docstrings must quote the measured values, not the hex one (G5)."""
    from tensorsph import initial_density

    square = lattice_bias(1.0, 1.3, kind="square")
    hex_ = lattice_bias(1.0, 1.3, kind="hex")
    assert square == pytest.approx(-5.323104e-05, rel=1e-4)
    assert hex_ == pytest.approx(5.883522e-03, rel=1e-4)
    doc = initial_density.__doc__ or ""
    assert "-5.3e-5" in doc, "square-lattice bias at h/dx=1.3 must be quoted"
    assert "5.9e-3" in doc, "hex-lattice bias at h/dx=1.3 must be quoted"


def test_bias_is_dx_independent():
    """``b`` depends on ``h/dx`` only -- so one scalar fixes every resolution."""
    for kind, ref in (("square", lattice_bias(0.002, 1.5 * 0.002)),
                      ("hex", lattice_bias(0.002, 1.3 * 0.002, kind="hex"))):
        for dx in (0.001, 0.003, 0.01):
            hf = 1.5 if kind == "square" else 1.3
            assert math.isclose(lattice_bias(dx, hf * dx, kind=kind), ref,
                                rel_tol=0, abs_tol=1e-12)