"""Regression gates for the P1 initialisation mass calibration (m2971 C2).

A regular lattice does not partition unity exactly: with the lattice-constant
mass the interior sum-density is ``rho0 (1 + b)`` with a small ``b`` set by the
kernel and the packing.  On the hexagonal drop used by the Young-Laplace
benchmarks ``b = +5.9e-3``, which at ``c0 = 20 m/s`` is a spurious *initial*
pressure of ``B[(1+b)^gamma - 1]`` = 2.4 kPa -- 6.7 x the Laplace signal of the
static-droplet benchmark.  ``calibrate_mass`` removes it with one pass of
``m -> m / (1 + b)``.

These tests pin the *initialisation* facts (milliseconds each, no time
integration): the measured bias, the post-calibration interior density, and the
acceptance criterion ``p_max / (rho0 c0^2) < 1e-4``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    calibrate_mass,
    density_from_neighbor_pairs,
    lattice_bias,
    make_particles,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_law"

RADIUS = 0.02
RHO0 = 1000.0
GAMMA = 7.0
H_FACTOR = 1.3
C0 = 20.0
TARGET = 1.0e-4


def _hexagonal_droplet():
    spec = importlib.util.spec_from_file_location(
        "laplace_law_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hexagonal_droplet


def _hex_state(dx: float):
    hexagonal_droplet = _hexagonal_droplet()
    pos, mass = hexagonal_droplet(RADIUS, dx, RHO0)
    pos = pos.to(torch.float32)
    mass = mass.to(torch.float32)
    cfg = Config(dx=dx, rho0=RHO0, sound_speed=C0, gamma=GAMMA)
    return pos, mass, cfg


def _interior_ratio(pos, mass, cfg, core_cut: float = 0.5) -> float:
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
    return float(rho[r < core_cut].mean() / cfg.rho0)


def test_lattice_bias_measures_the_hexagonal_offset():
    pos, mass, cfg = _hex_state(0.002)
    b = lattice_bias(pos, mass, cfg, torch.device("cpu"))
    # The hexagonal packing over-covers: the interior is pre-compressed.
    assert 5.0e-3 < b < 7.0e-3


def test_calibrate_mass_restores_the_interior_density():
    pos, mass, cfg = _hex_state(0.002)
    mass_cal = calibrate_mass(pos, mass, cfg, torch.device("cpu"))
    assert _interior_ratio(pos, mass_cal, cfg) == pytest.approx(1.0, abs=1e-5)
    # The uncalibrated state is measurably off.
    assert _interior_ratio(pos, mass, cfg) == pytest.approx(1.0 + 5.9e-3, abs=2e-4)


def test_initial_pressure_below_the_acceptance_target():
    pos, mass, cfg = _hex_state(0.002)
    mass_cal = calibrate_mass(pos, mass, cfg, torch.device("cpu"))
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass_cal, h, pairs)
    p = tait_pressure(rho, cfg)
    assert float(p.max()) / (cfg.rho0 * cfg.sound_speed ** 2) < TARGET


def test_uncalibrated_hexagonal_drop_is_precompressed():
    # Sanity: the bias is real, so the fix has something to remove.
    pos, mass, cfg = _hex_state(0.002)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    p = tait_pressure(rho, cfg)
    assert float(p.max()) / (cfg.rho0 * cfg.sound_speed ** 2) > 10.0 * TARGET


def test_bias_is_independent_of_the_sound_speed():
    # b is a lattice property, not an EOS property.
    pos, mass, cfg = _hex_state(0.002)
    b20 = lattice_bias(pos, mass, cfg, torch.device("cpu"))
    cfg5 = Config(dx=cfg.dx, rho0=RHO0, sound_speed=5.0, gamma=GAMMA)
    b5 = lattice_bias(pos, mass, cfg5, torch.device("cpu"))
    assert b20 == pytest.approx(b5, rel=1e-9)


def test_make_particles_calibration_flag():
    device = torch.device("cpu")
    cfg_off = Config()
    cfg_on = Config(mass_calibration=True)
    pos_off, mass_off = make_particles(cfg_off, device)
    pos_on, mass_on = make_particles(cfg_on, device)
    # Off by default: the lattice-constant mass is untouched (reproducibility).
    assert torch.equal(mass_off, torch.full_like(mass_off, cfg_off.rho0 * cfg_off.dx ** 2))
    assert torch.equal(pos_off, pos_on)
    # On: the interior sum-density is exactly rho0.
    h = cfg_on.h_factor * cfg_on.dx
    pairs = cell_linked_pairs(pos_on, 2.0 * h)
    rho = density_from_neighbor_pairs(pos_on, mass_on, h, pairs)
    plateau = rho >= 0.999 * rho.max()
    assert float(rho[plateau].mean() / cfg_on.rho0) == pytest.approx(1.0, abs=1e-5)
    p = tait_pressure(rho, cfg_on)
    assert float(p.max()) / (cfg_on.rho0 * cfg_on.sound_speed ** 2) < TARGET
