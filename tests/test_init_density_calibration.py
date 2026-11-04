"""Initial lattice density calibration (backlog P1 / C2 of m2971).

The tests are written against a *closed-form* reference, not against recorded
output: ``analytic_kernel_sum`` evaluates ``sum_j W_ij V_j`` over the exact
lattice shells, so every assertion below compares the code with the kernel
definition rather than with a previous run.

Evidence level before this work: the summation density carried a lattice
deficit ``b`` that was neither measured nor removed. After it: ``b`` is measured
from the particles, matches the analytic value, and one mass pass removes it.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    calibrate_initial_mass,
    density_from_neighbor_pairs,
    drop_wall_wall_pairs,
    full_support_mask,
    kernel_consistency_scale,
    make_particles,
    tait_pressure,
    wall_lattice,
    wall_mass,
)
from tensorsph.neighbor_search import cell_linked_pairs

_RUN = (Path(__file__).resolve().parents[1]
        / "benchmarks" / "init_density_calibration" / "run.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location("init_density_run", _RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_harness()
RHO0 = 1000.0
H_FACTOR = 1.3


def _disc(lattice="hex", dx=0.01, radius=0.1):
    builder = H.hex_disc if lattice == "hex" else H.square_disc
    pos, mass = builder(radius, dx)
    h = H_FACTOR * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    core = H.core_mask(pos, radius, 2.0 * h)
    return pos, mass, h, pairs, core


# --------------------------------------------------------------------------- #
# The analytic reference itself
# --------------------------------------------------------------------------- #
def test_analytic_lattice_sum_is_a_kernel_identity():
    """Square lattice at h/dx = 1.0 must sum to 1 exactly (kernel normalisation)."""
    # A square lattice with dx = h has its first ring outside the cubic spline's
    # peak but inside the support; the sum is *not* 1. The identity that does
    # hold is the continuum limit: halving dx must converge to 1 quadratically.
    coarse = H.analytic_kernel_sum("square", 1.0)
    finer = H.analytic_kernel_sum("square", 2.0)
    assert abs(finer - 1.0) < abs(coarse - 1.0)
    assert abs(H.analytic_kernel_sum("square", 4.0) - 1.0) < 2e-3


@pytest.mark.parametrize("lattice", ["square", "hex"])
def test_analytic_matches_measurement(lattice):
    """Measured ``K`` from the particle set == analytic lattice sum to < 1e-5."""
    pos, mass, h, pairs, core = _disc(lattice)
    scale = kernel_consistency_scale(pos, mass, RHO0, h, pairs, core=core)
    # kernel_consistency_scale returns the *mass* factor 1/K.
    measured = 1.0 / scale
    analytic = H.analytic_kernel_sum(lattice, H_FACTOR)
    assert measured == pytest.approx(analytic, rel=1e-5)


def test_square_lattice_deficit_is_tiny_and_hex_is_not():
    """The bias is a lattice property, so it does not shrink with dx."""
    square = 1.0 - H.analytic_kernel_sum("square", H_FACTOR)
    hex_ = 1.0 - H.analytic_kernel_sum("hex", H_FACTOR)
    assert 0.0 < square < 1e-4          # 5.3e-5: de-compressed, Tait clamps it
    assert hex_ < 0.0                    # +0.59 %: pre-compressed, 2.4 kPa
    assert abs(hex_) > 50 * square


# --------------------------------------------------------------------------- #
# The calibration utility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lattice", ["hex", "square"])
def test_calibration_removes_the_core_bias(lattice):
    pos, mass, h, pairs, core = _disc(lattice)
    rho_before = density_from_neighbor_pairs(pos, mass, h, pairs)
    mass_cal, scale = calibrate_initial_mass(pos, mass, RHO0, h, pairs, core=core)
    rho_after = density_from_neighbor_pairs(pos, mass_cal, h, pairs)
    # Square is de-compressed by only 5.3e-5, hex pre-compressed by 5.9e-3.
    assert abs(float(rho_before[core].mean() / RHO0 - 1.0)) > 1e-5
    assert abs(float(rho_after[core].mean() / RHO0 - 1.0)) < 1e-5


def test_calibration_scales_every_particle_by_the_same_factor():
    """It is a reference-state fix, not a per-particle density fit."""
    pos, mass, h, pairs, core = _disc("hex")
    mass_cal, scale = calibrate_initial_mass(pos, mass, RHO0, h, pairs, core=core)
    ratio = mass_cal / mass
    assert float(ratio.max() / ratio.min() - 1.0) < 1e-6   # float32 rounding
    assert float(ratio.mean()) == pytest.approx(scale, rel=1e-6)


def test_calibration_is_a_fixed_point():
    """Applying it twice changes nothing: the second factor is 1."""
    pos, mass, h, pairs, core = _disc("hex")
    mass_cal, _ = calibrate_initial_mass(pos, mass, RHO0, h, pairs, core=core)
    second = kernel_consistency_scale(pos, mass_cal, RHO0, h, pairs, core=core)
    assert second == pytest.approx(1.0, abs=1e-6)


def test_full_support_mask_selects_the_interior():
    pos, mass, h, pairs, core = _disc("hex")
    mask = full_support_mask(pos, mass, h, pairs)
    assert bool(mask.any())
    # Every full-support particle is inside the disc, none is on the rim.
    r = torch.linalg.vector_norm(pos, dim=1)
    assert float(r[mask].max()) < 0.1


def test_full_support_mask_matches_an_eroded_core():
    """The geometry-free measure agrees with an independent geometric core.

    Two bodies sit inside one another's 2h here (drop 4 cm above the pool,
    2h = 6.5 cm), which is the configuration that could inflate the maximum
    kernel sum. It does not: the gap particles also lose lateral support, so the
    two measures and the analytic square-lattice sum agree.
    """
    cfg = Config(init_density_calibration=False)
    device = torch.device("cpu")
    pos, mass = make_particles(cfg, device)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    naive = kernel_consistency_scale(pos, mass, RHO0, h, pairs)
    band = 2.0 * h
    x, y = pos[:, 0], pos[:, 1]
    drop_bottom = cfg.drop_center_y - cfg.drop_radius
    core = ((x > band) & (x < cfg.pool_width - band) & (y > band)
            & (y < cfg.pool_depth - band) & (y < drop_bottom - band))
    measured = kernel_consistency_scale(pos, mass, RHO0, h, pairs, core=core)
    analytic = 1.0 / H.analytic_kernel_sum("square", cfg.h_factor)
    assert measured == pytest.approx(analytic, rel=1e-5)
    assert naive == pytest.approx(analytic, rel=1e-5)


# --------------------------------------------------------------------------- #
# The solver path: make_particles / wall_mass
# --------------------------------------------------------------------------- #
def test_make_particles_calibrates_by_default():
    cfg = Config()
    device = torch.device("cpu")
    pos, mass = make_particles(cfg, device)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    band = 2.0 * h
    x, y = pos[:, 0], pos[:, 1]
    core = ((x > band) & (x < cfg.pool_width - band) & (y > band)
            & (y < cfg.pool_depth - band)
            & (y < cfg.drop_center_y - cfg.drop_radius - band))
    assert abs(float(rho[core].mean() / RHO0 - 1.0)) < 1e-5
    expected = 1.0 / H.analytic_kernel_sum("square", cfg.h_factor)
    assert cfg.init_density_scale == pytest.approx(expected, rel=1e-4)


def test_calibration_can_be_switched_off_for_a_b():
    cfg = Config(init_density_calibration=False)
    device = torch.device("cpu")
    pos, mass = make_particles(cfg, device)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    band = 2.0 * h
    deep = ((pos[:, 0] > band) & (pos[:, 0] < cfg.pool_width - band)
            & (pos[:, 1] > band) & (pos[:, 1] < 0.25))
    assert float(rho[deep].mean() / RHO0 - 1.0) < 0.0
    assert cfg.init_density_scale == 1.0


def test_wall_masses_follow_the_fluid_scale():
    """Otherwise the first fluid layer next to a wall keeps half the bias."""
    device = torch.device("cpu")
    cfg_on = Config()
    make_particles(cfg_on, device)
    wall = wall_lattice(cfg_on, device)
    m_on = wall_mass(cfg_on, wall.shape[0], device)
    base = cfg_on.rho0 * cfg_on.dx * cfg_on.dx
    assert float(m_on[0]) == pytest.approx(base * cfg_on.init_density_scale, rel=1e-7)
    assert cfg_on.init_density_scale != pytest.approx(1.0, abs=1e-6)
    # Explicit override still wins, and the default is 1 when never calibrated.
    explicit = wall_mass(cfg_on, 4, device, scale=1.0)
    assert float(explicit[0]) == pytest.approx(base, rel=1e-7)
    assert float(wall_mass(Config(), 4, device)[0]) == pytest.approx(base, rel=1e-7)


def test_impact_case_initial_pressure_meets_the_p1_criterion():
    """P1 acceptance: p_max/(rho0 c0^2) < 1e-4 at t = 0, with and without walls."""
    for grid, c0 in ((False, 20.0), (True, 20.0), (True, 5.0)):
        cfg = Config(sound_speed=c0)
        device = torch.device("cpu")
        pos, mass = make_particles(cfg, device)
        h = cfg.h_factor * cfg.dx
        if grid:
            wall = wall_lattice(cfg, device)
            n = pos.shape[0]
            all_pos = torch.cat((pos, wall), dim=0)
            all_mass = torch.cat((mass, wall_mass(cfg, wall.shape[0], device)), dim=0)
            pairs = drop_wall_wall_pairs(cell_linked_pairs(all_pos, 2.0 * h), n)
            rho = density_from_neighbor_pairs(all_pos, all_mass, h, pairs)[:n]
        else:
            rho = density_from_neighbor_pairs(pos, mass, h)
        p_max = float(tait_pressure(rho, cfg).max())
        assert p_max / (cfg.rho0 * c0 * c0) < 1e-4


# --------------------------------------------------------------------------- #
# Physics: the hex disc the surface-tension benchmarks use
# --------------------------------------------------------------------------- #
def test_hex_disc_starts_pre_compressed_and_calibration_clears_it():
    pos, mass, h, pairs, core = _disc("hex", dx=0.01)
    cfg = Config(dx=0.01, sound_speed=20.0)
    p_before = float(tait_pressure(density_from_neighbor_pairs(pos, mass, h, pairs),
                                   cfg).max())
    mass_cal, _ = calibrate_initial_mass(pos, mass, RHO0, h, pairs, core=core)
    p_after = float(tait_pressure(
        density_from_neighbor_pairs(pos, mass_cal, h, pairs), cfg).max())
    # b = +5.8836e-3 behind the Tait law at c0 = 20 m/s: B ((1+b)^gamma - 1).
    bulk = RHO0 * 20.0**2 / 7.0
    expected = bulk * ((1.0 + 5.8836e-3) ** 7.0 - 1.0)
    assert p_before == pytest.approx(expected, rel=0.01)
    assert p_before > 2000.0                       # 6.7x the 360 Pa Laplace jump
    assert p_after < 1.0
    assert p_before / (RHO0 * 20.0**2) > 5e-3      # 50x the 1e-4 criterion
    assert p_after / (RHO0 * 20.0**2) < 1e-4


def test_bias_launches_an_acoustic_transient():
    """The consequence, measured: 40 steps of Verlet on the hex disc."""
    series = {}
    for calibrated in (False, True):
        pos, mass, h, pairs, core = _disc("hex", dx=0.01)
        cfg = Config(dx=0.01, sound_speed=20.0, gravity=0.0, dt=1.0e-4)
        if calibrated:
            mass, _ = calibrate_initial_mass(pos, mass, RHO0, h, pairs, core=core)
        rows = H.verlet(pos, torch.zeros_like(pos), mass, cfg, 40)
        series[calibrated] = max(r["max_speed"] for r in rows[1:])
    # Acoustic estimate of a 2395 Pa step: p / (rho0 c0) = 0.12 m/s.
    assert 0.05 < series[False] < 0.5
    assert series[True] < 1e-2
    assert series[False] / series[True] > 100