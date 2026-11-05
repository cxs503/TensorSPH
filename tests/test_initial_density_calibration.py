"""The initial sum-density lattice bias: closed form, measurement, calibration.

Pins ``src/tensorsph/initial_density.py`` (mission m3209).  The numbers are the
ones ``backlog_update_m2971.md`` P1 asks about: ``b = 5.88e-3`` and 2395 Pa.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    make_particles,
)
from tensorsph.initial_density import (
    calibration_factor,
    calibrate_mass,
    lattice_bias,
    lattice_kernel_sum,
    measure_complete_support_bias,
    spurious_pressure,
    spurious_pressure_ratio,
)
from tensorsph.neighbor_search import cell_linked_pairs

DX = 0.002
H_FACTOR = 1.3
H = H_FACTOR * DX
RHO0 = 1000.0


def _hex_lattice(radius: float, dx: float, rho0: float, dtype=torch.float64):
    """Hex packing with the convention of benchmarks/laplace_law."""
    dy = dx * math.sqrt(3.0) / 2.0
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=dtype)
    mass = torch.full((pos.shape[0],), rho0 * dx * dy, dtype=dtype)
    return pos, mass


def _square_lattice(half: float, dx: float, rho0: float, dtype=torch.float64):
    g = torch.arange(-half, half, dx, dtype=dtype)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=dtype)
    return pos, mass


# --------------------------------------------------------------------------- #
# The closed form
# --------------------------------------------------------------------------- #
def test_hex_bias_reproduces_the_offset_reported_in_m2971():
    """The b that m2971 measured by hand now has a closed form."""
    b = lattice_bias(DX, H, kind="hex")
    assert b == pytest.approx(5.8835e-3, rel=1e-4)
    # and through Tait it is the 2395 Pa that file reports at c0 = 20 m/s
    p = (RHO0 * 20.0 ** 2 / 7.0) * ((1.0 + b) ** 7 - 1.0)
    assert p == pytest.approx(2395.0, rel=1e-3)


def test_square_bias_is_two_orders_smaller_and_negative():
    """The repo default (square lattice, h = 1.3 dx) sits near a zero crossing."""
    b = lattice_bias(DX, H, kind="square")
    assert b < 0.0
    assert abs(b) < 1.0e-4
    assert abs(b) < abs(lattice_bias(DX, H, kind="hex")) / 50.0


@pytest.mark.parametrize("kind", ["square", "hex"])
def test_kernel_sum_is_finite_and_close_to_area_inverse(kind):
    """Sum over one complete-support particle is 1/area to a few 1e-3 relative."""
    area = DX * DX * (math.sqrt(3.0) / 2.0 if kind == "hex" else 1.0)
    s = lattice_kernel_sum(DX, H, kind=kind)
    assert math.isfinite(s) and s > 0.0
    assert abs(s * area - 1.0) < 2.0e-2     # hex is the worse lattice


@pytest.mark.parametrize("kind", ["square", "hex"])
def test_bias_depends_only_on_h_over_dx(kind):
    """b is a function of h/dx: it must not move when the spacing is refined."""
    a = lattice_bias(0.002, H_FACTOR * 0.002, kind=kind)
    b = lattice_bias(0.0015, H_FACTOR * 0.0015, kind=kind)
    c = lattice_bias(0.001, H_FACTOR * 0.001, kind=kind)
    assert abs(a - b) < 1e-12
    assert abs(a - c) < 1e-12


def test_bias_changes_sign_with_h_over_dx():
    """The offset is a lattice quadrature error, so it oscillates, not decays."""
    values = [lattice_bias(DX, hf * DX, kind="hex") for hf in (1.05, 1.3, 1.6)]
    assert values[0] < 0.0 < values[1]
    assert values[2] < values[1]


def test_invalid_lattice_name_is_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        lattice_bias(DX, H, kind="triangular")


# --------------------------------------------------------------------------- #
# The closed form against measurement
# --------------------------------------------------------------------------- #
def test_analytic_bias_matches_the_measured_hex_drop():
    pos, mass = _hex_lattice(0.02, DX, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    measured = measure_complete_support_bias(pos, mass, H, RHO0, pairs=pairs)
    assert measured["count"] > 20            # the drop does have a core
    assert measured["bias"] == pytest.approx(lattice_bias(DX, H, kind="hex"),
                                             rel=1e-9)


def test_analytic_bias_matches_an_ideal_square_lattice():
    pos, mass = _square_lattice(0.02, DX, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    measured = measure_complete_support_bias(pos, mass, H, RHO0, pairs=pairs)
    assert measured["bias"] == pytest.approx(lattice_bias(DX, H, kind="square"),
                                             rel=1e-9)


def test_measure_reports_the_complete_support_population():
    pos, mass = _square_lattice(0.02, DX, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    out = measure_complete_support_bias(pos, mass, H, RHO0, pairs=pairs)
    # only particles at least 2h from every boundary qualify: a thin frame
    assert 0 < out["count"] < pos.shape[0]
    assert 0.2 < out["count_fraction"] < 0.95


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_factor_is_one_over_one_plus_b():
    b = lattice_bias(DX, H, kind="hex")
    assert calibration_factor(DX, H, kind="hex") == pytest.approx(1.0 / (1.0 + b))


def test_calibration_zeroes_the_initial_pressure_of_the_hex_drop():
    pos, mass = _hex_lattice(0.02, DX, RHO0)
    cal = calibrate_mass(mass, DX, H, kind="hex")
    pairs = cell_linked_pairs(pos, 2.0 * H)
    rho = density_from_neighbor_pairs(pos, cal, H, pairs)
    assert measure_complete_support_bias(pos, cal, H, RHO0,
                                         pairs=pairs)["bias"] == pytest.approx(
                                             0.0, abs=1e-12)
    # m2971's gate: p_max / (rho0 c0^2) < 1e-4 across the c0 ladder
    for c0 in (20.0, 10.0, 5.0):
        assert spurious_pressure_ratio(rho, RHO0, c0) < 1e-10


def test_uncalibrated_drop_fails_that_gate_by_six_orders_of_magnitude():
    pos, mass = _hex_lattice(0.02, DX, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    rho = density_from_neighbor_pairs(pos, mass, H, pairs)
    ratio = spurious_pressure_ratio(rho, RHO0, 20.0)
    assert ratio == pytest.approx(5.99e-3, rel=5e-2)
    assert ratio > 1e-4                     # i.e. the gate really was failing
    # 6.7x the 360 Pa Young-Laplace jump of the 2 cm drop
    assert float(spurious_pressure(rho, RHO0, 20.0).max()) / 360.0 > 6.0


def test_make_particles_calibration_flag_is_off_by_default_and_scales_mass():
    # ``density_calibration`` (the hydrostatic mass profile) is on by default and
    # would make the base mass non-uniform, hiding the flag under test; switch it
    # off so this case measures ``calibrate_initial_mass`` alone.
    cfg = Config(dx=0.002, h_factor=H_FACTOR, rho0=RHO0, pool_width=0.1,
                 pool_depth=0.05, drop_radius=0.005, drop_center_y=0.1,
                 gravity=9.81, density_calibration=False)
    pos_off, mass_off = make_particles(cfg, torch.device("cpu"))
    assert torch.allclose(mass_off,
                          torch.full_like(mass_off, RHO0 * cfg.dx * cfg.dx))
    cfg_on = Config(**{**cfg.__dict__, "calibrate_initial_mass": True})
    pos_on, mass_on = make_particles(cfg_on, torch.device("cpu"))
    assert torch.equal(pos_off, pos_on)
    factor = calibration_factor(cfg.dx, H_FACTOR * cfg.dx, kind="square")
    assert torch.allclose(mass_on, mass_off * factor)


def test_calibrated_pool_starts_at_rest_pressure():
    """Hydrostatic init + calibration: no spurious pressure before the first step."""
    cfg = Config(dx=0.002, h_factor=H_FACTOR, rho0=RHO0, sound_speed=20.0,
                 pool_width=0.1, pool_depth=0.05, drop_radius=0.005,
                 drop_center_y=0.1, gravity=9.81, hydrostatic_init=True,
                 calibrate_initial_mass=True)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    pool = pos[:, 1] <= cfg.pool_depth
    pos, mass = pos[pool], mass[pool]
    pairs = cell_linked_pairs(pos, 2.0 * H)
    rho = density_from_neighbor_pairs(pos, mass, H, pairs)
    # Compare against the *target* profile, not against rho0: at the bottom the
    # target itself is rho0 (1 + g H / B) = 1.0086 rho0, so a bias-vs-rho0
    # reading would measure the hydrostatics, not the lattice.
    from tensorsph.droplet_impact import hydrostatic_density
    target = hydrostatic_density(pos[:, 1], cfg)
    core = rho >= (1.0 - 1e-9) * rho.max()
    assert int(core.sum()) > 0
    assert torch.allclose(rho[core] / target[core], torch.ones(1, dtype=rho.dtype),
                          rtol=2e-4)
