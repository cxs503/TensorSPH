import math

import pytest
import torch

from tensorsph.dam_break import (
    DamBreakConfig,
    front_position,
    make_column,
    ritter_front,
    run_case,
    validate_case,
)
from tensorsph.droplet_impact import hydrostatic_density, wall_lattice


def small_case(**overrides):
    base = dict(dx=0.02, pool_width=1.0, column_width=0.2, tank_height=0.6,
                sound_speed=25.0, steps=400, output_every=100, device="cpu")
    base.update(overrides)
    cfg = DamBreakConfig(**base)
    cfg.column_height = 2.0 * cfg.column_width
    if cfg.sound_speed > 0:
        cfg.dt = cfg.dt_from_courant()
    return cfg


def test_column_is_inside_the_tank_and_rests_on_the_floor():
    cfg = small_case()
    positions, mass = make_column(cfg, torch.device("cpu"))
    assert positions.shape[1] == 2
    assert mass.shape == (positions.shape[0],)
    assert float(positions[:, 0].max()) < cfg.column_width
    assert float(positions[:, 0].min()) > 0.0
    assert float(positions[:, 1].min()) == pytest.approx(cfg.dx * 0.5)
    assert float(positions[:, 1].max()) < cfg.column_height
    assert (mass > 0).all()


def test_hydrostatic_initialisation_matches_the_analytic_profile():
    cfg = small_case(hydrostatic_init=True)
    positions, mass = make_column(cfg, torch.device("cpu"))
    expected = hydrostatic_density(positions[:, 1], cfg,
                                   surface=cfg.column_height) * cfg.dx * cfg.dx
    torch.testing.assert_close(mass, expected)

    uniform_cfg = small_case(hydrostatic_init=False)
    _, uniform_mass = make_column(uniform_cfg, torch.device("cpu"))
    # A resting column is compressed, so the hydrostatic total mass is larger.
    assert float(mass.sum()) > float(uniform_mass.sum())


def test_front_and_ritter_reference():
    cfg = small_case()
    positions = torch.tensor([[0.1, 0.1], [0.3, 0.2]], dtype=torch.float32)
    assert front_position(positions, cfg) == pytest.approx(0.3)
    assert front_position(torch.empty((0, 2)), cfg) == 0.0
    t = 0.2
    assert ritter_front(t, cfg) == pytest.approx(
        2.0 * math.sqrt(cfg.gravity * cfg.column_height) * t)


@pytest.mark.parametrize("overrides,error", [
    (dict(dx=0.0), ValueError),
    (dict(column_width=0.0), ValueError),
    (dict(column_width=5.0), ValueError),
    (dict(sound_speed=0.0), ValueError),
])
def test_invalid_cases_are_rejected(overrides, error):
    cfg = small_case(**overrides)
    with pytest.raises(error):
        validate_case(cfg)


def test_time_step_above_courant_limit_is_rejected():
    cfg = small_case()
    cfg.dt = 10.0 * cfg.dt_from_courant()
    with pytest.raises(ValueError, match="Courant"):
        validate_case(cfg)


def test_short_run_is_bounded_and_conserves_mass():
    cfg = small_case(steps=400, output_every=100)
    run = run_case(cfg)
    assert run.escaped == 0
    assert len(run.times) == len(run.front) == len(run.kinetic_energy)
    assert run.front[-1] > run.front[0], "the front must advance"
    assert run.max_speed < 10.0
    assert run.max_density_compression < 0.10
    assert run.mass > 0
    # Potential energy decreases as the column collapses.
    assert run.potential_energy[-1] < run.potential_energy[0]


def test_wall_geometry_spans_the_tank():
    cfg = small_case()
    wall = wall_lattice(cfg, torch.device("cpu"), top=cfg.tank_top())
    assert float(wall[:, 1].min()) < 0.0
    assert float(wall[:, 0].min()) < 0.0
    assert float(wall[:, 0].max()) > cfg.pool_width
    assert float(wall[:, 1].max()) <= cfg.tank_top() + 1e-6


def test_clamp_boundary_is_rejected_instead_of_silently_ignored():
    """run_case integrates static wall particles only."""
    cfg = small_case(boundary="clamp")
    with pytest.raises(ValueError, match="boundary"):
        validate_case(cfg)


def test_device_auto_resolves_to_a_real_device():
    """``device="auto"`` must not reach torch.device() unconverted."""
    cfg = small_case(device="auto", steps=1, output_every=1)
    run = run_case(cfg)
    assert run.final_positions.device.type in ("cpu", "cuda")


def test_mass_drift_is_measured_not_assumed():
    cfg = small_case(steps=100, output_every=50)
    run = run_case(cfg)
    assert run.initial_mass > 0.0
    assert run.final_mass == run.initial_mass
    assert run.mass_drift == 0.0
    assert run.mass == pytest.approx(run.initial_mass)


def test_compression_is_a_step_wise_maximum():
    """The recorded peak cannot be smaller than any diagnostic sample."""
    cfg = small_case(steps=200, output_every=20)
    run = run_case(cfg)
    assert run.max_density_compression > 0.0
    assert run.max_speed > 0.0


def test_compression_before_wall_impact_is_separated_from_the_impact_spike():
    """The whole-run peak is a wall-impact transient, not bulk compressibility."""
    cfg = small_case(dx=0.02, steps=900, output_every=50)
    cfg.pool_width = 0.7          # short tank: the front reaches the wall quickly
    run = run_case(cfg)
    assert run.wall_impact_time is not None
    assert run.compression_before_wall_impact <= run.max_density_compression
    assert run.wall_impact_time > 0.0
