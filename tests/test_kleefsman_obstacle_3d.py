"""Tests for the 3-D Kleefsman obstacle case.

The 2-D case already has sixteen test modules; these tests cover only what the
third dimension adds: the box geometry, non-penetration of the box, the gauge
that measures "metres of water over the blocked width", and the bypass that the
2-D model cannot represent.

Assertions are on invariants, not on the presence of numbers: a gauge that
returns 0.0 for a full slab or a clamp that moves a particle but leaves it
inside the box both have to fail.
"""
import math

import pytest
import torch

from tensorsph.kleefsman_obstacle_3d import (
    Kleefsman3DConfig,
    arrival_time,
    bypass_count,
    clamp_out_of_obstacle,
    obstacle_lattice_3d,
    peak_height,
    run_case,
    score,
    slab_height,
    validate_case,
)

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def base_cfg(**kw) -> Kleefsman3DConfig:
    params = {"dx": 0.05, "t_end": 0.05, "output_every": 10}
    params.update(kw)
    return Kleefsman3DConfig(**params)


def test_the_published_apparatus_validates():
    validate_case(base_cfg())


@pytest.mark.parametrize("field,value,needle", [
    ("pool_width", 3.0, "tank length"),
    ("tank_height", 1.2, "tank height"),
    ("tank_depth", 0.6, "tank width"),
    ("column_width", 1.0, "column length"),
    ("column_height", 0.6, "column height"),
    ("column_depth", 0.5, "full tank width"),
    ("obstacle_x", 2.2, "upstream face"),
    ("obstacle_height", 0.2, "obstacle height"),
    ("obstacle_width", 0.5, "obstacle width"),
])
def test_a_geometry_that_is_not_the_experiment_is_rejected(field, value, needle):
    cfg = base_cfg(**{field: value})
    with pytest.raises(ValueError, match=needle):
        validate_case(cfg)


def test_a_box_that_does_not_resolve_is_rejected():
    # 0.161 m box at dx = 0.06 m is 2.7 cells: under the 3-cell floor
    with pytest.raises(ValueError, match="under-resolved"):
        validate_case(Kleefsman3DConfig(dx=0.06, t_end=0.05))


def test_a_box_outside_the_tank_width_is_rejected():
    cfg = base_cfg(obstacle_z=0.7)          # 0.7 + 0.403 > 1.00
    with pytest.raises(ValueError, match="inside the tank width"):
        validate_case(cfg)


def test_the_obstacle_lattice_has_the_measured_extent():
    cfg = base_cfg(dx=0.05)
    box = obstacle_lattice_3d(cfg, CPU)
    # 0.161 m is 3 cells at dx = 0.05; the 0.403 m width is 8 cells (0.40 m)
    assert box.shape == (3 * 3 * 8, 3)
    assert float(box[:, 0].min()) >= cfg.obstacle_x
    assert float(box[:, 0].max()) <= cfg.obstacle_x1()
    assert float(box[:, 1].min()) >= 0.0
    assert float(box[:, 1].max()) <= cfg.obstacle_height
    assert float(box[:, 2].min()) >= cfg.obstacle_z
    assert float(box[:, 2].max()) <= cfg.obstacle_z1()


def test_the_blocked_fraction_is_the_experimental_40_percent():
    assert base_cfg().blocked_fraction() == pytest.approx(0.403, rel=1e-12)


def test_the_box_can_be_centred_instead_of_flush():
    cfg = base_cfg(obstacle_z=0.5 * (1.0 - 0.403))
    validate_case(cfg)
    assert cfg.obstacle_z1() == pytest.approx(1.0 - cfg.obstacle_z)


# --------------------------------------------------------------------------- #
# Non-penetration of the box
# --------------------------------------------------------------------------- #

def _inside_box(pos, cfg):
    return ((pos[:, 0] > cfg.obstacle_x) & (pos[:, 0] < cfg.obstacle_x1())
            & (pos[:, 1] > 0.0) & (pos[:, 1] < cfg.obstacle_height)
            & (pos[:, 2] > cfg.obstacle_z) & (pos[:, 2] < cfg.obstacle_z1()))


def test_the_clamp_empties_the_box_and_reports_how_many_it_moved():
    cfg = base_cfg()
    # two particles inside the box, one obviously outside
    pos = torch.tensor([
        [cfg.obstacle_x + 0.5 * cfg.obstacle_length, 0.5 * cfg.obstacle_height,
         cfg.obstacle_z + 0.5 * cfg.obstacle_width],
        [cfg.obstacle_x + 0.02, 0.02, cfg.obstacle_z + 0.02],
        [1.0, 0.3, 0.5],
    ], dtype=torch.float64)
    vel = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
                       dtype=torch.float64)
    moved = clamp_out_of_obstacle(pos, vel, cfg)
    assert moved == 2
    assert not _inside_box(pos, cfg).any()
    # the untouched particle keeps its velocity exactly
    assert torch.equal(vel[2], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
    # every moved particle lost the velocity component normal to the face it
    # left through, and only that one
    for k in (0, 1):
        assert int((vel[k] == 0).sum()) == 1


def test_the_clamp_is_idempotent():
    cfg = base_cfg()
    pos = torch.tensor([[cfg.obstacle_x + 0.08, 0.08,
                         cfg.obstacle_z + 0.2]], dtype=torch.float64)
    vel = torch.zeros_like(pos)
    assert clamp_out_of_obstacle(pos, vel, cfg) == 1
    assert clamp_out_of_obstacle(pos, vel, cfg) == 0


# --------------------------------------------------------------------------- #
# Gauges
# --------------------------------------------------------------------------- #

def test_the_slab_gauge_reports_metres_of_water_over_the_blocked_width():
    cfg = base_cfg()
    # a full lattice column of n particles in the slab must read
    # n * dx^3 / (aperture * blocked_width) metres
    dx = cfg.dx
    xs = cfg.station_x() + torch.tensor([-0.02, 0.0, 0.02], dtype=torch.float64)
    ys = torch.arange(0.5, 4.5, 1.0, dtype=torch.float64) * dx      # 4 layers
    zs = cfg.obstacle_z + (torch.arange(8, dtype=torch.float64) + 0.5) * dx
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1)
    n = pos.shape[0]
    expected = n * dx ** 3 / (cfg.gauge_aperture * cfg.obstacle_width)
    assert slab_height(pos, cfg) == pytest.approx(expected, rel=1e-12)
    # 3 * 4 * 8 particles of dx^3 over 0.08 * 0.403 m^2
    assert slab_height(pos, cfg) == pytest.approx(
        3 * 4 * 8 * dx ** 3 / (0.08 * 0.403), rel=1e-12)
    # the peak operator reads the top layer, not the average
    assert peak_height(pos, cfg) == pytest.approx(float(ys[-1]), rel=1e-12)


def test_the_slab_gauge_ignores_water_that_bypassed_the_box():
    cfg = base_cfg()
    # same height, but placed in the open spanwise gap: the H2 gauge must not
    # see it, because in the experiment that water goes round the box
    z_open = cfg.obstacle_z1() + 0.5 * cfg.gauge_aperture
    pos = torch.tensor([[cfg.station_x(), 0.30, z_open]], dtype=torch.float64)
    assert slab_height(pos, cfg) == 0.0
    assert peak_height(pos, cfg) == 0.0


def test_the_slab_gauge_ignores_water_upstream_or_downstream_of_the_slab():
    cfg = base_cfg()
    away = cfg.station_x() - cfg.gauge_aperture
    pos = torch.tensor([[away, 0.30, cfg.obstacle_z + 0.2]], dtype=torch.float64)
    assert slab_height(pos, cfg) == 0.0


def test_an_empty_particle_set_has_zero_height():
    cfg = base_cfg()
    empty = torch.zeros((0, 3), dtype=torch.float64)
    assert slab_height(empty, cfg) == 0.0
    assert peak_height(empty, cfg) == 0.0
    assert bypass_count(empty, cfg) == 0


# --------------------------------------------------------------------------- #
# Bypass
# --------------------------------------------------------------------------- #

def test_bypass_counts_only_water_past_the_box_and_outside_its_span():
    cfg = base_cfg()
    pos = torch.tensor([
        [cfg.obstacle_x1() + 0.05, 0.05, cfg.obstacle_z + 0.2],      # past, blocked span
        [cfg.obstacle_x1() + 0.05, 0.05, cfg.obstacle_z1() + 0.2],   # past, open span
        [cfg.obstacle_x + 0.05, 0.05, cfg.obstacle_z1() + 0.2],      # not past yet
    ], dtype=torch.float64)
    assert bypass_count(pos, cfg) == 1


# --------------------------------------------------------------------------- #
# Arrival / scoring
# --------------------------------------------------------------------------- #

def test_arrival_is_linearly_interpolated_between_samples():
    times = [0.0, 0.1, 0.2]
    values = [0.0, 0.01, 0.03]
    # 0.02 lies halfway between 0.01 and 0.03
    assert arrival_time(times, values, 0.02) == pytest.approx(0.15)


def test_arrival_is_none_when_the_series_never_crosses():
    assert arrival_time([0.0, 0.1], [0.0, 0.01], 0.02) is None


def test_scoring_uses_ten_percent_and_does_not_cherry_pick():
    cfg = base_cfg()
    # a run whose slab peak is exactly the reference and whose arrival is 30 %
    # late must FAIL on the arrival row
    from tensorsph.marin_probes import H2_ARRIVAL_S, H2_PEAK_M, H2_PEAK_TIME_S
    run = _fake_run(cfg, peak=H2_PEAK_M, peak_t=H2_PEAK_TIME_S, arrival=H2_ARRIVAL_S * 1.30)
    verdict = score(run)
    peaks = {r["metric"]: r for r in verdict["rows"]}
    assert peaks["peak_height"]["pass"] is True
    assert peaks["arrival_time"]["pass"] is False
    assert verdict["verdict"] == "FAIL"


def test_a_censored_peak_cannot_pass():
    from tensorsph.kleefsman_obstacle_3d import Kleefsman3DRun

    cfg = base_cfg(t_end=0.5)
    run = Kleefsman3DRun(cfg=cfg)
    # the record stops at 0.5 s, well before the experimental peak at 1.70 s
    run.times = [0.0, 0.2, 0.35, 0.45, 0.50]
    run.slab = [0.0, 0.01, 0.05, 0.20, 0.370]     # maximum on the last sample
    run.height = list(run.slab)
    run.front = [0.0, 1.0, 2.0, 2.4, 2.6]
    run.bypass = [0.0] * 5
    verdict = score(run)
    assert verdict["censored"] is True
    assert verdict["verdict"] == "FAIL"
    assert verdict["window_covers_reference_peak"] is False


def _fake_run(cfg, peak, peak_t, arrival):
    """A run whose slab series crosses 0.02 m exactly at ``arrival``.

    The crossing is placed between two samples 0.1 s apart (0.01 -> 0.03 m), so
    the interpolated arrival is the midpoint by construction; no run of the
    solver is involved and the test therefore checks the scoring rule alone.
    """
    from tensorsph.kleefsman_obstacle_3d import Kleefsman3DRun

    run = Kleefsman3DRun(cfg=cfg)
    run.times = [0.0, arrival - 0.05, arrival + 0.05, peak_t, peak_t + 0.5]
    run.slab = [0.0, 0.01, 0.03, peak, 0.9 * peak]
    run.height = list(run.slab)
    run.front = [0.0, 1.0, 2.0, 2.5, 2.6]
    run.bypass = [0.0, 0.0, 1.0, 2.0, 3.0]
    return run
