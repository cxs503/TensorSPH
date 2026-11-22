import numpy as np
import pytest
import torch

from tensorsph.kleefsman_obstacle_3d import Kleefsman3DConfig, clamp_out_of_obstacle
from tensorsph.marin_probes import centreline_height, sensor_positions, tap_pressures


def test_tap_coordinates_and_one_sided_hydrostatic_pressure():
    cfg = Kleefsman3DConfig(dx=0.02)
    taps = sensor_positions(cfg, torch.empty(0, dtype=torch.float64))
    assert taps[:4, 1].tolist() == pytest.approx([0.021, 0.061, 0.101, 0.141])
    assert (taps[4:, 0] - cfg.obstacle_x).tolist() == pytest.approx([0.021, 0.061, 0.101, 0.141])
    assert taps[0, 2] == pytest.approx(cfg.obstacle_z + 0.176)
    assert taps[4, 2] == pytest.approx(cfg.obstacle_z1() - 0.176)
    positions = taps.clone()
    positions[:4, 0] -= 0.01
    positions[4:, 1] += 0.01
    # Contaminated values inside the solid must have zero weight on both faces.
    inside = taps[0] + taps.new_tensor([0.003, 0.0, 0.0])
    positions = torch.cat([positions, inside[None]])
    density = torch.full_like(positions[:, 0], 1000.0)
    pressure = density * cfg.gravity * (0.3 - positions[:, 1])
    pressure[-1] = 1e8
    p, n = tap_pressures(positions, pressure, density, density.reciprocal(), cfg, taps)
    torch.testing.assert_close(p, 1000.0 * cfg.gravity * (0.3 - taps[:, 1]))
    assert (n >= 1).all()
    p, n = tap_pressures(positions[:0], pressure[:0], density[:0], density[:0], cfg, taps)
    assert not n.any() and not p.any()


def test_h2_measures_horizontal_volume_and_ignores_remote_water():
    cfg = Kleefsman3DConfig(dx=0.015)
    span = (torch.arange(-6, 7, dtype=torch.float64) + 0.5) * cfg.dx
    y = (torch.arange(8, dtype=span.dtype) + 0.5) * cfg.dx
    cloud = torch.cartesian_prod(cfg.station_x() + span, y, 0.5 * cfg.tank_depth + span)
    volume = torch.full_like(cloud[:, 0], cfg.dx**3)
    assert float(centreline_height(cloud, volume, cfg)) == pytest.approx(8 * cfg.dx, rel=0.005)
    cloud[:, 2] += 0.3
    assert centreline_height(cloud, volume, cfg) == 0


def test_contact_impulse_is_opposite_the_actual_fluid_momentum_jump():
    cfg = Kleefsman3DConfig()
    pos = torch.tensor([[cfg.obstacle_x + 0.001, 0.05, 0.5], [0.0, 0.05, 0.5]], dtype=torch.float64)
    vel = pos.new_tensor([[3.0, 2.0, -1.0], [0.1, 0.0, 0.0]])
    mass = pos.new_tensor([0.75, 1.0])
    initial = (mass[:, None] * vel).sum(0)
    impulse = pos.new_zeros(3)
    assert clamp_out_of_obstacle(pos, vel, cfg, mass=mass, body_impulse=impulse) == 1
    torch.testing.assert_close(initial - (mass[:, None] * vel).sum(0), impulse)
    assert impulse.tolist() == pytest.approx([2.25, 0.0, 0.0])


def test_fixed_window_pressure_integral_and_censored_record():
    import runpy
    from pathlib import Path

    score = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmarks/marin_box_loads/compare.py")
    )
    t = np.linspace(0, 1, 1001)
    p = 35.0 + 4000.0 * np.maximum(t - 0.35, 0.0)
    result = score["tap_event"](t, p)
    assert result["impulse_Pa_s"] == pytest.approx(0.5 * 4000 * 0.35**2)
    assert result["arrival_s"] == pytest.approx(0.475)
    with pytest.raises(ValueError, match="does not span"):
        score["tap_event"](t[:420], p[:420])


def test_solver_target_uses_the_workbook_peak_and_arrival():
    import json
    from pathlib import Path

    from tensorsph.marin_probes import H2_ARRIVAL_S, H2_PEAK_M, H2_PEAK_TIME_S

    path = Path(__file__).resolve().parents[1] / "benchmarks/marin_box_loads/comparison.json"
    measured = json.loads(path.read_text())["runs"]["dx50"]["H2"]
    assert (H2_PEAK_M, H2_PEAK_TIME_S, H2_ARRIVAL_S) == pytest.approx(
        (measured["measured_peak_m"], measured["measured_peak_t_s"], measured["observed_arrival_s"])
    )


def test_fixed_area_disk_quadrature_and_dry_area():
    from tensorsph.marin_probes import disk_average, tap_disk_nodes

    for dx in (.05, .032):
        cfg = Kleefsman3DConfig(dx=dx)
        taps = sensor_positions(cfg, torch.empty(0, dtype=torch.float64))
        radius = .01
        nodes = tap_disk_nodes(cfg, taps, 2 * radius)
        diff = nodes - taps[:, None, :]
        assert torch.all(diff[:4, :, 0] == 0) and torch.all(diff[4:, :, 1] == 0)
        r2 = diff.square().sum(-1)
        torch.testing.assert_close(disk_average(r2), torch.full_like(taps[:, 0], radius**2/2))
        torch.testing.assert_close(disk_average(r2**2), torch.full_like(taps[:, 0], radius**4/3))
        wet = torch.zeros_like(r2)
        wet[:, 0] = 1000.
        assert torch.all(disk_average(wet) == 250.)
        with pytest.raises(ValueError, match="beyond"):
            tap_disk_nodes(cfg, taps, .05)


def test_pressure_peaks_require_every_matching_solver_step(monkeypatch):
    import runpy
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1] / "benchmarks" / "marin_box_loads"
    monkeypatch.syspath_prepend(str(bench))
    check = runpy.run_path(str(bench / "footprint.py"))["check_impact_clock"]
    dt = 0.00043875
    steps = 1596
    meta = dict(dt=dt, steps=steps, output_every=1, simulated_time_s=steps * dt)
    t = np.round(np.arange(steps + 1) * dt, 6)
    check(t, meta)
    with pytest.raises(ValueError, match="every solver step"):
        check(np.delete(t, 1000), meta)
    t[1100] += 0.0001
    with pytest.raises(ValueError, match="event clock"):
        check(t, meta)
