"""Unit and smoke tests for the Kleefsman dam-break-with-obstacle case."""
import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.kleefsman_obstacle import (
    KleefsmanConfig,
    _peak_metrics,
    make_water,
    obstacle_lattice,
    run_case,
    validate_case,
    water_height_at,
)

REF = json.loads(Path(__file__).resolve().parents[1].joinpath(
    "benchmarks/kleefsman_obstacle/reference.json").read_text())


def cfg(**kw):
    g = REF["geometry_m"]
    base = dict(
        dx=0.05, t_end=0.05, device="cpu",
        tank_length=g["tank_length"], tank_height=g["tank_height"],
        column_length=g["column_length"], column_height=g["column_height"],
        obstacle_x=g["obstacle_upstream_face_x"],
        obstacle_length=g["obstacle_length"], obstacle_height=g["obstacle_height"],
        pool_width=g["tank_length"], pool_depth=g["tank_height"],
        dt=1.0e-4, output_every=5,
    )
    base.update(kw)
    return KleefsmanConfig(**base)


def test_geometry_is_consistent():
    g = REF["geometry_m"]
    assert abs(g["column_length"] + (g["obstacle_upstream_face_x"] - g["column_length"])
               + g["downstream_gap"] - g["tank_length"]) < 1e-12
    assert abs(g["obstacle_upstream_face_x"] + g["obstacle_length"]
               + (g["downstream_gap"] - g["obstacle_length"]) - g["tank_length"]) < 1e-12


def test_validate_rejects_bad_probe_and_box():
    validate_case(cfg())
    with pytest.raises(ValueError):
        validate_case(cfg(probe_offset=-0.5))          # probe upstream of the gate
    with pytest.raises(ValueError):
        validate_case(cfg(obstacle_length=2.0))        # box pokes out of the tank
    with pytest.raises(ValueError):
        validate_case(cfg(boundary="periodic"))


def test_obstacle_lattice_covers_box_interior():
    c = cfg(dx=0.01)
    wall = obstacle_lattice(c, torch.device("cpu"))
    assert wall.shape[0] == 16 * 16
    assert float(wall[:, 0].min()) >= c.obstacle_x
    assert float(wall[:, 0].max()) <= c.obstacle_x + c.obstacle_length
    assert float(wall[:, 1].max()) <= c.obstacle_height


def test_water_column_lattice_and_mass():
    c = cfg(dx=0.02)
    pos, mass = make_water(c, torch.device("cpu"))
    # cell_centres() uses floor(extent/dx) so the last cell never pokes through
    # the wall; the count must use the same convention (round() over-counted the
    # 0.55 m column at dx = 0.02: 28 rows asserted vs 27 rows built).
    assert pos.shape[0] == math.floor(c.column_length / c.dx) * math.floor(c.column_height / c.dx)
    assert bool((mass > 0).all())
    assert float(mass.sum()) > 0.9 * c.rho0 * c.column_length * c.column_height
    assert float(pos[:, 0].max()) <= c.column_length
    assert float(pos[:, 1].max()) <= c.column_height


def _column(c, top, y0=0.0):
    ys = torch.arange(y0 + c.dx, top + 0.5 * c.dx, c.dx)
    return torch.stack((torch.full_like(ys, c.probe_x()), ys), dim=-1)


def test_water_height_at_strip():
    c = cfg(dx=0.01, surface_particles=1)
    pos = _column(c, 0.25)                     # contiguous column, surface 0.25
    assert abs(water_height_at(pos, c) - 0.25) < 1e-6
    # a particle outside the probe strip is ignored
    far = torch.tensor([[c.probe_x() + 10 * c.dx, 0.9]], dtype=torch.float32)
    assert abs(water_height_at(torch.cat((pos, far)), c) - 0.25) < 1e-6
    empty = torch.tensor([[0.1, 0.3]], dtype=torch.float32)
    assert water_height_at(empty, c) == 0.0


def test_surface_estimator_rejects_detached_spray():
    """The dx = 0.01 run reported peak = 1.000 m (the tank roof) because the
    top-3 mean over the whole strip counted ballistic spray.  The contiguous
    stack must report the water body, and the legacy mode the spray."""
    c = cfg(dx=0.01, surface_particles=3)
    body = _column(c, 0.25)
    spray = torch.tensor([[c.probe_x(), 0.90], [c.probe_x(), 0.92],
                          [c.probe_x() + 0.5 * c.dx, 0.95]], dtype=torch.float32)
    pos = torch.cat((body, spray))
    robust = water_height_at(pos, c)
    assert 0.23 <= robust <= 0.26, robust          # water body, not spray
    c_legacy = cfg(dx=0.01, surface_particles=3, robust_surface=False)
    assert water_height_at(pos, c_legacy) > 0.9    # reproduces the old defect


def test_surface_estimator_survives_a_small_surface_hole():
    """One missing particle in the column is not a detached droplet."""
    c = cfg(dx=0.01, surface_particles=1)
    body = _column(c, 0.25)
    holed = body[(body[:, 1] < 0.225) | (body[:, 1] > 0.235)]  # drop the 0.23 row
    assert abs(water_height_at(holed, c) - 0.25) < 1e-6


def test_peak_metrics_interpolates_arrival():
    times = [0.0, 0.1, 0.2, 0.3]
    height = [0.0, 0.0, 0.04, 0.02]
    peak, t_peak, arrival = _peak_metrics(times, height, 0.02)
    assert peak == 0.04 and t_peak == 0.2
    assert abs(arrival - 0.15) < 1e-12          # crossing between 0.1 and 0.2
    assert math.isnan(_peak_metrics(times, [0.0] * 4, 0.02)[2])


def test_chunked_run_matches_a_single_run():
    """A time-budgeted run must be bit-identical to an uninterrupted one.

    The build node kills a shell command after ~120 s, so the fine grids have to
    be advanced in chunks; if resuming perturbed the trajectory the fine-grid
    numbers would not be comparable with the coarse ones.
    """
    c = cfg(dx=0.05, t_end=0.01, output_every=1)
    whole = run_case(c)
    state, pieces, part = None, 0, None
    while pieces < 500:
        part = run_case(c, resume=state, time_budget=0.0)
        state = part.state
        pieces += 1
        if part.finished:
            break
    assert part.finished and pieces > 3
    assert abs(part.simulated_time - whole.simulated_time) < 1e-12
    assert part.times == whole.times
    assert part.height == whole.height
    assert part.front == whole.front
    assert part.escaped == whole.escaped


def test_short_run_conserves_mass_and_keeps_fluid_in_tank():
    c = cfg(dx=0.05, t_end=0.02)
    run = run_case(c)
    assert run.escaped == 0
    assert run.mass_error < 1e-12
    assert run.times and run.times[-1] >= c.t_end
    assert all(h >= 0.0 for h in run.height)
    assert run.front and run.front[-1] <= c.tank_length
    assert math.isfinite(run.max_density_compression)
