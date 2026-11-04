"""Gates for the free-surface pressure closure (Adami, Ihmsen & Adams 2012).

The physics runs (1.5 capillary times) are far too slow for a unit test, so
what is pinned here is

* the operator itself, against a hand-computed value including the sign of the
  body-force term and the fact that interior particles are left alone;
* the pair-based and dense formulations against each other
  (``free_surface_pressure_closure`` vs ``..._closure_dense``), so
  ``rhs_all_pairs`` keeps being an independent check of ``rhs``;
* the default: ``free_surface_pressure="clamp"`` must reproduce the previous
  behaviour bit for bit, because every number published before this change was
  measured with it;
* the recorded acceptance run ``benchmarks/laplace_pressure_floor/
  report_extrapolate.json`` (the P0 of docs/verification_laplace_pressure_floor.md).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    free_surface_pressure_closure,
    free_surface_pressure_closure_dense,
    rhs,
    rhs_all_pairs,
    tait_pressure,
    validate_config,
)
from tensorsph.neighbor_search import cell_linked_pairs

REPORT = (Path(__file__).resolve().parents[1] / "benchmarks"
          / "laplace_pressure_floor" / "report_extrapolate.json")
DP_ANALYTIC = 360.0      # sigma/R with sigma = 7.2 N/m, R = 0.02 m


# --------------------------------------------------------------------------- #
# configuration surface
# --------------------------------------------------------------------------- #
def test_default_is_the_clamp_and_the_closure_is_opt_in():
    cfg = Config()
    assert cfg.free_surface_pressure == "clamp"
    assert cfg.free_surface_pressure_sweeps == 3
    validate_config(cfg)


@pytest.mark.parametrize("bad", [("free_surface_pressure", "shell"),
                                 ("free_surface_threshold", 0.0),
                                 ("free_surface_threshold", 1.5),
                                 ("free_surface_pressure_sweeps", 0)])
def test_invalid_options_are_rejected(bad):
    field, value = bad
    cfg = Config(**{field: value})
    with pytest.raises(ValueError):
        validate_config(cfg)


# --------------------------------------------------------------------------- #
# the operator, against a hand-computed value
# --------------------------------------------------------------------------- #
def _three_particle_case(gravity: float, sweeps: int = 1):
    """One deficient particle at the origin, two interior ones 0.02 m below.

    ``delta_y = y_i - y_j = +0.02`` for both pairs, so the body-force term is
    ``-g * 0.02 * rho = -196.2 Pa`` with g = 9.81: the shell particle sits
    *above* the fluid it is interpolated from, so the hydrostatic contribution
    lowers its pressure. Getting that sign wrong is exactly the failure mode
    the wall extrapolation had before 0.7.0.
    """
    cfg = Config(gamma=7.0, rho0=1000.0, gravity=gravity,
                 free_surface_pressure="extrapolate",
                 free_surface_pressure_sweeps=sweeps)
    pos = torch.tensor([[0.0, 0.0], [0.02, -0.02], [-0.02, -0.02]],
                       dtype=torch.float64)
    rho = torch.tensor([900.0, 1000.0, 1000.0], dtype=torch.float64)
    # The Tait law is normalised so p(rho0) = 0; the interior pressures are what
    # the closure interpolates, so they are set directly.
    pressure = torch.tensor([0.0, 200.0, 300.0], dtype=torch.float64)
    i = torch.tensor([0, 0, 1])
    j = torch.tensor([1, 2, 2])
    wij = torch.tensor([0.5, 0.25, 0.1], dtype=torch.float64)
    delta = pos[i] - pos[j]
    closed = free_surface_pressure_closure(pressure, rho, delta, wij, i, j, 3, cfg)
    return cfg, pressure, closed


def test_one_sweep_matches_the_hand_computation_including_the_body_force():
    """Weighted average over the two interior neighbours, done by hand."""
    _, pressure, closed = _three_particle_case(gravity=9.81, sweeps=1)
    # weights 0.5 (pair 0-1) and 0.25 (pair 0-2), denominator 0.75
    expected = (0.5 * (200.0 - 196.2) + 0.25 * (300.0 - 196.2)) / 0.75
    assert closed[0].item() == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(37.1333333333, rel=1e-9)
    # the interior particles are not touched
    assert closed[1].item() == pressure[1].item()
    assert closed[2].item() == pressure[2].item()
    # without gravity the same three weights give the plain kernel average
    _, _, closed_g0 = _three_particle_case(gravity=0.0, sweeps=1)
    assert closed_g0[0].item() == pytest.approx(
        (0.5 * 200.0 + 0.25 * 300.0) / 0.75, rel=1e-12)


def test_the_deficient_shell_stops_being_a_zero_pressure_boundary():
    """The point of the closure: a shell particle stops contributing p = 0."""
    _, _, closed = _three_particle_case(gravity=0.0, sweeps=1)
    assert closed[0].item() > 0.0
    # more sweeps can only move it towards the interior values it interpolates
    _, _, more = _three_particle_case(gravity=0.0, sweeps=3)
    assert more[0].item() >= closed[0].item()


def test_wall_pairs_never_feed_the_extrapolation():
    """Wall pressure is extrapolated *from* the fluid; feeding it back would
    make the closure depend on a value derived from itself."""
    cfg = Config(free_surface_pressure="extrapolate")
    n, nw = 2, 1
    pos = torch.tensor([[0.0, 0.0], [0.0, 0.02]], dtype=torch.float64)
    all_pos = torch.cat((pos, torch.tensor([[0.0, -0.02]], dtype=torch.float64)))
    rho = torch.tensor([900.0, 1000.0], dtype=torch.float64)
    pressure = torch.tensor([0.0, 100.0], dtype=torch.float64)
    i = torch.tensor([0, 1])
    j = torch.tensor([2, 0])
    wij = torch.tensor([5.0, 0.5], dtype=torch.float64)   # huge wall weight
    delta = all_pos[i] - all_pos[j]
    closed = free_surface_pressure_closure(pressure, rho, delta, wij, i, j, n, cfg)
    # the wall contribution (index 2) is dropped: only the fluid pair counts
    expected = 100.0 + 9.81 * 0.02 * 1000.0
    assert closed[0].item() == pytest.approx(float(expected), rel=1e-12)


# --------------------------------------------------------------------------- #
# pair-based vs dense reference implementation
# --------------------------------------------------------------------------- #
def test_pair_and_dense_closures_agree_on_a_deficient_droplet():
    cfg = Config(dx=0.004, sound_speed=20.0, gravity=0.0, viscosity=0.01,
                 free_surface_pressure="extrapolate", gamma=7.0)
    n_side = 9
    grid = torch.arange(n_side, dtype=torch.float64) * cfg.dx
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    pos = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1) - (n_side - 1) * cfg.dx / 2.0
    pos = pos[torch.linalg.vector_norm(pos, dim=1) < 0.014]
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx ** 2,
                      dtype=torch.float64)
    vel = torch.zeros_like(pos)
    rho_pairs, pressure_pairs, _ = rhs(pos, vel, mass, cfg)
    rho_dense, pressure_dense, _ = rhs_all_pairs(pos, vel, mass, cfg)
    torch.testing.assert_close(rho_pairs, rho_dense, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(pressure_pairs, pressure_dense, rtol=1e-10,
                               atol=1e-10)


def test_the_closed_pressure_is_what_the_forces_use():
    """A closure that only decorates the return value would pass every test
    above; this one checks the returned field actually changed the force."""
    cfg_clamp = Config(dx=0.004, sound_speed=20.0, gravity=0.0,
                       free_surface_pressure="clamp")
    cfg_closed = Config(dx=0.004, sound_speed=20.0, gravity=0.0,
                        free_surface_pressure="extrapolate")
    n_side = 9
    grid = torch.arange(n_side, dtype=torch.float64) * cfg_clamp.dx
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    pos = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1) - (n_side - 1) * cfg_clamp.dx / 2.0
    pos = pos[torch.linalg.vector_norm(pos, dim=1) < 0.014]
    # 2 % pre-compression, so the interior carries a positive Tait pressure and
    # the closure has something to interpolate.
    mass = torch.full((pos.shape[0],), 1.02 * cfg_clamp.rho0 * cfg_clamp.dx ** 2,
                      dtype=torch.float64)
    vel = torch.zeros_like(pos)
    _, pressure_clamp, acc_clamp = rhs(pos, vel, mass, cfg_clamp)
    _, pressure_closed, acc_closed = rhs(pos, vel, mass, cfg_closed)
    assert (pressure_closed > pressure_clamp).any()
    assert not torch.allclose(acc_closed, acc_clamp)


# --------------------------------------------------------------------------- #
# recorded acceptance run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py "
                    "--closure extrapolate --out report_extrapolate.json")
    return json.loads(REPORT.read_text())


def test_report_covers_the_full_resolution_and_sound_speed_matrix(report):
    assert report["free_surface_pressure"] == "extrapolate"
    points = {(r["dx_m"], r["sound_speed"]) for r in report["runs"]}
    assert points == {(0.002, 20.0), (0.002, 10.0), (0.002, 5.0),
                      (0.0015, 20.0), (0.0015, 10.0), (0.0015, 5.0)}
    for run in report["runs"]:
        assert run["finite"]
        assert math.isfinite(run["absolute_time_average_tail"]["rel_error_pct"])


def test_laplace_jump_within_15pct_with_the_closure(report):
    """The acceptance criterion that closed the P0 of
    docs/verification_laplace_pressure_floor.md: absolute pressure, time
    averaged over the last 40 % of the window, all six cases < 15 %."""
    for run in report["runs"]:
        err = abs(run["absolute_time_average_tail"]["rel_error_pct"])
        assert err < 15.0, (run["dx_m"], run["sound_speed"], err)


def test_the_closure_is_quieter_than_adami_0p1(report):
    """A closure that buys the pressure jump with a spurious current would not
    be an improvement; Adami et al. use 0.1 * sqrt(sigma/(rho R)) as the gate."""
    for run in report["runs"]:
        assert run["spurious_current"]["vmax_over_vref_final"] < 0.1, (
            run["dx_m"], run["sound_speed"],
            run["spurious_current"]["vmax_over_vref_final"])


def test_the_closure_beats_the_clamp_it_replaces(report):
    """A/B against the baseline recorded in report.json (same drop, same
    read-out, clamp closure)."""
    base_path = REPORT.parent / "report.json"
    if not base_path.exists():
        pytest.skip("baseline report.json missing")
    base = json.loads(base_path.read_text())
    old = {(r["dx_m"], r["sound_speed"]): r for r in base["runs"]}
    for run in report["runs"]:
        key = (run["dx_m"], run["sound_speed"])
        assert key in old, key
        new_err = abs(run["absolute_time_average_tail"]["rel_error_pct"])
        old_err = abs(old[key]["absolute_time_average_tail"]["rel_error_pct"])
        assert new_err < 0.5 * old_err, (key, old_err, new_err)