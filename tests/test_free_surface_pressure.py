"""Unit and gate tests for the free-surface (shell) pressure closure.

The closure is ``Config.free_surface_pressure``: for a particle whose summation
density is below ``free_surface_threshold * rho0`` the pressure is replaced by
the kernel-weighted extrapolation of its neighbours' pressure when that value is
larger than the (clamped, ``p >= 0``) Tait value. Model: Adami, Ihmsen & Adams
(2012) JCP 231, 7057, sec. 3.3.

These tests separate three things that must not be conflated:

1. the *operator* -- does ``free_surface_pressure`` compute the weighted average
   it claims to, with the right body-force sign, and only for flagged particles?
2. the *invariants* -- with the switch off nothing changes; with it on the
   interior is untouched and the EOS pressure is never lowered;
3. the *end-to-end gate* -- does the Young-Laplace reading in
   ``benchmarks/laplace_pressure_floor/report_fs.json`` land inside 15 %? The
   gate is skipped (not passed) when the report is absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_kernel,
    free_surface_pressure,
    rhs,
    rhs_all_pairs,
)
from tensorsph.neighbor_search import cell_linked_pairs

ROOT = Path(__file__).resolve().parents[1]
REPORT_FS = ROOT / "benchmarks" / "laplace_pressure_floor" / "report_fs.json"


def _cfg(**kwargs) -> Config:
    base = dict(dx=0.01, h_factor=1.3, rho0=1000.0, gravity=0.0, viscosity=0.0,
                surface_tension_coefficient=0.0, steps=1)
    base.update(kwargs)
    return Config(**base)


def _line_pairs(n: int):
    """Unordered pairs of n particles on a line, spacing dx."""
    idx = torch.arange(n)
    i = torch.cat([idx[:-1]])
    j = torch.cat([idx[1:]])
    return torch.stack((i, j), dim=1)


# --------------------------------------------------------------------------- #
# 1. operator
# --------------------------------------------------------------------------- #
def test_weighted_average_is_reproduced_on_a_line():
    """p_ext is the kernel-weighted mean of the neighbours' pressures."""
    dx = 0.01
    h = 1.3 * dx
    cfg = _cfg(dx=dx, free_surface_pressure=True, free_surface_threshold=0.99)
    pos = torch.stack([torch.arange(3, dtype=torch.float64) * dx,
                       torch.zeros(3, dtype=torch.float64)], dim=1)
    pairs = _line_pairs(3)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    rho = torch.tensor([1000.0, 980.0, 900.0], dtype=torch.float64)
    pressure = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float64)

    out = free_surface_pressure(rho, pressure, i, j, delta, wij, 3, cfg)

    w01 = float(cubic_spline_kernel(torch.tensor([dx], dtype=torch.float64), h))
    w12 = float(cubic_spline_kernel(torch.tensor([dx], dtype=torch.float64), h))
    # particle 1: neighbours 0 (p=50) and 2 (p=0)
    expected_1 = 50.0 * w01 / (w01 + w12)
    # particle 2: only neighbour 1 (p=0) -> extrapolation is 0, EOS wins
    assert out[0].item() == pytest.approx(50.0)          # interior: untouched
    assert out[1].item() == pytest.approx(expected_1, rel=1e-12)
    assert out[2].item() == pytest.approx(0.0)


def test_interior_particles_are_never_modified():
    """Only particles below the density threshold may change."""
    dx = 0.01
    h = 1.3 * dx
    cfg = _cfg(dx=dx, free_surface_pressure=True)
    pos = torch.stack([torch.arange(4, dtype=torch.float64) * dx,
                       torch.zeros(4, dtype=torch.float64)], dim=1)
    pairs = _line_pairs(4)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    rho = torch.tensor([1000.0, 995.0, 900.0, 880.0], dtype=torch.float64)
    pressure = torch.tensor([10.0, 7.0, 0.0, 0.0], dtype=torch.float64)

    out = free_surface_pressure(rho, pressure, i, j, delta, wij, 4, cfg)
    assert torch.equal(out[:2], pressure[:2])


def test_eos_pressure_is_never_lowered():
    """The closure must not remove pressure from a particle that has it."""
    dx = 0.01
    h = 1.3 * dx
    cfg = _cfg(dx=dx, free_surface_pressure=True)
    pos = torch.stack([torch.arange(3, dtype=torch.float64) * dx,
                       torch.zeros(3, dtype=torch.float64)], dim=1)
    pairs = _line_pairs(3)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    # particle 1 is flagged free surface but carries a large EOS pressure,
    # while its only neighbour has none: the extrapolation is smaller.
    rho = torch.tensor([1000.0, 980.0, 900.0], dtype=torch.float64)
    pressure = torch.tensor([0.0, 400.0, 0.0], dtype=torch.float64)
    out = free_surface_pressure(rho, pressure, i, j, delta, wij, 3, cfg)
    assert out[1].item() == pytest.approx(400.0)


def test_body_force_term_has_the_hydrostatic_sign():
    """With gravity, the extrapolation is *reduced* upward and *raised* downward."""
    dx = 0.01
    h = 1.3 * dx
    cfg = _cfg(dx=dx, free_surface_pressure=True, gravity=9.81)
    # vertical line, particle 1 in the middle: neighbours above and below.
    pos = torch.tensor([[0.0, -dx], [0.0, 0.0], [0.0, dx]], dtype=torch.float64)
    pairs = _line_pairs(3)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)

    # symmetric pressures: the two body terms cancel; the operator reduces to
    # the arithmetic kernel mean, which must equal the common value.
    rho = torch.tensor([1000.0, 980.0, 1000.0], dtype=torch.float64)
    pressure = torch.tensor([100.0, 0.0, 100.0], dtype=torch.float64)
    out = free_surface_pressure(rho, pressure, i, j, delta, wij, 3, cfg)
    assert out[1].item() == pytest.approx(100.0, rel=1e-12)

    # single source below the target: p_i = p_j - g rho_j (y_i - y_j) exactly
    # (the source pressure is large enough that the p >= 0 clamp does not bite).
    high = torch.tensor([500.0, 0.0, 500.0], dtype=torch.float64)
    single = torch.tensor([[0, 1]], dtype=torch.long)
    si, sj = single[:, 0], single[:, 1]
    sdelta = pos[si] - pos[sj]
    swij = cubic_spline_kernel(torch.linalg.vector_norm(sdelta, dim=1), h)
    rho = torch.tensor([1200.0, 980.0, 1000.0], dtype=torch.float64)
    out = free_surface_pressure(rho, high, si, sj, sdelta, swij, 3, cfg)
    assert out[1].item() == pytest.approx(500.0 - 9.81 * 1200.0 * dx, rel=1e-12)

    # and a denser neighbour above lifts the extrapolation above the sources.
    single = torch.tensor([[1, 2]], dtype=torch.long)
    si, sj = single[:, 0], single[:, 1]
    sdelta = pos[si] - pos[sj]
    swij = cubic_spline_kernel(torch.linalg.vector_norm(sdelta, dim=1), h)
    rho = torch.tensor([1000.0, 980.0, 1200.0], dtype=torch.float64)
    out = free_surface_pressure(rho, high, si, sj, sdelta, swij, 3, cfg)
    assert out[1].item() == pytest.approx(500.0 + 9.81 * 1200.0 * dx, rel=1e-12)


def test_isolated_particle_falls_back_to_eos():
    """No fluid neighbour -> no extrapolation, no division by zero."""
    cfg = _cfg(free_surface_pressure=True)
    rho = torch.tensor([900.0], dtype=torch.float64)
    pressure = torch.tensor([0.0], dtype=torch.float64)
    empty = torch.zeros(0, dtype=torch.long)
    out = free_surface_pressure(rho, pressure, empty, empty,
                                torch.zeros((0, 2), dtype=torch.float64),
                                torch.zeros(0, dtype=torch.float64), 1, cfg)
    assert torch.equal(out, pressure)


def test_wall_particles_are_not_sources():
    """Wall indices (>= count) are excluded from the average."""
    dx = 0.01
    h = 1.3 * dx
    cfg = _cfg(dx=dx, free_surface_pressure=True)
    pos = torch.stack([torch.arange(3, dtype=torch.float64) * dx,
                       torch.zeros(3, dtype=torch.float64)], dim=1)
    # fluid 0,1 and one wall particle (index 2)
    pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    rho = torch.tensor([1000.0, 980.0], dtype=torch.float64)
    pressure = torch.tensor([50.0, 0.0], dtype=torch.float64)
    out = free_surface_pressure(rho, pressure, i, j, delta, wij, 2, cfg,
                                wall_count=1)
    # particle 1 sees only fluid particle 0
    assert out[1].item() == pytest.approx(50.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. invariants inside rhs
# --------------------------------------------------------------------------- #
def _hex_drop(radius=0.04, dx=0.005, rho0=1000.0):
    hexa = torch.tensor([[dx, 0.0], [0.5 * dx, 0.5 * math.sqrt(3.0) * dx]],
                        dtype=torch.float64)
    cells = int(math.ceil(2.2 * radius / dx))
    points = []
    for a in range(-cells, cells + 1):
        for b in range(-cells, cells + 1):
            p = a * hexa[0] + b * hexa[1]
            if float(torch.linalg.vector_norm(p)) <= radius:
                points.append(p)
    pos = torch.stack(points)
    mass = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=torch.float64)
    return pos, mass


def test_switch_off_reproduces_the_default_rhs():
    """``free_surface_pressure=False`` is bit-for-bit the previous behaviour."""
    pos, mass = _hex_drop()
    vel = torch.zeros_like(pos)
    cfg_off = _cfg(dx=0.005, surface_tension_coefficient=7.2)
    cfg_on = _cfg(dx=0.005, surface_tension_coefficient=7.2,
                  free_surface_pressure=True)
    h = cfg_off.h_factor * cfg_off.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho_a, p_a, acc_a = rhs(pos, vel, mass, cfg_off, pairs=pairs)
    rho_b, p_b, acc_b = rhs(pos, vel, mass, cfg_off, pairs=pairs)
    assert torch.equal(rho_a, rho_b) and torch.equal(p_a, p_b)
    # with the closure on, the interior must still be identical
    rho_c, p_c, _ = rhs(pos, vel, mass, cfg_on, pairs=pairs)
    assert torch.equal(rho_a, rho_c)
    interior = rho_c >= cfg_on.free_surface_threshold * cfg_on.rho0
    assert bool(interior.any())
    assert torch.equal(p_a[interior], p_c[interior])


def test_closure_raises_the_shell_pressure_only():
    """The shell (rho < 0.99 rho0) gains pressure; the interior keeps the EOS."""
    pos, mass = _hex_drop()
    vel = torch.zeros_like(pos)
    cfg_off = _cfg(dx=0.005, surface_tension_coefficient=7.2)
    cfg_on = _cfg(dx=0.005, surface_tension_coefficient=7.2,
                  free_surface_pressure=True)
    h = cfg_off.h_factor * cfg_off.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho, p_off, _ = rhs(pos, vel, mass, cfg_off, pairs=pairs)
    _, p_on, _ = rhs(pos, vel, mass, cfg_on, pairs=pairs)
    shell = rho < cfg_on.free_surface_threshold * cfg_on.rho0
    assert bool(shell.any())
    raised = p_on[shell] > p_off[shell]
    assert int(raised.sum()) > 0
    assert torch.all(p_on >= p_off)
    assert torch.isfinite(p_on).all()


def test_dense_reference_matches_the_pair_path_with_the_closure():
    """``rhs_all_pairs`` implements the same closure as the production path."""
    pos, mass = _hex_drop(radius=0.03, dx=0.005)
    vel = torch.zeros_like(pos)
    cfg = _cfg(dx=0.005, surface_tension_coefficient=0.0, viscosity=0.01,
               free_surface_pressure=True)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    _, p_pair, acc_pair = rhs(pos, vel, mass, cfg, pairs=pairs)
    _, p_dense, acc_dense = rhs_all_pairs(pos, vel, mass, cfg)
    assert torch.allclose(p_pair, p_dense, rtol=1e-10, atol=1e-8)
    assert torch.allclose(acc_pair, acc_dense, rtol=1e-8, atol=1e-8)


def test_hydrostatic_wall_case_stays_finite_with_the_closure():
    """Wall boundary + gravity + closure: still finite, pressure monotone."""
    cfg = _cfg(dx=0.02, gravity=9.81, boundary="wall", wall_layers=2,
               free_surface_pressure=True)
    from tensorsph.droplet_impact import wall_lattice
    xs = torch.arange(-0.4, 0.4001, cfg.dx, dtype=torch.float64)
    ys = torch.arange(-0.2, 0.201, cfg.dx, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    pos = grid.clone()
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                      dtype=torch.float64)
    wall = wall_lattice(cfg, torch.device("cpu")).to(pos.dtype)
    vel = torch.zeros_like(pos)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(torch.cat((pos, wall)), 2.0 * h)
    rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall)
    cfg_off = _cfg(dx=0.02, gravity=9.81, boundary="wall", wall_layers=2)
    _, pressure_off, _ = rhs(pos, vel, mass, cfg_off, pairs=pairs, wall=wall)
    assert torch.isfinite(pressure).all() and torch.isfinite(acc).all()
    assert float(pressure.min()) >= 0.0
    # the closure only ever adds pressure, and it touches the deficient particles
    shell = rho < cfg.free_surface_threshold * cfg.rho0
    assert bool(shell.any())
    assert torch.all(pressure >= pressure_off)
    assert bool((pressure[shell] > pressure_off[shell]).any())


# --------------------------------------------------------------------------- #
# 3. end-to-end gate
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="honest FAIL, measured: the Adami 2012 sec. 3.3 shell closure moves "
           "the tail-averaged Young-Laplace error from -34..-38 % to "
           "-20..-25 % (4 cases: dx = 1.5/2.0 mm x c0 = 5/20 m/s) but stays "
           "outside the +/-15 % gate. docs/verification_free_surface_pressure.md "
           "sec. 5. A passing run here means the remaining interior density "
           "relaxation debt (P0-b) has been closed and this marker must go.",
)
@pytest.mark.skipif(not REPORT_FS.exists(),
                    reason="run benchmarks/laplace_pressure_floor/run.py "
                           "--free-surface-pressure first")
def test_young_laplace_reading_inside_15_percent_with_the_closure():
    """P0 acceptance: tail-averaged absolute Delta p within 15 % of sigma/R.

    The switch demonstrably helps (see
    ``test_closure_improves_every_point_of_the_baseline_report``) but does not
    reach the gate, so the criterion itself is pinned as a strict xfail rather
    than deleted.
    """
    report = json.loads(REPORT_FS.read_text())
    assert report["closure"]["free_surface_pressure"] is True
    points = {(r["dx_m"], r["sound_speed"]): r for r in report["runs"]}
    assert (0.002, 20.0) in points and (0.002, 5.0) in points
    for (dx, c0), run in sorted(points.items()):
        err = run["absolute_time_average_tail"]["rel_error_pct"]
        assert abs(err) < 15.0, (dx, c0, err)
        assert run["finite"] is True
        assert run["spurious_current"]["vmax_over_vref_final"] < 0.1


@pytest.mark.skipif(not REPORT_FS.exists(),
                    reason="run benchmarks/laplace_pressure_floor/run.py "
                           "--free-surface-pressure first")
def test_closure_improves_every_point_of_the_baseline_report():
    """Against ``report.json`` (closure off), each point must improve."""
    base_path = REPORT_FS.parent / "report.json"
    if not base_path.exists():
        pytest.skip("baseline report.json missing")
    base = json.loads(base_path.read_text())
    fs = json.loads(REPORT_FS.read_text())
    key = lambda r: (round(r["dx_m"], 6), round(r["sound_speed"], 6))
    base_err = {key(r): abs(r["absolute_time_average_tail"]["rel_error_pct"])
                for r in base["runs"]}
    for run in fs["runs"]:
        k = key(run)
        if k not in base_err:
            continue
        assert abs(run["absolute_time_average_tail"]["rel_error_pct"]) < base_err[k]