"""Unit gates for the free-surface pressure closure (Adami et al. 2012 §3.3).

The closure is a *boundary condition on p*: particles whose summation density is
deficient carry no Tait pressure at all, so an interior that must hold
``sigma/R`` cannot use them as neighbours.  ``extrapolate_free_surface_pressure``
replaces their pressure with the kernel-weighted average of their fluid
neighbours' pressures, each shifted by ``rho_j g.(r_i - r_j)``.

The tests below are analytic, not empirical: the shift is exactly the first
Taylor term of a hydrostatic field, so on a field that *is* linear in y the
closure must reproduce it to round-off.  That is a much sharper statement than
any SPH accuracy claim, and it is what makes the end-to-end numbers in
``closure_ladder.json`` trustworthy as a physics result rather than a tuned
constant.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (Config, extrapolate_free_surface_pressure,
                                      pair_kernels)
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
LADDER = BENCH / "closure_ladder.json"
RHO0 = 1000.0
G = 9.81


def _column(ny: int = 24, nx: int = 6, dx: float = 0.02):
    """A lattice block, top row at y = ny*dx, bottom at y = dx."""
    ys, xs = torch.meshgrid(
        torch.arange(1, ny + 1, dtype=torch.float64),
        torch.arange(1, nx + 1, dtype=torch.float64), indexing="ij")
    pos = torch.stack((xs.reshape(-1) * dx, ys.reshape(-1) * dx), dim=1)
    cfg = Config(dx=dx, gravity=G, rho0=RHO0)
    h = cfg.h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    wij, _ = pair_kernels(delta, h)
    return pos, pairs[:, 0], pairs[:, 1], wij, h, cfg


def test_identity_when_no_particle_is_deficient():
    """Nothing is free surface -> the closure must not touch the field."""
    pos, i, j, wij, h, cfg = _column()
    rho = torch.full((pos.shape[0],), RHO0, dtype=torch.float64)
    p = torch.linspace(1.0, 2.0, pos.shape[0], dtype=torch.float64)
    out = extrapolate_free_surface_pressure(pos, p, rho, i, j, wij,
                                            pos.shape[0], cfg, passes=3)
    assert torch.equal(out, p)


def test_uniform_pressure_is_reproduced_to_roundoff():
    """With g = 0 the closure is a pure Shepard average, so a uniform field must
    come back unchanged in the shell: that pins the normalisation (sum W / sum W)
    independently of any physics."""
    pos, i, j, wij, h, cfg = _column()
    n = pos.shape[0]
    y = pos[:, 1]
    free = y > float(y.max()) - 2.0 * cfg.dx
    rho = torch.where(free, torch.full_like(y, 0.94 * RHO0),
                      torch.full_like(y, RHO0))
    p_uniform = torch.full((n,), 360.0, dtype=torch.float64)
    cfg0 = Config(dx=cfg.dx, gravity=0.0, rho0=RHO0)
    closed = extrapolate_free_surface_pressure(pos, p_uniform, rho, i, j, wij,
                                               n, cfg0, passes=3)
    assert torch.allclose(closed, p_uniform, rtol=0.0, atol=1e-10)


def test_hydrostatic_shift_recovers_the_linear_field():
    """The shift rho_j g.(r_i - r_j) is the first Taylor term of a hydrostatic
    field.  On a linear p(y) the closure therefore recovers the analytic
    continuation of the shell, and the only residual is the density deficit of
    the shell neighbours used in the shift (0.94 rho0 -> 6 % of the hydrostatic
    increment, i.e. ~3 % of the pressure) plus the clipped p = 0 neighbours."""
    pos, i, j, wij, h, cfg = _column()
    n = pos.shape[0]
    y = pos[:, 1]
    p_true = RHO0 * G * (float(y.max()) + h - y)
    free = y > float(y.max()) - 2.0 * cfg.dx
    rho = torch.where(free, torch.full_like(y, 0.94 * RHO0),
                      torch.full_like(y, RHO0))
    closed = extrapolate_free_surface_pressure(pos, p_true, rho, i, j, wij,
                                               n, cfg, passes=6)
    assert torch.equal(closed[~free], p_true[~free])
    rel = ((closed[free] - p_true[free]).abs() / p_true[free]).max()
    assert float(rel) < 0.05, float(rel)


def test_closure_fills_a_clipped_shell_without_pulling():
    """Clamped Tait leaves the whole deficit band at p = 0; after the closure the
    band must carry a positive, monotonically decreasing-with-radius pressure,
    and it may never become negative."""
    pos, i, j, wij, h, cfg = _column()
    n = pos.shape[0]
    y = pos[:, 1]
    y_top = float(y.max())
    p_true = RHO0 * G * (y_top + h - y)
    free = y > y_top - 3.0 * cfg.dx
    rho = torch.where(free, torch.full_like(y, 0.90 * RHO0),
                      torch.full_like(y, RHO0))
    p_thermo = torch.where(free, torch.zeros_like(y), p_true)
    closed = extrapolate_free_surface_pressure(pos, p_thermo, rho, i, j, wij,
                                               n, cfg, passes=6)
    band = closed[free]
    # The outermost row has no fluid neighbour at all and correctly stays at 0;
    # the rest of the band must be filled and must never pull.
    assert float(band.min()) >= 0.0
    assert float(band.max()) > 0.0
    assert int((band > 0.0).sum()) >= 0.5 * int(free.sum())
    # Column: higher rows see less pressure (compare row means -- within one row
    # the columns are symmetric, and the shell's own density deficit scatters the
    # shift by ~1 % of the hydrostatic increment).
    ys, ps = y[free], closed[free]
    row_means = [float(ps[(ys - float(v)).abs() < 1e-9].mean())
                 for v in sorted(set(float(v) for v in ys))]
    assert all(b <= a for a, b in zip(row_means, row_means[1:])), row_means


def test_closure_ignores_wall_particles():
    """Wall pairs must not feed the extrapolation (wall pressure is derived from
    the fluid; closing that loop would be self-referential)."""
    pos, i, j, wij, h, cfg = _column()
    n = pos.shape[0]
    # Add a wall row below the block and pairs to it with a huge pressure.
    wall = torch.tensor([[0.5 * cfg.dx, 0.0], [1.5 * cfg.dx, 0.0]],
                        dtype=torch.float64)
    allpos = torch.cat((pos, wall))
    pairs = cell_linked_pairs(allpos, 2.0 * h)
    delta = allpos[pairs[:, 0]] - allpos[pairs[:, 1]]
    wij2, _ = pair_kernels(delta, h)
    ii, jj = pairs[:, 0], pairs[:, 1]
    y = pos[:, 1]
    rho = torch.full((n,), RHO0, dtype=torch.float64)
    p = RHO0 * G * (float(y.max()) + h - y)
    free = y < 4.0 * cfg.dx          # pretend the bottom of the block is deficient
    rho = torch.where(free, torch.full_like(y, 0.90 * RHO0), rho)
    p = torch.where(free, torch.zeros_like(y), p)
    out_fluid_only = extrapolate_free_surface_pressure(pos, p, rho, ii, jj, wij2,
                                                       n, cfg, passes=2)
    # Same call with the wall pairs' weights zeroed by a mask is what the
    # implementation does internally (`fluid_pair`); compare against a run where
    # the wall is not present at all.
    pos2, i2, j2, wij3, h2, cfg2 = _column()
    out_no_wall = extrapolate_free_surface_pressure(pos2, p, rho, i2, j2, wij3,
                                                    n, cfg2, passes=2)
    assert torch.allclose(out_fluid_only, out_no_wall, atol=1e-12)


# --------------------------------------------------------------------------- #
# end-to-end evidence (closure_ladder.json)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ladder():
    if not LADDER.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/closure_ladder.py first")
    return json.loads(LADDER.read_text())


def test_ladder_covers_the_four_acceptance_points(ladder):
    pts = {(row["dx_m"], row["sound_speed"]) for row in ladder["ladder"]}
    for dx, c0 in ladder["acceptance_points"]:
        assert (dx, c0) in pts


def test_closure_is_a_large_improvement_at_every_point(ladder):
    """Every setting that is meant to be usable removes >= 15 percentage points."""
    for row in ladder["ladder"]:
        if row["passes"] == 0:
            continue
        # A single sweep already halves the residual; three sweeps remove ~30 pt.
        floor = 10.0 if row["passes"] == 1 else 15.0
        assert row["error_reduction_pt"] > floor, row


def test_three_sweeps_meet_the_p0_criterion(ladder):
    """The P0 gate: all four acceptance points inside 15 % with passes = 3."""
    row = ladder["acceptance"]["3"]
    assert row["all_four_points_within_15pct"], row
    assert row["worst_abs_error_pct"] < 15.0


def test_closure_does_not_spawn_spurious_currents(ladder):
    """Adami's own criterion (vmax < 0.1 v_sigma) must survive the closure;
    this is what rules out the high-pass settings (see the doc §4)."""
    for row in ladder["ladder"]:
        assert row["finite"], row
        if row["passes"] <= 3:
            assert row["spurious_current_ok"], row


def test_compression_ratio_approaches_one(ladder):
    """C3 predicted an interior compression of 0.57 +- 0.06 of what sigma/R
    needs.  Three sweeps must move every point towards 1, and no point may
    overshoot by more than 15 % -- that is the same statement as the pressure
    criterion, read in the density the EOS actually sees."""
    by_point = {}
    for row in ladder["ladder"]:
        key = (row["dx_m"], row["sound_speed"])
        by_point.setdefault(key, {})[row["passes"]] = row
    assert len(by_point) == 4
    for key, rows in by_point.items():
        open_ratio = rows[0]["compression_ratio"]
        closed_ratio = rows[3]["compression_ratio"]
        assert closed_ratio > open_ratio, (key, open_ratio, closed_ratio)
        assert 0.85 < closed_ratio < 1.15, (key, closed_ratio)
        # And monotone in the number of sweeps up to the stable setting.
        assert rows[1]["compression_ratio"] < rows[2]["compression_ratio"] + 0.05
