"""Gates for the free-surface pressure closure (Adami, Hu & Adams 2012, §3.3).

The closure is the fix identified by
``docs/verification_laplace_pressure_floor.md`` §7 as the single blocking item
of the WITHHELD Young-Laplace verification: with the Tait law clamped at
``p >= 0`` the outer shell of a drop carries zero pressure, and the interior
settles at ~2/3 of ``sigma/R``. Extrapolating the shell pressure from the fluid
neighbours removes that floor.

What is pinned here:
  * off by default, and bit-identical to the clamped Tait law when off
    (the closure must not perturb any published number);
  * on, it only ever *raises* pressure, and only where the kernel support is
    truncated;
  * its value is the kernel-weighted neighbour average -- checked analytically
    on a hand-built three-particle configuration, not against a run;
  * the dense reference path (``rhs_all_pairs``) and the production pair path
    (``rhs``) agree with the closure on, so the two do not drift apart;
  * the acceptance numbers: the four (c0, dx) points of
    ``benchmarks/laplace_pressure_floor/report.json`` -> ``free_surface_runs``
    are within 15 % of ``sigma/R``.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
    extrapolate_free_surface_pressure,
    rhs,
    rhs_all_pairs,
    tait_pressure,
)

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
REPORT = BENCH / "report.json"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("lpf_fsp", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lpf = _load_benchmark_module()


def _cfg(**kwargs) -> Config:
    base = dict(dx=0.002, h_factor=1.3, rho0=1000.0, sound_speed=20.0, gamma=7.0,
                gravity=0.0, viscosity=0.01, surface_tension_coefficient=7.2)
    base.update(kwargs)
    return Config(**base)


def _drop(dx: float = 0.004, radius: float = 0.02):
    """A small 2-D hex lattice drop, density by summation (deficient shell)."""
    cfg = _cfg(dx=dx)
    h = cfg.h_factor * cfg.dx
    pos, mass = lpf.hexagonal_droplet(radius, dx, cfg.rho0)
    rho = density_from_neighbor_pairs(pos, mass, h)
    return cfg, pos, mass, rho, h


# --------------------------------------------------------------------------- #
# 1. default off, bit-identical when off
# --------------------------------------------------------------------------- #
def test_closure_is_off_by_default():
    cfg = Config()
    assert cfg.free_surface_pressure is False
    assert cfg.free_surface_threshold == pytest.approx(0.99)
    assert cfg.free_surface_iterations == 3


def test_off_path_returns_the_clamped_tait_pressure_bit_for_bit():
    """With the flag off the helper is the identity: no published number moves."""
    cfg, pos, _, rho, _ = _drop()
    p = tait_pressure(rho, cfg)
    i = torch.arange(0, pos.shape[0] - 1)
    j = i + 1
    out = extrapolate_free_surface_pressure(
        pos, rho, p, cfg, i, j, torch.ones_like(i, dtype=p.dtype),
        torch.zeros((i.numel(), pos.shape[1])), pos.shape[0])
    assert torch.equal(out, p)


# --------------------------------------------------------------------------- #
# 2. the value is the kernel-weighted neighbour average (analytic)
# --------------------------------------------------------------------------- #
def test_value_is_the_kernel_weighted_neighbour_average():
    cfg = _cfg(free_surface_pressure=True, free_surface_iterations=1,
               free_surface_threshold=0.99)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rho = torch.tensor([980.0, 1000.0, 1000.0])   # particle 0 is truncated
    pressure = torch.tensor([0.0, 120.0, 80.0])
    i = torch.tensor([0, 1, 0, 2])
    j = torch.tensor([1, 0, 2, 0])
    w = torch.tensor([2.0, 2.0, 1.0, 1.0])
    delta = pos[i] - pos[j]
    out = extrapolate_free_surface_pressure(pos, rho, pressure, cfg, i, j, w,
                                            delta, pos.shape[0])
    # (120*2 + 80*1) / (2 + 1) = 106.67 for the shell particle
    assert float(out[0]) == pytest.approx((120.0 * 2 + 80.0 * 1) / 3.0, rel=1e-6)
    # interior particles are untouched even though they receive pairs
    assert float(out[1]) == pytest.approx(120.0)
    assert float(out[2]) == pytest.approx(80.0)


def test_closure_never_lowers_pressure_and_only_raises_the_truncated_shell():
    cfg, pos, mass, rho, h = _drop()
    p0 = tait_pressure(rho, cfg)
    n = pos.shape[0]
    from tensorsph.neighbor_search import cell_linked_pairs
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    w = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)

    cfg_on = _cfg(dx=cfg.dx, free_surface_pressure=True)
    p1 = extrapolate_free_surface_pressure(pos, rho, p0, cfg_on, i, j, w, delta, n)

    assert bool((p1 >= p0 - 1e-9).all()), "closure must be monotone"
    surface = rho / cfg.rho0 < cfg.free_surface_threshold
    assert bool((p1[surface] >= p0[surface]).all())
    assert float((p1 - p0).abs().max()) > 0.0, "closure did nothing"
    assert float((p1[~surface] - p0[~surface]).abs().max()) == pytest.approx(0.0)


def test_interior_pressure_is_unchanged_on_a_complete_lattice():
    """Particles whose support is complete (rho/rho0 >= threshold) are skipped."""
    cfg, pos, _, rho, h = _drop(dx=0.003, radius=0.03)
    from tensorsph.neighbor_search import cell_linked_pairs
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    w = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    p0 = tait_pressure(rho, cfg)
    cfg_on = _cfg(dx=cfg.dx, free_surface_pressure=True)
    p1 = extrapolate_free_surface_pressure(pos, rho, p0, cfg_on, i, j, w, delta,
                                           pos.shape[0])
    core = rho / cfg.rho0 >= cfg.free_surface_threshold
    assert int(core.sum()) > 0
    assert torch.equal(p1[core], p0[core])


# --------------------------------------------------------------------------- #
# 3. the two implementations agree with the closure on
# --------------------------------------------------------------------------- #
def test_dense_and_pair_paths_agree_with_the_closure_on():
    """Both implementations must run the *same* closure, not merely compile.

    Surface tension is switched off here on purpose: ``rhs_all_pairs`` is the
    dense reference for the pressure/viscosity assembly and does not implement
    the CSF term, so comparing with it while CSF is active would test the
    reference's missing force rather than the closure.
    """
    cfg, pos, mass, _, _ = _drop(dx=0.004, radius=0.02)
    vel = torch.zeros_like(pos)
    cfg = _cfg(dx=cfg.dx, free_surface_pressure=True,
               surface_tension_coefficient=0.0)
    _, p_pair, a_pair = rhs(pos, vel, mass, cfg, dt=1e-6)
    _, p_dense, a_dense = rhs_all_pairs(pos, vel, mass, cfg, dt=1e-6)
    assert torch.allclose(p_pair, p_dense, rtol=1e-6, atol=1e-6)
    assert torch.allclose(a_pair, a_dense, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. acceptance: the four (c0, dx) points within 15 %
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def free_surface_runs():
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py first")
    runs = json.loads(REPORT.read_text()).get("free_surface_runs")
    if not runs:
        pytest.skip("run the benchmark with --free-surface / --free-surface-only")
    return runs


def test_four_point_matrix_covers_the_acceptance_grid(free_surface_runs):
    grid = {(r["dx_m"], r["sound_speed"]) for r in free_surface_runs}
    for dx in (0.002, 0.0015):
        for c0 in (20.0, 5.0):
            assert (dx, c0) in grid, f"missing acceptance point dx={dx} c0={c0}"


def test_laplace_tail_within_15pct_with_closure(free_surface_runs):
    """The P0 acceptance criterion of docs/verification_laplace_pressure_floor.md.

    ``absolute_time_average_tail`` -- absolute pressure read-out, averaged over
    the last 40 % of 1.5 capillary times -- must be within 15 % of ``sigma/R``
    at all four points. This is the criterion the closure was written against;
    it is the reason the default is worth flipping once the other benchmarks
    have been re-validated.
    """
    for run in free_surface_runs:
        if run["free_surface_iterations"] != 3:
            continue
        err = run["absolute_time_average_tail"]["rel_error_pct"]
        assert abs(err) < 15.0, (
            f"dx={run['dx_m']} c0={run['sound_speed']} iterations=3: {err:+.1f} %")


def test_closure_beats_the_clamped_default_on_every_point(free_surface_runs):
    """A/B against the same point without the closure (``runs``)."""
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py first")
    report = json.loads(REPORT.read_text())
    base = {(r["dx_m"], r["sound_speed"], r["free_surface_iterations"]
             if "free_surface_iterations" in r else 0): r for r in report["runs"]}
    for run in free_surface_runs:
        if run["free_surface_iterations"] != 3:
            continue
        ref = base.get((run["dx_m"], run["sound_speed"], 0))
        if ref is None:
            continue
        assert abs(run["absolute_time_average_tail"]["rel_error_pct"]) < abs(
            ref["absolute_time_average_tail"]["rel_error_pct"]), (
            f"closure did not help at dx={run['dx_m']} c0={run['sound_speed']}")


def test_production_iteration_count_keeps_the_spurious_current_small(
        free_surface_runs):
    """n=3 is the production value; it must stay in the quiet regime."""
    for run in free_surface_runs:
        assert run["finite"], "closure produced a non-finite state"
        for name in ("absolute", "bandA", "bandB"):
            assert math.isfinite(run[f"definition_{name}"]["delta_p_Pa"])
        if run["free_surface_iterations"] != 3:
            continue
        assert run["spurious_current"]["vmax_over_vref_final"] < 0.1, (
            f"dx={run['dx_m']} c0={run['sound_speed']}: spurious current "
            f"{run['spurious_current']['vmax_over_vref_final']:.3f}")


def test_over_iterating_the_closure_is_measurably_worse(free_surface_runs):
    """The iteration count is a *stability* parameter, not a convergence one.

    Six passes propagate the interior value past the shell's own force balance:
    the measured spurious current goes from ~0.03-0.09 (n=3) to 0.7-1.2, i.e.
    past Adami's 0.1 sanity limit. Pinned so nobody "improves" the default to a
    larger n without re-measuring.
    """
    by_n = {}
    for run in free_surface_runs:
        by_n.setdefault(run["free_surface_iterations"], []).append(
            abs(run["absolute_time_average_tail"]["rel_error_pct"]))
    if 3 not in by_n or not any(n >= 6 for n in by_n):
        pytest.skip("report does not contain both n=3 and n>=6")
    high = [v for n, vals in by_n.items() if n >= 6 for v in vals]
    assert max(high) > 15.0, "over-iteration no longer shows up in the error"