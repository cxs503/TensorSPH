"""Unit and record tests for the free-surface pressure closure (P0 of m2971).

Two halves:

``unit``    the closure itself, exercised on the benchmark drop with and without
            ``Config.free_surface_pressure``.  These run in seconds and do not
            need ``report_shell.json``.

``record``  the acceptance criterion ``|tail-averaged Delta p error| < 15 %`` at
            the four points of ``benchmarks/laplace_pressure_floor/report_shell.json``
            (c0 = 20 / 5 m/s x dx = 0.002 / 0.0015 m), plus the paired comparison
            against the un-closed baseline kept in ``report.json``.  The
            companion test
            ``test_laplace_pressure_floor.py::test_laplace_jump_within_15pct``
            stays red-by-design for the un-closed model: the two records
            document the same drop before and after the closure.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config, rhs
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
REPORT_SHELL = BENCH / "report_shell.json"


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "laplace_pressure_floor_run_closure", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lpf = _load_run_module()


@pytest.fixture(scope="module")
def report_shell():
    if not REPORT_SHELL.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run_closure.py first")
    return json.loads(REPORT_SHELL.read_text())


@pytest.fixture(scope="module")
def lattice():
    """The raw hexagonal drop: mass = rho0 dx dy, so the interior carries the
    lattice bias +5.9e-3 (p ~ 2.4 kPa at c0 = 20) while the outer shell is
    deficient and clamped to p = 0.  A calibrated drop would have p = 0
    everywhere at t = 0 and could not separate the two closures."""
    dx = 0.002
    pos, mass = lpf.hexagonal_droplet(lpf.RADIUS, dx, lpf.RHO0)
    h = lpf.H_FACTOR * dx
    return dict(pos=pos, mass=mass, h=h, pairs=cell_linked_pairs(pos, 2.0 * h),
                dx=dx)


@pytest.fixture(scope="module")
def drop(lattice):
    """Pressure of the same lattice with each closure."""
    dx = lattice["dx"]
    pos, mass, pairs = lattice["pos"], lattice["mass"], lattice["pairs"]
    common = dict(dx=dx, h_factor=lpf.H_FACTOR, rho0=lpf.RHO0, sound_speed=20.0,
                  gamma=lpf.GAMMA, gravity=0.0, viscosity=lpf.VISCOSITY,
                  surface_tension_coefficient=lpf.SIGMA,
                  csf_scheme="consistent")
    cfg_off = Config(**common)
    cfg_on = Config(**common, free_surface_pressure=True)
    vel = torch.zeros_like(pos)
    rho_off, p_off, _ = rhs(pos, vel, mass, cfg_off, pairs=pairs)
    rho_on, p_on, _ = rhs(pos, vel, mass, cfg_on, pairs=pairs)
    return dict(pos=pos, rho_off=rho_off, rho_on=rho_on, p_off=p_off, p_on=p_on,
                pairs=pairs, cfg_off=cfg_off, cfg_on=cfg_on, mass=mass, vel=vel)


# --------------------------------------------------------------------------- #
# unit: the closure is opt-in and touches exactly the deficient particles
# --------------------------------------------------------------------------- #
def test_closure_is_off_by_default():
    """Every published free-surface record stays bit-reproducible until re-run."""
    assert Config().free_surface_pressure is False
    assert 0.0 < Config().free_surface_density_ratio < 1.0


def test_closed_pressure_is_zero_on_the_unclosed_surface(drop):
    """The defect the closure targets: the clamped Tait law gives p = 0 there."""
    rel = drop["rho_off"] / drop["cfg_off"].rho0
    deficit = rel < drop["cfg_off"].free_surface_density_ratio
    assert bool(deficit.any()), "the drop must have a deficient surface layer"
    assert float(drop["p_off"][deficit].abs().max()) == 0.0
    # ... while the complete-kernel interior does carry a pressure
    assert float(drop["p_off"][~deficit].max()) > 0.0


def test_closure_raises_the_pressure_on_the_surface_layer(drop):
    rel = drop["rho_off"] / drop["cfg_off"].rho0
    deficit = rel < drop["cfg_off"].free_surface_density_ratio
    closed = drop["p_on"][deficit]
    assert float(closed.min()) >= 0.0
    assert float(closed.mean()) > 0.0
    assert float(closed.max()) > 0.0


def test_closure_leaves_complete_kernel_particles_bitwise_unchanged(drop):
    """Only deficient particles are rewritten; the interior is untouched."""
    rel = drop["rho_on"] / drop["cfg_on"].rho0
    interior = ~(rel < drop["cfg_on"].free_surface_density_ratio)
    assert bool(interior.any())
    assert torch.equal(drop["p_on"][interior], drop["p_off"][interior])


def test_closure_does_not_change_the_density_or_the_clamp(drop):
    """The closure is a pressure-assembly change only: rho is the same."""
    assert torch.equal(drop["rho_on"], drop["rho_off"])
    assert torch.equal(drop["p_on"].clamp(max=0.0),
                       torch.zeros_like(drop["p_on"]))


def test_closed_pressure_is_bounded_by_its_donors(drop):
    """A kernel average cannot exceed the largest donor source; no blow-up."""
    rel = drop["rho_on"] / drop["cfg_on"].rho0
    deficit = rel < drop["cfg_on"].free_surface_density_ratio
    interior = ~deficit
    donor_max = float(drop["p_on"][interior].max())
    assert float(drop["p_on"][deficit].max()) <= donor_max * (1.0 + 1e-6)


def test_closure_is_deterministic(drop):
    _, p_a, _ = rhs(drop["pos"], drop["vel"], drop["mass"], drop["cfg_on"],
                    pairs=drop["pairs"])
    assert torch.equal(p_a, drop["p_on"])


def test_closure_body_force_term_is_active_and_finite(lattice):
    """g.(r_j - r_i) enters eq. (11); with gravity on the closed pressure must
    differ from the gravity-free closure and stay finite and non-negative."""
    pos, mass, pairs = lattice["pos"], lattice["mass"], lattice["pairs"]
    dx = lattice["dx"]
    common = dict(dx=dx, h_factor=lpf.H_FACTOR, rho0=lpf.RHO0, sound_speed=5.0,
                  gamma=lpf.GAMMA, viscosity=lpf.VISCOSITY,
                  surface_tension_coefficient=lpf.SIGMA, csf_scheme="consistent",
                  free_surface_pressure=True)
    vel = torch.zeros_like(pos)
    _, p_no_g, acc_no_g = rhs(pos, vel, mass, Config(gravity=0.0, **common),
                              pairs=pairs)
    _, p_g, acc_g = rhs(pos, vel, mass, Config(gravity=9.81, **common),
                        pairs=pairs)
    for p in (p_no_g, p_g):
        assert bool(torch.isfinite(p).all())
        assert float(p.min()) >= 0.0
    assert bool(torch.isfinite(acc_g).all())
    assert not torch.equal(p_no_g, p_g)


# --------------------------------------------------------------------------- #
# record: the acceptance criterion
# --------------------------------------------------------------------------- #
def test_shell_report_records_the_acceptance_points(report_shell):
    assert report_shell["pressure_closure"] == "shell"
    points = {(run["dx_m"], run["sound_speed"]) for run in report_shell["runs"]}
    assert points == {(0.002, 20.0), (0.002, 5.0), (0.0015, 20.0), (0.0015, 5.0)}


def test_shell_runs_are_finite_and_same_stepping_as_the_baseline(report_shell):
    for run in report_shell["runs"]:
        assert run["finite"], run
        base = report_shell["baseline"][f"{run['dx_m']:g}/{run['sound_speed']:g}"]
        assert run["steps"] == base["steps"], "paired comparison must share dt"
        assert math.isfinite(run["absolute_time_average_tail"]["rel_error_pct"])


def test_laplace_jump_within_15pct_with_the_shell_closure(report_shell):
    """ACCEPTANCE (m2971 P0): every recorded case inside 15 %."""
    for run in report_shell["runs"]:
        err = run["absolute_time_average_tail"]["rel_error_pct"]
        assert abs(err) < 15.0, (run["dx_m"], run["sound_speed"], err)


def test_shell_closure_beats_the_clamp_at_every_point(report_shell):
    """Paired: the same run with the closure is closer to sigma/R everywhere."""
    for key, row in report_shell["comparison"].items():
        assert row["tait_tail_rel_error_pct"] is not None, key
        assert row["improvement_pt"] > 20.0, (key, row)
        assert abs(row["shell_tail_rel_error_pct"]) < abs(row["tait_tail_rel_error_pct"]), key


def test_shell_closure_improves_but_does_not_meet_3pct_everywhere(report_shell):
    """Honest record: 15 % is met at all four points, 3 % only at some of them.
    Do not overstate the result as a 3 %-level validation."""
    rows = report_shell["comparison"]
    assert all(row["passes_15pct"] for row in rows.values())
    within_3 = [k for k, row in rows.items() if row["passes_3pct"]]
    assert 0 < len(within_3) < len(rows), within_3
    # the coarse, high-c0 point is the worst: it is the one that must not be
    # reported as a 3 % case
    assert not rows["0.002/20"]["passes_3pct"]


def test_shell_closure_raises_the_spurious_current_and_that_is_recorded(report_shell):
    """Known cost: the closure injects a surface pressure that does work, so the
    residual velocity grows (0.03 -> 0.14 of the capillary reference). The
    Adami 0.1 criterion is therefore NOT met; the number is pinned here so a
    future improvement cannot silently regress it."""
    ratios = [run["spurious_current"]["vmax_over_vref_final"]
              for run in report_shell["runs"]]
    assert all(r < 0.5 for r in ratios), ratios
    assert any(r > 0.1 for r in ratios), ratios
    assert report_shell["reference"]["delta_p_Pa"] == pytest.approx(360.0)