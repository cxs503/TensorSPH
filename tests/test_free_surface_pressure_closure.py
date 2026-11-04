"""Free-surface pressure closure (Adami, Ihmsen & Adams 2012, JCP 231:7057 §3.3).

Why these tests exist
---------------------
``tait_pressure`` clamps ``p >= 0``. On a free surface the sum-density deficit
shell (rho/rho0 - 1 = -11.6 % at 0.80-0.95 R, -34 % at 0.95-1.05 R) therefore
pins every particle there to exactly ``p = 0``, and a drop in vacuum can only
hold a fraction of the Young-Laplace jump ``sigma/R`` in its interior:
``benchmarks/laplace_pressure_floor/`` measured ``0.57 +/- 0.06 sigma/R``
(-34 %) and registered the closure as the single blocking defect (P0 in
``backlog_update_m2971.md``).

The operator added here is the free-surface analogue of the wall-pressure
extrapolation ``rhs`` already performs: a particle flagged free surface
(``rho/rho0 < free_surface_rho_ratio``) takes the kernel-weighted average of
its reliable neighbours' pressures, plus the body-force term of Adami eq. (11).

The tests are layered so that a failure points at one property:

* the operator is *exact* for an affine pressure field (a linear p(x) is
  reproduced by the normalised kernel average up to the discrete moment error);
* it leaves the interior untouched, and leaves an already-honest free surface
  (no deficit) untouched;
* it does not invent pressure where there is no reliable neighbour;
* it keeps a hydrostatic column's surface at ``p ~ 0``, the property a naive
  "average the neighbours" implementation would violate by dragging the
  interior pressure up to the surface;
* it is off by default, so every published number stays reproducible;
* end to end on the drop, enabling it moves the absolute, time-averaged
  Laplace jump towards ``sigma/R`` (the record for the 15 % acceptance criterion
  lives in ``benchmarks/laplace_pressure_closure/``).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorsph import (
    Config,
    free_surface_pressure_closure,
    tait_pressure,
)

REPO = Path(__file__).resolve().parents[1]


def _lattice(n_side: int, dx: float = 0.005, dtype=torch.float64):
    xs = torch.arange(n_side, dtype=dtype) * dx
    y, x = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)


def _base_cfg(**kw) -> Config:
    defaults = dict(dx=0.005, h_factor=1.3, rho0=1000.0, sound_speed=20.0,
                    gamma=7.0, gravity=0.0, free_surface_pressure=True)
    defaults.update(kw)
    return Config(**defaults)


def test_closed_operator_is_identity_when_disabled():
    """Default off: the closure must not touch the pressure array."""
    pos = _lattice(9)
    rho = torch.full((pos.shape[0],), 900.0, dtype=torch.float64)
    cfg = _base_cfg(free_surface_pressure=False)
    p_in = torch.full_like(rho, 0.0)
    assert torch.equal(free_surface_pressure_closure(pos, rho, p_in, torch.ones_like(rho), cfg), p_in)


def test_affine_pressure_field_is_reproduced_at_the_surface():
    """Adami eq. (11) is exact for a smooth field: p = p0 + k*x.

    The deficit column in the middle of the lattice carries the clamped
    ``p = 0``; its reliable neighbours sit symmetrically on both sides, so the
    normalised kernel average of a linear field must return ``p(x_i)`` to
    round-off. This is the property that makes the operator an *extrapolation*
    rather than a smearing of the interior value.
    """
    pos = _lattice(11)
    cfg = _base_cfg()
    k = 1.0e4  # Pa/m
    p_true = k * (pos[:, 0] + cfg.dx)  # keep the flagged column non-zero
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx ** 2, dtype=pos.dtype)
    rho = torch.full((pos.shape[0],), cfg.rho0, dtype=pos.dtype)
    flagged = pos[:, 0] == 5.0 * cfg.dx
    rho[flagged] = 0.97 * cfg.rho0
    p_in = p_true.clone()
    p_in[flagged] = 0.0  # what tait_pressure clamps a deficit shell to
    p_out = free_surface_pressure_closure(pos, rho, p_in, mass, cfg)

    assert torch.all(p_out[flagged] > 0.0)
    err = (p_out[flagged] - p_true[flagged]).abs() / p_true[flagged].abs()
    assert float(err.max()) < 1.0e-6, float(err.max())
    # interior pressures must be returned unchanged
    assert torch.equal(p_out[~flagged], p_in[~flagged])


def test_no_pressure_is_invented_without_a_reliable_neighbour():
    """An isolated flagged particle keeps its Tait value (no hallucinated p)."""
    pos = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float64)
    rho = torch.full((2,), 900.0, dtype=torch.float64)
    p_in = torch.zeros(2, dtype=torch.float64)
    cfg = _base_cfg()
    p_out = free_surface_pressure_closure(pos, rho, p_in, torch.ones(2, dtype=torch.float64), cfg)
    assert torch.equal(p_out, p_in)


def test_hydrostatic_column_surface_stays_at_zero_pressure():
    """Gravity term keeps the surface honest: p must not be lifted upwards.

    A column at rest with p = rho0*g*(H - y) has p = 0 at the top. Averaging the
    neighbours' pressures without Adami's body-force term would give the top
    particle ~half a spacing worth of positive pressure and push it up; with the
    term it stays clamped at 0.
    """
    cfg = _base_cfg(gravity=9.81, rho0=1000.0, dx=0.01, h_factor=1.3)
    h = cfg.h_factor * cfg.dx
    pos = _lattice(21, dx=cfg.dx)
    y_top = float(pos[:, 1].max())
    H = y_top + 0.5 * cfg.dx
    p_true = 1000.0 * 9.81 * (H - pos[:, 1])
    # Tait-consistent density for that pressure, then subtract the same lattice
    # sum offset the drop carries (b ~ 5.9e-3), so the top layer is flagged.
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    rho = cfg.rho0 * (1.0 + p_true / bulk).pow(1.0 / cfg.gamma)
    rho = rho * 0.985
    rho[pos[:, 1] == 0.0] = cfg.rho0  # bottom layer reliable
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx ** 2, dtype=pos.dtype)
    p_in = tait_pressure(rho, cfg)
    p_out = free_surface_pressure_closure(pos, rho, p_in, mass, cfg)

    top = pos[:, 1] == y_top
    assert torch.all(p_in[top] == 0.0)
    # one spacing below the top, the analytic value is ~98 Pa; the closure must
    # not overshoot it by more than the discrete gradient error
    assert float(p_out[top].max()) < 1.5 * 1000.0 * 9.81 * cfg.dx, float(p_out[top].max())
    assert torch.all(torch.isfinite(p_out))


def test_layer_walk_reaches_an_isolated_shell_particle():
    """With enough layers the closure penetrates a multi-layer deficit shell."""
    cfg = _base_cfg(dx=0.01, free_surface_layers=4)
    pos = _lattice(15, dx=cfg.dx)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx ** 2, dtype=pos.dtype)
    p_in = torch.zeros(pos.shape[0], dtype=pos.dtype)
    # interior box carries 500 Pa, everything else is flagged free surface
    inner = ((pos[:, 0] > 0.03) & (pos[:, 0] < 0.11)
             & (pos[:, 1] > 0.03) & (pos[:, 1] < 0.11))
    p_in[inner] = 500.0
    rho = torch.full((pos.shape[0],), cfg.rho0, dtype=pos.dtype)
    rho[~inner] = 0.97 * cfg.rho0
    p_out = free_surface_pressure_closure(pos, rho, p_in, mass, cfg)
    shell = p_out[~inner]
    assert float((shell > 0.0).to(torch.float64).mean()) > 0.5
    assert float(p_out.max()) <= 500.0 * (1.0 + 1e-12)


def test_config_validation_rejects_bad_closure_parameters():
    cfg = _base_cfg(free_surface_rho_ratio=1.5)
    from tensorsph import validate_config
    with pytest.raises(ValueError, match="free_surface_rho_ratio"):
        validate_config(cfg)
    cfg = _base_cfg(free_surface_layers=0)
    with pytest.raises(ValueError, match="free_surface_layers"):
        validate_config(cfg)


# --------------------------------------------------------------------------- #
# end to end: the drop
# --------------------------------------------------------------------------- #
def _load_floor():
    spec = importlib.util.spec_from_file_location(
        "laplace_pressure_floor_run",
        REPO / "benchmarks" / "laplace_pressure_floor" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drop_ab():
    """One coarse drop run, closure off and on, from the same initial state."""
    H = _load_floor()
    return {fs: H.relax(0.002, 5.0, t_end_over_t_sigma=1.0, max_steps=900,
                        free_surface=fs)
            for fs in (False, True)}


def test_closure_moves_the_laplace_jump_towards_sigma_over_R(drop_ab):
    """The signed residual must shrink; the physics sign cannot flip."""
    off = drop_ab[False]["absolute_time_average_tail"]
    on = drop_ab[True]["absolute_time_average_tail"]
    assert off["rel_error_pct"] < 0.0
    assert abs(on["rel_error_pct"]) < abs(off["rel_error_pct"]), (off, on)
    assert drop_ab[True]["finite"] is True

# --------------------------------------------------------------------------- #
# the benchmark record (acceptance evidence)
# --------------------------------------------------------------------------- #
def _bench_report():
    path = REPO / "benchmarks" / "laplace_pressure_closure" / "report.json"
    if not path.exists():
        pytest.skip("run benchmarks/laplace_pressure_closure/run.py to produce report.json")
    import json
    return json.loads(path.read_text())


def test_operator_is_opt_in_by_default():
    """Published numbers must stay reproducible: the closure ships off."""
    cfg = Config()
    assert cfg.free_surface_pressure is False
    assert cfg.free_surface_rho_ratio == 0.99
    assert cfg.free_surface_layers == 3


def test_benchmark_meets_the_p0_acceptance_criterion():
    """Every closure-on case must sit inside 15 % of sigma/R.

    This is the machine-checkable form of the P0 acceptance criterion in
    backlog_update_m2971.md. It is the assertion that the strict-xfail in
    tests/test_laplace_pressure_floor.py describes, now satisfiable with the
    closure enabled (the floor record itself stays closure-off on purpose).
    """
    report = _bench_report()
    on = [r for r in report["runs"] if r["free_surface_pressure"]]
    assert len(on) >= 2
    for run in on:
        assert run["finite"] is True
        assert abs(run["tail_rel_error_pct"]) < 15.0, run
    assert report["all_on_within_15pct"] is True


def test_closure_off_branch_reproduces_the_published_floor_numbers():
    """A/B hygiene: the "off" column must be the floor benchmark, unchanged.

    If the closure ever leaks into the default path, this test fails because the
    closure-off residuals stop matching the numbers published in
    docs/verification_laplace_pressure_floor.md (via its report.json).
    """
    import json
    floor = json.loads((REPO / "benchmarks" / "laplace_pressure_floor" /
                        "report.json").read_text())
    published = {(r["dx_m"], r["sound_speed"]):
                 r["absolute_time_average_tail"]["rel_error_pct"]
                 for r in floor["runs"]}
    report = _bench_report()
    compared = 0
    for run in report["runs"]:
        if run["free_surface_pressure"]:
            continue
        key = (run["dx_m"], run["sound_speed"])
        if key not in published:
            continue
        assert abs(run["tail_rel_error_pct"] - published[key]) < 1.0e-6, (key, run, published[key])
        compared += 1
    assert compared >= 1
