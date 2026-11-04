"""Unit gates for the free-surface pressure closure (open P0 of the Laplace work).

The closure (``Config.free_surface_pressure``) replaces the EOS pressure of a
truncated-support particle by the Adami, Ihmsen & Adams (2012, sec. 3.3)
extrapolation of its neighbourhood. These tests pin the *mechanism* -- off is
exactly the old behaviour, on lifts only the shell, the body-force term points
down the gravity vector, and non-physical thresholds are rejected. The
end-to-end effect on the Young-Laplace jump is pinned in
``tests/test_laplace_pressure_floor.py`` from the recorded benchmark numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    free_surface_pressure_closure,
    rhs,
    tait_pressure,
    validate_config,
)

DX = 0.002
RADIUS = 0.02


def _lattice_drop(dx: float = DX, radius: float = RADIUS, rho0: float = 1000.0):
    """Hexagonal disc of particles, one lattice mass each (no calibration)."""
    dy = dx * 3.0 ** 0.5 / 2.0
    rows = int(round(radius / dy)) + 1
    cols = int(round(radius / dx)) + 1
    pos = []
    for j in range(-rows, rows + 1):
        x_off = 0.5 * dx if (j % 2) else 0.0
        for i in range(-cols, cols + 1):
            x = i * dx + x_off
            y = j * dy
            if x * x + y * y <= radius * radius:
                pos.append((x, y))
    pos = torch.tensor(pos, dtype=torch.float64)
    mass = torch.full((pos.shape[0],), rho0 * dx * dy, dtype=torch.float64)
    return pos, mass


def _config(**kwargs):
    base = dict(dx=DX, h_factor=1.3, rho0=1000.0, sound_speed=20.0, gamma=7.0,
                gravity=0.0, viscosity=0.0, dt=1.0e-5,
                surface_tension_coefficient=0.0)
    base.update(kwargs)
    return Config(**base)


def test_threshold_zero_reproduces_the_eos_pressure_field():
    """The default is a no-op: the returned pressure is Tait(rho) elementwise."""
    pos, mass = _lattice_drop()
    cfg = _config()
    assert cfg.free_surface_pressure == 0.0
    rho, pressure, _ = rhs(pos, torch.zeros_like(pos), mass, cfg)
    assert torch.equal(pressure, tait_pressure(rho, cfg))


def test_closure_lifts_only_the_flagged_shell():
    """Flagged particles go from p = 0 to p > 0; the interior is bit-identical."""
    pos, mass = _lattice_drop()
    vel = torch.zeros_like(pos)
    rho, p_off, _ = rhs(pos, vel, mass, _config())
    rho_on, p_on, _ = rhs(pos, vel, mass, _config(free_surface_pressure=0.99))
    # The closure is a pressure boundary condition only: the density field is
    # untouched by it.
    assert torch.equal(rho, rho_on)

    shell = rho < 0.99 * 1000.0
    assert 0 < int(shell.sum()) < rho.numel(), "need both a shell and an interior"
    # The defect being fixed: every shell particle sits on the Tait clamp.
    assert torch.all(p_off[shell] == 0.0)
    assert torch.all(p_on[shell] > 0.0)
    # Interpolation only redefines the shell; the interior keeps the EOS value.
    assert torch.equal(p_on[~shell], p_off[~shell])


def test_body_term_points_down_the_gravity_vector():
    """Hand check on two particles: the lower one gains rho*g*dy, the upper one
    would go negative and is clamped to zero."""
    dx = DX
    pos = torch.tensor([[0.0, 0.0], [0.0, dx]], dtype=torch.float64)
    rho = torch.tensor([1000.0, 1000.0], dtype=torch.float64)
    pressure = torch.tensor([100.0, 0.0], dtype=torch.float64)
    cfg = _config(gravity=9.81, free_surface_pressure=1.05)
    pair_i = torch.tensor([0])
    pair_j = torch.tensor([1])
    wij = torch.tensor([1.0], dtype=torch.float64)

    closed, pressure_all, density_force = free_surface_pressure_closure(
        rho, pressure, pressure.clone(), rho.clone(), pos, pair_i, pair_j,
        wij, cfg)
    # Lower particle (y = 0) takes its pressure from the upper one plus the
    # hydrostatic step rho*g*dy; the upper one takes the lower pressure minus the
    # same step.
    assert closed[0].item() == 0.0 + 1000.0 * 9.81 * dx
    assert closed[1].item() == 100.0 - 1000.0 * 9.81 * dx
    assert torch.equal(pressure_all, closed)

    # If the extrapolation goes negative (fluid above a vacuum) it is clamped:
    # the closure must never manufacture tension.
    thin = torch.tensor([10.0, 0.0], dtype=torch.float64)
    clamped, _, clamped_density = free_surface_pressure_closure(
        rho, thin, thin.clone(), rho.clone(), pos, pair_i, pair_j, wij, cfg)
    assert clamped[0].item() == 0.0 + 1000.0 * 9.81 * dx
    assert clamped[1].item() == 0.0               # 10 - rho*g*dy < 0 -> clamp

    # The pressure term needs the density that belongs to the closed pressure
    # (inverted Tait law), not the truncated-support summation density.
    bulk = 1000.0 * 20.0 ** 2 / 7.0
    for k in (0, 1):
        expected = 1000.0 * (closed[k].item() / bulk + 1.0) ** (1.0 / 7.0)
        assert density_force[k].item() == pytest.approx(expected, rel=1e-12)


def test_closure_uses_the_wall_block_without_mutating_it():
    """With wall particles present the wall pressures are passed through and the
    fluid block stays fluid-sized."""
    pos, mass = _lattice_drop()
    wall = torch.tensor([[0.0, -0.002], [0.004, -0.002], [-0.004, -0.002]],
                        dtype=torch.float64)
    cfg = _config(free_surface_pressure=0.99)
    rho, pressure, acc = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
    assert rho.shape[0] == pos.shape[0]
    assert pressure.shape[0] == pos.shape[0]
    assert acc.shape == pos.shape
    assert torch.isfinite(pressure).all() and torch.isfinite(acc).all()
    assert float(pressure.min()) >= 0.0


def test_closure_changes_the_pressure_force():
    """The closed field is what the momentum equation sees (not a display-only
    post-processing step)."""
    pos, mass = _lattice_drop()
    vel = torch.zeros_like(pos)
    _, _, acc_off = rhs(pos, vel, mass, _config())
    _, _, acc_on = rhs(pos, vel, mass, _config(free_surface_pressure=0.99))
    assert not torch.allclose(acc_off, acc_on)
    assert torch.isfinite(acc_on).all()


def test_validate_config_rejects_non_physical_thresholds():
    """The value is a density fraction, so it must live in [0, 2)."""
    for bad in (-0.1, 2.0, 5.0):
        cfg = _config(free_surface_pressure=bad)
        try:
            validate_config(cfg)
        except ValueError as exc:
            assert "free_surface_pressure" in str(exc)
        else:  # pragma: no cover - the assertion is the point
            raise AssertionError(f"threshold {bad} was accepted")
    validate_config(_config(free_surface_pressure=0.99))

# --------------------------------------------------------------------------- #
# recorded benchmark gates (benchmarks/laplace_free_surface_closure/report.json)
# --------------------------------------------------------------------------- #
CLOSURE_REPORT = (Path(__file__).resolve().parents[1] / "benchmarks"
                  / "laplace_free_surface_closure" / "report.json")


@pytest.fixture(scope="module")
def closure_report():
    if not CLOSURE_REPORT.exists():
        pytest.skip("run benchmarks/laplace_free_surface_closure/run.py first")
    return json.loads(CLOSURE_REPORT.read_text())


def test_closure_improves_every_measured_case(closure_report):
    """The closure must not be a cosmetic change: every paired case improves by
    at least 10 percentage points of |error|."""
    pairs = closure_report["pairs"]
    assert pairs, "no paired cases recorded"
    for p in pairs:
        # Coarse dx with the lowest c0 already sits closer to sigma/R before the
        # closure, so the gain there is smaller; it must still be positive and
        # not a rounding artefact.
        assert p["improvement_points"] > 3.0, p


def test_residual_after_closure_is_flat_in_c0(closure_report):
    """The residual is a systematic offset, not c0-dependent noise: the spread
    across c0 at fixed dx stays well below the size of the offset itself."""
    by_dx = {}
    for p in closure_report["pairs"]:
        by_dx.setdefault(p["dx_m"], []).append(p["error_after_pct"])
    for dx, errs in by_dx.items():
        assert max(errs) - min(errs) < 12.0, (dx, errs)


def test_recorded_best_case(closure_report):
    """Pins the best recorded reading so a silent change of the closure, the
    measurement window or the EOS cannot drift unnoticed."""
    best = min(closure_report["pairs"], key=lambda p: abs(p["error_after_pct"]))
    assert abs(best["error_after_pct"]) == pytest.approx(12.3, abs=2.5), best
    assert best["dx_m"] == pytest.approx(0.003)


def test_closure_keeps_the_interior_compression_physical(closure_report):
    """The inverted-Tait density is what makes the closed field a consistent EOS
    state: after the closure the interior compression ratio must sit in a
    physical band for *every* case, including the coarse-dx case whose EOS ratio
    is non-physical (it can even be negative, i.e. below rho0)."""
    for p in closure_report["pairs"]:
        assert 0.4 < p["compression_ratio_after"] < 1.3, p
