"""Gates for the conservation / invariance / integration-order audit.

Committed-report checks read ``benchmarks/conservation/report.json``; the live
check re-measures the strongest invariant (pair-wise cancellation of the
pressure gradient) on a small configuration so the claim cannot rot behind a
stale artefact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config, rhs
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "conservation"
REPORT = BENCH / "report.json"
ROUNDOFF = 1.0e-12


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text())


def _hex_droplet(radius: float, dx: float, rho0: float):
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), mass, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# G1 — the pressure gradient cancels pairwise
# --------------------------------------------------------------------------- #
def test_pressure_gradient_conserves_momentum_live():
    """One step, no viscosity, no wall: the net force must be roundoff."""
    cfg = Config(dx=0.004, h_factor=1.3, rho0=1000.0, sound_speed=20.0,
                 gamma=7.0, gravity=0.0, viscosity=0.0)
    pos, mass = _hex_droplet(0.03, cfg.dx, cfg.rho0)
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx)
    _, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=cfg.dt)
    force = mass[:, None] * acc
    leak = float(force.sum(dim=0).norm() / force.abs().sum())
    assert leak < ROUNDOFF, f"net force does not vanish: {leak:.3e}"


def test_pressure_gradient_leak_is_roundoff_in_report(report):
    rep = report["momentum"]["pressure_only"]
    assert rep["leak_median"] < ROUNDOFF
    assert rep["momentum_drift_rel"] < ROUNDOFF
    assert rep["leak_nan_steps"] == 0


# --------------------------------------------------------------------------- #
# G1b — the viscous term's momentum leak has a closed form
# --------------------------------------------------------------------------- #
def test_viscous_pair_leak_matches_closed_form(report):
    pair = report["viscous_pair"]
    ratio = pair["closed_form_ratio"]
    assert ratio is not None
    # Measured net pair force equals C nu m_i m_j (v_i-v_j) L (1/rho_i-1/rho_j).
    assert ratio == pytest.approx(1.0, abs=1.0e-9)
    assert pair["relative_density_contrast"] > 0.05
    # ... and the same pair with equal densities leaks exactly nothing.
    assert pair["equal_density_net_pair_force_N"] < 1.0e-25


def test_viscous_pair_leak_is_live():
    """Cheap live repeat of the closed form with an independent state."""
    from tensorsph.droplet_impact import VISCOUS_LAPLACIAN_FACTOR, pair_kernels

    dx, rho0, nu = 0.004, 1000.0, 0.01
    h = 1.3 * dx
    pos = torch.tensor([[0.0, 0.0], [dx, 0.0]], dtype=torch.float64)
    mass = torch.tensor([3.0, 1.5], dtype=torch.float64) * rho0 * dx * dx
    vel = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    cfg_nu = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=20.0, gamma=7.0,
                    gravity=0.0, viscosity=nu)
    cfg_0 = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=20.0, gamma=7.0,
                   gravity=0.0, viscosity=0.0)
    rho, _, a_nu = rhs(pos, vel, mass, cfg_nu, pairs=pairs, wall=None)
    _, _, a_0 = rhs(pos, vel, mass, cfg_0, pairs=pairs, wall=None)
    net = (mass[:, None] * (a_nu - a_0)).sum(dim=0)
    delta = pos[0] - pos[1]
    dist = torch.linalg.vector_norm(delta)
    _, grad_w = pair_kernels(delta.unsqueeze(0), h)
    lap = torch.relu(-(delta * grad_w[0]).sum()) / (dist * dist + 0.01 * h * h)
    predicted = (VISCOUS_LAPLACIAN_FACTOR * nu * mass[0] * mass[1]
                 * (vel[0] - vel[1]) * lap * (1.0 / rho[0] - 1.0 / rho[1]))
    assert float(predicted.norm()) > 0.0
    assert float(net.norm() / predicted.norm()) == pytest.approx(1.0, abs=1e-9)


def test_nonconserving_operators_are_recorded(report):
    """The audit must not claim exact conservation: both leaks are in the record."""
    mom = report["momentum"]
    assert mom["viscous"]["leak_median"] > 1.0e-6
    assert mom["csf_only"]["st_leak_median"] > 1.0e-4
    assert mom["csf_only"]["momentum_drift_rel"] > 1.0e-6
    assert mom["default"]["momentum_drift_rel"] > 1.0e-6


# --------------------------------------------------------------------------- #
# G3 — Galilean invariance
# --------------------------------------------------------------------------- #
def test_galilean_invariance(report):
    galf = report["galilean_field"]
    assert galf["velocity_field_mismatch_rel"] < 1.0e-9
    com = report["galilean"]["com_drift_m"]
    assert abs(com[0]) < 1.0e-6
    assert abs(com[1]) < 1.0e-6


# --------------------------------------------------------------------------- #
# G4 — closed-form first-order trajectory offset
# --------------------------------------------------------------------------- #
def test_ballistic_lag_matches_closed_form(report):
    bal = report["ballistic"]
    rows = bal["rows"]
    assert len(rows) >= 3
    for row in rows:
        assert row["finite"]
        assert row["lag_measured_m"] > 0.0
        # lag = (1/2) g dt t, derived from semi-implicit Euler: no free parameter
        assert row["lag_ratio"] == pytest.approx(1.0, abs=0.03), row
        # ... and it is a fraction dt/T of the fall distance, by construction
        assert row["lag_over_fall_distance"] == pytest.approx(
            row["dt_over_t"], rel=1e-6, abs=1e-9)


def test_observed_order_is_first_order(report):
    order = report["ballistic"]["observed_order"]
    assert order is not None
    assert 0.9 <= order <= 1.1, f"observed order {order:.3f} is not 1"


def test_verdicts_are_internally_consistent(report):
    v = report["verdicts"]
    mom = report["momentum"]
    assert v["mass_conservation_exact"] is True
    assert v["G1_pressure_gradient_symmetric"] is True
    assert v["G4_observed_order"] == pytest.approx(
        report["ballistic"]["observed_order"])
    assert v["G1b_viscous_pair_closed_form_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert v["G2_viscous_leak_median"] == pytest.approx(
        mom["viscous"]["leak_median"])
    # the module-default configuration is worse than pressure-only by orders
    assert mom["default"]["leak_median"] > 1.0e6 * mom["pressure_only"]["leak_median"]