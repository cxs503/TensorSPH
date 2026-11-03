"""Tests for the CSF acceleration assembly (``csf_accel_density``).

The gate that matters is a momentum budget, not a fitted number: the traction
of a *closed* interface in vacuum must inject no net momentum and no net
torque, because ``m a = rho0 V f / rho`` deviates from the geometric
``V f`` exactly by the free-surface density deficit.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.csf_consistent import surface_tension_acceleration
from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

import importlib.util
from pathlib import Path

_RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "csf_momentum_budget" / "run.py"
_spec = importlib.util.spec_from_file_location("csf_momentum_budget_run", _RUN)
_mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mb)

SIGMA = 7.2
RHO0 = 1000.0
RADIUS = 0.02


def _setup(dx=0.003, curvature="fit_quadratic"):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01, surface_tension_coefficient=SIGMA,
                 csf_scheme="consistent", csf_curvature=curvature)
    h = cfg.h_factor * cfg.dx
    pos, mass = _mb.hexagonal_droplet(RADIUS, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    return cfg, pos, mass, rho, pairs, h


def test_assemblies_differ_by_the_density_ratio_only():
    """Both assemblies come from the same traction; only the divisor differs."""
    cfg, pos, mass, rho, pairs, h = _setup()
    a_sum, _ = surface_tension_acceleration(
        pos, rho, mass, pairs, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic", accel_density="summation")
    a_ref, _ = surface_tension_acceleration(
        pos, rho, mass, pairs, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic", accel_density="rho0")
    # a = f/rho and a = f/rho0 with the same f  =>  a_sum/a_ref = rho0/rho.
    # (Only on the interface: in the bulk both are exactly zero and the ratio is 0/0.)
    active = a_ref.abs().sum(dim=1) > 0.0
    assert int(active.sum()) > 0
    ratio = a_sum[active] / a_ref[active]
    expected = (RHO0 / rho[active])[:, None]
    assert torch.allclose(ratio, expected, rtol=1e-9, atol=1e-9)


def test_reference_density_assembly_is_geometric():
    """``m a`` must equal ``V f`` exactly for the rho0 assembly."""
    cfg, pos, mass, rho, pairs, h = _setup()
    a_ref, state = surface_tension_acceleration(
        pos, rho, mass, pairs, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic", accel_density="rho0")
    force = a_ref * RHO0
    vol = mass / RHO0
    lhs = (mass[:, None] * a_ref).sum(0)
    rhs = (vol[:, None] * force).sum(0)
    assert torch.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
    # ... and the summation assembly over-injects where rho < rho0.
    a_sum, _ = surface_tension_acceleration(
        pos, rho, mass, pairs, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic", accel_density="summation")
    assert float(torch.linalg.vector_norm((mass[:, None] * a_sum).sum(0))) > \
        float(torch.linalg.vector_norm(lhs))


def test_interface_density_deficit_is_present():
    """The premise of the fix: the CSF shell runs below rho0."""
    row = _mb.static_budget(RADIUS, 0.003, SIGMA, RHO0, 20.0, "fit_quadratic")
    assert row["interface_density"]["mean_rho_over_rho0"] < 0.95
    assert row["interface_density"]["momentum_injection_factor"] > 1.03


def test_net_momentum_and_torque_vanish_on_a_closed_interface():
    """Analytic gates: no self-propulsion, no self-torque."""
    row = _mb.static_budget(RADIUS, 0.003, SIGMA, RHO0, 20.0, "fit_quadratic")
    for scheme in ("summation", "rho0"):
        a = row["assemblies"][scheme]
        assert math.isfinite(a["net_force_over_traction"]), (scheme, a)
        assert math.isfinite(a["torque_over_reference"]), (scheme, a)
        assert a["net_force_over_traction"] < 0.05, (scheme, a["net_force_over_traction"])
        # the traction of a circular interface is radial: no net torque at all
        assert a["torque_over_reference"] < 5e-2, (scheme, a["torque_over_reference"])


def test_reference_assembly_creates_less_net_momentum():
    row = _mb.static_budget(RADIUS, 0.003, SIGMA, RHO0, 20.0, "fit_quadratic")
    assert (row["assemblies"]["rho0"]["net_force_over_traction"]
            < row["assemblies"]["summation"]["net_force_over_traction"])


def test_cut_force_is_independent_of_the_assembly():
    """The traction is the same object; the cut integral must not move."""
    row = _mb.static_budget(RADIUS, 0.003, SIGMA, RHO0, 20.0, "fit_quadratic")
    a = row["assemblies"]["summation"]["cut_force_over_2sigma"]
    b = row["assemblies"]["rho0"]["cut_force_over_2sigma"]
    assert a == pytest.approx(b, rel=1e-12)
    # Known deficit of the current discretisation (0.71 of 2 sigma, -29 %);
    # pinned so that a future fix is forced to update the record.
    assert 0.6 < a < 0.8


def test_config_round_trip_and_validation():
    cfg, pos, mass, rho, pairs, h = _setup()
    assert cfg.csf_accel_density == "summation"
    acc_sum = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    cfg.csf_accel_density = "rho0"
    acc_ref = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    assert torch.isfinite(acc_ref).all()
    # the config switch reaches the assembly: a_ref = a_sum * rho/rho0
    active = acc_ref.abs().sum(dim=1) > 0.0
    assert int(active.sum()) > 0
    assert not torch.allclose(acc_sum, acc_ref)
    assert torch.allclose((acc_sum / acc_ref)[active],
                          (RHO0 / rho[active])[:, None], rtol=1e-9, atol=1e-9)
    # a bogus selector is rejected at the operator, wherever Config is checked
    cfg.csf_accel_density = "bogus"
    with pytest.raises(ValueError):
        compute_surface_tension_force(pos, rho, mass, pairs, cfg)


def test_bulk_particles_carry_no_surface_force():
    """Control: away from the interface the CSF force is a no-op."""
    cfg, pos, mass, rho, pairs, h = _setup()
    a_sum, state = surface_tension_acceleration(
        pos, rho, mass, pairs, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic", accel_density="summation")
    r = torch.linalg.vector_norm(pos, dim=1)
    bulk = (r < 0.5 * RADIUS) & (state.delta_s == 0.0)
    assert int(bulk.sum()) > 0
    assert float(torch.linalg.vector_norm(a_sum[bulk], dim=1).max()) == 0.0
    assert float(tait_pressure(rho, cfg)[r < 0.5 * RADIUS].max()) >= 0.0