"""Quantitative verification of the contact-angle boundary condition.

``tests/test_contact_angle.py`` checks shapes, the "far particles untouched"
property and magnitude preservation on hand-made particle sets.  It never
measures the angle that actually comes out, and its two integration tests use a
six-particle cluster whose CSF curvature is identically zero, so they cannot
distinguish a working boundary condition from a silent no-op.

These tests score the identity the boundary condition is supposed to realise,

    n_grad . n_wall = cos(theta)          (n_grad = grad(c)/|grad(c)|)

on a hex-packed droplet resting on a flat wall lattice, for
theta = 30 .. 150 deg, and they pin the two properties that a rotation must
have: |grad(c)| unchanged and theta = 90 deg neutral (no wall-normal part).
Numbers come from ``benchmarks/contact_angle/run.py``, which writes the same
quantities to ``benchmarks/contact_angle/results.json`` for the report.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    compute_surface_tension_force,
    csf_colour_field,
)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "contact_angle" / "run.py"
THETAS_DEG = (30.0, 60.0, 90.0, 120.0, 150.0)


def _bench_module():
    spec = importlib.util.spec_from_file_location("contact_angle_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def geometry():
    """Hex drop (R = 15 mm, dx = 2.5 mm) resting on a two-row wall lattice."""
    bench = _bench_module()
    radius, dx, rho0 = 0.015, 0.0025, 1000.0
    cfg = Config(dx=dx, rho0=rho0, gravity=0.0, sound_speed=20.0,
                 surface_tension_coefficient=0.072)
    h = cfg.h_factor * dx
    drop, mass = bench.hexagonal_drop(radius, dx, rho0, center=(0.0, radius + 0.5 * dx))
    wall = bench.wall_strip(dx, half_width=1.5 * radius)
    density = torch.full((drop.shape[0],), rho0, dtype=torch.float64)
    pairs = cell_linked_pairs(drop, 2.0 * h)
    return dict(bench=bench, cfg=cfg, h=h, pos=drop, mass=mass, density=density,
                pairs=pairs, wall=wall, radius=radius, dx=dx)


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_realised_normal_satisfies_contact_angle_identity(geometry, theta_deg):
    """The enforced gradient normal must realise n . n_wall = cos(theta)."""
    g = geometry
    row = g["bench"].measure(g["pos"], g["mass"], g["density"], g["pairs"],
                             g["wall"], g["h"], theta_deg)
    assert row["n_particles_scored"] >= 3, "degenerate set-up: no near-wall interface"
    assert row["cos_grad_max_abs_residual"] < 1.0e-6, row


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_rotation_preserves_gradient_magnitude(geometry, theta_deg):
    """Enforcing the angle must rotate the gradient, never rescale it."""
    g = geometry
    row = g["bench"].measure(g["pos"], g["mass"], g["density"], g["pairs"],
                             g["wall"], g["h"], theta_deg)
    assert row["grad_magnitude_rel_change_max"] < 1.0e-12, row


def test_neutral_angle_removes_the_wall_normal_component(geometry):
    """theta = 90 deg is the neutral case: cos(n, n_wall) = 0."""
    g = geometry
    row = g["bench"].measure(g["pos"], g["mass"], g["density"], g["pairs"],
                             g["wall"], g["h"], 90.0)
    assert abs(row["cos_grad_after_median"]) < 1.0e-8, row


def test_boundary_condition_actually_changes_the_field(geometry):
    """Without the BC the drop's bottom interface is parallel to the wall."""
    g = geometry
    row = g["bench"].measure(g["pos"], g["mass"], g["density"], g["pairs"],
                             g["wall"], g["h"], 60.0)
    # grad(c) points into the fluid, i.e. along +n_wall at the flat bottom
    assert row["cos_grad_before_median"] > 0.9, row
    assert abs(row["cos_grad_after_median"] - 0.5) < 1.0e-6, row


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_force_stays_finite_and_nonzero(geometry, theta_deg):
    """The CSF force with the BC must be finite and non-trivial."""
    g = geometry
    cfg = Config(dx=g["dx"], rho0=1000.0, gravity=0.0, sound_speed=20.0,
                 surface_tension_coefficient=0.072,
                 contact_angle=math.radians(theta_deg))
    acc = compute_surface_tension_force(g["pos"], g["density"], g["mass"],
                                        g["pairs"], cfg, wall=g["wall"])
    assert torch.isfinite(acc).all()
    assert float(torch.linalg.vector_norm(acc, dim=1).max()) > 0.0


def test_contact_angle_none_is_bit_identical_to_no_wall(geometry):
    """contact_angle=None must keep the pre-existing behaviour exactly."""
    g = geometry
    cfg_none = Config(dx=g["dx"], rho0=1000.0, gravity=0.0, sound_speed=20.0,
                      surface_tension_coefficient=0.072, contact_angle=None)
    cfg_wall = Config(dx=g["dx"], rho0=1000.0, gravity=0.0, sound_speed=20.0,
                      surface_tension_coefficient=0.072,
                      contact_angle=math.pi / 3)
    a = compute_surface_tension_force(g["pos"], g["density"], g["mass"],
                                      g["pairs"], cfg_none, wall=g["wall"])
    b = compute_surface_tension_force(g["pos"], g["density"], g["mass"],
                                      g["pairs"], cfg_none, wall=None)
    c = compute_surface_tension_force(g["pos"], g["density"], g["mass"],
                                      g["pairs"], cfg_wall, wall=g["wall"])
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_committed_verdict_is_self_consistent():
    """The archived benchmark numbers must agree with the identity check."""
    import json

    path = ROOT / "benchmarks" / "contact_angle" / "results.json"
    rep = json.loads(path.read_text())
    assert rep["setup"]["theory"].startswith("n . n_wall = cos(theta)")
    verdict = rep["verdict"]
    assert verdict["within_1e-6_cos"] is True
    assert verdict["magnitude_preserved_to_1e-6"] is True
    for row in rep["angle_identity"]:
        assert row["cos_grad_max_abs_residual"] < 1.0e-6, row