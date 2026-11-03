"""``csf_fields`` must hand every caller the same field the force is built from.

The interface-fit curvature operator (``csf_curvature.py``) needs a stencil
taken along the *arc*, not along the kernel support: the cubic-spline kernel
dies at 2h, so a shell particle in a 2h pair list only sees 3-4 lattice
neighbours and the quadratic graph fit then returns a curvature that is about
twice too large.  The pair-list widening used to live only inside
``surface_tension_acceleration``, so the production force was fine while any
*direct* ``csf_fields`` consumer (diagnostics, acceptance criteria, tests)
silently received a wildly wrong curvature -- +96 % on a circle at
dx = 1.5 mm, with no error and nothing visibly wrong in the returned state.

These tests pin the cure: ``csf_fields`` widens the pair list itself when a fit
operator is requested.  They also pin the invariance that makes the cure safe,
namely that widening an MLS (kernel-based) operator's pair list cannot change
any number, because kernel weights vanish beyond 2h.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorsph.csf_consistent import csf_fields, surface_tension_acceleration  # noqa: E402
from tensorsph.droplet_impact import Config, density_from_neighbor_pairs  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

RADIUS = 0.02
SIGMA = 7.2
RHO0 = 1000.0
DX = 0.002


def _hex_droplet():
    spec = importlib.util.spec_from_file_location(
        "_ll_law_test", ROOT / "benchmarks" / "laplace_law" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.hexagonal_droplet


@pytest.fixture(scope="module")
def drop():
    cfg = Config(dx=DX, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature="fit_quadratic")
    h = cfg.h_factor * cfg.dx
    pos, mass = _hex_droplet()(RADIUS, DX, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    return cfg, pos, mass, rho, pairs, h


def test_fit_operator_is_pair_list_invariant(drop):
    """2h and 4h pair lists must give the same field for the fit operator."""
    cfg, pos, mass, rho, pairs2h, h = drop
    pairs4h = cell_linked_pairs(pos, 4.0 * h)
    narrow = csf_fields(pos, mass, pairs2h, h, RHO0, curvature_scheme="fit_quadratic")
    wide = csf_fields(pos, mass, pairs4h, h, RHO0, curvature_scheme="fit_quadratic")
    assert torch.allclose(narrow.kappa, wide.kappa, rtol=1e-10, atol=1e-8)
    assert torch.allclose(narrow.delta_s, wide.delta_s, rtol=1e-10, atol=1e-12)
    assert torch.allclose(narrow.color, wide.color, rtol=1e-12, atol=1e-12)


def test_mls_operator_is_pair_list_invariant(drop):
    """The kernel dies at 2h, so a wider list cannot move the MLS numbers."""
    cfg, pos, mass, rho, pairs2h, h = drop
    pairs4h = cell_linked_pairs(pos, 4.0 * h)
    narrow = csf_fields(pos, mass, pairs2h, h, RHO0, curvature_scheme="mls")
    wide = csf_fields(pos, mass, pairs4h, h, RHO0, curvature_scheme="mls")
    assert torch.allclose(narrow.kappa, wide.kappa, rtol=1e-10, atol=1e-8)


def test_direct_csf_fields_matches_the_force_state(drop):
    """The state a diagnostic scores must be the state the force used."""
    cfg, pos, mass, rho, pairs2h, h = drop
    _, state_force = surface_tension_acceleration(
        pos, rho, mass, pairs2h, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic")
    state_direct = csf_fields(pos, mass, pairs2h, h, RHO0,
                              curvature_scheme="fit_quadratic")
    assert torch.allclose(state_force.kappa, state_direct.kappa, rtol=1e-10,
                          atol=1e-8)


def test_curvature_on_a_circle_is_inside_the_gate_when_called_directly(drop):
    """Regression pin for the silent 2x: the delta_s-weighted curvature of the
    circle R = 20 mm must land within 15 % of 1/R when ``csf_fields`` is called
    with the ordinary 2h pair list."""
    cfg, pos, mass, rho, pairs2h, h = drop
    state = csf_fields(pos, mass, pairs2h, h, RHO0,
                       curvature_scheme="fit_quadratic")
    w = state.delta_s
    kappa_bar = float((state.kappa.abs() * w).sum() / w.sum())
    rel = (kappa_bar - 1.0 / RADIUS) / (1.0 / RADIUS)
    assert abs(rel) < 0.15, f"delta_s-weighted curvature off by {rel * 100:+.1f}%"


def test_force_rejects_unknown_curvature_scheme(drop):
    cfg, pos, mass, rho, pairs2h, h = drop
    with pytest.raises(ValueError):
        surface_tension_acceleration(pos, rho, mass, pairs2h, h, SIGMA,
                                     rho0=RHO0, curvature_scheme="nope")