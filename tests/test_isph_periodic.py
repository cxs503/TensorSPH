"""Invariants of the periodic ISPH operators and of the Taylor-Green measurement.

Context
-------
``XVAL-SPH-METHOD-SELECTION-2026-11-12`` still lists "XVAL-SPH-PROJECTION-
TAYLOR-GREEN" as residual debt: the method-selection registry has a projection
(ISPH) record from a *wall-bounded channel* only, so a smooth inviscid vortex
profile comes back ``WITHHELD``.  ``benchmarks/isph_taylor_green/run.py`` now
measures that missing number, and the measurement is **negative** (23.6 % error
at ``dx = 0.05``, ``Co = 0.1`` against the 15 % owner standard; unstable at
``dx = 0.04``).

These tests pin what *is* solid, so the negative result cannot be blamed on the
operators, and pin the selection consequence, so a future "fix" cannot quietly
turn the refusal into a SELECTED route without new evidence:

* the periodic operator triple is consistent on a lattice (divergence of the
  exact Taylor-Green field is zero to round-off, gradient/Laplacian reproduce
  the analytic derivatives to a few percent);
* the projection really is a projection (it reduces the discrete divergence);
* viscosity dissipates energy (sign, not just "a" change);
* the strictly-adjoint pair ``D D^T`` of ``projection_adjoint`` has a
  *checkerboard* null mode on a collocated lattice -- the reason this module
  uses an M-matrix (graph) Laplacian instead of the adjoint composition;
* no vortex-class record has been added to the registry, i.e. the selector still
  refuses the Gresho/Taylor-Green class.

All numbers are measured on a 20x20 lattice (400 particles): the file runs in a
few seconds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import cubic_spline_gradient
from tensorsph.isph_periodic import (divergence, divergence_norm, gradient,
                                     kernel_moments, laplacian, pair_geometry,
                                     periodic_pairs, project, viscous_laplacian)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "outputs" / "isph_taylor_green" / "results.json"
REGISTRY = REPO / "docs" / "reports" / "method_selection_registry.json"

L, DX, HF = 1.0, 0.05, 1.3
H = HF * DX
U0, RE = 1.0, 10.0
K = 2.0 * math.pi / L
NU = U0 / (RE * K)


@pytest.fixture(scope="module")
def cloud():
    torch.set_default_dtype(torch.float64)
    n = int(round(L / DX))
    line = (torch.arange(n, dtype=torch.float64) + 0.5) * DX
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    vol = torch.full((pos.shape[0],), (L * L) / pos.shape[0], dtype=torch.float64)
    return pos, vol


def _exact_velocity(pos, t=0.0):
    amp = U0 * math.exp(-2.0 * NU * K * K * t)
    return torch.stack((amp * torch.sin(K * pos[:, 0]) * torch.cos(K * pos[:, 1]),
                        -amp * torch.cos(K * pos[:, 0]) * torch.sin(K * pos[:, 1])), dim=-1)


def test_pairs_use_the_minimum_image():
    pos = torch.tensor([[0.02, 0.02], [0.98, 0.98]], dtype=torch.float64)
    pairs = periodic_pairs(pos, 0.1, 1.0)
    assert pairs.shape == (1, 2), "the two corners are neighbours across the torus"
    _i, _j, delta, _gw = pair_geometry(pos, pairs, 0.065, 1.0)
    assert float(delta.norm()) == pytest.approx(0.04 * math.sqrt(2.0), rel=1e-12)


def test_divergence_of_the_exact_field_is_zero_to_round_off(cloud):
    pos, vol = cloud
    pairs = periodic_pairs(pos, 2.0 * H, L)
    rms = divergence_norm(pos, _exact_velocity(pos), vol, pairs, H, L)
    assert rms < 1e-12, "a lattice must reproduce the analytic solenoidal field"


def test_gradient_is_first_order_consistent(cloud):
    pos, vol = cloud
    pairs = periodic_pairs(pos, 2.0 * H, L)
    f = torch.sin(K * pos[:, 0])
    target = K * torch.cos(K * pos[:, 0])
    got = gradient(pos, f, vol, pairs, H, L)[:, 0]
    assert float((got - target).norm() / target.norm()) < 0.05
    b, _a = kernel_moments(pos, vol, pairs, H, L)
    assert float(b.mean()) == pytest.approx(1.0, rel=0.05), \
        "trace/dim first moment is the gradient normaliser on a 2D lattice"


def test_laplacian_reproduces_minus_nabla_squared(cloud):
    pos, vol = cloud
    pairs = periodic_pairs(pos, 2.0 * H, L)
    f = torch.sin(K * pos[:, 0]) * torch.cos(K * pos[:, 1])
    exact = -2.0 * K * K * f          # nabla^2 f ; the operator approximates -nabla^2
    got = laplacian(pos, f, vol, pairs, H, L)
    assert float((got + exact).norm() / exact.norm()) < 0.05


def test_viscosity_dissipates_energy(cloud):
    pos, vol = cloud
    pairs = periodic_pairs(pos, 2.0 * H, L)
    u = _exact_velocity(pos)
    e0 = float(u.square().sum())
    u1 = u + 1e-3 * viscous_laplacian(pos, u, vol, pairs, H, L, NU) * 0.0
    assert float(u1.square().sum()) == pytest.approx(e0, rel=1e-15)
    step = 0.05 * H / U0
    u2 = u + step * viscous_laplacian(pos, u, vol, pairs, H, L, NU)
    assert float(u2.square().sum()) < e0, "positive nu must remove energy"


def test_projection_reduces_the_discrete_divergence(cloud):
    pos, vol = cloud
    pairs = periodic_pairs(pos, 2.0 * H, L)
    u = _exact_velocity(pos).clone()
    u[:, 0] = u[:, 0] * (1.0 + 0.2 * torch.sin(3.0 * K * pos[:, 1]))   # deliberately compressible
    before = divergence_norm(pos, u, vol, pairs, H, L)
    u_p, _p, info = project(pos, u, vol, pairs, H, L, 1000.0, 0.05 * H / U0)
    after = divergence_norm(pos, u_p, vol, pairs, H, L)
    assert after < before
    assert info["div_after_over_div_before"] == pytest.approx(after / before, rel=1e-9)


def test_adjoint_composition_D_Dt_has_a_checkerboard_null_mode(cloud):
    """Why the Poisson operator here is a graph Laplacian, not ``D D^T``.

    ``projection_adjoint.adjoint_gradient`` pairs the difference divergence with
    the *symmetric* coefficient ``V_j p_i + V_i p_j``.  On a collocated lattice
    the sawtooth field ``p = (-1)^(i+j)`` makes every neighbour pair satisfy
    ``p_i + p_j = 0``, so ``D^T p = 0`` and ``D D^T`` is singular there: CG then
    diverges along that null direction.  Measured on this 20x20 lattice the
    pressure reached +-4.4e19 and the velocity correction 3.9e15 (see
    ``docs/xval_isph_projection_taylor_green.md`` §4).  The graph Laplacian used
    by :mod:`tensorsph.isph_periodic` has no such mode.
    """
    pos, vol = cloud
    pairs = periodic_pairs(pos, 2.0 * H, L)
    i, j, _delta, gw = pair_geometry(pos, pairs, H, L)
    n = int(round(L / DX))
    grid = torch.arange(n, dtype=torch.float64)
    gx, gy = torch.meshgrid(grid, grid, indexing="xy")
    saw = ((-1.0) ** (gx + gy)).reshape(-1)

    def adjoint_gradient_symmetric(p):
        coef = vol[j] * p[i] + vol[i] * p[j]
        out = torch.zeros((pos.shape[0], 2), dtype=torch.float64)
        out = out.index_add(0, i, coef[:, None] * gw)
        out = out.index_add(0, j, -coef[:, None] * gw)
        return out

    assert float(adjoint_gradient_symmetric(saw).abs().max()) < 1e-12
    # The graph Laplacian used instead has only the constant as null mode; on
    # the same sawtooth field it returns a non-zero action (so its Poisson
    # problem is not singular there).
    assert float(laplacian(pos, saw, vol, pairs, H, L).abs().max()) > 1e-6


def test_the_vortex_class_is_still_refused_by_the_selector():
    """The measurement is negative, so the evidence hole must stay open."""
    from tensorsph.method_selection import CaseProfile, load_registry, select_method

    registry = load_registry()
    vortex = CaseProfile(acoustic_mach=0.05, reynolds=1e4, domain="periodic",
                         smooth_inviscid_vortex=True)
    selection = select_method(vortex, registry=registry)
    statuses = {d.method: d.status for d in selection.candidates}
    assert statuses.get("isph_projection", "WITHHELD") == "WITHHELD"
    assert all(d.status != "SELECTED" for d in selection.candidates)

    doc = json.loads(REGISTRY.read_text())
    vortex_cases = [r["case"] for r in doc["records"]
                    if "taylor" in r["case"].lower() or "vortex" in r["case"].lower()]
    assert vortex_cases == [], (
        "a vortex-class record may only be added once a run meets the 15 % "
        f"standard; found {vortex_cases}")


def test_committed_benchmark_record_is_an_honest_fail():
    if not RESULTS.exists():
        pytest.skip("benchmark artefact not present in this checkout")
    doc = json.loads(RESULTS.read_text())
    assert doc["tolerance_pct"] == 15.0
    assert doc["pass"] is False, "the stored run must not be dressed up as a pass"
    worst = min(row["velocity_rel_l2_pct"] for row in doc["rows"])
    assert worst > 15.0
    for row in doc["rows"]:
        assert row["pass"] == (row["velocity_rel_l2_pct"] <= 15.0)