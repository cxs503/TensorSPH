"""Invariants of the adjoint-consistent projection used for the Young-Laplace drop.

The point of these tests is that the projection is a *projection*: the operator
that measures divergence and the operator that corrects it must be a discrete
adjoint pair, otherwise the correction re-injects the divergence it was supposed
to remove.  Every assertion here is an identity or a measured ratio on a small
(R/dx = 10, N = 1203) hex block, so the file runs in seconds.

The counters are deliberate: ``test_mls_pair_is_not_adjoint`` and
``test_mls_projection_leaves_divergence`` pin the *old* behaviour, so the tests
fail if either (a) the new pair silently degrades to the old one, or (b) someone
"fixes" the old operators without updating the record in
``docs/verification_laplace_sph_isph.md`` §5.1.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import mls_divergence, mls_gradient
from tensorsph.laplace_isph import (hex_block, pressure_poisson)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.projection_adjoint import (
    adjoint_gradient,
    adjoint_laplacian,
    adjointness_defect,
    pair_divergence,
    poisson_adjoint,
    project_velocity,
    relax_projection_consistent,
)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "benchmarks" / "laplace_sph" / "results_dynamic.json"
DOC = HERE / "docs" / "verification_laplace_sph_isph_dynamic.md"

R, DX, HF = 0.02, 0.002, 1.3
H = HF * DX
RHO = 1000.0


@pytest.fixture(scope="module")
def cloud():
    pos, mass, liquid, half = hex_block(R, DX, 0.012, RHO)
    vol = mass / RHO
    pairs = cell_linked_pairs(pos, 2.0 * H)
    return pos, vol, pairs, liquid, half


def _mls_adjoint_defect(pos, vol, pairs, h, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = pos.shape[0]
    u = torch.randn((n, 2), generator=g, dtype=pos.dtype)
    q = torch.randn((n,), generator=g, dtype=pos.dtype)
    lhs = (mls_divergence(pos, u, vol, pairs, h) * q).sum()
    rhs = -(u * mls_gradient(pos, q, vol, pairs, h)).sum()
    return float((lhs - rhs).abs() / lhs.abs())


# --------------------------------------------------------------------------- #
# the adjoint pair
# --------------------------------------------------------------------------- #
def test_pair_adjointness_is_roundoff(cloud):
    pos, vol, pairs, _, _ = cloud
    assert adjointness_defect(pos, vol, pairs, H) < 1e-12


def test_mls_pair_is_not_adjoint(cloud):
    """Negative control: the shipped MLS pair is off by O(10)."""
    pos, vol, pairs, _, _ = cloud
    assert _mls_adjoint_defect(pos, vol, pairs, H) > 10.0


def test_adjoint_gradient_recovers_a_linear_field(cloud):
    """``G p`` with ``G = -D^T`` must be aligned with the analytic ``grad p``."""
    pos, vol, pairs, _, _ = cloud
    g = adjoint_gradient(pos, pos[:, 0].clone(), vol, pairs, H)
    interior = torch.linalg.vector_norm(pos, dim=1) < 0.6 * R
    mean = g[interior].mean(dim=0)
    assert mean[0] == pytest.approx(1.0, rel=0.02)   # measured 1.008
    assert abs(float(mean[1])) < 1e-6


def test_adjoint_laplacian_is_symmetric_and_negative(cloud):
    pos, vol, pairs, _, _ = cloud
    g = torch.Generator().manual_seed(7)
    n = pos.shape[0]
    q1 = torch.randn(n, generator=g, dtype=pos.dtype)
    q2 = torch.randn(n, generator=g, dtype=pos.dtype)
    a = float((q1 * adjoint_laplacian(pos, q2, vol, pairs, H)).sum())
    b = float((q2 * adjoint_laplacian(pos, q1, vol, pairs, H)).sum())
    assert abs(a - b) / abs(a) < 1e-10          # symmetry
    assert float((q1 * adjoint_laplacian(pos, q1, vol, pairs, H)).sum()) < 0.0


# --------------------------------------------------------------------------- #
# the projection itself
# --------------------------------------------------------------------------- #
def test_projection_removes_the_discrete_divergence(cloud):
    pos, vol, pairs, _, _ = cloud
    g = torch.Generator().manual_seed(11)
    n = pos.shape[0]
    u = torch.randn((n, 2), generator=g, dtype=pos.dtype) * 1e-3
    _, _, info = project_velocity(pos, u, vol, pairs, H, RHO, 1e-4, gauge=0,
                                  rel_tol=1e-10)
    assert info["div_after_over_div_before"] < 1e-8   # measured 8.7e-13


def test_mls_projection_leaves_divergence(cloud):
    """Negative control: the old operator pair cannot project."""
    pos, vol, pairs, _, _ = cloud
    g = torch.Generator().manual_seed(11)
    n = pos.shape[0]
    dt = 1e-4
    u = torch.randn((n, 2), generator=g, dtype=pos.dtype) * 1e-3
    div_star = mls_divergence(pos, u, vol, pairs, H)
    dirichlet = torch.zeros(n, dtype=torch.bool)
    dirichlet[0] = True
    p, _ = pressure_poisson(pos, RHO * div_star / dt, pairs, vol, H, dirichlet,
                            rel_tol=1e-10, max_iter=5000)
    u_new = u - dt * mls_gradient(pos, p, vol, pairs, H) / RHO
    before = float(torch.linalg.vector_norm(div_star))
    after = float(torch.linalg.vector_norm(
        pair_divergence(pos, u_new, vol, pairs, H)))
    assert after / before > 0.5, after / before


def test_poisson_is_exact_on_a_manufactured_field(cloud):
    """``CG`` must invert its own operator: ``A p = b`` -> ``A p - b`` ~ 0."""
    pos, vol, pairs, _, _ = cloud
    n = pos.shape[0]
    g = torch.Generator().manual_seed(3)
    q = torch.randn(n, generator=g, dtype=pos.dtype)
    b = -adjoint_laplacian(pos, q, vol, pairs, H)
    p, _ = poisson_adjoint(pos, b, vol, pairs, H, gauge=0, rel_tol=1e-11,
                           max_iter=8000)
    resid = -adjoint_laplacian(pos, p, vol, pairs, H) - b
    keep = torch.ones(n, dtype=torch.bool)
    keep[0] = False
    assert float(torch.linalg.vector_norm(resid[keep]) /
                 torch.linalg.vector_norm(b[keep])) < 1e-9


# --------------------------------------------------------------------------- #
# the committed dynamics artefact
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dyn():
    if not RES.exists():
        pytest.skip("results_dynamic.json not built")
    return json.loads(RES.read_text())


def test_dynamics_artifact_shape(dyn):
    assert dyn["case"] == "CASE-LAPLACE-SPH-DYNAMIC"
    assert len(dyn["runs"]) >= 2


def test_legacy_path_still_diverges(dyn):
    """The record of the old failure must not be silently dropped."""
    assert dyn["legacy"]["diverged"] is True
    assert dyn["legacy"]["steps_to_divergence"] >= 1


def test_damped_runs_do_not_diverge(dyn):
    for r in dyn["runs"]:
        assert r["diverged"] is False, (r["nu_factor"], r["R_over_dx"])


def test_projection_stays_exact_through_the_march(dyn):
    for r in dyn["runs"]:
        for h in r["history"]:
            assert h["div_after_over_div_before"] < 1e-5, (r["nu_factor"],
                                                           h["step"])


def test_best_run_meets_the_laplace_gate(dyn):
    best = dyn["best"]
    assert abs(best["dp_rel_error_pct"]) <= 15.0
    assert abs(best["area_ratio_final"] - 1.0) <= 0.02


def test_documented_page_matches_rendering(dyn):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "laplace_sph_make_doc_dynamic",
        HERE / "benchmarks" / "laplace_sph" / "make_doc_dynamic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert DOC.read_text() == mod.render(json.loads(RES.read_text()))