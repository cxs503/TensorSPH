"""Unit and analytic verification tests for the interface-fit curvature operator.

``tensorsph.csf_curvature.fit_curvature`` replaces ``kappa = div n`` inside the
CSF model.  ``div n`` read +25 % on a circular drop
(``docs/verification_surface_tension_curvature.md``), which is outside the 15 %
platform gate.  The tests here pin the operator to analytic curvature:

1. points exactly on a circle of radius ``R`` -> ``kappa = 1/R`` (both fits);
2. points on an ellipse -> the analytic ``kappa`` at every sample point;
3. the lattice droplet used by the benchmarks -> the ``delta_s``-weighted mean
   stays inside the 15 % gate and the sign never flips;
4. degenerate stencils (bulk particles, 3-D input, empty pair list) fall back to
   the caller's MLS curvature instead of returning NaN.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_curvature import fit_curvature
from tensorsph.droplet_impact import Config

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_law" / "run.py"


def _bench_module():
    spec = importlib.util.spec_from_file_location("laplace_law_run", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pairs_within(pos, cutoff):
    d = torch.cdist(pos, pos)
    i, j = torch.nonzero((d <= cutoff) & (d > 0.0), as_tuple=True)
    return torch.stack([i, j], dim=1)


def _circle(n, radius):
    theta = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    pos = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    normal = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return pos, normal


def _ellipse(n, a, b):
    theta = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    pos = torch.stack([a * torch.cos(theta), b * torch.sin(theta)], dim=1)
    # Outward unit normal of x^2/a^2 + y^2/b^2 = 1.
    grad = torch.stack([pos[:, 0] / a ** 2, pos[:, 1] / b ** 2], dim=1)
    normal = grad / torch.linalg.vector_norm(grad, dim=1, keepdim=True)
    # Analytic curvature ab / (a^2 sin^2 + b^2 cos^2)^{3/2}.
    s, c = torch.sin(theta), torch.cos(theta)
    kappa = a * b / (a * a * s * s + b * b * c * c) ** 1.5
    return pos, normal, kappa


@pytest.mark.parametrize("method", ["fit_quadratic", "fit_circle"])
@pytest.mark.parametrize("radius,n", [(0.02, 60), (0.05, 200)])
def test_circle_curvature_is_analytic(method, radius, n):
    """Exact circle points: the fit must return 1/R with the right sign."""
    pos, normal = _circle(n, radius)
    h = 2.0 * radius * math.pi / n
    pairs = _pairs_within(pos, 4.0 * h)
    delta_s = torch.ones(n)
    kappa = fit_curvature(pos, normal, delta_s, pairs, h, method=method)
    exact = 1.0 / radius
    # The quadratic fit truncates at O((stencil/R)^2); at n = 60 the 4h stencil
    # spans a 48 deg arc, which is where the 5 % comes from.  The circle fit is
    # algebraically exact on exact circle data.
    tol = 1.0e-3 if method == "fit_circle" else (0.07 if n < 100 else 0.02)
    assert torch.all(kappa > 0.0), "outward normals on a convex circle give kappa > 0"
    assert abs(float(kappa.mean()) - exact) / exact < tol
    assert float((kappa - exact).abs().max()) / exact < 3.0 * tol


@pytest.mark.parametrize("a,b", [(0.03, 0.015), (0.02, 0.02)])
def test_ellipse_curvature_matches_analytic(a, b):
    """Non-circular interface: pointwise curvature, not just the mean."""
    pos, normal, kappa_exact = _ellipse(400, a, b)
    h = 2.0 * math.pi * max(a, b) / 400
    pairs = _pairs_within(pos, 4.0 * h)
    kappa = fit_curvature(pos, normal, torch.ones(400), pairs, h,
                          method="fit_quadratic")
    rel = ((kappa - kappa_exact) / kappa_exact).abs().max()
    assert float(rel) < 0.02, f"max pointwise curvature error {float(rel):.3%}"


def test_lattice_droplet_curvature_inside_gate():
    """The benchmark droplet: delta_s-weighted mean kappa within 15 % of 1/R."""
    law = _bench_module()
    from tensorsph.csf_consistent import csf_fields
    from tensorsph.neighbor_search import cell_linked_pairs

    radius, dx = 0.02, 0.002
    cfg = Config(dx=dx, h_factor=1.3, rho0=1000.0, sound_speed=20.0,
                 surface_tension_coefficient=7.2, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    pos, mass = law.hexagonal_droplet(radius, dx, 1000.0)
    pairs = cell_linked_pairs(pos, 4.0 * h)
    state = csf_fields(pos, mass, pairs, h, 1000.0,
                       curvature_scheme="fit_quadratic")
    w = state.delta_s
    kappa_w = float((state.kappa.abs() * w).sum() / w.sum())
    exact = 1.0 / radius
    assert abs(kappa_w - exact) / exact < 0.15, (
        f"kappa {kappa_w:.2f} vs {exact:.2f} "
        f"({(kappa_w - exact) / exact:+.1%})")
    # Force direction is set by the normal, but a curvature sign flip would make
    # the drop explode; the shell must never be negative.
    r = torch.linalg.vector_norm(pos, dim=1)
    shell = (r / radius > 0.85) & (r / radius < 1.02)
    assert bool((state.kappa[shell] > 0.0).all())


def test_mls_scheme_is_the_identity_on_the_fallback():
    pos, normal = _circle(40, 0.02)
    pairs = _pairs_within(pos, 0.01)
    fallback = torch.full((40,), 7.0)
    out = fit_curvature(pos, normal, torch.ones(40), pairs, 0.0025,
                        fallback=fallback, method="mls")
    assert torch.equal(out, fallback)


def test_degenerate_stencils_fall_back():
    """Bulk particles, empty pair lists and 3-D input keep the fallback."""
    pos, normal = _circle(12, 0.02)
    fallback = torch.full((12,), -3.0)
    # No pairs at all.
    out = fit_curvature(pos, normal, torch.ones(12), torch.zeros((0, 2), dtype=torch.long),
                        0.0025, fallback=fallback)
    assert torch.equal(out, fallback)
    # delta_s below the interface threshold -> bulk, no fit.
    out = fit_curvature(pos, normal, torch.zeros(12), _pairs_within(pos, 0.01),
                        0.0025, fallback=fallback)
    assert torch.equal(out, fallback)
    # 3-D positions are not fitted (a quadric/sphere fit is a different operator).
    pos3 = torch.zeros((5, 3))
    out = fit_curvature(pos3, torch.zeros((5, 3)), torch.ones(5),
                        torch.tensor([[0, 1], [1, 2]]), 0.0025, fallback=fallback[:5])
    assert torch.equal(out, fallback[:5])
    # No NaN/Inf can leak out of the linear solves.
    out = fit_curvature(pos, normal, torch.ones(12), _pairs_within(pos, 0.01), 0.0025)
    assert torch.isfinite(out).all()


def test_config_rejects_unknown_curvature_scheme():
    from tensorsph.droplet_impact import validate_config
    with pytest.raises(ValueError):
        validate_config(Config(csf_curvature="parabola"))