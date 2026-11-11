"""Kernel-level tests for the WCSPH verification chain."""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.wcsph import kernels


def _quadrature(dim: int, kind: str, n: int = 200001) -> float:
    q = torch.linspace(0.0, kernels.SUPPORT_FACTOR[kind], n, dtype=torch.float64)
    f, _ = kernels.shape(q, kind)
    # full radial measure: the 1-D kernel is even, so the half-line integral
    # carries a factor 2 (the 2-D and 3-D polar measures are already complete)
    surface = {1: 2.0 * torch.ones_like(q), 2: 2.0 * math.pi * q,
               3: 4.0 * math.pi * q * q}[dim]
    return float(torch.trapezoid(f * surface, q)) * kernels.normalisation(dim, kind)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_quintic_is_normalised(dim):
    """The tabulated constants are the ones that make the kernel integrate to 1."""
    assert _quadrature(dim, "quintic") == pytest.approx(1.0, abs=2.0e-6)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_gaussian_is_normalised_within_its_truncation(dim):
    """Truncation at q = 3 costs at most ~1e-3 of the mass, and nothing else.

    The 1-D/2-D tails are ``erfc(3) = 2.2e-5``; the 3-D tail is larger
    (``erfc(3) + 6 exp(-9)/sqrt(pi) = 7.6e-4``) because the measure grows as
    ``q^2``.  Both are stated here rather than hidden in one loose tolerance.
    """
    tolerance = 1.0e-3 if dim == 3 else 2.0e-4
    assert _quadrature(dim, "gaussian") == pytest.approx(1.0, abs=tolerance)


def test_kernel_vanishes_at_its_support():
    for kind in ("quintic", "gaussian"):
        for dim in (1, 2, 3):
            h = 0.013
            w = kernels.kernel_value(
                torch.tensor([kernels.SUPPORT_FACTOR[kind] * h + 1e-12],
                             dtype=torch.float64), h, dim, kind)
            assert float(w.abs().max()) == pytest.approx(0.0, abs=1.0e-12)


def test_quintic_is_continuous_across_its_piecewise_boundaries():
    """The three polynomial pieces must join with no jump in value or slope."""
    eps = 1.0e-9
    for q0 in (1.0, 2.0):
        lo = torch.tensor([q0 - eps], dtype=torch.float64)
        hi = torch.tensor([q0 + eps], dtype=torch.float64)
        f_lo, df_lo = kernels.shape(lo, "quintic")
        f_hi, df_hi = kernels.shape(hi, "quintic")
        assert float((f_lo - f_hi).abs().max()) < 1.0e-6
        assert float((df_lo - df_hi).abs().max()) < 1.0e-6


@pytest.mark.parametrize("kind", ["quintic", "gaussian"])
@pytest.mark.parametrize("dim", [2, 3])
def test_gradient_matches_finite_difference(kind, dim):
    h = 0.02
    delta = torch.tensor([[0.013, 0.004, 0.002][:dim],
                          [-0.021, 0.007, -0.003][:dim]], dtype=torch.float64)
    _, grad = kernels.kernel_pairs(delta, h, dim, kind)
    eps = 1.0e-7
    fd = torch.zeros_like(grad)
    for a in range(dim):
        shift = torch.zeros(dim, dtype=torch.float64)
        shift[a] = eps
        w_plus = kernels.kernel_value(
            torch.linalg.vector_norm(delta + shift, dim=1), h, dim, kind)
        w_minus = kernels.kernel_value(
            torch.linalg.vector_norm(delta - shift, dim=1), h, dim, kind)
        fd[:, a] = (w_plus - w_minus) / (2.0 * eps)
    assert torch.allclose(grad, fd, rtol=1e-5, atol=1e-2)


def test_pair_gradient_is_antisymmetric():
    delta = torch.tensor([[0.01, -0.02]], dtype=torch.float64)
    _, g = kernels.kernel_pairs(delta, 0.02, 2, "quintic")
    _, g_neg = kernels.kernel_pairs(-delta, 0.02, 2, "quintic")
    assert torch.allclose(g, -g_neg)


def test_kernel_is_radially_symmetric():
    h = 0.02
    r = torch.tensor([0.011], dtype=torch.float64)
    w1 = kernels.kernel_value(r, h, 2, "quintic")
    w2 = kernels.kernel_value(r, h, 2, "quintic")
    assert torch.equal(w1, w2)
    _, g = kernels.kernel_pairs(torch.tensor([[0.0, 0.011]], dtype=torch.float64),
                                h, 2, "quintic")
    assert float(g[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_self_contribution_matches_kernel_at_zero():
    h = torch.tensor(0.013, dtype=torch.float64)
    assert float(kernels.self_contribution(h, 2, "quintic")) == pytest.approx(
        float(kernels.kernel_value(torch.zeros((), dtype=torch.float64), h, 2,
                                   "quintic")))


def test_adaptive_h_follows_the_benz_scaling_and_is_clipped():
    rho = torch.tensor([1000.0, 1100.0, 10.0], dtype=torch.float64)
    h = kernels.adaptive_h(rho, 1000.0, 0.013, 2)
    assert float(h[0]) == pytest.approx(0.013)
    assert float(h[1]) < 0.013
    assert float(h[2]) == pytest.approx(1.25 * 0.013)


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError):
        kernels.normalisation(2, "quartic")
    with pytest.raises(ValueError):
        kernels.shape(torch.zeros(3), "quartic")
    with pytest.raises(ValueError):
        kernels.normalisation(4, "quintic")


def test_convergence_order_is_the_log_ratio():
    assert kernels.convergence_order([0.04, 0.01], [0.02, 0.01]) == pytest.approx(2.0)
    assert kernels.convergence_order([0.01, 0.02], [0.02, 0.01]) == pytest.approx(-1.0)
    assert math.isinf(kernels.convergence_order([1.0, 0.0], [0.02, 0.01]))
    assert math.isnan(kernels.convergence_order([0.0, 1.0], [0.02, 0.01]))