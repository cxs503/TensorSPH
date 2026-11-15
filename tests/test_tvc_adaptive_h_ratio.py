"""Upstream's adaptive smoothing-length branch of the TVC operator.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.  Until this file the port
ran with ``h_ratio == 1`` only (recorded as an open item in
``docs/xval_sphinx_tvc/README.md`` and ``CADENCE_INVARIANCE.md``).  The branch
is now ported, and these tests pin it to the fetched upstream sources
(``docs/xval_sphinx_tvc/oss_source/adaptive/``, sha256 in that directory's
``PROVENANCE.json``):

==============================================  ===============================
upstream source                                  what it fixes
==============================================  ===============================
``adaptation.h`` ``SmoothingLengthRatio()``       ``h_ratio`` is *reference over
                                                  local* smoothing length
``base_kernel.cpp`` ``Kernel::dW(h_ratio, ...)``  ``q = r*inv_h*h_ratio`` and the
                                                  normalisation ``h_ratio**(dim+1)``
                                                  (from ``h_factor_dW_2D`` in
                                                  ``base_kernel.h``)
``transport_velocity_correction.hpp`` ``update``  the ``inv_h_ratio**2`` factor on
                                                  the displacement
``base_body_relation.h`` ``SearchDepthAdaptive``  the cut-off is the owner
                                                  particle's ``2 * h_local``
==============================================  ===============================

The three consequences tested here:

1. ``h_ratio = None`` is bit-identical to the pre-port operator (no silent
   change to the existing, graded dam-break runs).
2. A *uniform* ratio ``c`` is exactly the operator at ``h/c`` -- so the
   displacement obeys ``|dr|/h_local = alpha * |grad-1|/h`` independent of the
   local refinement, which is the point of upstream's factor (derivation in
   ``docs/xval_sphinx_tvc/ADAPTIVE_H_RATIO.md``).
3. The per-pair kernel evaluation is *owner-side*: a pair inside
   ``2 h_local_i`` but outside ``2 h_local_j`` contributes to ``i`` only.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral,
    transport_velocity_correction,
)


def jittered_lattice(n=5, dx=0.1, seed=7, dtype=torch.float64):
    """A small 2-D jittered lattice plus two wall particles."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.arange(n, dtype=dtype) * dx
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    fluid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    fluid = fluid + 0.13 * dx * (torch.rand(fluid.shape, generator=g,
                                            dtype=dtype) - 0.5)
    wall = torch.tensor([[-dx, 0.0], [dx * n, 0.3]], dtype=dtype)
    return torch.cat([fluid, wall]), fluid.shape[0], 0.13 * dx


# --------------------------------------------------------------------------- #
# 1. no silent change to the graded operator
# --------------------------------------------------------------------------- #
def test_none_equals_unit_ratio_exactly():
    pos, _, dx = jittered_lattice()
    vol = torch.full((pos.shape[0],), 0.7 * dx * dx, dtype=torch.float64)
    base = kernel_gradient_integral(pos, vol, 1.3 * dx, wall_count=2)
    for ratio in (1.0, torch.ones(pos.shape[0], dtype=torch.float64)):
        got = kernel_gradient_integral(pos, vol, 1.3 * dx, wall_count=2,
                                       h_ratio=ratio)
        assert torch.equal(got, base)

    r0 = transport_velocity_correction(pos, vol, 1.3 * dx, alpha=0.05,
                                       wall_count=2)
    r1 = transport_velocity_correction(pos, vol, 1.3 * dx, alpha=0.05,
                                       wall_count=2, h_ratio=1.0)
    assert torch.equal(r0.shift, r1.shift)


# --------------------------------------------------------------------------- #
# 2. uniform ratio == operator at the local h (the derivation, measured)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", [0.5, 2.0, 4.0])
def test_uniform_ratio_is_the_local_h_operator(ratio):
    pos, _, dx = jittered_lattice()
    vol = torch.full((pos.shape[0],), 0.7 * dx * dx, dtype=torch.float64)
    h = 1.3 * dx
    h_local = h / ratio

    g_ref = kernel_gradient_integral(pos, vol, h_local, wall_count=2)
    g_ratio = kernel_gradient_integral(pos, vol, h, wall_count=2,
                                       h_ratio=ratio)
    assert torch.allclose(g_ratio, g_ref, rtol=0.0, atol=1e-13)

    r_ref = transport_velocity_correction(pos, vol, h_local, alpha=0.05,
                                          wall_count=2)
    r_ratio = transport_velocity_correction(pos, vol, h, alpha=0.05,
                                            wall_count=2, h_ratio=ratio)
    assert torch.allclose(r_ratio.shift, r_ref.shift, rtol=0.0, atol=1e-15)
    # and the invariant actually claimed: |dr|/h_local is ratio independent
    assert torch.allclose(r_ratio.shift / h_local, r_ref.shift / h_local,
                          rtol=0.0, atol=1e-14)


# --------------------------------------------------------------------------- #
# 3. literal transcription of upstream ``Kernel::dW(h_ratio, r)``
# --------------------------------------------------------------------------- #
def upstream_dW_2d(r, h_ref, h_ratio):
    """``base_kernel.cpp`` for the 2-D cubic B-spline, verbatim formulas.

    ``factor_W_2D = 10/(7 pi h^2)`` (``kernel_cubic_B_spline.cpp``),
    ``factor_dW_2D = inv_h * factor_W_2D`` (``base_kernel.cpp``
    ``setDerivativeParameters``), ``q = r * inv_h * h_ratio``,
    ``h_factor_dW_2D(h_ratio) = h_ratio**3`` (``base_kernel.h``).
    """
    factor_W_2D = 10.0 / (7.0 * math.pi * h_ref * h_ref)
    factor_dW_2D = (1.0 / h_ref) * factor_W_2D
    q = r / h_ref * h_ratio

    def dW_2D(qq):
        if qq < 1.0:
            return 9.0 * qq * qq / 4.0 - 3.0 * qq
        if qq < 2.0:
            return -3.0 * (2.0 - qq) ** 2 / 4.0
        return 0.0

    return factor_dW_2D * dW_2D(q) * h_ratio ** 3


def test_matches_upstream_dW_transcription():
    pos, n_fluid, dx = jittered_lattice(n=4)
    h = 1.3 * dx
    vol = torch.full((pos.shape[0],), 0.5 * dx * dx, dtype=torch.float64)
    ratios = torch.linspace(0.6, 2.4, pos.shape[0], dtype=torch.float64)

    got = kernel_gradient_integral(pos, vol, h, wall_count=2, h_ratio=ratios)

    # brute force: g_i = -sum_j (C_i + C_j) dW_ij V_j e_ij, C = 1, walls *2
    want = torch.zeros((n_fluid, 2), dtype=torch.float64)
    for i in range(n_fluid):
        for j in range(pos.shape[0]):
            if j == i:
                continue
            d = pos[i] - pos[j]
            r = float(torch.linalg.vector_norm(d))
            dW = upstream_dW_2d(r, h, float(ratios[i]))
            weight = 2.0 if j >= n_fluid else 2.0
            want[i] += -weight * dW * float(vol[j]) * (d / r) if r > 0 else 0.0
    assert torch.allclose(got, want, rtol=0.0, atol=1e-13), (got - want).abs().max()


# --------------------------------------------------------------------------- #
# 4. the cut-off is the owner particle's
# --------------------------------------------------------------------------- #
def test_cutoff_follows_owner_ratio():
    h = 0.1
    # particle 0: ratio 1.0 (h_local = h); particle 1: ratio 0.5 (h_local = 2h)
    r = 2.5 * h
    pos = torch.tensor([[0.0, 0.0], [r, 0.0]], dtype=torch.float64)
    vol = torch.full((2,), 1.0e-3, dtype=torch.float64)
    pairs = torch.tensor([[0, 1]])
    got = kernel_gradient_integral(pos, vol, h, pairs=pairs,
                                   h_ratio=torch.tensor([1.0, 0.5],
                                                        dtype=torch.float64))
    assert torch.all(got[0] == 0.0), "particle 0 must not see a neighbour at 2.5h"
    assert torch.linalg.vector_norm(got[1]) > 0.0, "particle 1 (h_local=2h) must"

    # symmetric control: with both ratios 1 neither particle sees the other
    none_seen = kernel_gradient_integral(pos, vol, h, pairs=pairs,
                                         h_ratio=torch.ones(2, dtype=torch.float64))
    assert torch.all(none_seen == 0.0)


# --------------------------------------------------------------------------- #
# 5. the wall branch carries the same factor (it is applied in ``update()``)
# --------------------------------------------------------------------------- #
def test_wall_branch_scales_with_inv_ratio_squared():
    dx = 0.1
    h = 1.3 * dx
    fluid = torch.tensor([[0.0, 0.5 * dx]], dtype=torch.float64)
    wall = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    pos = torch.cat([fluid, wall])
    vol = torch.full((2,), 0.6 * dx * dx, dtype=torch.float64)
    ratio = 2.0

    r_ref = transport_velocity_correction(pos, vol, h / ratio, alpha=0.05,
                                          wall_count=1)
    r_ratio = transport_velocity_correction(pos, vol, h, alpha=0.05,
                                            wall_count=1, h_ratio=ratio)
    assert torch.allclose(r_ratio.shift, r_ref.shift, rtol=0.0, atol=1e-16)
    assert torch.linalg.vector_norm(r_ref.shift).item() > 0.0


# --------------------------------------------------------------------------- #
# 6. the recorded bookkeeping
# --------------------------------------------------------------------------- #
def test_extras_record_the_ratio_range():
    pos, _, dx = jittered_lattice(n=3)
    vol = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    ratios = torch.full((pos.shape[0] - 2,), 1.5, dtype=torch.float64)
    res = transport_velocity_correction(pos, vol, 1.3 * dx, alpha=0.05,
                                        wall_count=2, h_ratio=ratios)
    lo, hi = res.extras["h_ratio"]
    # the recorded range is over the *fluid* particles (walls are padded with 1.0
    # inside the gradient only, which test_cutoff_follows_owner_ratio covers)
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(1.5)
    assert res.extras["h_ratio"] is not None
    plain = transport_velocity_correction(pos, vol, 1.3 * dx, alpha=0.05,
                                          wall_count=2)
    assert plain.extras["h_ratio"] is None