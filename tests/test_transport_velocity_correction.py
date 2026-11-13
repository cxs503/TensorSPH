"""Cross-checks of the transport-velocity-correction port.

The point of these tests is not "the code runs" but "the port is the upstream
operator": sign, ``(C_i + C_j)`` weight, wall factor and the displacement law
are each pinned by an independent computation (a literal transcription of
upstream's loop, a numerical derivative, a symmetry identity).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.droplet_impact import cubic_spline_kernel  # noqa: E402
from tensorsph.transport_velocity_correction import (  # noqa: E402
    SPHINXSYS_DEFAULT_COEFFICIENT, SPHINXSYS_SINGLE_STEP_FRACTION,
    kernel_gradient_integral, transport_velocity_correction)


def lattice(n=25, dx=0.02, dtype=torch.float64):
    xs = torch.arange(n, dtype=dtype) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    return pos, torch.full((pos.shape[0],), dx * dx, dtype=dtype)


def test_default_coefficients_match_upstream_header():
    # ``TransportVelocityCorrection(..., Real coefficient = 0.2)`` and "if single
    # (acoustic) time step is used, decrease to about 1/4 of the default".
    assert SPHINXSYS_DEFAULT_COEFFICIENT == 0.2
    assert SPHINXSYS_SINGLE_STEP_FRACTION == 0.25


def test_uniform_lattice_is_a_fixed_point():
    """Complete support -> the grad-1 integral vanishes -> no spurious drift."""
    dx = 0.02
    h = 1.3 * dx
    pos, vol = lattice(25, dx)
    g = kernel_gradient_integral(pos, vol, h)
    mag = torch.linalg.vector_norm(g, dim=1)
    # interior particles only: the outermost ring has a truncated support
    interior = ((pos[:, 0] > 2.2 * h) & (pos[:, 0] < 24 * dx - 2.2 * h)
                & (pos[:, 1] > 2.2 * h) & (pos[:, 1] < 24 * dx - 2.2 * h))
    assert interior.sum() > 100
    assert float(mag[interior].max()) * h < 1.0e-10


def test_matches_literal_upstream_transcription():
    """Brute-force transcription of ``interaction()``: sign and weight.

    Upstream inner loop (``transport_velocity_correction.hpp``)::

        inconsistency -= (C_i + C_j) * dW_ij * Vol_j * e_ij

    with ``e_ij`` the unit vector of the i-j separation oriented so that
    ``dW_ij * e_ij == grad_i W_ij`` (see the module docstring's derivation from
    ``pressure_gradient.hpp``).  Here ``dW/dr`` is evaluated by a numerical
    derivative of the kernel, i.e. independently of the analytic gradient used
    by the implementation.
    """
    dx = 0.02
    h = 1.3 * dx
    pos, vol = lattice(25, dx)
    # a non-symmetric, non-lattice configuration: shifts every particle
    gen = torch.Generator().manual_seed(7)
    pos = pos + 0.15 * dx * torch.randn(pos.shape, generator=gen,
                                        dtype=torch.float64)
    g = kernel_gradient_integral(pos, vol, h)

    eps = 1.0e-5 * h
    ref = torch.zeros_like(g)
    cutoff = 2.0 * h
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i == j:
                continue
            d = pos[i] - pos[j]
            r = float(torch.linalg.vector_norm(d))
            if r >= cutoff or r == 0.0:
                continue
            # float64 kernel arguments: in float32 the difference quotient of
            # the kernel loses every significant digit (W/eps amplifies the
            # representation error of W).
            w_plus = cubic_spline_kernel(torch.tensor(r + eps, dtype=torch.float64), h)
            w_minus = cubic_spline_kernel(torch.tensor(r - eps, dtype=torch.float64), h)
            dw = float(w_plus - w_minus) / (2.0 * eps)
            e = d / r                       # orientation: dW * e == grad_i W
            ref[i] -= (1.0 + 1.0) * dw * float(vol[j]) * e   # C_i + C_j = 2
    assert torch.allclose(g[::37], ref[::37], atol=5.0e-6 * float(
        torch.linalg.vector_norm(ref, dim=1).max()))


def test_surface_correction_points_out_of_the_fluid():
    """De-clumping direction: at a free surface the shift is outward.

    The grad-1 integral points into the fluid (gradient theorem), upstream's
    minus sign flips it, so the corrected particle leaves the bulk.  A sign
    error here is not cosmetic: it compresses the fluid (measured blow-up in
    ``docs/xval_sphinx_tvc``).
    """
    dx = 0.02
    h = 1.3 * dx
    pos, vol = lattice(25, dx)
    g = kernel_gradient_integral(pos, vol, h)
    top = pos[:, 1] == pos[:, 1].max()
    # particles on the top row: support truncated from above
    assert float(g[:, 1][top].mean()) > 0.0
    bottom = pos[:, 1] == 0.0
    assert float(g[:, 1][bottom].mean()) < 0.0


def test_wall_contact_uses_upstream_factor_two():
    dx = 0.02
    h = 1.3 * dx
    pos, vol = lattice(9, dx)
    wall = torch.tensor([[float(pos[:, 0].mean()), -dx],
                         [float(pos[:, 0].mean()) - dx, -dx]],
                        dtype=torch.float64)
    wall_vol = torch.full((wall.shape[0],), dx * dx, dtype=torch.float64)
    both = torch.cat((pos, wall))
    g_all = kernel_gradient_integral(both, torch.cat((vol, wall_vol)), h,
                                     wall_count=wall.shape[0])
    g_fluid = kernel_gradient_integral(pos, vol, h)
    assert not torch.allclose(g_all, g_fluid)
    # removing the wall particles moves the bottom row; everything else that is
    # far from them is unchanged
    far = pos[:, 1] > 4 * dx
    assert torch.allclose(g_all[far], g_fluid[far], atol=1.0e-12)


def test_shift_law_and_scope_mask():
    dx = 0.02
    h = 1.3 * dx
    pos, vol = lattice(25, dx)
    res = transport_velocity_correction(pos, vol, h, alpha=0.05)
    assert res.alpha == pytest.approx(0.05)
    assert res.shift.shape == pos.shape
    bound = 0.05 * h * h * torch.linalg.vector_norm(res.gradient_integral, dim=1)
    assert torch.all(torch.linalg.vector_norm(res.shift, dim=1) <= bound + 1e-14)
    mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    mask[0] = True
    res_mask = transport_velocity_correction(pos, vol, h, alpha=0.05,
                                             active=mask)
    assert float(torch.linalg.vector_norm(res_mask.shift[1:])) == 0.0
    assert float(torch.linalg.vector_norm(res_mask.shift[0])) == float(
        torch.linalg.vector_norm(res.shift[0]))


def test_net_translation_of_a_symmetric_blob_is_zero():
    """Symmetry: a centred symmetric blob must not drift as a whole."""
    dx = 0.02
    h = 1.3 * dx
    pos, vol = lattice(21, dx)
    centre = pos.mean(dim=0)
    pos = pos - centre
    res = transport_velocity_correction(pos, vol, h, alpha=0.05)
    net = res.shift.sum(dim=0)
    scale = torch.linalg.vector_norm(res.shift, dim=1).sum()
    assert float(torch.linalg.vector_norm(net)) < 1.0e-10 * max(float(scale), 1.0)


def test_disabled_correction_leaves_the_solver_bit_identical():
    from tensorsph.dam_break import DamBreakConfig, run_case
    cfg = DamBreakConfig(dx=0.04, dt=1.0e-4, steps=40, output_every=20,
                         device="cpu")
    a = run_case(cfg)
    b = run_case(DamBreakConfig(dx=0.04, dt=1.0e-4, steps=40, output_every=20,
                                device="cpu", tvc_every=0, tvc_alpha=0.05))
    assert torch.equal(a.final_positions, b.final_positions)


def test_correction_keeps_the_column_bounded_and_massive():
    """A short run with the correction stays physical (no blow-up)."""
    from tensorsph.dam_break import DamBreakConfig, run_case
    cfg = DamBreakConfig(dx=0.04, dt=1.0e-4, steps=120, output_every=40,
                         device="cpu", tvc_every=1, tvc_alpha=0.05)
    run = run_case(cfg)
    assert run.mass_drift == 0.0
    assert run.escaped == 0
    assert math.isfinite(run.max_tvc_shift)
    assert run.max_tvc_shift / (cfg.h_factor * cfg.dx) < 0.5
    assert run.max_density_compression < 0.1