"""Wendland C2 kernel in the WCSPH verification chain (m14479, red-first).

Before this change ``tensorsph.wcsph.kernels`` shipped exactly two kernels, the
quintic spline and a truncated Gaussian (``SUPPORT_FACTOR == {"quintic": 3,
"gaussian": 3}``).  Every production SPH code in the field defaults to the
Wendland C2 kernel instead (Wendland 1995; Monaghan 2005; DualSPHysics uses it
as the default because its Fourier transform is positive, which removes the
pairing/clumping instability of the spline family at high neighbour counts).
The tree already carries two *independent* Wendland C2 implementations used by
other cases -- 2-D in :mod:`tensorsph.droplet_impact`, 3-D in
:mod:`tensorsph.threed.kernel3` -- but the WCSPH chain could not use the
kernel at all, so its verification face was kernel-family-limited.

This file is the red half of the change: it was run and captured before
``kernels.py`` learned ``"wendland"`` and every test below failed with
``ValueError: unknown kernel 'wendland'`` (the log is quoted in
``docs/wendland_wcsph_kernel_m14479.md``).  After the implementation it is the
green half: constants re-derived by quadrature, compact support and C2
continuity checked, gradient checked against finite differences, and the new
kernel cross-checked against the two existing independent implementations plus
a cross-kernel dam-break run that must give the same front within a stated
tolerance.

Conventions: ``q = r / h``, support ``q < 2``, profile
``f(q) = (1 - q/2)^4 (2q + 1)``, ``f'(q) = -5 q (1 - q/2)^3``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.wcsph import WCSPHConfig, front_at, kernels, particle_pair
from tensorsph.wcsph.cases import DamBreakConfig, dam_break

#: Closed-form normalisations ``alpha_d`` with ``int W dV = 1`` over the 2h ball
#: (``d = 1``: ``2 alpha h int_0^2 f dq = 4/3 alpha h``; ``d = 2``: ``2 pi h^2
#: alpha int f q dq = (4 pi / 7) alpha h^2``; ``d = 3``: ``4 pi h^3 alpha int
#: f q^2 dq = (16 pi / 21) alpha h^3``).
ALPHA = {1: 3.0 / 4.0, 2: 7.0 / (4.0 * math.pi), 3: 21.0 / (16.0 * math.pi)}


def _quadrature(dim: int, n: int = 400001) -> float:
    """``int W dV`` over the 2h ball, on the analytic ``q`` variable."""
    q = torch.linspace(0.0, kernels.SUPPORT_FACTOR["wendland"], n,
                       dtype=torch.float64)
    f, _ = kernels.shape(q, "wendland")
    surface = {
        1: 2.0 * torch.ones_like(q),
        2: 2.0 * math.pi * q,
        3: 4.0 * math.pi * q * q,
    }[dim]
    return float(torch.trapezoid(f * surface, q)) * kernels.normalisation(
        dim, "wendland")


# --------------------------------------------------------------------------- #
# 1. the kernel itself: support, normalisation, continuity
# --------------------------------------------------------------------------- #
def test_support_factor_is_two():
    assert kernels.SUPPORT_FACTOR["wendland"] == 2.0


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_normalisation_is_the_closed_form(dim):
    assert kernels.normalisation(dim, "wendland") == pytest.approx(
        ALPHA[dim], rel=0.0, abs=1.0e-15)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_kernel_integrates_to_one(dim):
    assert _quadrature(dim) == pytest.approx(1.0, abs=2.0e-6)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_compact_support_is_exact(dim):
    h = 0.017
    q = torch.tensor([2.0, 2.0000001, 2.5, 10.0], dtype=torch.float64)
    w = kernels.kernel_value(q * h, h, dim, "wendland")
    assert torch.equal(w, torch.zeros_like(w))


def test_profile_and_slope_are_c2_at_the_support_edge():
    """Wendland C2 vanishes with zero slope *and* zero curvature at q = 2."""
    eps = 1.0e-6
    q = torch.tensor([2.0 - eps], dtype=torch.float64)
    f, df = kernels.shape(q, "wendland")
    assert float(f.abs().max()) < 1.0e-22
    assert float(df.abs().max()) < 2.0e-18
    # the (1 - q/2)^4 profile guarantees f ~ eps^4 and f' ~ eps^3
    q2 = torch.tensor([2.0 - 0.5 * eps], dtype=torch.float64)
    f2, df2 = kernels.shape(q2, "wendland")
    assert float(f2.abs().max()) == pytest.approx(16.0 * float(f.abs().max()),
                                                  rel=1.0e-3)
    assert float(df2.abs().max()) == pytest.approx(8.0 * float(df.abs().max()),
                                                   rel=1.0e-3)


def test_self_contribution_is_f_one():
    h = torch.tensor(0.013, dtype=torch.float64)
    for dim in (1, 2, 3):
        w0 = float(kernels.self_contribution(h, dim, "wendland"))
        expected = float(kernels.normalisation(dim, "wendland") / h ** dim)
        assert w0 == pytest.approx(expected, rel=1.0e-14)


# --------------------------------------------------------------------------- #
# 2. gradient: finite differences, antisymmetry, no r = 0 singularity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_gradient_matches_finite_difference(dim):
    h = 0.02
    delta = torch.tensor([[0.013, 0.004, 0.002][:dim],
                          [-0.021, 0.007, -0.003][:dim],
                          [0.030, -0.011, 0.005][:dim]], dtype=torch.float64)
    _, grad = kernels.kernel_pairs(delta, h, dim, "wendland")
    eps = 1.0e-7
    fd = torch.zeros_like(grad)
    for a in range(dim):
        shift = torch.zeros(dim, dtype=torch.float64)
        shift[a] = eps
        w_plus = kernels.kernel_value(
            torch.linalg.vector_norm(delta + shift, dim=1), h, dim, "wendland")
        w_minus = kernels.kernel_value(
            torch.linalg.vector_norm(delta - shift, dim=1), h, dim, "wendland")
        fd[:, a] = (w_plus - w_minus) / (2.0 * eps)
    assert torch.allclose(grad, fd, rtol=1.0e-5, atol=1.0e-2)


def test_gradient_is_antisymmetric_and_finite_at_the_origin():
    delta = torch.tensor([[0.01, -0.02], [-0.01, 0.02], [0.0, 0.0]],
                         dtype=torch.float64)
    _, g = kernels.kernel_pairs(delta, 0.02, 2, "wendland")
    _, g_neg = kernels.kernel_pairs(-delta, 0.02, 2, "wendland")
    assert torch.allclose(g, -g_neg)
    assert torch.equal(g[2], torch.zeros(2, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# 3. cross-check against the two independent Wendland implementations in-tree
# --------------------------------------------------------------------------- #
def test_2d_agrees_with_droplet_impact_implementation():
    from tensorsph.droplet_impact import (
        wendland_c2_gradient, wendland_c2_kernel)
    h = 0.021
    r = torch.tensor([0.0, 0.004, 0.017, 0.040, 0.042], dtype=torch.float64)
    ref = wendland_c2_kernel(r, h)
    got = kernels.kernel_value(r, h, 2, "wendland")
    assert torch.allclose(got, ref, rtol=1.0e-14, atol=0.0)
    delta = torch.tensor([[0.011, -0.006], [-0.025, 0.008]], dtype=torch.float64)
    _, got_g = kernels.kernel_pairs(delta, h, 2, "wendland")
    ref_g = wendland_c2_gradient(delta, h)
    assert torch.allclose(got_g, ref_g, rtol=1.0e-13, atol=1.0e-14)


def test_3d_agrees_with_threed_implementation():
    from tensorsph.threed.kernel3 import (
        wendland_c2_gradient, wendland_c2_kernel)
    h = 0.019
    r = torch.tensor([0.0, 0.005, 0.021, 0.038], dtype=torch.float64)
    ref = wendland_c2_kernel(r, h)
    got = kernels.kernel_value(r, h, 3, "wendland")
    assert torch.allclose(got, ref, rtol=1.0e-14, atol=0.0)
    delta = torch.tensor([[0.009, -0.004, 0.006], [-0.030, 0.011, -0.008]],
                         dtype=torch.float64)
    _, got_g = kernels.kernel_pairs(delta, h, 3, "wendland")
    ref_g = wendland_c2_gradient(delta, h)
    assert torch.allclose(got_g, ref_g, rtol=1.0e-13, atol=1.0e-14)


# --------------------------------------------------------------------------- #
# 4. the chain accepts the kernel: config, pair case, dam break
# --------------------------------------------------------------------------- #
def test_solver_config_accepts_the_kernel_and_halves_the_support():
    cfg = WCSPHConfig(dim=2, dx=0.01, kernel="wendland")
    cfg.validate()
    assert cfg.support == pytest.approx(2.0 * cfg.h0)


def test_particle_pair_conserves_momentum_with_the_new_kernel():
    run = particle_pair(v0=0.5, c0=20.0, steps=400, kernel="wendland")
    mom0 = run.momentum[0]
    worst = max(float((m - mom0).abs().max()) for m in run.momentum)
    assert worst < 1.0e-10
    assert all(math.isfinite(float(m.abs().sum())) for m in run.momentum)
    assert run.separation[-1] > run.separation[0]  # they bounced, not merged


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_dam_break_front_is_kernel_independent():
    """The same physics must come out of the spline and the Wendland kernel.

    A coarse, short dam break (dx = 0.02 m, t = 0.20 s) is run with both
    kernels; the leading-edge position at t = 0.15 s is compared.  The measured
    spread is reported in ``docs/wendland_wcsph_kernel_m14479.md``; the bound
    here (10 %) is the gate, not the measurement.
    """
    def run(kind: str):
        cfg = DamBreakConfig(column_width=0.1, column_height=0.2,
                             tank_length=1.0, tank_height=0.3, dx=0.02,
                             c0=15.0, alpha_av=0.1, kernel=kind,
                             t_end=0.2, sample_times=(0.15,))
        return dam_break(cfg)

    spline = run("quintic")
    wend = run("wendland")
    assert spline.simulated_time >= 0.19 and wend.simulated_time >= 0.19
    for r in (spline, wend):
        assert r.max_courant <= 0.5 + 1.0e-9
        assert math.isfinite(r.max_compression)

    f_spline = front_at(spline.times, spline.front_toe, 0.15)
    f_wend = front_at(wend.times, wend.front_toe, 0.15)
    assert f_spline is not None and f_wend is not None
    assert f_spline > 0.1  # the front actually left the column
    assert abs(f_wend - f_spline) / f_spline < 0.10
