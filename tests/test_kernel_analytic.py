"""Analytic verification of the SPH kernels: cubic spline and Wendland C2.

This module hardens the *analytic* verification surface of the four shipped
radial kernels -- the 2-D and 3-D cubic splines and the 2-D and 3-D Wendland
C2 -- with three groups of assertions.  Every number asserted here is either
an exact closed-form value or produced by quadrature at run time; nothing is
copied from a previous run of this code (zero hand-transcription).

Groups
------
(1) Normalisation / compact support / symmetry
    * ``int W dV = 1`` by radial quadrature (trapezoid, fine grid);
    * ``W = 0`` for ``r >= 2h`` and ``W > 0`` inside the support;
    * the gradient is antisymmetric, ``grad W(delta) = -grad W(-delta)``, and
      its magnitude depends on ``|delta|`` only (radial kernel).
(2) Gradient vs central difference at analytic points
    * ``|grad_analytic - central_diff| < 1e-6`` (relative) at several points
      inside the support, on both polynomial branches.
(3) 1-D linear density field kernel interpolation
    * the self-normalising interpolation ``f_hat = sum f_j W / sum W``
      reproduces ``f(x) = a + b x`` to ``< 1 %`` (zeroth + first moment
      consistency).  The cubic spline's 1-D restriction is a cubic B-spline and
      reproduces linear fields to machine precision; the Wendland C2 carries a
      small ``O((dx/h)^2)`` lattice error, still far below the threshold.

Pre-declared thresholds (fixed before inspecting results)
---------------------------------------------------------
``NORM_TOL = 1e-9``   -- |int W dV - 1|
``GRAD_TOL = 1e-6``   -- relative, analytic gradient vs central difference
``LINEAR_TOL = 1e-2`` -- relative, 1-D linear field reproduction
"""

import math

import pytest
import torch

from tensorsph import (
    cubic_spline_gradient,
    cubic_spline_kernel,
    wendland_c2_gradient,
    wendland_c2_kernel,
)
from tensorsph.threed import (
    cubic_spline_gradient as cubic_spline_gradient_3d,
    cubic_spline_kernel as cubic_spline_kernel_3d,
    wendland_c2_gradient as wendland_c2_gradient_3d,
    wendland_c2_kernel as wendland_c2_kernel_3d,
)

# --- pre-declared thresholds (see module docstring) -------------------------
NORM_TOL = 1.0e-9
GRAD_TOL = 1.0e-6
LINEAR_TOL = 1.0e-2

DT = torch.float64

# (name, W(r,h), grad(delta,h), dim) for the four kernels under test.
KERNELS = [
    ("cubic_spline_2d", cubic_spline_kernel, cubic_spline_gradient, 2),
    ("wendland_c2_2d", wendland_c2_kernel, wendland_c2_gradient, 2),
    ("cubic_spline_3d", cubic_spline_kernel_3d, cubic_spline_gradient_3d, 3),
    ("wendland_c2_3d", wendland_c2_kernel_3d, wendland_c2_gradient_3d, 3),
]
KERNEL_IDS = [k[0] for k in KERNELS]


def _radial_integral(W, h: float, dim: int, n: int = 200_000) -> float:
    """``int W dV = S_{d-1} int_0^{2h} W r^{d-1} dr`` by the trapezoid rule.

    Both endpoints contribute zero (``W(2h) = 0`` and ``r^{d-1} = 0`` at the
    origin for ``d >= 2``), so a plain sum times ``dr`` is the trapezoid.
    """
    r = torch.linspace(0.0, 2.0 * h, n, dtype=DT)
    w = W(r, h)
    dr = float(r[1] - r[0])
    if dim == 2:
        return float((2.0 * math.pi * w * r).sum() * dr)
    return float((4.0 * math.pi * w * r.square()).sum() * dr)


# --------------------------------------------------------------------------- #
# (1) Normalisation / compact support / symmetry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,W,G,dim", KERNELS, ids=KERNEL_IDS)
def test_kernel_normalises_to_one(name, W, G, dim):
    """``int W dV = 1`` over the 2h ball, for several smoothing lengths."""
    for h in (0.01, 0.05, 0.13):
        integral = _radial_integral(W, h, dim)
        assert abs(integral - 1.0) < NORM_TOL, (name, h, integral)


@pytest.mark.parametrize("name,W,G,dim", KERNELS, ids=KERNEL_IDS)
def test_kernel_is_compact_and_positive_inside_support(name, W, G, dim):
    h = 0.05
    r_in = torch.linspace(0.0, 2.0 * h - 1.0e-9, 200, dtype=DT)
    w_in = W(r_in, h)
    assert bool((w_in >= 0.0).all())
    assert float(w_in.max()) > 0.0
    # exactly at the support edge and beyond: identically zero
    r_out = torch.tensor([2.0 * h, 2.0 * h + 1.0e-6, 3.0 * h, 5.0 * h], dtype=DT)
    assert float(W(r_out, h).abs().max()) == 0.0


@pytest.mark.parametrize("name,W,G,dim", KERNELS, ids=KERNEL_IDS)
def test_gradient_is_antisymmetric_and_radial(name, W, G, dim):
    """``grad W(delta) = -grad W(-delta)`` and ``|grad W|`` depends on ``|delta|`` only."""
    h = 0.05
    d = torch.zeros(dim, dtype=DT)
    d[0] = 0.07
    d[1 % dim] = 0.03
    g1 = G(d.unsqueeze(0), h)[0]
    g2 = G((-d).unsqueeze(0), h)[0]
    assert float((g1 + g2).abs().max()) < 1.0e-12
    # a different direction with the same |delta| gives the same |grad|
    d_rot = torch.zeros(dim, dtype=DT)
    d_rot[0] = 0.03
    d_rot[1 % dim] = 0.07
    g3 = G(d_rot.unsqueeze(0), h)[0]
    assert abs(float(g1.norm()) - float(g3.norm())) < 1.0e-12


# --------------------------------------------------------------------------- #
# (2) Gradient vs central difference at analytic points
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,W,G,dim", KERNELS, ids=KERNEL_IDS)
def test_gradient_matches_central_difference(name, W, G, dim):
    """Analytic gradient vs central difference of ``W``: relative error < 1e-6."""
    h = 0.1
    eps = 1.0e-5
    for q in (0.5, 0.75, 1.25, 1.75):
        d = q * h
        wp = float(W(torch.tensor(d + eps, dtype=DT), h))
        wm = float(W(torch.tensor(d - eps, dtype=DT), h))
        cd = (wp - wm) / (2.0 * eps)
        delta = torch.zeros(dim, dtype=DT)
        delta[0] = d
        g = float(G(delta.unsqueeze(0), h)[0, 0])
        assert abs(g - cd) < GRAD_TOL * max(abs(g), abs(cd), 1.0e-300), (name, q, g, cd)


# --------------------------------------------------------------------------- #
# (3) 1-D linear density field kernel interpolation
# --------------------------------------------------------------------------- #
def _interp_linear_1d(W, h_over_dx: float, offset: float,
                      a: float = 1.0, b: float = 2.0, x0: float = 5.0):
    """Self-normalising SPH interpolation of ``f(x) = a + b x`` on a 1-D lattice.

    Particles sit on a uniform lattice of spacing ``dx``; the evaluation point
    is offset by ``offset * dx`` from a lattice node so the support window is
    not symmetric about it.  Uniform particle masses cancel in the ratio.
    """
    dx = 0.01
    h = h_over_dx * dx
    x = x0 + offset * dx
    jmin = math.floor((x - 2.0 * h) / dx) - 1
    jmax = math.ceil((x + 2.0 * h) / dx) + 1
    xs = torch.arange(jmin, jmax + 1, dtype=DT) * dx
    r = (xs - x).abs()
    w = W(r, h)
    f = a + b * xs
    return float((f * w).sum() / w.sum()), a + b * x


@pytest.mark.parametrize("name,W,G,dim",
                         [k for k in KERNELS if k[3] == 2],
                         ids=[k[0] for k in KERNELS if k[3] == 2])
def test_linear_density_field_is_reproduced(name, W, G, dim):
    """A 1-D linear field is reproduced by kernel interpolation to < 1 %."""
    h_over_dx = 4.0
    worst = 0.0
    for offset in (0.0, 0.13, 0.37, 0.5, 0.62, 0.73, 0.91):
        val, exact = _interp_linear_1d(W, h_over_dx, offset)
        worst = max(worst, abs(val - exact) / abs(exact))
    assert worst < LINEAR_TOL, (name, worst)
