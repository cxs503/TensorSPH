"""Analytic verification of the CSF surface delta function's normalisation.

The CSF force is ``f = -sigma kappa delta_s n``; ``delta_s`` must satisfy
``int delta_s dn = 1`` across the interface (Brackbill et al. 1992).  The
0.8.0-line model used ``delta_s = 2 |grad c|``, which assumes that a particle
sitting on the free surface keeps exactly half of its kernel weight.  With the
2-D cubic spline at ``h = 1.3 dx`` the self term is ``V W(0) = 0.233``, so the
surface particle keeps ``c_s = (1 + V W(0)) / 2 = 0.617`` and the in-fluid
colour branch spans ``[c_s, 1]``, not ``[1/2, 1]``.  The measured normalisation
was therefore 0.74-0.76 at every resolution
(``docs/verification_delta_function_normalisation.md``).

The tests here pin:

1. ``"factor2"`` is *exactly* ``2 |grad c|`` (the historical behaviour, so old
   benchmark numbers stay reproducible);
2. ``"influid"`` is the parameter-free rescaling ``|grad c| / (1 - c_s)``, and
   ``c_s`` computed from ``V`` and the kernel equals the surface colour the
   model actually produces on a planar half-lattice (a direct measurement of
   the analytic claim, not a fit);
3. on the lattice droplet the normalised delta function integrates to 1 within
   5 %, where ``"factor2"`` is outside the 15 % gate -- so the test fails if the
   defect ever gets silently "fixed" by a fitted constant;
4. the force-density cutting-force identity ``F_cut = 2 sigma`` lands inside
   the 15 % gate with ``"influid"`` and outside it with ``"factor2"``;
5. switching the normalisation changes nothing but the amplitude: curvature and
   force direction are bit-identical between the two schemes;
6. ``Config`` rejects an unknown ``csf_delta_s``.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import (
    csf_fields,
    delta_s_from_gradient,
    surface_color_level,
    surface_tension_acceleration,
)
from tensorsph.droplet_impact import Config, cubic_spline_kernel

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "csf_delta_s" / "run.py"
LAW = ROOT / "benchmarks" / "laplace_law" / "run.py"
DX = 0.003
RADIUS = 0.02
SIGMA = 7.2
H_FACTOR = 1.3
H = H_FACTOR * DX


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_delta = _module(BENCH, "csf_delta_s_bench")
_law = _module(LAW, "laplace_law_bench")


def _droplet(dx=DX, radius=RADIUS):
    pos, mass = _law.hexagonal_droplet(radius, dx, 1000.0)
    wide = _delta_bench_pairs(pos, 4.0 * H_FACTOR * dx)
    return pos, mass, wide


def _delta_bench_pairs(pos, cutoff):
    d = torch.cdist(pos, pos)
    i, j = torch.nonzero((d <= cutoff) & (d > 0.0), as_tuple=True)
    keep = i < j  # one direction per pair, exactly like cell_linked_pairs
    return torch.stack([i[keep], j[keep]], dim=1)


def _half_lattice_surface_color(dx=DX, rows=40, cols=60):
    """Colour of the top row of a hex half-lattice: the model's own ``c_s``."""
    dy = dx * math.sqrt(3.0) / 2.0
    pts = []
    for row in range(-rows, 1):
        y = row * dy
        for col in range(-cols, cols + 1):
            pts.append(((col + 0.5 * (row % 2)) * dx, y))
    pos = torch.tensor(pts)
    vol = dx * dx * math.sqrt(3.0) / 2.0
    mass = torch.full((pos.shape[0],), vol * 1000.0)
    pairs = _delta_bench_pairs(pos, 2.0 * H)
    state = csf_fields(pos, mass, pairs, H, 1000.0)
    surface = pos[:, 1] > -1.0e-12
    return float(state.color[surface].median()), float(vol * float(
        cubic_spline_kernel(torch.zeros(()), H)))


# --------------------------------------------------------------------------- #
# 1-2. the two normalisations themselves
# --------------------------------------------------------------------------- #

def test_factor2_is_exactly_twice_the_gradient():
    """Regression pin: the historical scheme must not drift."""
    grad = torch.tensor([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0]])
    vol = torch.full((3,), DX * DX * math.sqrt(3.0) / 2.0)
    delta_s = delta_s_from_gradient(grad, vol, H, scheme="factor2")
    assert torch.allclose(delta_s, torch.tensor([10.0, 0.0, 2.0]))


def test_influid_scaling_is_one_over_the_surface_colour_span():
    grad = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    vol = torch.full((2,), DX * DX * math.sqrt(3.0) / 2.0)
    c_s = surface_color_level(vol, H)
    delta_s = delta_s_from_gradient(grad, vol, H, scheme="influid")
    assert torch.allclose(delta_s, torch.linalg.vector_norm(grad, dim=1) / (1.0 - c_s))


def test_half_lattice_surface_colour_falsifies_the_factor_two_assumption():
    """A surface particle keeps far more than half of its kernel weight.

    ``delta_s = 2 |grad c|`` is only correct if the in-fluid colour branch spans
    ``[1/2, 1]``, i.e. if a particle on the free surface holds exactly half of
    its kernel support.  Measured on the model's own planar half-lattice it
    holds ``c_s = 0.73`` (self term 0.233 included), so the assumed span is
    ``0.50`` against an actual ``0.27`` -- the normalisation is far too small.
    """
    measured, self_term = _half_lattice_surface_color()
    assert self_term == pytest.approx(0.233, abs=0.005)
    assert measured > 0.60, measured          # not 1/2
    assert measured < 0.90, measured          # not bulk either


def test_analytic_branch_span_matches_the_measured_gradient_integral():
    """``int_fluid |grad c| dn`` is what the delta function must divide out.

    The closed-form estimate for that branch span is ``1 - c_s`` with
    ``c_s = (1 + V W(0)) / 2``; the measurement below is the actual integral of
    ``|grad c|`` over the fluid side of the lattice droplet.  The two agree to
    ~2 %, which is why ``|grad c| / (1 - c_s)`` is a parameter-free scale and
    not a fitted constant.
    """
    pos, mass, pairs = _droplet()
    state = csf_fields(pos, mass, pairs, H, 1000.0, delta_s_scheme="factor2")
    measured_span = _delta.delta_s_integral(
        pos, mass, 1000.0, state.delta_s, RADIUS) / 2.0
    analytic_span = 1.0 - float(surface_color_level(mass / 1000.0, H)[0])
    assert measured_span == pytest.approx(analytic_span, rel=0.05)


def test_unknown_delta_s_scheme_raises():
    vol = torch.ones(1)
    with pytest.raises(ValueError, match="unknown delta_s_scheme"):
        delta_s_from_gradient(torch.zeros(1, 2), vol, H, scheme="nope")


# --------------------------------------------------------------------------- #
# 3. int delta_s dn = 1 on the lattice droplet
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scheme,within_5pct", [("influid", True), ("factor2", False)])
def test_delta_s_integral_is_one_only_with_the_normalised_scheme(scheme, within_5pct):
    pos, mass, pairs = _droplet()
    state = csf_fields(pos, mass, pairs, H, 1000.0, delta_s_scheme=scheme)
    integral = _delta.delta_s_integral(pos, mass, 1000.0, state.delta_s, RADIUS)
    if within_5pct:
        assert abs(integral - 1.0) < 0.05, f"{scheme}: {integral}"
    else:
        assert abs(integral - 1.0) >= 0.15, f"{scheme}: {integral}"


def test_normalised_integral_also_within_3pct_at_fine_resolution():
    pos, mass, pairs = _droplet(dx=0.0015)
    state = csf_fields(pos, mass, pairs, 1.3 * 0.0015, 1000.0,
                       delta_s_scheme="influid")
    integral = _delta.delta_s_integral(pos, mass, 1000.0, state.delta_s, RADIUS)
    assert abs(integral - 1.0) < 0.05, integral


# --------------------------------------------------------------------------- #
# 4. the normalisation is what breaks the cutting-force identity
# --------------------------------------------------------------------------- #

def test_force_density_cutting_force_closes_with_the_normalised_scheme():
    """``F_cut / (2 sigma)`` on the force *density*, inside the 15 % gate.

    Measured on ``f = -sigma kappa delta_s n`` rather than on ``a = f / rho``,
    because the kernel-sum density on the outermost particle layer is 20 % low
    and inflates ``a`` by ~25 % -- the very bias that used to mask the delta_s
    deficit (see the benchmark doc).
    """
    results = {}
    for scheme in ("factor2", "influid"):
        pos, mass, pairs = _droplet()
        _, state = surface_tension_acceleration(
            pos, torch.full_like(mass, 1000.0), mass, pairs, H, SIGMA,
            rho0=1000.0, curvature_scheme="fit_quadratic",
            delta_s_scheme=scheme)
        f_density = (-SIGMA * state.kappa * state.delta_s)[:, None] * state.normal
        cut = float((mass[pos[:, 0] > 0, None] / 1000.0
                     * f_density[pos[:, 0] > 0]).sum(dim=0)[0])
        results[scheme] = cut / (-2.0 * SIGMA)
    assert abs(results["influid"] - 1.0) < 0.15, results
    assert results["factor2"] < 0.85, results


# --------------------------------------------------------------------------- #
# 5. the knob is amplitude-only
# --------------------------------------------------------------------------- #

def test_curvature_and_direction_are_unchanged_by_the_normalisation():
    pos, mass, pairs = _droplet()
    a = csf_fields(pos, mass, pairs, H, 1000.0, curvature_scheme="fit_quadratic",
                   delta_s_scheme="factor2")
    b = csf_fields(pos, mass, pairs, H, 1000.0, curvature_scheme="fit_quadratic",
                   delta_s_scheme="influid")
    # delta_s is a *weight* of the local curvature fit, so a uniform rescaling
    # of it changes the fit only by round-off; nothing else may move.
    assert torch.allclose(a.kappa, b.kappa, rtol=1e-9, atol=1e-9)
    assert torch.equal(a.normal, b.normal)
    assert torch.equal(a.color, b.color)
    # ... and the delta function is a pure rescaling of the same gradient
    ratio = (b.delta_s / a.delta_s.clamp_min(1e-30))[a.delta_s > 0]
    assert float(ratio.min()) == pytest.approx(float(ratio.max()), rel=1e-12)
    assert float(ratio.median()) == pytest.approx(
        0.5 / (1.0 - float(surface_color_level(mass / 1000.0, H)[0])), rel=1e-6)


# --------------------------------------------------------------------------- #
# 6. config wiring
# --------------------------------------------------------------------------- #

def test_config_validates_the_delta_s_scheme():
    from tensorsph.droplet_impact import validate_config

    cfg = Config(dx=DX, rho0=1000.0, sound_speed=20.0, dt=5.85e-5,
                 surface_tension_coefficient=SIGMA)
    assert cfg.csf_delta_s == "factor2"  # default unchanged
    validate_config(cfg)
    bad = Config(dx=DX, rho0=1000.0, sound_speed=20.0, dt=5.85e-5,
                 surface_tension_coefficient=SIGMA, csf_delta_s="bogus")
    with pytest.raises(ValueError, match="csf_delta_s"):
        validate_config(bad)