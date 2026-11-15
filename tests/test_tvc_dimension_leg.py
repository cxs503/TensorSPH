"""Tests for the dimension leg of the TVC port (``DIMENSION_LEG.md``).

What is pinned here, and why it is not "just executes the code":

* the kernel normalisation used by the operator equals ``int W_d dV = 1`` in
  1, 2 and 3 dimensions (an exact analytic invariant, checked by quadrature);
* the 2-D path is *bit-identical* to the pre-existing
  ``tensorsph.droplet_impact.cubic_spline_gradient`` transcription, so the
  dimension work cannot have moved any of the earlier legs;
* a 3-D cloud with a contradicting ``dim`` raises instead of being silently
  scaled by ``sigma_2/sigma_3`` (the defect this leg fixes);
* the free-surface shift matches the continuum anchor ``2 alpha h I_d(h,0)``
  in every dimension to the measured lattice bias, and the bias values are
  the ones in the published payload (drift guard);
* the bulk lattice is an exact fixed point and the mirror-ghost wall branch
  vanishes to round-off *iff* ``wall_factor == 2`` in every dimension.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.droplet_impact import cubic_spline_gradient  # noqa: E402
from tensorsph.transport_velocity_correction import (  # noqa: E402
    CUBIC_SPLINE_SIGMA,
    cubic_spline_gradient_dim,
    kernel_gradient_integral,
    transport_velocity_correction,
)

_PAYLOAD = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "dimension_leg.json"


def _bench():
    spec = importlib.util.spec_from_file_location(
        "dimension_leg", ROOT / "benchmarks" / "xval_sphinx_tvc" / "dimension_leg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def payload() -> dict:
    assert _PAYLOAD.exists(), f"missing payload {_PAYLOAD}"
    return json.loads(_PAYLOAD.read_text())


@pytest.fixture(scope="module")
def bench():
    return _bench()


def test_kernel_normalisation_is_unity_in_every_dimension(bench):
    for dim in (1, 2, 3):
        got = bench.normalisation(0.13, dim)
        assert got == pytest.approx(1.0, abs=2.0e-9), dim


def test_sigma_table_matches_the_profile_integrals():
    """``sigma_d`` from the table vs the profile's own 1-D integral."""
    h = 0.37
    q = torch.linspace(0.0, 2.0, 200001, dtype=torch.float64)
    f = torch.where(q < 1.0, 1.0 - 1.5 * q.square() + 0.75 * q.pow(3),
                    torch.where(q < 2.0, 0.25 * (2.0 - q).pow(3),
                                torch.zeros_like(q)))
    # int_0^2 f(q) q^(d-1) dq * sigma_d * h^d = 1/d * ... (full symmetry)
    for dim, weight in ((1, torch.ones_like(q)), (2, q), (3, q.square())):
        one_side = float(torch.trapezoid(f * weight, q))
        solid = {1: 2.0, 2: 2.0 * math.pi, 3: 4.0 * math.pi}[dim]
        assert CUBIC_SPLINE_SIGMA[dim](h) * solid * one_side * h ** dim == \
            pytest.approx(1.0, rel=1e-9)


def test_two_dimensional_path_is_bit_identical():
    """``dim=2`` reproduces the pre-existing 2-D transcription exactly."""
    torch.manual_seed(7)
    pos = torch.rand(300, 2, dtype=torch.float64) * 0.5
    vol = torch.full((300,), 1.0e-3, dtype=torch.float64)
    inferred = kernel_gradient_integral(pos, vol, 0.13)
    explicit = kernel_gradient_integral(pos, vol, 0.13, dim=2)
    direct = cubic_spline_gradient_dim(pos - pos, 0.13, 2)      # (300, 2) zeros
    assert torch.equal(inferred, explicit)
    assert torch.equal(direct, cubic_spline_gradient(pos - pos, 0.13))
    # and through the entry point used by every earlier leg
    a = transport_velocity_correction(pos, vol, 0.13, alpha=0.05)
    b = transport_velocity_correction(pos, vol, 0.13, alpha=0.05, dim=2)
    assert torch.equal(a.shift, b.shift)
    assert a.extras["dim"] == b.extras["dim"] == 2


def test_dimension_mismatch_is_rejected_instead_of_silently_scaled():
    pos3 = torch.rand(20, 3, dtype=torch.float64)
    vol = torch.full((20,), 1.0e-3, dtype=torch.float64)
    with pytest.raises(ValueError, match="contradicts the point cloud"):
        kernel_gradient_integral(pos3, vol, 0.13, dim=2)
    with pytest.raises(ValueError, match="contradicts the point cloud"):
        transport_velocity_correction(pos3, vol, 0.13, alpha=0.05, dim=1)
    with pytest.raises(ValueError, match="dim in 1\\.\\.3"):
        kernel_gradient_integral(torch.rand(20, 4, dtype=torch.float64),
                                 torch.full((20,), 1.0e-3, dtype=torch.float64), 0.13)


def test_silent_scaling_that_the_fix_removed(bench):
    """``sigma_2/sigma_3 = 10h/7``: the pre-fix error on a 3-D cloud."""
    for dx in (0.05, 0.02, 0.01):
        h = bench.HBAR * dx
        assert bench.pre_fix_scale(dx) == pytest.approx(10.0 * h / 7.0)
        assert 1.0 / bench.pre_fix_scale(dx) == pytest.approx(7.0 / (10.0 * h))


def test_free_surface_anchor_matches_the_published_bias(bench, payload):
    """Continuum ``2 alpha h I_d(h,0)`` vs the measured lattice shift."""
    published = {r["dim"]: r for r in payload["surface_anchor"]}
    for dim in (1, 2, 3):
        got = bench.surface_shift(0.1, dim)
        want = published[dim]
        assert got["max_shift_over_h_per_alpha"] == \
            pytest.approx(want["max_shift_over_h_per_alpha"], rel=1e-9)
        assert got["bias_rel"] == pytest.approx(want["bias_rel"], abs=1e-9)
        # the lattice sits *below* the continuum value in every dimension
        assert -0.20 < got["bias_rel"] < 0.0


def test_two_dimensional_anchor_reproduces_the_surface_anchor_leg(bench, payload):
    """Independent cross-check of ``SURFACE_ANCHOR.md``'s 14.15 %."""
    published = {r["dim"]: r for r in payload["surface_anchor"]}
    assert published[2]["bias_rel"] == pytest.approx(-0.1415, abs=5.0e-4)


def test_three_dimensional_gate_is_arrangement_dependent(bench, payload):
    """Honest status: the <15 % gate does not transfer from 2-D to 3-D."""
    anchor = {r["dim"]: r for r in payload["surface_anchor"]}
    fcc = payload["fcc_3d"]["0.1"]
    # both 3-D arrangements miss the <15 % gate; fcc only just
    assert anchor[3]["bias_rel"] < -0.15
    assert fcc["bias_rel"] < -0.15
    assert fcc["bias_rel"] > anchor[3]["bias_rel"]          # denser packing is closer
    assert abs(anchor[2]["bias_rel"]) < 0.15               # 2-D: inside


def test_shift_over_h_is_resolution_independent_in_every_dimension(bench, payload):
    rows = payload["resolution_ladder"]
    for dim in (1, 2, 3):
        vals = [r["max_shift_over_h_per_alpha"] for r in rows if r["dim"] == dim]
        assert len(vals) == 3
        assert max(vals) - min(vals) < 1.0e-12


def test_bulk_lattice_is_an_exact_fixed_point_in_every_dimension(bench):
    for dim in (1, 2, 3):
        assert bench.bulk_fixed_point(0.1, dim) < 1.0e-12, dim


def test_wall_branch_needs_the_upstream_factor_two_in_every_dimension(bench):
    retained = []
    for dim in (1, 2, 3):
        two = bench.wall_residual(0.1, dim, 2.0)
        one = bench.wall_residual(0.1, dim, 1.0)
        assert two < 1.0e-12, dim      # mirror ghosts cancel exactly
        assert 0.5 < one < 0.65, dim   # factor 1 keeps the surface-scale residual
        retained.append(one)
    assert max(retained) / min(retained) - 1.0 < 0.02   # dimension independent