"""Cadence leg of ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

The upstream ``TransportVelocityCorrection::update()`` carries **no ``dt``**: it
adds ``coefficient * h_ref**2 * limiter(...) * g_i`` to the position on every
call, so the coefficient is a per-call displacement amplitude and the physical
correction *rate* depends on how often the call is made.  Upstream prescribes
``coefficient = 0.2`` for one call per convection step (dual-criteria scheme)
and "about 1/4" -- ``0.05`` -- for one call per acoustic step.  Those two
prescriptions can only describe the same physics if ``N`` calls with ``a / N``
reproduce one call with ``a``.

These tests pin that statement quantitatively:

* the one-call shift is exactly linear in ``alpha`` (same configuration), so the
  coefficient is a clean prefactor and not entangled with the kernel;
* the split-vs-single composition error is **second order in ``alpha``** and
  grows like ``(1 - 1/N)`` -- i.e. the equivalence of upstream's two
  coefficients is an approximation with a measured, non-vanishing first-order
  error.  The constants asserted below are the measured values of
  ``benchmarks/xval_sphinx_tvc/cadence_invariance.py`` (operator probe); they
  are pinned so a change of the operator's normalisation cannot pass silently.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

from tensorsph.transport_velocity_correction import (  # noqa: E402
    SPHINXSYS_DEFAULT_COEFFICIENT,
    SPHINXSYS_SINGLE_STEP_FRACTION,
    transport_velocity_correction,
)
from cadence_invariance import operator_composition  # noqa: E402

DX = 0.03


@pytest.fixture(scope="module")
def one_call():
    """A single correction call on the dam-break initial state."""
    from tensorsph.dam_break import (DamBreakConfig, make_column,  # noqa: E402
                                    wall_lattice)
    from tensorsph.droplet_impact import lattice_mass_unit  # noqa: E402

    cfg = DamBreakConfig(dx=DX, column_width=0.2, column_height=0.4)
    device = torch.device("cpu")
    fluid, mass = make_column(cfg, device)
    wall = wall_lattice(cfg, device, top=cfg.tank_top())
    pos = torch.cat([fluid, wall], dim=0)
    volume = torch.cat((
        mass / lattice_mass_unit(cfg) * DX ** fluid.shape[1],
        torch.full((wall.shape[0],), DX ** fluid.shape[1], dtype=fluid.dtype)))
    h = cfg.h_factor * DX

    def call(alpha):
        return transport_velocity_correction(pos, volume, h, alpha=alpha,
                                             wall_count=int(wall.shape[0]),
                                             h_ref=h)

    return {"cfg": cfg, "pos": pos, "volume": volume, "h": h,
            "n_wall": int(wall.shape[0]), "call": call}


def test_shift_is_exactly_linear_in_alpha(one_call):
    """``alpha`` is a pure prefactor: one call at ``2a`` = 2 x one call at ``a``."""
    a = one_call["call"](0.05).shift
    b = one_call["call"](0.10).shift
    assert torch.allclose(b, 2.0 * a, rtol=0.0, atol=1e-12)
    # ... and the shift is not identically zero (a free surface has g != 0).
    assert float(a.norm(dim=-1).max()) > 0.0


def test_alpha_none_is_the_upstream_single_step_coefficient(one_call):
    none = one_call["call"](None).shift
    explicit = one_call["call"](
        SPHINXSYS_DEFAULT_COEFFICIENT * SPHINXSYS_SINGLE_STEP_FRACTION).shift
    assert torch.equal(none, explicit)
    assert math.isclose(SPHINXSYS_SINGLE_STEP_FRACTION, 0.25)


def test_split_calls_are_not_equivalent_single_calls(one_call):
    """``N`` calls of ``a/N`` differ from one call of ``a`` -- measured, not 0.

    If this ever became exactly zero the coefficient would be a rate
    coefficient and upstream's per-cadence table would be redundant; the
    operator is a position correction, so it must not be.
    """
    r1 = operator_composition(DX, 0.2, 1, "none")
    r4 = operator_composition(DX, 0.2, 4, "none")
    assert r1["composition_error_mean_over_shift"] == pytest.approx(0.0, abs=1e-14)
    assert r4["composition_error_mean_over_shift"] > 0.10


def test_composition_error_is_second_order_in_alpha():
    """Error / shift ~ alpha (a second-order position error over a first-order
    shift), so doubling ``alpha`` doubles the relative error."""
    e05 = operator_composition(DX, 0.05, 4, "none")["composition_error_mean_over_shift"]
    e10 = operator_composition(DX, 0.10, 4, "none")["composition_error_mean_over_shift"]
    e20 = operator_composition(DX, 0.20, 4, "none")["composition_error_mean_over_shift"]
    assert 0.02 < e05 < 0.06
    assert e10 / e05 == pytest.approx(2.0, rel=0.05)
    assert e20 / e10 == pytest.approx(2.0, rel=0.05)


def test_composition_error_saturates_like_one_minus_one_over_n():
    """``Phi_{a/N}^N`` differs from ``Phi_a`` by ``(a^2/2)(1 - 1/N) F'F``.

    Ratio ``err(N=4)/err(N=8)`` must therefore be ``(3/4)/(7/8) = 0.857``.
    """
    e4 = operator_composition(DX, 0.10, 4, "none")["composition_error_mean_over_shift"]
    e8 = operator_composition(DX, 0.10, 8, "none")["composition_error_mean_over_shift"]
    assert e4 / e8 == pytest.approx((1.0 - 0.25) / (1.0 - 0.125), rel=0.03)


def test_truncated_linear_limiter_damps_but_does_not_change_the_law():
    """The shipped limiter attenuates the shift a little (noise floor); the
    composition law is unchanged because the limiter is still smooth there."""
    r = operator_composition(DX, 0.10, 1, "truncated_linear")
    base = operator_composition(DX, 0.10, 1, "none")
    assert 0.9 < r["mean_shift_over_h"] / base["mean_shift_over_h"] < 1.0
    e4_lim = operator_composition(DX, 0.10, 4, "truncated_linear")
    e8_lim = operator_composition(DX, 0.10, 8, "truncated_linear")
    ratio = e4_lim["composition_error_mean_over_shift"] / \
        e8_lim["composition_error_mean_over_shift"]
    assert ratio == pytest.approx((1.0 - 0.25) / (1.0 - 0.125), rel=0.05)


def test_rate_matched_dam_break_runs_are_reproducible_and_distinct():
    """End-to-end pin: the rate-matched cadences agree, and reproducibly so.

    ``alpha = 0.05`` every acoustic step and ``alpha = 0.20`` every 4th step
    carry the same rate (4 x 0.05 = 0.20), so INV says their front trajectories
    coincide; they must coincide *within the predeclared gates* -- not
    identically, because the operator composition is only first order -- and the
    recorded distance must be reproduced bit for bit (deterministic solver).
    """
    import json

    from cadence_invariance import curve_distance
    from xval_sphinx_tvc.run import run_variant

    payload_path = ROOT / "docs/xval_sphinx_tvc/data/cadence_invariance.json"
    dx = 0.03
    runs = {}
    for name, variant in (
            ("s005_e1", dict(tvc_every=1, tvc_alpha=0.05, tvc_scope="all")),
            ("d020_e4", dict(tvc_every=4, tvc_alpha=0.20, tvc_scope="all"))):
        r = run_variant(name, variant, dx, 0.5)
        runs[name] = r
    d = curve_distance(runs["s005_e1"], runs["d020_e4"])
    # INV: same rate => same trajectory, inside the predeclared gates ...
    assert d["inv_abs_pass"] and d["inv_rel_pass"], d
    # ... but not identically: the composition error is first order in alpha,
    # so a bit-identical collapse would mean the operator lost its shift
    # amplitude dependence.
    assert d["mean_abs_dz_over_a"] > 1e-6
    if payload_path.exists():
        payload = json.loads(payload_path.read_text())
        rec = payload["analysis"][str(dx)]["rate_matched_pair"]
        assert d["mean_abs_dz_over_a"] == pytest.approx(
            rec["mean_abs_dz_over_a"], rel=1e-6)
        assert payload["analysis"][str(dx)]["k_star_variant"] == \
            min(payload["analysis"][str(dx)]["pairs_vs_s005_e1"],
                key=lambda k: payload["analysis"][str(dx)]["pairs_vs_s005_e1"][k]
                ["mean_abs_dz_over_a"])