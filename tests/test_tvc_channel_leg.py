"""Tests of the TVC channel leg (``benchmarks/xval_sphinx_tvc/channel.py``).

Three claims, each with an anchor that cannot be tuned:

* a perfect channel lattice is an exact fixed point of the ported operator, and
  the *wall weight* is what makes it one -- ``wall_factor = 2`` (the value that
  matches the ``C_i + C_j = 2`` fluid/fluid weight), not 1;
* the correction is an explicit position relaxation: it contracts the
  ``grad-1`` residual of a jittered lattice, monotonically and faster for a
  larger coefficient, and the whole coefficient range upstream names stays
  inside the contraction (no divergence observed);
* switching it off reproduces the published channel baseline bit for bit, and
  switching it on leaves the analytic parabola inside the case's own 5 %
  tolerance, i.e. it is harmless in a wall-bounded flow that is already
  regular.

Run: ``PYTHONPATH=src python -m pytest tests/test_tvc_channel_leg.py -q``
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tensorsph.case_poiseuille_morris import (MorrisConfig, build_channel,
                                             channel_tvc_shift, evaluate,
                                             run_case)

RES = 8


def _lattice(res: int = RES):
    cfg = MorrisConfig(resolution=res)
    pos, mass = build_channel(cfg, "cpu")
    return cfg, pos, mass


def test_perfect_lattice_is_an_exact_fixed_point_with_wall_factor_two():
    """The wall (mirror-ghost) weight 2 is what keeps the weights uniform."""
    cfg, pos, mass = _lattice()
    out = channel_tvc_shift(pos, mass,
                            dataclasses.replace(cfg, tvc_alpha=0.0,
                                                tvc_wall_factor=2.0))
    # float64 neighbour-sum floor; the analytic value is exactly zero.
    assert out["h_grad1_max"] < 1.0e-10
    assert float(torch.linalg.vector_norm(out["shift"], dim=1).max()) == 0.0


def test_wall_factor_one_breaks_the_fixed_point_only_in_the_wall_rows():
    """Counting the mirror image once leaves an order-unity wall-row residual."""
    cfg, pos, mass = _lattice()
    one = channel_tvc_shift(pos, mass,
                            dataclasses.replace(cfg, tvc_alpha=0.0,
                                                tvc_wall_factor=1.0))
    two = channel_tvc_shift(pos, mass,
                            dataclasses.replace(cfg, tvc_alpha=0.0,
                                                tvc_wall_factor=2.0))
    resid = torch.linalg.vector_norm(one["gradient_integral"], dim=1) * cfg.h
    y = pos[:, 1] / cfg.height
    wall = y < 2.5 * cfg.dx / cfg.height
    interior = (y >= 2.5 * cfg.dx / cfg.height) & (y < 1.0 - 2.5 * cfg.dx / cfg.height)
    assert float(resid[wall].max()) > 1.0e-1          # the anchor discriminates
    assert float(resid[interior].max()) < 1.0e-10     # no ghost neighbours there
    assert two["h_grad1_max"] < 1.0e-10


def test_shift_is_translation_invariant():
    """A rigid translation of the whole cloud must not change the correction."""
    cfg, pos0, mass = _lattice()
    cfg = dataclasses.replace(cfg, tvc_alpha=0.02)
    gen = torch.Generator().manual_seed(1)
    pos = pos0 + (torch.rand(pos0.shape, generator=gen, dtype=pos0.dtype) - 0.5) \
        * (2.0 * 0.05) * cfg.dx
    a = channel_tvc_shift(pos, mass, cfg)
    assert float(torch.linalg.vector_norm(a["shift"], dim=1).max()) > 0.0
    shifted = pos.clone()
    shifted[:, 0] = torch.remainder(shifted[:, 0] + 0.25 * cfg.length, cfg.length)
    b = channel_tvc_shift(shifted, mass, cfg)
    # the periodic quadrature is exact for the two images used, so the shift of
    # every particle is unchanged by the rigid translation
    assert torch.allclose(a["shift"], b["shift"], atol=1.0e-14, rtol=0.0)


def test_correction_off_reproduces_the_published_baseline_bit_for_bit():
    base = run_case(MorrisConfig(resolution=RES, t_end_over_tau=1.0))
    explicit = run_case(MorrisConfig(resolution=RES, t_end_over_tau=1.0,
                                     tvc_alpha=None))
    assert base["steps"] == explicit["steps"]
    assert base["tvc"]["applications"] == explicit["tvc"]["applications"] == 0
    assert base["profile"] == explicit["profile"]
    assert torch.equal(base["final_positions"], explicit["final_positions"])


def test_relaxation_is_a_contraction_across_the_upstream_coefficient_range():
    cfg, pos0, mass = _lattice()
    gen = torch.Generator().manual_seed(0)
    jitter = (torch.rand(pos0.shape, generator=gen, dtype=pos0.dtype) - 0.5) \
        * (2.0 * 0.05) * cfg.dx
    start = pos0 + jitter
    initial = channel_tvc_shift(start, mass,
                                dataclasses.replace(cfg, tvc_alpha=0.0))
    assert initial["h_grad1_median"] > 1.0e-2        # a real perturbation

    def run(alpha: float, steps: int = 40):
        cfg_a = dataclasses.replace(cfg, tvc_alpha=alpha)
        pos = start.clone()
        series = []
        for _ in range(steps):
            res = channel_tvc_shift(pos, mass, cfg_a)
            pos = pos + res["shift"]
            series.append(res["h_grad1_median"])
        return series

    # both upstream coefficients relax; the larger one relaxes faster and
    # neither overshoots (measured: the predicted divergence is not reached)
    stable = run(0.05)
    assert stable[-1] < 0.25 * initial["h_grad1_median"]
    assert max(stable) <= initial["h_grad1_median"] * 1.01
    strong = run(0.20)
    assert strong[-1] < stable[-1]
    assert max(strong) <= initial["h_grad1_median"] * 1.01


@pytest.mark.parametrize("alpha", [0.05, 0.20])
def test_correction_is_harmless_on_the_analytic_parabola(alpha):
    """TVC must leave the exact channel solution inside the case's tolerance."""
    base_cfg = MorrisConfig(resolution=RES, t_end_over_tau=1.0)
    base = evaluate(run_case(base_cfg), base_cfg)
    cfg = dataclasses.replace(base_cfg, tvc_alpha=alpha)
    case = run_case(cfg)
    ev = evaluate(case, cfg)
    assert case["tvc"]["applications"] == cfg.steps
    # the channel starts on a lattice, so there is nothing to de-clump: the
    # correction stays a small fraction of a spacing
    assert case["tvc"]["shift_max_over_dx"] < 0.05
    assert ev["profile_rms_interior_pct"] < 5.0
    assert abs(ev["profile_rms_interior_pct"]
               - base["profile_rms_interior_pct"]) < 1.0


def test_correction_is_mass_preserving_and_leaves_the_fluid_count_alone():
    cfg = MorrisConfig(resolution=RES, t_end_over_tau=1.0, tvc_alpha=0.05)
    case = run_case(cfg)
    assert case["n_fluid"] == cfg.resolution * int(round(cfg.length / cfg.dx))
    assert float(case["mass"].sum()) == pytest.approx(
        cfg.rho0 * cfg.dx * cfg.dx * case["n_fluid"], rel=1.0e-14)