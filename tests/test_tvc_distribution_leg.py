"""Analytic-anchor tests for the TVC distribution leg.

The distribution leg is only worth something if its anchors are *exact*: the
metrics of ``benchmarks/xval_sphinx_tvc/distribution.py`` are defined so that a
perfect square lattice gives a known value (1 for ``nn_median_over_dx``, 0 for
the rest).  These tests pin that on constructed configurations whose metric
values are known analytically, so a broken metric cannot pass by producing
plausible-looking numbers on a simulation.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

from distribution import (ANCHORS, distribution_error,  # noqa: E402
                          lattice_metrics, lattice_reference)

DX = 0.02
H = 1.3 * DX


def square_lattice(n: int = 21, dx: float = DX, dtype=torch.float64,
                   h: float = H) -> torch.Tensor:
    xs = torch.arange(n, dtype=dtype) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def test_perfect_lattice_reproduces_every_anchor_to_round_off():
    pos = square_lattice()
    m = lattice_metrics(pos, DX, H)
    assert m["nn_median_over_dx"] == pytest.approx(1.0, abs=1e-12)
    assert m["nn_cv"] == pytest.approx(0.0, abs=1e-12)
    assert m["close_pair_fraction"] == 0.0
    assert m["bulk_grad1_mean_h"] == pytest.approx(0.0, abs=1e-12)
    assert m["bulk_concentration_error"] == pytest.approx(0.0, abs=1e-12)
    err = distribution_error(m)
    assert err["worst_relative"] == pytest.approx(0.0, abs=1e-12)


def test_metrics_are_invariant_under_uniform_scale_change():
    """A lattice is a lattice at any dx: the normalised metrics cannot move."""
    base = lattice_metrics(square_lattice(), DX, H)
    for scale in (0.37, 1.7, 5.0):
        m = lattice_metrics(square_lattice(dx=DX * scale, h=H * scale),
                            DX * scale, H * scale)
        for key in ("nn_median_over_dx", "nn_cv", "close_pair_fraction",
                    "bulk_grad1_mean_h", "bulk_concentration_error"):
            assert m[key] == pytest.approx(base[key], abs=1e-11), key


def test_bulk_subset_is_the_geometric_interior():
    """Full support is a geometric count, not a concentration threshold."""
    n = 21
    m = lattice_metrics(square_lattice(n), DX, H)
    lattice = lattice_reference(DX, H)
    # 2h = 2.6 dx -> every particle whose 5x5 neighbourhood fits has
    # (5 - 2)^2 = 9 ... (n - 4)^2 particles have all 20 lattice neighbours.
    assert lattice["neighbour_count"] == 20
    assert m["bulk_particles"] == (n - 4) ** 2
    assert m["bulk_fraction"] == pytest.approx((n - 4) ** 2 / n ** 2, abs=1e-12)


def test_uniform_compression_is_detected_and_matches_the_analytic_ratio():
    """Squeezing x by 0.6 makes every nearest neighbour 0.6 dx away."""
    pos = square_lattice()
    centre = pos[:, 0].mean()
    pos = pos.clone()
    pos[:, 0] = centre + (pos[:, 0] - centre) * 0.6
    m = lattice_metrics(pos, DX, H)
    assert m["nn_median_over_dx"] == pytest.approx(0.6, abs=1e-12)
    assert m["close_pair_fraction"] > 0.0
    assert m["nn_cv"] > 0.0                      # two spacings now, not one
    assert m["bulk_concentration_error"] > 0.3   # C = 1/0.6 > 1
    assert m["bulk_grad1_mean_h"] > 0.1          # anchor 0 is violated badly


def test_uniform_void_is_reported_as_loss_of_full_support():
    """Stretching x by 1.25 dilutes the cloud below the lattice neighbour count.

    ``r_nn`` stays at ``dx`` (the stretched direction is the longer one), so the
    nearest-neighbour metric alone cannot see the void; the full-support subset
    can, and it must report *no* anchor-carrying sample instead of a bogus zero.
    """
    pos = square_lattice()
    centre = pos[:, 0].mean()
    pos = pos.clone()
    pos[:, 0] = centre + (pos[:, 0] - centre) * 1.25
    m = lattice_metrics(pos, DX, H)
    assert m["nn_median_over_dx"] == pytest.approx(1.0, abs=1e-12)
    assert m["close_pair_fraction"] == 0.0
    # 12 lattice neighbours instead of 20 -> nothing carries the analytic anchor
    lattice = lattice_reference(DX, H)
    assert lattice["neighbour_count"] == 20
    assert m["bulk_particles"] == 0
    assert m["bulk_grad1_mean_h"] is None
    assert m["bulk_concentration_error"] is None


def test_single_clumped_pair_is_located():
    pos = square_lattice()
    # move one particle almost onto its neighbour
    pos = pos.clone()
    pos[1] = pos[0] + torch.tensor([1e-3 * DX, 0.0], dtype=pos.dtype)
    m = lattice_metrics(pos, DX, H)
    # exactly one *pair* out of the sampled pairs is below 0.7 dx
    assert m["close_pair_fraction"] == pytest.approx(
        1.0 / m["close_pair_sample"], rel=1e-9)
    assert m["nn_median_over_dx"] == pytest.approx(1.0, abs=1e-12)


def test_empty_samples_are_none_not_zero():
    m = lattice_metrics(torch.zeros((1, 2), dtype=torch.float64), DX, H)
    assert m["nn_median_over_dx"] is None
    assert m["bulk_grad1_mean_h"] is None
    assert distribution_error(m)["worst_relative"] is None


def test_error_dict_only_makes_relative_errors_where_the_anchor_is_nonzero():
    m = {"nn_median_over_dx": 1.15, "nn_cv": 0.2, "close_pair_fraction": 0.07}
    err = distribution_error(m)
    assert err["relative"] == {"nn_median_over_dx": pytest.approx(0.15)}
    assert err["absolute"]["nn_cv"] == pytest.approx(0.2)
    assert err["worst_relative"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# The ladder analysis (slope fits, gate bookkeeping)
# ---------------------------------------------------------------------------

def _record(dx: float, exponent: float, prefactor: float, variant: str) -> dict:
    """Synthetic ladder record following an exact power law."""
    nn = 1.0 + prefactor * dx ** exponent
    return {
        "variant": variant,
        "dx": dx,
        "metrics": {"nn_median_over_dx": nn, "nn_cv": 0.0,
                    "bulk_grad1_mean_h": 0.0, "close_pair_fraction": 0.0,
                    "bulk_fraction": 0.5},
        "error": distribution_error({"nn_median_over_dx": nn}),
        "front_score": {"mae": 0.0, "mean_rel_error": 0.0},
    }


def test_analysis_recovers_a_known_exponent_and_books_the_gate():
    from distribution_ladder import analyse

    # error(dx) = 0.158 * (dx / 0.01) ** -1.5 -> 15.8 % at the finest level
    records = [_record(dx, -1.5, 0.158 * 0.01 ** 1.5, "baseline")
               for dx in (0.04, 0.03, 0.02, 0.01)]
    out = analyse(records, gate=0.15)
    slope = out["slopes"]["baseline"]["worst_relative_vs_dx"]
    assert slope == pytest.approx(-1.5, rel=1e-9)
    per = out["by_variant"]["baseline"]
    # 1.98 % at dx=0.04 ... 15.8 % at dx=0.01, so only the finest level
    # crosses the 15 % gate -- the bookkeeping must say exactly that (levels
    # are reported in ascending dx).
    assert per["gate_pass_levels"] == [0.02, 0.03, 0.04]
    assert per["gate_fail_levels"] == [0.01]


# ---------------------------------------------------------------------------
# The published ladder, if it is present in this checkout
# ---------------------------------------------------------------------------

LADDER = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "distribution_ladder.json"


@pytest.mark.skipif(not LADDER.exists(), reason="ladder payload not generated")
def test_published_ladder_is_internally_consistent():
    payload = json.loads(LADDER.read_text())
    gate = payload["gates"]["owner_rel"]
    assert gate == 0.15
    tol = payload["gates"]["anchor_tol"]
    for rec in payload["records"]:
        # every level starts on a lattice, so the metrics must hit the anchors
        assert rec["initial_error_max"] <= tol
        # the reported relative error must be the one the metrics imply
        nn = rec["metrics"]["nn_median_over_dx"]
        assert rec["error"]["relative"]["nn_median_over_dx"] == pytest.approx(
            abs(nn - 1.0), rel=1e-12)
        worst = rec["error"]["worst_relative"]
        assert (worst <= gate) == (rec["dx"] in payload["analysis"]
                                   ["by_variant"][rec["variant"]]
                                   ["gate_pass_levels"])