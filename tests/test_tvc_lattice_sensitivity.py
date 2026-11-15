"""Lattice-dependence of the free-surface TVC anchor (second verification leg).

``docs/xval_tvc_surface_anchor.md`` graded the free-surface branch of the
transport-velocity-correction operator against the exact continuum limit
``g -> 2 I(h, d) n`` and reported **14.15 %** deficit at the production
resolution ``h/dx = 1.3`` -- i.e. the ``< 15 %`` owner gate passed with 0.85
points of margin, measured on **one** particle arrangement (square lattice,
surface row exactly on the plane).

These tests widen the arrangement family and assert the *numbers* the new leg
measured (``benchmarks/xval_sphinx_tvc/lattice_sensitivity.py``,
``docs/xval_tvc_lattice_sensitivity.md``):

* the square lattice reproduces the published 14.15 % exactly (regression);
* the densest hexagonal packing has a *smaller* deficit (10.33 %) at the same
  ``h/dx`` and both converge at local order ~2;
* a lattice rotated against the surface normal is closer to the continuum
  limit than the aligned one and develops a tangential component (the
  correction stops being purely normal);
* uniform jitter keeps the bias (-14 %) but widens the scatter, and at
  ``zeta = 0.30 dx`` the ensemble MAE crosses the 15 % gate;
* the wall branch's *exact* anchor (``|g| = 0`` iff ``wall_factor = 2``)
  holds on the triangular lattice too;
* the bulk is a fixed point on both lattices (``|g| = 0`` to round-off).

Every gate below is a measured value with a band; none of them is a tautology
(the operator could pass a "returns a tensor" test while being wrong).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "benchmarks" / "xval_sphinx_tvc"))

from lattice_sensitivity import (  # noqa: E402
    build_lattice,
    line_integrals,
    measure_lattice,
    wall_hex_case,
)
from surface_term import HBAR, closed_form_I0, line_integral  # noqa: E402

H = 0.026          # production smoothing length of this repo
PROD = HBAR        # production h/dx = 1.3
PUBLISHED_SQUARE = -0.1415   # docs/xval_tvc_surface_anchor.md


def _row(kind: str, h_over_dx: float = PROD) -> dict:
    return measure_lattice(build_lattice(kind, H, h_over_dx), H)


# --------------------------------------------------------------------------- #
# the batched anchor quadrature must equal the loop version
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("depth_over_h", [0.0, 0.25, 0.75, 1.0])
def test_batched_anchor_matches_reference_quadrature(depth_over_h):
    d = depth_over_h * H
    loop = line_integral(H, d)
    vec = float(line_integrals(H, torch.tensor([d], dtype=torch.float64)))
    assert vec == pytest.approx(loop, rel=1e-10)


def test_closed_form_surface_line_integral():
    assert line_integral(H, 0.0) == pytest.approx(closed_form_I0(H), rel=1e-9)


# --------------------------------------------------------------------------- #
# regression: the square-lattice number of the published leg
# --------------------------------------------------------------------------- #
def test_square_lattice_reproduces_published_deficit():
    row = _row("square")
    assert row["mean_rel"] == pytest.approx(PUBLISHED_SQUARE, rel=2e-3), row
    assert row["cos"] > 0.9999, "aligned square lattice must be purely normal"


# --------------------------------------------------------------------------- #
# hexagonal packing
# --------------------------------------------------------------------------- #
def test_hex_lattice_has_smaller_deficit_than_square():
    sq, hx = _row("square"), _row("hex")
    assert hx["mean_rel"] < 0.0
    assert abs(hx["mean_rel"]) < 0.15, "hex must stay inside the owner gate"
    assert abs(hx["mean_rel"]) < 0.85 * abs(sq["mean_rel"]), (sq, hx)
    assert hx["cos"] > 0.9999


@pytest.mark.parametrize("kind", ["square", "hex"])
def test_bulk_is_a_fixed_point_on_every_lattice(kind):
    """Full support => the grad-1 integral vanishes exactly, not approximately."""
    row = _row(kind)
    assert row["n_deep"] > 10
    assert row["bulk_rel"] < 1e-10, f"{kind}: bulk must be a fixed point"


@pytest.mark.parametrize("kind", ["square", "hex"])
def test_refinement_converges_at_second_order(kind):
    errs = [abs(_row(kind, n)["mae_rel"]) for n in (PROD, 2.0, 4.0, 8.0)]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    order = math.log(errs[0] / errs[1]) / math.log(2.0 / PROD)
    assert order > 1.5, f"{kind}: local order {order:.2f} below 1.5"
    assert errs[-1] < 0.005, f"{kind}: h/dx=8 must be under 0.5 %, got {errs[-1]:.4f}"


# --------------------------------------------------------------------------- #
# orientation of the lattice relative to the free surface
# --------------------------------------------------------------------------- #
def test_aligned_lattice_is_not_the_best_orientation():
    """The published 14.15 % is the *aligned* case; rotation moves it either way."""
    aligned = _row("square")
    rotated = {}
    for deg in (15.0, 30.0, 45.0):
        lat = build_lattice("square", H, PROD, rotation_deg=deg)
        rotated[deg] = measure_lattice(lat, H)
    best = min(rotated.items(), key=lambda kv: abs(kv[1]["mean_rel"]))
    assert abs(best[1]["mean_rel"]) < 0.5 * abs(aligned["mean_rel"]), rotated
    # the bias is not a universal constant of the operator: it changes sign
    assert any(r["mean_rel"] > 0.0 for r in rotated.values()), rotated


def test_rotated_lattice_leaks_tangentially():
    """A non-symmetric surface makes |g| point away from the normal."""
    lat = build_lattice("square", H, PROD, rotation_deg=30.0)
    row = measure_lattice(lat, H)
    assert row["cos"] < 0.995, row
    assert row["cos"] > 0.9, "tangential leak must stay a perturbation"


# --------------------------------------------------------------------------- #
# disorder: what a real run's distribution does to the same anchor
# --------------------------------------------------------------------------- #
def test_jitter_is_reproducible_for_a_fixed_seed():
    a = measure_lattice(build_lattice("square", H, PROD, jitter=0.15, seed=7), H)
    b = measure_lattice(build_lattice("square", H, PROD, jitter=0.15, seed=7), H)
    c = measure_lattice(build_lattice("square", H, PROD, jitter=0.15, seed=8), H)
    assert a["mae_rel"] == pytest.approx(b["mae_rel"], rel=0, abs=0)
    assert a["mae_rel"] != pytest.approx(c["mae_rel"], rel=1e-6)


def test_jitter_keeps_the_bias_but_widens_the_tail():
    """Disorder costs *tail*, not *mean* -- the mean hypothesis was falsified.

    Falsified: "more jitter => larger ensemble-mean deviation".  On 6 and on 24
    seeds the mean at ``zeta = 0.30`` (13.7 % / 16.4 %) is within noise of the
    mean at ``zeta = 0.05`` (13.8 % / 14.1 %): the random part of the quadrature
    error averages out, while the *systematic* -14 % lattice bias does not.  What
    does grow monotonically is the tail -- per-realisation max and p95.
    """
    seeds = range(8)
    rows = {zeta: [measure_lattice(
        build_lattice("square", H, PROD, jitter=zeta, seed=s), H) for s in seeds]
        for zeta in (0.05, 0.30)}
    small = [r["mae_rel"] for r in rows[0.05]]
    large = [r["mae_rel"] for r in rows[0.30]]
    mean_small = sum(small) / len(small)
    mean_large = sum(large) / len(large)
    spread_small = max(small) - min(small)
    spread_large = max(large) - min(large)
    assert mean_small < 0.15, "mild disorder must stay inside the gate"
    assert spread_large > 2.0 * spread_small, (spread_small, spread_large)
    assert max(large) > 0.15, (
        "at zeta=0.30 dx a single realisation must be able to cross 15 %; "
        f"measured {max(large):.3f}")
    assert abs(mean_large - mean_small) < 0.5 * spread_large, "tail, not mean"
    # the systematic part stays the lattice bias whatever the disorder
    for r in rows[0.05] + rows[0.30]:
        assert r["mean_rel"] < 0.0


# --------------------------------------------------------------------------- #
# wall branch: the exact mirror anchor on a triangular lattice
# --------------------------------------------------------------------------- #
def test_wall_branch_is_exact_on_the_triangular_lattice():
    two = wall_hex_case(H, PROD, wall_factor=2.0)
    one = wall_hex_case(H, PROD, wall_factor=1.0)
    assert two["n_graded"] > 10
    assert two["g_max_over_scale"] < 1e-12, two
    assert one["g_max_over_scale"] > 0.3, one
    assert two["g_max_over_scale"] < 1e-8 * one["g_max_over_scale"]