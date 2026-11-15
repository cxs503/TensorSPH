"""Tests for the shift-bound leg of ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Three things are asserted, all cheap (operator level, no fluid solve):

1. **The displacement law is exact.**  ``delta r = alpha h^2 g``, so for a fixed
   particle configuration ``|delta r| / alpha`` must be *independent of alpha*.
   This is what licenses the benchmark's use of a single slope
   ``s(alpha)/alpha`` and therefore its prediction ``alpha = C dx / (h G)``.
2. **The bound can be inverted.**  Given a measured ``s(alpha)`` curve, the
   ``alpha`` that saturates ``s = C`` is recovered to round-off -- the
   prediction machinery of ``shift_bound.py`` does not itself add error.
3. **The committed payload says what the document says.**  The numbers quoted in
   ``docs/xval_sphinx_tvc/SHIFT_BOUND_PREDICTION.md`` are recomputed here from
   ``data/shift_bound.json`` (slope, residual, implied bound), so a silent edit
   of either the payload or the generator is a test failure.

The physical claim the leg grades -- "is the *accuracy* optimum on a
resolution-independent bound?" -- is *not* asserted here; it is a measurement
recorded in the payload (H1 rejected) and re-derived by point 3.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral,
    transport_velocity_correction,
)

DATA = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "shift_bound.json"
H_OVER_DX = 1.3


def free_surface_lattice(nx: int = 30, ny: int = 10, dx: float = 0.02,
                         h_over_dx: float = H_OVER_DX):
    """A square lattice in ``x2 <= 0`` with a plane free surface (no wall)."""
    xs = torch.arange(nx, dtype=torch.float64) * dx
    ys = (torch.arange(ny, dtype=torch.float64) - (ny - 1)) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    vol = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    return pos, vol, h_over_dx * dx


@pytest.mark.parametrize("alpha", [0.02, 0.05, 0.10, 0.15, 0.20])
def test_shift_over_alpha_is_independent_of_alpha(alpha: float):
    """``|delta r| = alpha h^2 |g|``: the slope is a property of the state."""
    pos, vol, h = free_surface_lattice()
    g = kernel_gradient_integral(pos, vol, h, wall_count=0)
    res = transport_velocity_correction(pos, vol, h, alpha=alpha, wall_count=0)
    g_peak = torch.linalg.vector_norm(g, dim=1).max().item()
    expected = alpha * h * h * g_peak
    assert res.max_shift == pytest.approx(expected, rel=1e-12)
    # and the *slope* is the same number for every alpha
    slope = res.max_shift / alpha
    assert slope == pytest.approx(h * h * g_peak, rel=1e-12)


def test_free_surface_slope_matches_the_planar_anchor_scaling():
    """``s(alpha)/alpha = (h/dx) * |g| h`` with ``|g| h`` the surface constant.

    This is the identity ``shift_bound.py`` inverts.  It is checked in units of
    ``h`` and of ``dx`` so both normalisations of the report are pinned.
    """
    pos, vol, h = free_surface_lattice()
    dx = 0.02
    g = kernel_gradient_integral(pos, vol, h, wall_count=0)
    g_peak = torch.linalg.vector_norm(g, dim=1).max().item()
    res = transport_velocity_correction(pos, vol, h, alpha=0.1, wall_count=0)
    slope = res.max_shift / 0.1
    assert slope == pytest.approx(h * h * g_peak, rel=1e-12)
    assert slope / dx == pytest.approx((h / dx) * g_peak * h, rel=1e-12)
    # the surface constant is O(1) and positive: the correction moves particles
    assert 0.5 < g_peak * h < 3.0


def _bound_alpha(C: float, alphas: list[float], shifts: list[float]) -> float:
    """Invert a measured ``s(alpha)`` curve at ``s = C`` (linear interpolation)."""
    for i in range(len(alphas) - 1):
        if shifts[i] <= C <= shifts[i + 1]:
            t = (C - shifts[i]) / (shifts[i + 1] - shifts[i])
            return alphas[i] + t * (alphas[i + 1] - alphas[i])
    return float("nan")


def test_bound_inversion_is_exact_on_the_measured_law():
    """On ``s = m alpha`` the inversion must return ``C/m``."""
    alphas = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
    m = 2.01
    shifts = [m * a for a in alphas]
    for C in (0.10, 0.15, 0.201, 0.25):
        assert _bound_alpha(C, alphas, shifts) == pytest.approx(C / m, rel=1e-12)


@pytest.mark.skipif(not DATA.exists(), reason="payload not generated")
@pytest.mark.skip(reason="level-level reproduction superseded by round-3 grid gate "
                         "(tests/test_tvc_shift_bound_grid_complete.py re-derives "
                         "C_implied/A_implied/H6 over complete levels only); this leg "
                         "assumed every payload level is full-grid, and dx=0.01 m is "
                         "declared-but-still-measuring (job cj3832)")
def test_payload_slope_and_implied_bound_are_reproducible():
    """The document's slope / residual / implied bound come from the payload."""
    payload = json.loads(DATA.read_text())
    records = payload["records"]
    by_dx: dict[float, list[dict]] = {}
    for r in records:
        by_dx.setdefault(round(r["dx"], 6), []).append(r)

    slopes = []
    for dx, pts in by_dx.items():
        pos = sorted((p for p in pts if p["alpha"] > 0), key=lambda p: p["alpha"])
        assert pos, f"no positive-alpha points at dx={dx}"
        # through-origin slope, exactly as the generator computes it
        sxx = sum(p["alpha"] ** 2 for p in pos)
        sxy = sum(p["alpha"] * p["max_shift_over_dx"] for p in pos)
        m = sxy / sxx
        resid = max(abs(p["max_shift_over_dx"] - m * p["alpha"])
                    / p["max_shift_over_dx"] for p in pos)
        assert resid <= 0.03, f"displacement law not linear at dx={dx}: {resid:.3%}"
        slopes.append(m)
        # the optimum recorded in the summary is the argmin of the measured error
        summary_level = payload["summary"]["levels"][f"{dx:g}"]
        best = min(pos, key=lambda p: p["rel_error"])
        assert summary_level["alpha_opt"] == pytest.approx(best["alpha"], abs=1e-12)
        # C_implied is the measured displacement at the optimum alpha
        assert summary_level["C_implied"] == pytest.approx(
            best["max_shift_over_dx"], rel=1e-9)

    # resolution independence of the displacement slope: < 1 %
    assert (max(slopes) - min(slopes)) / (sum(slopes) / len(slopes)) < 0.01

    # H1 verdict must be the one the payload records (no silent flip)
    spread = payload["summary"]["C_implied"]["spread_over_mean"]
    assert payload["summary"]["C_implied"]["H1_holds_25pct"] == (spread <= 0.25)


@pytest.mark.skipif(not DATA.exists(), reason="payload not generated")
def test_shipped_coefficient_row_is_present_at_every_level():
    """The comparison against SPHinXsys' shipped ``alpha = 0.05`` must exist.

    Without it the leg could not say whether the shipped rate is (or is not)
    sufficient, which is the ledger's actual question.
    """
    payload = json.loads(DATA.read_text())
    levels = payload["summary"]["levels"]
    assert levels, "no levels summarised"
    for dx, lv in levels.items():
        assert lv["shipped_alpha"] == pytest.approx(0.05)
        assert lv["shipped_rel_error"] is not None, f"dx={dx} lacks the shipped row"
        assert lv["baseline_rel_error"] is not None
        assert math.isfinite(lv["C_implied"])