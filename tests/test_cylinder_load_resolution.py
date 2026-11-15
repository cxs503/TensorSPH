"""Gates for the cylinder load arm at the *coupled* fsi2 scale (R = 0.05 m).

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.  Every earlier static
audit of the cylinder load ran the SPHinXsys scale (``R = 0.5 m``,
``h/R <= 0.10``).  The coupled Turek-Hron runs use ``D = 0.1 m`` -> ``R = 0.05 m``
with ``dx = 0.02 m``, i.e. ``h/R = 0.52`` and a support ``2h`` that exceeds the
body radius.  These tests pin what the two arms do *there*:

* the plane-fit arm is exact on a linear field (so the quadrature and the
  tributary arc are not the problem),
* the kernel-average arm over-loads by a measured, h/R-controlled factor
  (31 % at the coupled configuration), which is *upward*: it cannot explain a
  drag coefficient that is 60x too small,
* no station at the coupled configuration is dry (coverage 1, nearest fluid
  particle 0.5 dx away), so the ``X4 dry body`` defect does not recur here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

import cylinder_resolution_audit as A  # noqa: E402


def _coupled():
    """The coupled fsi2 configuration: R = D/2 = 0.05 m, dx = 0.02 m."""
    import math
    n_circ = max(int(math.ceil(2.0 * math.pi * 0.05 / 0.02)), 8)
    return A.score(0.05, 0.02, n_circ)


def test_plane_fit_arm_is_exact_at_benchmark_scale():
    row = _coupled()
    assert abs(row["fit_err_x"]) < 1e-5
    assert abs(row["fit_err_y"]) < 1e-5


def test_kernel_arm_overload_at_benchmark_scale_is_pinned():
    """31 % over-load, measured; upward sign is the point of the assertion."""
    row = _coupled()
    assert row["kernel_err_x"] > 0.0
    assert 0.20 < row["kernel_err_x"] < 0.45
    assert abs(row["kernel_err_x"] - row["kernel_err_y"]) < 1e-9


def test_kernel_bias_is_controlled_by_h_over_R():
    rows = [A.score(0.05, dx, max(int(2 * 3.141592653589793 * 0.05 / dx), 8))
            for dx in (0.04, 0.02, 0.01)]
    errs = [r["kernel_err_x"] for r in rows]
    assert errs[0] > errs[1] > errs[2], errs
    # and it is the same defect on the SPHinXsys scale, just weaker
    big = A.score(0.5, 0.04, 79)
    assert 0.0 < big["kernel_err_x"] < errs[0]


def test_no_dry_stations_at_benchmark_scale():
    row = _coupled()
    assert row["coverage"] == pytest.approx(1.0)
    assert row["gap_min_over_dx"] >= 0.5 - 1e-9


def test_station_refinement_does_not_move_the_arms():
    """Quadrature is converged: only the support error depends on h."""
    rows = [A.score(0.05, 0.02, n) for n in (16, 32, 64)]
    spread = max(r["kernel_err_x"] for r in rows) - min(r["kernel_err_x"] for r in rows)
    assert spread < 5e-3
    for r in rows:
        assert abs(r["fit_err_x"]) < 1e-5