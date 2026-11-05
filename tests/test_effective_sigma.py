"""Effective surface tension of the CSF operator (verification debt, m3188).

The historical pin in ``test_laplace_pressure.py`` (0.79 +/- 0.06) was measured
with the ``mls`` curvature operator, which was the default until m2979 switched
the default to ``fit_quadratic``.  The static circular-droplet force ratio
``|F_right| / (2 sigma)`` is the global quantity the Young-Laplace balance
sees; the analytic target is 1.0.

These tests pin the *current* model state per scheme (so a silent change of
either operator is caught) and record the honest verdict: neither scheme
converges to 1.0 in the tested range, so the effective sigma is a systematic
bias of 21-29 % (WITHHELD, > 15 % platform gate).  See
``docs/verification_effective_sigma.md``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from tensorsph import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure" / "run.py"
_spec = importlib.util.spec_from_file_location("laplace_benchmark", _BENCH)
laplace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(laplace)

SIGMA = 0.072
DX = 0.005
RADIUS = 0.04


def _ratio(curvature: str, dx: float = DX, radius: float = RADIUS,
           sigma: float = SIGMA) -> float:
    cfg = Config(dx=dx, rho0=1000.0, sound_speed=20.0, gravity=0.0,
                 surface_tension_coefficient=sigma, csf_curvature=curvature)
    pos, mass, density = laplace.circular_droplet(radius, dx)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx)
    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    force = mass[:, None] * acc
    right = pos[:, 0] > 0.0
    return float(torch.linalg.vector_norm(force[right].sum(dim=0)) / (2.0 * sigma))


def test_current_default_delivers_the_documented_ratio():
    """fit_quadratic (default since m2979) at the documented R/h = 6.15.

    Measured 2026-11-04: 0.7106.  The pre-m2979 value 0.7907 (mls) is pinned
    separately below; the gap between the two is the m2979 operator swap.
    """
    ratio = _ratio("fit_quadratic")
    assert abs(ratio - 0.7106) < 0.02, ratio


def test_mls_operator_still_delivers_the_historical_ratio():
    """The pre-m2979 default (mls) must remain reachable and unchanged.

    This is the honest record of the old pin: 0.7907 at R/h = 6.15.  Keeping it
    as an explicit-scheme assertion (instead of deleting it) preserves the
    history the 0.79 pin encoded.
    """
    ratio = _ratio("mls")
    assert abs(ratio - 0.7907) < 0.02, ratio


def test_effective_sigma_is_below_the_analytic_amplitude():
    """The operator under-delivers, it does not overshoot.

    A ratio above 1.0 would mean the CSF force is *stronger* than the analytic
    traction -- a different failure mode (e.g. a missing normalisation) that
    the two pins above would not distinguish from a healthy 1.0.
    """
    for curv in ("fit_quadratic", "mls"):
        assert _ratio(curv) < 1.0, curv


def test_fit_quadratic_does_not_improve_under_refinement():
    """Guard the recorded trend: refinement moves fit_quadratic *away* from 1.0.

    0.7106 (R/h = 6.15) -> 0.6818 (R/h = 12.31).  If a future fix makes the
    integrated force converge, this test must be rewritten, not deleted.
    """
    coarse = _ratio("fit_quadratic", dx=DX)
    fine = _ratio("fit_quadratic", dx=DX / 2.0)
    assert fine < coarse, (coarse, fine)
