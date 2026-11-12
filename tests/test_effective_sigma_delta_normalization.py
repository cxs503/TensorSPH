"""Delta-function normalisation of the CSF operator vs the analytic 2*sigma.

Pins the two measurements of
``docs/verification_effective_sigma_delta_normalization.md`` on a static
circular droplet (R = 0.04 m, dx = 5 mm):

* ``int delta_s dV / (2 pi R)`` -- the delta function must integrate to the
  interface length (unit depth).  ``legacy`` (``2 |grad c|``) reaches only
  ~2/3 of it; ``local_range`` (``|grad c| / (c_bulk - c_void)``) reaches ~0.91.
* ``|F_right| / (2 sigma)`` -- the CSF traction integrated over the right half
  of the droplet.  Exact answer 1.0.  ``legacy`` = 0.71 (-29 %),
  ``local_range`` = 0.97 (-2.8 %).

Both numbers are static force evaluations (no time integration), so the test
costs under two seconds.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph import Config
from tensorsph.csf_consistent import csf_fields
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "laplace_benchmark", ROOT / "benchmarks" / "laplace_pressure" / "run.py")
laplace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(laplace)

SIGMA = 0.072
RHO0 = 1000.0
RADIUS = 0.04
DX = 0.005


def _droplet(delta_norm: str):
    cfg = Config(dx=DX, rho0=RHO0, sound_speed=20.0, gravity=0.0,
                 surface_tension_coefficient=SIGMA,
                 csf_curvature="fit_quadratic",
                 csf_delta_normalization=delta_norm)
    pos, mass, density = laplace.circular_droplet(RADIUS, DX)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx)
    return cfg, pos, mass, density, pairs


def _force_ratio(delta_norm: str) -> float:
    cfg, pos, mass, density, pairs = _droplet(delta_norm)
    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    force = mass[:, None] * acc
    right = force[pos[:, 0] > 0.0].sum(dim=0)
    return float(torch.linalg.vector_norm(right) / (2.0 * SIGMA))


def _delta_integral(delta_norm: str) -> float:
    cfg, pos, mass, _, pairs = _droplet(delta_norm)
    h = cfg.h_factor * cfg.dx
    dpairs = cell_linked_pairs(pos, 4.0 * h)
    state = csf_fields(pos, mass, dpairs, h, RHO0,
                       curvature_scheme="fit_quadratic", curvature_stencil=4.0,
                       delta_normalization=delta_norm)
    ds = state.delta_s.clone()
    mag = torch.linalg.vector_norm(state.grad_c, dim=1)
    ds[mag <= 1.0e-3 * mag.max()] = 0.0
    return float((mass / RHO0 * ds).sum() / (2.0 * math.pi * RADIUS))


def test_legacy_normalisation_reproduces_the_documented_29_percent_deficit():
    """Baseline: the default stays bit-compatible with the published 0.71."""
    assert Config(dx=0.01).csf_delta_normalization == "legacy"
    assert _force_ratio("legacy") == pytest.approx(0.711, abs=0.03)
    assert _delta_integral("legacy") == pytest.approx(0.668, abs=0.02)


def test_local_range_normalisation_brings_the_force_inside_5_percent():
    """local_range recovers the interface length and the 2 sigma force."""
    assert _delta_integral("local_range") == pytest.approx(0.914, abs=0.02)
    assert abs(_force_ratio("local_range") - 1.0) < 0.05


def test_delta_integral_and_force_move_together():
    """The force deficit tracks the missing interface length, not curvature."""
    for norm in ("legacy", "local_range"):
        length_deficit = _delta_integral(norm)
        # force ratio should sit below the recovered length by the (measured,
        # resolution-limited) curvature bias only, i.e. the same order.
        assert abs(_force_ratio(norm) - length_deficit) < 0.12


def test_unknown_delta_normalisation_is_rejected():
    from tensorsph.droplet_impact import validate_config
    cfg = Config(dx=DX, csf_delta_normalization="bogus")
    with pytest.raises(ValueError, match="csf_delta_normalization"):
        validate_config(cfg)
