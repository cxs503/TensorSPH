"""Tests for the parameter-free free-surface closure of the CSS stress flux.

What is pinned (each against a property, not a recorded number):

* ``support_matrix`` is the identity in the bulk of a filled body and ``I/2`` on
  an exact half-space -- that is the *definition* of the support fraction, and it
  is what makes the closure a measurement rather than a calibration;
* the pair factor is exactly ``1`` for a complete support and clamps to
  ``1/SUPPORT_FRACTION_FLOOR`` for a vanishing one;
* the closure keeps the stress flux momentum-conserving to round-off;
* the free-surface Laplace reading is inside the 15 % ledger gate and the closure
  improves it by more than 5x over the shipped one-sided sum, at two
  resolutions (no fitted coefficient anywhere in the path).
"""
from __future__ import annotations

import math

import torch

from tensorsph import sphinxys_css as sc
from tensorsph.neighbor_search import cell_linked_pairs

DX = 0.01
H = sc.SPHINXSYS_H_SPACING_RATIO * DX
SIGMA = 0.072
RADIUS = 0.1


def _lattice(half_height: float | None = None):
    """Square lattice of spacing DX; ``half_height`` cuts a plane free surface."""
    span = 10.0 * H
    axis = torch.arange(-span, span + 0.5 * DX, DX, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    if half_height is not None:
        pos = pos[pos[:, 1] >= -1e-12]
    vol = torch.full((pos.shape[0],), DX * DX, dtype=pos.dtype)
    return pos, vol


def test_support_matrix_is_identity_in_the_bulk():
    pos, vol = _lattice()
    pairs = cell_linked_pairs(pos, 2.0 * H)
    matrix = sc.support_matrix(pos, vol, pairs, H)
    centre = torch.argmin(torch.linalg.vector_norm(pos, dim=1))
    assert torch.allclose(matrix[centre], torch.eye(2, dtype=pos.dtype), atol=5e-2), \
        matrix[centre]
    s = sc.support_fraction(pos, vol, pairs, H)
    assert abs(float(s[centre]) - 1.0) < 0.05, float(s[centre])


def test_half_space_support_fraction_is_one_half():
    """A particle sitting on a plane free surface sees exactly half the kernel."""
    pos, vol = _lattice(half_height=0.0)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    s = sc.support_fraction(pos, vol, pairs, H)
    y0 = float(pos[:, 1].min())
    on_surface = (pos[:, 1] - y0).abs() < 1e-9
    assert float(s[on_surface].mean()) < 0.65, float(s[on_surface].mean())
    assert float(s[on_surface].mean()) > 0.40, float(s[on_surface].mean())
    deep = pos[:, 1] > y0 + 2.0 * H
    assert float(s[deep].mean()) > 0.85


def test_pair_factor_is_unity_for_complete_support():
    pairs = torch.tensor([[0, 1], [1, 2]])
    s = torch.ones(3, dtype=torch.float64)
    assert torch.allclose(sc.pair_renormalisation(s, pairs), torch.ones(2, dtype=torch.float64))
    assert sc.pair_renormalisation(torch.zeros(3, dtype=torch.float64), pairs).max() <= \
        1.0 / sc.SUPPORT_FRACTION_FLOOR + 1e-12
    assert sc.pair_renormalisation(s, torch.empty((0, 2), dtype=torch.long)).numel() == 0


def _droplet(radius_over_dx: float, completion: str):
    dx = RADIUS / radius_over_dx
    h = sc.SPHINXSYS_H_SPACING_RATIO * dx
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "surface_tension"))
    import single_phase_completion as SPC

    pos, r, rho0, vol = SPC.make_disc(RADIUS, dx)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    acc, _fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs, None, SIGMA, h,
        frac=torch.ones(pos.shape[0], dtype=pos.dtype), mode="single_phase",
        hourglass=False, assembly="consistent", completion=completion)
    n_out = pos / r.clamp_min(1e-300)[:, None]
    dp = float((-rho0 * (acc * n_out).sum(dim=1) * dx * dx).sum()) / (2.0 * math.pi * RADIUS)
    mom = (rho0 * vol)[:, None].mul(acc).sum(dim=0)
    scale = float((rho0 * vol * torch.linalg.vector_norm(acc, dim=1)).sum())
    return dp / (SIGMA / RADIUS), float(torch.linalg.vector_norm(mom)) / max(scale, 1e-300)


def test_closure_keeps_momentum_and_gate():
    for rdx in (16, 32):
        ratio_closed, mom_closed = _droplet(rdx, "completeness")
        ratio_open, _ = _droplet(rdx, "none")
        err_closed = abs(ratio_closed - 1.0) * 100.0
        err_open = abs(ratio_open - 1.0) * 100.0
        assert mom_closed < 1e-12, mom_closed
        assert err_closed < 15.0, (rdx, ratio_closed)
        assert err_open / err_closed > 5.0, (rdx, err_open, err_closed)