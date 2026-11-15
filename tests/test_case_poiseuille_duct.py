"""Tests for CASE-DUCT-POISEUILLE-3D (reference layer + mirror planes + dtype).

The duct reference is graded against three independent statements before any
SPH number is trusted:

* the double series reproduces the plane-Poiseuille parabola as ``W/H -> inf``,
* the square duct reproduces the classical centre-velocity ratio 0.5894,
* an independently coded finite-difference Poisson solve agrees with the series.

Plus the two code paths the case needs and every other 3-D caller benefits from:
the mirror planes for an arbitrary normal axis, and the 3-D operator set on a
float64 cloud (it used to raise a Float/Double mismatch in ``index_add``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_poiseuille_duct import (  # noqa: E402
    SQUARE_TO_PLANE, DuctConfig, build_duct, cfg_mean, duct_series, fd_reference,
    plane_limit_table, truncation_table,
)
from tensorsph.wall_mirror import mirror_plane, plane_walls  # noqa: E402


def test_series_matches_plane_poiseuille_limit():
    """W/H -> 16 must reproduce g0 (H-y) y / (2 nu) on the centre plane."""
    cfg = DuctConfig(aspect=16.0)
    y = np.linspace(0.02, 0.08, 7)
    z = np.full_like(y, cfg.width / 2)
    got = duct_series(cfg, y, z)
    want = cfg.g0 * (cfg.height - y) * y / (2.0 * cfg.nu)
    assert np.max(np.abs(got - want) / want) < 1e-3


def test_square_duct_centre_ratio():
    """Classical square-duct centre velocity over the plane-channel value."""
    cfg = DuctConfig(aspect=1.0)
    uc = float(duct_series(cfg, [cfg.height / 2], [cfg.width / 2])[0])
    assert abs(uc / cfg.u_plane - SQUARE_TO_PLANE) / SQUARE_TO_PLANE < 5e-3


def test_bulk_velocity_matches_quadrature():
    """Closed-form bulk velocity vs a fine quadrature of the series."""
    cfg = DuctConfig(aspect=1.0)
    n = 200
    y = (np.arange(n) + 0.5) * cfg.height / n
    z = (np.arange(n) + 0.5) * cfg.width / n
    Y, Z = np.meshgrid(y, z, indexing="ij")
    quad = float(duct_series(cfg, Y.ravel(), Z.ravel()).mean())
    assert abs(quad - cfg_mean(cfg)) / cfg_mean(cfg) < 1e-3


def test_fd_reference_agrees_with_series():
    """Independent finite-difference Poisson solve is the reference's own gate."""
    for aspect in (1.0, 4.0):
        cfg = DuctConfig(aspect=aspect)
        fd = fd_reference(cfg, 64, 64)
        assert abs(fd["centre_error_pct"]) < 1.0
        assert abs(fd["l2_error_pct"]) < 1.0


def test_truncation_is_negligible():
    cfg = DuctConfig()
    rows = truncation_table(DuctConfig(aspect=1.0), nmax_list=(41, 81))
    assert all(abs(r["error_vs_nmax321_pct"]) < 0.1 for r in rows)


def test_plane_limit_table_monotone():
    rows = plane_limit_table(aspects=(1.0, 4.0, 8.0))
    ratios = [r["ratio_to_plane"] for r in rows]
    assert ratios[0] < ratios[1] < ratios[2] <= 1.0 + 1e-3


def test_mirror_plane_axis_z():
    """A wall normal to ``z`` mirrors that coordinate and no other."""
    pos = torch.tensor([[0.0, 0.05, 0.01], [0.0, 0.05, 0.09]], dtype=torch.float64)
    vel = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    mass = torch.ones(2, dtype=torch.float64)
    gp, gv, _ = mirror_plane(pos, vel, mass, 0.0, 0.2, side=+1, axis=2)
    assert gp.shape == pos.shape
    assert torch.allclose(gp[:, 2], -pos[:, 2])
    assert torch.allclose(gp[:, :2], pos[:, :2])
    assert torch.allclose(gv, -vel)          # no-slip: velocity reversed


def test_plane_walls_closes_a_box():
    """Four planes with per-plane axes: ghosts on both transverse pairs."""
    cfg = DuctConfig()
    pos = torch.tensor([[0.0, 0.01, 0.01], [0.0, 0.09, 0.09]], dtype=torch.float64)
    vel = torch.zeros_like(pos)
    mass = torch.ones(pos.shape[0], dtype=torch.float64)
    planes = [(0.0, +1, None, 1), (cfg.height, -1, None, 1),
              (0.0, +1, None, 2), (cfg.width, -1, None, 2)]
    gp, gv, gm = plane_walls(pos, vel, mass, planes, 0.05)
    assert gp.shape[0] == 4                       # one ghost per plane per particle
    assert gm.shape[0] == gp.shape[0]
    assert gv.abs().max() == 0.0


def test_build_duct_is_cell_centred():
    cfg = DuctConfig(aspect=4.0)
    pos, mass, counts, dx = build_duct(cfg, 6)
    nx, ny, nz = counts
    assert (nx, ny, nz) == (4, 6, 24)
    assert float(pos[:, 1].min()) == pytest.approx(0.5 * dx)
    assert float(pos[:, 2].max()) == pytest.approx(cfg.width - 0.5 * dx)
    assert float(mass.min()) == pytest.approx(float(mass.max()))


def test_operators3_accept_float64_cloud():
    """Regression: the 3-D pair kernels must match the cloud's precision."""
    from tensorsph.threed.neighbors3 import box_pairs
    from tensorsph.threed.operators3 import accelerations
    torch.manual_seed(0)
    n = 4
    pos = torch.rand((n ** 3, 3), dtype=torch.float64) * 0.1
    mass = torch.full((pos.shape[0],), 1e-3, dtype=torch.float64)
    vel = torch.zeros_like(pos)
    pairs = box_pairs(pos, 0.05)
    acc, rho, pressure = accelerations(pos, vel, mass, 0.02, pairs, rho0=1000.0,
                                       sound_speed=20.0, viscosity=0.01)
    assert acc.dtype == torch.float64 and rho.dtype == torch.float64
    assert torch.isfinite(acc).all()
