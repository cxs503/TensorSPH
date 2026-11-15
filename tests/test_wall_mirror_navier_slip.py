"""Tests for the ``slip="navier"`` (partial-slip) mirror boundary condition.

The Navier mode is a one-parameter family; what is asserted here are the
*invariants* that pin it down, not merely that the code runs:

* exact reduction to ``no_slip`` as ``lam -> 0`` and to ``free_slip`` as
  ``lam -> inf`` (so every previously verified ``lam = 0`` number is a member of
  the same family);
* the ghost tangential velocity is ``w + alpha (v - w)`` with
  ``alpha = (lam - d)/(lam + d)``, ``d`` measured to the plane;
* consistency with the analytic slip solution: for a wall layer that already
  carries ``u(d) = lam G + G d`` (slip velocity ``lam G``, wall gradient ``G``)
  the rule returns exactly ``lam G`` -- this is what makes the discrete BC the
  first-order discretisation of ``u_w = lam du/dn`` rather than a fit;
* the normal component is mirrored (no mass flux through the wall), and a wall
  moving tangentially with the fluid produces no tangential jump;
* argument validation, and pass-through of ``slip_length`` via ``plane_walls``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorsph.wall_mirror import mirror_plane, plane_walls


def _state(ys, ux=None, uy=None):
    ys = torch.tensor(ys, dtype=torch.float64)
    n = ys.numel()
    pos = torch.stack((torch.full_like(ys, 0.5), ys), dim=1)
    vel = torch.zeros((n, 2), dtype=torch.float64)
    if ux is not None:
        vel[:, 0] = torch.as_tensor(ux, dtype=torch.float64)
    if uy is not None:
        vel[:, 1] = torch.as_tensor(uy, dtype=torch.float64)
    mass = torch.ones(n, dtype=torch.float64)
    return pos, vel, mass


def test_navier_reduces_to_no_slip_as_lambda_goes_to_zero():
    pos, vel, mass = _state([0.1, 0.2], ux=[1.7, -0.4], uy=[0.3, 0.9])
    gp0, gv0, _ = mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="no_slip")
    gp1, gv1, _ = mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="navier",
                               slip_length=1e-12)
    assert torch.allclose(gp0, gp1)
    assert torch.allclose(gv0, gv1, atol=1e-9)


def test_navier_reduces_to_free_slip_for_huge_lambda():
    pos, vel, mass = _state([0.1, 0.2], ux=[1.7, -0.4], uy=[0.3, 0.9])
    _, gv_free, _ = mirror_plane(pos, vel, mass, 0.0, 0.5, +1,
                                 slip="free_slip")
    _, gv_nav, _ = mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="navier",
                                slip_length=1e12)
    assert torch.allclose(gv_nav, gv_free, atol=1e-9)


def test_ghost_tangential_follows_alpha_formula():
    lam, plane_y = 0.05, 0.0
    ys = [0.01, 0.02, 0.04, 0.08]
    pos, vel, mass = _state(ys, ux=[0.3, 0.6, 1.0, 1.4], uy=[0.1, 0.2, 0.3, 0.4])
    _, gv, _ = mirror_plane(pos, vel, mass, plane_y, 0.5, +1, slip="navier",
                            slip_length=lam)
    for k, y in enumerate(ys):
        d = y - plane_y
        alpha = (lam - d) / (lam + d)
        assert gv[k, 0].item() == pytest.approx(alpha * vel[k, 0].item(),
                                                rel=1e-12, abs=1e-14)
        # normal component is mirrored exactly as in no_slip
        assert gv[k, 1].item() == pytest.approx(-vel[k, 1].item(),
                                                rel=1e-12, abs=1e-14)


def test_slip_rule_is_exactly_consistent_with_the_analytic_wall_layer():
    """u(d) = lam G + G d  =>  implied wall velocity is exactly lam G."""
    lam = 0.03
    grad = 2.5
    for ys in ([0.006, 0.012, 0.018], [0.0005, 0.001, 0.004]):
        d = torch.tensor(ys, dtype=torch.float64)
        u = lam * grad + grad * d
        pos, vel, mass = _state(ys, ux=u.tolist())
        _, gv, _ = mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="navier",
                                slip_length=lam)
        for k, y in enumerate(ys):
            implied_wall = 0.5 * (gv[k, 0].item() + vel[k, 0].item())
            assert implied_wall == pytest.approx(lam * grad, rel=1e-10)


def test_tangentially_moving_wall_has_no_relative_slip():
    lam, plane_y, w = 0.02, 0.0, 1.25
    ys = [0.01, 0.03]
    pos, vel, mass = _state(ys, ux=[w, w], uy=[0.0, 0.0])
    _, gv, _ = mirror_plane(pos, vel, mass, plane_y, 0.5, +1, slip="navier",
                            slip_length=lam, plane_vel=(w, 0.0))
    # fluid moving exactly with the wall: no relative shear, ghost keeps w
    assert torch.allclose(gv[:, 0], torch.full((2,), w, dtype=torch.float64))


def test_navier_leaves_ghost_positions_and_masses_untouched():
    pos, vel, mass = _state([0.01, 0.3], ux=[1.0, 2.0])
    gp_n, _, gm_n = mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="navier",
                                 slip_length=0.05)
    gp_0, _, gm_0 = mirror_plane(pos, vel, mass, 0.0, 0.5, +1)
    assert torch.allclose(gp_n, gp_0)
    assert torch.allclose(gm_n, gm_0)
    # mirrored through the plane, not through a spacing
    assert gp_n[0, 1].item() == pytest.approx(2.0 * 0.0 - 0.01)


def test_top_plane_side_uses_outward_distance():
    lam, plane_y = 0.05, 1.0
    pos, vel, mass = _state([0.99, 0.97], ux=[0.5, 0.8])
    _, gv, _ = mirror_plane(pos, vel, mass, plane_y, 0.5, -1, slip="navier",
                            slip_length=lam)
    for k, y in enumerate([0.99, 0.97]):
        d = (y - plane_y) * -1
        alpha = (lam - d) / (lam + d)
        assert gv[k, 0].item() == pytest.approx(alpha * vel[k, 0].item(),
                                                rel=1e-12)


def test_navier_requires_positive_slip_length():
    pos, vel, mass = _state([0.01], ux=[1.0])
    with pytest.raises(ValueError):
        mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="navier")
    with pytest.raises(ValueError):
        mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="navier",
                     slip_length=0.0)
    with pytest.raises(ValueError):
        mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="navier",
                     slip_length=-1e-3)
    with pytest.raises(ValueError):
        mirror_plane(pos, vel, mass, 0.0, 0.5, +1, slip="partial")


def test_plane_walls_forwards_slip_length():
    pos, vel, mass = _state([0.02, 0.98], ux=[1.0, -1.0])
    planes = [(0.0, +1), (1.0, -1)]
    _, gv, _ = plane_walls(pos, vel, mass, planes, 0.5, slip="navier",
                           slip_length=0.05)
    assert gv.shape[0] == 2
    # bottom ghost: lam > d -> tangential still positive (partial slip)
    assert gv[0, 0].item() > 0.0
    # top ghost: alpha > 0 as well, and the sign of the tangential part follows
    # the fluid, so the mirrored flow is not reversed as in no_slip
    assert gv[1, 0].item() < 0.0


def test_empty_wall_layer_returns_empty_ghosts():
    pos, vel, mass = _state([0.4], ux=[1.0])
    gp, gv, gm = mirror_plane(pos, vel, mass, 0.0, 0.1, +1, slip="navier",
                              slip_length=0.05)
    assert gp.shape[0] == 0 and gv.shape[0] == 0 and gm.shape[0] == 0